"""
engine_runners.py
------------------
Translates (model, engine, batching, kv_cache) into an actual `docker run`
and returns a RunningContainer. One function per engine, all with the same
signature so run_benchmark.py can call them interchangeably.

Each function raises a clear exception (ContainerStartError / NotImplementedError
/ FileNotFoundError) when the run genuinely cannot happen -- that exception
message becomes the `failure_reason` in the results CSV. This is intentional:
a missing TRT-LLM engine or an unmapped NIM image is a real, reportable
finding ("engine X has no supported path for model Y on this hardware"),
not a bug to hide.
"""

import os
import uuid

from DGX_BENCHMARK.DGX_BEN.config import ENGINES
from DGX_BENCHMARK.DGX_BEN.docker_utils import RunningContainer, docker_run_detached, ContainerStartError

HF_CACHE_MOUNT = ["-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface"]


def start_vllm(model: dict, batching: str, kv_cache: str, port: int) -> RunningContainer:
    cfg = ENGINES["vllm"]
    name = f"bench_vllm_{model['id']}_{uuid.uuid4().hex[:6]}"

    max_num_seqs = 1 if batching == "no_batching" else 256
    quant = model["quantization_default"]

    entrypoint_args = [
        "--model", model["hf_path"],
        "--port", str(port),
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
    ]
    if quant != "bf16":
        entrypoint_args += ["--quantization", quant]
    if kv_cache == "fp8_kv_cache":
        entrypoint_args += ["--kv-cache-dtype", "fp8"]

    docker_args = ["--rm", *HF_CACHE_MOUNT, "-e", "HF_TOKEN"]

    cid = docker_run_detached(cfg["image"], name, port, docker_args, entrypoint_args)
    return RunningContainer(name=name, port=port, container_id=cid)


def start_trtllm(model: dict, batching: str, kv_cache: str, port: int) -> RunningContainer:
    cfg = ENGINES["trtllm"]
    quant = model["quantization_default"]
    engine_dir = cfg["engine_dir_template"].format(
        model_id=model["id"], quant=quant, batching=batching
    )

    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(
            f"No pre-built TensorRT-LLM engine at '{engine_dir}'. "
            f"Build it first with `trtllm-build` for model={model['hf_path']}, "
            f"quant={quant}, batching={batching}. Logging this as a failed run."
        )

    name = f"bench_trtllm_{model['id']}_{uuid.uuid4().hex[:6]}"
    docker_args = ["--rm", "-v", f"{os.path.abspath(engine_dir)}:/engine"]
    entrypoint_args = [
        "trtllm-serve", "/engine",
        "--port", str(port),
    ]
    if kv_cache == "fp8_kv_cache":
        entrypoint_args += ["--kv_cache_type", "fp8"]
    # NOTE: continuous batching in TRT-LLM is normally baked into the engine
    # at build time (max_batch_size). If batching == "no_batching", the
    # engine directory itself should have been built with max_batch_size=1.

    cid = docker_run_detached(cfg["image"], name, port, docker_args, entrypoint_args)
    return RunningContainer(name=name, port=port, container_id=cid)


def start_nim(model: dict, batching: str, kv_cache: str, port: int) -> RunningContainer:
    cfg = ENGINES["nim"]
    image = cfg["image_by_model"].get(model["id"])
    if image is None:
        raise NotImplementedError(
            f"No NIM image mapped for model '{model['id']}' in config.py "
            f"ENGINES['nim']['image_by_model']. Logging this as a failed run "
            f"(NIM does not currently support this model on this rig)."
        )

    name = f"bench_nim_{model['id']}_{uuid.uuid4().hex[:6]}"
    docker_args = [
        "--rm",
        "-e", "NGC_API_KEY",
        "-v", f"{os.path.expanduser('~')}/.cache/nim:/opt/nim/.cache",
    ]
    entrypoint_args = []  # NIM containers self-configure; batching/KV knobs
    # are mostly fixed per NIM profile. If the specific NIM image exposes
    # env vars for this (e.g. NIM_MAX_BATCH_SIZE), add them here:
    if batching == "no_batching":
        docker_args += ["-e", "NIM_MAX_BATCH_SIZE=1"]
    if kv_cache == "fp8_kv_cache":
        docker_args += ["-e", "NIM_KV_CACHE_DTYPE=fp8"]

    cid = docker_run_detached(image, name, port, docker_args, entrypoint_args)
    return RunningContainer(name=name, port=port, container_id=cid)


ENGINE_RUNNERS = {
    "vllm": start_vllm,
    "trtllm": start_trtllm,
    "nim": start_nim,
}
