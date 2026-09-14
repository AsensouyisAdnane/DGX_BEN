"""
docker_utils.py
----------------
Everything that touches Docker or nvidia-smi.

Design goal: keep every run on IDENTICAL infrastructure. That means a full
teardown (stop + remove containers, prune the network/container layer --
NOT the pulled images, since those are multi-GB downloads you don't want
to repeat) before every single experiment, not just between models.
"""

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests


def run_cmd(cmd: list, timeout: Optional[int] = None, check: bool = False):
    """Run a shell command, always capturing output instead of letting it
    crash the whole script."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from e


def prepare_benchmark_prerequisites(matrix: list, config_module) -> None:
    """Pull every image and download every HF model before any experiment.

    TRT-LLM engines are local build artifacts, so their presence is checked
    here. NIM model assets are image/cache-specific; pulling the mapped image
    and starting it during the first experiment is the only portable way to
    validate those assets.
    """
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for model preparation")
    if any(exp["engine"] == "nim" for exp in matrix) and not os.environ.get("NGC_API_KEY"):
        raise RuntimeError("NGC_API_KEY is required for NIM preparation")

    check = run_cmd(["docker", "version"], timeout=30)
    if check.returncode != 0:
        raise RuntimeError(f"Docker is not available: {check.stderr.strip()}")
    check = run_cmd(["nvidia-smi"], timeout=30)
    if check.returncode != 0:
        raise RuntimeError(f"nvidia-smi is not available: {check.stderr.strip()}")

    images = set()
    for exp in matrix:
        engine_cfg = config_module.ENGINES[exp["engine"]]
        image = engine_cfg.get("image_by_model", {}).get(exp["model"]["id"], engine_cfg.get("image"))
        if not image:
            raise RuntimeError(f"No image configured for {exp['engine']} / {exp['model']['id']}")
        images.add(image)
    for image in sorted(images):
        result = run_cmd(["docker", "pull", image], timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Could not pull image '{image}': {result.stderr.strip()}")

    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    models = {exp["model"]["hf_path"] for exp in matrix}
    vllm_image = config_module.ENGINES["vllm"]["image"]
    for model_id in sorted(models):
        result = run_cmd([
            "docker", "run", "--rm", "--gpus", "all", "-e", "HF_TOKEN",
            "-v", f"{cache_dir}:/root/.cache/huggingface",
            "--entrypoint", "python", vllm_image, "-c",
            "from huggingface_hub import snapshot_download; snapshot_download(repo_id='" + model_id + "')",
        ], timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Could not prepare Hugging Face model '{model_id}': {result.stderr.strip()}")

    for exp in matrix:
        if exp["engine"] != "trtllm":
            continue
        cfg = config_module.ENGINES["trtllm"]
        path = cfg["engine_dir_template"].format(
            model_id=exp["model"]["id"], quant=exp["model"]["quantization_default"], batching=exp["batching"]
        )
        if not os.path.isdir(path) or not os.listdir(path):
            raise RuntimeError(f"Missing pre-built TensorRT-LLM engine: {path}")


# -------------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------------

def docker_full_cleanup(keep_images: bool = True) -> None:
    """Stop and remove ALL containers, prune networks, and make sure the
    GPU is free before the next experiment starts. Images are kept by
    default since model weights baked into containers (NIM) or pulled
    layers (vLLM/TRT-LLM) are expensive to re-download.
    """
    # Stop every running container
    ps = run_cmd(["docker", "ps", "-q"])
    ids = ps.stdout.split()
    if ids:
        run_cmd(["docker", "stop", *ids], timeout=60)

    # Remove every container (stopped or exited)
    ps_all = run_cmd(["docker", "ps", "-aq"])
    ids_all = ps_all.stdout.split()
    if ids_all:
        run_cmd(["docker", "rm", "-f", *ids_all], timeout=60)

    # Prune dangling networks/volumes created by previous runs (not images)
    run_cmd(["docker", "network", "prune", "-f"], timeout=30)
    run_cmd(["docker", "container", "prune", "-f"], timeout=30)

    if not keep_images:
        run_cmd(["docker", "image", "prune", "-af"], timeout=120)

    # Wait for the GPU driver to release memory after container teardown.
    wait_for_gpu_idle()


def gpu_is_idle(max_mem_mb: int = 500) -> bool:
    """Sanity check that no stray process is holding GPU memory before we
    start the next experiment. Returns True if usage is below threshold."""
    snap = get_gpu_snapshot()
    if snap is None:
        return True  # can't check -> don't block the run
    return snap["memory_used_mb"] <= max_mem_mb


def wait_for_gpu_idle(max_mem_mb: int = 500, timeout_s: int = 60,
                      poll_interval_s: float = 3.0) -> None:
    """Wait for GPU memory to be released after container teardown.

    The bounded wait avoids both a needless fixed delay and an infinite loop
    if another process is legitimately using the GPU.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gpu_is_idle(max_mem_mb):
            return
        time.sleep(poll_interval_s)

    snapshot = get_gpu_snapshot()
    if snapshot is not None and snapshot["memory_used_mb"] > max_mem_mb:
        raise RuntimeError(
            f"GPU memory did not fall below {max_mem_mb} MB within {timeout_s}s "
            f"(currently {snapshot['memory_used_mb']:.0f} MB)"
        )


# -------------------------------------------------------------------------
# GPU telemetry
# -------------------------------------------------------------------------

def get_gpu_snapshot() -> Optional[dict]:
    """One-shot nvidia-smi read: memory, utilization, power, temperature."""
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # DGX Spark has a single unified GPU; if multiple lines show up, take GPU0
    line = result.stdout.strip().splitlines()[0]
    mem_used, mem_total, util, power, temp = [x.strip() for x in line.split(",")]
    return {
        "memory_used_mb": float(mem_used),
        "memory_total_mb": float(mem_total),
        "utilization_pct": float(util),
        "power_draw_w": float(power) if power not in ("N/A", "[N/A]") else None,
        "temperature_c": float(temp),
    }


class GpuMonitor:
    """Background sampler. Start it right before load-testing begins, stop
    it right after, then pull peak/avg stats for the CSV row."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop_event.is_set():
            snap = get_gpu_snapshot()
            if snap:
                self._samples.append(snap)
            self._stop_event.wait(self.interval_s)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self._samples:
            return {
                "gpu_mem_used_peak_mb": None,
                "gpu_mem_used_avg_mb": None,
                "gpu_util_avg_pct": None,
                "gpu_power_avg_w": None,
                "gpu_temp_avg_c": None,
                "gpu_temp_peak_c": None,
            }
        mems = [s["memory_used_mb"] for s in self._samples]
        utils = [s["utilization_pct"] for s in self._samples]
        temps = [s["temperature_c"] for s in self._samples]
        powers = [s["power_draw_w"] for s in self._samples if s["power_draw_w"] is not None]
        return {
            "gpu_mem_used_peak_mb": max(mems),
            "gpu_mem_used_avg_mb": sum(mems) / len(mems),
            "gpu_util_avg_pct": sum(utils) / len(utils),
            "gpu_power_avg_w": (sum(powers) / len(powers)) if powers else None,
            "gpu_temp_avg_c": sum(temps) / len(temps),
            "gpu_temp_peak_c": max(temps),
        }


def get_driver_versions() -> dict:
    """Capture environment metadata so results are reproducible later."""
    versions = {"driver_version": None, "cuda_version": None}
    result = run_cmd(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=15,
    )
    if result.returncode == 0 and result.stdout.strip():
        versions["driver_version"] = result.stdout.strip().splitlines()[0]

    result2 = run_cmd(["nvidia-smi"], timeout=15)
    if result2.returncode == 0:
        for line in result2.stdout.splitlines():
            if "CUDA Version" in line:
                # e.g. "| NVIDIA-SMI 550.xx  Driver Version: 550.xx  CUDA Version: 12.4 |"
                try:
                    versions["cuda_version"] = line.split("CUDA Version:")[1].split("|")[0].strip()
                except IndexError:
                    pass
    return versions


# -------------------------------------------------------------------------
# Container lifecycle
# -------------------------------------------------------------------------

@dataclass
class RunningContainer:
    name: str
    port: int
    container_id: str = ""
    start_time: float = field(default_factory=time.time)


class ContainerStartError(RuntimeError):
    pass


def docker_run_detached(
    image: str,
    container_name: str,
    port: int,
    docker_args: list,
    entrypoint_args: list,
) -> str:
    """Generic `docker run -d` wrapper. Engine-specific argument building
    lives in engine_runners.py -- this function just executes the command
    and returns the container id, or raises ContainerStartError."""
    cmd = [
        "docker", "run", "-d",
        "--gpus", "all",
        "--name", container_name,
        "-p", f"{port}:{port}",
        "--ipc=host",
        *docker_args,
        image,
        *entrypoint_args,
    ]
    result = run_cmd(cmd, timeout=60)
    if result.returncode != 0:
        raise ContainerStartError(
            f"docker run failed for image '{image}': {result.stderr.strip()}"
        )
    return result.stdout.strip()  # container id


def stop_container(container: RunningContainer) -> None:
    if container is None:
        return
    run_cmd(["docker", "stop", container.name], timeout=60)
    run_cmd(["docker", "rm", "-f", container.name], timeout=30)


def wait_for_ready(port: int, timeout_s: int, poll_interval_s: int) -> float:
    """Poll the OpenAI-compatible /v1/models endpoint. Returns load time
    in seconds, or raises TimeoutError."""
    start = time.time()
    url = f"http://localhost:{port}/v1/models"
    while time.time() - start < timeout_s:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return time.time() - start
        except requests.RequestException:
            pass
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Service on port {port} did not become healthy within {timeout_s}s")


def get_served_model(port: int) -> str:
    """Return the model id advertised by the OpenAI-compatible service."""
    response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models or not models[0].get("id"):
        raise RuntimeError("/v1/models returned no served model id")
    return models[0]["id"]
