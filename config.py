"""
config.py
---------
Single source of truth for the DGX Spark benchmark matrix.

Edit THIS file to add/remove models, engines, or configurations.
Nothing in run_benchmark.py should need to change when you tweak the matrix.
"""

# =========================================================================
# 1. MODELS  (ordered smallest -> biggest on purpose; the orchestrator
#             preserves this order so cheap/fast runs happen first and
#             you get partial results quickly)
# =========================================================================
#
# quantization_default: the precision the run should use by default on a
#   single DGX Spark (128GB unified memory). Anything that will not fit
#   in bf16 alongside a reasonable KV cache is pre-quantized.
#
# engines: which inference engines we will ATTEMPT for this model. Not
#   every engine ships a prebuilt path for every model -- if a container
#   or engine file is missing, the run is logged as a clean FAILURE
#   (worked=no, failure_reason=...) rather than crashing the script.
#
# startup_timeout_s is optional per model. It is a hung-startup safety limit,
# not evidence of readiness; Docker state, engine logs, GPU memory, API health,
# and a smoke inference request provide the actual trace.

MODELS = [
    {
        "id": "gpt-oss-20b",
        "hf_path": "openai/gpt-oss-20b",
        "size_b": 20,
        "active_params_b": None,          # unpublished / not needed for the report
        "type": "MoE",
        "quantization_default": "mxfp4",  # ships natively quantized
        "engines": ["vllm", "trtllm", "nim"],
    },
    {
        "id": "nemotron-3.5-lightning-30b-a3b",
        "hf_path": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "size_b": 30,
        "active_params_b": 3,
        "type": "Hybrid Mamba+MoE",
        "quantization_default": "nvfp4",  # checkpoint is pre-quantized
        "engines": ["vllm", "trtllm", "nim"],
    },
    {
        "id": "qwen3.8-27b",
        "hf_path": "Qwen/Qwen3.8-27B",
        "size_b": 27,
        "active_params_b": 27,
        "type": "Dense",
        "quantization_default": "bf16",   # fits natively on 128GB unified mem
        "engines": ["vllm", "trtllm", "nim"],
    },
    {
        "id": "deepseek-r1-distill-llama-70b",
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "size_b": 70,
        "active_params_b": 70,
        "type": "Dense (reasoning-distilled)",
        "quantization_default": "fp8",    # bf16 (~140GB) will not fit + KV cache
        "engines": ["vllm", "trtllm", "nim"],
    },
    {
        "id": "llama-4-scout-17b-16e",
        "hf_path": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "size_b": 109,
        "active_params_b": 17,
        "type": "MoE (natively multimodal)",
        "quantization_default": "fp8",
        "engines": ["vllm", "trtllm", "nim"],
    },
    {
        "id": "gpt-oss-120b",
        "hf_path": "openai/gpt-oss-120b",
        "size_b": 120,
        "active_params_b": None,
        "type": "MoE",
        "quantization_default": "mxfp4",  # ships natively quantized (~65GB)
        "engines": ["vllm", "trtllm", "nim"],
    },
]

# =========================================================================
# 2. ENGINES  -- docker image / launch templates.
#    >>> YOU MUST EDIT THE IMAGE TAGS <<< to match what you've pulled.
# =========================================================================

ENGINES = {
    "vllm": {
        "image": "vllm/vllm-openai:latest",           # EDIT if pinning a version
        "container_port": 8000,
        # Keep headroom for the OS/runtime on DGX Spark. This matches the
        # NVIDIA launch example and avoids startup failure at vLLM's default.
        "gpu_memory_utilization": 0.8,
    },
    "trtllm": {
        # TensorRT-LLM engines must be built per (model, precision, batching)
        # ahead of time with `trtllm-build`. This script does NOT build them
        # for you (that step is model-specific and slow). It expects a
        # pre-built engine directory at:
        #   ./trtllm_engines/<model_id>__<quantization>__<batching>/
        # If that path is missing, the run is logged as a failed experiment
        # with a clear reason instead of crashing.
        "image": "nvcr.io/nvidia/tensorrt-llm/release:latest",  # EDIT
        "container_port": 8001,
        "engine_dir_template": "./trtllm_engines/{model_id}__{quant}__{batching}",
    },
    "nim": {
        # NIM containers are per-model microservices pulled from NGC.
        # EDIT this mapping to the exact NIM image for each model you have
        # access to. Models with no known NIM image will fail cleanly.
        "image_by_model": {
            "nemotron-3.5-lightning-30b-a3b": "nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:latest",
            "gpt-oss-20b": "nvcr.io/nim/openai/gpt-oss-20b:latest",
            "gpt-oss-120b": "nvcr.io/nim/openai/gpt-oss-120b:latest",
            # models below intentionally have no entry yet -> NIM runs
            # for them will fail cleanly and get logged as "no NIM image"
        },
        "container_port": 8002,
    },
}

# =========================================================================
# 3. RUN CONFIGURATIONS (batching x KV-cache x prompt case)
# =========================================================================

BATCHING_MODES = [
    "no_batching",          # max_num_seqs = 1 (serialize requests)
    "continuous_batching",  # engine default dynamic/continuous batching
]

KV_CACHE_MODES = [
    "default",     # engine default KV cache dtype (usually fp16/bf16)
    "fp8_kv_cache",  # quantized KV cache, where the engine supports it
]

# Concurrency sweep used for the "multiple requests handling" metric.
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
REQUESTS_PER_CONCURRENCY_LEVEL = 20   # keep short; this is a benchmark, not a soak test

# Public, reproducible benchmark corpus. Keep the exact text in this file (or
# replace it with a versioned dataset path) so prompt provenance is explicit.
BENCHMARK_DATASET = "inline-inference-engineering-v1"
_LONG_CONTEXT_PASSAGE = """\
Modern LLM inference platforms sit at the intersection of several \
independent design decisions, and it is easy to attribute a latency or \
throughput result to the wrong one. Model architecture sets a floor: a \
dense transformer activates every parameter on every token, while a \
mixture-of-experts model routes each token through a small subset of \
experts, which lowers compute per token but raises memory bandwidth \
pressure since the router can touch different weights every step.

Weight quantization changes the memory footprint and, on hardware with \
native low-precision matrix units, the achievable compute throughput. \
FP8 and INT4 weights roughly halve or quarter memory versus BF16, but \
quality loss and dequantization overhead vary by method and are not \
free to ignore in an enterprise evaluation.

KV-cache dtype is a separate lever from weight quantization. A model can \
run BF16 weights with an FP8 KV cache, or the reverse. Since the KV \
cache grows linearly with context length and concurrent sequences, its \
precision has an outsized effect on how many simultaneous requests a \
GPU can hold in memory before evicting or rejecting new ones.

Batching policy determines whether requests are processed one at a time \
or interleaved via continuous batching, where new requests join a \
running batch as soon as a slot frees up. Continuous batching is what \
allows throughput to keep climbing with concurrency instead of flatlining \
after the first request, but it also introduces scheduling overhead and \
can raise tail latency for unlucky requests that land behind a long one.

Context length interacts with all of the above: longer prompts increase \
prefill compute, which is the dominant cost behind time to first token, \
while longer generations increase decode steps, which is the dominant \
cost behind total latency and tokens-per-second throughput. Treating \
these as one variable hides which stage of inference is actually the \
bottleneck.

To compare vLLM, TensorRT-LLM, and NIM fairly, a report should hold the \
prompt and output length fixed per test case, vary concurrency \
independently, and report time to first token, inter-token latency, \
end-to-end latency percentiles, tokens-per-second throughput, peak and \
average GPU memory, and power draw -- broken out by the actual input and \
output token counts used, not just a qualitative "short/medium/long" \
label.
"""

PROMPT_CASES = [
    {
        "id": "short_in_short_out",
        "text": "What is continuous batching and why can it improve LLM serving throughput?",
        "max_output_tokens": 64,
    },
    {
        "id": "short_in_long_out",
        "text": (
            "Explain in detail how continuous batching works in modern LLM "
            "inference engines, including how it differs from static batching "
            "and what scheduling trade-offs it introduces."
        ),
        "max_output_tokens": 512,
    },
    {
        "id": "long_in_short_out",
        "text": _LONG_CONTEXT_PASSAGE + "\nSummarize the above in exactly one sentence.",
        "max_output_tokens": 64,
    },
    {
        "id": "long_in_long_out",
        "text": _LONG_CONTEXT_PASSAGE + (
            "\nBased on the above, write a detailed comparison of how each "
            "design decision affects time to first token versus total latency."
        ),
        "max_output_tokens": 512,
    },
]
SAMPLING_PARAMETERS = [
    {"id": "greedy", "temperature": 0.0, "top_p": 1.0},
    {"id": "sampled", "temperature": 0.7, "top_p": 0.9},
]

# =========================================================================
# 4. TIMEOUTS / SAFETY LIMITS
# =========================================================================

# None means wait for the engine's readiness callback instead of imposing a
# total download/load deadline. Set an integer only when you want a hard cap.
HEALTH_CHECK_TIMEOUT_S = None
HEALTH_CHECK_POLL_INTERVAL_S = 5
PER_REQUEST_TIMEOUT_S = 180
GPU_MONITOR_INTERVAL_S = 2
PROGRESS_INTERVAL_S = 5
TRACE_DIR = "docker_logs"

# =========================================================================
# 5. OUTPUT FILES
# =========================================================================

RESULTS_CSV = "results_summary.csv"
DETAILED_CSV = "results_detailed_requests.csv"
PREFLIGHT_CSV = "engine_preflight.csv"
LOG_FILE = "benchmark_run.log"
