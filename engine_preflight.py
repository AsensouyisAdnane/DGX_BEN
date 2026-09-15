"""Per-model/per-engine smoke checks run before the full benchmark matrix."""

import os
import time
import logging
from datetime import datetime, timezone

from bench_client import send_one_request
from csv_logger import ResultLogger
from docker_utils import (
    docker_full_cleanup,
    get_gpu_snapshot,
    get_served_model,
    ensure_docker_image,
    run_cmd,
    save_container_logs,
    stop_container,
    wait_for_ready,
)
from engine_runners import ENGINE_RUNNERS, model_artifact_path

PREFLIGHT_COLUMNS = [
    "timestamp",
    "check_id",
    "model_id",
    "model_hf_path",
    "engine",
    "engine_image",
    "network_check",
    "quantization",
    "status",
    "failure_reason",
    "container_log_file",
    "load_time_s",
    "smoke_latency_s",
    "served_model",
    "hf_token_present",
    "ngc_api_key_present",
]

STATUS_DONE = "done"
STATUS_CANNOT_RUN = "cannot_run"
STATUS_TERMINATED = "terminated_with_error"
log = logging.getLogger("dgx_bench")


def log_loading_progress(index: int, total: int, model_id: str, engine: str,
                         elapsed: int, timeout_s: int, container_state: str | None,
                         container_log: str, cache_size_mb: int | None,
                         cache_delta_mb: int | None) -> None:
    snapshot = get_gpu_snapshot()
    memory = "GPU memory=N/A"
    if snapshot:
        if snapshot["memory_used_mb"] is not None:
            memory = (f"GPU memory={snapshot['memory_used_mb']:.0f}/"
                      f"{snapshot['memory_total_mb']:.0f} MB")
        else:
            memory = (f"GPU memory=N/A, utilization={snapshot['utilization_pct'] or 0:.0f}%, "
                      f"power={snapshot['power_draw_w'] or 0:.1f}W, "
                      f"temperature={snapshot['temperature_c'] or 0:.0f}C")
    cache = "model cache=unavailable" if cache_size_mb is None else f"model cache={cache_size_mb} MB"
    if cache_delta_mb:
        cache += f" ({cache_delta_mb:+d} MB)"
    log_suffix = f", engine_log={container_log}" if container_log else ""
    limit = "no total deadline" if timeout_s is None else f"{timeout_s}s limit"
    log.info("[preflight %d/%d] %s / %s: loading %ds, %s, container=%s, %s, %s%s",
             index, total, model_id, engine, elapsed, limit,
             container_state or "unknown", memory, cache, log_suffix)


def engine_image(model: dict, engine: str, config_module) -> str:
    engine_config = config_module.ENGINES[engine]
    return engine_config.get("image_by_model", {}).get(model["id"], engine_config.get("image", ""))


def check_id(model: dict, engine: str, config_module) -> str:
    return "__".join((model["id"], model["hf_path"], engine, model["quantization_default"],
                      engine_image(model, engine, config_module) or "unconfigured"))


def load_cached_checks(logger: ResultLogger) -> dict:
    if not os.path.exists(logger.csv_path):
        return {}
    latest = {}
    with open(logger.csv_path, newline="") as file:
        import csv
        for row in csv.DictReader(file):
            if row.get("check_id"):
                latest[row["check_id"]] = row
    return latest


def classify_failure(error: Exception) -> str:
    message = str(error).lower()
    permanent_markers = (
        "out of memory", "cuda out of memory", "does not fit", "not enough memory",
        "not supported", "no nim image", "missing pre-built", "repository not found",
        "model not found", "access denied", "unauthorized", "invalid model",
        "no image configured", "ngc_api_key",
    )
    if isinstance(error, (FileNotFoundError, NotImplementedError, KeyError)) or any(
        marker in message for marker in permanent_markers
    ):
        return STATUS_CANNOT_RUN
    return STATUS_TERMINATED


def verify_vllm_dns(image: str) -> str:
    """Verify DNS from the actual vLLM image before model download begins."""
    result = run_cmd([
        "docker", "run", "--rm", "--entrypoint", "python3", image, "-c",
        "import socket; print(socket.gethostbyname('huggingface.co'))",
    ], timeout=30)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()[-1000:]
        raise ConnectionError(f"Container DNS cannot resolve huggingface.co: {detail}")
    return result.stdout.strip()


def run_preflight(matrix: list, config_module, port: int,
                  refresh: bool = False) -> dict:
    """Return the latest status for each (model_id, engine) pair."""
    logger = ResultLogger(config_module.PREFLIGHT_CSV, PREFLIGHT_COLUMNS)
    cached = load_cached_checks(logger)
    statuses = {}
    seen = set()
    checks = []
    for experiment in matrix:
        pair = (experiment["model"]["id"], experiment["engine"])
        if pair not in seen:
            seen.add(pair)
            checks.append((pair[0], pair[1], experiment["model"]))

    for index, (model_id, engine, model) in enumerate(checks, 1):
        pair = (model_id, engine)

        current_check_id = check_id(model, engine, config_module)
        cached_row = cached.get(current_check_id)
        if not refresh and cached_row and cached_row.get("status") in {STATUS_DONE, STATUS_CANNOT_RUN}:
            statuses[pair] = cached_row["status"]
            log.info("[preflight %d/%d] %s / %s: cached status=%s",
                     index, len(checks), model["id"], engine, cached_row["status"])
            continue

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "check_id": current_check_id,
            "model_id": model["id"],
            "model_hf_path": model["hf_path"],
            "engine": engine,
            "engine_image": engine_image(model, engine, config_module),
            "network_check": "not_required",
            "quantization": model["quantization_default"],
            "hf_token_present": bool(os.environ.get("HF_TOKEN")),
            "ngc_api_key_present": bool(os.environ.get("NGC_API_KEY")),
        }
        startup_timeout_s = model.get("startup_timeout_s", config_module.HEALTH_CHECK_TIMEOUT_S)
        row["container_log_file"] = (
            f"{config_module.TRACE_DIR}/preflight_{model['id']}__{engine}.log"
        )
        container = None
        try:
            log.info("[preflight %d/%d] %s / %s: checking image and starting service...",
                     index, len(checks), model["id"], engine)
            if not row["engine_image"]:
                raise NotImplementedError(f"No image configured for {model['id']} / {engine}")
            log.info("[preflight %d/%d] %s / %s: ensuring image is available locally...",
                     index, len(checks), model["id"], engine)
            ensure_docker_image(row["engine_image"], timeout_s=3600)
            if engine == "vllm":
                log.info("[preflight %d/%d] %s / vllm: checking container DNS for huggingface.co...",
                         index, len(checks), model["id"])
                row["network_check"] = f"huggingface.co={verify_vllm_dns(row['engine_image'])}"
            docker_full_cleanup()
            container = ENGINE_RUNNERS[engine](model, "continuous_batching", "default", port)
            log.info("[preflight %d/%d] %s / %s: loading model...",
                     index, len(checks), model["id"], engine)
            row["load_time_s"] = round(wait_for_ready(
                port, startup_timeout_s,
                config_module.HEALTH_CHECK_POLL_INTERVAL_S, container.name,
                lambda elapsed, state, engine_log, cache_size, cache_delta: log_loading_progress(
                    index, len(checks), model["id"], engine, elapsed,
                    startup_timeout_s, state, engine_log, cache_size, cache_delta,
                ),
                config_module.PROGRESS_INTERVAL_S,
                model_artifact_path(model, engine),
            ), 2)
            served_model = get_served_model(port)
            row["served_model"] = served_model
            prompt = config_module.PROMPT_CASES[0]
            log.info("[preflight %d/%d] %s / %s: sending smoke request...",
                     index, len(checks), model["id"], engine)
            result = send_one_request(
                f"http://localhost:{port}", prompt["text"], prompt["max_output_tokens"],
                config_module.PER_REQUEST_TIMEOUT_S, 1, served_model, 0.0, 1.0
            )
            if not result.success:
                raise RuntimeError(result.error)
            row["smoke_latency_s"] = round(result.total_latency_s, 3)
            row["status"] = STATUS_DONE
            row["failure_reason"] = ""
            log.info("[preflight %d/%d] %s / %s: PASS (load %.1fs, request %.3fs)",
                     index, len(checks), model["id"], engine,
                     row["load_time_s"], row["smoke_latency_s"])
        except Exception as error:
            row["status"] = classify_failure(error)
            row["failure_reason"] = str(error)[:2000]
            log.warning("[preflight %d/%d] %s / %s: %s — %s",
                        index, len(checks), model["id"], engine,
                        row["status"], row["failure_reason"])
        finally:
            try:
                save_container_logs(container, row["container_log_file"])
            except Exception as trace_error:
                log.warning("Could not save container trace: %s", trace_error)
            stop_container(container)
            try:
                docker_full_cleanup()
            except Exception as cleanup_error:
                if row.get("status") == STATUS_DONE:
                    row["status"] = STATUS_TERMINATED
                    row["failure_reason"] = f"Cleanup failed: {cleanup_error}"[:2000]
            logger.append_row(row)

        statuses[pair] = row["status"]

    return statuses
