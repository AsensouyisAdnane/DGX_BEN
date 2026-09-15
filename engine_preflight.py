"""Per-model/per-engine smoke checks run before the full benchmark matrix."""

import os
import time
from datetime import datetime, timezone

from bench_client import send_one_request
from csv_logger import ResultLogger
from docker_utils import docker_full_cleanup, get_served_model, stop_container, wait_for_ready
from engine_runners import ENGINE_RUNNERS

PREFLIGHT_COLUMNS = [
    "timestamp",
    "check_id",
    "model_id",
    "model_hf_path",
    "engine",
    "engine_image",
    "quantization",
    "status",
    "failure_reason",
    "load_time_s",
    "smoke_latency_s",
    "served_model",
    "hf_token_present",
    "ngc_api_key_present",
]

STATUS_DONE = "done"
STATUS_CANNOT_RUN = "cannot_run"
STATUS_TERMINATED = "terminated_with_error"


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


def run_preflight(matrix: list, config_module, port: int,
                  refresh: bool = False) -> dict:
    """Return the latest status for each (model_id, engine) pair."""
    logger = ResultLogger(config_module.PREFLIGHT_CSV, PREFLIGHT_COLUMNS)
    cached = load_cached_checks(logger)
    statuses = {}
    seen = set()

    for experiment in matrix:
        model = experiment["model"]
        engine = experiment["engine"]
        pair = (model["id"], engine)
        if pair in seen:
            continue
        seen.add(pair)

        current_check_id = check_id(model, engine, config_module)
        cached_row = cached.get(current_check_id)
        if not refresh and cached_row and cached_row.get("status") in {STATUS_DONE, STATUS_CANNOT_RUN}:
            statuses[pair] = cached_row["status"]
            continue

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "check_id": current_check_id,
            "model_id": model["id"],
            "model_hf_path": model["hf_path"],
            "engine": engine,
            "engine_image": engine_image(model, engine, config_module),
            "quantization": model["quantization_default"],
            "hf_token_present": bool(os.environ.get("HF_TOKEN")),
            "ngc_api_key_present": bool(os.environ.get("NGC_API_KEY")),
        }
        container = None
        try:
            if not row["engine_image"]:
                raise NotImplementedError(f"No image configured for {model['id']} / {engine}")
            docker_full_cleanup()
            container = ENGINE_RUNNERS[engine](model, "continuous_batching", "default", port)
            row["load_time_s"] = round(wait_for_ready(
                port, config_module.HEALTH_CHECK_TIMEOUT_S,
                config_module.HEALTH_CHECK_POLL_INTERVAL_S, container.name
            ), 2)
            served_model = get_served_model(port)
            row["served_model"] = served_model
            prompt = config_module.PROMPT_CASES[0]
            result = send_one_request(
                f"http://localhost:{port}", prompt["text"], prompt["max_output_tokens"],
                config_module.PER_REQUEST_TIMEOUT_S, 1, served_model, 0.0, 1.0
            )
            if not result.success:
                raise RuntimeError(result.error)
            row["smoke_latency_s"] = round(result.total_latency_s, 3)
            row["status"] = STATUS_DONE
            row["failure_reason"] = ""
        except Exception as error:
            row["status"] = classify_failure(error)
            row["failure_reason"] = str(error)[:2000]
        finally:
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
