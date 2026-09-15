#!/usr/bin/env python3
"""
run_benchmark.py
-----------------
DGX Spark inference benchmark orchestrator.

For every (model x engine x batching x kv_cache x prompt x sampling) combination, in that
order, smallest model first:
  1. Full docker + GPU cleanup (identical infra for every run)
  2. Start the container for this configuration
  3. Wait for it to become healthy (records cold-start / load time)
  4. Run the concurrency sweep load test
  5. Tear the container down, clean up again
  6. Append ONE row to results_summary.csv immediately (fsync'd)
  7. Append the raw per-request rows to results_detailed_requests.csv

If ANY step fails, the experiment is logged with worked=no and a specific
failure_reason, and the loop moves on to the next combination. A crash of
the whole Python process loses at most the single in-flight experiment,
because every previous result was already flushed to disk.

Usage:
    python3 run_benchmark.py                # run everything, resume if interrupted
    python3 run_benchmark.py --no-resume    # start the matrix over from scratch
    python3 run_benchmark.py --dry-run      # print the planned matrix, run nothing
"""

import argparse
import itertools
import logging
import sys
import time
from datetime import datetime, timezone

import config
from csv_logger import ResultLogger, SUMMARY_COLUMNS, DETAILED_COLUMNS
from docker_utils import (
    docker_full_cleanup,
    GpuMonitor,
    get_driver_versions,
    get_gpu_snapshot,
    save_container_logs,
    stop_container,
    wait_for_ready,
    get_served_model,
    prepare_benchmark_prerequisites,
)
from engine_runners import ENGINE_RUNNERS, model_artifact_path
from bench_client import run_full_sweep
from engine_preflight import STATUS_DONE, run_preflight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dgx_bench")

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_CANNOT_RUN = "cannot_run"
STATUS_TERMINATED = "terminated_with_error"


def log_loading_progress(model_id: str, engine: str, elapsed: int, timeout_s: int,
                         container_state: str | None, container_log: str,
                         cache_size_mb: int | None, cache_delta_mb: int | None) -> None:
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
    log.info("Loading %s on %s: %ss, %s, container=%s, %s, %s%s",
             model_id, engine, elapsed, limit,
             container_state or "unknown", memory, cache, log_suffix)


def log_benchmark_gpu(snapshot: dict) -> None:
    memory = "N/A" if snapshot["memory_used_mb"] is None else (
        f"{snapshot['memory_used_mb']:.0f}/{snapshot['memory_total_mb']:.0f} MB"
    )
    log.info("Benchmark GPU: memory=%s, utilization=%s%%, power=%s W, temperature=%s C",
             memory,
             f"{snapshot['utilization_pct']:.0f}" if snapshot["utilization_pct"] is not None else "N/A",
             f"{snapshot['power_draw_w']:.0f}" if snapshot["power_draw_w"] is not None else "N/A",
             f"{snapshot['temperature_c']:.0f}" if snapshot["temperature_c"] is not None else "N/A")


def build_experiment_matrix() -> list:
    """Cartesian product of model x engine x batching x kv_cache, in the
    order models are declared in config.py (smallest -> biggest)."""
    matrix = []
    for model in config.MODELS:
        for engine in model["engines"]:
            for batching, kv_cache, prompt_case, sampling in itertools.product(
                config.BATCHING_MODES, config.KV_CACHE_MODES,
                config.PROMPT_CASES, config.SAMPLING_PARAMETERS
            ):
                exp_id = (f"{model['id']}__{engine}__{batching}__{kv_cache}__"
                          f"{prompt_case['id']}__{sampling['id']}")
                matrix.append({"experiment_id": exp_id, "model": model,
                               "engine": engine, "batching": batching,
                               "kv_cache": kv_cache, "prompt_case": prompt_case,
                               "sampling": sampling})
    return matrix


def base_row(exp: dict) -> dict:
    model = exp["model"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp["experiment_id"],
        "model_id": model["id"],
        "model_hf_path": model["hf_path"],
        "model_size_b": model["size_b"],
        "model_active_params_b": model["active_params_b"],
        "model_type": model["type"],
        "engine": exp["engine"],
        "quantization": model["quantization_default"],
        "batching_mode": exp["batching"],
        "kv_cache_mode": exp["kv_cache"],
        "benchmark_dataset": config.BENCHMARK_DATASET,
        "prompt_case": exp["prompt_case"]["id"],
        "prompt_chars": len(exp["prompt_case"]["text"]),
        "prompt_words_estimate": len(exp["prompt_case"]["text"].split()),
        "max_output_tokens": exp["prompt_case"]["max_output_tokens"],
        "sampling_parameters": (f"temperature={exp['sampling']['temperature']},"
                                 f"top_p={exp['sampling']['top_p']}"),
        "status": STATUS_RUNNING,
    }


def classify_failure(error: Exception) -> str:
    """Separate permanent engine/configuration failures from retryable errors."""
    message = str(error).lower()
    permanent_types = (FileNotFoundError, NotImplementedError, KeyError)
    permanent_markers = (
        "out of memory", "cuda out of memory", "does not fit", "not enough memory",
        "not supported", "no nim image", "missing pre-built", "repository not found",
        "model not found", "access denied", "unauthorized", "invalid model",
    )
    if isinstance(error, permanent_types) or any(marker in message for marker in permanent_markers):
        return STATUS_CANNOT_RUN
    return STATUS_TERMINATED


def run_one_experiment(exp: dict, summary_logger: ResultLogger,
                        detailed_logger: ResultLogger, port: int) -> None:
    row = base_row(exp)
    startup_timeout_s = exp["model"].get("startup_timeout_s", config.HEALTH_CHECK_TIMEOUT_S)
    row["container_log_file"] = f"{config.TRACE_DIR}/{exp['experiment_id']}.log"
    versions = get_driver_versions()
    row.update(versions)

    t_exp_start = time.time()
    container = None
    monitor = GpuMonitor(
        interval_s=config.GPU_MONITOR_INTERVAL_S,
        sample_callback=log_benchmark_gpu,
    )

    log.info(f"=== START {exp['experiment_id']} ===")
    # Persist an in-flight marker before Docker/GPU work starts. If the
    # machine loses power, the next invocation will retry this one only.
    summary_logger.append_row(row)

    try:
        log.info("Phase 1/4 cleanup: removing previous benchmark state and waiting for GPU memory...")
        docker_full_cleanup()

        runner = ENGINE_RUNNERS[exp["engine"]]
        log.info(f"Phase 2/4 startup: engine={exp['engine']} model={exp['model']['id']} "
                 f"batching={exp['batching']} kv_cache={exp['kv_cache']}...")
        container = runner(exp["model"], exp["batching"], exp["kv_cache"], port)

        log.info("Phase 3/4 loading: waiting for the model service to become ready...")
        load_time_s = wait_for_ready(
            port, startup_timeout_s, config.HEALTH_CHECK_POLL_INTERVAL_S,
            container.name,
            lambda elapsed, state, engine_log, cache_size, cache_delta: log_loading_progress(
                exp["model"]["id"], exp["engine"], elapsed, startup_timeout_s, state, engine_log,
                cache_size, cache_delta,
            ),
            config.PROGRESS_INTERVAL_S,
            model_artifact_path(exp["model"], exp["engine"]),
        )
        row["load_time_s"] = round(load_time_s, 2)
        served_model = get_served_model(port)

        log.info("Phase 4/4 benchmarking: running concurrency sweep...")
        monitor.start()
        sweep = run_full_sweep(
            base_url=f"http://localhost:{port}",
            concurrency_levels=config.CONCURRENCY_LEVELS,
            num_requests_per_level=config.REQUESTS_PER_CONCURRENCY_LEVEL,
            prompt=exp["prompt_case"]["text"],
            max_tokens=exp["prompt_case"]["max_output_tokens"],
            timeout_s=config.PER_REQUEST_TIMEOUT_S,
            model_name=served_model,
            temperature=exp["sampling"]["temperature"],
            top_p=exp["sampling"]["top_p"],
            progress_callback=log.info,
        )
        monitor.stop()
        row.update(monitor.summary())

        if not sweep["overall_success"]:
            row["worked"] = "no"
            row["status"] = STATUS_TERMINATED
            row["failure_reason"] = "All requests failed at every concurrency level"
        else:
            best = sweep["best_level_summary"]
            row["worked"] = "yes"
            row["status"] = STATUS_DONE
            row["failure_reason"] = ""
            input_token_counts = [
                r.input_tokens for level in sweep["per_level"]
                for r in level["raw_results"] if r.success and r.input_tokens is not None
            ]
            if input_token_counts:
                row["prompt_tokens_actual"] = input_token_counts[0]
            row["max_stable_concurrency"] = sweep["max_stable_concurrency"]
            row["throughput_tokens_per_s"] = best["throughput_tokens_per_s"]
            row["throughput_req_per_s"] = best["throughput_req_per_s"]
            row["ttft_mean_s"] = best["ttft_mean_s"]
            row["ttft_p50_s"] = best["ttft_p50_s"]
            row["ttft_p95_s"] = best["ttft_p95_s"]
            row["latency_mean_s"] = best["latency_mean_s"]
            row["latency_p50_s"] = best["latency_p50_s"]
            row["latency_p95_s"] = best["latency_p95_s"]

            # Flatten per-request raw results for the detailed CSV
            detailed_rows = []
            for level in sweep["per_level"]:
                for r in level["raw_results"]:
                    detailed_rows.append({
                        "experiment_id": exp["experiment_id"],
                        "concurrency": r.concurrency,
                        "success": r.success,
                        "ttft_s": r.ttft_s,
                        "total_latency_s": r.total_latency_s,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "error": r.error,
                    })
            detailed_logger.append_rows(detailed_rows)

        log.info(f"Result: worked={row.get('worked')} "
                  f"max_stable_concurrency={row.get('max_stable_concurrency')}")

    except KeyboardInterrupt:
        monitor.stop()
        row["worked"] = "no"
        row["status"] = STATUS_TERMINATED
        row["failure_reason"] = "Benchmark interrupted by user"
        log.warning(f"Experiment interrupted: {exp['experiment_id']}")
        raise
    except Exception as e:
        monitor.stop()
        row["worked"] = "no"
        row["status"] = classify_failure(e)
        row["failure_reason"] = str(e)[:500]
        log.warning(f"Experiment failed: status={row['status']} reason={row['failure_reason']}")

    finally:
        try:
            save_container_logs(container, row["container_log_file"])
        except Exception as trace_error:
            log.warning("Could not save container trace: %s", trace_error)
        stop_container(container)
        log.info("Cleanup: stopping container and waiting for GPU memory release...")
        try:
            docker_full_cleanup()
        except Exception as cleanup_err:
            log.error(f"Cleanup after experiment also failed: {cleanup_err}")

        row["run_duration_s"] = round(time.time() - t_exp_start, 2)
        summary_logger.append_row(row)
        log.info(f"=== SAVED {exp['experiment_id']} (duration {row['run_duration_s']}s) ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore existing results_summary.csv and rerun everything")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the planned experiment matrix and exit")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run cached model/engine smoke checks, then exit")
    parser.add_argument("--refresh-preflight", action="store_true",
                        help="Rerun smoke checks even when a cached final result exists")
    parser.add_argument("--port", type=int, default=8000,
                         help="Local port used to reach each container's API")
    args = parser.parse_args()

    matrix = build_experiment_matrix()

    if args.dry_run:
        print(f"Planned experiments: {len(matrix)}")
        for exp in matrix:
            print(f"  {exp['experiment_id']}")
        return

    log.info("Checking Docker and GPU availability...")
    prepare_benchmark_prerequisites(matrix, config)

    log.info("Running model/engine smoke checks before the full matrix...")
    preflight_statuses = run_preflight(
        matrix, config, args.port, refresh=args.refresh_preflight
    )
    passed_pairs = {
        pair for pair, status in preflight_statuses.items() if status == STATUS_DONE
    }
    log.info("Preflight passed for %d/%d model/engine pairs. Results: %s",
             len(passed_pairs), len(preflight_statuses), config.PREFLIGHT_CSV)

    if args.preflight_only:
        return

    summary_logger = ResultLogger(config.RESULTS_CSV, SUMMARY_COLUMNS)
    detailed_logger = ResultLogger(config.DETAILED_CSV, DETAILED_COLUMNS)

    completed = set() if args.no_resume else summary_logger.load_completed_experiment_ids()
    if completed:
        log.info(f"Resuming: {len(completed)} experiments already completed, will skip those.")

    remaining = [
        exp for exp in matrix
        if exp["experiment_id"] not in completed
        and (exp["model"]["id"], exp["engine"]) in passed_pairs
    ]
    log.info(f"Total experiments: {len(matrix)} | Remaining to run: {len(remaining)}")

    for i, exp in enumerate(remaining, 1):
        log.info(f"[{i}/{len(remaining)}] {exp['experiment_id']}")
        try:
            run_one_experiment(exp, summary_logger, detailed_logger, args.port)
        except KeyboardInterrupt:
            log.warning("Interrupted by user. Progress so far is saved in "
                        f"{config.RESULTS_CSV}. Re-run the script to resume.")
            sys.exit(1)
        except Exception as e:
            # Belt-and-suspenders: even an unexpected bug in the orchestrator
            # itself should not lose already-saved rows or kill the whole run.
            log.error(f"Unexpected orchestrator error on {exp['experiment_id']}: {e}")
            continue

    log.info("All experiments complete. See results_summary.csv and "
              "results_detailed_requests.csv for analysis in your notebook.")


if __name__ == "__main__":
    main()
