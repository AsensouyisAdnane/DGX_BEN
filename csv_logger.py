"""
csv_logger.py
--------------
Appends one row per finished experiment, flushing + fsyncing immediately so
that a crash (or you pulling the power on the DGX) never loses more than
the single experiment that was in flight.

Also supports RESUME: on startup, read whatever rows already exist and
skip any (model, engine, batching, kv_cache, prompt, sampling) combo already completed, so a
restarted run continues instead of starting the whole matrix over.
"""

import csv
import os
from typing import Optional

SUMMARY_COLUMNS = [
    "timestamp",
    "experiment_id",
    "model_id",
    "model_hf_path",
    "model_size_b",
    "model_active_params_b",
    "model_type",
    "engine",
    "quantization",
    "batching_mode",
    "kv_cache_mode",
    "benchmark_dataset",
    "prompt_case",
    "prompt_chars",
    "prompt_words_estimate",
    "prompt_tokens_actual",
    "max_output_tokens",
    "sampling_parameters",
    "status",
    "worked",                 # yes / no
    "failure_reason",
    "load_time_s",
    "max_stable_concurrency",
    "throughput_tokens_per_s",
    "throughput_req_per_s",
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "latency_mean_s",
    "latency_p50_s",
    "latency_p95_s",
    "gpu_mem_used_peak_mb",
    "gpu_mem_used_avg_mb",
    "gpu_util_avg_pct",
    "gpu_power_avg_w",
    "gpu_temp_avg_c",
    "gpu_temp_peak_c",
    "driver_version",
    "cuda_version",
    "run_duration_s",
]

DETAILED_COLUMNS = [
    "experiment_id",
    "concurrency",
    "success",
    "ttft_s",
    "total_latency_s",
    "input_tokens",
    "output_tokens",
    "error",
]


class ResultLogger:
    def __init__(self, csv_path: str, columns: list):
        self.csv_path = csv_path
        self.columns = columns
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def append_row(self, row: dict) -> None:
        # Fill any missing columns with empty string so schema drift doesn't crash the run
        safe_row = {col: row.get(col, "") for col in self.columns}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(safe_row)
            f.flush()
            os.fsync(f.fileno())  # survive a hard crash / power loss right after this line

    def append_rows(self, rows: list) -> None:
        if not rows:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            for row in rows:
                safe_row = {col: row.get(col, "") for col in self.columns}
                writer.writerow(safe_row)
            f.flush()
            os.fsync(f.fileno())

    def load_completed_experiment_ids(self) -> set:
        """Return experiments that are safe to skip when resuming.

        The CSV is append-only, so an experiment can have a starting
        ``running`` row followed by a final row. The last row wins.
        Legacy files without ``status`` are treated conservatively: only
        rows with worked=yes are considered complete.
        """
        if not os.path.exists(self.csv_path):
            return set()
        latest = {}
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_id = row.get("experiment_id")
                if exp_id:
                    latest[exp_id] = row

        skip_statuses = {"done", "cannot_run", "can't_run"}
        return {
            exp_id for exp_id, row in latest.items()
            if row.get("status") in skip_statuses
            or (not row.get("status") and row.get("worked") == "yes")
        }
