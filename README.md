# DGX Spark Inference Benchmark

Runs the full model x engine x batching x KV-cache x prompt x sampling matrix on a single DGX
Spark, cleaning the Docker/GPU environment before **every** run so all
configurations are tested on identical infrastructure, and saving one row
to CSV **immediately** after each experiment finishes so a crash never
costs more than the single run in flight.

## Files

| File | Purpose |
|---|---|
| `config.py` | The experiment matrix: models, engines, batching/KV-cache modes, timeouts. **Edit this first.** |
| `engine_runners.py` | Builds the `docker run` command for each engine (vLLM / TRT-LLM / NIM). **Edit the image tags and, for TRT-LLM, the pre-built engine paths.** |
| `docker_utils.py` | Cleanup, container start/stop, health checks, GPU telemetry (nvidia-smi polling). |
| `bench_client.py` | Load-testing client: streams requests, measures TTFT/latency/throughput across a concurrency sweep. |
| `csv_logger.py` | Crash-safe CSV writer (fsync after every row) + resume support. |
| `run_benchmark.py` | Orchestrator — run this. |

## Before you run

1. The script performs a strict preflight before the first experiment. It
   checks Docker and NVIDIA, pulls every configured engine image, and uses the
   vLLM image to download every Hugging Face model into the shared cache. It
   also checks that every configured TRT-LLM engine directory exists. A failed
   preflight stops the benchmark; it does not create misleading partial rows.

2. **Edit `config.py`:**
   - `ENGINES["vllm"]["image"]`, `ENGINES["trtllm"]["image"]` — pin versions.
   - `ENGINES["nim"]["image_by_model"]` — map each model id to its real NIM
     image tag. Every model/engine combination in the matrix must be mapped;
     an unmapped model stops preflight so the run cannot silently omit coverage.

3. **Pre-build TensorRT-LLM engines.** Unlike vLLM, TRT-LLM needs a compiled
   engine per (model, quantization, batching) before this script runs. Put each one at:
   ```
   ./trtllm_engines/<model_id>__<quantization>__<batching>/
   ```
   e.g. `./trtllm_engines/gpt-oss-20b__mxfp4__continuous_batching/`.
   If any required directory is missing, preflight stops with the exact path.

3. Set credentials the containers need:
   ```bash
   export HF_TOKEN=your_huggingface_token
   export NGC_API_KEY=your_ngc_key
   ```

4. `pip install -r requirements.txt --break-system-packages`

## Running it

```bash
# See the planned matrix without touching Docker or the GPU
python3 run_benchmark.py --dry-run

# Run everything (resumes automatically if it was interrupted before)
python3 run_benchmark.py

# Force a full restart, ignoring any previous results_summary.csv
python3 run_benchmark.py --no-resume
```

The script logs to both the console and `benchmark_run.log`. Leave it
running in `tmux`/`screen` — the default 432-experiment matrix across 6 models
will take many hours, most of it model load time for the bigger models.

## If the DGX crashes or you need to stop it

Just re-run `python3 run_benchmark.py`. It reads `results_summary.csv`,
skips every `experiment_id` already present, and continues where it left
off. Nothing before the crash is lost — every row was fsync'd to disk the
moment that experiment finished.

## Output

- **`results_summary.csv`** — one row per experiment: model, engine,
  batching, KV-cache mode, worked (yes/no) + failure reason, load time,
  max stable concurrency, throughput (tokens/s and req/s), TTFT and
  latency (mean/p50/p95), GPU memory/utilization/power/temperature
  (peak + avg), driver/CUDA version, total run duration. This is the file
  to load into your notebook for the report and charts.

- **`results_detailed_requests.csv`** — every individual request's raw
  TTFT/latency/token count, tagged with its `experiment_id` and
  concurrency level. Use this if you want distributions/histograms rather
  than just the aggregated percentiles in the summary file.

## Benchmark dataset and prompts

The default dataset is `inline-inference-engineering-v1`, defined directly in
`config.py` as three deterministic prompt cases: `short`, `medium`, and
`long`. The summary records the case id, exact character count, a word-count
estimate, maximum output tokens, temperature, and top-p. The long case is a
repeated inference-engineering brief to exercise a larger input context. These
are performance prompts, not a quality/evaluation dataset; add a versioned
dataset loader and quality metrics if answer quality is part of the report.

## Extending the metrics

`bench_client.py` currently reports TTFT, TPOT-derived latency, and
throughput. To add power-over-time or thermal-throttling detection for the
report, `docker_utils.GpuMonitor` already samples `nvidia-smi` every
`GPU_MONITOR_INTERVAL_S` seconds during the load test — its raw
`self._samples` list has a full timeseries per experiment if you want to
plot temperature/power curves instead of just peak/avg in the notebook.

## Known limitations (be upfront about these in the report)

- The NIM launch parameters for batching/KV-cache are placeholders
  (`NIM_MAX_BATCH_SIZE`, `NIM_KV_CACHE_DTYPE` env vars) — verify these
  against the actual NIM image's supported configuration env vars, they
  vary per model microservice.
- TRT-LLM engines must be built ahead of time; this script deliberately
  does not automate `trtllm-build` because the right flags are highly
  model- and precision-specific.
- Docker image layers are kept between runs (`keep_images=True` in
  `docker_full_cleanup`) so multi-GB pulls aren't repeated — only
  containers/networks are torn down between experiments.
