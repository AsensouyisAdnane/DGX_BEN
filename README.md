# DGX Spark Inference Benchmark

Runs the full engine x model x quantization x batching x KV-cache x prompt x sampling matrix
on a single DGX Spark. It completes all configurations for one inference engine before moving
to the next, while cleaning the Docker/GPU environment before **every** run so all
configurations are tested on identical infrastructure. It writes an in-flight
status marker before each run and a final status row immediately after it
finishes, so a restart can identify the single experiment that was in flight.

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

1. The script checks Docker and NVIDIA, then smoke-tests every model/engine
   pair before the full matrix. Each check starts the engine, which pulls the
   image and downloads/caches required model assets, waits for readiness, and
   sends one request. Failed pairs are recorded in `engine_preflight.csv` and
   excluded from the full matrix.

2. **Edit `config.py`:**
   - `ENGINES["vllm"]["image"]`, `ENGINES["trtllm"]["image"]` — pin versions.
   - Add an optional `quantizations_by_engine` list inside a model to test
     additional weight-quantization candidates. The default is used when it
     is omitted. Keep NIM to the quantization embedded in its image.
   - `ENGINES["nim"]["image_by_model"]` — map each model id to its real NIM
     image tag. An unmapped or unsupported model is recorded as `cannot_run`
     in the preflight report and its matrix rows are skipped.

3. TensorRT-LLM uses the local Hugging Face cache directly with the configured
   `1.3.0rc24` image, so no `trtllm_engines/` directory is required. The
   checkpoint must be supported by that TensorRT-LLM release.

4. Set credentials required by the selected model images:
   ```bash
   export HF_TOKEN=your_huggingface_token
   export NGC_API_KEY=your_ngc_personal_api_key  # only when the NIM image needs it
   ```

5. `pip install -r requirements.txt --break-system-packages`

## Running it

```bash
# See the planned matrix without touching Docker or the GPU
python3 run_benchmark.py --dry-run

# Start every model/engine once, run one request, and save engine_preflight.csv
python3 run_benchmark.py --preflight-only

# Run everything (resumes automatically if it was interrupted before)
python3 run_benchmark.py

# Ignore cached final preflight checks and run them again
python3 run_benchmark.py --refresh-preflight

# Force a full restart, ignoring any previous results_summary.csv
python3 run_benchmark.py --no-resume
```

The script logs to both the console and `benchmark_run.log`. Each startup log
includes the engine, model, selected quantization, batching, and KV-cache mode.
Leave it running in `tmux`/`screen` — the default 576-experiment matrix across 6 models
will take many hours, most of it model load time for the bigger models.

## If the DGX crashes or you need to stop it

Just re-run `python3 run_benchmark.py`. It reads the latest row for each
`experiment_id` in `results_summary.csv`, skips only `done` and
`cannot_run` experiments, and retries `terminated_with_error` and stale
`running` experiments. Each status row is fsync'd to disk.

## Output

- **`results_summary.csv`** — experiment status (`done`, `cannot_run`, or
  `terminated_with_error`), model, engine, batching, KV-cache mode, worked
  (yes/no) + failure reason, load time,
  max stable concurrency, throughput (tokens/s and req/s), TTFT and
  latency (mean/p50/p95), driver/CUDA version, total run duration. This is the file
  to load into your notebook for the report and charts.

- **`results_detailed_requests.csv`** — every individual request's raw
  TTFT/latency/token count, tagged with its `experiment_id` and
  concurrency level. Use this if you want distributions/histograms rather
  than just the aggregated percentiles in the summary file.

- **`engine_preflight.csv`** — one smoke check per model/engine pair. A check
  starts the container with the conservative single-sequence setting and the default KV cache,
  waits for `/v1/models`, then sends one greedy request. The full matrix runs
  only for pairs whose preflight status is `done`. `cannot_run` results are
  cached; use `--refresh-preflight` after changing an image, token, engine,
  or model configuration.

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
- TensorRT-LLM direct checkpoint serving depends on model support in the
  selected image; unsupported architectures are recorded as failed runs.
- Docker image layers are kept between runs (`keep_images=True` in
  `docker_full_cleanup`) so multi-GB pulls aren't repeated — only
  containers/networks are torn down between experiments.
