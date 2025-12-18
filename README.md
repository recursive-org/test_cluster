# Cluster scripts

Run the scripts to test the performance of a cluster.

All scripts now live in `./yuandong`.

For Nebius cluster, login to any worker node, download the docker image with `docker pull lmsysorg/sglang:latest` and run: 
```
python yuandong/container_perf_runner.py
```
It will automatically launch docker, wait until it is ready, and run performance test. The test is simple, just to load the SGLang server (all kernels are recompiled from scratch) and then send single/batch requests. We want to check the loading time and the thoroughputs with various TP settings.   

For Andromeda cluster, login to the main logging node, and run 
```
python yuandong/container_perf_runner.py
``` 
If docker is not available, the runner falls back to Slurm via `sbatch` and uses `~/test_cluster/launch.sbatch` by default (override with `SBATCH_SCRIPT=/path/to/your.sbatch`).
This repo also includes `./launch.sbatch`, and the runner prefers the local `launch.sbatch` if present.

## Async orchestrator

For more reliable staged launches (start all services with limited concurrency, then benchmark and clean up) use the asyncio-based runner:
```
python3 yuandong/run_perf_async.py --start-max 2 --bench-max 1
```
Environment defaults are mirrored: `LOG_DIR`, `PORT_BASE`, `PORT_STEP`, `STAGED_MAX_LAUNCH` (start-max), `BENCH_MAX`, `WAIT_SECONDS`, prompt/token settings, and batch prompts. Override configs via `--configs "model[:mem][:tp]" ...`.
The async runner uses the same default prompts and batch prompts as the original bash runner (long Chinese prompt, batch prompt list, max_tokens 10000, warmup_max_tokens 256, repeat_requests 10).

If you just want to launch a simple Qwen3-8B container, just use `yuandong/start_qwen_container.sh`.

Logs: by default all JSONL run logs go under `perf_logs` in the repo root (override with `LOG_DIR`/`--log-dir`). State files live in `perf_logs/state/` when using staged/async modes.
