# Cluster scripts

Run the scripts to test the performance of a cluster.

For Nebius cluster, login to any worker node, download the docker image with `docker pull lmsysorg/sglang:latest` and run the following command. 
```
bash run_perf.sh
```
It will automatically launch docker, wait until it is ready, and run performance test. The test is simple, just to load the SGLang server (all kernels are recompiled from scratch) and then send single/batch requests. We want to check the loading time and the thoroughputs with various TP settings.   

For Andromeda cluster, login to the main logging node, and run 
```
STAGED=1 STAGED_MAX_LAUNCH=2 bash run_perf.sh
``` 
If docker is not available, the runner falls back to Slurm via `sbatch` and uses `~/test_cluster/launch.sbatch` by default (override with `SBATCH_SCRIPT=/path/to/your.sbatch`).
This repo also includes `./launch.sbatch`, and the runner prefers the local `launch.sbatch` if present.
To overlap startup waits across multiple jobs while still benchmarking one-by-one, run with `STAGED=1` (e.g. `STAGED=1 bash run_perf.sh`).

If you just want to launch a simple Qwen3-8B container, just use `start_qwen_container.sh`.
