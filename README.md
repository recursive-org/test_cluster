# Cluster scripts

Run the scripts to test the performance of a cluster.

For Nebius cluster, login to any worker node, download the docker image with `docker pull lmsysorg/sglang:latest` and run the following command. 
```
bash run_perf.sh
```
It will automatically launch docker, wait until it is ready, and run performance test. The test is simple, just to load the SGLang server (all kernels are recompiled from scratch) and then send single/batch requests. We want to check the loading time and the thoroughputs with various TP settings.   

For Andromeda cluster, login to the main logging node, and run 
```
bash run_perf.sh
``` 

If you just want to launch a simple Qwen3-8B container, just use `start_qwen_container.sh`.
