#!/usr/bin/env python3
"""
Async orchestrator for container performance runs.

Starts multiple services (docker or sbatch), waits for readiness, benchmarks,
and cleans up. Also supports a fake runner for tests.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_CONFIGS = [
    ("Qwen/Qwen3-8B", None, 1),
    ("Qwen/Qwen3-8B", None, 2),
    ("Qwen/Qwen3-8B", None, 4),
    ("Qwen/Qwen3-8B", None, 8),
    # ("deepseek-ai/DeepSeek-V3-0324", None, 8),
    # ("deepseek-ai/DeepSeek-V3-0324", "0.9", 4),
    # ("deepseek-ai/DeepSeek-V3-0324", "0.9", 8),
    # ("deepseek-ai/DeepSeek-V3.2", "0.9", 8),
    # ("nvidia/DeepSeek-V3-0324-NVFP4", None, 4),
    # ("nvidia/DeepSeek-V3-0324-NVFP4", None, 8),
]


@dataclass
class ServiceConfig:
    idx: int
    model: str
    mem_fraction_static: Optional[float]
    tp: int
    port: int
    log_dir: Path
    wait_seconds: int
    poll_interval: int
    image: str
    run_mode: Optional[str]
    sbatch_script: Optional[Path]
    hf_cache: Path
    prompt: str
    max_tokens: int
    warmup_requests: int
    warmup_max_tokens: int
    repeat_requests: int
    temperature: float
    batch_enabled: bool
    batch_prompts: str
    batch_max_tokens: int
    batch_temperature: float
    batch_model: str
    container_name: str
    fake_runner: Optional[Path]


def parse_config(entry: str) -> Tuple[str, Optional[str], int]:
    parts = entry.split(":")
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1] or None, int(parts[2] or 1)


def detect_run_mode(forced: Optional[str]) -> str:
    if forced:
        return forced
    if shutil.which("docker"):
        return "docker"
    if shutil.which("sbatch"):
        return "sbatch"
    raise SystemExit("Neither docker nor sbatch found.")


def find_sbatch_script(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    candidates = [
        Path(__file__).resolve().parent / "launch.sbatch",
        Path.home() / "test_cluster" / "launch.sbatch",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec: trusted host
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
        return False


async def run_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None, label: str = "") -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}{line}")
    return await proc.wait()


def run_cmd(
    cmd: List[str],
    check: bool = False,
    capture: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, env=env)


def start_docker(cfg: ServiceConfig) -> None:
    run_cmd(["docker", "rm", "-f", cfg.container_name], check=False)
    cmd = [
        "docker",
        "run",
        "--gpus",
        "all",
        "--shm-size",
        "32g",
        "-d",
        "--rm",
        "-p",
        f"{cfg.port}:30000",
        "--name",
        cfg.container_name,
        "-v",
        f"{cfg.hf_cache}:/root/.cache/huggingface",
        cfg.image,
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model,
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--tp",
        str(cfg.tp),
        "--mem-fraction-static",
        str(cfg.mem_fraction_static or 0.0),
        "--trust-remote-code",
    ]
    run_cmd(cmd, check=True)


def start_sbatch(cfg: ServiceConfig) -> str:
    script = cfg.sbatch_script
    if not script or not script.exists():
        raise SystemExit(f"sbatch script not found ({script})")
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": cfg.model,
            "PORT": str(cfg.port),
            "MEM_FRACTION_STATIC": str(cfg.mem_fraction_static or ""),
            "IMAGE": cfg.image,
            "TP": str(cfg.tp),
            "HF_CACHE": str(cfg.hf_cache),
        }
    )
    cp = run_cmd(["sbatch", "--parsable", str(script)], capture=True, env=env)
    out = (cp.stdout or "").strip().splitlines()
    if not out:
        raise SystemExit(f"Failed to submit sbatch: {cp.stdout} {cp.stderr}")
    job_id = out[-1].strip()
    if not job_id:
        raise SystemExit(f"Failed to parse job id: {out}")
    return job_id


def job_state(job_id: str) -> str:
    cp = run_cmd(["squeue", "-j", job_id, "-h", "-o", "%T"], capture=True)
    state = (cp.stdout or "").splitlines()
    if state:
        return state[0].strip()
    cp = run_cmd(["scontrol", "show", "job", job_id], capture=True)
    for token in (cp.stdout or "").split():
        if token.startswith("JobState="):
            return token.split("=", 1)[1]
    return ""


def job_host(job_id: str) -> str:
    cp = run_cmd(["squeue", "-j", job_id, "-h", "-o", "%N"], capture=True)
    host = (cp.stdout or "").splitlines()
    if host:
        return host[0].strip()
    cp = run_cmd(["scontrol", "show", "job", job_id], capture=True)
    for token in (cp.stdout or "").split():
        if token.startswith("BatchHost="):
            return token.split("=", 1)[1]
    return ""


async def wait_ready(cfg: ServiceConfig, run_mode: str, job_id: str | None, host_hint: Optional[str]) -> str:
    host = host_hint or ("127.0.0.1" if run_mode == "docker" else "")
    start = time.time()
    last_state = ""
    while time.time() - start < cfg.wait_seconds:
        if run_mode == "sbatch" and job_id:
            state = job_state(job_id)
            last_state = state
            if state in {"FAILED", "TIMEOUT", "CANCELLED", "COMPLETED"}:
                raise SystemExit(f"sbatch job {job_id} terminal state {state}")
            if not host:
                host = job_host(job_id)
        target = host or ""
        if target:
            url = f"http://{target}:{cfg.port}/v1/models"
            if http_ok(url, timeout=2.0):
                print(f"[ready:{cfg.idx}] Ready after {int(time.time()-start)}s on {target}")
                return target
        await asyncio.sleep(cfg.poll_interval)
    raise SystemExit(f"Timed out waiting for service (last_state={last_state})")


def curl_json(url: str, payload: Dict, timeout: float = 30.0) -> Dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec
        return json.loads(resp.read().decode())


def run_requests(cfg: ServiceConfig, host: str, run_id: str) -> Path:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    run_log = cfg.log_dir / f"container-run-{run_id}.jsonl"
    print(f"[bench:{cfg.idx}] Logging to {run_log}")

    def log_line(obj: Dict) -> None:
        with run_log.open("a") as f:
            f.write(json.dumps(obj) + "\n")

    def do_chat(iter_label: str, prompt: str, max_tokens: int) -> Dict:
        url = f"http://{host}:{cfg.port}/v1/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": cfg.temperature,
        }
        t0 = time.time()
        try:
            resp = curl_json(url, payload)
            t1 = time.time()
            usage = resp.get("usage", {})
            ct = usage.get("completion_tokens")
            tokens_per_sec = (ct or 0) / max(t1 - t0, 1e-6) if ct else None
            return {
                "iteration": iter_label,
                "status": "ok",
                "elapsed_sec": t1 - t0,
                "completion_tokens": ct,
                "tokens_per_sec": tokens_per_sec,
                "response": resp,
            }
        except Exception as e:  # noqa: BLE001
            return {"iteration": iter_label, "status": "error", "error": str(e)}

    def do_batch(iter_label: str) -> Dict:
        url = f"http://{host}:{cfg.port}/v1/completions"
        prompts = [p for p in cfg.batch_prompts.splitlines() if p.strip()]
        payload = {
            "model": cfg.batch_model,
            "prompt": prompts,
            "max_tokens": cfg.batch_max_tokens,
            "temperature": cfg.batch_temperature,
        }
        t0 = time.time()
        try:
            resp = curl_json(url, payload)
            t1 = time.time()
            usage = resp.get("usage", {})
            ct = usage.get("completion_tokens")
            tokens_per_sec = (ct or 0) / max(t1 - t0, 1e-6) if ct else None
            return {
                "iteration": iter_label,
                "status": "ok",
                "elapsed_sec": t1 - t0,
                "completion_tokens": ct,
                "tokens_per_sec": tokens_per_sec,
                "response": resp,
            }
        except Exception as e:  # noqa: BLE001
            return {"iteration": iter_label, "status": "error", "error": str(e)}

    for i in range(1, cfg.warmup_requests + 1):
        res = do_chat(f"warmup-{i}", cfg.prompt, cfg.warmup_max_tokens)
        log_line(res)

    tokens: List[float] = []
    elapsed: List[float] = []
    for i in range(1, cfg.repeat_requests + 1):
        res = do_chat(f"run-{i}", cfg.prompt, cfg.max_tokens)
        log_line(res)
        if res.get("tokens_per_sec") is not None:
            tokens.append(res["tokens_per_sec"])
        if res.get("elapsed_sec") is not None:
            elapsed.append(res["elapsed_sec"])

    if cfg.batch_enabled:
        res = do_batch("batch-1")
        log_line(res)

    def stats(vals: List[float]) -> Dict:
        if not vals:
            return {"count": 0, "mean": None, "std": None}
        import math
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        return {"count": len(vals), "mean": mean, "std": std}

    summary = {
        "mem_fraction_static": cfg.mem_fraction_static,
        "tokens_per_sec_stats": stats(tokens),
        "elapsed_sec_stats": stats(elapsed),
        "host": host,
        "port": cfg.port,
    }
    log_line(summary)
    return run_log


async def launch_bench_cleanup(cfg: ServiceConfig, start_sem: asyncio.Semaphore, bench_sem: asyncio.Semaphore) -> bool:
    run_mode = detect_run_mode(cfg.run_mode)
    job_id: Optional[str] = None
    host_hint: Optional[str] = None
    try:
        async with start_sem:
            if cfg.fake_runner:
                env = os.environ.copy()
                env.update({"PHASE": "start", "IDX": str(cfg.idx), "PORT": str(cfg.port), "MODEL_PATH": cfg.model})
                rc = await run_subprocess([sys.executable, str(cfg.fake_runner)], env=env, label=f"start:{cfg.idx}")
                if rc != 0:
                    return False
                host_hint = "127.0.0.1"
            else:
                if run_mode == "docker":
                    await asyncio.to_thread(start_docker, cfg)
                    host_hint = "127.0.0.1"
                else:
                    job_id = await asyncio.to_thread(start_sbatch, cfg)
                    node = job_host(job_id) or "<pending>"
                    node_ip = resolve_host_ip(node) if node and node != "<pending>" else "<pending>"
                    print(f"[start:{cfg.idx}] Submitted sbatch job {job_id} on {node} ({node_ip}), expected port {cfg.port}")

            if cfg.fake_runner:
                host = host_hint or "127.0.0.1"
            else:
                host = await wait_ready(cfg, run_mode, job_id, host_hint)

        async with bench_sem:
            if cfg.fake_runner:
                env = os.environ.copy()
                env.update({"PHASE": "bench", "IDX": str(cfg.idx), "PORT": str(cfg.port), "MODEL_PATH": cfg.model})
                rc = await run_subprocess([sys.executable, str(cfg.fake_runner)], env=env, label=f"bench:{cfg.idx}")
                ok = rc == 0
            else:
                run_id = time.strftime("%Y%m%d-%H%M%S")
                await asyncio.to_thread(run_requests, cfg, host, run_id)
                ok = True
            if ok:
                print(f"[bench:{cfg.idx}] completed")
            else:
                print(f"[bench:{cfg.idx}] failed")

        return True
    except Exception as e:  # noqa: BLE001
        print(f"[error:{cfg.idx}] {e}")
        return False
    finally:
        try:
            if cfg.fake_runner:
                env = os.environ.copy()
                env.update({"PHASE": "cleanup", "IDX": str(cfg.idx)})
                await run_subprocess([sys.executable, str(cfg.fake_runner)], env=env, label=f"cleanup:{cfg.idx}")
            else:
                if run_mode == "docker":
                    await asyncio.to_thread(run_cmd, ["docker", "stop", cfg.container_name], False, False)
                elif run_mode == "sbatch" and job_id:
                    await asyncio.to_thread(run_cmd, ["scancel", job_id], False, False)
        except Exception as e:  # noqa: BLE001
            print(f"[cleanup:{cfg.idx}] {e}")


def build_service_configs(args: argparse.Namespace) -> List[ServiceConfig]:
    configs_in = [parse_config(c) for c in (args.configs or [])] or DEFAULT_CONFIGS
    cfgs: List[ServiceConfig] = []

    sbatch_script = find_sbatch_script(args.sbatch_script)
    print(f"sbatch_script: {sbatch_script}")
    for idx, (model, mem, tp) in enumerate(configs_in):
        port = args.port_base + idx * args.port_step
        cfgs.append(
            ServiceConfig(
                idx=idx,
                model=model,
                mem_fraction_static=float(mem) if mem else None,
                tp=tp,
                port=port,
                log_dir=Path(args.log_dir),
                wait_seconds=args.wait_seconds,
                poll_interval=args.poll_interval,
                image=args.image,
                run_mode=args.run_mode,
                sbatch_script=sbatch_script,
                hf_cache=Path(args.hf_cache),
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                warmup_requests=args.warmup_requests,
                warmup_max_tokens=args.warmup_max_tokens,
                repeat_requests=args.repeat_requests,
                temperature=args.temperature,
                batch_enabled=args.batch_enabled,
                batch_prompts=args.batch_prompts,
                batch_max_tokens=args.batch_max_tokens,
                batch_temperature=args.batch_temperature,
                batch_model=args.batch_model,
                container_name=f"bench-{idx}-tp-{tp}",
                fake_runner=Path(args.fake_runner).resolve() if args.fake_runner else None,
            )
        )
    return cfgs


async def main_async(args: argparse.Namespace) -> int:
    start_sem = asyncio.Semaphore(max(args.start_max, 1))
    bench_sem = asyncio.Semaphore(max(args.bench_max, 1))

    cfgs = build_service_configs(args)

    tasks = [asyncio.create_task(launch_bench_cleanup(cfg, start_sem, bench_sem)) for cfg in cfgs]
    results = await asyncio.gather(*tasks)
    failures = [i for i, ok in enumerate(results) if not ok]
    if failures:
        print(f"Failures in jobs: {failures}")
        return 1
    print("All benchmarks completed.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async container perf runner")
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "perf_logs"))
    parser.add_argument("--port-base", type=int, default=int(os.environ.get("PORT_BASE", 30000)))
    parser.add_argument("--port-step", type=int, default=int(os.environ.get("PORT_STEP", 10)))
    parser.add_argument("--start-max", type=int, default=int(os.environ.get("STAGED_MAX_LAUNCH", 2)))
    parser.add_argument("--bench-max", type=int, default=int(os.environ.get("BENCH_MAX", 1)))
    parser.add_argument("--wait-seconds", type=int, default=int(os.environ.get("WAIT_SECONDS", 7200)))
    parser.add_argument("--poll-interval", type=int, default=int(os.environ.get("POLL_INTERVAL", 5)))
    parser.add_argument("--image", default=os.environ.get("IMAGE", "lmsysorg/sglang:latest"))
    parser.add_argument("--run-mode", default=os.environ.get("RUN_MODE"))
    parser.add_argument("--sbatch-script", default=os.environ.get("SBATCH_SCRIPT"))
    parser.add_argument("--hf-cache", default=os.environ.get("HF_CACHE", str(Path.home() / ".cache" / "huggingface")))
    parser.add_argument(
        "--prompt",
        default=os.environ.get("PROMPT", "仿照过秦论写篇过隋论。请经过认真思考后写出，长度约为一万字。"),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", 10000)))
    parser.add_argument("--warmup-requests", type=int, default=int(os.environ.get("WARMUP_REQUESTS", 1)))
    parser.add_argument("--warmup-max-tokens", type=int, default=int(os.environ.get("WARMUP_MAX_TOKENS", 256)))
    parser.add_argument("--repeat-requests", type=int, default=int(os.environ.get("REPEAT_REQUESTS", 10)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.7)))
    parser.add_argument("--batch-enabled", action="store_true", default=os.environ.get("BATCH_ENABLED", "1") == "1")
    parser.add_argument(
        "--batch-prompts",
        default=os.environ.get(
            "BATCH_PROMPTS",
            "What is the meaning of life?\n"
            "Write a one-sentence summary of why the sky is blue.\n"
            'Explain Docker port conflicts and how to fix "port is already allocated".\n'
            "Give three tips to speed up Python code without changing results.\n"
            "Draft a polite email asking for an invoice correction.\n"
            "List five creative uses for a Raspberry Pi at home.\n"
            "Explain the difference between TCP and UDP in simple terms.\n"
            "Write a haiku about winter in San Francisco.\n"
            "Suggest a 2-day itinerary for Vancouver in December.\n"
            "Provide a bash one-liner to find processes listening on port 30000.\n"
            'Explain what "post-money valuation" means with a numeric example.\n'
            "Generate a JSON object describing a user profile (name, hobbies, skills).\n"
            "Write a short bedtime story featuring a teddy bear astronaut.\n"
            'Summarize the key idea of "grokking" in neural networks in two sentences.\n'
            "Give a checklist for reviewing a startup equity agreement.\n"
            'Explain what "anti-dilution" is and the common types.\n'
            "Propose five A/B test ideas for a landing page conversion boost.\n"
            "Write a Python function to compute moving average over a list.\n"
            "Explain how to batch requests with curl using JSON arrays.\n"
            'Give a concise definition of "representation theory" for ML engineers.\n'
            "Suggest a healthy 15-minute lunch recipe with common ingredients.",
        ),
    )
    parser.add_argument("--batch-max-tokens", type=int, default=int(os.environ.get("BATCH_MAX_TOKENS", 10000)))
    parser.add_argument("--batch-temperature", type=float, default=float(os.environ.get("BATCH_TEMPERATURE", 0.7)))
    parser.add_argument("--batch-model", default=os.environ.get("BATCH_MODEL", "default"))
    parser.add_argument("--configs", nargs="*", help="model[:mem][:tp] entries")
    parser.add_argument("--fake-runner", help="Path to fake runner for tests (start/bench/cleanup via PHASE env)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
