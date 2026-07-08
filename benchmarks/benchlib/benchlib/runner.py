# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Parallel job runner with GPU distribution."""

import os
import subprocess
import sys
from math import ceil


def run_parallel(script: str, all_jobs: list[dict], gpus: list[int], jobs_per_gpu: int = 1, extra_args: list = None):
    """Distribute jobs across GPUs and run in parallel subprocesses.

    Args:
        script: Path to the runner script (e.g. baselines/optimus5p/run.py)
        all_jobs: List of job dicts, each with keys to pass as CLI args
        gpus: List of GPU IDs to use
        jobs_per_gpu: Max concurrent jobs per GPU
        extra_args: Additional CLI args to pass to each worker
    """
    total_workers = len(gpus) * jobs_per_gpu
    # Assign GPU to each worker slot (round-robin)
    gpu_assignments = [gpus[i % len(gpus)] for i in range(total_workers)]

    # Split jobs into chunks for each worker
    chunk_size = ceil(len(all_jobs) / total_workers)
    chunks = [all_jobs[i:i + chunk_size] for i in range(0, len(all_jobs), chunk_size)]

    processes = []
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        gpu_id = gpu_assignments[worker_id]
        names = [j["name"] for j in chunk]

        cmd = [
            sys.executable, script,
            "--gpu", str(gpu_id),
            "--names", *names,
            "--worker-id", str(worker_id),
            *(extra_args or []),
        ]
        # Pass through config if present in first job
        if "config" in chunk[0]:
            cmd.extend(["--config", chunk[0]["config"]])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Ensure conda env's libstdc++ is found
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"

        print(f"  [worker {worker_id}] gpu={gpu_id} names={names}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all
    failures = 0
    for p in processes:
        p.wait()
        if p.returncode != 0:
            failures += 1

    print(f"\nDone. {len(processes)} workers, {failures} failures.")
    return failures
