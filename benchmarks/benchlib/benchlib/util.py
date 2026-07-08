# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Shared small utilities for the benchmark runners: cache check, atomic JSON
write, and a timing context manager."""
import json
import os
import time


def check_cached(output_dir: str):
    """Return cached metrics if predictions.tsv + a valid metrics.json exist, else None.

    A malformed metrics.json (caught mid-write by a concurrent reader, or a
    partially/doubly-written file left by an interrupted run) must NOT crash the
    worker. Retry a few times to ride out a transient concurrent write, then treat
    a persistently unreadable cache as "not cached" (the fold reruns).
    """
    pred_path = os.path.join(output_dir, "predictions.tsv")
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not (os.path.exists(pred_path) and os.path.exists(metrics_path)):
        return None
    for _ in range(3):
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(0.2)
    return None


def write_json_atomic(path: str, obj):
    """Write JSON atomically (temp file + os.replace) so concurrent readers never
    observe a truncated / partial / doubly-written file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class Timer:
    """Context manager that records elapsed seconds."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.time() - self._start, 1)
