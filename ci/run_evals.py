#!/usr/bin/env python3
"""Eval regression gate.

Runs the golden-task suite against a live bridge and exits non-zero when the
pass rate drops below --min-pass (default 1.0). Intended for CI and the
nightly cron. Requires EVAL_API_KEY (admin or user key) and EVAL_BASE_URL.

Usage: python ci/run_evals.py [--min-pass 1.0]
"""
import argparse
import json
import os
import sys
import time

import httpx

BASE = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8001")
KEY = os.environ.get("EVAL_API_KEY", "")
COMBO = os.environ.get("EVAL_COMBO_ID", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pass", type=float, default=1.0)
    args = ap.parse_args()

    if not KEY:
        print("EVAL_API_KEY not set — required (admin or user key)")
        return 2
    with httpx.Client(timeout=900.0) as c:
        r = c.post(f"{BASE}/evals/run", json={},
                   headers={"Authorization": f"Bearer {KEY}"})
        if r.status_code != 200:
            print(f"eval run failed: HTTP {r.status_code}: {r.text[:300]}")
            return 2
        body = r.json()

    def _pass_rate(run: dict) -> float:
        for field in ("task_results", "task_results_detail", "results"):
            tasks = run.get(field) or []
            if tasks:
                passed = sum(1 for t in tasks if t.get("passed") or t.get("ok") or t.get("score", 0) >= 1)
                return passed / len(tasks) if tasks else 1.0
        return float(run.get("pass_rate", 1.0))

    rate = _pass_rate(body)
    print(f"eval pass rate: {rate:.2f} (min {args.min_pass})")
    if rate < args.min_pass:
        print("REGRESSION GATE: FAILED")
        return 1
    print("REGRESSION GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
