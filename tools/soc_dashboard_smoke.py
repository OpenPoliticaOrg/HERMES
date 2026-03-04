#!/usr/bin/env python3
"""Smoke test for SOC dashboard client."""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SOC dashboard smoke test")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="Timeout waiting for dashboard readiness.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON report.",
    )
    return parser.parse_args()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _http_json(url, method="GET", payload=None, timeout=2.0):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    dashboard_script = repo_root / "tools" / "soc_dashboard_client.py"
    if not dashboard_script.exists():
        print(f"[FAIL] dashboard script missing: {dashboard_script}")
        return 1

    grpc_port = _free_port()
    dash_port = _free_port()
    grpc_target = f"127.0.0.1:{grpc_port}"
    base = f"http://127.0.0.1:{dash_port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(dashboard_script),
            "--repo-root",
            str(repo_root),
            "--grpc-target",
            grpc_target,
            "--host",
            "127.0.0.1",
            "--port",
            str(dash_port),
            "--spawn-grpc-server",
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = {"status": "error", "base_url": base, "grpc_target": grpc_target}
    try:
        deadline = time.time() + max(2.0, float(args.timeout_seconds))
        ready = False
        while time.time() < deadline:
            try:
                _ = _http_json(f"{base}/api/health")
                ready = True
                break
            except Exception:
                time.sleep(0.2)
        if not ready:
            raise RuntimeError("dashboard did not become ready before timeout")

        ingest = _http_json(f"{base}/api/ingest-demo", method="POST")
        if ingest.get("status") != "ok":
            raise RuntimeError("dashboard ingest-demo returned non-ok status")

        snapshot = _http_json(f"{base}/api/snapshot")
        if int(snapshot.get("threat_event_count", 0)) < 1:
            raise RuntimeError("snapshot threat_event_count < 1")
        if int(snapshot.get("case_count", 0)) < 1:
            raise RuntimeError("snapshot case_count < 1")
        case_items = snapshot.get("recent_cases", [])
        if not isinstance(case_items, list) or len(case_items) < 1:
            raise RuntimeError("snapshot recent_cases missing")
        case_id = str((case_items[0] or {}).get("case_id", ""))
        if not case_id:
            raise RuntimeError("snapshot case_id missing")

        ack = _http_json(
            f"{base}/api/case-action",
            method="POST",
            payload={
                "action": "ack",
                "case_id": case_id,
                "analyst_id": "analyst_smoke",
                "reason": "ack from dashboard smoke",
            },
        )
        if ack.get("status") != "ok":
            raise RuntimeError("dashboard case ack returned non-ok status")

        confirm = _http_json(
            f"{base}/api/case-action",
            method="POST",
            payload={
                "action": "confirm",
                "case_id": case_id,
                "analyst_id": "analyst_smoke",
                "reason": "confirm from dashboard smoke",
            },
        )
        if confirm.get("status") != "ok":
            raise RuntimeError("dashboard case confirm returned non-ok status")

        snapshot_after = _http_json(f"{base}/api/snapshot")
        case_after = None
        for item in snapshot_after.get("recent_cases", []):
            if str((item or {}).get("case_id", "")) == case_id:
                case_after = item
                break
        if not isinstance(case_after, dict) or str(case_after.get("state", "")) != "confirmed":
            raise RuntimeError("case state was not confirmed after dashboard action")

        report["status"] = "ok"
        report["checks"] = {
            "ingest": ingest,
            "snapshot_counts": {
                "threat_event_count": snapshot.get("threat_event_count"),
                "case_count": snapshot.get("case_count"),
                "profile_id": snapshot.get("profile_id"),
            },
            "case_actions": {
                "case_id": case_id,
                "ack_status": ack.get("status"),
                "confirm_status": confirm.get("status"),
                "final_state": case_after.get("state"),
            },
        }
        if args.print_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("[OK] SOC dashboard smoke test passed.")
            print("[OK] dashboard:", base)
            print("[OK] grpc:", grpc_target)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        if args.print_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"[FAIL] SOC dashboard smoke test failed: {exc}")
        return 1
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
