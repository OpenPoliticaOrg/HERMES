#!/usr/bin/env python3
"""
Environment/setup doctor for context-conditional event + Markov pipeline.
"""

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str  # "required" | "optional"
    detail: str
    fix: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Context Markov setup doctor")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model checkpoint path to validate.",
    )
    parser.add_argument(
        "--video-source",
        default=None,
        help="Optional camera index (e.g. 0) or video path to probe.",
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run the fast smoke script as part of doctor checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    return parser.parse_args()


def _resolve(root, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def _module_exists(module_name):
    return importlib.util.find_spec(module_name) is not None


def _add_import_check(results, module_name, severity="required", pip_name=None):
    ok = _module_exists(module_name)
    pip_name = pip_name or module_name
    fix = f"pip install {pip_name}" if not ok else ""
    results.append(
        CheckResult(
            name=f"python_module:{module_name}",
            ok=ok,
            severity=severity,
            detail="installed" if ok else "missing",
            fix=fix,
        )
    )


def _add_runtime_import_check(
    results, module_name, severity="required", fix="", prepend_paths=None
):
    added_paths = []
    for p in prepend_paths or []:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)
            added_paths.append(p)
    try:
        __import__(module_name)
        ok = True
        detail = "imported"
        fix_msg = ""
    except Exception as exc:
        ok = False
        detail = f"import failed: {exc}"
        fix_msg = fix or "Install missing dependencies from requirements.txt."
    finally:
        for p in added_paths:
            if p in sys.path:
                sys.path.remove(p)
    results.append(
        CheckResult(
            name=f"runtime_import:{module_name}",
            ok=ok,
            severity=severity,
            detail=detail,
            fix=fix_msg,
        )
    )


def _add_file_check(results, path, severity="required"):
    ok = Path(path).exists()
    results.append(
        CheckResult(
            name=f"file:{Path(path)}",
            ok=ok,
            severity=severity,
            detail="exists" if ok else "missing",
            fix="" if ok else f"Expected file at {path}",
        )
    )


def _probe_video_source(video_source):
    try:
        import cv2
    except Exception as exc:
        return False, f"cv2 unavailable: {exc}"

    src = video_source
    try:
        src = int(video_source)
    except (TypeError, ValueError):
        pass

    cap = cv2.VideoCapture(src)
    ok = cap.isOpened()
    cap.release()
    if ok:
        return True, "opened successfully"
    return False, f"failed to open source: {video_source}"


def _run_smoke(root):
    script = root / "run_scripts/context_markov/test.sh"
    if not script.exists():
        return False, "smoke script missing"
    try:
        proc = subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        msg = proc.stdout.strip() + "\n" + proc.stderr.strip()
        return False, msg.strip()
    except Exception as exc:
        return False, str(exc)


def _torch_detail():
    try:
        import torch

        return True, {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0,
        }
    except Exception as exc:
        return False, {"error": str(exc)}


def _python_version_ok():
    major, minor = sys.version_info[:2]
    return (major, minor) >= (3, 8), f"{major}.{minor}"


def run_checks(args):
    root = Path(args.repo_root).resolve()
    results = []

    # System checks
    py_ok, py_ver = _python_version_ok()
    results.append(
        CheckResult(
            name="python_version",
            ok=py_ok,
            severity="required",
            detail=f"detected {py_ver}",
            fix="Use Python 3.8+",
        )
    )
    results.append(
        CheckResult(
            name="repo_root",
            ok=root.exists(),
            severity="required",
            detail=str(root),
            fix="Pass --repo-root /path/to/HERMES",
        )
    )

    # Required project files
    for rel in [
        "stream_online.py",
        "tools/context_markov_smoke.py",
        "run_scripts/context_markov/test.sh",
        "run_scripts/context_markov/live.sh",
        "run_scripts/context_markov/sweep.sh",
        "data/taxonomy/example_event_taxonomy.json",
        "data/taxonomy/example_observation_classifiers.json",
        "data/taxonomy/example_markov_chain.json",
    ]:
        _add_file_check(results, root / rel, severity="required")

    # Module checks
    # Minimal for smoke
    _add_import_check(results, "numpy", severity="required")
    # Full stack for run/inference
    _add_import_check(results, "torch", severity="required")
    _add_import_check(results, "torchvision", severity="required")
    _add_import_check(results, "omegaconf", severity="required")
    _add_import_check(results, "transformers", severity="required")
    _add_import_check(results, "einops", severity="required")
    _add_import_check(results, "iopath", severity="required")
    _add_import_check(results, "webdataset", severity="required")
    _add_import_check(results, "fairscale", severity="required")
    _add_import_check(results, "pycocoevalcap", severity="required")
    _add_import_check(results, "decord", severity="optional")
    _add_import_check(results, "cv2", severity="required", pip_name="opencv-python")
    _add_runtime_import_check(
        results,
        "lavis",
        severity="required",
        fix="Run `pip install -r requirements.txt` and ensure local lavis imports resolve.",
        prepend_paths=[root],
    )

    # Torch CUDA visibility (optional)
    torch_ok, torch_info = _torch_detail()
    if torch_ok:
        results.append(
            CheckResult(
                name="torch_cuda",
                ok=True,
                severity="optional",
                detail=json.dumps(torch_info, sort_keys=True),
                fix="",
            )
        )
    else:
        results.append(
            CheckResult(
                name="torch_cuda",
                ok=False,
                severity="optional",
                detail=json.dumps(torch_info, sort_keys=True),
                fix="Install torch with CUDA if GPU inference is needed.",
            )
        )

    # Optional checkpoint existence
    if args.checkpoint:
        ckpt = _resolve(root, args.checkpoint)
        _add_file_check(results, ckpt, severity="required")

    # Optional video source probe
    if args.video_source is not None:
        ok, detail = _probe_video_source(args.video_source)
        results.append(
            CheckResult(
                name=f"video_source:{args.video_source}",
                ok=ok,
                severity="optional",
                detail=detail,
                fix="Check camera permissions or video file path.",
            )
        )

    # Optional smoke test execution
    if args.run_smoke:
        ok, detail = _run_smoke(root)
        results.append(
            CheckResult(
                name="smoke_test",
                ok=ok,
                severity="required",
                detail=detail,
                fix="Run bash run_scripts/context_markov/test.sh and inspect output.",
            )
        )

    return results


def summarize(results):
    required_failures = [r for r in results if (not r.ok and r.severity == "required")]
    optional_failures = [r for r in results if (not r.ok and r.severity == "optional")]

    smoke_ready = len(required_failures) == 0
    live_ready = smoke_ready and all(
        r.ok
        for r in results
        if (
            r.name.startswith("python_module:")
            and r.name.split(":", 1)[1]
            in {
                "torch",
                "torchvision",
                "omegaconf",
                "transformers",
                "einops",
                "iopath",
                "webdataset",
                "fairscale",
                "pycocoevalcap",
                "cv2",
            }
        )
        or r.name == "runtime_import:lavis"
    )
    return {
        "smoke_ready": smoke_ready,
        "live_ready": live_ready,
        "required_failures": len(required_failures),
        "optional_failures": len(optional_failures),
    }


def print_human(results, summary):
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Smoke Ready: {'YES' if summary['smoke_ready'] else 'NO'}")
    print(f"Live Ready:  {'YES' if summary['live_ready'] else 'NO'}")
    print("")

    for r in results:
        status = "OK" if r.ok else ("WARN" if r.severity == "optional" else "FAIL")
        print(f"[{status}] {r.name} :: {r.detail}")
        if (not r.ok) and r.fix:
            print(f"      fix: {r.fix}")


def main():
    args = parse_args()
    results = run_checks(args)
    summary = summarize(results)

    if args.json:
        payload = {
            "summary": summary,
            "results": [r.__dict__ for r in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print_human(results, summary)
    return 0 if summary["smoke_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
