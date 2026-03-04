#!/usr/bin/env python3
"""Environment/setup doctor for SOC readiness foundation workflows."""

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str  # required|optional
    detail: str
    fix: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description="SOC readiness doctor")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run tools/soc_readiness_smoke.py as part of checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    return parser.parse_args()


def _module_exists(module_name):
    return importlib.util.find_spec(module_name) is not None


def _binary_exists(name):
    return shutil.which(str(name)) is not None


def _add_import_check(results, module_name, severity="required", pip_name=None):
    ok = _module_exists(module_name)
    pip_name = pip_name or module_name
    results.append(
        CheckResult(
            name=f"python_module:{module_name}",
            ok=ok,
            severity=severity,
            detail="installed" if ok else "missing",
            fix="" if ok else f"pip install {pip_name}",
        )
    )


def _add_file_check(results, path, severity="required"):
    path = Path(path)
    ok = path.exists()
    results.append(
        CheckResult(
            name=f"file:{path}",
            ok=ok,
            severity=severity,
            detail="exists" if ok else "missing",
            fix="" if ok else f"Expected file at {path}",
        )
    )


def _run_smoke(repo_root):
    smoke_script = repo_root / "tools" / "soc_readiness_smoke.py"
    if not smoke_script.exists():
        return False, "smoke script missing"
    proc = subprocess.run(
        [sys.executable, str(smoke_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
    return False, detail


def _run_mlops_smoke(repo_root):
    smoke_script = repo_root / "tools" / "soc_mlops_smoke.py"
    if not smoke_script.exists():
        return False, "mlops smoke script missing"
    proc = subprocess.run(
        [sys.executable, str(smoke_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
    return False, detail


def _run_services_smoke(repo_root):
    smoke_script = repo_root / "tools" / "soc_services_smoke.py"
    if not smoke_script.exists():
        return False, "services smoke script missing"
    proc = subprocess.run(
        [sys.executable, str(smoke_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
    return False, detail


def _run_integration_probe(repo_root):
    probe_script = repo_root / "tools" / "soc_integration_probe.py"
    if not probe_script.exists():
        return False, "integration probe script missing"
    proc = subprocess.run(
        [sys.executable, str(probe_script), "--json"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
    return False, detail


def _run_grpc_smoke(repo_root):
    smoke_script = repo_root / "tools" / "soc_grpc_smoke.py"
    if not smoke_script.exists():
        return False, "grpc smoke script missing"
    proc = subprocess.run(
        [sys.executable, str(smoke_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
    return False, detail


def run_checks(args):
    root = Path(args.repo_root).resolve()
    results = []

    results.append(
        CheckResult(
            name="python_version",
            ok=sys.version_info[:2] >= (3, 8),
            severity="required",
            detail=f"detected {sys.version_info.major}.{sys.version_info.minor}",
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

    required_files = [
        "tools/soc_readiness_smoke.py",
        "tools/soc_readiness_doctor.py",
        "tools/soc_coordination_sim.py",
        "tools/soc_mlops_smoke.py",
        "tools/soc_services_smoke.py",
        "tools/soc_integration_probe.py",
        "tools/soc_grpc_server.py",
        "tools/soc_grpc_smoke.py",
        "data/soc/example_threat_taxonomy_v2.json",
        "data/soc/example_soc_runtime_config.json",
        "data/soc/example_onvif_inventory.json",
        "lavis/common/soc/runtime.py",
        "lavis/common/soc/runtime_services.py",
        "lavis/common/soc/schemas.py",
        "lavis/common/soc/message_bus.py",
        "lavis/common/soc/stores.py",
        "lavis/common/soc/federation.py",
        "lavis/common/soc/calibration.py",
        "lavis/common/soc/mlops.py",
        "lavis/common/soc/security.py",
        "docs/security_readiness_roadmap_implementation.md",
    ]
    for rel in required_files:
        _add_file_check(results, root / rel, severity="required")

    _add_import_check(results, "numpy", severity="required")
    _add_import_check(results, "scipy", severity="optional")
    _add_import_check(results, "torch", severity="optional")
    _add_import_check(results, "cv2", severity="optional", pip_name="opencv-python")

    # Optional integrations for production adapters.
    _add_import_check(results, "nats", severity="optional", pip_name="nats-py")
    _add_import_check(results, "redis", severity="optional", pip_name="redis")
    _add_import_check(
        results, "clickhouse_connect", severity="optional", pip_name="clickhouse-connect"
    )
    _add_import_check(results, "grpc", severity="optional", pip_name="grpcio")
    results.append(
        CheckResult(
            name="binary:protoc",
            ok=_binary_exists("protoc"),
            severity="optional",
            detail="installed" if _binary_exists("protoc") else "missing",
            fix="" if _binary_exists("protoc") else "Install protoc (Protocol Buffers compiler).",
        )
    )

    if args.run_smoke:
        ok, detail = _run_smoke(root)
        results.append(
            CheckResult(
                name="soc_smoke_test",
                ok=ok,
                severity="required",
                detail=detail,
                fix="Run python tools/soc_readiness_smoke.py and inspect failures.",
            )
        )
        ok_mlops, detail_mlops = _run_mlops_smoke(root)
        results.append(
            CheckResult(
                name="soc_mlops_smoke_test",
                ok=ok_mlops,
                severity="required",
                detail=detail_mlops,
                fix="Run python tools/soc_mlops_smoke.py and inspect failures.",
            )
        )
        ok_services, detail_services = _run_services_smoke(root)
        results.append(
            CheckResult(
                name="soc_services_smoke_test",
                ok=ok_services,
                severity="required",
                detail=detail_services,
                fix="Run python tools/soc_services_smoke.py and inspect failures.",
            )
        )
        ok_probe, detail_probe = _run_integration_probe(root)
        results.append(
            CheckResult(
                name="soc_integration_probe",
                ok=ok_probe,
                severity="required",
                detail=detail_probe,
                fix="Run python tools/soc_integration_probe.py --json and inspect failures.",
            )
        )
        grpc_ready = _module_exists("grpc") and _binary_exists("protoc")
        if grpc_ready:
            ok_grpc, detail_grpc = _run_grpc_smoke(root)
            results.append(
                CheckResult(
                    name="soc_grpc_smoke_test",
                    ok=ok_grpc,
                    severity="required",
                    detail=detail_grpc,
                    fix="Run python tools/soc_grpc_smoke.py and inspect failures.",
                )
            )
        else:
            reasons = []
            if not _module_exists("grpc"):
                reasons.append("missing grpcio")
            if not _binary_exists("protoc"):
                reasons.append("missing protoc")
            results.append(
                CheckResult(
                    name="soc_grpc_smoke_test",
                    ok=False,
                    severity="optional",
                    detail=f"skipped ({', '.join(reasons)})",
                    fix="Install grpcio and protoc, then run python tools/soc_grpc_smoke.py.",
                )
            )

    return results


def summarize(results):
    required_failures = [r for r in results if not r.ok and r.severity == "required"]
    optional_failures = [r for r in results if not r.ok and r.severity == "optional"]
    return {
        "soc_ready": len(required_failures) == 0,
        "required_failures": len(required_failures),
        "optional_failures": len(optional_failures),
    }


def print_human(results, summary):
    print(f"System: {platform.system()} {platform.release()}")
    print(f"SOC Ready: {'YES' if summary['soc_ready'] else 'NO'}")
    print("")
    for r in results:
        status = "OK" if r.ok else ("WARN" if r.severity == "optional" else "FAIL")
        print(f"[{status}] {r.name} :: {r.detail}")
        if not r.ok and r.fix:
            print(f"      fix: {r.fix}")


def main():
    args = parse_args()
    results = run_checks(args)
    summary = summarize(results)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "results": [r.__dict__ for r in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary["soc_ready"] else 1

    print_human(results, summary)
    return 0 if summary["soc_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
