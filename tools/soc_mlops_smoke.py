#!/usr/bin/env python3
"""Smoke test for SOC security + MLOps primitives."""

import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SOC MLOps smoke test")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON report.",
    )
    return parser.parse_args()


def _load_soc_modules(repo_root):
    soc_dir = repo_root / "lavis" / "common" / "soc"
    pkg_name = "soc_mlops_local"

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(soc_dir)]
    sys.modules[pkg_name] = pkg

    modules = {}
    for name in ["mlops", "security"]:
        path = soc_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
        sys.modules[f"{pkg_name}.{name}"] = module
    return modules


def fail(message):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()

    soc = _load_soc_modules(root)
    mlops = soc["mlops"]
    security = soc["security"]

    # 1) Model registry sign/verify.
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact = Path(tmp_dir) / "artifact.bin"
        artifact.write_bytes(b"demo_model_payload_v1")

        registry = mlops.SignedModelRegistry(signing_key="unit_test_key")
        entry = registry.register_model(
            model_id="demo_model",
            version="v1",
            artifact_path=str(artifact),
            metadata={"task": "test"},
        )
        if not registry.verify_model(entry):
            fail("Model registry verification failed for registered artifact.")

    # 2) Canary rollout deterministic assignment.
    canary = mlops.CanaryRolloutManager(default_ratio=0.2)
    canary.set_rollout(
        rollout_id="rollout_a",
        baseline_profile="edge_cpu_profile",
        canary_profile="edge_gpu_profile",
        canary_ratio=0.2,
    )
    assignments = [
        canary.resolve_profile("rollout_a", site_id=f"site_{i}", camera_id="cam0")
        for i in range(100)
    ]
    canary_count = sum(1 for x in assignments if x == "edge_gpu_profile")
    if canary_count <= 0 or canary_count >= 100:
        fail("Canary assignment is degenerate; expected mixed baseline/canary outputs.")

    # 2b) Guardrail policy + rollback trigger.
    guardrails = mlops.RolloutGuardrailPolicy(
        enabled=True,
        consecutive_alarm_windows=2,
        min_samples_for_slo=8,
        rollback_on_drift=True,
        rollback_on_slo=True,
    )
    decision = {}
    for _ in range(2):
        decision = guardrails.evaluate(
            drift_metrics={
                "class_prior_drift_alarm": True,
                "embedding_drift_alarm": False,
                "alert_volume_alarm": False,
            },
            slo_metrics={"latency_alarm": False, "availability_alarm": False, "samples": 16},
        )
    if not decision.get("triggered", False):
        fail("Guardrail policy did not trigger on sustained alarms.")
    rollback_cfg = canary.rollback("rollout_a")
    if not isinstance(rollback_cfg, dict) or rollback_cfg.get("active", True):
        fail("Canary rollback did not deactivate rollout as expected.")
    rollback_record = guardrails.record_rollback(
        rollout_id="rollout_a",
        reason="drift:class_prior_drift_alarm",
        timestamp_utc="2026-03-04T00:00:00Z",
    )
    if rollback_record.get("rollout_id") != "rollout_a":
        fail("Guardrail rollback history did not capture rollout id.")

    # 3) Drift + SLO monitors.
    drift = mlops.DriftMonitor(class_window=64, embedding_window=64, alert_window=64)
    drift_report = {}
    for idx in range(80):
        drift_report = drift.update(
            class_event_ids=["security:intrusion" if idx % 3 else "security:fire_smoke"],
            embedding_refs=[f"emb_{idx % 5}"],
            alert_count=2 if idx % 10 else 6,
        )
    if "class_prior_drift" not in drift_report:
        fail("Drift monitor did not emit class_prior_drift.")

    slo = mlops.SLOMonitor(latency_window=64)
    slo_report = {}
    for idx in range(70):
        slo_report = slo.record(
            processing_seconds=0.08 + (idx % 5) * 0.01,
            success=True,
            profile_id="edge_cpu_profile",
        )
    if slo_report.get("latency_p95_s") is None:
        fail("SLO monitor did not emit p95 latency.")

    # 4) RBAC + immutable audit.
    rbac = security.RBACPolicyEngine()
    if not rbac.authorize("svc_soc_runtime", "emit:threat_candidate", site_id="site_west_01"):
        fail("RBAC denied expected runtime action.")
    if rbac.authorize("unknown_account", "emit:threat_candidate", site_id="site_west_01"):
        fail("RBAC unexpectedly authorized unknown account.")

    audit = security.ImmutableAuditLog()
    audit.append(actor="svc_soc_runtime", action="process.start", details={"seq": 1}, timestamp_utc="2026-03-04T00:00:00Z")
    audit.append(actor="svc_soc_runtime", action="process.end", details={"seq": 1}, timestamp_utc="2026-03-04T00:00:01Z")
    integrity = audit.verify()
    if not integrity.get("valid", False):
        fail("Immutable audit chain integrity failed.")

    report = {
        "status": "ok",
        "checks": {
            "model_registry": registry.snapshot(),
            "canary_counts": {
                "canary": canary_count,
                "baseline": 100 - canary_count,
            },
            "guardrail_decision": decision,
            "guardrail_snapshot": guardrails.snapshot(),
            "drift_report": drift_report,
            "slo_report": slo_report,
            "rbac_snapshot": rbac.snapshot(),
            "audit_integrity": integrity,
        },
    }

    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[OK] SOC MLOps smoke test passed.")


if __name__ == "__main__":
    main()
