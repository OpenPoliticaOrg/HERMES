#!/usr/bin/env python3
"""Smoke test for SOC runtime service handlers (gRPC-aligned local services)."""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SOC service handlers smoke test")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--soc-config",
        default="data/soc/example_soc_runtime_config.json",
        help="SOC config path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON report.",
    )
    return parser.parse_args()


def _resolve(root, path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


def _load_soc_modules(repo_root):
    soc_dir = repo_root / "lavis" / "common" / "soc"
    pkg_name = "soc_services_local"

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(soc_dir)]
    sys.modules[pkg_name] = pkg

    modules = {}
    for name in ["runtime", "runtime_services"]:
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
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        fail(f"repo root missing: {repo_root}")

    modules = _load_soc_modules(repo_root)
    runtime_mod = modules["runtime"]
    services_mod = modules["runtime_services"]

    soc_config_path = _resolve(repo_root, args.soc_config)
    if not soc_config_path.exists():
        fail(f"soc config missing: {soc_config_path}")

    orchestrator = runtime_mod.SOCOrchestrator.from_json_config(str(soc_config_path))
    suite = services_mod.SOCRuntimeServiceSuite(orchestrator)

    profile_resp = suite.inference_profile.resolve_profile(
        {"site_id": "site_west_01", "camera_id": "cam_a01", "role_hint": "edge"}
    )
    if not profile_resp.get("profile_id"):
        fail("resolve_profile did not return profile_id.")

    ingest_resp = suite.ingest_gateway.ingest_observation(
        {
            "track_event": {
                "event_id": "security:intrusion",
                "timestamp_utc": "2026-03-04T01:00:00Z",
                "site_id": "site_west_01",
                "camera_id": "cam_a01",
                "entity_id_local": "person_101",
                "track_confidence": 0.92,
                "lifecycle_state": "entered",
                "context_label": "warehouse_entry",
                "context_confidence": 0.96,
                "observation_source": "detector_tracker",
                "bbox": {"x1": 0.11, "y1": 0.21, "x2": 0.33, "y2": 0.74},
                "reid_embedding_ref": "emb_person_101",
            }
        }
    )
    if ingest_resp.get("status") != "ok":
        fail("ingest_observation did not return ok status.")
    soc_payload = ingest_resp.get("soc_payload", {})
    if len(soc_payload.get("entity_track_events", [])) < 1:
        fail("ingest_observation did not produce entity track events.")
    if len(soc_payload.get("threat_events", [])) < 1:
        fail("ingest_observation did not produce threat events.")

    fusion_resp = suite.entity_fusion.upsert_entity_track(
        {
            "track_event": {
                "event_id": "security:intrusion",
                "timestamp_utc": "2026-03-04T01:00:01Z",
                "entity_id_local": "person_101",
                "reid_embedding_ref": "emb_person_101",
                "track_confidence": 0.9,
            }
        }
    )
    if not fusion_resp.get("entity_id_global"):
        fail("entity fusion did not return entity_id_global.")

    score_resp = suite.threat_scoring.score_threat(
        {
            "track_event": {
                "event_id": "security:intrusion",
                "timestamp_utc": "2026-03-04T01:00:02Z",
                "entity_id_local": "person_101",
                "entity_id_global": fusion_resp.get("entity_id_global"),
                "track_confidence": 0.95,
                "context_label": "warehouse_entry",
                "observation_source": "detector_tracker",
            }
        }
    )
    threat_event = score_resp.get("threat_event")
    if score_resp.get("status") != "ok" or not isinstance(threat_event, dict):
        fail("threat scoring did not emit threat_event.")

    dispatch_resp = suite.alert_dispatch.dispatch_alert(
        {"threat_event": threat_event, "confirmed": False, "max_dispatch": 4}
    )
    if dispatch_resp.get("dispatched_count", 0) < 1:
        fail("alert dispatch did not dispatch any messages.")

    case_updates = soc_payload.get("case_updates", [])
    if len(case_updates) < 1:
        fail("No case was opened from ingest flow.")
    case_id = case_updates[0].get("case_id")
    feedback_resp = suite.feedback_ingest.ingest_feedback(
        {
            "case_id": case_id,
            "analyst_id": "analyst_42",
            "label": "true_positive",
            "notes": "Validated in service smoke test.",
        }
    )
    if feedback_resp.get("status") != "ok":
        fail("feedback ingest failed.")

    ack_resp = suite.case_management.acknowledge_case(
        {
            "case_id": case_id,
            "analyst_id": "analyst_42",
            "reason": "ack in services smoke",
        }
    )
    if ack_resp.get("status") != "ok":
        fail("case acknowledge failed.")
    confirm_resp = suite.case_management.confirm_case(
        {
            "case_id": case_id,
            "analyst_id": "analyst_42",
            "reason": "confirm in services smoke",
        }
    )
    if confirm_resp.get("status") != "ok":
        fail("case confirm failed.")

    snapshot_resp = suite.runtime_status.get_runtime_snapshot({"max_items": 5})
    if int(snapshot_resp.get("case_count", 0)) < 1:
        fail("runtime status snapshot did not report cases.")
    if int(snapshot_resp.get("threat_event_count", 0)) < 1:
        fail("runtime status snapshot did not report threat events.")

    report = {
        "status": "ok",
        "profile": profile_resp,
        "ingest_counts": {
            "entity_track_events": len(soc_payload.get("entity_track_events", [])),
            "threat_events": len(soc_payload.get("threat_events", [])),
            "case_updates": len(case_updates),
        },
        "fusion": fusion_resp,
        "threat_score": score_resp,
        "dispatch": dispatch_resp,
        "feedback": feedback_resp,
        "case_ack": ack_resp,
        "case_confirm": confirm_resp,
        "runtime_snapshot": snapshot_resp,
    }
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[OK] SOC runtime services smoke test passed.")
        print("[OK] profile:", profile_resp.get("profile_id"))
        print("[OK] case_id:", case_id)


if __name__ == "__main__":
    main()
