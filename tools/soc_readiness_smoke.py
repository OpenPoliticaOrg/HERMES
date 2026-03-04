#!/usr/bin/env python3
"""Smoke test for SOC production-readiness foundation modules."""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SOC readiness smoke test")
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
    pkg_name = "soc_local"

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(soc_dir)]
    sys.modules[pkg_name] = pkg

    modules = {}
    for name in [
        "schemas",
        "calibration",
        "federation",
        "ingestion_health",
        "interop",
        "message_bus",
        "profiles",
        "routing",
        "security",
        "stores",
        "workflow",
        "threat_intel",
        "mlops",
        "runtime",
    ]:
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
        fail(f"repo root not found: {repo_root}")

    soc_modules = _load_soc_modules(repo_root)
    runtime_mod = soc_modules["runtime"]
    routing_mod = soc_modules["routing"]
    interop_mod = soc_modules["interop"]

    soc_config_path = _resolve(repo_root, args.soc_config)
    if not soc_config_path.exists():
        fail(f"soc config missing: {soc_config_path}")

    # 1) ONVIF discovery/profile sync check.
    inventory_path = repo_root / "data" / "soc" / "example_onvif_inventory.json"
    onvif = interop_mod.ONVIFDiscoveryService.from_json_inventory(str(inventory_path))
    sync = onvif.sync_profiles()
    if int(sync.get("count", 0)) < 1:
        fail("ONVIF profile sync returned no cameras.")

    # 2) SOC orchestrator check with synthetic stream output.
    orchestrator = runtime_mod.SOCOrchestrator.from_json_config(str(soc_config_path))

    stream_result = {
        "sequence_id": "cam_a01",
        "window_index": 4,
        "frame_index": 120,
        "timestamp_utc": "2026-03-03T20:00:00Z",
        "ecological_context": "warehouse_entry",
        "entity_observation_source": "detector_tracker",
        "markov_state": {"event_id": "security:intrusion", "prob": 0.78},
        "markov_posterior": [
            {"event_id": "security:intrusion", "prob": 0.78},
            {"event_id": "security:fire_smoke", "prob": 0.22},
        ],
        "entity_event_sequences": [
            {
                "entity_id": "person_001",
                "lifecycle_state": "entered",
                "sequence_length": 6,
                "markov_state": {"event_id": "security:intrusion", "prob": 0.78},
                "event_predictions": [{"label": "intrusion"}],
                "metadata": {
                    "bbox_xyxy_norm": [0.12, 0.22, 0.34, 0.75],
                    "reid_embedding_ref": "emb_person_001",
                },
            },
            {
                "entity_id": "person_002",
                "lifecycle_state": "reentered",
                "sequence_length": 11,
                "markov_state": {"event_id": "security:assault_fight", "prob": 0.74},
                "event_predictions": [{"label": "fight"}],
                "metadata": {
                    "bbox_xyxy_norm": [0.40, 0.18, 0.65, 0.81],
                    "reid_embedding_ref": "emb_person_002",
                },
            },
        ],
        "entity_lifecycle": {
            "window_step": 4,
            "entered_entities": ["person_001"],
            "reentered_entities": ["person_002"],
            "active_entities": ["person_001", "person_002"],
            "exited_entities": [],
            "active_count": 2,
            "total_tracked_entities": 2,
        },
    }

    soc_out = orchestrator.process_result(stream_result)
    track_events = soc_out.get("entity_track_events", [])
    threat_events = soc_out.get("threat_events", [])
    case_updates = soc_out.get("case_updates", [])

    if len(track_events) != 2:
        fail("SOC orchestrator did not emit expected entity track events.")
    if len(threat_events) < 1:
        fail("SOC orchestrator did not emit threat events.")
    if len(case_updates) < 1:
        fail("SOC workflow did not open candidate cases.")
    if len(soc_out.get("message_bus_publish_results", [])) < 1:
        fail("SOC orchestrator did not publish dispatched messages to message bus.")
    if int((soc_out.get("hot_store_stats") or {}).get("keys", 0)) < 1:
        fail("SOC hot-state store was not updated.")
    if int(
        (soc_out.get("entity_federation_snapshot") or {}).get("global_entities", 0)
    ) < 1:
        fail("Entity federation did not produce any global entities.")
    if str((soc_out.get("confidence_calibration") or {}).get("method", "")) == "":
        fail("Confidence calibrator metadata is missing.")
    if (soc_out.get("security") or {}).get("service_account_id") != "svc_soc_runtime":
        fail("Security service account was not propagated correctly.")
    if not bool(
        ((soc_out.get("security") or {}).get("audit_integrity") or {}).get("valid", False)
    ):
        fail("Immutable audit log integrity check failed.")
    if not isinstance(soc_out.get("drift_metrics", {}), dict):
        fail("Drift metrics payload is missing.")
    if not isinstance(soc_out.get("slo_metrics", {}), dict):
        fail("SLO metrics payload is missing.")
    if not isinstance(soc_out.get("rollout_guardrails", {}), dict):
        fail("Rollout guardrail payload is missing.")
    if float(soc_out.get("processing_seconds", 0.0)) <= 0.0:
        fail("Processing latency metric was not recorded.")
    if int(((soc_out.get("mlops") or {}).get("model_registry") or {}).get("model_count", 0)) < 1:
        fail("Model registry bootstrap did not register artifacts.")
    if not isinstance(((soc_out.get("mlops") or {}).get("guardrails") or {}), dict):
        fail("Guardrail policy snapshot missing from mlops payload.")

    clip_refs = [x.get("clip_ref") for x in threat_events if x.get("clip_ref")]
    if len(clip_refs) == 0:
        fail("Threat clip references were not persisted.")
    for clip_ref in clip_refs[:3]:
        if not Path(clip_ref).exists():
            fail(f"Clip reference path not found: {clip_ref}")

    # 3) Workflow transition + feedback checks.
    first_case_id = case_updates[0]["case_id"]
    ack = orchestrator.workflow_service.acknowledge(first_case_id, actor="analyst_1")
    if not ack or ack.get("state") != "analyst_review":
        fail("Workflow acknowledge transition failed.")
    confirmed = orchestrator.workflow_service.transition(
        first_case_id,
        "confirmed",
        actor="analyst_1",
        reason="verified by cross-camera corroboration",
    )
    if not confirmed or confirmed.get("state") != "confirmed":
        fail("Workflow confirm transition failed.")
    feedback = orchestrator.workflow_service.ingest_feedback(
        first_case_id,
        actor="analyst_1",
        label="true_positive",
        notes="Threat was valid and handled by on-site security.",
    )
    if feedback is None:
        fail("Feedback ingest failed.")

    # 4) Backpressure + priority retention check.
    queue_tester = routing_mod.RuntimeRoutingPolicyService(
        max_backlog=6,
        congestion_soft_limit=4,
        max_retries=1,
        priority_drop_threshold=60,
    )
    for idx in range(20):
        queue_tester.publish(
            routing_mod.NATS_SUBJECTS["video_obs_raw"],
            {"idx": idx, "kind": "low_priority_obs"},
        )
    queue_tester.publish(
        routing_mod.NATS_SUBJECTS["threat_alert_confirmed"],
        {"idx": "critical_1", "kind": "critical_alert"},
    )
    dispatched = queue_tester.dispatch_step(max_dispatch=3)
    if not any(
        item.get("subject") == routing_mod.NATS_SUBJECTS["threat_alert_confirmed"]
        for item in dispatched
    ):
        fail("Priority routing failed to dispatch critical alert under congestion.")

    # 5) Service/profile sanity checks.
    profile = soc_out.get("profile", {})
    if not profile.get("profile_id"):
        fail("Profile routing returned empty profile.")

    report = {
        "status": "ok",
        "paths": {
            "repo_root": str(repo_root),
            "soc_config": str(soc_config_path),
            "onvif_inventory": str(inventory_path),
        },
        "checks": {
            "onvif_sync": sync,
            "selected_profile": profile,
            "entity_track_event_count": len(track_events),
            "threat_event_count": len(threat_events),
            "case_update_count": len(case_updates),
            "routing_metrics": soc_out.get("routing_metrics", {}),
            "message_bus_publish_count": len(
                soc_out.get("message_bus_publish_results", [])
            ),
            "hot_store_stats": soc_out.get("hot_store_stats", {}),
            "event_store_stats": soc_out.get("event_store_stats", {}),
            "entity_federation_snapshot": soc_out.get("entity_federation_snapshot", {}),
            "confidence_calibration": soc_out.get("confidence_calibration", {}),
            "drift_metrics": soc_out.get("drift_metrics", {}),
            "slo_metrics": soc_out.get("slo_metrics", {}),
            "rollout_guardrails": soc_out.get("rollout_guardrails", {}),
            "security": soc_out.get("security", {}),
            "mlops": soc_out.get("mlops", {}),
            "processing_seconds": soc_out.get("processing_seconds"),
            "post_confirm_case_state": confirmed.get("state"),
            "feedback_label": feedback.get("label"),
            "queue_tester_metrics": queue_tester.metrics(),
        },
    }

    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[OK] SOC readiness smoke test passed.")
        print("[OK] profile:", profile.get("profile_id"))
        print("[OK] threat events:", len(threat_events))
        print("[OK] case updates:", len(case_updates))


if __name__ == "__main__":
    main()
