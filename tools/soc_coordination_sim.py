#!/usr/bin/env python3
"""Simulate coordination and message passing across surveillance feed nodes."""

import argparse
import importlib.util
import json
import math
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SOC multi-feed coordination simulator")
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
        "--cameras",
        default="cam_a01,cam_a02,cam_a03",
        help="Comma-separated camera IDs used as surveillance nodes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Simulation steps per run.",
    )
    parser.add_argument(
        "--output-json",
        default="logs/soc_coordination/summary.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print report JSON to stdout.",
    )
    return parser.parse_args()


def _resolve(root, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def _load_soc_modules(repo_root):
    soc_dir = repo_root / "lavis" / "common" / "soc"
    pkg_name = "soc_coord_local"

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(soc_dir)]
    import sys

    sys.modules[pkg_name] = pkg

    modules = {}
    for name in [
        "schemas",
        "calibration",
        "federation",
        "ingestion_health",
        "interop",
        "message_bus",
        "mlops",
        "profiles",
        "routing",
        "security",
        "stores",
        "workflow",
        "threat_intel",
        "runtime",
    ]:
        path = soc_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
        sys.modules[f"{pkg_name}.{name}"] = module
    return modules


def _camera_list(raw):
    out = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out if out else ["cam_a01", "cam_a02", "cam_a03"]


def _iso_at(base, step_idx):
    ts = base + timedelta(seconds=int(step_idx))
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _entity_payload(entity_id, lifecycle_state, event_id, event_label, prob, embedding_ref):
    return {
        "entity_id": entity_id,
        "lifecycle_state": lifecycle_state,
        "sequence_length": 1,
        "markov_state": {"event_id": event_id, "prob": float(prob)},
        "event_predictions": [{"label": event_label, "confidence": float(prob)}],
        "metadata": {
            "bbox_xyxy_norm": [0.2, 0.2, 0.5, 0.8],
            "reid_embedding_ref": embedding_ref,
        },
    }


def _scenario_for_step(step_idx, cameras):
    """Simple handoff scenario for one target entity plus distractors."""
    if len(cameras) < 3:
        cameras = list(cameras) + [f"cam_x{i}" for i in range(3 - len(cameras))]

    cam0, cam1, cam2 = cameras[0], cameras[1], cameras[2]
    target = {
        "camera": None,
        "entity_id": None,
        "lifecycle": "continued",
        "event_id": "security:intrusion",
        "event_label": "intrusion",
        "prob": 0.82,
        "embedding_ref": "person_alpha_global_signature",
    }

    if step_idx < 8:
        target["camera"] = cam0
        target["entity_id"] = "track_1001"
        target["lifecycle"] = "entered" if step_idx == 0 else "continued"
    elif step_idx < 16:
        target["camera"] = cam1
        target["entity_id"] = "track_2207"
        target["lifecycle"] = "reentered" if step_idx == 8 else "continued"
    else:
        target["camera"] = cam2
        target["entity_id"] = "track_3199"
        target["lifecycle"] = "reentered" if step_idx == 16 else "continued"

    per_camera = {cam: [] for cam in cameras}
    per_camera[target["camera"]].append(
        _entity_payload(
            entity_id=target["entity_id"],
            lifecycle_state=target["lifecycle"],
            event_id=target["event_id"],
            event_label=target["event_label"],
            prob=target["prob"],
            embedding_ref=target["embedding_ref"],
        )
    )

    # Add distractor entities.
    for idx, cam in enumerate(cameras):
        if cam == target["camera"]:
            continue
        per_camera[cam].append(
            _entity_payload(
                entity_id=f"distractor_{idx}_{step_idx}",
                lifecycle_state="continued",
                event_id="coin:put_on_hair_extensions",
                event_label="put on hair extensions",
                prob=0.61,
                embedding_ref=f"distractor_signature_{idx}_{step_idx}",
            )
        )

    return target, per_camera


def _build_result(timestamp_utc, camera_id, frame_index, entities):
    active_entities = [e.get("entity_id") for e in entities]
    return {
        "timestamp_utc": timestamp_utc,
        "sequence_id": camera_id,
        "frame_index": int(frame_index),
        "window_index": int(frame_index // 10),
        "ecological_context": "perimeter",
        "entity_observation_source": "detector_tracker",
        "event_predictions": [{"label": "intrusion", "confidence": 0.81}],
        "markov_state": {"event_id": "security:intrusion", "prob": 0.81},
        "markov_posterior": [{"event_id": "security:intrusion", "prob": 0.81}],
        "entity_event_sequences": entities,
        "entity_lifecycle": {
            "window_step": int(frame_index // 10),
            "entered_entities": [
                x.get("entity_id") for x in entities if x.get("lifecycle_state") == "entered"
            ],
            "reentered_entities": [
                x.get("entity_id") for x in entities if x.get("lifecycle_state") == "reentered"
            ],
            "exited_entities": [],
            "active_entities": active_entities,
            "active_count": len(active_entities),
            "total_tracked_entities": len(active_entities),
        },
    }


def _p95(values):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    idx = int(math.ceil(0.95 * len(vals))) - 1
    idx = max(0, min(idx, len(vals) - 1))
    return float(vals[idx])


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()

    soc_modules = _load_soc_modules(root)
    federation_mod = soc_modules["federation"]
    bus_mod = soc_modules["message_bus"]
    routing_mod = soc_modules["routing"]
    runtime_mod = soc_modules["runtime"]
    stores_mod = soc_modules["stores"]

    soc_config = _resolve(root, args.soc_config)
    cameras = _camera_list(args.cameras)
    steps = max(2, int(args.steps))

    shared_federation = federation_mod.EntityFederationService(
        max_time_delta_seconds=30.0,
        min_embedding_similarity=0.75,
    )
    shared_bus = bus_mod.InMemoryMessageBus()
    shared_hot = stores_mod.InMemoryHotStateStore()
    shared_event_store = stores_mod.InMemoryEventStore(max_events_per_table=50000)

    orchestrators = {}
    for camera_id in cameras:
        orch = runtime_mod.SOCOrchestrator.from_json_config(str(soc_config))
        orch.camera_id = str(camera_id)
        orch.entity_federation_service = shared_federation
        orch.message_bus = shared_bus
        orch.hot_state_store = shared_hot
        orch.event_store = shared_event_store
        orchestrators[camera_id] = orch

    base_ts = datetime(2026, 3, 4, 10, 0, 0, tzinfo=timezone.utc)
    handoff_refs = []
    processing_lat_ms = []
    total_threats = 0
    total_cases = 0

    for step in range(steps):
        ts = _iso_at(base_ts, step)
        target, per_camera_entities = _scenario_for_step(step, cameras)

        for camera_id in cameras:
            result = _build_result(
                timestamp_utc=ts,
                camera_id=camera_id,
                frame_index=step * 10,
                entities=per_camera_entities[camera_id],
            )

            t0 = time.perf_counter()
            soc_out = orchestrators[camera_id].process_result(result)
            t1 = time.perf_counter()
            processing_lat_ms.append((t1 - t0) * 1000.0)

            total_threats += len(soc_out.get("threat_events", []))
            total_cases += len(soc_out.get("case_updates", []))

            for item in soc_out.get("entity_track_events", []):
                if item.get("entity_id_local") == target.get("entity_id"):
                    handoff_refs.append(
                        {
                            "step": step,
                            "camera_id": camera_id,
                            "global_id": item.get("entity_id_global"),
                        }
                    )

    # Handoff consistency across camera transitions.
    by_stage = {"cam0": None, "cam1": None, "cam2": None}
    for item in handoff_refs:
        step = int(item["step"])
        if step < 8:
            by_stage["cam0"] = item.get("global_id")
        elif step < 16:
            by_stage["cam1"] = item.get("global_id")
        else:
            by_stage["cam2"] = item.get("global_id")

    pair_checks = []
    pair_checks.append(by_stage["cam0"] is not None and by_stage["cam1"] is not None and by_stage["cam0"] == by_stage["cam1"])
    pair_checks.append(by_stage["cam1"] is not None and by_stage["cam2"] is not None and by_stage["cam1"] == by_stage["cam2"])
    handoff_consistency = sum(1 for x in pair_checks if x) / float(len(pair_checks))

    # Congestion test for critical channel preservation.
    congestion_orch = next(iter(orchestrators.values()))
    for idx in range(200):
        congestion_orch.routing_service.publish(
            routing_mod.NATS_SUBJECTS["video_obs_raw"],
            {"idx": idx, "type": "noise"},
        )
    congestion_orch.routing_service.publish(
        routing_mod.NATS_SUBJECTS["threat_alert_confirmed"],
        {"idx": "critical_event", "type": "critical"},
    )
    dispatch = congestion_orch.routing_service.dispatch_step(max_dispatch=8)
    critical_preserved = any(
        item.get("subject") == routing_mod.NATS_SUBJECTS["threat_alert_confirmed"]
        for item in dispatch
        if isinstance(item, dict)
    )

    report = {
        "status": "ok",
        "config": {
            "soc_config": str(soc_config),
            "cameras": cameras,
            "steps": steps,
        },
        "coordination_kpis": {
            "cross_camera_handoff_consistency": handoff_consistency,
            "critical_channel_preserved_under_congestion": bool(critical_preserved),
            "p95_processing_latency_ms": _p95(processing_lat_ms),
            "total_threat_events": int(total_threats),
            "total_case_updates": int(total_cases),
            "global_entity_count": int(shared_federation.snapshot().get("global_entities", 0)),
            "message_subject_counts": shared_bus.snapshot(),
        },
        "routing_metrics": {
            cam: orch.routing_service.metrics() for cam, orch in orchestrators.items()
        },
        "store_stats": {
            "hot_store": shared_hot.stats(),
            "event_store": shared_event_store.stats(),
            "entity_federation": shared_federation.snapshot(),
        },
    }

    output_json = _resolve(root, args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fp:
        json.dump(report, fp, indent=2, sort_keys=True)

    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"[OK] SOC coordination simulation report written to {output_json}")
        print(
            "[OK] handoff consistency:",
            report["coordination_kpis"]["cross_camera_handoff_consistency"],
        )
        print(
            "[OK] critical channel preserved:",
            report["coordination_kpis"]["critical_channel_preserved_under_congestion"],
        )


if __name__ == "__main__":
    main()
