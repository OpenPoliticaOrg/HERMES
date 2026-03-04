#!/usr/bin/env python3
"""Run a gRPC server exposing SOC runtime services."""

import argparse
import importlib
import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import types
from concurrent import futures
from pathlib import Path

import grpc


def parse_args():
    parser = argparse.ArgumentParser(description="SOC gRPC runtime server")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--soc-config",
        default="data/soc/example_soc_runtime_config.json",
        help="SOC runtime config path (relative to repo root unless absolute).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=50051, help="Bind port.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="gRPC server thread pool size.",
    )
    parser.add_argument(
        "--proto-out-dir",
        default="logs/soc_proto_gen",
        help="Directory where Python proto module is generated.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def _resolve(root, path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


def _load_soc_modules(repo_root):
    soc_dir = repo_root / "lavis" / "common" / "soc"
    pkg_name = "soc_grpc_local"

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


def _ensure_proto_module(repo_root, out_dir):
    proto_path = repo_root / "docs" / "proto" / "hermes_soc_services.proto"
    if not proto_path.exists():
        raise RuntimeError(f"Proto not found: {proto_path}")

    protoc = shutil.which("protoc")
    if not protoc:
        raise RuntimeError("protoc not found in PATH; install Protocol Buffers compiler.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pb2_path = out_dir / "hermes_soc_services_pb2.py"
    needs_regen = (not pb2_path.exists()) or (
        pb2_path.stat().st_mtime < proto_path.stat().st_mtime
    )
    if needs_regen:
        cmd = [
            protoc,
            f"--proto_path={proto_path.parent}",
            f"--python_out={out_dir}",
            str(proto_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
            raise RuntimeError(f"protoc failed: {detail}")

    # Homebrew protoc can emit runtime-version checks requiring newer protobuf than
    # some environments ship. Strip that guard for this generated local module.
    raw = pb2_path.read_text()
    if "runtime_version" in raw:
        patched = raw.replace(
            "from google.protobuf import runtime_version as _runtime_version\n", ""
        )
        patched = re.sub(
            r"_runtime_version\.ValidateProtobufRuntimeVersion\([\s\S]*?\)\n",
            "",
            patched,
            count=1,
        )
        if patched != raw:
            pb2_path.write_text(patched)

    out_dir_s = str(out_dir.resolve())
    if out_dir_s not in sys.path:
        sys.path.insert(0, out_dir_s)
    return importlib.import_module("hermes_soc_services_pb2")


def _pb_bbox_to_dict(msg):
    if msg is None or len(msg.ListFields()) == 0:
        return None
    return {
        "x1": float(msg.x1),
        "y1": float(msg.y1),
        "x2": float(msg.x2),
        "y2": float(msg.y2),
    }


def _pb_entity_track_to_dict(msg):
    data = {
        "event_id": str(msg.event_id),
        "timestamp_utc": str(msg.timestamp_utc),
        "site_id": str(msg.site_id),
        "camera_id": str(msg.camera_id),
        "entity_id_local": str(msg.entity_id_local),
        "entity_id_global": str(msg.entity_id_global) if msg.entity_id_global else None,
        "track_confidence": float(msg.track_confidence),
        "reid_embedding_ref": str(msg.reid_embedding_ref) if msg.reid_embedding_ref else None,
        "lifecycle_state": str(msg.lifecycle_state) if msg.lifecycle_state else "continued",
        "context_label": str(msg.context_label) if msg.context_label else None,
        "context_confidence": float(msg.context_confidence),
        "observation_source": str(msg.observation_source)
        if msg.observation_source
        else "detector_tracker",
    }
    if msg.HasField("bbox"):
        data["bbox"] = _pb_bbox_to_dict(msg.bbox)
    return data


def _dict_to_pb_threat(payload, pb2):
    payload = payload if isinstance(payload, dict) else {}
    msg = pb2.ThreatEvent()
    msg.threat_type = str(payload.get("threat_type", ""))
    msg.severity = str(payload.get("severity", ""))
    msg.confidence_calibrated = float(payload.get("confidence_calibrated", 0.0))
    msg.entity_refs.extend([str(x) for x in payload.get("entity_refs", [])])
    msg.camera_refs.extend([str(x) for x in payload.get("camera_refs", [])])
    if payload.get("clip_ref") is not None:
        msg.clip_ref = str(payload.get("clip_ref"))
    if payload.get("markov_state") is not None:
        msg.markov_state = str(payload.get("markov_state"))
    msg.anomaly_score = float(payload.get("anomaly_score", 0.0))
    msg.fusion_score = float(payload.get("fusion_score", 0.0))
    msg.policy_action = str(payload.get("policy_action", ""))
    msg.explanations.extend([str(x) for x in payload.get("explanations", [])])
    msg.timestamp_utc = str(payload.get("timestamp_utc", ""))
    msg.site_id = str(payload.get("site_id", ""))
    return msg


def _pb_threat_to_dict(msg):
    return {
        "threat_type": str(msg.threat_type),
        "severity": str(msg.severity),
        "confidence_calibrated": float(msg.confidence_calibrated),
        "entity_refs": [str(x) for x in msg.entity_refs],
        "camera_refs": [str(x) for x in msg.camera_refs],
        "clip_ref": str(msg.clip_ref) if msg.clip_ref else None,
        "markov_state": str(msg.markov_state) if msg.markov_state else None,
        "anomaly_score": float(msg.anomaly_score),
        "fusion_score": float(msg.fusion_score),
        "policy_action": str(msg.policy_action),
        "explanations": [str(x) for x in msg.explanations],
        "timestamp_utc": str(msg.timestamp_utc),
        "site_id": str(msg.site_id),
    }


def _dict_to_pb_runtime_snapshot(payload, pb2):
    payload = payload if isinstance(payload, dict) else {}
    out = pb2.RuntimeSnapshotResponse()
    out.timestamp_utc = str(payload.get("timestamp_utc", ""))
    out.site_id = str(payload.get("site_id", ""))
    out.camera_id = str(payload.get("camera_id", ""))
    out.profile_id = str(payload.get("profile_id", ""))
    out.threat_event_count = int(payload.get("threat_event_count", 0))
    out.case_count = int(payload.get("case_count", 0))
    out.backlog_total = int(payload.get("backlog_total", 0))
    out.dead_letter_count = int(payload.get("dead_letter_count", 0))
    out.congested = bool(payload.get("congested", False))

    for item in payload.get("recent_threats", []):
        item = item if isinstance(item, dict) else {}
        row = out.recent_threats.add()
        row.threat_type = str(item.get("threat_type", ""))
        row.severity = str(item.get("severity", ""))
        row.confidence_calibrated = float(item.get("confidence_calibrated", 0.0))
        row.timestamp_utc = str(item.get("timestamp_utc", ""))
        row.site_id = str(item.get("site_id", ""))
        row.entity_ref = str(item.get("entity_ref", ""))
        row.policy_action = str(item.get("policy_action", ""))

    for item in payload.get("recent_cases", []):
        item = item if isinstance(item, dict) else {}
        row = out.recent_cases.add()
        row.case_id = str(item.get("case_id", ""))
        row.state = str(item.get("state", ""))
        row.severity = str(item.get("severity", ""))
        row.threat_type = str(item.get("threat_type", ""))
        row.confidence = float(item.get("confidence", 0.0))
        row.created_at_utc = str(item.get("created_at_utc", ""))
        row.updated_at_utc = str(item.get("updated_at_utc", ""))
    return out


def _dict_to_pb_case_action_response(payload, pb2):
    payload = payload if isinstance(payload, dict) else {}
    out = pb2.CaseActionResponse()
    out.status = str(payload.get("status", "error"))
    out.error = str(payload.get("error", ""))
    case = payload.get("case", {})
    if isinstance(case, dict):
        out.case.case_id = str(case.get("case_id", ""))
        out.case.state = str(case.get("state", ""))
        out.case.severity = str(case.get("severity", ""))
        out.case.threat_type = str(case.get("threat_type", ""))
        out.case.confidence = float(case.get("confidence", 0.0))
        out.case.created_at_utc = str(case.get("created_at_utc", ""))
        out.case.updated_at_utc = str(case.get("updated_at_utc", ""))
    return out


def build_server(repo_root, soc_config_path, pb2, max_workers):
    modules = _load_soc_modules(repo_root)
    runtime_mod = modules["runtime"]
    services_mod = modules["runtime_services"]

    orchestrator = runtime_mod.SOCOrchestrator.from_json_config(str(soc_config_path))
    suite = services_mod.SOCRuntimeServiceSuite(orchestrator)

    class IngestGatewayServicer:
        def IngestObservation(self, request, context):
            payload = {"track_event": _pb_entity_track_to_dict(request.track_event)}
            try:
                suite.ingest_gateway.ingest_observation(payload)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"ingest failed: {exc}")
            return pb2.Empty()

    class InferenceProfileServicer:
        def ResolveProfile(self, request, context):
            payload = {
                "site_id": request.site_id,
                "camera_id": request.camera_id,
                "role_hint": request.role_hint,
            }
            try:
                profile = suite.inference_profile.resolve_profile(payload)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"profile resolve failed: {exc}")
            out = pb2.ResolveProfileResponse()
            out.profile_id = str(profile.get("profile_id", ""))
            out.detector_name = str(profile.get("detector_name", ""))
            out.tracker_name = str(profile.get("tracker_name", ""))
            out.reid_name = str(profile.get("reid_name", ""))
            return out

    class EntityFusionServicer:
        def UpsertEntityTrack(self, request, context):
            payload = {"track_event": _pb_entity_track_to_dict(request.track_event)}
            try:
                suite.entity_fusion.upsert_entity_track(payload)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"entity fusion failed: {exc}")
            return pb2.Empty()

    class ThreatScoringServicer:
        def ScoreThreat(self, request, context):
            payload = {"track_event": _pb_entity_track_to_dict(request.track_event)}
            try:
                score = suite.threat_scoring.score_threat(payload)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"threat scoring failed: {exc}")
            out = pb2.ThreatScoreResponse()
            if isinstance(score, dict) and isinstance(score.get("threat_event"), dict):
                out.threat_event.CopyFrom(_dict_to_pb_threat(score["threat_event"], pb2))
            return out

    class AlertDispatchServicer:
        def DispatchAlert(self, request, context):
            payload = {
                "threat_event": _pb_threat_to_dict(request.threat_event),
                "confirmed": False,
                "max_dispatch": 4,
            }
            try:
                suite.alert_dispatch.dispatch_alert(payload)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"dispatch failed: {exc}")
            return pb2.Empty()

    class FeedbackIngestServicer:
        def IngestFeedback(self, request, context):
            payload = {
                "case_id": request.case_id,
                "analyst_id": request.analyst_id,
                "label": request.label,
                "notes": request.notes,
            }
            try:
                out = suite.feedback_ingest.ingest_feedback(payload)
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"feedback failed: {exc}")
            if (out or {}).get("status") != "ok":
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"feedback ingest failed: {(out or {}).get('error', 'unknown')}",
                )
            return pb2.Empty()

    class RuntimeStatusServicer:
        def GetRuntimeSnapshot(self, request, context):
            payload = {"max_items": int(request.max_items or 10)}
            try:
                snapshot = suite.runtime_status.get_runtime_snapshot(payload)
            except Exception as exc:
                context.abort(
                    grpc.StatusCode.INTERNAL, f"runtime snapshot failed: {exc}"
                )
            return _dict_to_pb_runtime_snapshot(snapshot, pb2)

    class CaseManagementServicer:
        @staticmethod
        def _payload(request):
            return {
                "case_id": request.case_id,
                "analyst_id": request.analyst_id,
                "reason": request.reason,
            }

        def AcknowledgeCase(self, request, context):
            try:
                out = suite.case_management.acknowledge_case(self._payload(request))
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"ack case failed: {exc}")
            if (out or {}).get("status") != "ok":
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"ack case failed: {(out or {}).get('error', 'unknown')}",
                )
            return _dict_to_pb_case_action_response(out, pb2)

        def ConfirmCase(self, request, context):
            try:
                out = suite.case_management.confirm_case(self._payload(request))
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"confirm case failed: {exc}")
            if (out or {}).get("status") != "ok":
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"confirm case failed: {(out or {}).get('error', 'unknown')}",
                )
            return _dict_to_pb_case_action_response(out, pb2)

        def DismissCase(self, request, context):
            try:
                out = suite.case_management.dismiss_case(self._payload(request))
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, f"dismiss case failed: {exc}")
            if (out or {}).get("status") != "ok":
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"dismiss case failed: {(out or {}).get('error', 'unknown')}",
                )
            return _dict_to_pb_case_action_response(out, pb2)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.IngestGatewayService",
                {
                    "IngestObservation": grpc.unary_unary_rpc_method_handler(
                        IngestGatewayServicer().IngestObservation,
                        request_deserializer=pb2.IngestObservationRequest.FromString,
                        response_serializer=pb2.Empty.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.InferenceProfileService",
                {
                    "ResolveProfile": grpc.unary_unary_rpc_method_handler(
                        InferenceProfileServicer().ResolveProfile,
                        request_deserializer=pb2.ResolveProfileRequest.FromString,
                        response_serializer=pb2.ResolveProfileResponse.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.EntityFusionService",
                {
                    "UpsertEntityTrack": grpc.unary_unary_rpc_method_handler(
                        EntityFusionServicer().UpsertEntityTrack,
                        request_deserializer=pb2.EntityFusionRequest.FromString,
                        response_serializer=pb2.Empty.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.ThreatScoringService",
                {
                    "ScoreThreat": grpc.unary_unary_rpc_method_handler(
                        ThreatScoringServicer().ScoreThreat,
                        request_deserializer=pb2.ThreatScoreRequest.FromString,
                        response_serializer=pb2.ThreatScoreResponse.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.AlertDispatchService",
                {
                    "DispatchAlert": grpc.unary_unary_rpc_method_handler(
                        AlertDispatchServicer().DispatchAlert,
                        request_deserializer=pb2.AlertDispatchRequest.FromString,
                        response_serializer=pb2.Empty.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.FeedbackIngestService",
                {
                    "IngestFeedback": grpc.unary_unary_rpc_method_handler(
                        FeedbackIngestServicer().IngestFeedback,
                        request_deserializer=pb2.FeedbackIngestRequest.FromString,
                        response_serializer=pb2.Empty.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.RuntimeStatusService",
                {
                    "GetRuntimeSnapshot": grpc.unary_unary_rpc_method_handler(
                        RuntimeStatusServicer().GetRuntimeSnapshot,
                        request_deserializer=pb2.RuntimeSnapshotRequest.FromString,
                        response_serializer=pb2.RuntimeSnapshotResponse.SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "hermes.soc.v1.CaseManagementService",
                {
                    "AcknowledgeCase": grpc.unary_unary_rpc_method_handler(
                        CaseManagementServicer().AcknowledgeCase,
                        request_deserializer=pb2.CaseActionRequest.FromString,
                        response_serializer=pb2.CaseActionResponse.SerializeToString,
                    ),
                    "ConfirmCase": grpc.unary_unary_rpc_method_handler(
                        CaseManagementServicer().ConfirmCase,
                        request_deserializer=pb2.CaseActionRequest.FromString,
                        response_serializer=pb2.CaseActionResponse.SerializeToString,
                    ),
                    "DismissCase": grpc.unary_unary_rpc_method_handler(
                        CaseManagementServicer().DismissCase,
                        request_deserializer=pb2.CaseActionRequest.FromString,
                        response_serializer=pb2.CaseActionResponse.SerializeToString,
                    ),
                },
            ),
        )
    )

    return server


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = Path(args.repo_root).resolve()
    soc_config = _resolve(repo_root, args.soc_config)
    if not soc_config.exists():
        raise SystemExit(f"SOC config not found: {soc_config}")

    pb2 = _ensure_proto_module(repo_root, _resolve(repo_root, args.proto_out_dir))
    server = build_server(
        repo_root=repo_root,
        soc_config_path=soc_config,
        pb2=pb2,
        max_workers=max(1, int(args.max_workers)),
    )

    bind_addr = f"{args.host}:{int(args.port)}"
    server.add_insecure_port(bind_addr)
    server.start()
    logging.info(f"SOC gRPC server listening on {bind_addr}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("SOC gRPC server stopping...")
        server.stop(grace=2.0)


if __name__ == "__main__":
    main()
