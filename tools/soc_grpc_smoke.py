#!/usr/bin/env python3
"""Smoke test for SOC gRPC runtime server."""

import argparse
import importlib
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc


def parse_args():
    parser = argparse.ArgumentParser(description="SOC gRPC smoke test")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--soc-config",
        default="data/soc/example_soc_runtime_config.json",
        help="SOC runtime config path.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=12.0,
        help="Timeout waiting for server readiness.",
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


def _find_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _ensure_proto_module(repo_root, out_dir):
    proto_path = repo_root / "docs" / "proto" / "hermes_soc_services.proto"
    if not proto_path.exists():
        raise RuntimeError(f"Proto not found: {proto_path}")
    protoc = shutil.which("protoc")
    if not protoc:
        raise RuntimeError("protoc not found in PATH")

    out_dir.mkdir(parents=True, exist_ok=True)
    pb2_path = out_dir / "hermes_soc_services_pb2.py"
    needs_regen = (not pb2_path.exists()) or (
        pb2_path.stat().st_mtime < proto_path.stat().st_mtime
    )
    if needs_regen:
        proc = subprocess.run(
            [
                protoc,
                f"--proto_path={proto_path.parent}",
                f"--python_out={out_dir}",
                str(proto_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()
            raise RuntimeError(f"protoc failed: {detail}")

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


def _call(channel, method, req_serializer, resp_deserializer, request, timeout_s=3.0):
    rpc = channel.unary_unary(
        method,
        request_serializer=req_serializer,
        response_deserializer=resp_deserializer,
    )
    return rpc(request, timeout=timeout_s)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    soc_config = _resolve(repo_root, args.soc_config)
    if not soc_config.exists():
        print(f"[FAIL] SOC config not found: {soc_config}")
        return 1

    pb2 = _ensure_proto_module(repo_root, repo_root / "logs" / "soc_proto_gen")
    server_script = repo_root / "tools" / "soc_grpc_server.py"
    if not server_script.exists():
        print(f"[FAIL] server script missing: {server_script}")
        return 1

    port = _find_free_port()
    bind = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(server_script),
            "--repo-root",
            str(repo_root),
            "--soc-config",
            str(soc_config),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = {"status": "error", "bind": bind, "checks": {}}
    try:
        deadline = time.time() + float(max(2.0, args.timeout_seconds))
        ready = False
        with grpc.insecure_channel(bind) as channel:
            while time.time() < deadline:
                try:
                    _call(
                        channel,
                        "/hermes.soc.v1.InferenceProfileService/ResolveProfile",
                        pb2.ResolveProfileRequest.SerializeToString,
                        pb2.ResolveProfileResponse.FromString,
                        pb2.ResolveProfileRequest(
                            site_id="site_west_01",
                            camera_id="cam_a01",
                            role_hint="edge",
                        ),
                    )
                    ready = True
                    break
                except Exception:
                    time.sleep(0.2)

            if not ready:
                raise RuntimeError("server did not become ready before timeout")

            profile = _call(
                channel,
                "/hermes.soc.v1.InferenceProfileService/ResolveProfile",
                pb2.ResolveProfileRequest.SerializeToString,
                pb2.ResolveProfileResponse.FromString,
                pb2.ResolveProfileRequest(
                    site_id="site_west_01",
                    camera_id="cam_a01",
                    role_hint="edge",
                ),
            )
            if not profile.profile_id:
                raise RuntimeError("ResolveProfile returned empty profile_id")
            report["checks"]["resolve_profile"] = {
                "profile_id": profile.profile_id,
                "detector_name": profile.detector_name,
                "tracker_name": profile.tracker_name,
            }

            ingest_req = pb2.IngestObservationRequest(
                track_event=pb2.EntityTrackEvent(
                    event_id="security:intrusion",
                    timestamp_utc="2026-03-04T02:00:00Z",
                    site_id="site_west_01",
                    camera_id="cam_a01",
                    entity_id_local="person_501",
                    track_confidence=0.91,
                    lifecycle_state="entered",
                    context_label="warehouse_entry",
                    context_confidence=0.94,
                    observation_source="detector_tracker",
                )
            )
            _call(
                channel,
                "/hermes.soc.v1.IngestGatewayService/IngestObservation",
                pb2.IngestObservationRequest.SerializeToString,
                pb2.Empty.FromString,
                ingest_req,
            )
            report["checks"]["ingest_observation"] = {"ok": True}

            fusion_req = pb2.EntityFusionRequest(
                track_event=pb2.EntityTrackEvent(
                    event_id="security:intrusion",
                    timestamp_utc="2026-03-04T02:00:01Z",
                    site_id="site_west_01",
                    camera_id="cam_a01",
                    entity_id_local="person_501",
                    track_confidence=0.88,
                    lifecycle_state="continued",
                    context_label="warehouse_entry",
                    context_confidence=0.90,
                    observation_source="detector_tracker",
                )
            )
            _call(
                channel,
                "/hermes.soc.v1.EntityFusionService/UpsertEntityTrack",
                pb2.EntityFusionRequest.SerializeToString,
                pb2.Empty.FromString,
                fusion_req,
            )
            report["checks"]["entity_fusion"] = {"ok": True}

            score_req = pb2.ThreatScoreRequest(
                track_event=pb2.EntityTrackEvent(
                    event_id="security:intrusion",
                    timestamp_utc="2026-03-04T02:00:02Z",
                    site_id="site_west_01",
                    camera_id="cam_a01",
                    entity_id_local="person_501",
                    track_confidence=0.93,
                    lifecycle_state="continued",
                    context_label="warehouse_entry",
                    context_confidence=0.95,
                    observation_source="detector_tracker",
                )
            )
            score_resp = _call(
                channel,
                "/hermes.soc.v1.ThreatScoringService/ScoreThreat",
                pb2.ThreatScoreRequest.SerializeToString,
                pb2.ThreatScoreResponse.FromString,
                score_req,
            )
            if not score_resp.threat_event.threat_type:
                raise RuntimeError("ScoreThreat returned empty threat_type")
            report["checks"]["score_threat"] = {
                "threat_type": score_resp.threat_event.threat_type,
                "severity": score_resp.threat_event.severity,
            }

            dispatch_req = pb2.AlertDispatchRequest(threat_event=score_resp.threat_event)
            _call(
                channel,
                "/hermes.soc.v1.AlertDispatchService/DispatchAlert",
                pb2.AlertDispatchRequest.SerializeToString,
                pb2.Empty.FromString,
                dispatch_req,
            )
            report["checks"]["dispatch_alert"] = {"ok": True}

            feedback_req = pb2.FeedbackIngestRequest(
                case_id="case_00000001",
                analyst_id="analyst_smoke",
                label="true_positive",
                notes="grpc smoke validation",
            )
            _call(
                channel,
                "/hermes.soc.v1.FeedbackIngestService/IngestFeedback",
                pb2.FeedbackIngestRequest.SerializeToString,
                pb2.Empty.FromString,
                feedback_req,
            )
            report["checks"]["feedback_ingest"] = {"ok": True}

        report["status"] = "ok"
        if args.print_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("[OK] SOC gRPC smoke test passed.")
            print("[OK] bind:", bind)
            print("[OK] profile:", report["checks"]["resolve_profile"]["profile_id"])
        return 0
    except Exception as exc:
        out = ""
        err = ""
        try:
            out, err = proc.communicate(timeout=0.5)
        except Exception:
            pass
        report["error"] = str(exc)
        if out.strip():
            report["server_stdout"] = out.strip()
        if err.strip():
            report["server_stderr"] = err.strip()
        if args.print_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("[FAIL] SOC gRPC smoke test failed:", exc)
            if out.strip():
                print("[server stdout]")
                print(out.strip())
            if err.strip():
                print("[server stderr]")
                print(err.strip())
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
