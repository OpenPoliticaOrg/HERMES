#!/usr/bin/env python3
"""Minimal web dashboard client for SOC gRPC runtime status."""

import argparse
import importlib
import json
import random
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc


def parse_args():
    parser = argparse.ArgumentParser(description="SOC dashboard client")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--grpc-target",
        default="127.0.0.1:50051",
        help="SOC gRPC server target host:port.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dashboard HTTP bind host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Dashboard HTTP bind port.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=1000,
        help="Browser refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="Maximum recent threats/cases shown in dashboard.",
    )
    parser.add_argument(
        "--demo-stream",
        action="store_true",
        help="Generate demo ingest events continuously.",
    )
    parser.add_argument(
        "--demo-interval-seconds",
        type=float,
        default=1.0,
        help="Demo ingest interval seconds when --demo-stream is enabled.",
    )
    parser.add_argument(
        "--spawn-grpc-server",
        action="store_true",
        help="Spawn local SOC gRPC server subprocess automatically.",
    )
    parser.add_argument(
        "--soc-config",
        default="data/soc/example_soc_runtime_config.json",
        help="SOC config for spawned gRPC server.",
    )
    parser.add_argument(
        "--proto-out-dir",
        default="logs/soc_proto_gen",
        help="Directory where Python proto module is generated.",
    )
    parser.add_argument(
        "--analyst-id",
        default="analyst_dashboard",
        help="Default analyst id used for case actions.",
    )
    return parser.parse_args()


def _resolve(root, path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


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


class DashboardState:
    def __init__(self, grpc_target, pb2, max_items=20, analyst_id="analyst_dashboard"):
        self.grpc_target = str(grpc_target)
        self.pb2 = pb2
        self.max_items = max(1, int(max_items))
        self.analyst_id = str(analyst_id or "analyst_dashboard")
        self.lock = threading.Lock()
        self.entity_seq = 0

    def _call(self, method, req_ser, resp_deser, request, timeout_s=3.0):
        with grpc.insecure_channel(self.grpc_target) as channel:
            rpc = channel.unary_unary(
                method,
                request_serializer=req_ser,
                response_deserializer=resp_deser,
            )
            return rpc(request, timeout=timeout_s)

    def get_snapshot(self):
        req = self.pb2.RuntimeSnapshotRequest(max_items=int(self.max_items))
        response = self._call(
            "/hermes.soc.v1.RuntimeStatusService/GetRuntimeSnapshot",
            self.pb2.RuntimeSnapshotRequest.SerializeToString,
            self.pb2.RuntimeSnapshotResponse.FromString,
            req,
        )
        return {
            "timestamp_utc": response.timestamp_utc,
            "site_id": response.site_id,
            "camera_id": response.camera_id,
            "profile_id": response.profile_id,
            "threat_event_count": int(response.threat_event_count),
            "case_count": int(response.case_count),
            "backlog_total": int(response.backlog_total),
            "dead_letter_count": int(response.dead_letter_count),
            "congested": bool(response.congested),
            "recent_threats": [
                {
                    "threat_type": x.threat_type,
                    "severity": x.severity,
                    "confidence_calibrated": float(x.confidence_calibrated),
                    "timestamp_utc": x.timestamp_utc,
                    "site_id": x.site_id,
                    "entity_ref": x.entity_ref,
                    "policy_action": x.policy_action,
                }
                for x in response.recent_threats
            ],
            "recent_cases": [
                {
                    "case_id": x.case_id,
                    "state": x.state,
                    "severity": x.severity,
                    "threat_type": x.threat_type,
                    "confidence": float(x.confidence),
                    "created_at_utc": x.created_at_utc,
                    "updated_at_utc": x.updated_at_utc,
                }
                for x in response.recent_cases
            ],
        }

    def ingest_demo_event(self):
        with self.lock:
            self.entity_seq += 1
            entity_id = f"demo_entity_{self.entity_seq:05d}"
        now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        event_candidates = [
            ("security:intrusion", "warehouse_entry"),
            ("security:assault_fight", "parking_lot"),
            ("security:fire_smoke", "warehouse_entry"),
        ]
        event_id, context = random.choice(event_candidates)
        conf = random.uniform(0.72, 0.98)
        req = self.pb2.IngestObservationRequest(
            track_event=self.pb2.EntityTrackEvent(
                event_id=event_id,
                timestamp_utc=now_ts,
                site_id="site_west_01",
                camera_id="cam_a01",
                entity_id_local=entity_id,
                track_confidence=float(conf),
                lifecycle_state="entered",
                context_label=context,
                context_confidence=0.95,
                observation_source="detector_tracker",
            )
        )
        self._call(
            "/hermes.soc.v1.IngestGatewayService/IngestObservation",
            self.pb2.IngestObservationRequest.SerializeToString,
            self.pb2.Empty.FromString,
            req,
        )
        return {"status": "ok", "entity_id": entity_id, "event_id": event_id}

    def case_action(self, action, case_id, analyst_id=None, reason=""):
        action = str(action or "").strip().lower()
        case_id = str(case_id or "").strip()
        actor = str(analyst_id or self.analyst_id).strip() or self.analyst_id
        reason = str(reason or "")
        if not case_id:
            return {"status": "error", "error": "missing_case_id"}
        method_map = {
            "ack": "/hermes.soc.v1.CaseManagementService/AcknowledgeCase",
            "confirm": "/hermes.soc.v1.CaseManagementService/ConfirmCase",
            "dismiss": "/hermes.soc.v1.CaseManagementService/DismissCase",
        }
        if action not in method_map:
            return {"status": "error", "error": "unsupported_action"}
        req = self.pb2.CaseActionRequest(
            case_id=case_id, analyst_id=actor, reason=reason
        )
        resp = self._call(
            method_map[action],
            self.pb2.CaseActionRequest.SerializeToString,
            self.pb2.CaseActionResponse.FromString,
            req,
        )
        out = {
            "status": str(resp.status),
            "error": str(resp.error),
            "case": {
                "case_id": resp.case.case_id,
                "state": resp.case.state,
                "severity": resp.case.severity,
                "threat_type": resp.case.threat_type,
                "confidence": float(resp.case.confidence),
                "created_at_utc": resp.case.created_at_utc,
                "updated_at_utc": resp.case.updated_at_utc,
            },
        }
        return out


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HERMES SOC Dashboard</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #1f2a44;
      --muted: #5d6b89;
      --accent: #0f7b6c;
      --warn: #b6541a;
      --bad: #b42318;
      --grid: #d9e0ef;
    }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: linear-gradient(160deg, #edf3ff 0%, #f9fafc 100%);
      color: var(--ink);
    }
    .wrap { padding: 16px; max-width: 1200px; margin: 0 auto; }
    .title { display: flex; justify-content: space-between; align-items: center; }
    .title h1 { margin: 0; font-size: 24px; letter-spacing: 0.3px; }
    .meta { color: var(--muted); font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 14px 0; }
    .card {
      background: var(--card);
      border: 1px solid var(--grid);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(18, 32, 66, 0.05);
    }
    .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .v { font-size: 22px; font-weight: 700; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; background: var(--card); }
    th, td { border-bottom: 1px solid var(--grid); padding: 7px 8px; text-align: left; }
    th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    .controls { margin: 8px 0 12px; display: flex; gap: 8px; align-items: center; }
    input {
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
    }
    button {
      background: var(--accent); color: #fff; border: none; border-radius: 8px;
      padding: 9px 12px; font-weight: 600; cursor: pointer;
    }
    button:hover { opacity: 0.92; }
    .badge { border-radius: 8px; padding: 2px 6px; font-size: 12px; }
    .sev-critical,.sev-high { background: #fbe4e2; color: var(--bad); }
    .sev-medium { background: #fff0df; color: var(--warn); }
    .sev-low,.sev-info { background: #e4f4ef; color: #0d6a5d; }
    .err { color: var(--bad); font-size: 13px; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">
      <h1>HERMES SOC Runtime Dashboard</h1>
      <div class="meta" id="meta">connecting...</div>
    </div>
    <div class="controls">
      <button onclick="injectDemo()">Inject Demo Event</button>
      <input id="analyst_id" value="__ANALYST_ID__" />
      <span class="meta">Refresh: __REFRESH_MS__ ms</span>
    </div>
    <div class="grid">
      <div class="card"><div class="k">Profile</div><div class="v" id="profile">-</div></div>
      <div class="card"><div class="k">Threat Events</div><div class="v" id="threat_count">0</div></div>
      <div class="card"><div class="k">Cases</div><div class="v" id="case_count">0</div></div>
      <div class="card"><div class="k">Backlog</div><div class="v" id="backlog">0</div></div>
      <div class="card"><div class="k">Dead Letter</div><div class="v" id="dead_letter">0</div></div>
    </div>
    <div class="row">
      <div class="card">
        <h3 style="margin-top:0">Recent Threats</h3>
        <table>
          <thead><tr><th>Time</th><th>Type</th><th>Severity</th><th>Confidence</th><th>Entity</th></tr></thead>
          <tbody id="threat_rows"></tbody>
        </table>
      </div>
      <div class="card">
        <h3 style="margin-top:0">Recent Cases</h3>
        <table>
          <thead><tr><th>Case</th><th>State</th><th>Threat</th><th>Severity</th><th>Confidence</th><th>Actions</th></tr></thead>
          <tbody id="case_rows"></tbody>
        </table>
      </div>
    </div>
    <div id="error" class="err"></div>
  </div>
  <script>
    const refreshMs = __REFRESH_MS__;
    async function fetchSnapshot() {
      try {
        const r = await fetch('/api/snapshot');
        if (!r.ok) throw new Error(await r.text());
        const d = await r.json();
        document.getElementById('error').textContent = '';
        document.getElementById('meta').textContent =
          `${d.site_id}/${d.camera_id} | updated ${d.timestamp_utc} | congested=${d.congested}`;
        document.getElementById('profile').textContent = d.profile_id || '-';
        document.getElementById('threat_count').textContent = d.threat_event_count;
        document.getElementById('case_count').textContent = d.case_count;
        document.getElementById('backlog').textContent = d.backlog_total;
        document.getElementById('dead_letter').textContent = d.dead_letter_count;

        const tBody = document.getElementById('threat_rows');
        tBody.innerHTML = '';
        (d.recent_threats || []).forEach(t => {
          const tr = document.createElement('tr');
          const sev = (t.severity || '').toLowerCase();
          tr.innerHTML = `
            <td>${t.timestamp_utc || ''}</td>
            <td>${t.threat_type || ''}</td>
            <td><span class="badge sev-${sev}">${t.severity || ''}</span></td>
            <td>${(t.confidence_calibrated || 0).toFixed(3)}</td>
            <td>${t.entity_ref || ''}</td>`;
          tBody.appendChild(tr);
        });

        const cBody = document.getElementById('case_rows');
        cBody.innerHTML = '';
        (d.recent_cases || []).forEach(c => {
          const sev = (c.severity || '').toLowerCase();
          const caseId = c.case_id || '';
          const tr = document.createElement('tr');
          const actions = caseId
            ? `<button onclick="caseAction('${caseId}','ack')">Ack</button>
               <button onclick="caseAction('${caseId}','confirm')">Confirm</button>
               <button onclick="caseAction('${caseId}','dismiss')">Dismiss</button>`
            : '';
          tr.innerHTML = `
            <td>${c.case_id || ''}</td>
            <td>${c.state || ''}</td>
            <td>${c.threat_type || ''}</td>
            <td><span class="badge sev-${sev}">${c.severity || ''}</span></td>
            <td>${(c.confidence || 0).toFixed(3)}</td>
            <td>${actions}</td>`;
          cBody.appendChild(tr);
        });
      } catch (e) {
        document.getElementById('error').textContent = `snapshot error: ${e}`;
      }
    }

    async function injectDemo() {
      try {
        const r = await fetch('/api/ingest-demo', { method: 'POST' });
        if (!r.ok) throw new Error(await r.text());
      } catch (e) {
        document.getElementById('error').textContent = `inject error: ${e}`;
      }
    }

    async function caseAction(caseId, action) {
      try {
        const analystId = (document.getElementById('analyst_id').value || '').trim() || 'analyst_dashboard';
        let reason = '';
        if (action === 'confirm' || action === 'dismiss') {
          reason = window.prompt(`Reason for ${action} ${caseId}`, '') || '';
        }
        const r = await fetch('/api/case-action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ case_id: caseId, action, analyst_id: analystId, reason }),
        });
        if (!r.ok) throw new Error(await r.text());
        await fetchSnapshot();
      } catch (e) {
        document.getElementById('error').textContent = `case action error: ${e}`;
      }
    }

    fetchSnapshot();
    setInterval(fetchSnapshot, refreshMs);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    state = None
    refresh_ms = 1000
    analyst_id = "analyst_dashboard"

    def _json(self, status, payload):
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_TEMPLATE.replace("__REFRESH_MS__", str(self.refresh_ms))
            html = html.replace("__ANALYST_ID__", str(self.analyst_id))
            self._html(html)
            return
        if self.path == "/api/snapshot":
            try:
                payload = self.state.get_snapshot()
                self._json(HTTPStatus.OK, payload)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"status": "error", "error": str(exc)},
                )
            return
        if self.path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})

    def do_POST(self):
        if self.path == "/api/ingest-demo":
            try:
                payload = self.state.ingest_demo_event()
                self._json(HTTPStatus.OK, payload)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"status": "error", "error": str(exc)},
                )
            return
        if self.path == "/api/case-action":
            body = self._read_json()
            try:
                payload = self.state.case_action(
                    action=body.get("action"),
                    case_id=body.get("case_id"),
                    analyst_id=body.get("analyst_id") or self.analyst_id,
                    reason=body.get("reason", ""),
                )
                if str(payload.get("status", "")).lower() != "ok":
                    self._json(HTTPStatus.BAD_REQUEST, payload)
                else:
                    self._json(HTTPStatus.OK, payload)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"status": "error", "error": str(exc)},
                )
            return
        self._json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})

    def log_message(self, fmt, *args):
        return


class DemoPump(threading.Thread):
    def __init__(self, state, interval_seconds=1.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.state.ingest_demo_event()
            except Exception:
                pass
            self._stop_evt.wait(self.interval_seconds)


def _spawn_grpc_server(repo_root, grpc_target, soc_config):
    host, port = grpc_target.split(":")
    server_script = repo_root / "tools" / "soc_grpc_server.py"
    return subprocess.Popen(
        [
            sys.executable,
            str(server_script),
            "--repo-root",
            str(repo_root),
            "--soc-config",
            str(soc_config),
            "--host",
            str(host),
            "--port",
            str(port),
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_grpc_ready(state, timeout_seconds=12.0):
    deadline = time.time() + float(max(2.0, timeout_seconds))
    last_err = None
    while time.time() < deadline:
        try:
            state.get_snapshot()
            return True
        except Exception as exc:
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"gRPC target not ready: {last_err}")


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    pb2 = _ensure_proto_module(repo_root, _resolve(repo_root, args.proto_out_dir))

    state = DashboardState(
        grpc_target=args.grpc_target,
        pb2=pb2,
        max_items=args.max_items,
        analyst_id=args.analyst_id,
    )
    spawned_proc = None
    if args.spawn_grpc_server:
        soc_config = _resolve(repo_root, args.soc_config)
        spawned_proc = _spawn_grpc_server(repo_root, args.grpc_target, soc_config)
        _wait_grpc_ready(state, timeout_seconds=12.0)

    DashboardHandler.state = state
    DashboardHandler.refresh_ms = max(200, int(args.refresh_ms))
    DashboardHandler.analyst_id = str(args.analyst_id or "analyst_dashboard")
    server = ThreadingHTTPServer((args.host, int(args.port)), DashboardHandler)

    demo_pump = None
    if args.demo_stream:
        demo_pump = DemoPump(state, interval_seconds=args.demo_interval_seconds)
        demo_pump.start()

    url = f"http://{args.host}:{int(args.port)}"
    print(f"[OK] SOC dashboard listening: {url}")
    print(f"[OK] gRPC target: {args.grpc_target}")
    if args.spawn_grpc_server:
        print("[OK] spawned SOC gRPC server subprocess")
    if args.demo_stream:
        print(f"[OK] demo stream enabled every {args.demo_interval_seconds:.2f}s")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if demo_pump is not None:
            demo_pump.stop()
        if spawned_proc is not None and spawned_proc.poll() is None:
            spawned_proc.terminate()
            try:
                spawned_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                spawned_proc.kill()
                spawned_proc.wait(timeout=2.0)


if __name__ == "__main__":
    main()
