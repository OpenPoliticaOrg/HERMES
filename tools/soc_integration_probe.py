#!/usr/bin/env python3
"""Probe external SOC integrations (NATS, Redis, ClickHouse, transport files)."""

import argparse
import asyncio
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProbeResult:
    name: str
    configured: bool
    ok: bool
    detail: str
    duration_ms: float
    required: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="SOC external integration probe")
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
        "--timeout-seconds",
        type=float,
        default=2.0,
        help="Connection timeout for external probes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output.",
    )
    return parser.parse_args()


def _resolve(root, path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


def _module_exists(module_name):
    return importlib.util.find_spec(module_name) is not None


def _probe_nats(servers, timeout_seconds):
    t0 = time.perf_counter()
    if not _module_exists("nats"):
        return ProbeResult(
            name="message_bus:nats",
            configured=True,
            ok=False,
            detail="nats-py missing (pip install nats-py)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )

    async def _run():
        from nats.aio.client import Client as NATSClient

        nc = NATSClient()
        await nc.connect(servers=servers, connect_timeout=timeout_seconds)
        await nc.flush(timeout=timeout_seconds)
        await nc.close()

    try:
        asyncio.run(_run())
        return ProbeResult(
            name="message_bus:nats",
            configured=True,
            ok=True,
            detail=f"connected to {servers}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )
    except Exception as exc:
        return ProbeResult(
            name="message_bus:nats",
            configured=True,
            ok=False,
            detail=f"connect failed: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )


def _probe_redis(url, timeout_seconds):
    t0 = time.perf_counter()
    if not _module_exists("redis"):
        return ProbeResult(
            name="hot_store:redis",
            configured=True,
            ok=False,
            detail="redis missing (pip install redis)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )
    try:
        import redis

        client = redis.Redis.from_url(
            url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        pong = client.ping()
        return ProbeResult(
            name="hot_store:redis",
            configured=True,
            ok=bool(pong),
            detail=f"ping={'ok' if pong else 'failed'} url={url}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )
    except Exception as exc:
        return ProbeResult(
            name="hot_store:redis",
            configured=True,
            ok=False,
            detail=f"connect failed: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )


def _probe_clickhouse(cfg, timeout_seconds):
    t0 = time.perf_counter()
    if not _module_exists("clickhouse_connect"):
        return ProbeResult(
            name="event_store:clickhouse",
            configured=True,
            ok=False,
            detail="clickhouse-connect missing (pip install clickhouse-connect)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )
    try:
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=cfg.get("host", "127.0.0.1"),
            port=int(cfg.get("port", 8123)),
            username=cfg.get("username", "default"),
            password=cfg.get("password", ""),
            database=cfg.get("database", "default"),
            connect_timeout=max(0.1, float(timeout_seconds)),
        )
        row = client.query("SELECT 1").result_rows[0][0]
        return ProbeResult(
            name="event_store:clickhouse",
            configured=True,
            ok=(int(row) == 1),
            detail=f"query={'ok' if int(row) == 1 else 'bad_result'} host={cfg.get('host', '127.0.0.1')}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )
    except Exception as exc:
        return ProbeResult(
            name="event_store:clickhouse",
            configured=True,
            ok=False,
            detail=f"connect failed: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )


def _probe_transport_files(transport_cfg):
    t0 = time.perf_counter()
    transport_cfg = transport_cfg if isinstance(transport_cfg, dict) else {}
    mtls = bool(transport_cfg.get("mtls_internal", False))
    tls = bool(transport_cfg.get("tls_external", False))
    enabled = mtls or tls
    if not enabled:
        return ProbeResult(
            name="security:transport_files",
            configured=False,
            ok=True,
            detail="transport security disabled in config",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=False,
        )
    missing = []
    for key in ["cert_path", "key_path", "ca_path"]:
        p = transport_cfg.get(key, "")
        if p and not Path(p).exists():
            missing.append(f"{key}={p}")
    return ProbeResult(
        name="security:transport_files",
        configured=True,
        ok=(len(missing) == 0),
        detail="ok" if len(missing) == 0 else f"missing {', '.join(missing)}",
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        required=True,
    )


def _probe_clip_store(cfg, root):
    t0 = time.perf_counter()
    cfg = cfg if isinstance(cfg, dict) else {}
    store_type = str(cfg.get("type", "none")).lower().strip()
    if store_type not in {"filesystem", "fs"}:
        return ProbeResult(
            name="clip_store:filesystem",
            configured=False,
            ok=True,
            detail=f"clip store type={store_type}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=False,
        )
    root_dir = str(cfg.get("root_dir", "logs/soc_clips"))
    p = _resolve(root, root_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".probe_write_test"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        return ProbeResult(
            name="clip_store:filesystem",
            configured=True,
            ok=True,
            detail=f"writable {p}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )
    except Exception as exc:
        return ProbeResult(
            name="clip_store:filesystem",
            configured=True,
            ok=False,
            detail=f"write failed: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            required=True,
        )


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cfg_path = _resolve(repo_root, args.soc_config)
    if not cfg_path.exists():
        print(f"[FAIL] config not found: {cfg_path}")
        return 1

    with open(cfg_path, "r") as fp:
        cfg = json.load(fp)

    integrations = cfg.get("integrations", {}) if isinstance(cfg.get("integrations"), dict) else {}
    message_bus_cfg = integrations.get("message_bus", {})
    hot_store_cfg = integrations.get("hot_store", {})
    event_store_cfg = integrations.get("event_store", {})
    clip_store_cfg = integrations.get("clip_store", {})
    security_cfg = cfg.get("security", {}) if isinstance(cfg.get("security"), dict) else {}
    transport_cfg = security_cfg.get("transport", {})

    results = []

    bus_type = str(message_bus_cfg.get("type", "in_memory")).lower().strip()
    if bus_type == "nats":
        results.append(
            _probe_nats(
                servers=message_bus_cfg.get("servers", ["nats://127.0.0.1:4222"]),
                timeout_seconds=args.timeout_seconds,
            )
        )
    else:
        results.append(
            ProbeResult(
                name="message_bus:nats",
                configured=False,
                ok=True,
                detail=f"message_bus type={bus_type}",
                duration_ms=0.0,
                required=False,
            )
        )

    hot_type = str(hot_store_cfg.get("type", "in_memory")).lower().strip()
    if hot_type == "redis":
        results.append(
            _probe_redis(
                url=hot_store_cfg.get("url", "redis://127.0.0.1:6379/0"),
                timeout_seconds=args.timeout_seconds,
            )
        )
    else:
        results.append(
            ProbeResult(
                name="hot_store:redis",
                configured=False,
                ok=True,
                detail=f"hot_store type={hot_type}",
                duration_ms=0.0,
                required=False,
            )
        )

    event_type = str(event_store_cfg.get("type", "in_memory")).lower().strip()
    if event_type == "clickhouse":
        results.append(_probe_clickhouse(event_store_cfg, timeout_seconds=args.timeout_seconds))
    else:
        results.append(
            ProbeResult(
                name="event_store:clickhouse",
                configured=False,
                ok=True,
                detail=f"event_store type={event_type}",
                duration_ms=0.0,
                required=False,
            )
        )

    results.append(_probe_clip_store(clip_store_cfg, repo_root))
    results.append(_probe_transport_files(transport_cfg))

    required_failures = [r for r in results if r.required and not r.ok]
    summary = {
        "ready": len(required_failures) == 0,
        "required_failures": len(required_failures),
        "configured_checks": sum(1 for r in results if r.configured),
        "total_checks": len(results),
    }

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "results": [asdict(r) for r in results],
                    "config_path": str(cfg_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"SOC Integration Ready: {'YES' if summary['ready'] else 'NO'}")
        print(f"Config: {cfg_path}")
        for item in results:
            status = "OK" if item.ok else "FAIL"
            if not item.configured:
                status = "SKIP"
            suffix = " (required)" if item.required else ""
            print(f"[{status}] {item.name}{suffix} :: {item.detail}")
        if not summary["ready"]:
            print("Fix required integrations before production deployment.")

    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
