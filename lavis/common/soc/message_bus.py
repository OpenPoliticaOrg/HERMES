"""Message bus adapters for SOC routing/export surfaces."""

import json
from collections import defaultdict
from datetime import datetime, timezone


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _to_payload(payload):
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


class BaseMessageBus:
    def publish(self, subject, payload, headers=None):
        raise NotImplementedError

    def flush(self):
        return True


class InMemoryMessageBus(BaseMessageBus):
    def __init__(self):
        self.messages = defaultdict(list)

    def publish(self, subject, payload, headers=None):
        subject = str(subject)
        item = {
            "subject": subject,
            "payload": _to_payload(payload),
            "headers": headers if isinstance(headers, dict) else {},
            "published_at_utc": utc_now_iso(),
        }
        self.messages[subject].append(item)
        return item

    def get_subject(self, subject):
        return list(self.messages.get(str(subject), []))

    def snapshot(self):
        return {k: len(v) for k, v in self.messages.items()}


class NATSJetStreamMessageBus(BaseMessageBus):
    """Best-effort NATS JetStream publisher.

    This adapter is synchronous but intentionally lightweight. If `nats-py` is
    unavailable or a connection cannot be established, the class raises at init.
    """

    def __init__(self, servers=None, stream_name="HERMES", timeout_seconds=2.0):
        try:
            from nats.aio.client import Client as NATSClient
        except Exception as exc:
            raise RuntimeError(
                "nats-py is required for NATSJetStreamMessageBus. Install with `pip install nats-py`."
            ) from exc

        self._NATSClient = NATSClient
        self.servers = servers or ["nats://127.0.0.1:4222"]
        self.stream_name = str(stream_name)
        self.timeout_seconds = float(max(0.1, timeout_seconds))
        self._connected = False
        self._error_count = 0

    def _sync_publish(self, subject, payload, headers=None):
        import asyncio

        async def _run_once():
            nc = self._NATSClient()
            await nc.connect(servers=self.servers, connect_timeout=self.timeout_seconds)
            try:
                js = nc.jetstream()
                data = json.dumps(_to_payload(payload)).encode("utf-8")
                ack = await js.publish(str(subject), data, headers=headers)
                return {
                    "subject": str(subject),
                    "stream": self.stream_name,
                    "seq": int(getattr(ack, "seq", 0)),
                    "duplicate": bool(getattr(ack, "duplicate", False)),
                    "published_at_utc": utc_now_iso(),
                }
            finally:
                await nc.close()

        return asyncio.run(_run_once())

    def publish(self, subject, payload, headers=None):
        try:
            item = self._sync_publish(subject=subject, payload=payload, headers=headers)
            self._connected = True
            return item
        except Exception as exc:
            self._error_count += 1
            raise RuntimeError(f"NATS publish failed: {exc}") from exc

    def flush(self):
        return self._connected and self._error_count == 0

    def stats(self):
        return {
            "connected": bool(self._connected),
            "error_count": int(self._error_count),
            "servers": list(self.servers),
            "stream_name": self.stream_name,
        }
