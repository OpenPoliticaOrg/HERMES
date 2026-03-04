"""Storage adapters for SOC hot state, analytics events, and clips."""

import json
from collections import defaultdict, deque
from pathlib import Path


def _safe_payload(value):
    if isinstance(value, dict):
        return value
    return {"value": value}


class BaseHotStateStore:
    def set(self, key, value, ttl_seconds=None):
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError

    def stats(self):
        return {}


class InMemoryHotStateStore(BaseHotStateStore):
    def __init__(self):
        self._map = {}

    def set(self, key, value, ttl_seconds=None):
        self._map[str(key)] = _safe_payload(value)
        return True

    def get(self, key, default=None):
        return self._map.get(str(key), default)

    def stats(self):
        return {"keys": len(self._map)}


class RedisHotStateStore(BaseHotStateStore):
    def __init__(self, url="redis://127.0.0.1:6379/0", key_prefix="hermes:soc"):
        try:
            import redis
        except Exception as exc:
            raise RuntimeError("redis package required. Install with `pip install redis`.") from exc

        self.client = redis.Redis.from_url(url)
        self.key_prefix = str(key_prefix)

    def _k(self, key):
        return f"{self.key_prefix}:{key}"

    def set(self, key, value, ttl_seconds=None):
        k = self._k(key)
        payload = json.dumps(_safe_payload(value))
        if ttl_seconds is not None:
            self.client.setex(k, int(max(1, ttl_seconds)), payload)
        else:
            self.client.set(k, payload)
        return True

    def get(self, key, default=None):
        raw = self.client.get(self._k(key))
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default

    def stats(self):
        try:
            info = self.client.info()
            return {
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
            }
        except Exception:
            return {}


class BaseEventStore:
    def append(self, table, event):
        raise NotImplementedError

    def query_recent(self, table, limit=100):
        raise NotImplementedError

    def stats(self):
        return {}


class InMemoryEventStore(BaseEventStore):
    def __init__(self, max_events_per_table=10000):
        self.max_events_per_table = max(100, int(max_events_per_table))
        self.tables = defaultdict(lambda: deque(maxlen=self.max_events_per_table))

    def append(self, table, event):
        table = str(table)
        self.tables[table].append(_safe_payload(event))
        return True

    def query_recent(self, table, limit=100):
        table = str(table)
        limit = max(1, int(limit))
        return list(self.tables.get(table, []))[-limit:]

    def stats(self):
        return {"tables": {k: len(v) for k, v in self.tables.items()}}


class ClickHouseEventStore(BaseEventStore):
    """JSONEachRow append adapter for ClickHouse event persistence."""

    def __init__(self, host="127.0.0.1", port=8123, username="default", password="", database="default"):
        try:
            import clickhouse_connect
        except Exception as exc:
            raise RuntimeError(
                "clickhouse-connect package required. Install with `pip install clickhouse-connect`."
            ) from exc

        self.client = clickhouse_connect.get_client(
            host=host,
            port=int(port),
            username=username,
            password=password,
            database=database,
        )

    def append(self, table, event):
        table = str(table)
        event = _safe_payload(event)
        # Generic fallback: store a single JSON payload column named `payload`.
        self.client.insert(table, [{"payload": event}], column_names=["payload"])
        return True

    def query_recent(self, table, limit=100):
        table = str(table)
        limit = max(1, int(limit))
        query = f"SELECT payload FROM {table} ORDER BY timestamp DESC LIMIT {limit}"
        rows = self.client.query(query).result_rows
        return [row[0] for row in rows]


class BaseClipStore:
    def store_clip(self, clip_id, clip_payload, metadata=None):
        raise NotImplementedError


class FilesystemClipStore(BaseClipStore):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def store_clip(self, clip_id, clip_payload, metadata=None):
        clip_id = str(clip_id)
        path = self.root_dir / f"{clip_id}.json"
        payload = {
            "clip_id": clip_id,
            "clip_payload": _safe_payload(clip_payload),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        with open(path, "w") as fp:
            json.dump(payload, fp)
        return str(path)
