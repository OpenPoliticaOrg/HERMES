"""Cross-camera/site metadata federation for entity continuity."""

from datetime import datetime


def _to_epoch(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    text = str(ts)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _tokenize_embedding_ref(value):
    text = str(value or "").strip().lower()
    if not text:
        return set()
    tokens = []
    for token in text.replace("-", "_").split("_"):
        token = token.strip()
        if token:
            tokens.append(token)
    return set(tokens)


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


class EntityFederationService:
    """Resolve global entity IDs across camera/site boundaries using metadata only."""

    def __init__(self, max_time_delta_seconds=20.0, min_embedding_similarity=0.85):
        self.max_time_delta_seconds = float(max(0.1, max_time_delta_seconds))
        self.min_embedding_similarity = float(max(0.0, min(min_embedding_similarity, 1.0)))

        self._local_to_global = {}
        self._global_records = {}
        self._counter = 0

    def _new_global_id(self):
        self._counter += 1
        return f"global_entity_{self._counter:08d}"

    def _local_key(self, site_id, camera_id, entity_id_local):
        return f"{site_id}|{camera_id}|{entity_id_local}"

    def _match_existing(self, timestamp_utc, embedding_ref):
        ts = _to_epoch(timestamp_utc)
        emb_tokens = _tokenize_embedding_ref(embedding_ref)

        best_global = None
        best_score = -1.0
        for gid, rec in self._global_records.items():
            rec_ts = rec.get("last_seen_ts")
            if ts is not None and rec_ts is not None:
                if abs(float(ts) - float(rec_ts)) > self.max_time_delta_seconds:
                    continue
            rec_tokens = rec.get("embedding_tokens", set())
            sim = _jaccard(emb_tokens, rec_tokens)
            if sim >= self.min_embedding_similarity and sim > best_score:
                best_score = sim
                best_global = gid

        return best_global

    def resolve(self, site_id, camera_id, entity_id_local, timestamp_utc=None, embedding_ref=None):
        local_key = self._local_key(site_id, camera_id, entity_id_local)
        if local_key in self._local_to_global:
            gid = self._local_to_global[local_key]
            self._touch(gid, timestamp_utc=timestamp_utc, embedding_ref=embedding_ref)
            return gid

        gid = self._match_existing(timestamp_utc=timestamp_utc, embedding_ref=embedding_ref)
        if gid is None:
            gid = self._new_global_id()

        self._local_to_global[local_key] = gid
        self._touch(gid, timestamp_utc=timestamp_utc, embedding_ref=embedding_ref)
        return gid

    def _touch(self, global_id, timestamp_utc=None, embedding_ref=None):
        rec = self._global_records.setdefault(
            str(global_id),
            {
                "last_seen_ts": None,
                "embedding_tokens": set(),
                "observations": 0,
            },
        )
        ts = _to_epoch(timestamp_utc)
        if ts is not None:
            rec["last_seen_ts"] = float(ts)
        emb_tokens = _tokenize_embedding_ref(embedding_ref)
        if emb_tokens:
            if not rec["embedding_tokens"]:
                rec["embedding_tokens"] = set(emb_tokens)
            else:
                rec["embedding_tokens"] = set(rec["embedding_tokens"]).union(emb_tokens)
        rec["observations"] = int(rec.get("observations", 0)) + 1

    def snapshot(self):
        return {
            "global_entities": len(self._global_records),
            "local_links": len(self._local_to_global),
        }
