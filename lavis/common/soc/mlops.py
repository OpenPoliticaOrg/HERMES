"""MLOps and reliability helpers for SOC model lifecycle and monitoring."""

import hashlib
import json
from collections import defaultdict, deque


def _clamp01(value):
    try:
        x = float(value)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _jensen_shannon_divergence(p, q, eps=1e-9):
    keys = sorted(set(p.keys()).union(set(q.keys())))
    if not keys:
        return 0.0

    p_vec = []
    q_vec = []
    p_sum = 0.0
    q_sum = 0.0
    for key in keys:
        pv = max(_safe_float(p.get(key, 0.0), 0.0), 0.0)
        qv = max(_safe_float(q.get(key, 0.0), 0.0), 0.0)
        p_vec.append(pv)
        q_vec.append(qv)
        p_sum += pv
        q_sum += qv

    if p_sum <= eps:
        p_vec = [1.0 / float(len(keys))] * len(keys)
    else:
        p_vec = [x / p_sum for x in p_vec]

    if q_sum <= eps:
        q_vec = [1.0 / float(len(keys))] * len(keys)
    else:
        q_vec = [x / q_sum for x in q_vec]

    m_vec = [(a + b) * 0.5 for a, b in zip(p_vec, q_vec)]

    def _kl(a, b):
        total = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, eps)
            bi = max(bi, eps)
            total += ai * (ai / bi)
        return total

    # Convert from multiplicative KL approximation above into bounded surrogate.
    kl_pm = _kl(p_vec, m_vec)
    kl_qm = _kl(q_vec, m_vec)
    js = 0.5 * (kl_pm + kl_qm) - 1.0
    return max(0.0, js)


class SignedModelRegistry:
    """Registers model artifacts with deterministic hash signatures."""

    def __init__(self, signing_key=""):
        self.signing_key = str(signing_key or "")
        self.entries = []

    @classmethod
    def from_dict(cls, payload):
        payload = payload if isinstance(payload, dict) else {}
        return cls(signing_key=payload.get("signing_key", ""))

    def _sign(self, digest):
        if self.signing_key:
            return _hash_text(f"{self.signing_key}:{digest}")
        return _hash_text(f"unsigned:{digest}")

    def register_model(self, model_id, version, artifact_path, metadata=None):
        digest = _hash_file(artifact_path)
        signature = self._sign(digest)
        entry = {
            "model_id": str(model_id),
            "version": str(version),
            "artifact_path": str(artifact_path),
            "sha256": digest,
            "signature": signature,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        self.entries.append(entry)
        return entry

    def verify_model(self, entry, artifact_path=None):
        entry = entry if isinstance(entry, dict) else {}
        path = artifact_path or entry.get("artifact_path")
        if not path:
            return False
        digest = _hash_file(path)
        expected_sig = self._sign(digest)
        return digest == entry.get("sha256") and expected_sig == entry.get("signature")

    def latest(self, model_id):
        model_id = str(model_id)
        candidates = [x for x in self.entries if x.get("model_id") == model_id]
        if not candidates:
            return None
        return candidates[-1]

    def snapshot(self):
        return {
            "model_count": len(self.entries),
            "models": list(self.entries),
        }


class CanaryRolloutManager:
    """Deterministic site/camera canary assignment with rollback support."""

    def __init__(self, default_ratio=0.1):
        self.default_ratio = _clamp01(default_ratio)
        self.rollouts = {}

    @classmethod
    def from_dict(cls, payload):
        payload = payload if isinstance(payload, dict) else {}
        return cls(default_ratio=payload.get("default_ratio", 0.1))

    def set_rollout(self, rollout_id, baseline_profile, canary_profile, canary_ratio=None):
        ratio = self.default_ratio if canary_ratio is None else _clamp01(canary_ratio)
        self.rollouts[str(rollout_id)] = {
            "baseline_profile": str(baseline_profile),
            "canary_profile": str(canary_profile),
            "canary_ratio": float(ratio),
            "active": True,
        }
        return self.rollouts[str(rollout_id)]

    def rollback(self, rollout_id):
        cfg = self.rollouts.get(str(rollout_id))
        if cfg is None:
            return None
        cfg["active"] = False
        return cfg

    def resolve_profile(self, rollout_id, site_id, camera_id=None):
        cfg = self.rollouts.get(str(rollout_id))
        if cfg is None or not bool(cfg.get("active", False)):
            return None

        key = f"{site_id}|{camera_id or ''}"
        h = int(_hash_text(key), 16) % 10000
        frac = h / 10000.0
        if frac < float(cfg.get("canary_ratio", 0.0)):
            return cfg.get("canary_profile")
        return cfg.get("baseline_profile")

    def snapshot(self):
        return {"rollouts": dict(self.rollouts)}


class DriftMonitor:
    """Monitors class-prior drift, embedding drift, and alert-volume anomalies."""

    def __init__(
        self,
        class_window=200,
        embedding_window=200,
        alert_window=300,
        class_drift_threshold=0.12,
        embedding_drift_threshold=0.30,
        alert_z_threshold=3.0,
    ):
        self.class_window = max(16, int(class_window))
        self.embedding_window = max(16, int(embedding_window))
        self.alert_window = max(16, int(alert_window))
        self.class_drift_threshold = float(max(0.0, class_drift_threshold))
        self.embedding_drift_threshold = float(max(0.0, embedding_drift_threshold))
        self.alert_z_threshold = float(max(0.5, alert_z_threshold))

        self.class_events = deque(maxlen=self.class_window)
        self.embedding_values = deque(maxlen=self.embedding_window)
        self.alert_counts = deque(maxlen=self.alert_window)

        self.class_baseline = defaultdict(float)
        self.embedding_baseline = None
        self.alert_ewma = None

    @staticmethod
    def _hash_embedding(value):
        digest = _hash_text(value)
        return int(digest[:8], 16) / float(16 ** 8)

    def _class_distribution(self, data):
        counts = defaultdict(float)
        for item in data:
            counts[str(item)] += 1.0
        total = sum(counts.values())
        if total <= 1e-9:
            return {}
        return {k: v / total for k, v in counts.items()}

    def update(self, class_event_ids=None, embedding_refs=None, alert_count=0):
        class_event_ids = class_event_ids or []
        embedding_refs = embedding_refs or []

        for event_id in class_event_ids:
            self.class_events.append(str(event_id))

        for emb in embedding_refs:
            self.embedding_values.append(self._hash_embedding(emb))

        self.alert_counts.append(max(0, int(alert_count)))

        class_current = self._class_distribution(self.class_events)
        if len(self.class_baseline) == 0 and class_current:
            self.class_baseline.update(class_current)

        class_js = _jensen_shannon_divergence(self.class_baseline, class_current)

        emb_mean = None
        emb_drift = 0.0
        if self.embedding_values:
            emb_mean = sum(self.embedding_values) / float(len(self.embedding_values))
            if self.embedding_baseline is None:
                self.embedding_baseline = emb_mean
            emb_drift = abs(float(emb_mean) - float(self.embedding_baseline))

        mean_alert = None
        std_alert = None
        z_alert = 0.0
        if len(self.alert_counts) >= 2:
            mean_alert = sum(self.alert_counts) / float(len(self.alert_counts))
            variance = sum((x - mean_alert) ** 2 for x in self.alert_counts) / float(
                max(1, len(self.alert_counts) - 1)
            )
            std_alert = max(variance ** 0.5, 1e-6)
            z_alert = (self.alert_counts[-1] - mean_alert) / std_alert

            if self.alert_ewma is None:
                self.alert_ewma = mean_alert
            alpha = 0.05
            self.alert_ewma = (1.0 - alpha) * self.alert_ewma + alpha * self.alert_counts[-1]

        return {
            "class_prior_drift": float(class_js),
            "class_prior_drift_alarm": bool(class_js >= self.class_drift_threshold),
            "embedding_drift": float(emb_drift),
            "embedding_drift_alarm": bool(emb_drift >= self.embedding_drift_threshold),
            "alert_volume_zscore": float(z_alert),
            "alert_volume_alarm": bool(abs(z_alert) >= self.alert_z_threshold),
            "class_distribution": class_current,
            "alert_ewma": _safe_float(self.alert_ewma, 0.0),
            "alert_mean": _safe_float(mean_alert, 0.0),
            "alert_std": _safe_float(std_alert, 0.0),
        }


class SLOMonitor:
    """Tracks latency/availability SLOs and alert-rate budget signals."""

    def __init__(
        self,
        latency_window=500,
        target_p95_latency_gpu_s=2.0,
        target_p95_latency_cpu_s=4.0,
        error_budget=0.01,
    ):
        self.latency_window = max(16, int(latency_window))
        self.target_p95_latency_gpu_s = float(max(0.1, target_p95_latency_gpu_s))
        self.target_p95_latency_cpu_s = float(max(0.1, target_p95_latency_cpu_s))
        self.error_budget = _clamp01(error_budget)

        self.latencies = deque(maxlen=self.latency_window)
        self.successes = deque(maxlen=self.latency_window)

    @staticmethod
    def _p95(values):
        if not values:
            return None
        vals = sorted(values)
        idx = int(round(0.95 * (len(vals) - 1)))
        idx = max(0, min(idx, len(vals) - 1))
        return float(vals[idx])

    def record(self, processing_seconds, success=True, profile_id=""):
        self.latencies.append(max(0.0, float(processing_seconds)))
        self.successes.append(1 if bool(success) else 0)

        p95 = self._p95(self.latencies)
        availability = sum(self.successes) / float(len(self.successes)) if self.successes else 1.0

        profile_id = str(profile_id)
        if "cpu" in profile_id:
            target = self.target_p95_latency_cpu_s
        else:
            target = self.target_p95_latency_gpu_s

        latency_alarm = p95 is not None and p95 > target
        availability_alarm = availability < (1.0 - self.error_budget)

        return {
            "latency_p95_s": p95,
            "latency_target_s": float(target),
            "latency_alarm": bool(latency_alarm),
            "availability": float(availability),
            "availability_target": float(1.0 - self.error_budget),
            "availability_alarm": bool(availability_alarm),
            "samples": len(self.latencies),
        }


class RolloutGuardrailPolicy:
    """Sustained-alarm guardrail that decides when to rollback canary rollout."""

    def __init__(
        self,
        enabled=True,
        consecutive_alarm_windows=3,
        min_samples_for_slo=32,
        rollback_on_drift=True,
        rollback_on_slo=True,
        drift_alarm_fields=None,
        slo_alarm_fields=None,
        max_rollback_history=64,
    ):
        self.enabled = bool(enabled)
        self.consecutive_alarm_windows = max(1, int(consecutive_alarm_windows))
        self.min_samples_for_slo = max(1, int(min_samples_for_slo))
        self.rollback_on_drift = bool(rollback_on_drift)
        self.rollback_on_slo = bool(rollback_on_slo)
        self.drift_alarm_fields = list(
            drift_alarm_fields
            if isinstance(drift_alarm_fields, list) and len(drift_alarm_fields) > 0
            else [
                "class_prior_drift_alarm",
                "embedding_drift_alarm",
                "alert_volume_alarm",
            ]
        )
        self.slo_alarm_fields = list(
            slo_alarm_fields
            if isinstance(slo_alarm_fields, list) and len(slo_alarm_fields) > 0
            else ["latency_alarm", "availability_alarm"]
        )
        self.rollback_history = deque(maxlen=max(8, int(max_rollback_history)))
        self._consecutive_alarm_hits = 0
        self._last_decision = {}

    @classmethod
    def from_dict(cls, payload):
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            enabled=payload.get("enabled", True),
            consecutive_alarm_windows=payload.get("consecutive_alarm_windows", 3),
            min_samples_for_slo=payload.get("min_samples_for_slo", 32),
            rollback_on_drift=payload.get("rollback_on_drift", True),
            rollback_on_slo=payload.get("rollback_on_slo", True),
            drift_alarm_fields=payload.get("drift_alarm_fields", None),
            slo_alarm_fields=payload.get("slo_alarm_fields", None),
            max_rollback_history=payload.get("max_rollback_history", 64),
        )

    @staticmethod
    def _active_alarm_fields(metrics, fields):
        metrics = metrics if isinstance(metrics, dict) else {}
        out = []
        for name in fields:
            if bool(metrics.get(name, False)):
                out.append(str(name))
        return out

    def evaluate(self, drift_metrics=None, slo_metrics=None):
        drift_metrics = drift_metrics if isinstance(drift_metrics, dict) else {}
        slo_metrics = slo_metrics if isinstance(slo_metrics, dict) else {}

        if not self.enabled:
            decision = {
                "enabled": False,
                "triggered": False,
                "alarm_reasons": [],
                "consecutive_alarm_hits": 0,
                "required_consecutive_hits": self.consecutive_alarm_windows,
                "slo_sample_gate": True,
            }
            self._last_decision = decision
            return decision

        alarm_reasons = []
        if self.rollback_on_drift:
            active = self._active_alarm_fields(drift_metrics, self.drift_alarm_fields)
            for field_name in active:
                alarm_reasons.append(f"drift:{field_name}")

        slo_samples = int(max(0, slo_metrics.get("samples", 0)))
        slo_sample_gate = slo_samples >= self.min_samples_for_slo
        if self.rollback_on_slo and slo_sample_gate:
            active = self._active_alarm_fields(slo_metrics, self.slo_alarm_fields)
            for field_name in active:
                alarm_reasons.append(f"slo:{field_name}")

        if len(alarm_reasons) > 0:
            self._consecutive_alarm_hits += 1
        else:
            self._consecutive_alarm_hits = 0

        triggered = (
            len(alarm_reasons) > 0
            and self._consecutive_alarm_hits >= self.consecutive_alarm_windows
        )

        decision = {
            "enabled": True,
            "triggered": bool(triggered),
            "alarm_reasons": alarm_reasons,
            "consecutive_alarm_hits": int(self._consecutive_alarm_hits),
            "required_consecutive_hits": int(self.consecutive_alarm_windows),
            "slo_sample_gate": bool(slo_sample_gate),
            "slo_samples": int(slo_samples),
        }
        self._last_decision = decision
        return decision

    def record_rollback(self, rollout_id, reason, timestamp_utc):
        item = {
            "rollout_id": str(rollout_id),
            "reason": str(reason),
            "timestamp_utc": str(timestamp_utc),
        }
        self.rollback_history.append(item)
        self._consecutive_alarm_hits = 0
        return item

    def snapshot(self):
        return {
            "enabled": bool(self.enabled),
            "consecutive_alarm_windows": int(self.consecutive_alarm_windows),
            "min_samples_for_slo": int(self.min_samples_for_slo),
            "rollback_on_drift": bool(self.rollback_on_drift),
            "rollback_on_slo": bool(self.rollback_on_slo),
            "drift_alarm_fields": list(self.drift_alarm_fields),
            "slo_alarm_fields": list(self.slo_alarm_fields),
            "rollback_history": list(self.rollback_history),
            "last_decision": dict(self._last_decision),
        }
