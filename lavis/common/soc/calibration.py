"""Confidence calibration utilities for threat/event probabilities."""

import math


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


def _logit(p, eps=1e-6):
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.5


class ConfidenceCalibrator:
    """Calibrates confidence scores with identity/temperature/isotonic strategies."""

    def __init__(self, method="identity", temperature=1.0, isotonic_points=None):
        self.method = str(method or "identity")
        self.temperature = float(max(1e-6, temperature))
        self.isotonic_points = self._normalize_points(isotonic_points)

    @staticmethod
    def _normalize_points(points):
        if not isinstance(points, (list, tuple)):
            return []
        out = []
        for item in points:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            x = _clamp01(item[0])
            y = _clamp01(item[1])
            out.append((x, y))
        out.sort(key=lambda t: t[0])
        if len(out) == 0:
            return []

        monotonic = []
        max_y = 0.0
        for x, y in out:
            max_y = max(max_y, y)
            monotonic.append((x, max_y))
        return monotonic

    @classmethod
    def from_dict(cls, payload):
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            method=payload.get("method", "identity"),
            temperature=payload.get("temperature", 1.0),
            isotonic_points=payload.get("isotonic_points", []),
        )

    def calibrate(self, value):
        p = _clamp01(value)

        if self.method == "temperature":
            z = _logit(p)
            return _clamp01(_sigmoid(z / self.temperature))

        if self.method == "isotonic":
            return self._isotonic_map(p)

        return p

    def _isotonic_map(self, p):
        pts = self.isotonic_points
        if len(pts) == 0:
            return p
        if p <= pts[0][0]:
            return pts[0][1]
        if p >= pts[-1][0]:
            return pts[-1][1]

        for idx in range(1, len(pts)):
            x0, y0 = pts[idx - 1]
            x1, y1 = pts[idx]
            if p < x0 or p > x1:
                continue
            if abs(x1 - x0) <= 1e-12:
                return y1
            ratio = (p - x0) / (x1 - x0)
            return _clamp01(y0 + ratio * (y1 - y0))
        return p

    def fit_temperature(self, scores, labels):
        """Simple grid-search fitting for temperature scaling."""
        xs = [_clamp01(x) for x in (scores or [])]
        ys = [1 if int(y) == 1 else 0 for y in (labels or [])]
        n = min(len(xs), len(ys))
        if n == 0:
            return self.temperature

        best_t = self.temperature
        best_loss = None
        for t in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]:
            loss = 0.0
            for idx in range(n):
                p = _sigmoid(_logit(xs[idx]) / t)
                y = ys[idx]
                p = min(max(p, 1e-6), 1.0 - 1e-6)
                loss += -y * math.log(p) - (1 - y) * math.log(1.0 - p)
            loss /= float(n)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = float(best_t)
        self.method = "temperature"
        return self.temperature

    def fit_isotonic(self, scores, labels, bins=10):
        """Monotonic reliability-table fit usable as isotonic surrogate."""
        xs = [_clamp01(x) for x in (scores or [])]
        ys = [1 if int(y) == 1 else 0 for y in (labels or [])]
        n = min(len(xs), len(ys))
        if n == 0:
            return []

        bins = max(2, int(bins))
        bucket_sum = [0.0] * bins
        bucket_count = [0] * bins
        for i in range(n):
            idx = min(int(xs[i] * bins), bins - 1)
            bucket_sum[idx] += float(ys[i])
            bucket_count[idx] += 1

        points = []
        for b in range(bins):
            x_mid = (b + 0.5) / float(bins)
            if bucket_count[b] == 0:
                continue
            y_hat = bucket_sum[b] / float(bucket_count[b])
            points.append((x_mid, y_hat))

        self.isotonic_points = self._normalize_points(points)
        self.method = "isotonic"
        return list(self.isotonic_points)

    def to_dict(self):
        return {
            "method": self.method,
            "temperature": float(self.temperature),
            "isotonic_points": [list(x) for x in self.isotonic_points],
        }
