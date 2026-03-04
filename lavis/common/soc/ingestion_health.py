"""Ingestion health checks: stream drop, timestamp skew, and FPS drift."""

import time
from collections import defaultdict, deque
from datetime import datetime, timezone


def _to_unix_seconds(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        value = ts.strip()
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return None
    return None


def _safe_float(value, default_value):
    try:
        return float(value)
    except Exception:
        return float(default_value)


class IngestionHealthMonitor:
    def __init__(
        self,
        expected_fps=10.0,
        fps_window=60,
        drop_timeout_seconds=5.0,
        max_timestamp_skew_seconds=2.0,
        max_fps_drift_ratio=0.35,
    ):
        self.expected_fps = max(0.1, float(expected_fps))
        self.fps_window = max(4, int(fps_window))
        self.drop_timeout_seconds = max(0.5, float(drop_timeout_seconds))
        self.max_timestamp_skew_seconds = max(0.0, float(max_timestamp_skew_seconds))
        self.max_fps_drift_ratio = max(0.0, float(max_fps_drift_ratio))
        self._state = defaultdict(self._new_camera_state)

    def _new_camera_state(self):
        return {
            "frame_times": deque(maxlen=self.fps_window),
            "last_ingest_wall_time": None,
            "last_capture_ts": None,
            "last_frame_index": None,
            "drops_detected": 0,
        }

    def update(self, site_id, camera_id, capture_timestamp_utc, frame_index=None, now_time=None):
        key = (str(site_id), str(camera_id))
        st = self._state[key]
        wall_now = time.time() if now_time is None else float(now_time)
        capture_ts = _to_unix_seconds(capture_timestamp_utc)

        if st["last_ingest_wall_time"] is not None:
            gap = wall_now - float(st["last_ingest_wall_time"])
            if gap > self.drop_timeout_seconds:
                st["drops_detected"] += 1
        st["last_ingest_wall_time"] = wall_now

        if capture_ts is not None:
            st["frame_times"].append(capture_ts)
            st["last_capture_ts"] = capture_ts
        elif st["last_capture_ts"] is not None:
            # fallback to ingest time if source timestamp is unavailable
            fallback_ts = wall_now
            st["frame_times"].append(fallback_ts)
            st["last_capture_ts"] = fallback_ts

        if frame_index is not None:
            st["last_frame_index"] = int(frame_index)

        fps_estimate = self._estimate_fps(st["frame_times"])
        fps_drift_ratio = abs(fps_estimate - self.expected_fps) / self.expected_fps
        timestamp_skew = None
        if capture_ts is not None:
            timestamp_skew = abs(wall_now - capture_ts)

        status = {
            "site_id": str(site_id),
            "camera_id": str(camera_id),
            "expected_fps": float(self.expected_fps),
            "fps_estimate": float(fps_estimate),
            "fps_drift_ratio": float(fps_drift_ratio),
            "fps_drift_alarm": bool(fps_drift_ratio > self.max_fps_drift_ratio),
            "timestamp_skew_seconds": (
                float(timestamp_skew) if timestamp_skew is not None else None
            ),
            "timestamp_skew_alarm": bool(
                timestamp_skew is not None
                and timestamp_skew > self.max_timestamp_skew_seconds
            ),
            "stream_drop_alarm": bool(st["drops_detected"] > 0),
            "drops_detected": int(st["drops_detected"]),
            "last_frame_index": st["last_frame_index"],
            "updated_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        return status

    @staticmethod
    def _estimate_fps(frame_times):
        if len(frame_times) < 2:
            return 0.0
        dt = _safe_float(frame_times[-1], 0.0) - _safe_float(frame_times[0], 0.0)
        if dt <= 1e-9:
            return 0.0
        return float(len(frame_times) - 1) / float(dt)
