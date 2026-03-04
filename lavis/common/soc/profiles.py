"""Hardware-aware inference profile routing for edge/core deployments."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _safe_int_env(name, default_value):
    try:
        return int(os.environ.get(name, default_value))
    except Exception:
        return int(default_value)


def detect_runtime_hardware():
    cuda_available = False
    cuda_devices = 0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:
        cuda_available = False
        cuda_devices = 0

    cpu_count = os.cpu_count() or 1
    memory_gb_hint = _safe_int_env("HERMES_MEMORY_GB_HINT", 0)
    site_role = os.environ.get("HERMES_SITE_ROLE", "edge").strip().lower()
    if site_role not in {"edge", "core"}:
        site_role = "edge"

    return {
        "site_role": site_role,
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "cpu_count": int(cpu_count),
        "memory_gb_hint": int(memory_gb_hint),
    }


@dataclass
class InferenceProfile:
    profile_id: str
    detector_name: str
    tracker_name: str
    reid_name: Optional[str]
    expected_latency_ms: int
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "profile_id": self.profile_id,
            "detector_name": self.detector_name,
            "tracker_name": self.tracker_name,
            "reid_name": self.reid_name,
            "expected_latency_ms": int(self.expected_latency_ms),
            "metadata": dict(self.metadata),
        }


class InferenceProfileService:
    """Dynamic profile selector with edge/core hardware fallbacks."""

    def __init__(self):
        self.profiles = {
            "edge_gpu_profile": InferenceProfile(
                profile_id="edge_gpu_profile",
                detector_name="lightweight_detector",
                tracker_name="bytetrack",
                reid_name="compact_reid",
                expected_latency_ms=450,
                metadata={"hardware": "gpu", "tier": "edge"},
            ),
            "edge_cpu_profile": InferenceProfile(
                profile_id="edge_cpu_profile",
                detector_name="quantized_lightweight_detector",
                tracker_name="sparse_tracker",
                reid_name=None,
                expected_latency_ms=1500,
                metadata={"hardware": "cpu", "tier": "edge"},
            ),
            "core_gpu_profile": InferenceProfile(
                profile_id="core_gpu_profile",
                detector_name="high_accuracy_detector",
                tracker_name="strongsort_style",
                reid_name="strong_reid",
                expected_latency_ms=650,
                metadata={"hardware": "gpu", "tier": "core"},
            ),
            "auto_motion_fallback": InferenceProfile(
                profile_id="auto_motion_fallback",
                detector_name="auto_motion",
                tracker_name="motion_blob_association",
                reid_name=None,
                expected_latency_ms=120,
                metadata={"hardware": "agnostic", "tier": "fallback"},
            ),
        }

    def select_profile(self, hardware_snapshot=None, force_profile=None):
        if force_profile and force_profile in self.profiles:
            return self.profiles[force_profile]

        hw = hardware_snapshot or detect_runtime_hardware()
        role = str(hw.get("site_role", "edge")).lower()
        cuda_available = bool(hw.get("cuda_available", False))

        if role == "core" and cuda_available:
            return self.profiles["core_gpu_profile"]
        if role == "edge" and cuda_available:
            return self.profiles["edge_gpu_profile"]
        if role in {"edge", "core"}:
            return self.profiles["edge_cpu_profile"]
        return self.profiles["auto_motion_fallback"]

    def list_profiles(self):
        return [self.profiles[k].to_dict() for k in sorted(self.profiles.keys())]
