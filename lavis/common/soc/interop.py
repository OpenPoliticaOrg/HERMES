"""Ingestion interoperability abstractions for ONVIF/VMS-first integration."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraProfile:
    site_id: str
    camera_id: str
    rtsp_url: str
    onvif_endpoint: Optional[str] = None
    model_name: Optional[str] = None
    stream_profile: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "rtsp_url": self.rtsp_url,
            "onvif_endpoint": self.onvif_endpoint,
            "model_name": self.model_name,
            "stream_profile": self.stream_profile,
            "location": self.location,
            "metadata": dict(self.metadata),
        }


class CameraSourceProvider(ABC):
    @abstractmethod
    def list_cameras(self):
        """Return a list of CameraProfile objects."""


class ClipExporter(ABC):
    @abstractmethod
    def export_clip(self, site_id, camera_id, start_ts_utc, end_ts_utc, reason):
        """Return a clip reference string/URL."""


class PTZControlProxy(ABC):
    @abstractmethod
    def get_status(self, site_id, camera_id):
        """Return read-only PTZ state (phase-1)."""


class ONVIFDiscoveryService:
    """Profile sync helper that can run against static inventory or provider SDK wrappers."""

    def __init__(self, provider=None):
        self.provider = provider
        self._cache = {}  # key=(site_id,camera_id) -> CameraProfile

    @staticmethod
    def from_json_inventory(path):
        with open(path, "r") as fp:
            payload = json.load(fp)
        items = payload.get("cameras", payload if isinstance(payload, list) else [])
        profiles = []
        for item in items:
            if not isinstance(item, dict):
                continue
            site_id = str(item.get("site_id", "site_default"))
            camera_id = str(item.get("camera_id", "camera_unknown"))
            rtsp_url = str(item.get("rtsp_url", ""))
            if not rtsp_url:
                continue
            profiles.append(
                CameraProfile(
                    site_id=site_id,
                    camera_id=camera_id,
                    rtsp_url=rtsp_url,
                    onvif_endpoint=item.get("onvif_endpoint"),
                    model_name=item.get("model_name"),
                    stream_profile=item.get("stream_profile"),
                    location=item.get("location"),
                    metadata=item.get("metadata", {}),
                )
            )
        service = ONVIFDiscoveryService(provider=None)
        for profile in profiles:
            service._cache[(profile.site_id, profile.camera_id)] = profile
        return service

    def discover(self):
        if self.provider is None:
            return list(self._cache.values())
        out = self.provider.list_cameras()
        profiles = []
        for item in out:
            if isinstance(item, CameraProfile):
                profiles.append(item)
            elif isinstance(item, dict):
                try:
                    profiles.append(CameraProfile(**item))
                except Exception:
                    continue
        return profiles

    def sync_profiles(self):
        """Return profile sync delta: added/updated/removed camera IDs."""
        current = {
            (p.site_id, p.camera_id): p
            for p in self.discover()
            if isinstance(p, CameraProfile)
        }
        prev_keys = set(self._cache.keys())
        new_keys = set(current.keys())

        added = sorted(list(new_keys - prev_keys))
        removed = sorted(list(prev_keys - new_keys))
        updated = []

        for key in sorted(prev_keys.intersection(new_keys)):
            before = self._cache[key].to_dict()
            after = current[key].to_dict()
            if before != after:
                updated.append(key)

        self._cache = current

        return {
            "added": [{"site_id": a[0], "camera_id": a[1]} for a in added],
            "updated": [{"site_id": a[0], "camera_id": a[1]} for a in updated],
            "removed": [{"site_id": a[0], "camera_id": a[1]} for a in removed],
            "count": len(self._cache),
        }

    def get_profile(self, site_id, camera_id):
        return self._cache.get((str(site_id), str(camera_id)))


class StaticCameraSourceProvider(CameraSourceProvider):
    def __init__(self, profiles=None):
        self.profiles = [x for x in (profiles or []) if isinstance(x, CameraProfile)]

    def list_cameras(self):
        return list(self.profiles)


class NullClipExporter(ClipExporter):
    def export_clip(self, site_id, camera_id, start_ts_utc, end_ts_utc, reason):
        return (
            f"clip://{site_id}/{camera_id}?start={start_ts_utc}&end={end_ts_utc}"
            f"&reason={reason}"
        )


class ReadOnlyPTZProxy(PTZControlProxy):
    def get_status(self, site_id, camera_id):
        return {
            "site_id": str(site_id),
            "camera_id": str(camera_id),
            "status": "read_only",
            "pan": None,
            "tilt": None,
            "zoom": None,
        }
