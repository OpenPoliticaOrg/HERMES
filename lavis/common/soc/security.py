"""Security and governance primitives for SOC runtime."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _hash_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


@dataclass
class TransportSecurityConfig:
    mtls_internal: bool = False
    tls_external: bool = False
    cert_path: str = ""
    key_path: str = ""
    ca_path: str = ""

    def validate(self):
        issues = []
        if self.mtls_internal or self.tls_external:
            for field_name, value in [
                ("cert_path", self.cert_path),
                ("key_path", self.key_path),
                ("ca_path", self.ca_path),
            ]:
                if value and not Path(value).exists():
                    issues.append(f"missing_{field_name}:{value}")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "mtls_internal": bool(self.mtls_internal),
            "tls_external": bool(self.tls_external),
        }


class RBACPolicyEngine:
    """Simple least-privilege RBAC evaluator."""

    def __init__(self, roles=None, service_accounts=None):
        self.roles = roles or {
            "soc_runtime": [
                "emit:video_obs_raw",
                "emit:entity_track",
                "emit:event_posterior",
                "emit:threat_candidate",
                "emit:case_update",
                "emit:bus_publish",
                "store:hot_state",
                "store:event",
                "store:clip",
                "monitor:drift",
                "monitor:slo",
                "audit:write",
            ],
            "soc_dispatcher": [
                "emit:threat_confirmed",
                "emit:case_update",
            ],
            "soc_analyst": [
                "case:ack",
                "case:confirm",
                "case:dismiss",
                "feedback:ingest",
            ],
            "soc_admin": ["*"],
        }
        self.service_accounts = service_accounts or {
            "svc_soc_runtime": {
                "role": "soc_runtime",
                "sites": ["*"],
            }
        }

    @classmethod
    def from_dict(cls, payload):
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            roles=payload.get("roles", None),
            service_accounts=payload.get("service_accounts", None),
        )

    def authorize(self, account_id, action, site_id=None):
        account = self.service_accounts.get(str(account_id))
        if account is None:
            return False

        role = str(account.get("role", ""))
        permissions = self.roles.get(role, [])
        if not isinstance(permissions, list):
            permissions = []

        site_id = str(site_id) if site_id is not None else None
        allowed_sites = account.get("sites", ["*"])
        if not isinstance(allowed_sites, list):
            allowed_sites = ["*"]
        site_allowed = "*" in allowed_sites or site_id is None or site_id in allowed_sites
        if not site_allowed:
            return False

        if "*" in permissions:
            return True
        return str(action) in permissions

    def snapshot(self):
        return {
            "roles": sorted(list(self.roles.keys())),
            "service_accounts": sorted(list(self.service_accounts.keys())),
        }


class ImmutableAuditLog:
    """Append-only hash-chained audit log."""

    def __init__(self):
        self.entries = []

    def append(self, actor, action, details=None, timestamp_utc=None):
        prev_hash = self.entries[-1]["entry_hash"] if self.entries else "genesis"
        body = {
            "index": len(self.entries),
            "timestamp_utc": timestamp_utc,
            "actor": str(actor),
            "action": str(action),
            "details": details if isinstance(details, dict) else {},
            "prev_hash": prev_hash,
        }
        entry_hash = _hash_text(json.dumps(body, sort_keys=True))
        body["entry_hash"] = entry_hash
        self.entries.append(body)
        return body

    def verify(self):
        if not self.entries:
            return {"valid": True, "entries": 0, "broken_at": None}

        prev_hash = "genesis"
        for idx, item in enumerate(self.entries):
            compare = dict(item)
            item_hash = compare.pop("entry_hash", None)
            expected_prev = compare.get("prev_hash")
            if expected_prev != prev_hash:
                return {"valid": False, "entries": len(self.entries), "broken_at": idx}
            expected_hash = _hash_text(json.dumps(compare, sort_keys=True))
            if item_hash != expected_hash:
                return {"valid": False, "entries": len(self.entries), "broken_at": idx}
            prev_hash = item_hash
        return {"valid": True, "entries": len(self.entries), "broken_at": None}

    def snapshot(self, limit=50):
        limit = max(1, int(limit))
        return {
            "entries": list(self.entries[-limit:]),
            "integrity": self.verify(),
        }
