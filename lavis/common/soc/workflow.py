"""SOC human-in-loop alert workflow, SLA tracking, and audit logs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schemas import ThreatEvent, utc_now_iso


VALID_STATES = {"candidate", "analyst_review", "confirmed", "dismissed"}
TERMINAL_STATES = {"confirmed", "dismissed"}


def _iso_to_unix(ts_utc):
    try:
        value = str(ts_utc)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


@dataclass
class AlertCase:
    case_id: str
    state: str
    created_at_utc: str
    updated_at_utc: str
    site_id: Optional[str]
    severity: str
    threat_type: str
    policy_action: str
    confidence: float
    acknowledged_at_utc: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolution_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "state": self.state,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "site_id": self.site_id,
            "severity": self.severity,
            "threat_type": self.threat_type,
            "policy_action": self.policy_action,
            "confidence": float(self.confidence),
            "acknowledged_at_utc": self.acknowledged_at_utc,
            "acknowledged_by": self.acknowledged_by,
            "resolution_reason": self.resolution_reason,
            "metadata": dict(self.metadata),
        }


class AlertWorkflowService:
    def __init__(self, sla_seconds_by_severity=None, runbooks=None):
        self.sla_seconds_by_severity = sla_seconds_by_severity or {
            "critical": 30,
            "high": 120,
            "medium": 300,
            "low": 900,
            "info": 1800,
        }
        self.runbooks = runbooks or {}
        self.case_counter = 0
        self.cases = {}
        self.audit_log = []
        self.feedback_log = []

    def _new_case_id(self):
        self.case_counter += 1
        return f"case_{self.case_counter:08d}"

    def _append_audit(self, case_id, action, actor, details=None):
        self.audit_log.append(
            {
                "timestamp_utc": utc_now_iso(),
                "case_id": str(case_id),
                "action": str(action),
                "actor": str(actor),
                "details": details if isinstance(details, dict) else {},
            }
        )

    def open_candidate_case(self, threat_event):
        if isinstance(threat_event, ThreatEvent):
            data = threat_event.to_dict()
        else:
            data = threat_event if isinstance(threat_event, dict) else {}

        case_id = self._new_case_id()
        now = utc_now_iso()
        case = AlertCase(
            case_id=case_id,
            state="candidate",
            created_at_utc=now,
            updated_at_utc=now,
            site_id=data.get("site_id"),
            severity=str(data.get("severity", "medium")),
            threat_type=str(data.get("threat_type", "unknown")),
            policy_action=str(data.get("policy_action", "review_required")),
            confidence=float(data.get("confidence_calibrated", 0.0)),
            metadata={
                "entity_refs": data.get("entity_refs", []),
                "camera_refs": data.get("camera_refs", []),
                "markov_state": data.get("markov_state"),
                "anomaly_score": data.get("anomaly_score"),
                "fusion_score": data.get("fusion_score"),
                "explanations": data.get("explanations", []),
            },
        )
        self.cases[case_id] = case
        self._append_audit(case_id, "open_candidate", "system", case.to_dict())
        return case.to_dict()

    def transition(self, case_id, new_state, actor="analyst", reason=None):
        case = self.cases.get(str(case_id))
        if case is None:
            return None
        if new_state not in VALID_STATES:
            return None
        if case.state in TERMINAL_STATES:
            return case.to_dict()

        if case.state == "candidate" and new_state not in {"analyst_review", "dismissed", "confirmed"}:
            return None
        if case.state == "analyst_review" and new_state not in {"confirmed", "dismissed"}:
            return None

        case.state = str(new_state)
        case.updated_at_utc = utc_now_iso()
        if new_state in TERMINAL_STATES and reason is not None:
            case.resolution_reason = str(reason)
        self._append_audit(
            case.case_id,
            "transition",
            actor,
            {
                "new_state": case.state,
                "reason": case.resolution_reason,
            },
        )
        return case.to_dict()

    def acknowledge(self, case_id, actor="analyst"):
        case = self.cases.get(str(case_id))
        if case is None:
            return None
        now = utc_now_iso()
        case.acknowledged_at_utc = now
        case.acknowledged_by = str(actor)
        case.updated_at_utc = now
        if case.state == "candidate":
            case.state = "analyst_review"
        self._append_audit(case.case_id, "acknowledge", actor, {"state": case.state})
        return case.to_dict()

    def bind_runbook(self, severity, checklist_items):
        self.runbooks[str(severity)] = [str(x) for x in checklist_items or []]

    def runbook_for_case(self, case_id):
        case = self.cases.get(str(case_id))
        if case is None:
            return []
        return list(self.runbooks.get(case.severity, []))

    def ingest_feedback(self, case_id, actor, label, notes=""):
        case = self.cases.get(str(case_id))
        if case is None:
            return None
        payload = {
            "timestamp_utc": utc_now_iso(),
            "case_id": case.case_id,
            "actor": str(actor),
            "label": str(label),
            "notes": str(notes),
            "threat_type": case.threat_type,
            "severity": case.severity,
        }
        self.feedback_log.append(payload)
        self._append_audit(case.case_id, "feedback", actor, payload)
        return payload

    def sla_breaches(self, now_utc=None):
        now = _iso_to_unix(now_utc or utc_now_iso())
        breached = []
        for case in self.cases.values():
            if case.state in TERMINAL_STATES:
                continue
            created = _iso_to_unix(case.created_at_utc)
            if created is None or now is None:
                continue
            limit = float(self.sla_seconds_by_severity.get(case.severity, 600))
            if (now - created) > limit:
                breached.append(
                    {
                        "case_id": case.case_id,
                        "state": case.state,
                        "severity": case.severity,
                        "elapsed_seconds": max(0.0, now - created),
                        "sla_seconds": limit,
                    }
                )
        return breached

    def snapshot(self):
        return {
            "cases": [self.cases[k].to_dict() for k in sorted(self.cases.keys())],
            "audit_size": len(self.audit_log),
            "feedback_size": len(self.feedback_log),
        }
