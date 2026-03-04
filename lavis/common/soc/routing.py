"""Runtime message routing with congestion control and dead-letter handling."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict

from .schemas import utc_now_iso


NATS_SUBJECTS = {
    "video_obs_raw": "video.obs.raw",
    "video_entity_tracks": "video.entity.tracks",
    "video_event_posterior": "video.event.posterior",
    "threat_alert_candidate": "threat.alert.candidate",
    "threat_alert_confirmed": "threat.alert.confirmed",
    "soc_case_updates": "soc.case.updates",
}


class AlertPriority:
    HIGH = 100
    MEDIUM = 60
    LOW = 20


def _subject_priority(subject):
    if subject == NATS_SUBJECTS["threat_alert_confirmed"]:
        return AlertPriority.HIGH
    if subject == NATS_SUBJECTS["threat_alert_candidate"]:
        return 90
    if subject == NATS_SUBJECTS["soc_case_updates"]:
        return 80
    if subject == NATS_SUBJECTS["video_event_posterior"]:
        return AlertPriority.MEDIUM
    if subject == NATS_SUBJECTS["video_entity_tracks"]:
        return 50
    return AlertPriority.LOW


@dataclass
class RoutedMessage:
    subject: str
    payload: Dict[str, Any]
    priority: int
    attempts: int = 0
    created_at_utc: str = field(default_factory=utc_now_iso)
    last_error: str = ""

    def to_dict(self):
        return {
            "subject": self.subject,
            "payload": dict(self.payload),
            "priority": int(self.priority),
            "attempts": int(self.attempts),
            "created_at_utc": self.created_at_utc,
            "last_error": self.last_error,
        }


class RuntimeRoutingPolicyService:
    """Subject-prioritized queueing with retry + dead-letter semantics."""

    def __init__(
        self,
        max_backlog=5000,
        congestion_soft_limit=3000,
        max_retries=3,
        priority_drop_threshold=60,
    ):
        self.max_backlog = max(1, int(max_backlog))
        self.congestion_soft_limit = max(1, int(congestion_soft_limit))
        self.max_retries = max(0, int(max_retries))
        self.priority_drop_threshold = int(priority_drop_threshold)

        self.queues = defaultdict(deque)
        self.retry_queue = deque()
        self.dead_letter_queue = deque()

        self.counters = {
            "published": 0,
            "dispatched": 0,
            "dropped_congestion": 0,
            "retried": 0,
            "dead_lettered": 0,
            "failed_dispatch": 0,
        }

    def _total_backlog(self):
        base = sum(len(q) for q in self.queues.values())
        base += len(self.retry_queue)
        return int(base)

    def _drop_low_priority_if_needed(self):
        while self._total_backlog() > self.max_backlog:
            dropped = False
            for subject, queue in sorted(self.queues.items()):
                if not queue:
                    continue
                if _subject_priority(subject) >= self.priority_drop_threshold:
                    continue
                queue.popleft()
                self.counters["dropped_congestion"] += 1
                dropped = True
                break
            if not dropped:
                # backlog exceeded with only high-priority messages; keep and break
                break

    def publish(self, subject, payload):
        msg = RoutedMessage(
            subject=str(subject),
            payload=payload if isinstance(payload, dict) else {"value": payload},
            priority=_subject_priority(subject),
        )
        self.queues[msg.subject].append(msg)
        self.counters["published"] += 1
        self._drop_low_priority_if_needed()
        return msg.to_dict()

    def on_delivery_failure(self, message, error_text):
        if not isinstance(message, RoutedMessage):
            return
        message.attempts += 1
        message.last_error = str(error_text)
        self.counters["failed_dispatch"] += 1
        if message.attempts > self.max_retries:
            self.dead_letter_queue.append(message)
            self.counters["dead_lettered"] += 1
            return
        self.retry_queue.append(message)
        self.counters["retried"] += 1

    def dispatch_step(self, max_dispatch=64, fail_predicate=None):
        max_dispatch = max(1, int(max_dispatch))
        dispatched = []

        while self.retry_queue and len(dispatched) < max_dispatch:
            msg = self.retry_queue.popleft()
            if callable(fail_predicate) and fail_predicate(msg):
                self.on_delivery_failure(msg, "retry delivery failure")
                continue
            self.counters["dispatched"] += 1
            dispatched.append(msg.to_dict())

        subjects = sorted(self.queues.keys(), key=lambda s: _subject_priority(s), reverse=True)
        for subject in subjects:
            queue = self.queues[subject]
            while queue and len(dispatched) < max_dispatch:
                msg = queue.popleft()
                if callable(fail_predicate) and fail_predicate(msg):
                    self.on_delivery_failure(msg, "dispatch delivery failure")
                    continue
                self.counters["dispatched"] += 1
                dispatched.append(msg.to_dict())
            if len(dispatched) >= max_dispatch:
                break

        return dispatched

    def metrics(self):
        by_subject = {
            subject: len(queue) for subject, queue in self.queues.items() if len(queue) > 0
        }
        return {
            "backlog_total": self._total_backlog(),
            "backlog_retry": len(self.retry_queue),
            "backlog_dead_letter": len(self.dead_letter_queue),
            "by_subject": by_subject,
            "counters": dict(self.counters),
            "congested": self._total_backlog() >= self.congestion_soft_limit,
        }

    def dead_letter_snapshot(self, limit=25):
        limit = max(1, int(limit))
        out = []
        for msg in list(self.dead_letter_queue)[:limit]:
            out.append(msg.to_dict())
        return out
