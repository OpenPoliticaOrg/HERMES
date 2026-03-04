"""Internal service contracts matching planned gRPC surface."""

from abc import ABC, abstractmethod


class IngestGatewayService(ABC):
    @abstractmethod
    def ingest_observation(self, request):
        pass


class InferenceProfileServiceContract(ABC):
    @abstractmethod
    def resolve_profile(self, request):
        pass


class EntityFusionService(ABC):
    @abstractmethod
    def upsert_entity_track(self, request):
        pass


class ThreatScoringService(ABC):
    @abstractmethod
    def score_threat(self, request):
        pass


class AlertDispatchService(ABC):
    @abstractmethod
    def dispatch_alert(self, request):
        pass


class FeedbackIngestService(ABC):
    @abstractmethod
    def ingest_feedback(self, request):
        pass
