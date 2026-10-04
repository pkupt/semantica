"""
Provenance-enabled wrappers for document ingestion.

Tracks: file paths, pages, metadata, ingestion timestamps

Usage:
    from semantica.ingest.ingest_provenance import PDFIngestorWithProvenance

    ingestor = PDFIngestorWithProvenance(provenance=True)
    docs = ingestor.ingest("document.pdf")

Author: Semantica Contributors
License: MIT
"""

from typing import Optional
from datetime import datetime
import uuid


class IngestProvenanceMixin:
    """Mixin for ingest provenance tracking."""

    def __init__(
        self,
        provenance: bool = False,
        agent_id: Optional[str] = None,
        is_automated: bool = True,
        **kwargs,
    ):
        self.provenance = provenance
        self._prov_manager = None
        self._agent_id = agent_id or self.__class__.__name__
        self._is_automated = is_automated

        if provenance:
            try:
                from semantica.provenance import ProvenanceManager
                self._prov_manager = ProvenanceManager()
            except ImportError:
                self.provenance = False


class PDFIngestorWithProvenance(IngestProvenanceMixin):
    """PDF ingestor with provenance tracking."""

    def __init__(
        self,
        provenance: bool = False,
        agent_id: Optional[str] = None,
        is_automated: bool = True,
        **config,
    ):
        from .file_ingestor import FileIngestor

        IngestProvenanceMixin.__init__(
            self, provenance=provenance, agent_id=agent_id, is_automated=is_automated
        )
        self._ingestor = FileIngestor(**config)

    def ingest(self, file_path: str, **kwargs):
        """Ingest PDF with provenance tracking."""
        activity_started_at_time = datetime.utcnow().isoformat()
        docs = self._ingestor.ingest(file_path, **kwargs)
        activity_ended_at_time = datetime.utcnow().isoformat()

        if self.provenance and self._prov_manager:
            for doc in docs:
                doc_id = getattr(doc, 'id', f"doc_{uuid.uuid4().hex[:8]}")
                self._prov_manager.track_entity(
                    entity_id=doc_id,
                    source=file_path,
                    entity_type="document",
                    agent_id=self._agent_id,
                    agent_type="software_agent",
                    is_automated=self._is_automated,
                    activity_started_at_time=activity_started_at_time,
                    activity_ended_at_time=activity_ended_at_time,
                    metadata={
                        "file_type": "pdf",
                        "pages": getattr(doc, 'page_count', None)
                    }
                )

        return docs

    def __getattr__(self, name):
        return getattr(self._ingestor, name)


__all__ = ['PDFIngestorWithProvenance', 'IngestProvenanceMixin']
