"""
dbt Ingestion Module

This module provides dbt ingestion capabilities for the Semantica framework,
turning dbt's ``manifest.json`` / ``catalog.json`` build artifacts into
documents and lineage records that plug directly into ``GraphBuilder``.

dbt artifacts are one of the richest readily available sources of column-level
lineage and model-to-model relationships in a modern data stack, and parsing
them is purely local (no network dependency) for the common path.

Key Features:
    - Local parsing of ``manifest.json`` and ``catalog.json`` artifacts
    - Model, seed and snapshot summaries with column metadata
    - Source (external table) summaries from the ``sources`` block
    - Model-to-model and source-to-model lineage edges
    - Optional dbt Cloud Metadata API access (Phase 2) through the shared,
      SSRF-guarded ``requests`` session
    - ``export_as_documents`` output contract for ``GraphBuilder``
    - Progress tracking and error handling

Main Classes:
    - DbtIngestor: Main dbt ingestion class
    - DbtConnector: Artifact resolution / loading (+ optional Cloud client)
    - DbtData: Data representation for dbt ingestion

Example Usage:
    >>> from semantica.ingest import DbtIngestor
    >>> ingestor = DbtIngestor(project_dir="/path/to/dbt/project")
    >>> data = ingestor.ingest_artifacts()
    >>> documents = ingestor.export_as_documents(data)

Author: Semantica Contributors
License: MIT
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker

try:
    import requests

    REQUESTS_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - depends on env
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from .ssrf import request_with_ssrf_guard
except (ImportError, OSError):  # pragma: no cover - depends on env
    request_with_ssrf_guard = None

__all__ = [
    "DbtData",
    "DbtConnector",
    "DbtIngestor",
]

_MISSING_EXTRA_HINT = (
    "requests is required for the dbt connector. "
    "Install it with: pip install 'semantica[ingest-dbt]'"
)

# Manifest node types that other nodes can reference. Tests and analyses are
# skipped: they are build-time QA, not graph-worthy data assets.
_REFABLE_RESOURCE_TYPES = frozenset({"model", "seed", "snapshot"})


@dataclass
class DbtData:
    """dbt data representation.

    Attributes:
        models: Parsed model/seed/snapshot summaries (one dict per node).
        sources: Parsed dbt source summaries (external tables).
        lineage: Lineage edges, each ``{"source": unique_id, "target": unique_id}``.
        metadata: Run metadata (artifact paths, counts).
        ingested_at: Timestamp of the ingestion run.
    """

    models: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    lineage: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=datetime.now)


class DbtConnector:
    """
    dbt artifact resolution and loading.

    The primary path is local: dbt writes ``manifest.json`` and
    ``catalog.json`` into the project's ``target/`` directory on every
    ``dbt run`` / ``dbt docs generate``, and those files are read straight
    from disk -- no network access and no extra dependency involved.

    For hosted projects (Phase 2), the connector can also fetch metadata
    from the dbt Cloud Metadata API. Those outbound requests go through
    ``request_with_ssrf_guard`` so user-supplied endpoints cannot reach
    private/loopback/link-local address space.

    Example Usage:
        >>> connector = DbtConnector(project_dir="/path/to/dbt/project")
        >>> manifest = connector.load_artifact(connector.get_manifest_path())
    """

    def __init__(
        self,
        project_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        metadata_url: Optional[str] = None,
        cloud_token: Optional[str] = None,
        allow_private_ips: Optional[bool] = None,
        **config,
    ):
        """
        Initialize dbt connector.

        Args:
            project_dir: Path to the dbt project root (defaults to
                ``DBT_PROJECT_DIR`` or the current directory).
            target_dir: Directory containing the dbt artifacts
                (defaults to ``<project_dir>/target``).
            metadata_url: dbt Cloud Metadata API endpoint (Phase 2),
                e.g. ``https://metadata.cloud.getdbt.com/graphql``.
            cloud_token: dbt Cloud service or personal token (Phase 2).
            allow_private_ips: Explicitly allow requests to private networks
                for the Cloud API (for self-hosted dbt Cloud behind a VPN).
            **config: Additional configuration.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(_MISSING_EXTRA_HINT)

        self.logger = get_logger("dbt_connector")

        self.project_dir = Path(
            project_dir or os.getenv("DBT_PROJECT_DIR") or "."
        ).expanduser()
        self.target_dir = (
            Path(target_dir).expanduser() if target_dir else self.project_dir / "target"
        )
        self.metadata_url = metadata_url or os.getenv("DBT_METADATA_URL")
        self.cloud_token = cloud_token or os.getenv("DBT_CLOUD_TOKEN")
        self.allow_private_ips = allow_private_ips
        self.config = config

        self.logger.debug(
            "dbt connector initialized (project_dir=%s, target_dir=%s)",
            self.project_dir,
            self.target_dir,
        )

    def get_artifact_path(self, artifact: str) -> Path:
        """
        Resolve the path of a dbt artifact inside the target directory.

        Args:
            artifact: Artifact file name, e.g. ``"manifest.json"``.

        Returns:
            Path: Resolved artifact path.

        Raises:
            ValidationError: If the artifact does not exist.
        """
        if not artifact:
            raise ValidationError("Artifact name is required (e.g. 'manifest.json').")
        path = self.target_dir / artifact
        if not path.is_file():
            raise ValidationError(
                f"dbt artifact not found: {path}. Run 'dbt run' (and "
                f"'dbt docs generate' for catalog.json) first, or point "
                f"'target_dir' at the directory containing the artifacts."
            )
        return path

    def get_manifest_path(self) -> Path:
        """Resolve the ``manifest.json`` path (raises if absent)."""
        return self.get_artifact_path("manifest.json")

    def get_catalog_path(self) -> Path:
        """Resolve the ``catalog.json`` path (raises if absent)."""
        return self.get_artifact_path("catalog.json")

    def load_artifact(self, path) -> Dict[str, Any]:
        """
        Load a dbt artifact JSON file from disk.

        Args:
            path: Path to the artifact file.

        Returns:
            dict: Parsed artifact contents.

        Raises:
            ProcessingError: If the file cannot be read or parsed.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ProcessingError(
                f"Invalid JSON in dbt artifact {path}: {exc}"
            ) from exc
        except OSError as exc:
            raise ProcessingError(f"Failed to read dbt artifact {path}: {exc}") from exc

    def get_cloud_metadata(
        self,
        account_id: int,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Run a GraphQL query against the dbt Cloud Metadata API (Phase 2).

        Outbound requests go through ``request_with_ssrf_guard``.

        Args:
            account_id: Numeric dbt Cloud account ID.
            query: GraphQL query string.
            variables: GraphQL variables (optional).
            **options: Extra request options forwarded to the guard.

        Returns:
            dict: Parsed GraphQL response.

        Raises:
            ValidationError: If metadata_url / cloud_token are not configured.
            ProcessingError: If the request fails.
        """
        if not self.metadata_url:
            raise ValidationError(
                "dbt Cloud Metadata API endpoint is required for cloud access. "
                "Provide 'metadata_url' or set DBT_METADATA_URL."
            )
        if not self.cloud_token:
            raise ValidationError(
                "dbt Cloud token is required for cloud access. "
                "Provide 'cloud_token' or set DBT_CLOUD_TOKEN."
            )

        payload = {"query": query, "variables": variables or {}}
        try:
            response = request_with_ssrf_guard(
                self.metadata_url,
                method="POST",
                json=payload,
                headers={
                    "Authorization": f"Token {self.cloud_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "semantica-dbt-ingestor",
                },
                allow_private_ips=self.allow_private_ips,
                **options,
            )
            body = response.json()
            if response.status_code >= 400:
                raise ProcessingError(
                    f"dbt Cloud Metadata API returned {response.status_code}: "
                    f"{str(body)[:200]}"
                )
            return body
        except ProcessingError:
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Failed to query dbt Cloud Metadata API: {exc}"
            ) from exc

    def close(self):
        """No persistent connections are held; kept for connector symmetry."""


class DbtIngestor:
    """
    dbt ingestion handler.

    Parses dbt ``manifest.json`` / ``catalog.json`` artifacts into model,
    source and lineage records, and exports them as flat documents that
    plug directly into ``GraphBuilder``.

    Features:
        - Local artifact parsing (no network dependency)
        - Model / seed / snapshot summaries with merged column metadata
        - Source summaries from the manifest ``sources`` block
        - Model-to-model and source-to-model lineage edges
        - Optional dbt Cloud Metadata API access (Phase 2)
        - Progress tracking and error handling

    Example Usage:
        >>> ingestor = DbtIngestor(project_dir="/path/to/dbt/project")
        >>> data = ingestor.ingest_artifacts()
        >>> documents = ingestor.export_as_documents(data)
    """

    def __init__(
        self,
        project_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        metadata_url: Optional[str] = None,
        cloud_token: Optional[str] = None,
        allow_private_ips: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize dbt ingestor.

        Args:
            project_dir: Path to the dbt project root.
            target_dir: Directory containing the dbt artifacts.
            metadata_url: dbt Cloud Metadata API endpoint (Phase 2).
            cloud_token: dbt Cloud token (Phase 2).
            allow_private_ips: Allow private networks for the Cloud API.
            config: Optional configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(_MISSING_EXTRA_HINT)

        self.logger = get_logger("dbt_ingestor")
        self.config = config or {}
        self.config.update(kwargs)

        self.connector = DbtConnector(
            project_dir=project_dir,
            target_dir=target_dir,
            metadata_url=metadata_url,
            cloud_token=cloud_token,
            allow_private_ips=allow_private_ips,
            **self.config,
        )

        self.progress_tracker = get_progress_tracker()
        if not self.progress_tracker.enabled:
            self.progress_tracker.enabled = True

        self.logger.debug("dbt ingestor initialized")

    def ingest_artifacts(
        self,
        manifest_path: Optional[str] = None,
        catalog_path: Optional[str] = None,
        **options,
    ) -> DbtData:
        """
        Parse dbt artifacts into models, sources and lineage edges.

        ``manifest.json`` is required; ``catalog.json`` is optional and only
        enriches nodes with physical column types. When ``catalog_path`` is
        not given, the connector looks for ``catalog.json`` next to the
        manifest and silently skips it if absent.

        Args:
            manifest_path: Explicit path to ``manifest.json`` (optional;
                resolved from ``target_dir`` by default).
            catalog_path: Explicit path to ``catalog.json`` (optional).
            **options: Additional options (reserved).

        Returns:
            DbtData: Parsed models, sources and lineage edges.

        Raises:
            ProcessingError: If artifact parsing fails.
        """
        tracking_id = self.progress_tracker.start_tracking(
            file=str(self.connector.target_dir),
            module="ingest",
            submodule="DbtIngestor",
            message="Parsing dbt artifacts",
        )

        try:
            manifest_path = manifest_path or self.connector.get_manifest_path()
            manifest = self.connector.load_artifact(manifest_path)

            if catalog_path is None:
                try:
                    catalog_path = self.connector.get_catalog_path()
                except ValidationError:
                    catalog_path = None
            catalog = (
                self.connector.load_artifact(catalog_path) if catalog_path else None
            )

            nodes = manifest.get("nodes", {}) or {}
            source_block = manifest.get("sources", {}) or {}
            catalog_nodes = (catalog or {}).get("nodes", {}) or {}
            catalog_sources = (catalog or {}).get("sources", {}) or {}

            models: List[Dict[str, Any]] = []
            for unique_id, node in nodes.items():
                if node.get("resource_type") not in _REFABLE_RESOURCE_TYPES:
                    continue
                catalog_columns = (catalog_nodes.get(unique_id) or {}).get("columns")
                models.append(self._summarize_node(unique_id, node, catalog_columns))

            sources: List[Dict[str, Any]] = []
            for unique_id, source in source_block.items():
                catalog_columns = (catalog_sources.get(unique_id) or {}).get("columns")
                sources.append(
                    self._summarize_source(unique_id, source, catalog_columns)
                )

            lineage: List[Dict[str, Any]] = []
            refable_nodes = [
                (unique_id, node)
                for unique_id, node in nodes.items()
                if node.get("resource_type") in _REFABLE_RESOURCE_TYPES
            ]
            for unique_id, node in refable_nodes + list(source_block.items()):
                depends_on = (node.get("depends_on") or {}).get("nodes") or []
                for dependency in depends_on:
                    lineage.append({"source": dependency, "target": unique_id})

            data = DbtData(
                models=models,
                sources=sources,
                lineage=lineage,
                metadata={
                    "manifest_path": str(manifest_path),
                    "catalog_path": str(catalog_path) if catalog_path else None,
                    "model_count": len(models),
                    "source_count": len(sources),
                    "edge_count": len(lineage),
                },
            )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=(
                    f"Parsed {len(models)} node(s), {len(sources)} source(s), "
                    f"{len(lineage)} lineage edge(s)"
                ),
            )
            self.logger.info(
                "dbt artifact ingestion completed: %d node(s), %d source(s), "
                "%d lineage edge(s)",
                len(models),
                len(sources),
                len(lineage),
            )
            return data

        except (ValidationError, ProcessingError):
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message="dbt artifact parsing failed"
            )
            raise
        except Exception as exc:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(exc)
            )
            self.logger.error(f"Failed to ingest dbt artifacts: {exc}")
            raise ProcessingError(f"Failed to ingest dbt artifacts: {exc}") from exc

    def get_cloud_metadata(
        self,
        account_id: int,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """Query the dbt Cloud Metadata API (Phase 2). See
        :meth:`DbtConnector.get_cloud_metadata`."""
        return self.connector.get_cloud_metadata(
            account_id, query, variables=variables, **options
        )

    def export_as_documents(
        self,
        data: DbtData,
        include_code: bool = True,
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Convert parsed dbt data to document format for Semantica processing.

        One document is produced per node (model/seed/snapshot) and per
        source. Each document's metadata carries ``depends_on`` so downstream
        graph construction can wire model-to-model and source-to-model
        relationships without re-parsing the manifest.

        Args:
            data: DbtData object to convert.
            include_code: Include the model's raw SQL in the document text
                (default: True). Set to False for compact documents.
            **options: Additional options (reserved).

        Returns:
            list: List of document dictionaries.

        Example:
            >>> data = ingestor.ingest_artifacts()
            >>> documents = ingestor.export_as_documents(data, include_code=False)
        """
        documents: List[Dict[str, Any]] = []

        for node in data.models + data.sources:
            text_parts: List[str] = [
                f"{node.get('name')} ({node.get('resource_type')})"
            ]
            if node.get("description"):
                text_parts.append(node["description"])

            columns = node.get("columns") or {}
            if columns:
                column_lines = []
                for column_name in sorted(columns):
                    column = columns[column_name]
                    line = f"{column_name}: {column.get('type') or 'unknown'}"
                    if column.get("description"):
                        line += f" - {column['description']}"
                    column_lines.append(line)
                text_parts.append("Columns:\n" + "\n".join(column_lines))

            if include_code and node.get("code"):
                text_parts.append(node["code"])

            metadata: Dict[str, Any] = {
                "source": "dbt",
                "resource_type": node.get("resource_type"),
                "database": node.get("database"),
                "schema": node.get("schema"),
                "depends_on": node.get("depends_on") or [],
            }
            for key in ("path", "materialized", "tags", "source_name", "identifier"):
                if node.get(key) is not None:
                    metadata[key] = node[key]

            documents.append(
                {
                    "id": node["unique_id"],
                    "text": "\n\n".join(part for part in text_parts if part),
                    "metadata": metadata,
                }
            )

        self.logger.debug(f"Exported {len(documents)} documents")
        return documents

    def _summarize_node(
        self,
        unique_id: str,
        node: Dict[str, Any],
        catalog_columns: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a flat summary dict for one manifest node."""
        return {
            "unique_id": unique_id,
            "name": node.get("name"),
            "resource_type": node.get("resource_type"),
            "database": node.get("database"),
            "schema": node.get("schema"),
            "description": node.get("description") or "",
            "path": node.get("original_file_path") or node.get("path"),
            "materialized": (node.get("config") or {}).get("materialized"),
            "tags": node.get("tags") or [],
            "depends_on": list((node.get("depends_on") or {}).get("nodes") or []),
            "columns": self._merge_columns(node.get("columns"), catalog_columns),
            "code": node.get("raw_code") or node.get("compiled_code"),
        }

    def _summarize_source(
        self,
        unique_id: str,
        source: Dict[str, Any],
        catalog_columns: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a flat summary dict for one manifest source."""
        return {
            "unique_id": unique_id,
            "name": source.get("name"),
            "resource_type": "source",
            "source_name": source.get("source_name"),
            "database": source.get("database"),
            "schema": source.get("schema"),
            "identifier": source.get("identifier"),
            "loader": source.get("loader"),
            "description": source.get("description") or "",
            "tags": source.get("tags") or [],
            "depends_on": [],
            "columns": self._merge_columns(source.get("columns"), catalog_columns),
            "code": None,
        }

    @staticmethod
    def _merge_columns(
        manifest_columns: Optional[Dict[str, Any]],
        catalog_columns: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge manifest-declared columns with catalog physical metadata.

        The catalog (``dbt docs generate``) knows the physical column type;
        the manifest may additionally carry YAML descriptions. Types from the
        catalog win; descriptions fall back through manifest description to
        catalog comment.
        """
        merged: Dict[str, Dict[str, Any]] = {}

        for name, info in (manifest_columns or {}).items():
            merged[name] = {
                "name": name,
                "type": (info.get("type") or "").strip(),
                "description": info.get("description") or "",
            }

        for name, info in (catalog_columns or {}).items():
            entry = merged.setdefault(
                name, {"name": name, "type": "", "description": ""}
            )
            if info.get("type"):
                entry["type"] = info["type"]
            if info.get("comment") and not entry["description"]:
                entry["description"] = info["comment"]

        return merged

    def close(self):
        """Close the underlying connector (no-op, kept for symmetry)."""
        self.connector.close()
