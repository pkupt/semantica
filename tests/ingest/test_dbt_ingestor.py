"""Tests for the dbt artifact ingestor.

Phase 1 of the dbt connector is purely local: it parses ``manifest.json`` /
``catalog.json`` files from disk, so most tests drive it with fixture
artifacts written to ``tmp_path`` — no SDK mocking required for the local
path. The dbt Cloud Metadata API path (Phase 2) patches the single SSRF
entry point, mirroring ``tests/ingest/test_sap_ingestor.py``.

Covered:
- manifest parsing: models / seeds / snapshots kept, tests and analyses skipped
- source summaries from the ``sources`` block
- lineage edges for model-to-model and source-to-model dependencies
- catalog merge: physical types win, descriptions fall back to comments
- catalog.json is optional (absent file is silently skipped)
- ``export_as_documents`` flat document shape for ``GraphBuilder``
- lazy import: ``semantica.ingest`` does not import the connector eagerly,
  and a missing ``requests`` yields the ``semantica[ingest-dbt]`` hint
- Cloud API goes through the SSRF guard with token auth
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

import semantica.ingest as ingest_pkg
from semantica.ingest import DbtConnector, DbtData, DbtIngestor
from semantica.utils.exceptions import ProcessingError, ValidationError

MANIFEST = {
    "nodes": {
        "model.shop.stg_orders": {
            "name": "stg_orders",
            "resource_type": "model",
            "database": "analytics",
            "schema": "staging",
            "description": "Staged orders.",
            "original_file_path": "models/staging/stg_orders.sql",
            "config": {"materialized": "view"},
            "tags": ["staging"],
            "depends_on": {
                "nodes": [
                    "source.shop.raw_orders",
                    "model.shop.stg_customers",
                ]
            },
            "columns": {
                "order_id": {"name": "order_id", "description": "Primary key."},
                "status": {"name": "status", "description": ""},
            },
            "raw_code": "select id as order_id, status from {{ source('shop', 'raw_orders') }}",
        },
        "model.shop.stg_customers": {
            "name": "stg_customers",
            "resource_type": "model",
            "database": "analytics",
            "schema": "staging",
            "description": "",
            "original_file_path": "models/staging/stg_customers.sql",
            "config": {"materialized": "view"},
            "tags": [],
            "depends_on": {"nodes": []},
            "columns": {},
            "raw_code": "select 1 as customer_id",
        },
        "seed.shop.country_codes": {
            "name": "country_codes",
            "resource_type": "seed",
            "database": "analytics",
            "schema": "seeds",
            "description": "",
            "original_file_path": "seeds/country_codes.csv",
            "config": {"materialized": "table"},
            "tags": [],
            "depends_on": {"nodes": []},
            "columns": {},
            "raw_code": None,
        },
        "snapshot.shop.orders_snapshot": {
            "name": "orders_snapshot",
            "resource_type": "snapshot",
            "database": "analytics",
            "schema": "snapshots",
            "description": "",
            "original_file_path": "snapshots/orders.sql",
            "config": {"materialized": "snapshot"},
            "tags": [],
            "depends_on": {"nodes": ["model.shop.stg_orders"]},
            "columns": {},
            "raw_code": "select * from {{ ref('stg_orders') }}",
        },
        "test.shop.not_null_order_id": {
            "name": "not_null_order_id",
            "resource_type": "test",
            "depends_on": {"nodes": ["model.shop.stg_orders"]},
        },
        "analysis.shop.churn": {
            "name": "churn",
            "resource_type": "analysis",
            "depends_on": {"nodes": ["model.shop.stg_orders"]},
        },
    },
    "sources": {
        "source.shop.raw_orders": {
            "name": "raw_orders",
            "resource_type": "source",
            "source_name": "shop",
            "database": "raw",
            "schema": "public",
            "identifier": "raw_orders",
            "loader": "airbyte",
            "description": "Landing table for orders.",
            "tags": ["raw"],
            "depends_on": {"nodes": []},
            "columns": {"id": {"name": "id", "description": "Source order id."}},
        }
    },
}

CATALOG = {
    "nodes": {
        "model.shop.stg_orders": {
            "unique_id": "model.shop.stg_orders",
            "columns": {
                "order_id": {"name": "order_id", "type": "integer", "comment": ""},
                "status": {
                    "name": "status",
                    "type": "text",
                    "comment": "Order status.",
                },
                "amount": {"name": "amount", "type": "numeric", "comment": ""},
            },
        }
    },
    "sources": {
        "source.shop.raw_orders": {
            "unique_id": "source.shop.raw_orders",
            "columns": {"id": {"name": "id", "type": "bigint", "comment": ""}},
        }
    },
}


@pytest.fixture
def dbt_project(tmp_path):
    """A fake dbt project whose target/ holds manifest.json and catalog.json."""
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (target / "catalog.json").write_text(json.dumps(CATALOG), encoding="utf-8")
    return tmp_path


@pytest.fixture
def ingestor(dbt_project):
    return DbtIngestor(project_dir=str(dbt_project))


class TestLazyImport:
    def test_semantica_ingest_does_not_import_dbt_eagerly(self):
        # A fresh interpreter must not load the module on plain package import.
        code = (
            "import sys, semantica.ingest;"
            "assert 'semantica.ingest.dbt_ingestor' not in sys.modules;"
            "print('ok')"
        )
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_lazy_access_uses_friendly_message_without_requests(self, monkeypatch):
        import semantica.ingest.dbt_ingestor as dbt_mod

        monkeypatch.setattr(dbt_mod, "REQUESTS_AVAILABLE", False)
        monkeypatch.delattr(ingest_pkg, "DbtIngestor", raising=False)
        monkeypatch.delattr(ingest_pkg, "DbtConnector", raising=False)
        with pytest.raises(ImportError, match=r"semantica\[ingest-dbt\]"):
            ingest_pkg.DbtConnector


class TestArtifactParsing:
    def test_models_seeds_snapshots_kept_tests_skipped(self, ingestor):
        data = ingestor.ingest_artifacts()
        model_ids = {m["unique_id"] for m in data.models}
        assert model_ids == {
            "model.shop.stg_orders",
            "model.shop.stg_customers",
            "seed.shop.country_codes",
            "snapshot.shop.orders_snapshot",
        }

    def test_sources_parsed_with_identifiers(self, ingestor):
        data = ingestor.ingest_artifacts()
        assert len(data.sources) == 1
        source = data.sources[0]
        assert source["unique_id"] == "source.shop.raw_orders"
        assert source["source_name"] == "shop"
        assert source["identifier"] == "raw_orders"
        assert source["loader"] == "airbyte"

    def test_lineage_edges_model_and_source(self, ingestor):
        data = ingestor.ingest_artifacts()
        edges = {(e["source"], e["target"]) for e in data.lineage}
        assert ("source.shop.raw_orders", "model.shop.stg_orders") in edges
        assert ("model.shop.stg_customers", "model.shop.stg_orders") in edges
        assert ("model.shop.stg_orders", "snapshot.shop.orders_snapshot") in edges
        # tests / analyses are skipped entirely, so their edges must not appear
        assert all("test." not in e["target"] for e in data.lineage)
        assert all("analysis." not in e["target"] for e in data.lineage)

    def test_catalog_types_merged_and_descriptions_fall_back(self, ingestor):
        data = ingestor.ingest_artifacts()
        orders = next(
            m for m in data.models if m["unique_id"] == "model.shop.stg_orders"
        )
        columns = orders["columns"]
        # physical type from catalog wins
        assert columns["order_id"]["type"] == "integer"
        # description kept from manifest
        assert columns["order_id"]["description"] == "Primary key."
        # catalog-only column appears
        assert columns["amount"]["type"] == "numeric"
        # catalog comment used when manifest description empty
        assert columns["status"]["description"] == "Order status."

    def test_catalog_is_optional(self, dbt_project):
        (dbt_project / "target" / "catalog.json").unlink()
        ingestor = DbtIngestor(project_dir=str(dbt_project))
        data = ingestor.ingest_artifacts()
        assert data.metadata["catalog_path"] is None
        orders = next(
            m for m in data.models if m["unique_id"] == "model.shop.stg_orders"
        )
        assert orders["columns"]["order_id"]["type"] == ""

    def test_explicit_paths_override_target_dir(self, dbt_project, tmp_path_factory):
        other = tmp_path_factory.mktemp("artifacts")
        (other / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        ingestor = DbtIngestor(project_dir=str(dbt_project))
        data = ingestor.ingest_artifacts(manifest_path=str(other / "manifest.json"))
        assert data.metadata["model_count"] == 4

    def test_missing_manifest_raises_validation_error(self, tmp_path):
        ingestor = DbtIngestor(project_dir=str(tmp_path))
        with pytest.raises(ValidationError, match="manifest.json"):
            ingestor.ingest_artifacts()

    def test_invalid_manifest_json_raises_processing_error(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "manifest.json").write_text("{not json", encoding="utf-8")
        ingestor = DbtIngestor(project_dir=str(tmp_path))
        with pytest.raises(ProcessingError, match="Invalid JSON"):
            ingestor.ingest_artifacts()

    def test_dataclass_counts_in_metadata(self, ingestor):
        data = ingestor.ingest_artifacts()
        assert isinstance(data, DbtData)
        assert data.metadata["model_count"] == 4
        assert data.metadata["source_count"] == 1
        assert data.metadata["edge_count"] == len(data.lineage)


class TestExportAsDocuments:
    def test_document_shape(self, ingestor):
        data = ingestor.ingest_artifacts()
        documents = ingestor.export_as_documents(data)
        ids = {d["id"] for d in documents}
        assert "model.shop.stg_orders" in ids
        assert "source.shop.raw_orders" in ids
        assert len(documents) == len(data.models) + len(data.sources)

    def test_document_metadata_carries_lineage(self, ingestor):
        data = ingestor.ingest_artifacts()
        documents = {d["id"]: d for d in ingestor.export_as_documents(data)}
        orders = documents["model.shop.stg_orders"]
        assert orders["metadata"]["source"] == "dbt"
        assert orders["metadata"]["resource_type"] == "model"
        assert orders["metadata"]["database"] == "analytics"
        assert orders["metadata"]["schema"] == "staging"
        assert set(orders["metadata"]["depends_on"]) == {
            "source.shop.raw_orders",
            "model.shop.stg_customers",
        }

    def test_document_text_sections(self, ingestor):
        data = ingestor.ingest_artifacts()
        documents = {d["id"]: d for d in ingestor.export_as_documents(data)}
        orders = documents["model.shop.stg_orders"]
        assert "stg_orders (model)" in orders["text"]
        assert "Staged orders." in orders["text"]
        assert "order_id: integer - Primary key." in orders["text"]
        assert "status: text - Order status." in orders["text"]
        assert "select id as order_id" in orders["text"]

    def test_include_code_false_drops_sql(self, ingestor):
        data = ingestor.ingest_artifacts()
        documents = {
            d["id"]: d for d in ingestor.export_as_documents(data, include_code=False)
        }
        assert "select id as order_id" not in documents["model.shop.stg_orders"]["text"]


class TestConnectorPaths:
    def test_missing_artifact_raises_with_hint(self, dbt_project):
        connector = DbtConnector(project_dir=str(dbt_project))
        (dbt_project / "target" / "catalog.json").unlink()
        with pytest.raises(ValidationError, match="dbt docs generate"):
            connector.get_catalog_path()

    def test_load_artifact_roundtrip(self, dbt_project):
        connector = DbtConnector(project_dir=str(dbt_project))
        manifest = connector.load_artifact(connector.get_manifest_path())
        assert manifest["nodes"]["model.shop.stg_orders"]["name"] == "stg_orders"

    def test_custom_target_dir(self, dbt_project, tmp_path_factory):
        other = tmp_path_factory.mktemp("targets")
        (other / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
        connector = DbtConnector(project_dir=str(dbt_project), target_dir=str(other))
        assert connector.get_manifest_path() == other / "manifest.json"


class TestCloudMetadataApi:
    def test_requires_metadata_url(self, ingestor):
        with pytest.raises(ValidationError, match="endpoint is required"):
            ingestor.get_cloud_metadata(account_id=1, query="{ models { name } }")

    def test_requires_token(self, dbt_project):
        ingestor = DbtIngestor(
            project_dir=str(dbt_project),
            metadata_url="https://metadata.example.com/graphql",
        )
        with pytest.raises(ValidationError, match="token is required"):
            ingestor.get_cloud_metadata(account_id=1, query="{ models { name } }")

    @patch("semantica.ingest.dbt_ingestor.request_with_ssrf_guard")
    def test_query_goes_through_ssrf_guard(self, mock_request, dbt_project):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": {"models": [{"name": "stg_orders"}]}}
        mock_request.return_value = response

        ingestor = DbtIngestor(
            project_dir=str(dbt_project),
            metadata_url="https://metadata.example.com/graphql",
            cloud_token="secret-token",
        )
        result = ingestor.get_cloud_metadata(
            account_id=42,
            query="query($accountId: Int!) { models(accountId: $accountId) { name } }",
            variables={"accountId": 42},
        )

        assert result["data"]["models"][0]["name"] == "stg_orders"
        args, kwargs = mock_request.call_args
        assert args[0] == "https://metadata.example.com/graphql"
        assert kwargs["method"] == "POST"
        assert kwargs["headers"]["Authorization"] == "Token secret-token"
        assert kwargs["json"]["variables"] == {"accountId": 42}

    @patch("semantica.ingest.dbt_ingestor.request_with_ssrf_guard")
    def test_http_error_raises_processing_error(self, mock_request, dbt_project):
        response = MagicMock()
        response.status_code = 403
        response.json.return_value = {"errors": [{"message": "forbidden"}]}
        mock_request.return_value = response

        ingestor = DbtIngestor(
            project_dir=str(dbt_project),
            metadata_url="https://metadata.example.com/graphql",
            cloud_token="secret-token",
        )
        with pytest.raises(ProcessingError, match="403"):
            ingestor.get_cloud_metadata(account_id=1, query="{ models { name } }")
