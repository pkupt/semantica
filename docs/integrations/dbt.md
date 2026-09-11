---
title: "dbt Integration"
description: "Ingest dbt manifest and catalog artifacts — model lineage, sources, and column metadata — into Semantica's KG pipeline."
icon: "diagram-project"
---

> Turn dbt's `manifest.json` / `catalog.json` build artifacts into knowledge-graph documents. Every `dbt run` already writes a full model-to-model lineage graph — Semantica reads it straight from disk, no warehouse connection needed.


## Installation

```bash
# Install with dbt support
pip install "semantica[ingest-dbt]"

# Or install the dependency separately
pip install requests>=2.28.0
```

The connector parses local artifacts, which needs nothing beyond `requests` (used for the optional dbt Cloud path). Your core `semantica` install never pulls it in automatically.


## Basic Usage

```python
from semantica.ingest import DbtIngestor

ingestor = DbtIngestor(project_dir="/path/to/dbt/project")

# Reads <project_dir>/target/manifest.json and catalog.json (if present)
data = ingestor.ingest_artifacts()

print(data.metadata)
# {'manifest_path': '.../target/manifest.json', 'catalog_path': '.../target/catalog.json',
#  'model_count': 42, 'source_count': 7, 'edge_count': 55}

documents = ingestor.export_as_documents(data)
print(documents[0]["id"])        # e.g. "model.jaffle_shop.stg_orders"
print(documents[0]["metadata"])  # source='dbt', resource_type, database, schema, depends_on, ...
```

<Tip>
Run `dbt docs generate` first if you want physical column types from `catalog.json` merged into each node. Without it, the connector still works — you just get YAML-declared columns only.
</Tip>


## What You Get

| Output | Contents |
|---|---|
| `data.models` | One summary per model / seed / snapshot: name, database, schema, materialization, tags, columns (with types from the catalog), raw SQL |
| `data.sources` | One summary per dbt source: source name, database, schema, identifier, columns |
| `data.lineage` | One edge per dependency: `{"source": "<upstream unique_id>", "target": "<downstream unique_id>"}` — covers both model-to-model and source-to-model links |
| `documents` | Flat `{"id", "text", "metadata"}` dicts, one per node, with `depends_on` in metadata so `GraphBuilder` can wire relationships directly |


## Feeding GraphBuilder

```python
from semantica.ingest import DbtIngestor
from semantica.graph_builder import GraphBuilder

ingestor = DbtIngestor(project_dir="./my_dbt_project")
data = ingestor.ingest_artifacts()
documents = ingestor.export_as_documents(data, include_code=False)  # compact docs

builder = GraphBuilder()
builder.build_from_documents(documents)
```

Each document's `metadata.depends_on` lists the upstream node IDs, and the document `id` is the dbt `unique_id` — so model lineage becomes graph relationships out of the box.


## Explicit Artifact Paths

If your artifacts live outside the default `target/` directory (CI workspaces, copied artifacts):

```python
ingestor = DbtIngestor(target_dir="/ci/artifacts/current")
data = ingestor.ingest_artifacts(
    manifest_path="/ci/artifacts/current/manifest.json",
    catalog_path="/ci/artifacts/current/catalog.json",
)
```


## dbt Cloud Metadata API (Phase 2)

For hosted projects, the connector can also query the dbt Cloud Metadata API. Requests go through Semantica's SSRF guard, so user-supplied endpoints cannot reach private networks unless you explicitly allow it:

```python
ingestor = DbtIngestor(
    metadata_url="https://metadata.cloud.getdbt.com/graphql",
    cloud_token=os.getenv("DBT_CLOUD_TOKEN"),
)

response = ingestor.get_cloud_metadata(
    account_id=12345,
    query="""
    query($accountId: Int!) {
      models(accountId: $accountId) { name uniqueId }
    }
    """,
    variables={"accountId": 12345},
)
```

The local artifact path above remains the simplest option for most setups: it is offline, exact, and refreshed every time your dbt jobs run.
