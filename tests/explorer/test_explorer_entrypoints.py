import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from semantica.context.context_graph import ContextGraph  # noqa: E402
from semantica.explorer.runtime import install_mutation_bridge  # noqa: E402
from semantica.explorer.session import GraphSession  # noqa: E402


def test_mutation_bridge_supports_multiple_apps_for_one_graph(monkeypatch):
    graph = ContextGraph(advanced_analytics=False)
    first_session = GraphSession(graph)
    second_session = GraphSession(graph)
    received = []
    graph.mutation_callback = lambda *event: received.append(("original", event))

    monkeypatch.setattr(
        first_session,
        "handle_graph_mutation",
        lambda *event: received.append(("first", event)),
    )
    monkeypatch.setattr(
        second_session,
        "handle_graph_mutation",
        lambda *event: received.append(("second", event)),
    )

    first_app = FastAPI()
    first_app.state.event_loop = None
    first_app.state.ws_manager = None
    second_app = FastAPI()
    second_app.state.event_loop = None
    second_app.state.ws_manager = None

    install_mutation_bridge(first_app, first_session)
    install_mutation_bridge(second_app, second_session)
    graph.mutation_callback("UPDATE_NODE", "node-1", {"content": "Updated"})

    assert [receiver for receiver, _ in received] == ["second", "first", "original"]


def test_mutation_bridge_is_idempotent_for_one_app(monkeypatch):
    graph = ContextGraph(advanced_analytics=False)
    session = GraphSession(graph)
    received = []
    monkeypatch.setattr(
        session,
        "handle_graph_mutation",
        lambda *event: received.append(event),
    )
    app = FastAPI()
    app.state.event_loop = None
    app.state.ws_manager = None

    install_mutation_bridge(app, session)
    install_mutation_bridge(app, session)
    graph.mutation_callback("UPDATE_NODE", "node-1", {"content": "Updated"})

    assert len(received) == 1


def test_mutation_bridge_reinstalls_for_new_session_on_same_app(monkeypatch):
    first_session = GraphSession(ContextGraph(advanced_analytics=False))
    second_session = GraphSession(ContextGraph(advanced_analytics=False))
    received = []
    monkeypatch.setattr(
        first_session,
        "handle_graph_mutation",
        lambda *event: received.append(("first", event)),
    )
    monkeypatch.setattr(
        second_session,
        "handle_graph_mutation",
        lambda *event: received.append(("second", event)),
    )
    app = FastAPI()
    app.state.event_loop = None
    app.state.ws_manager = None

    install_mutation_bridge(app, first_session)
    install_mutation_bridge(app, second_session)
    second_session.graph.mutation_callback(
        "UPDATE_NODE",
        "node-2",
        {"content": "Updated"},
    )

    assert [receiver for receiver, _ in received] == ["second"]


def test_legacy_server_mounts_editable_markdown_routes(monkeypatch):
    monkeypatch.setenv("SEMANTICA_ALLOW_ANONYMOUS", "true")

    from semantica import server

    # fastapi>=0.141 nests included routers in app.routes as _IncludedRouter
    # entries whose own routes only appear via original_router — expand them
    # so mounted paths stay visible on every fastapi version.
    def _iter_paths(routes):
        for route in routes:
            nested = getattr(route, "original_router", None)
            if nested is not None:
                yield from _iter_paths(nested.routes)
            elif hasattr(route, "path"):
                yield route.path

    paths = set(_iter_paths(server.app.routes))
    assert "/api/markdown/{kind}/{resource_id:path}" in paths
    assert "/api/memories" in paths
    assert "/ws/graph-updates" in paths

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/graph-updates") as websocket:
            acknowledgement = websocket.receive_json()
            assert acknowledgement["event"] == "connection_ack"
            assert acknowledgement["data"] == {"connected": True}
            assert acknowledgement["timestamp"]
        info = client.get("/api/info")
        memories = client.get("/api/memories")
        server.app.state.session.graph.add_node(
            "server-node",
            "Note",
            "Original server content",
        )
        current = client.get("/api/markdown/context-node/server-node").json()
        saved = client.put(
            "/api/markdown/context-node/server-node",
            json={
                "markdown": current["source"].replace(
                    "Original server content",
                    "Updated server content",
                ),
                "expected_revision": current["revision"],
            },
        )

    assert info.json()["capabilities"]["agent_memory"] is False
    assert memories.status_code == 503
    assert memories.json()["detail"] == (
        "AgentMemory is not configured for this Explorer instance."
    )
    assert saved.status_code == 200
    assert saved.json()["body"] == "Updated server content"


def test_cli_main_configures_allowed_origins_for_custom_port(tmp_path, monkeypatch):
    """Regression test for #1257: semantica-explorer --port 8020 includes port in allowed_origins."""
    monkeypatch.setenv("SEMANTICA_ALLOW_ANONYMOUS", "true")
    monkeypatch.delenv("SEMANTICA_API_KEY", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("EXPLORER_CORS_ORIGINS", raising=False)

    graph_file = tmp_path / "test_graph.json"
    graph_file.write_text('{"nodes": [], "edges": []}', encoding="utf-8")

    from semantica.explorer import main
    import uvicorn

    captured_app = None
    captured_kwargs = {}

    def mock_run(app, **kwargs):
        nonlocal captured_app, captured_kwargs
        captured_app = app
        captured_kwargs = kwargs

    monkeypatch.setattr(uvicorn, "run", mock_run)

    main(["--graph", str(graph_file), "--port", "8020", "--no-browser"])

    assert captured_kwargs["port"] == 8020
    assert "http://127.0.0.1:8020" in captured_app.state.explorer_settings["allowed_origins"]
    assert "http://localhost:8020" in captured_app.state.explorer_settings["allowed_origins"]

    with TestClient(captured_app) as client:
        with client.websocket_connect(
            "/ws/graph-updates", headers={"Origin": "http://127.0.0.1:8020"}
        ) as ws:
            ack = ws.receive_json()
        assert ack["event"] == "connection_ack"

        # Hostile origin still rejected
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                "/ws/graph-updates", headers={"Origin": "https://evil.example"}
            ):
                pass
        assert excinfo.value.code == 4403


def test_cli_main_preserves_explicit_allowed_origins(tmp_path, monkeypatch):
    """Ensure explicit ALLOWED_ORIGINS is not overridden when --port is specified."""
    monkeypatch.setenv("SEMANTICA_ALLOW_ANONYMOUS", "true")
    monkeypatch.delenv("SEMANTICA_API_KEY", raising=False)
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://custom.example.com")
    monkeypatch.delenv("EXPLORER_CORS_ORIGINS", raising=False)

    graph_file = tmp_path / "test_graph.json"
    graph_file.write_text('{"nodes": [], "edges": []}', encoding="utf-8")

    from semantica.explorer import main
    import uvicorn

    captured_app = None

    def mock_run(app, **kwargs):
        nonlocal captured_app
        captured_app = app

    monkeypatch.setattr(uvicorn, "run", mock_run)

    main(["--graph", str(graph_file), "--port", "8020", "--no-browser"])

    assert captured_app.state.explorer_settings["allowed_origins"] == [
        "https://custom.example.com"
    ]
