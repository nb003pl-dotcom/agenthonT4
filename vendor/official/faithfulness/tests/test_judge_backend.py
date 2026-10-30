"""Tests for the served-judge backend switch (stdlib mock server, no weights)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from faithfulness.judge import (
    ENV_JUDGE_BACKEND,
    ENV_JUDGE_URL,
    ENV_JUDGE_TOKEN,
    NLI_MODEL_IDS,
    DeBERTaNLIJudge,
    EnsembleNLIJudge,
    ServedNLIJudge,
    build_judge,
)


class _MockJudgeServer:
    """In-process /entail endpoint with scriptable responses per model_id."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.scores: dict[str, float] = {}
        self.fail_next: int = 0  # number of requests to 500 before succeeding
        self.raw_body: bytes | None = None  # overrides JSON response if set

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                outer.headers.append(dict(self.headers))
                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                if outer.raw_body is not None:
                    body = outer.raw_body
                else:
                    score = outer.scores.get(payload["model_id"], 0.5)
                    body = json.dumps({"entailment": score}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:  # silence test output
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def server() -> Iterator[_MockJudgeServer]:
    mock = _MockJudgeServer()
    yield mock
    mock.close()


def make_judge(server: _MockJudgeServer, **kwargs: Any) -> ServedNLIJudge:
    defaults: dict[str, Any] = dict(model_id="model-a", url=server.url, max_retries=3)
    defaults.update(kwargs)
    return ServedNLIJudge(**defaults)


def test_served_judge_posts_payload_and_parses_score(server: _MockJudgeServer) -> None:
    server.scores["model-a"] = 0.9731
    judge = make_judge(server, token="sekrit")
    score = judge.entail("the premise text", "the hypothesis text")
    assert score == 0.9731
    [request] = server.requests
    assert request == {
        "model_id": "model-a",
        "premise": "the premise text",
        "hypothesis": "the hypothesis text",
    }
    assert server.headers[0].get("Authorization") == "Bearer sekrit"


def test_no_auth_header_without_token(server: _MockJudgeServer) -> None:
    make_judge(server).entail("p", "h")
    assert "Authorization" not in server.headers[0]


def test_empty_inputs_short_circuit_client_side(server: _MockJudgeServer) -> None:
    judge = make_judge(server)
    assert judge.entail("", "hypothesis") == 0.0
    assert judge.entail("premise", "   ") == 0.0
    assert server.requests == []  # never hit the network


def test_retries_then_succeeds(server: _MockJudgeServer) -> None:
    server.fail_next = 2
    server.scores["model-a"] = 0.25
    assert make_judge(server).entail("p", "h") == 0.25
    assert len(server.requests) == 3


def test_raises_after_exhausted_retries(server: _MockJudgeServer) -> None:
    server.fail_next = 99
    with pytest.raises(RuntimeError, match="all 3 attempts"):
        make_judge(server).entail("p", "h")


def test_out_of_range_score_rejected(server: _MockJudgeServer) -> None:
    server.raw_body = json.dumps({"entailment": 1.7}).encode()
    with pytest.raises(RuntimeError):
        make_judge(server).entail("p", "h")


def test_ensemble_of_served_judges_averages_client_side(
    server: _MockJudgeServer,
) -> None:
    server.scores = {"model-a": 0.9, "model-b": 0.5}
    ensemble = EnsembleNLIJudge(
        judges=[make_judge(server), make_judge(server, model_id="model-b")]
    )
    assert ensemble.entail("p", "h") == pytest.approx(0.7)


def test_build_judge_default_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_JUDGE_BACKEND, raising=False)
    ensemble = build_judge()
    members = ensemble._judges
    assert len(members) == len(NLI_MODEL_IDS)
    assert all(isinstance(j, DeBERTaNLIJudge) for j in members)


def test_build_judge_served(
    monkeypatch: pytest.MonkeyPatch, server: _MockJudgeServer
) -> None:
    monkeypatch.setenv(ENV_JUDGE_BACKEND, "served")
    monkeypatch.setenv(ENV_JUDGE_URL, server.url + "/")  # trailing slash stripped
    monkeypatch.setenv(ENV_JUDGE_TOKEN, "tok")
    ensemble = build_judge()
    # The list comprehension IS the type assertion: anything that is not a ServedNLIJudge is
    # dropped, and the length check catches it -- and mypy can then see model_id/url/token,
    # which the NLIJudge protocol (entail-only) deliberately does not carry.
    members = [j for j in ensemble._judges if isinstance(j, ServedNLIJudge)]
    assert len(members) == len(ensemble._judges)
    assert [j.model_id for j in members] == NLI_MODEL_IDS
    assert all(j.url == server.url and j.token == "tok" for j in members)


def test_build_judge_served_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_JUDGE_BACKEND, "served")
    monkeypatch.delenv(ENV_JUDGE_URL, raising=False)
    with pytest.raises(RuntimeError, match="T4_JUDGE_URL"):
        build_judge()


def test_build_judge_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_JUDGE_BACKEND, "quantum")
    with pytest.raises(RuntimeError, match="not a valid backend"):
        build_judge()
