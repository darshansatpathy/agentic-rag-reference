"""HTTP API.

Identity: this service sits behind an authenticating gateway (API gateway, service
mesh or BFF) that verifies the user's token (OIDC) and forwards `X-User-Id` and
`X-User-Groups`. When `ARAG_SERVICE_TOKEN` is set, requests must also carry
`Authorization: Bearer <token>`, so only that gateway can assert identities. Without it
the service runs in development mode and trusts the headers (logged at start-up).
"""

from __future__ import annotations

import hmac
import logging
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from agentic_rag import __version__
from agentic_rag.config import Settings, build_pipeline
from agentic_rag.observability import configure_logging, request_id_var
from agentic_rag.pipeline import RagPipeline
from agentic_rag.retrieval import Principal

log = logging.getLogger("arag.api")
_GROUP = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    include_trace: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)


class Metrics:
    """Prometheus text-format counters and a latency histogram, without a client library."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: Counter[tuple[str, int]] = Counter()
        self.outcomes: Counter[str] = Counter()
        self.latency: dict[str, list[int]] = {}
        self.latency_sum: dict[str, float] = {}

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(route, status)] += 1
            buckets = self.latency.setdefault(route, [0] * (len(_BUCKETS) + 1))
            for i, edge in enumerate(_BUCKETS):
                if seconds <= edge:
                    buckets[i] += 1
            buckets[-1] += 1
            self.latency_sum[route] = self.latency_sum.get(route, 0.0) + seconds

    def outcome(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.outcomes[name] += n

    def render(self) -> str:
        lines = ["# TYPE arag_http_requests_total counter"]
        with self._lock:
            for (route, status), n in sorted(self.requests.items()):
                lines.append(f'arag_http_requests_total{{route="{route}",status="{status}"}} {n}')
            lines.append("# TYPE arag_answers_total counter")
            for name, n in sorted(self.outcomes.items()):
                lines.append(f'arag_answers_total{{outcome="{name}"}} {n}')
            lines.append("# TYPE arag_request_seconds histogram")
            for route, buckets in sorted(self.latency.items()):
                for edge, n in zip(_BUCKETS, buckets, strict=False):
                    lines.append(f'arag_request_seconds_bucket{{route="{route}",le="{edge}"}} {n}')
                lines.append(f'arag_request_seconds_bucket{{route="{route}",le="+Inf"}} {buckets[-1]}')
                lines.append(f'arag_request_seconds_sum{{route="{route}"}} {self.latency_sum[route]:.6f}')
                lines.append(f'arag_request_seconds_count{{route="{route}"}} {buckets[-1]}')
        return "\n".join(lines) + "\n"


def create_app(pipeline: RagPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    rag = pipeline or build_pipeline(settings)
    metrics = Metrics()
    if not settings.service_token:
        log.warning("ARAG_SERVICE_TOKEN is not set: identity headers are trusted (development mode)")

    app = FastAPI(title="agentic-rag-reference", version=__version__)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", rid):
            rid = uuid.uuid4().hex
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.observe(path, response.status_code, time.perf_counter() - started)
        response.headers["X-Request-ID"] = rid
        return response

    def principal(
        authorization: str | None = Header(default=None),
        x_user_id: str | None = Header(default=None),
        x_user_groups: str | None = Header(default=None),
    ) -> Principal:
        if settings.service_token:
            presented = (authorization or "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(presented.encode(), settings.service_token.encode()):
                raise HTTPException(status_code=401, detail="missing or invalid service token")
        groups = [g.strip().lower() for g in (x_user_groups or "").split(",") if g.strip()]
        bad = [g for g in groups if not _GROUP.match(g)]
        if bad:
            raise HTTPException(status_code=400, detail=f"invalid group name(s): {', '.join(bad[:3])}")
        return Principal.of(x_user_id or "anonymous", groups)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "chunks": len(rag.chunks)}

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/ask")
    def ask(body: AskRequest, who: Principal = Depends(principal)) -> dict[str, Any]:  # noqa: B008
        answer = rag.ask(body.question, who)
        metrics.outcome("abstained" if answer.abstained else "answered")
        quarantined = sum(len(t.get("chunks", [])) for t in answer.trace if t.get("step") == "quarantine")
        if quarantined:
            metrics.outcome("quarantined_chunks", quarantined)
        data = answer.to_dict()
        if not body.include_trace:
            data.pop("trace", None)
        data["request_id"] = request_id_var.get()
        return data

    @app.post("/v1/search")
    def search(body: SearchRequest, who: Principal = Depends(principal)) -> dict[str, Any]:  # noqa: B008
        hits = rag.search(body.query, who, body.k)
        return {
            "hits": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "doc_id": h.chunk.doc_id,
                    "title": h.chunk.title,
                    "section": h.chunk.section,
                    "score": round(h.score, 6),
                    "ranks": h.ranks,
                    "text": h.chunk.text,
                }
                for h in hits[: body.k]
            ],
            "request_id": request_id_var.get(),
        }

    return app
