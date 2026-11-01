"""Structured logs, request IDs and optional OpenTelemetry spans for each pipeline stage."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("arag_request_id", default=None)
query_log = logging.getLogger("arag.query")

try:
    from opentelemetry import trace as _otel_trace

    _tracer: Any = _otel_trace.get_tracer("agentic-rag-reference")
except ImportError:  # pragma: no cover
    _tracer = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid:
            payload["request_id"] = rid
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    for name in ("arag", "agentic_rag"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level.upper())
        logger.propagate = False


def log_query(fields: dict[str, Any]) -> None:
    """Log one answered question. Question text is not logged by default: it can hold personal data."""
    query_log.info("query", extra={"fields": fields, "request_id": request_id_var.get()})


@contextlib.contextmanager
def span(name: str, attributes: dict[str, Any]) -> Iterator[Any]:
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for k, v in attributes.items():
            if v is not None:
                current.set_attribute(k, v)
        yield current
