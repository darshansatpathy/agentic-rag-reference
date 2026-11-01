"""Agentic RAG reference: hybrid retrieval, permission-aware, cited and able to say "I don't know"."""

from agentic_rag.models import Answer, Chunk, Citation, Document, Hit
from agentic_rag.pipeline import PipelineConfig, RagPipeline
from agentic_rag.retrieval import Mode, Principal

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "Hit",
    "Mode",
    "PipelineConfig",
    "Principal",
    "RagPipeline",
]
__version__ = "0.1.0"
