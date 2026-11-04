"""Visual place retrieval: FAISS index, temporal exclusion and candidate detection."""

from latent_loop.retrieval.detector import (
    RetrievalConfig,
    RetrievalResult,
    assert_causal,
    filter_candidates,
    retrieve,
)
from latent_loop.retrieval.index import FlatIPIndex
from latent_loop.retrieval.temporal_filter import TemporalExclusion

__all__ = [
    "FlatIPIndex",
    "RetrievalConfig",
    "RetrievalResult",
    "TemporalExclusion",
    "assert_causal",
    "filter_candidates",
    "retrieve",
]
