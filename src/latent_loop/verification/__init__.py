"""Phase 4: geometric verification of loop-closure candidates (ORB + essential matrix)."""

from latent_loop.verification.geometry import (
    REASONS,
    EssentialEstimate,
    VerificationConfig,
    VerificationResult,
    estimate_essential,
    ransac_essential_numpy,
    verify_features,
    verify_pair,
)
from latent_loop.verification.orb import (
    MatchConfig,
    Matches,
    OrbConfig,
    OrbFeatures,
    extract_orb,
    hamming_distance,
    match_features,
    raw_nearest_matches,
)
from latent_loop.verification.pipeline import (
    FeatureStore,
    VerificationRun,
    label_pairs,
    run_dirname,
    summarize,
    verify_candidates,
)

__all__ = [
    "REASONS",
    "EssentialEstimate",
    "FeatureStore",
    "MatchConfig",
    "Matches",
    "OrbConfig",
    "OrbFeatures",
    "VerificationConfig",
    "VerificationResult",
    "VerificationRun",
    "estimate_essential",
    "extract_orb",
    "hamming_distance",
    "label_pairs",
    "match_features",
    "ransac_essential_numpy",
    "raw_nearest_matches",
    "run_dirname",
    "summarize",
    "verify_candidates",
    "verify_features",
    "verify_pair",
]
