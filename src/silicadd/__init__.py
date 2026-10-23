"""silicadd -- computational chemistry add-ons.

Currently exposes the active learning virtual screening benchmark and an
AutoDock Vina docking engine that plugs into it.
"""

from .active_screener import (
    ActiveScreener,
    DockingCache,
    DockingRecord,
    EligibleLibrary,
    TopKReference,
    benchmark_strategies,
    build_eligible_library,
    canonical_smiles,
    canonicalize_reference_scores,
    compute_docking_config_id,
    featurize_smiles,
    predict_with_uncertainty,
    run_active_loop,
    stack_fingerprints,
)
from .vina_docking import (
    DockedPose,
    VinaDockingEngine,
    VinaDockingError,
    make_vina_docking_function,
)

__all__ = [
    "ActiveScreener",
    "DockedPose",
    "DockingCache",
    "DockingRecord",
    "EligibleLibrary",
    "TopKReference",
    "VinaDockingEngine",
    "VinaDockingError",
    "benchmark_strategies",
    "build_eligible_library",
    "canonical_smiles",
    "canonicalize_reference_scores",
    "compute_docking_config_id",
    "featurize_smiles",
    "make_vina_docking_function",
    "predict_with_uncertainty",
    "run_active_loop",
    "stack_fingerprints",
]
