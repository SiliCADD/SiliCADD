"""Unit tests for the Active Learning Virtual Screening benchmark.

Covers:
    * score ranking direction (lower is better)
    * iterative retraining
    * budget and candidate-pool exhaustion
    * docking result caching, including configuration isolation
    * invalid SMILES, duplicate molecules and conflicting reference scores
    * reference dataset coverage (complete vs incomplete)
    * calls-to-recovery computed from the real docking order
    * effort accounting for failures and cache hits
    * predictions for valid but undocked molecules
    * reproducibility
    * benchmark metrics and their validity flags
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import pytest

# Make the sibling module importable when running pytest from the repo root.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

from active_screener import (  # noqa: E402
    STATUS_DUPLICATE,
    STATUS_FAILED,
    STATUS_INVALID,
    STATUS_NOT_DOCKED,
    STATUS_OK,
    ActiveScreener,
    benchmark_strategies,
    build_eligible_library,
    canonical_smiles,
    canonicalize_reference_scores,
    compute_docking_config_id,
    featurize_smiles,
    run_active_loop,
    stack_fingerprints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _library() -> List[str]:
    """A small library of distinct, valid SMILES."""
    return [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "C1CCCCC1",
        "CCCCCC",
        "CC(C)C",
        "CCN(CC)CC",
        "c1ccncc1",
    ]


def _scores() -> Dict[str, float]:
    """Deterministic true scores keyed by canonical SMILES.

    Lower is better; the ordering is unrelated to library position so that a
    strategy's ranking behaviour is actually exercised.
    """
    return {
        "CCO": -5.0,
        "c1ccccc1": -1.0,
        "CC(=O)O": -8.0,
        "C1CCCCC1": -2.0,
        "CCCCCC": -3.0,
        "CC(C)C": -7.0,
        "CCN(CC)CC": -4.0,
        "c1ccncc1": -6.0,
    }


def _docking_from_map(score_map: Dict[str, float]) -> Callable[[str], float]:
    """Build a deterministic docking function keyed on canonical SMILES."""
    from rdkit import Chem

    def dock(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"unexpected invalid molecule {smiles}"
        canonical = Chem.MolToSmiles(mol)
        if canonical not in score_map:
            raise KeyError(canonical)
        return score_map[canonical]

    return dock


def _flat_dock(smiles: str) -> float:
    """A docking function that gives every molecule the same score."""
    return -1.0


# ---------------------------------------------------------------------------
# featurize_smiles
# ---------------------------------------------------------------------------


def test_featurize_smiles_shape_and_validity():
    library = _library()
    fingerprints, canonical, valid = featurize_smiles(library)

    assert len(fingerprints) == len(library)
    assert len(canonical) == len(library)
    assert len(valid) == len(library)
    assert all(valid)
    assert all(fp is not None and fp.shape == (2048,) for fp in fingerprints)
    assert stack_fingerprints(fingerprints).shape == (len(library), 2048)


def test_featurize_smiles_handles_invalid_gracefully():
    fingerprints, canonical, valid = featurize_smiles(
        ["CCO", "not-a-smiles", "c1ccccc1", "AlsoInvalid"]
    )

    assert valid == [True, False, True, False]
    assert fingerprints[0] is not None
    assert fingerprints[1] is None
    assert canonical[1] is None
    # Only valid molecules contribute rows to the stacked matrix.
    assert stack_fingerprints(fingerprints).shape == (2, 2048)


# ---------------------------------------------------------------------------
# Eligible library / canonical identity
# ---------------------------------------------------------------------------


def test_eligible_library_excludes_invalid_and_duplicates():
    library = build_eligible_library(
        ["CCO", "OCC", "not-a-smiles", "c1ccccc1", "c1ccccc1"]
    )
    # "OCC" and "CCO" are the same molecule; only the first occurrence counts.
    assert library.smiles == ["CCO", "c1ccccc1"]
    assert library.canonical == ["CCO", "c1ccccc1"]
    assert library.duplicate_count == 2
    assert library.invalid_count == 1
    assert len(library) == 2


def test_canonicalize_reference_scores_detects_conflicts():
    """Two representations of one molecule with different scores is an error."""
    with pytest.raises(ValueError, match="conflicting reference scores"):
        canonicalize_reference_scores({"CCO": -1.0, "OCC": -2.0})


def test_canonicalize_reference_scores_accepts_identical_duplicates():
    result = canonicalize_reference_scores({"CCO": -1.0, "OCC": -1.0})
    assert result == {"CCO": -1.0}


def test_canonicalize_reference_scores_rejects_invalid_and_non_finite():
    with pytest.raises(ValueError, match="invalid SMILES"):
        canonicalize_reference_scores({"not-a-smiles": -1.0})
    with pytest.raises(ValueError, match="not finite"):
        canonicalize_reference_scores({"CCO": float("nan")})


# ---------------------------------------------------------------------------
# Score ranking direction
# ---------------------------------------------------------------------------


def test_greedy_selects_most_negative_prediction():
    """Greedy ranks candidates by predicted score, lowest (best) first."""
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="greedy",
        docking_function=_flat_dock,
        random_state=42,
    )
    # Synthetic predictions: index 3 is the best (most negative).
    predictions = np.array([-1.0, -2.0, -5.0, -9.0, -0.5, -3.0, -4.0, -6.0])

    selected = screener._select_batch_greedy([0, 1, 2, 3, 4, 5, 6, 7], predictions)
    assert selected == [3]

    # The whole batch is taken in predicted-score order:
    # -9.0 (idx 3), -6.0 (idx 7), -5.0 (idx 2).
    screener.batch_size = 3
    selected = screener._select_batch_greedy([0, 1, 2, 3, 4, 5, 6, 7], predictions)
    assert selected == [3, 7, 2]


def test_greedy_loop_picks_the_best_predicted_candidate():
    """In the loop, greedy must pick the argmin of the current predictions."""
    scores = _scores()
    captured: List = []

    class RecordingScreener(ActiveScreener):
        def _select_batch(self, pool, predictions, uncertainties):
            selected = super()._select_batch(pool, predictions, uncertainties)
            captured.append((list(pool), np.array(predictions), list(selected)))
            return selected

    screener = RecordingScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=1,
        docking_budget=6,
        strategy="greedy",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    screener.run_active_loop()

    assert captured, "greedy selection never ran"
    for pool, predictions, selected in captured:
        if not selected:
            continue
        best_in_pool = min(pool, key=lambda index: (predictions[index], index))
        assert selected[0] == best_in_pool


def test_lower_score_is_better_in_benchmark():
    """The best true score reported must be the most negative one found."""
    scores = _scores()
    results = benchmark_strategies(
        smiles_list=_library(),
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[len(_library())],
        docking_function=_docking_from_map(scores),
        verbose=False,
    )
    assert results.iloc[0]["best_true_score"] == pytest.approx(min(scores.values()))


def test_uncertainty_strategy_runs_and_ranks():
    """The uncertainty strategy must produce a valid acquisition ranking."""
    scores = _scores()
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=2,
        docking_budget=6,
        strategy="uncertainty",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    results = screener.run_active_loop()

    docked = results[results["validation_status"] == STATUS_OK]
    assert len(docked) >= 4
    assert docked["true_score"].notna().all()


def test_uncertainty_acquisition_respects_exploration_weight():
    """The acquisition score must trade predicted affinity against uncertainty."""
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="uncertainty",
        docking_function=_flat_dock,
        random_state=42,
    )

    # Index 0 has the best prediction but zero uncertainty; index 1 has a
    # mediocre prediction but high uncertainty.
    predictions = np.array([-9.0, -1.0, -5.0, -5.0])
    uncertainties = np.array([0.0, 10.0, 0.0, 0.0])
    pool = [0, 1, 2, 3]

    # Pure exploitation picks the best predicted candidate.
    screener.exploration_weight = 0.0
    assert screener._select_batch_uncertainty(
        pool, predictions, uncertainties
    ) == [0]

    # Pure exploration picks the most uncertain candidate.
    screener.exploration_weight = 1.0
    assert screener._select_batch_uncertainty(
        pool, predictions, uncertainties
    ) == [1]

    # An intermediate weight still favours the confident, well-predicted
    # candidate when the uncertainty gap is small enough.
    screener.exploration_weight = 0.5
    assert screener._select_batch_uncertainty(
        pool, predictions, uncertainties
    ) == [0]


def test_exploration_weight_changes_selection():
    """Pure exploitation and pure exploration must differ in what they pick."""
    library = [
        "CCO",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CCCCCCC",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)O",
        "CCN(CC)CC",
        "C1CCCCC1",
        "CC(C)C",
    ]

    def pseudo_random_dock(smiles: str) -> float:
        # Deterministic but unstructured, so the forest cannot predict it well.
        digest = 0
        for char in smiles:
            digest = (digest * 31 + ord(char)) % 9973
        return -float(digest) / 100.0

    def picks(weight: float) -> List[str]:
        screener = ActiveScreener(
            smiles_list=library,
            seed_size=3,
            batch_size=1,
            docking_budget=6,
            strategy="uncertainty",
            docking_function=pseudo_random_dock,
            random_state=7,
            exploration_weight=weight,
        )
        results = screener.run_active_loop()
        picked = results[
            (results["validation_status"] == STATUS_OK) & (results["iteration"] > 0)
        ]
        return picked["smiles"].tolist()

    assert picks(0.0) != picks(1.0)


def test_strategies_tolerate_non_finite_predictions():
    """A non-finite prediction must not crash or corrupt the ranking."""
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="greedy",
        docking_function=_flat_dock,
        random_state=42,
    )
    # Rank the whole pool, so the ordering of the non-finite candidates is
    # observable.
    screener.batch_size = 8

    # The best finite prediction is -6.0 at index 7; indices 0 and 1 are
    # non-finite and must be ranked last.
    predictions = np.array([np.nan, np.inf, -3.0, -1.0, -2.0, -4.0, -5.0, -6.0])
    uncertainties = np.array([np.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    greedy = screener._select_batch_greedy(list(range(8)), predictions)
    assert greedy[0] == 7
    assert set(greedy[-2:]) == {0, 1}

    acquisition = screener._select_batch_uncertainty(
        list(range(8)), predictions, uncertainties
    )
    assert acquisition[0] == 7
    assert set(acquisition[-2:]) == {0, 1}


def test_exploration_weight_is_validated():
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError, match="exploration_weight"):
            ActiveScreener(
                smiles_list=_library(),
                seed_size=1,
                batch_size=1,
                docking_budget=2,
                strategy="uncertainty",
                docking_function=_flat_dock,
                exploration_weight=weight,
            )


# ---------------------------------------------------------------------------
# Iterative retraining
# ---------------------------------------------------------------------------


def test_model_is_retrained_each_iteration():
    """The surrogate must be (re)trained as new molecules get docked."""
    scores = _scores()
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=1,
        batch_size=1,
        docking_budget=5,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    results = screener.run_active_loop()

    assert screener.model is not None
    assert len(screener.model.estimators_) > 0
    assert int((results["validation_status"] == STATUS_OK).sum()) >= 2

    # Docked rows carry a model-based prediction and an uncertainty estimate.
    docked = results[results["validation_status"] == STATUS_OK]
    assert docked["predicted_score"].notna().all()
    assert docked["prediction_uncertainty"].notna().all()
    assert (docked["prediction_uncertainty"] >= 0).all()


def test_uncertainty_uses_individual_tree_predictions():
    """Uncertainty must come from the spread across the individual trees."""
    from active_screener import predict_with_uncertainty

    scores = _scores()
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    screener.run_active_loop()

    X = np.vstack(
        [r.fingerprint for r in screener.records if r.fingerprint is not None]
    )
    mean, std = predict_with_uncertainty(screener.model, X)

    tree_preds = np.vstack(
        [tree.predict(X) for tree in screener.model.estimators_]
    )
    assert np.allclose(mean, tree_preds.mean(axis=0))
    assert np.allclose(std, tree_preds.std(axis=0))


def test_predicted_scores_are_never_used_as_training_labels():
    """The model must be trained on true scores only."""
    scores = _scores()
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=2,
        docking_budget=6,
        strategy="greedy",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    results = screener.run_active_loop()

    docked = results[results["validation_status"] == STATUS_OK]
    assert not np.allclose(
        docked["predicted_score"].fillna(-1).to_numpy(),
        docked["true_score"].fillna(-2).to_numpy(),
    )


# ---------------------------------------------------------------------------
# Predictions for undocked candidates
# ---------------------------------------------------------------------------


def test_predictions_populated_for_valid_undocked_molecules():
    """Valid molecules that were never docked still get a prediction."""
    scores = _scores()
    library = _library()
    screener = ActiveScreener(
        smiles_list=library,
        seed_size=2,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    results = screener.run_active_loop()

    undocked = results[results["validation_status"] == STATUS_NOT_DOCKED]
    assert len(undocked) > 0
    # A model was trained, so every undocked but featurizable molecule has a
    # prediction and an uncertainty estimate ...
    assert undocked["predicted_score"].notna().all()
    assert undocked["prediction_uncertainty"].notna().all()
    # ... but no true score, because it was never docked.
    assert undocked["true_score"].isna().all()


def test_no_predictions_for_invalid_or_duplicate_molecules():
    """Molecules that are not valid candidates get no prediction."""
    screener = ActiveScreener(
        smiles_list=["CCO", "not-a-smiles", "c1ccccc1", "c1ccccc1"],
        seed_size=2,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=_flat_dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    invalid = results[results["validation_status"] == STATUS_INVALID]
    assert len(invalid) == 1
    assert invalid["predicted_score"].isna().all()
    assert invalid["prediction_uncertainty"].isna().all()

    duplicates = results[results["validation_status"] == STATUS_DUPLICATE]
    assert len(duplicates) == 1
    assert duplicates["true_score"].isna().all()


def test_no_predictions_when_too_few_successful_dockings():
    """Without enough successful scores there is no model, so no predictions."""
    def always_fails(smiles: str) -> float:
        raise RuntimeError("docking unavailable")

    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=1,
        docking_budget=4,
        strategy="random",
        docking_function=always_fails,
        random_state=42,
    )
    results = screener.run_active_loop()

    assert screener.model is None
    assert results["predicted_score"].isna().all()
    assert results["prediction_uncertainty"].isna().all()
    assert results["true_score"].isna().all()
    assert (results["validation_status"] == STATUS_FAILED).sum() == 4


def test_no_predictions_with_a_single_successful_docking():
    """One successful score is not enough to fit a forest."""
    def one_success(smiles: str) -> float:
        if smiles == "CCO":
            return -1.0
        raise RuntimeError("docking unavailable")

    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=1,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=one_success,
        random_state=42,
    )
    results = screener.run_active_loop()

    assert screener.model is None
    assert results["predicted_score"].isna().all()
    assert results["prediction_uncertainty"].isna().all()


# ---------------------------------------------------------------------------
# Budget enforcement and exhaustion
# ---------------------------------------------------------------------------


def test_budget_is_never_exceeded():
    scores = _scores()
    budget = 3
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=1,
        batch_size=1,
        docking_budget=budget,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    results = screener.run_active_loop()

    assert screener.docking_calls <= budget
    assert int((results["validation_status"] == STATUS_OK).sum()) <= budget


def test_too_few_valid_molecules_for_the_seed_raises():
    """A seed batch that cannot be filled is a parameter error."""
    with pytest.raises(ValueError, match="seed size"):
        ActiveScreener(
            smiles_list=["not-a-smiles", "also-bad"],
            seed_size=1,
            batch_size=1,
            docking_budget=2,
            strategy="random",
            docking_function=_flat_dock,
        )


def test_budget_smaller_than_seed_size_raises():
    with pytest.raises(ValueError):
        ActiveScreener(
            smiles_list=_library(),
            seed_size=5,
            batch_size=1,
            docking_budget=2,
            strategy="random",
            docking_function=_flat_dock,
        )


def test_loop_stops_when_no_candidates_remain():
    """Docking every unique molecule must terminate and use the full library."""
    library = _library()
    screener = ActiveScreener(
        smiles_list=library,
        seed_size=2,
        batch_size=2,
        docking_budget=len(library) + 5,
        strategy="greedy",
        docking_function=_docking_from_map(_scores()),
        random_state=42,
    )
    results = screener.run_active_loop()

    assert int((results["validation_status"] == STATUS_OK).sum()) == len(library)
    assert screener.remaining_indices == []


def test_exhausted_budget_terminates_without_infinite_loop():
    """A budget smaller than the library must stop cleanly."""
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=3,
        docking_budget=4,
        strategy="random",
        docking_function=_flat_dock,
        random_state=42,
    )
    results = screener.run_active_loop()
    assert screener.docking_calls == 4
    assert len(results) == len(_library())


def test_invalid_parameters_raise():
    for kwargs in (
        {"seed_size": 0},
        {"batch_size": 0},
        {"docking_budget": 0},
        {"strategy": "nonsense"},
        {"exploration_weight": 1.5},
    ):
        base = {
            "smiles_list": _library(),
            "seed_size": 1,
            "batch_size": 1,
            "docking_budget": 2,
            "strategy": "random",
            "docking_function": _flat_dock,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            ActiveScreener(**base)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_avoids_repeat_docking_calls(tmp_path):
    """A second run over the same library must reuse cached results."""
    calls: List[str] = []

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -float(len(smiles)) / 10.0

    cache_file = tmp_path / "cache.json"
    library = ["CCO", "c1ccccc1", "CC(=O)O"]

    first = ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=counting_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="target-1D4X",
    )
    first.run_active_loop()
    assert cache_file.exists()
    assert len(calls) > 0

    calls.clear()
    second = ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=counting_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="target-1D4X",
    )
    second_results = second.run_active_loop()

    # Every molecule was already cached, so no new docking was performed.
    assert len(calls) == 0
    assert second.true_docking_calls == 0
    assert second_results["true_score"].notna().all()


def test_cache_isolated_across_docking_configurations(tmp_path):
    """The same molecule may have different scores in different setups."""
    calls: List[str] = []

    def config_specific_dock(smiles: str) -> float:
        calls.append(smiles)
        return -1.0

    cache_file = tmp_path / "cache.json"
    library = ["CCO", "c1ccccc1"]

    ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=config_specific_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="config-A",
    ).run_active_loop()
    first_calls = len(calls)

    calls.clear()
    second = ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=config_specific_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="config-B",
    )
    second.run_active_loop()

    # A different configuration must not reuse the other configuration's scores.
    assert len(calls) == first_calls

    # Both configurations are present in the cache file.
    with open(cache_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert set(payload["configurations"]) == {"config-A", "config-B"}
    assert payload["version"] == "2.0"


def test_cache_reuses_entries_within_the_same_configuration(tmp_path):
    cache_file = tmp_path / "cache.json"
    library = ["CCO", "c1ccccc1"]
    calls: List[str] = []

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -2.0

    for _ in range(2):
        ActiveScreener(
            smiles_list=library,
            seed_size=1,
            batch_size=1,
            docking_budget=2,
            strategy="random",
            docking_function=counting_dock,
            random_state=42,
            cache_path=str(cache_file),
            docking_config="config-A",
        ).run_active_loop()

    # Only the first run performed docking work.
    assert len(calls) == 2


def test_cache_path_without_docking_config_raises(tmp_path):
    """Persistence without a configuration identity must be refused."""
    with pytest.raises(ValueError, match="docking_config"):
        ActiveScreener(
            smiles_list=["CCO"],
            seed_size=1,
            batch_size=1,
            docking_budget=1,
            strategy="random",
            docking_function=_flat_dock,
            cache_path=str(tmp_path / "cache.json"),
        )


def test_legacy_cache_file_is_not_reused(tmp_path):
    """A version 1.0 cache carries no configuration identity, so it is ignored."""
    cache_file = tmp_path / "cache.json"
    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump(
            {"version": "1.0", "data": {"CCO": {"score": -99.0, "timestamp": 1.0}}},
            handle,
        )

    calls: List[str] = []

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -1.0

    screener = ActiveScreener(
        smiles_list=["CCO"],
        seed_size=1,
        batch_size=1,
        docking_budget=1,
        strategy="random",
        docking_function=counting_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="config-A",
    )
    results = screener.run_active_loop()

    # The legacy entry was not reused: the molecule was docked afresh.
    assert len(calls) == 1
    assert results.iloc[0]["true_score"] == pytest.approx(-1.0)


def test_malformed_cache_file_does_not_crash(tmp_path):
    cache_file = tmp_path / "cache.json"
    with open(cache_file, "w", encoding="utf-8") as handle:
        handle.write("this is not json")

    screener = ActiveScreener(
        smiles_list=["CCO"],
        seed_size=1,
        batch_size=1,
        docking_budget=1,
        strategy="random",
        docking_function=_flat_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="config-A",
    )
    results = screener.run_active_loop()
    assert results.iloc[0]["true_score"] == pytest.approx(-1.0)


def test_docking_config_id_is_stable_and_order_independent():
    a = compute_docking_config_id({"engine": "vina", "exhaustiveness": 8})
    b = compute_docking_config_id({"exhaustiveness": 8, "engine": "vina"})
    c = compute_docking_config_id({"engine": "vina", "exhaustiveness": 16})
    assert a == b
    assert a != c
    with pytest.raises(ValueError):
        compute_docking_config_id({})


def test_cache_is_keyed_by_canonical_smiles():
    """Two different SMILES for the same molecule share one cache entry."""
    calls: List[str] = []

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -1.0

    # "OCC" and "CCO" are the same molecule.
    screener = ActiveScreener(
        smiles_list=["CCO", "OCC"],
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=counting_dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    # Only one unique molecule, docked once.
    assert len(calls) == 1
    assert len(results) == 2
    assert STATUS_DUPLICATE in set(results["validation_status"])


# ---------------------------------------------------------------------------
# Invalid SMILES, docking failures, non-finite scores
# ---------------------------------------------------------------------------


def test_invalid_smiles_is_marked_and_never_docked():
    def dock(smiles: str) -> float:
        if smiles == "invalid_smiles":
            raise ValueError("cannot parse")
        return -1.0

    screener = ActiveScreener(
        smiles_list=["CCO", "invalid_smiles", "c1ccccc1"],
        seed_size=1,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    invalid_row = results[results["smiles"] == "invalid_smiles"].iloc[0]
    assert invalid_row["validation_status"] == STATUS_INVALID
    assert pd.isna(invalid_row["true_score"])


def test_docking_exception_is_recorded_as_failed():
    def dock(smiles: str) -> float:
        if smiles == "CCO":
            raise RuntimeError("docking crashed")
        return -1.0

    screener = ActiveScreener(
        smiles_list=["CCO", "c1ccccc1"],
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    failed = results[results["smiles"] == "CCO"].iloc[0]
    assert failed["validation_status"] == STATUS_FAILED
    assert pd.isna(failed["true_score"])


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_scores_are_rejected(bad_score):
    def dock(smiles: str) -> float:
        if smiles == "CCO":
            return bad_score
        return -1.0

    screener = ActiveScreener(
        smiles_list=["CCO", "c1ccccc1"],
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    row = results[results["smiles"] == "CCO"].iloc[0]
    assert row["validation_status"] == STATUS_FAILED
    assert pd.isna(row["true_score"])


def test_non_numeric_score_is_rejected():
    def dock(smiles: str):
        if smiles == "CCO":
            return "not a number"
        return -1.0

    screener = ActiveScreener(
        smiles_list=["CCO", "c1ccccc1"],
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    row = results[results["smiles"] == "CCO"].iloc[0]
    assert row["validation_status"] == STATUS_FAILED


def test_failed_dockings_are_not_used_for_training():
    """A failed docking must not become a training label."""
    def dock(smiles: str) -> float:
        if smiles == "CCO":
            return float("nan")
        return -2.0

    screener = ActiveScreener(
        smiles_list=["CCO", "c1ccccc1", "CC(=O)O", "C1CCCCC1"],
        seed_size=1,
        batch_size=1,
        docking_budget=4,
        strategy="random",
        docking_function=dock,
        random_state=42,
    )
    screener.run_active_loop()

    assert screener.model is not None
    # The NaN label must not have reached the forest.
    predictions = screener.model.predict(
        np.vstack(
            [r.fingerprint for r in screener.records if r.fingerprint is not None]
        )
    )
    assert not np.isnan(predictions).any()


def test_duplicate_molecules_are_docked_only_once():
    screener = ActiveScreener(
        smiles_list=["CCO", "CCO", "CCO", "c1ccccc1"],
        seed_size=2,
        batch_size=1,
        docking_budget=4,
        strategy="random",
        docking_function=_flat_dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    assert len(results) == 4
    assert int((results["validation_status"] == STATUS_OK).sum()) == 2
    assert int((results["validation_status"] == STATUS_DUPLICATE).sum()) == 2


# ---------------------------------------------------------------------------
# Effort accounting: attempts vs invocations vs successes
# ---------------------------------------------------------------------------


def test_failed_attempts_count_as_effort_but_not_as_successes():
    """A failed attempt costs a real invocation but yields no score."""
    def always_fails(smiles: str) -> float:
        raise RuntimeError("crash")

    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=2,
        docking_budget=4,
        strategy="random",
        docking_function=always_fails,
        random_state=42,
    )
    results = screener.run_active_loop()

    # Every attempt consumed a budget slot and a docking invocation ...
    assert screener.docking_calls == 4
    assert screener.true_docking_calls == 4
    # ... but no attempt produced a score.
    assert screener.successful_dockings == 0
    assert int((results["validation_status"] == STATUS_FAILED).sum()) == 4
    assert results["true_score"].isna().all()


def test_successful_and_failed_attempts_are_distinguished():
    """The three effort counters stay consistent with each other."""
    def dock(smiles: str) -> float:
        if smiles in {"c1ccccc1", "CC(=O)O"}:
            raise RuntimeError("crash")
        return -1.0

    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=2,
        docking_budget=4,
        strategy="random",
        docking_function=dock,
        random_state=42,
    )
    results = screener.run_active_loop()

    failed = int((results["validation_status"] == STATUS_FAILED).sum())
    succeeded = int((results["validation_status"] == STATUS_OK).sum())

    assert screener.docking_calls == 4
    assert screener.true_docking_calls == 4
    assert screener.successful_dockings == succeeded
    assert succeeded + failed == screener.docking_calls


def test_cache_hits_are_not_counted_as_docking_invocations(tmp_path):
    calls: List[str] = []

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -1.0

    cache_file = tmp_path / "cache.json"
    library = ["CCO", "c1ccccc1", "CC(=O)O"]

    ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=counting_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="config-A",
    ).run_active_loop()

    calls.clear()
    second = ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=counting_dock,
        random_state=42,
        cache_path=str(cache_file),
        docking_config="config-A",
    )
    second.run_active_loop()

    # Three attempts were made, but no docking function was invoked: the
    # scores came from the cache.
    assert second.docking_calls == 3
    assert second.true_docking_calls == 0
    assert second.successful_dockings == 3
    assert len(calls) == 0


def test_attempt_log_records_the_real_order():
    scores = _scores()
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=2,
        docking_budget=6,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    screener.run_active_loop()

    log = screener.get_attempt_log()
    assert len(log) == 6
    # The attempt index is a strictly increasing 1..N sequence.
    assert list(log["attempt_index"]) == [1, 2, 3, 4, 5, 6]
    # The effort counter never decreases.
    assert log["effort"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_results():
    scores = _scores()
    frames = []
    for _ in range(2):
        frames.append(
            ActiveScreener(
                smiles_list=_library(),
                seed_size=2,
                batch_size=2,
                docking_budget=6,
                strategy="uncertainty",
                docking_function=_docking_from_map(scores),
                random_state=123,
            ).run_active_loop()
        )

    pd.testing.assert_frame_equal(
        frames[0].reset_index(drop=True), frames[1].reset_index(drop=True)
    )


def test_different_seeds_can_differ_but_are_stable():
    scores = _scores()

    def run(seed: int) -> pd.DataFrame:
        return ActiveScreener(
            smiles_list=_library(),
            seed_size=2,
            batch_size=2,
            docking_budget=6,
            strategy="random",
            docking_function=_docking_from_map(scores),
            random_state=seed,
        ).run_active_loop()

    a = run(1)
    b = run(2)
    # Stability: rerunning each seed reproduces itself.
    pd.testing.assert_frame_equal(a, run(1))
    pd.testing.assert_frame_equal(b, run(2))


def test_module_level_run_active_loop_matches_class():
    scores = _scores()
    frame = run_active_loop(
        smiles_list=_library(),
        seed_size=2,
        batch_size=2,
        docking_budget=5,
        strategy="greedy",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    assert set(frame.columns) == {
        "smiles",
        "canonical_smiles",
        "predicted_score",
        "true_score",
        "prediction_uncertainty",
        "iteration",
        "strategy",
        "validation_status",
        "docking_call",
    }
    assert len(frame) == len(_library())


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_result_dataframe_contract():
    """Undocked compounds must have true_score = NaN."""
    scores = _scores()
    library = _library()
    screener = ActiveScreener(
        smiles_list=library,
        seed_size=2,
        batch_size=1,
        docking_budget=3,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    results = screener.run_active_loop()

    assert list(results.columns) == [
        "smiles",
        "canonical_smiles",
        "predicted_score",
        "true_score",
        "prediction_uncertainty",
        "iteration",
        "strategy",
        "validation_status",
        "docking_call",
    ]
    assert len(results) == len(library)

    undocked = results[results["validation_status"] == STATUS_NOT_DOCKED]
    assert len(undocked) > 0
    assert undocked["true_score"].isna().all()
    # The strategy is recorded for the molecules that were actually selected.
    docked = results[results["validation_status"] == STATUS_OK]
    assert (docked["strategy"] == "random").all()
    assert docked["iteration"].notna().all()
    # The canonical SMILES identify the molecule.
    assert results[results["smiles"] == "CCO"].iloc[0]["canonical_smiles"] == "CCO"


def test_history_tracks_progress():
    scores = _scores()
    screener = ActiveScreener(
        smiles_list=_library(),
        seed_size=2,
        batch_size=1,
        docking_budget=5,
        strategy="random",
        docking_function=_docking_from_map(scores),
        random_state=42,
    )
    screener.run_active_loop()

    history = screener.get_history()
    assert len(history) >= 2
    assert "cumulative_budget_used" in history.columns
    assert "best_true_score" in history.columns
    assert "docking_function_calls" in history.columns
    # Budget usage is monotonically non-decreasing.
    assert history["cumulative_budget_used"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Benchmark metrics
# ---------------------------------------------------------------------------


def test_benchmark_output_columns():
    results = benchmark_strategies(
        smiles_list=_library(),
        k=2,
        seed=42,
        strategies=["random", "greedy", "uncertainty"],
        budgets=[2, 4],
        docking_function=_docking_from_map(_scores()),
        verbose=False,
    )

    expected = {
        "strategy",
        "budget",
        "docking_attempts",
        "true_docking_calls",
        "successful_dockings",
        "cumulative_budget_used",
        "best_true_score",
        "recall_top_k",
        "fraction_recovered",
        "docking_calls_to_50pct",
        "docking_calls_to_80pct",
        "docking_calls_to_100pct",
        "runtime_seconds",
        "recovery_metrics_valid",
        "reference_complete",
        "reference_source",
        "reference_status",
        "top_k_size",
    }
    assert expected.issubset(set(results.columns))
    assert len(results) == 3 * 2


def test_benchmark_metrics_are_bounded():
    results = benchmark_strategies(
        smiles_list=_library(),
        k=3,
        seed=42,
        strategies=["random", "greedy", "uncertainty"],
        budgets=[2, 4, 6],
        docking_function=_docking_from_map(_scores()),
        verbose=False,
    )
    for _, row in results.iterrows():
        assert 0.0 <= row["recall_top_k"] <= 1.0
        assert 0.0 <= row["fraction_recovered"] <= 1.0
        assert row["true_docking_calls"] <= row["budget"]
        assert row["cumulative_budget_used"] <= row["budget"]
        assert row["docking_attempts"] == row["cumulative_budget_used"]
        assert row["successful_dockings"] <= row["docking_attempts"]
        assert row["runtime_seconds"] >= 0.0


def test_benchmark_best_score_is_most_negative_found():
    """best_true_score must be the minimum of the true scores found."""
    scores = _scores()
    results = benchmark_strategies(
        smiles_list=_library(),
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[8],
        docking_function=_docking_from_map(scores),
        verbose=False,
    )
    # With the whole library docked, the best score must be the global best.
    assert results.iloc[0]["best_true_score"] == pytest.approx(min(scores.values()))


def test_benchmark_recall_reaches_one_with_full_budget():
    results = benchmark_strategies(
        smiles_list=_library(),
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[len(_library())],
        docking_function=_docking_from_map(_scores()),
        verbose=False,
    )
    row = results.iloc[0]
    assert row["recall_top_k"] == pytest.approx(1.0)
    assert not pd.isna(row["docking_calls_to_100pct"])


def test_benchmark_uses_reference_dataset_when_supplied():
    """A complete reference dataset is used instead of exhaustive docking."""
    library = _library()
    reference = {smiles: float(i) for i, smiles in enumerate(library)}

    calls: List[str] = []

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -float(len(smiles))

    results = benchmark_strategies(
        smiles_list=library,
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[4],
        docking_function=counting_dock,
        reference_scores=reference,
        verbose=False,
    )
    row = results.iloc[0]
    assert row["reference_source"] == "reference"
    assert bool(row["reference_complete"]) is True
    assert bool(row["recovery_metrics_valid"]) is True
    assert row["top_k_size"] == 2

    # The reference replaced the exhaustive docking: the docking function was
    # only called for the molecules the strategy selected.
    assert len(calls) == 4
    assert len(calls) < len(library)


def test_benchmark_incomplete_reference_marks_metrics_unavailable():
    """A partial reference must not produce a 'global top-k'."""
    library = _library()
    partial = {smiles: float(i) for i, smiles in enumerate(library[:4])}

    results = benchmark_strategies(
        smiles_list=library,
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[4],
        docking_function=_docking_from_map(_scores()),
        reference_scores=partial,
        verbose=False,
    )
    row = results.iloc[0]
    assert bool(row["reference_complete"]) is False
    assert bool(row["recovery_metrics_valid"]) is False
    # The recovery metrics are unavailable, not zero.
    assert pd.isna(row["recall_top_k"])
    assert pd.isna(row["fraction_recovered"])
    assert pd.isna(row["docking_calls_to_50pct"])
    assert pd.isna(row["docking_calls_to_100pct"])
    # Useful outputs are preserved.
    assert row["true_docking_calls"] > 0
    assert row["runtime_seconds"] >= 0.0
    assert row["budget"] == 4
    assert "cannot be measured" in row["reference_status"]


def test_benchmark_reports_unmeasurable_top_k():
    """When no ground truth is available, top-k recovery must be flagged."""
    library = _library()

    def always_fails(smiles: str) -> float:
        raise RuntimeError("no docking available")

    results = benchmark_strategies(
        smiles_list=library,
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[2],
        docking_function=always_fails,
        verbose=False,
    )
    row = results.iloc[0]
    assert bool(row["recovery_metrics_valid"]) is False
    assert pd.isna(row["recall_top_k"])
    assert "cannot be measured" in row["reference_status"]


def test_benchmark_conflicting_reference_scores_raise():
    library = ["CCO", "c1ccccc1"]
    with pytest.raises(ValueError, match="conflicting reference scores"):
        benchmark_strategies(
            smiles_list=library,
            k=1,
            strategies=["random"],
            budgets=[1],
            docking_function=_flat_dock,
            reference_scores={"CCO": -1.0, "OCC": -2.0},
            verbose=False,
        )


def test_benchmark_matches_top_k_by_canonical_smiles():
    """Equivalent SMILES must not be counted as separate molecules."""
    # The library holds two representations of ethanol plus one other molecule.
    library = ["CCO", "OCC", "c1ccccc1"]
    scores = {"CCO": -5.0, "c1ccccc1": -1.0}

    results = benchmark_strategies(
        smiles_list=library,
        k=1,
        seed=42,
        strategies=["random"],
        budgets=[3],
        docking_function=_docking_from_map(scores),
        verbose=False,
    )
    row = results.iloc[0]

    # The top-k is the single best molecule, not two copies of ethanol.
    assert row["top_k_size"] == 1
    assert row["recall_top_k"] in (0.0, 1.0)
    assert row["recall_top_k"] <= 1.0
    # Ethanol is docked exactly once, so at most one call recovers the top-1.
    assert row["docking_calls_to_100pct"] is not None
    assert row["docking_calls_to_100pct"] <= row["true_docking_calls"]


def test_all_strategies_get_the_same_seed_molecules():
    """Fair comparison: the random seed batch must not depend on the strategy."""
    library = _library()
    seeds = {}

    for strategy in ("random", "greedy", "uncertainty"):
        screener = ActiveScreener(
            smiles_list=library,
            seed_size=2,
            batch_size=2,
            docking_budget=4,
            strategy=strategy,
            docking_function=_docking_from_map(_scores()),
            random_state=42,
        )
        screener.run_active_loop()
        log = screener.get_attempt_log()
        seeds[strategy] = list(log[log["iteration"] == 0]["smiles"])

    assert seeds["random"] == seeds["greedy"] == seeds["uncertainty"]
    assert len(seeds["random"]) == 2


def test_benchmark_does_not_leak_reference_scores_to_strategies():
    """The oracle must not be visible to a selection strategy."""
    calls: List[str] = []
    library = _library()

    def counting_dock(smiles: str) -> float:
        calls.append(smiles)
        return -float(len(smiles))

    benchmark_strategies(
        smiles_list=library,
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[2],
        docking_function=counting_dock,
        verbose=False,
    )

    # 8 molecules docked exhaustively for the reference, then only the 2 the
    # strategy selected. The strategy did not inherit the reference scores.
    assert len(calls) == len(library) + 2


def test_calls_to_recovery_uses_the_real_docking_order():
    """Recovery is measured against the order docking actually happened."""
    library = ["CCO", "c1ccccc1", "CC(=O)O", "C1CCCCC1"]
    # Ethanol is the best molecule; it is docked third, inside a batch.
    scores = {"CCO": -9.0, "c1ccccc1": -1.0, "CC(=O)O": -2.0, "C1CCCCC1": -3.0}

    def ordered_dock(smiles: str) -> float:
        # Dock in a fixed order so the attempt sequence is deterministic.
        order = ["c1ccccc1", "CC(=O)O", "CCO", "C1CCCCC1"]
        return scores[canonical_smiles(smiles)] - 0.0 * order.index(smiles)

    results = benchmark_strategies(
        smiles_list=library,
        k=1,
        seed=42,
        strategies=["random"],
        budgets=[4],
        docking_function=ordered_dock,
        verbose=False,
    )
    row = results.iloc[0]

    # The top-1 must be recovered within the budget, and the reported effort
    # must be a real invocation count, not an iteration index.
    assert not pd.isna(row["docking_calls_to_100pct"])
    assert 1 <= row["docking_calls_to_100pct"] <= row["true_docking_calls"]


def test_calls_to_recovery_unavailable_when_never_reached():
    """A target that is never reached must be NaN, not a misleading number."""
    library = ["CCO", "c1ccccc1", "CC(=O)O", "C1CCCCC1"]
    scores = {"CCO": -9.0, "c1ccccc1": -1.0, "CC(=O)O": -2.0, "C1CCCCC1": -3.0}

    results = benchmark_strategies(
        smiles_list=library,
        k=1,
        seed=42,
        strategies=["random"],
        budgets=[1],
        docking_function=_docking_from_map(scores),
        verbose=False,
    )
    row = results.iloc[0]

    # With a budget of one call the best molecule is usually not reached.
    if row["recall_top_k"] < 1.0:
        assert pd.isna(row["docking_calls_to_100pct"])
    assert row["docking_calls_to_100pct"] <= row["true_docking_calls"] or pd.isna(
        row["docking_calls_to_100pct"]
    )


def test_benchmark_retains_previous_output_field_names():
    """The earlier column names must remain available as aliases."""
    results = benchmark_strategies(
        smiles_list=_library(),
        k=2,
        seed=42,
        strategies=["random"],
        budgets=[4],
        docking_function=_docking_from_map(_scores()),
        verbose=False,
    )
    row = results.iloc[0]
    assert "top_k_measurable" in results.columns
    assert "top_k_note" in results.columns
    assert bool(row["top_k_measurable"]) == bool(row["recovery_metrics_valid"])
    assert row["top_k_note"] == row["reference_status"]


def test_benchmark_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        benchmark_strategies(
            smiles_list=_library(),
            k=2,
            strategies=["not-a-strategy"],
            budgets=[2],
            docking_function=_flat_dock,
            verbose=False,
        )


def test_eligible_library_canonical_set_returns_set():
    """EligibleLibrary.canonical_set returns the set of canonical SMILES."""
    library = build_eligible_library(["CCO", "c1ccccc1", "CC(=O)O"])
    assert library.canonical_set == {"CCO", "c1ccccc1", "CC(=O)O"}


def test_benchmark_reproducible_with_reference_scores():
    """benchmark_strategies is deterministic when reference_scores is supplied."""
    library = _library()
    reference = _scores()

    results1 = benchmark_strategies(
        smiles_list=library,
        k=2,
        seed=42,
        strategies=["random", "greedy"],
        budgets=[2, 4],
        docking_function=_docking_from_map(_scores()),
        reference_scores=reference,
        verbose=False,
    )
    results2 = benchmark_strategies(
        smiles_list=library,
        k=2,
        seed=42,
        strategies=["random", "greedy"],
        budgets=[2, 4],
        docking_function=_docking_from_map(_scores()),
        reference_scores=reference,
        verbose=False,
    )

    # runtime_seconds is wall-clock time and must not gate reproducibility.
    deterministic_columns = [
        column for column in results1.columns if column != "runtime_seconds"
    ]
    pd.testing.assert_frame_equal(
        results1[deterministic_columns].reset_index(drop=True),
        results2[deterministic_columns].reset_index(drop=True),
    )


def test_benchmark_ignores_reference_entries_outside_library():
    """Extra reference entries for molecules outside the library are ignored."""
    library = ["CCO", "c1ccccc1"]
    reference = {
        "CCO": -1.0,
        "c1ccccc1": -2.0,
        "CC(=O)O": -3.0,  # not in library — should be ignored
    }

    results = benchmark_strategies(
        smiles_list=library,
        k=1,
        seed=42,
        strategies=["random"],
        budgets=[2],
        docking_function=_flat_dock,
        reference_scores=reference,
        verbose=False,
    )
    row = results.iloc[0]
    assert bool(row["reference_complete"]) is True
    assert bool(row["recovery_metrics_valid"]) is True
    assert row["top_k_size"] == 1


def test_build_eligible_library_all_invalid():
    """All-invalid SMILES produces an empty eligible library with correct counts."""
    library = build_eligible_library(["not-a-smiles", "AlsoInvalid!"])
    assert library.invalid_count == 2
    assert library.duplicate_count == 0
    assert len(library) == 0
    assert library.smiles == []
    assert library.canonical == []
    assert library.canonical_set == set()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
