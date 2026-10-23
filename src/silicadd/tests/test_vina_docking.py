"""Unit tests for the AutoDock Vina docking engine.

The Vina and Meeko interfaces are mocked, so the suite never needs a
receptor, a docking box on a real target, network access or a real docking
run. The only unmocked third-party code exercised here is RDKit, which is a
hard dependency of the project and is used to build the ligand conformers.

Covers:
    * receptor and docking-box validation (no invented target or box)
    * ligand preparation through RDKit and Meeko
    * the Vina call sequence (receptor and maps prepared once, reused)
    * the best-affinity selection over the docked poses
    * failure handling: no synthetic score is ever substituted
    * integration with the active learning loop and its result contract
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

# Make the sibling modules importable when running pytest from the repo root.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

import vina_docking  # noqa: E402
from active_screener import (  # noqa: E402
    STATUS_FAILED,
    STATUS_OK,
    ActiveScreener,
    compute_docking_config_id,
    run_active_loop,
)
from vina_docking import (  # noqa: E402
    VinaDockingEngine,
    VinaDockingError,
    make_vina_docking_function,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: The affinity values the fake Vina returns, one per pose.
FAKE_AFFINITIES = (-3.5, -7.25, -5.0)


class FakeVina:
    """A stand-in for the AutoDock Vina Python API.

    It records every call so the tests can assert on the call sequence, and
    it can be told to fail in order to exercise the error paths.
    """

    #: Set to an exception instance to make the docking run fail.
    fail_on_dock: Any = None
    #: Set to an exception instance to make the receptor setup fail.
    fail_on_setup: Any = None
    #: Affinities returned by ``energies``; one row per pose.
    affinities: Any = FAKE_AFFINITIES

    def __init__(self, sf_name="vina", cpu=0, seed=0, verbosity=1, **kwargs):
        self.sf_name = sf_name
        self.cpu = cpu
        self.seed = seed
        self.verbosity = verbosity
        self.kwargs = kwargs
        self.receptor: Any = None
        self.maps: Any = None
        self.ligand: Any = None
        self.dock_calls: List[Dict[str, Any]] = []
        self.energy_calls: List[Dict[str, Any]] = []
        self.set_receptor_calls = 0
        self.compute_maps_calls = 0

    def set_receptor(self, rigid_pdbqt_filename=None, flex_pdbqt_filename=None):
        if type(self).fail_on_setup is not None:
            raise type(self).fail_on_setup
        self.set_receptor_calls += 1
        self.receptor = rigid_pdbqt_filename

    def compute_vina_maps(
        self, center, box_size, spacing=0.375, force_even_voxels=False
    ):
        self.compute_maps_calls += 1
        self.maps = {
            "center": list(center),
            "box_size": list(box_size),
            "spacing": spacing,
        }

    def set_ligand_from_string(self, pdbqt_string):
        self.ligand = pdbqt_string

    def dock(self, exhaustiveness=8, n_poses=20, min_rmsd=1.0, max_evals=0):
        if type(self).fail_on_dock is not None:
            raise type(self).fail_on_dock
        self.dock_calls.append(
            {"exhaustiveness": exhaustiveness, "n_poses": n_poses}
        )

    def energies(self, n_poses=9, energy_range=3.0):
        self.energy_calls.append({"n_poses": n_poses})
        rows = []
        for affinity in type(self).affinities:
            # Vina returns [total, inter, intra, torsions, intra best pose].
            rows.append([affinity, affinity + 1.0, 0.5, 0.1, 0.5])
        return np.array(rows, dtype=float)


@pytest.fixture
def receptor(tmp_path):
    """A placeholder receptor file; its contents are never parsed."""
    path = tmp_path / "receptor.pdbqt"
    path.write_text("REMARK placeholder receptor\nEND\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_vina(monkeypatch):
    """Install the fake Vina and reset its shared state around each test."""
    FakeVina.fail_on_dock = None
    FakeVina.fail_on_setup = None
    FakeVina.affinities = FAKE_AFFINITIES
    monkeypatch.setattr(vina_docking, "_Vina", FakeVina)
    return FakeVina


def _engine(receptor, **kwargs):
    """Build an engine with a valid box, ready to be docked with."""
    defaults = {
        "receptor_pdbqt": str(receptor),
        "box_center": (10.0, 12.0, 8.0),
        "box_size": (22.0, 20.0, 22.0),
    }
    defaults.update(kwargs)
    return VinaDockingEngine(**defaults)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_missing_receptor_file_raises(tmp_path):
    """A receptor that does not exist must be reported, not invented."""
    with pytest.raises(VinaDockingError, match="receptor PDBQT file not found"):
        VinaDockingEngine(
            receptor_pdbqt=str(tmp_path / "absent.pdbqt"),
            box_center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
        )


def test_receptor_must_be_a_pdbqt_file(tmp_path):
    """Vina only accepts PDBQT receptors."""
    path = tmp_path / "receptor.pdb"
    path.write_text("ATOM\n", encoding="utf-8")
    with pytest.raises(VinaDockingError, match="must be a PDBQT file"):
        VinaDockingEngine(
            receptor_pdbqt=str(path),
            box_center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
        )


def test_box_center_is_required():
    """The binding site is never guessed: the box centre must be given."""
    with pytest.raises(ValueError, match="box_center"):
        VinaDockingEngine(
            receptor_pdbqt="receptor.pdbqt",
            box_center=None,
            box_size=(20.0, 20.0, 20.0),
        )


def test_box_size_is_required():
    """The box dimensions must be given explicitly too."""
    with pytest.raises(ValueError, match="box_size"):
        VinaDockingEngine(
            receptor_pdbqt="receptor.pdbqt",
            box_center=(0.0, 0.0, 0.0),
            box_size=None,
        )


@pytest.mark.parametrize(
    "center",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
    ],
)
def test_invalid_box_center_is_rejected(center):
    with pytest.raises(ValueError, match="box_center"):
        VinaDockingEngine(
            receptor_pdbqt="receptor.pdbqt",
            box_center=center,
            box_size=(20.0, 20.0, 20.0),
        )


@pytest.mark.parametrize(
    "size",
    [
        (0.0, 20.0, 20.0),
        (-5.0, 20.0, 20.0),
        (20.0, 20.0),
    ],
)
def test_invalid_box_size_is_rejected(size):
    with pytest.raises(ValueError, match="box_size"):
        VinaDockingEngine(
            receptor_pdbqt="receptor.pdbqt",
            box_center=(0.0, 0.0, 0.0),
            box_size=size,
        )


def test_invalid_engine_options_are_rejected(receptor):
    """Exhaustiveness, n_poses, spacing and scoring are validated."""
    for kwargs in (
        {"exhaustiveness": 0},
        {"n_poses": 0},
        {"spacing": 0.0},
        {"scoring": "not-a-scoring-function"},
    ):
        with pytest.raises(ValueError):
            _engine(receptor, **kwargs)


def test_missing_vina_package_is_reported(monkeypatch, receptor):
    """Without the Vina package the engine must say so, not fall back."""
    monkeypatch.setattr(vina_docking, "_Vina", None)
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="Vina Python API is not installed"):
        engine("CCO")


def test_missing_meeko_package_is_reported(monkeypatch, receptor):
    """Without Meeko the engine must say so, not fall back."""
    monkeypatch.setattr(vina_docking, "_MoleculePreparation", None)
    monkeypatch.setattr(vina_docking, "_PDBQTWriterLegacy", None)
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="Meeko is not installed"):
        engine("CCO")


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------


def test_invalid_smiles_raises_a_docking_error(receptor, fake_vina):
    """An unparseable SMILES is a failure, never a placeholder score."""
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="could not parse the SMILES"):
        engine("not-a-smiles")


def test_ligand_pdbqt_is_prepared_through_rdkit_and_meeko(receptor, fake_vina):
    """The ligand is parsed, protonated, embedded and written as PDBQT."""
    engine = _engine(receptor)
    pdbqt = engine.prepare_ligand_pdbqt("CCO")

    # A real PDBQT ligand: a root, atoms with coordinates and torsion count.
    assert "ROOT" in pdbqt
    assert "ATOM" in pdbqt
    assert "TORSDOF" in pdbqt
    # The heavy atoms of ethanol are present.
    assert pdbqt.count("ATOM") >= 3


def test_ligand_preparation_is_reproducible(receptor, fake_vina):
    """The same SMILES and seed must give the same conformer."""
    engine = _engine(receptor, seed=7)
    first = engine.prepare_ligand_pdbqt("CCO")
    second = engine.prepare_ligand_pdbqt("CCO")
    assert first == second


def test_ligand_preparation_failure_is_reported(receptor, monkeypatch):
    """A Meeko failure must surface as a docking error."""
    engine = _engine(receptor)

    class BrokenPreparator:
        def prepare(self, mol):
            raise RuntimeError("meeko exploded")

    monkeypatch.setattr(
        vina_docking, "_MoleculePreparation", lambda *a, **k: BrokenPreparator()
    )
    with pytest.raises(VinaDockingError, match="ligand preparation failed"):
        engine.prepare_ligand_pdbqt("CCO")


def test_meeko_write_failure_is_reported(receptor, monkeypatch):
    """A PDBQT that Meeko refuses to write must surface as an error."""

    class FakeSetup:
        pass

    class BrokenPreparator:
        def prepare(self, mol):
            return [FakeSetup()]

    class BrokenWriter:
        @staticmethod
        def write_string(setup, **kwargs):
            return "", False, "could not write"

    monkeypatch.setattr(
        vina_docking, "_MoleculePreparation", lambda *a, **k: BrokenPreparator()
    )
    monkeypatch.setattr(vina_docking, "_PDBQTWriterLegacy", BrokenWriter)

    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="could not write the PDBQT"):
        engine.prepare_ligand_pdbqt("CCO")


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------


def test_docking_returns_the_best_affinity(receptor, fake_vina):
    """The most negative pose is the score, in the loop's sign convention."""
    engine = _engine(receptor)
    affinity = engine("CCO")
    assert affinity == pytest.approx(min(FAKE_AFFINITIES))
    assert affinity < 0.0


def test_docking_uses_the_configured_box_and_seed(receptor, fake_vina):
    """The box and the seed reach the Vina API unchanged."""
    engine = _engine(receptor, seed=123, exhaustiveness=16)
    engine("CCO")

    instance = engine._vina
    assert instance.seed == 123
    assert instance.maps["center"] == [10.0, 12.0, 8.0]
    assert instance.maps["box_size"] == [22.0, 20.0, 22.0]
    assert instance.dock_calls[0]["exhaustiveness"] == 16


def test_receptor_and_maps_are_prepared_only_once(receptor, fake_vina):
    """The receptor setup is the expensive part and must not be repeated."""
    engine = _engine(receptor)
    for smiles in ("CCO", "c1ccccc1", "CC(=O)O"):
        engine(smiles)

    instance = engine._vina
    assert instance.set_receptor_calls == 1
    assert instance.compute_maps_calls == 1
    # ... but every molecule was docked.
    assert len(instance.dock_calls) == 3


def test_warm_up_prepares_the_receptor_once(receptor, fake_vina):
    """warm_up moves the setup cost out of the first docking attempt."""
    engine = _engine(receptor)
    engine.warm_up()
    engine.warm_up()
    assert engine._vina.set_receptor_calls == 1
    assert engine._vina.compute_maps_calls == 1


def test_receptor_setup_failure_is_reported(receptor, fake_vina):
    """A receptor Vina cannot load must be reported clearly."""
    fake_vina.fail_on_setup = RuntimeError("bad receptor file")
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="could not prepare the Vina"):
        engine("CCO")


def test_docking_failure_raises_and_returns_no_score(receptor, fake_vina):
    """A failed docking must raise -- never a synthetic score."""
    fake_vina.fail_on_dock = RuntimeError("vina crashed")
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="Vina docking failed"):
        engine("CCO")


def test_no_poses_raises_instead_of_returning_a_score(receptor, fake_vina):
    """An empty result set is a failure, not a score of zero."""
    fake_vina.affinities = ()
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="no poses"):
        engine("CCO")


def test_non_finite_affinities_raise(receptor, fake_vina):
    """Non-finite affinities are rejected rather than passed on."""
    fake_vina.affinities = (float("nan"), float("inf"))
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="no finite affinity"):
        engine("CCO")


def test_single_pose_energy_row_is_accepted(receptor, fake_vina):
    """A one-dimensional energy array is still a valid result."""
    fake_vina.affinities = (-4.25,)
    engine = _engine(receptor)
    assert engine("CCO") == pytest.approx(-4.25)


def test_engine_is_callable_and_returns_a_plain_float(receptor, fake_vina):
    """The engine satisfies the docking_function interface."""
    engine = _engine(receptor)
    score = engine("CCO")
    assert isinstance(score, float)
    assert not isinstance(score, bool)


def test_make_vina_docking_function_returns_a_callable(receptor, fake_vina):
    """The convenience factory produces the same callable interface."""
    dock = make_vina_docking_function(
        str(receptor), (10.0, 12.0, 8.0), (22.0, 20.0, 22.0)
    )
    assert callable(dock)
    assert dock("CCO") == pytest.approx(min(FAKE_AFFINITIES))


# ---------------------------------------------------------------------------
# Configuration identity
# ---------------------------------------------------------------------------


def test_config_settings_describe_the_setup(receptor, fake_vina):
    """The settings cover everything that materially affects the score."""
    engine = _engine(receptor, exhaustiveness=16, seed=99)
    settings = engine.config_settings()
    assert settings["engine"] == "vina"
    assert settings["exhaustiveness"] == 16
    assert settings["seed"] == 99
    assert settings["box"]["center"] == [10.0, 12.0, 8.0]
    assert settings["box"]["size"] == [22.0, 20.0, 22.0]
    assert settings["target"] == str(receptor)


def test_config_settings_hash_to_a_stable_id(receptor, fake_vina):
    """The settings can be used as a docking_config for the cache."""
    engine = _engine(receptor)
    config_id = compute_docking_config_id(engine.config_settings())
    assert config_id == compute_docking_config_id(engine.config_settings())
    assert len(config_id) == 32


def test_different_boxes_get_different_config_ids(receptor, fake_vina):
    """Two boxes are two different docking setups."""
    first = compute_docking_config_id(
        _engine(receptor, box_center=(0.0, 0.0, 0.0)).config_settings()
    )
    second = compute_docking_config_id(
        _engine(receptor, box_center=(5.0, 0.0, 0.0)).config_settings()
    )
    assert first != second


# ---------------------------------------------------------------------------
# Integration with the active learning loop
# ---------------------------------------------------------------------------


def test_engine_drives_the_active_learning_loop(receptor, fake_vina):
    """The engine plugs into run_active_loop without changing its contract."""
    engine = _engine(receptor)
    library = ["CCO", "c1ccccc1", "CC(=O)O"]

    results = run_active_loop(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="greedy",
        docking_function=engine,
        random_state=42,
    )

    # The result contract is unchanged.
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

    docked = results[results["validation_status"] == STATUS_OK]
    assert len(docked) == 2
    assert docked["true_score"].notna().all()
    assert set(docked["true_score"]) <= set(FAKE_AFFINITIES)


def test_failed_docking_is_recorded_as_failed_not_a_mock_score(
    receptor, fake_vina
):
    """A docking failure must never turn into a fabricated affinity."""
    fake_vina.fail_on_dock = RuntimeError("vina crashed")
    engine = _engine(receptor)
    library = ["CCO", "c1ccccc1", "CC(=O)O"]

    screener = ActiveScreener(
        smiles_list=library,
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=engine,
        random_state=42,
    )
    results = screener.run_active_loop()

    failed = results[results["validation_status"] == STATUS_FAILED]
    assert len(failed) == 2
    # The true score is NaN, not a placeholder number.
    assert failed["true_score"].isna().all()
    # The attempts were still charged to the effort budget.
    assert screener.docking_calls == 2
    assert screener.true_docking_calls == 2
    assert screener.successful_dockings == 0
    # And no placeholder leaked into the surrogate's training labels.
    assert screener.model is None


def test_failed_docking_never_becomes_a_training_label(receptor, fake_vina):
    """A failed docking must not teach the surrogate a made-up affinity."""
    fake_vina.fail_on_dock = RuntimeError("vina crashed")
    engine = _engine(receptor)

    screener = ActiveScreener(
        smiles_list=["CCO", "c1ccccc1"],
        seed_size=1,
        batch_size=1,
        docking_budget=2,
        strategy="random",
        docking_function=engine,
        random_state=42,
    )
    results = screener.run_active_loop()

    assert results["true_score"].isna().all()
    assert (results["validation_status"] == STATUS_FAILED).sum() == 2
    assert screener.model is None


def test_engine_works_with_a_configuration_aware_cache(receptor, fake_vina, tmp_path):
    """The engine's settings key the persistent docking cache."""
    engine = _engine(receptor)
    cache_file = tmp_path / "cache.json"

    def run_once():
        return ActiveScreener(
            smiles_list=["CCO", "c1ccccc1"],
            seed_size=1,
            batch_size=1,
            docking_budget=2,
            strategy="random",
            docking_function=engine,
            random_state=42,
            cache_path=str(cache_file),
            docking_config=engine.config_settings(),
        ).run_active_loop()

    run_once()
    dockings_after_first_run = len(engine._vina.dock_calls)
    assert dockings_after_first_run == 2

    results = run_once()
    # The second run reused the cache: no new docking was performed.
    assert len(engine._vina.dock_calls) == dockings_after_first_run
    assert results["true_score"].notna().all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
