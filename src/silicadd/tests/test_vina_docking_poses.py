"""Unit tests for the pose-level output of the AutoDock Vina docking engine.

:meth:`VinaDockingEngine.dock_pdbqt_poses` is the pose-level counterpart of
:meth:`VinaDockingEngine.dock_pdbqt`: it keeps the geometry of every pose so
that it can be compared against a reference (for example to measure the RMSD
of a redocked pose against the experimentally observed one).

The Vina interface is mocked, so the suite never needs a receptor, a docking
box on a real target, network access or a real docking run. The ligand
PDBQT is produced by the real RDKit and Meeko, so the pose reconstruction
exercises the genuine ``REMARK SMILES IDX`` round trip rather than a stub.

Covers:
    * the poses and their affinities are returned in Vina's ranking order
    * the reconstructed molecules keep the atom order of the input molecule
    * the best pose agrees with the single-affinity interface
    * the energy range reaches Vina unchanged
    * failure handling: no pose is ever invented, and every failure is
      reported as a VinaDockingError
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
from vina_docking import (  # noqa: E402
    DockedPose,
    VinaDockingEngine,
    VinaDockingError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: The affinities the fake Vina reports, one per pose.
FAKE_AFFINITIES = (-3.5, -7.25, -5.0)

#: A small molecule used to build a realistic ligand PDBQT.
TEST_SMILES = "CCO"


def _prepared_pdbqt(smiles: str = TEST_SMILES) -> str:
    """Return a real Meeko-prepared PDBQT for ``smiles``.

    The real preparation is used so the pose blocks the fake Vina returns
    carry a genuine ``REMARK SMILES IDX`` mapping, and the reconstruction
    therefore exercises the actual Meeko code path.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    setups = MoleculePreparation().prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    assert is_ok, error_msg
    return pdbqt_string


def _pose_block(
    ligand_pdbqt: str, index: int, affinity: float, offset: float = 0.0
) -> str:
    """Build one Vina-style ``MODEL`` block for a pose.

    The block mimics what Vina writes: a ``REMARK VINA RESULT`` header, the
    ligand's ``REMARK SMILES`` / ``REMARK SMILES IDX`` mapping and the
    torsion tree, with every atom shifted by ``offset`` so that the poses
    differ from one another geometrically.
    """
    lines = ligand_pdbqt.split("\n")
    last_idx = max(
        i for i, line in enumerate(lines) if line.startswith("REMARK SMILES IDX")
    )
    root = next(i for i, line in enumerate(lines) if line.startswith("ROOT"))
    header = "\n".join(lines[: last_idx + 1])
    body = "\n".join(lines[root:])

    shifted = []
    for line in body.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + offset
            shifted.append(line[:30] + f"{x:8.3f}" + line[38:])
        else:
            shifted.append(line)

    return (
        f"MODEL {index}\n"
        f"REMARK VINA RESULT:   {affinity:8.3f}      0.000      0.000\n"
        f"{header}\n"
        f"{chr(10).join(shifted)}\n"
        "ENDMDL\n"
    )


class FakeVina:
    """A stand-in for the AutoDock Vina Python API that also yields poses.

    It records every call so the tests can assert on the call sequence, and
    it can be told to fail in order to exercise the error paths.
    """

    #: Set to an exception instance to make the docking run fail.
    fail_on_dock: Any = None
    #: Set to an exception instance to make the receptor setup fail.
    fail_on_setup: Any = None
    #: Affinities returned by ``energies``; one row per pose.
    affinities: Any = FAKE_AFFINITIES
    #: Poses returned by ``poses``; a PDBQT string, or ``None`` to build one.
    poses_pdbqt: Any = None

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
        self.pose_calls: List[Dict[str, Any]] = []
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
        if type(self).fail_on_setup is not None:
            raise type(self).fail_on_setup
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
        self.energy_calls.append(
            {"n_poses": n_poses, "energy_range": energy_range}
        )
        rows = []
        for affinity in type(self).affinities:
            # Vina returns [total, inter, intra, torsions, intra best pose].
            rows.append([affinity, affinity + 1.0, 0.5, 0.1, 0.5])
        return np.array(rows, dtype=float)

    def poses(self, n_poses=9, energy_range=3.0, coordinates_only=False):
        self.pose_calls.append(
            {"n_poses": n_poses, "energy_range": energy_range}
        )
        if type(self).poses_pdbqt is not None:
            return type(self).poses_pdbqt
        return "".join(
            _pose_block(self.ligand, index, affinity, offset=0.5 * index)
            for index, affinity in enumerate(type(self).affinities, start=1)
        )


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
    FakeVina.poses_pdbqt = None
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
# Successful pose generation
# ---------------------------------------------------------------------------


def test_poses_are_returned_with_their_affinities(receptor, fake_vina):
    """Every docked pose is returned, carrying its own affinity."""
    engine = _engine(receptor)
    poses = engine.dock_pdbqt_poses(_prepared_pdbqt())

    assert len(poses) == len(FAKE_AFFINITIES)
    assert all(isinstance(pose, DockedPose) for pose in poses)
    assert [pose.affinity for pose in poses] == list(FAKE_AFFINITIES)


def test_poses_are_returned_in_vinas_ranking_order(receptor, fake_vina):
    """The pose order follows the affinity order Vina reports.

    Vina ranks its poses best first, so the affinities it reports are
    already in descending order of quality (most negative first).
    """
    fake_vina.affinities = (-9.0, -4.0, -1.0)
    engine = _engine(receptor)
    poses = engine.dock_pdbqt_poses(_prepared_pdbqt())

    affinities = [pose.affinity for pose in poses]
    assert affinities == [-9.0, -4.0, -1.0]
    # The first pose is the one Vina ranks best, which is the most negative.
    assert poses[0].affinity == min(affinities)


def test_pose_molecules_keep_the_atom_order_of_the_input(receptor, fake_vina):
    """The reconstruction keeps the input molecule's atom order.

    This is what makes the poses comparable by substructure matching: the
    ``REMARK SMILES IDX`` mapping survives the round trip, so no renumbering
    by hand is needed before an RMSD can be computed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    engine = _engine(receptor)
    poses = engine.dock_pdbqt_poses(_prepared_pdbqt())

    reference = Chem.MolFromSmiles(TEST_SMILES)
    reference = Chem.AddHs(reference)
    AllChem.EmbedMolecule(reference, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(reference)
    expected = [atom.GetSymbol() for atom in reference.GetAtoms()]

    for pose in poses:
        assert [atom.GetSymbol() for atom in pose.mol.GetAtoms()] == expected


def test_pose_molecules_carry_the_docked_coordinates(receptor, fake_vina):
    """Each pose holds a 3D conformer, not an empty molecule."""
    engine = _engine(receptor)
    poses = engine.dock_pdbqt_poses(_prepared_pdbqt())

    for pose in poses:
        assert pose.mol.GetNumConformers() == 1
        positions = pose.mol.GetConformer().GetPositions()
        assert positions.shape[1] == 3
        assert np.isfinite(positions).all()


def test_best_pose_agrees_with_the_single_affinity_interface(
    receptor, fake_vina
):
    """The pose-level and affinity-level views report the same best score."""
    engine = _engine(receptor)
    pdbqt = _prepared_pdbqt()

    poses = engine.dock_pdbqt_poses(pdbqt)
    assert min(pose.affinity for pose in poses) == engine.dock_pdbqt(pdbqt)


def test_existing_callable_interface_still_returns_one_affinity(
    receptor, fake_vina
):
    """The SMILES-in / float-out contract is unchanged."""
    engine = _engine(receptor)
    score = engine(TEST_SMILES)

    assert isinstance(score, float)
    assert score == pytest.approx(min(FAKE_AFFINITIES))


def test_energy_range_reaches_vina_unchanged(receptor, fake_vina):
    """The requested energy window is passed to Vina, not invented."""
    engine = _engine(receptor)
    engine.dock_pdbqt_poses(_prepared_pdbqt(), energy_range=42.5)

    assert engine._vina.pose_calls[0]["energy_range"] == 42.5
    assert engine._vina.energy_calls[0]["energy_range"] == 42.5


def test_docking_uses_the_configured_box_and_seed(receptor, fake_vina):
    """The box, seed and exhaustiveness reach the Vina API unchanged."""
    engine = _engine(receptor, seed=123, exhaustiveness=16)
    engine.dock_pdbqt_poses(_prepared_pdbqt())

    instance = engine._vina
    assert instance.seed == 123
    assert instance.maps["center"] == [10.0, 12.0, 8.0]
    assert instance.maps["box_size"] == [22.0, 20.0, 22.0]
    assert instance.dock_calls[0]["exhaustiveness"] == 16


def test_receptor_and_maps_are_prepared_only_once(receptor, fake_vina):
    """The receptor setup is the expensive part and must not be repeated."""
    engine = _engine(receptor)
    engine.dock_pdbqt_poses(_prepared_pdbqt())
    engine.dock_pdbqt_poses(_prepared_pdbqt())

    instance = engine._vina
    assert instance.set_receptor_calls == 1
    assert instance.compute_maps_calls == 1
    assert len(instance.dock_calls) == 2


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_non_string_pdbqt_is_reported(receptor, fake_vina):
    """A PDBQT that is not a string cannot be docked, and is not guessed at."""
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError):
        engine.dock_pdbqt_poses(12345)


def test_invalid_energy_range_is_rejected(receptor, fake_vina):
    """A non-numeric energy range is a programming error."""
    engine = _engine(receptor)
    with pytest.raises((ValueError, TypeError)):
        engine.dock_pdbqt_poses(_prepared_pdbqt(), energy_range="not-a-number")


def test_missing_meeko_package_is_reported(monkeypatch, receptor):
    """Without Meeko the poses cannot be reconstructed, and are not faked."""
    monkeypatch.setattr(vina_docking, "_PDBQTMolecule", None)
    monkeypatch.setattr(vina_docking, "_RDKitMolCreate", None)
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="Meeko is not installed"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


# ---------------------------------------------------------------------------
# Docking failures
# ---------------------------------------------------------------------------


def test_docking_failure_raises_and_returns_no_poses(receptor, fake_vina):
    """A failed docking must raise -- never return a fabricated pose."""
    fake_vina.fail_on_dock = RuntimeError("vina crashed")
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="Vina docking failed"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


def test_receptor_setup_failure_is_reported(receptor, fake_vina):
    """A receptor Vina cannot load must be reported clearly."""
    fake_vina.fail_on_setup = RuntimeError("bad receptor file")
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="could not prepare the Vina"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


def test_no_poses_raises_instead_of_returning_an_empty_list(
    receptor, fake_vina
):
    """An empty result set is a failure, not an empty success."""
    fake_vina.affinities = ()
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="no poses"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


def test_non_finite_affinities_raise(receptor, fake_vina):
    """Non-finite affinities are rejected rather than passed on."""
    fake_vina.affinities = (float("nan"), float("inf"))
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="no finite affinity"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


def test_mismatched_pose_count_is_reported(receptor, fake_vina):
    """Poses that cannot be matched to their scores are an error.

    Silently zipping a shorter list of poses against a longer list of
    affinities would report the wrong score for a pose.
    """
    engine = _engine(receptor)
    fake_vina.poses_pdbqt = _pose_block(_prepared_pdbqt(), 1, -3.5)

    with pytest.raises(VinaDockingError, match="could not be matched"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


def test_unreadable_pose_is_reported(receptor, fake_vina):
    """A pose that cannot be reconstructed must not yield a fake molecule."""
    # One readable pose and one empty one, so that the pose count matches
    # the number of affinities and the reconstruction itself is what fails.
    fake_vina.affinities = (-3.5, -7.25)
    fake_vina.poses_pdbqt = (
        _pose_block(_prepared_pdbqt(), 1, -3.5)
        + "MODEL 2\n"
        "REMARK VINA RESULT:  -7.250      0.000      0.000\n"
        "ENDMDL\n"
    )
    engine = _engine(receptor)
    with pytest.raises(VinaDockingError, match="could not reconstruct"):
        engine.dock_pdbqt_poses(_prepared_pdbqt())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
