"""
AutoDock Vina docking engine for the active learning virtual screening loop.

This module turns a SMILES string into a real Vina affinity in kcal/mol, in
the form the active learning loop expects: a callable that takes a SMILES
string and returns a floating point score where **lower (more negative) is
better**. A Vina affinity is already a binding energy in kcal/mol and is
negative for favourable binding, so it is used unchanged -- no rescaling and
no sign flip.

Pipeline
--------
For every molecule:

1. **Parse** the SMILES with RDKit.
2. **Add hydrogens** and **generate a 3D conformer** (ETKDGv3, seeded so the
   conformer is reproducible).
3. **Prepare the ligand** with Meeko, which writes the PDBQT string Vina
   expects.
4. **Dock** the ligand against the user-supplied receptor PDBQT with the
   official Vina Python API.
5. **Return** the best (most negative) finite affinity over the poses.

Poses as well as scores
------------------------
:meth:`VinaDockingEngine.dock_pdbqt_poses` is the pose-level counterpart of
:meth:`VinaDockingEngine.dock_pdbqt`. Where that method collapses a run to a
single affinity, this one keeps the geometry of every pose, so a prediction
can be compared against an experimental structure -- for example to measure
the RMSD of a redocked pose against the co-crystallised ligand. The poses
come back as RDKit molecules whose atom order matches the molecule the ligand
was prepared from, so two poses can be compared by substructure matching
without renumbering by hand.

Nothing is invented
-------------------
The engine never guesses a target, a binding site or a docking box. The
receptor PDBQT path, the box centre and the box dimensions are all required
arguments. The engine also never substitutes a synthetic score when something
goes wrong: every failure raises :class:`VinaDockingError`. That matters for
the active learning loop, which treats a raised exception as a *failed*
docking (recorded with ``validation_status == "failed"`` and a ``NaN`` true
score) rather than as a molecule with a plausible-looking but fabricated
affinity.

Cost
----
Preparing the receptor and computing the Vina affinity maps are the expensive
parts of a screening campaign, and both depend only on the receptor and the
box -- not on the ligand. The engine therefore sets up the receptor and the
maps **once** and reuses them for every molecule. Call :meth:`warm_up` to
force that setup up front, so that a bad receptor is reported before a long
campaign starts rather than on its first molecule.

Example
-------
>>> from silicadd import VinaDockingEngine, run_active_loop
>>> engine = VinaDockingEngine(          # doctest: +SKIP
...     receptor_pdbqt="prepared_receptor.pdbqt",
...     box_center=(12.5, 4.0, -3.2),
...     box_size=(22.0, 20.0, 22.0),
...     exhaustiveness=8,
...     seed=42,
... )
>>> results = run_active_loop(           # doctest: +SKIP
...     smiles_list=["CCO", "c1ccccc1", "CC(=O)O"],
...     seed_size=1,
...     batch_size=1,
...     docking_budget=2,
...     strategy="uncertainty",
...     docking_function=engine,
...     docking_config=engine.config_settings(),
... )
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# The Vina and Meeko Python APIs are optional at import time: this module must
# stay importable (and unit-testable) without them, and the failure has to be
# reported when docking is actually attempted rather than when the package is
# imported.
try:  # pragma: no cover - exercised only where the dependency is absent
    from vina import Vina as _Vina
except ImportError:  # pragma: no cover
    _Vina = None

try:  # pragma: no cover - exercised only where the dependency is absent
    from meeko import MoleculePreparation as _MoleculePreparation
    from meeko import PDBQTWriterLegacy as _PDBQTWriterLegacy
except ImportError:  # pragma: no cover
    _MoleculePreparation = None
    _PDBQTWriterLegacy = None

try:  # pragma: no cover - exercised only where the dependency is absent
    from meeko import PDBQTMolecule as _PDBQTMolecule
    from meeko import RDKitMolCreate as _RDKitMolCreate
except ImportError:  # pragma: no cover
    _PDBQTMolecule = None
    _RDKitMolCreate = None


class VinaDockingError(RuntimeError):
    """Raised when a Vina docking attempt cannot produce a valid affinity.

    Every operational failure of the engine is reported with this exception:
    a missing or unreadable receptor, a SMILES that RDKit cannot parse, a
    ligand that cannot be prepared into PDBQT, and a docking run that fails
    or returns no usable pose.

    The active learning loop catches docking exceptions and records the
    molecule as ``failed`` with a ``NaN`` true score, so raising is the
    correct way to report a failure -- returning a placeholder number would
    silently poison the surrogate model's training labels.
    """


@dataclass(frozen=True)
class DockedPose:
    """A single docked pose returned by :meth:`VinaDockingEngine.dock_pdbqt_poses`.

    Each pose carries its affinity and an RDKit molecule whose coordinates
    are the docked coordinates. The RDKit reconstruction keeps the atom order
    of the molecule the ligand was prepared from, so two poses of the same
    molecule can be compared by substructure matching and RMSD without any
    additional renumbering.
    """

    affinity: float
    mol: Any  # Chem.Mol


def _as_float_triple(values: Sequence[float], name: str) -> Tuple[float, ...]:
    """Validate that ``values`` is a sequence of three finite numbers.

    Args:
        values: The candidate triple.
        name: Parameter name, used in the error message.

    Returns:
        The triple as a tuple of floats.

    Raises:
        ValueError: If ``values`` is not three finite numbers.
    """
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError(f"{name} must be three numbers, got {values!r}")

    try:
        triple = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be three numbers, got {values!r}"
        ) from exc

    if len(triple) != 3:
        raise ValueError(
            f"{name} must be exactly three numbers (x, y, z), got "
            f"{len(triple)}"
        )
    if not all(math.isfinite(value) for value in triple):
        raise ValueError(f"{name} must be finite, got {values!r}")
    return triple


class VinaDockingEngine:
    """A callable AutoDock Vina docking engine.

    The engine is configured once -- receptor, docking box, exhaustiveness and
    seed -- and then used as the ``docking_function`` of an active learning
    run. It is stateful (the receptor and the affinity maps are prepared once
    and reused) and therefore **not** thread-safe: give each thread its own
    engine.

    Args:
        receptor_pdbqt: Path to a **prepared** rigid receptor PDBQT file. The
            receptor must already be protonated and converted to PDBQT (for
            example with ``prepare_receptor`` from AutoDockTools or
            ``meeko``'s receptor tooling). This engine does not prepare
            proteins.
        box_center: Centre of the docking box in Angstrom, ``(x, y, z)``.
            Required: the engine never guesses a binding site.
        box_size: Dimensions of the docking box in Angstrom, ``(a, b, c)``.
            Required.
        exhaustiveness: Number of Monte Carlo runs Vina performs per ligand.
        n_poses: Number of poses generated per ligand and inspected for the
            best affinity.
        seed: Random seed for the Vina search and the conformer generation,
            so that a run is reproducible.
        scoring: Vina scoring function, ``"vina"``, ``"vinardo"`` or
            ``"ad4"``.
        cpu: Number of CPUs Vina may use. ``0`` means all of them.
        spacing: Grid spacing of the affinity maps in Angstrom.
        verbosity: Vina verbosity, ``0`` (silent) to ``2`` (verbose).
        optimize_ligand: Whether to energy-minimise the generated conformer
            with MMFF before docking.
        max_opt_iters: Maximum number of MMFF optimisation steps.

    Raises:
        VinaDockingError: If the receptor file is missing or is not a PDBQT
            file.
        ValueError: If the box parameters or the other options are invalid.

    Example:
        >>> engine = VinaDockingEngine(      # doctest: +SKIP
        ...     receptor_pdbqt="receptor.pdbqt",
        ...     box_center=(10.0, 10.0, 10.0),
        ...     box_size=(20.0, 20.0, 20.0),
        ... )
        >>> affinity = engine("CCO")          # doctest: +SKIP
        >>> affinity <= 0.0                   # doctest: +SKIP
        True
    """

    VALID_SCORING_FUNCTIONS = ("vina", "vinardo", "ad4")

    def __init__(
        self,
        receptor_pdbqt: Union[str, Path],
        box_center: Sequence[float],
        box_size: Sequence[float],
        exhaustiveness: int = 8,
        n_poses: int = 9,
        seed: int = 42,
        scoring: str = "vina",
        cpu: int = 0,
        spacing: float = 0.375,
        verbosity: int = 1,
        optimize_ligand: bool = True,
        max_opt_iters: int = 200,
    ) -> None:
        receptor = Path(receptor_pdbqt)
        self.receptor_pdbqt = receptor

        # The arguments are validated first: a malformed box is a
        # programming error, and reporting it before looking at the file
        # system gives the more useful message.
        self.box_center = _as_float_triple(box_center, "box_center")
        self.box_size = _as_float_triple(box_size, "box_size")
        if any(dimension <= 0.0 for dimension in self.box_size):
            raise ValueError(
                f"box_size must be positive in every dimension, got {box_size!r}"
            )
        if exhaustiveness < 1:
            raise ValueError(
                f"exhaustiveness must be at least 1, got {exhaustiveness}"
            )
        if n_poses < 1:
            raise ValueError(f"n_poses must be at least 1, got {n_poses}")
        if spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        scoring = scoring.lower()
        if scoring not in self.VALID_SCORING_FUNCTIONS:
            raise ValueError(
                f"scoring must be one of {self.VALID_SCORING_FUNCTIONS}, got "
                f"{scoring!r}"
            )

        if not receptor.exists():
            raise VinaDockingError(
                f"receptor PDBQT file not found: {receptor}. The engine docks "
                "against a prepared receptor; prepare the protein first (for "
                "example with AutoDockTools' prepare_receptor) and pass the "
                "resulting .pdbqt path."
            )
        if receptor.suffix.lower() != ".pdbqt":
            raise VinaDockingError(
                f"receptor must be a PDBQT file, got {receptor}. Vina only "
                "accepts PDBQT receptors."
            )

        self.exhaustiveness = int(exhaustiveness)
        self.n_poses = int(n_poses)
        self.seed = int(seed)
        self.scoring = scoring
        self.cpu = int(cpu)
        self.spacing = float(spacing)
        self.verbosity = int(verbosity)
        self.optimize_ligand = bool(optimize_ligand)
        self.max_opt_iters = int(max_opt_iters)

        # The Vina object, the receptor and the affinity maps are prepared
        # lazily and then reused: they depend on the receptor and the box, not
        # on the ligand.
        self._vina: Optional[Any] = None
        self._preparator: Optional[Any] = None
        self._maps_computed = False

    # -- receptor and maps ------------------------------------------------

    def warm_up(self) -> None:
        """Prepare the receptor and the affinity maps ahead of time.

        Both depend only on the receptor and the docking box, so they are
        computed once and reused for every ligand. Calling this explicitly
        moves that cost out of the first docking attempt, so a bad receptor
        or an impossible box is reported before a campaign starts.

        Raises:
            VinaDockingError: If the receptor cannot be prepared or the
                affinity maps cannot be computed.
        """
        self._ensure_vina()

    def _ensure_vina(self) -> Any:
        """Return the Vina object, preparing the receptor and maps once.

        Raises:
            VinaDockingError: If Vina is not installed, or if the receptor
                cannot be prepared or the maps cannot be computed.
        """
        if self._vina is not None:
            return self._vina

        if _Vina is None:
            raise VinaDockingError(
                "the AutoDock Vina Python API is not installed; install the "
                "'vina' package (pip install vina) to run Vina docking"
            )

        try:
            vina = _Vina(
                sf_name=self.scoring,
                cpu=self.cpu,
                seed=self.seed,
                verbosity=self.verbosity,
            )
            vina.set_receptor(rigid_pdbqt_filename=str(self.receptor_pdbqt))
            vina.compute_vina_maps(
                center=list(self.box_center),
                box_size=list(self.box_size),
                spacing=self.spacing,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a docking error
            raise VinaDockingError(
                f"could not prepare the Vina receptor "
                f"{self.receptor_pdbqt} for the docking box centred at "
                f"{self.box_center}: {exc}"
            ) from exc

        self._vina = vina
        self._maps_computed = True
        logger.info(
            "Vina receptor %s prepared for the box centred at %s with size %s",
            self.receptor_pdbqt,
            self.box_center,
            self.box_size,
        )
        return self._vina

    def _ensure_preparator(self) -> Any:
        """Return the Meeko preparator, creating it once and reusing it.

        Raises:
            VinaDockingError: If Meeko is not installed.
        """
        if self._preparator is None:
            if _MoleculePreparation is None:
                raise VinaDockingError(
                    "Meeko is not installed; install the 'meeko' package "
                    "(pip install meeko) to prepare ligands for Vina docking"
                )
            self._preparator = _MoleculePreparation()
        return self._preparator

    # -- ligand preparation -----------------------------------------------

    def prepare_ligand_pdbqt(self, smiles: str) -> str:
        """Turn a SMILES string into a Vina-ready PDBQT ligand string.

        The molecule is parsed with RDKit, hydrogens are added, a 3D conformer
        is generated (seeded, so it is reproducible) and optionally
        energy-minimised, and Meeko writes the PDBQT.

        Args:
            smiles: The molecule as a SMILES string.

        Returns:
            The ligand as a PDBQT string.

        Raises:
            VinaDockingError: If the SMILES cannot be parsed, no conformer
                can be generated, or the ligand cannot be prepared.
        """
        if not isinstance(smiles, str):
            raise VinaDockingError(
                f"expected a SMILES string, got {type(smiles).__name__}"
            )

        preparator = self._ensure_preparator()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise VinaDockingError(
                f"RDKit could not parse the SMILES {smiles!r}"
            )

        try:
            mol = Chem.AddHs(mol)
            self._embed_conformer(mol, smiles)
            if self.optimize_ligand and AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=self.max_opt_iters)

            setups = preparator.prepare(mol)
            if not setups:
                raise VinaDockingError(
                    f"Meeko produced no ligand setup for {smiles!r}"
                )
            pdbqt_string, is_ok, error_msg = _PDBQTWriterLegacy.write_string(
                setups[0]
            )
        except VinaDockingError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a docking error
            raise VinaDockingError(
                f"ligand preparation failed for {smiles!r}: {exc}"
            ) from exc

        if not is_ok:
            raise VinaDockingError(
                f"Meeko could not write the PDBQT for {smiles!r}: {error_msg}"
            )
        if not pdbqt_string.strip():
            raise VinaDockingError(
                f"Meeko wrote an empty PDBQT for {smiles!r}"
            )

        return pdbqt_string

    def _embed_conformer(self, mol: Any, smiles: str) -> None:
        """Generate a 3D conformer for ``mol`` in place.

        ETKDGv3 is tried first; molecules it cannot embed are retried with
        random coordinates, which succeeds for most of them.

        Raises:
            VinaDockingError: If no usable conformer could be generated.
        """
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        params.useRandomCoords = False
        if AllChem.EmbedMolecule(mol, params) >= 0:
            if self._conformer_is_finite(mol):
                return

        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) >= 0:
            if self._conformer_is_finite(mol):
                return

        raise VinaDockingError(
            f"could not generate a 3D conformer for {smiles!r}"
        )

    @staticmethod
    def _conformer_is_finite(mol: Any) -> bool:
        """Whether ``mol`` has a conformer with finite coordinates."""
        if mol.GetNumConformers() == 0:
            return False
        for position in mol.GetConformer().GetPositions():
            if not all(math.isfinite(float(value)) for value in position):
                return False
        return True

    # -- docking -----------------------------------------------------------

    def dock_pdbqt(self, pdbqt_string: str) -> float:
        """Dock a prepared PDBQT ligand and return the best affinity.

        Args:
            pdbqt_string: The ligand as a PDBQT string.

        Returns:
            The best (most negative) finite Vina affinity in kcal/mol.

        Raises:
            VinaDockingError: If the docking run fails or produces no usable
                pose.
        """
        vina = self._ensure_vina()

        try:
            vina.set_ligand_from_string(pdbqt_string)
            vina.dock(
                exhaustiveness=self.exhaustiveness, n_poses=self.n_poses
            )
            energies = np.asarray(vina.energies(n_poses=self.n_poses), dtype=float)
        except Exception as exc:  # noqa: BLE001 - reported as a docking error
            raise VinaDockingError(f"Vina docking failed: {exc}") from exc

        return self._best_affinity(energies)

    def dock_pdbqt_poses(
        self, pdbqt_string: str, energy_range: float = 3.0
    ) -> List[DockedPose]:
        """Dock a prepared PDBQT ligand and return every pose it produced.

        This is the pose-level counterpart of :meth:`dock_pdbqt`: where that
        method collapses the run to a single affinity, this one keeps the
        geometry of each pose so it can be compared against a reference --
        for example to measure the RMSD of a redocked pose against the
        experimentally observed one.

        The poses are returned as RDKit molecules reconstructed from Vina's
        PDBQT output. Meeko's ``REMARK SMILES IDX`` block survives the round
        trip, so the reconstructed molecule keeps the atom order of the
        molecule the ligand was prepared from. Two poses of the same molecule
        can therefore be compared by substructure matching without any
        renumbering by hand.

        Args:
            pdbqt_string: The ligand as a PDBQT string.
            energy_range: Maximum energy difference from the best pose, in
                kcal/mol, for a pose to be reported. The default matches
                Vina's own default and the set of poses
                :meth:`dock_pdbqt` scores; pass a larger value to keep the
                weaker poses as well.

        Returns:
            The docked poses, ordered by Vina's own ranking (best affinity
            first). Each carries its affinity in kcal/mol and its docked
            geometry.

        Raises:
            VinaDockingError: If the docking run fails, produces no usable
                pose, or a pose cannot be reconstructed into an RDKit
                molecule.
            ValueError: If ``energy_range`` is not a positive number.
        """
        try:
            energy_range = float(energy_range)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"energy_range must be a number, got {energy_range!r}"
            ) from exc
        if not math.isfinite(energy_range) or energy_range <= 0.0:
            raise ValueError(
                f"energy_range must be a positive number, got {energy_range!r}"
            )

        if _PDBQTMolecule is None or _RDKitMolCreate is None:
            raise VinaDockingError(
                "Meeko is not installed; install the 'meeko' package "
                "(pip install meeko) to reconstruct the docked poses"
            )

        vina = self._ensure_vina()

        try:
            vina.set_ligand_from_string(pdbqt_string)
            vina.dock(
                exhaustiveness=self.exhaustiveness, n_poses=self.n_poses
            )
            energies = np.asarray(
                vina.energies(n_poses=self.n_poses, energy_range=energy_range),
                dtype=float,
            )
            poses_pdbqt = vina.poses(
                n_poses=self.n_poses, energy_range=energy_range
            )
        except Exception as exc:  # noqa: BLE001 - reported as a docking error
            raise VinaDockingError(f"Vina docking failed: {exc}") from exc

        affinities = self._pose_affinities(energies)
        blocks = self._split_pose_blocks(poses_pdbqt)
        if len(blocks) != len(affinities):
            raise VinaDockingError(
                f"Vina reported {len(affinities)} affinities but {len(blocks)} "
                "poses, so the docked poses could not be matched to their "
                "scores"
            )

        poses: List[DockedPose] = []
        for affinity, block in zip(affinities, blocks):
            poses.append(
                DockedPose(affinity=affinity, mol=self._pose_mol(block))
            )
        return poses

    @staticmethod
    def _pose_affinities(energies: Any) -> List[float]:
        """Extract the per-pose affinities from a Vina energy table.

        Vina returns one row per pose, with the total affinity in the first
        column. The rows are returned in Vina's own ranking order, which is
        the same order the poses come back in.

        Raises:
            VinaDockingError: If there is no pose, or no finite affinity.
        """
        array = np.asarray(energies, dtype=float)
        if array.size == 0:
            raise VinaDockingError(
                "Vina returned no poses, so no affinity could be reported"
            )
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise VinaDockingError(
                "Vina returned no poses, so no affinity could be reported"
            )

        totals = array[:, 0]
        finite = totals[np.isfinite(totals)]
        if finite.size == 0:
            raise VinaDockingError(
                "Vina returned no finite affinity values for the docked poses"
            )
        return [float(value) for value in totals]

    @staticmethod
    def _split_pose_blocks(poses_pdbqt: str) -> List[str]:
        """Split Vina's multi-model PDBQT output into one block per pose.

        Vina writes the poses as a sequence of ``MODEL`` / ``ENDMDL`` blocks.
        Each block is returned separately, so that it can be reconstructed on
        its own.
        """
        blocks: List[str] = []
        current: List[str] = []
        for line in poses_pdbqt.split("\n"):
            if line.startswith("MODEL"):
                current = [line]
                continue
            if line.startswith("ENDMDL"):
                current.append(line)
                blocks.append("\n".join(current))
                current = []
                continue
            if current:
                current.append(line)
        return blocks

    @staticmethod
    def _pose_mol(block: str) -> Any:
        """Reconstruct one Vina pose block into an RDKit molecule.

        Raises:
            VinaDockingError: If the block cannot be read, or if it carries no
                molecule.
        """
        try:
            pdbqt_mol = _PDBQTMolecule(block)
            mols = _RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        except Exception as exc:  # noqa: BLE001 - reported as a docking error
            raise VinaDockingError(
                f"could not reconstruct a docked pose into an RDKit molecule: "
                f"{exc}"
            ) from exc

        if not mols:
            raise VinaDockingError(
                "a docked pose could not be reconstructed into an RDKit "
                "molecule"
            )
        return mols[0]

    @staticmethod
    def _best_affinity(energies: Any) -> float:
        """Extract the best (most negative) finite affinity from ``energies``.

        Vina returns one row per pose, with the total affinity in the first
        column.

        Raises:
            VinaDockingError: If there is no pose, or no finite affinity.
        """
        array = np.asarray(energies, dtype=float)
        if array.size == 0:
            raise VinaDockingError(
                "Vina returned no poses, so no affinity could be reported"
            )
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise VinaDockingError(
                "Vina returned no poses, so no affinity could be reported"
            )

        totals = array[:, 0]
        finite = totals[np.isfinite(totals)]
        if finite.size == 0:
            raise VinaDockingError(
                "Vina returned no finite affinity values for the docked poses"
            )
        return float(finite.min())

    def __call__(self, smiles: str) -> float:
        """Dock one molecule and return its Vina affinity in kcal/mol.

        This is the ``docking_function`` interface of the active learning
        loop: it takes a SMILES string and returns a floating point score
        where lower is better.

        Args:
            smiles: The molecule as a SMILES string.

        Returns:
            The best (most negative) finite Vina affinity in kcal/mol.

        Raises:
            VinaDockingError: If the ligand cannot be prepared or the docking
                fails. The exception propagates to the active learning loop,
                which records the molecule as a failed docking.
        """
        return self.dock_pdbqt(self.prepare_ligand_pdbqt(smiles))

    # -- configuration -----------------------------------------------------

    def config_settings(self) -> Dict[str, Any]:
        """Describe the docking setup, for use as a ``docking_config``.

        The returned mapping covers every setting that materially affects the
        score, which is exactly what
        :func:`silicadd.active_screener.compute_docking_config_id` expects.
        Passing it to the active learning loop makes the docking cache
        configuration-aware, so scores are never reused across setups.

        Returns:
            A mapping describing the docking configuration.
        """
        return {
            "target": str(self.receptor_pdbqt),
            "engine": "vina",
            "scoring": self.scoring,
            "exhaustiveness": self.exhaustiveness,
            "n_poses": self.n_poses,
            "seed": self.seed,
            "box": {
                "center": list(self.box_center),
                "size": list(self.box_size),
            },
            "ligand_prep": {
                "protonation": "rdkit_addhs",
                "conformer": "etkdgv3",
                "optimized": self.optimize_ligand,
            },
        }


def make_vina_docking_function(
    receptor_pdbqt: Union[str, Path],
    box_center: Sequence[float],
    box_size: Sequence[float],
    **kwargs: Any,
) -> Callable[[str], float]:
    """Build a Vina docking function for the active learning loop.

    A convenience wrapper around :class:`VinaDockingEngine` for callers that
    only want a plain callable.

    Args:
        receptor_pdbqt: Path to a prepared rigid receptor PDBQT file.
        box_center: Centre of the docking box in Angstrom, ``(x, y, z)``.
        box_size: Dimensions of the docking box in Angstrom, ``(a, b, c)``.
        **kwargs: Further options, passed to :class:`VinaDockingEngine`.

    Returns:
        A callable mapping a SMILES string to a Vina affinity in kcal/mol.

    Example:
        >>> dock = make_vina_docking_function(   # doctest: +SKIP
        ...     "receptor.pdbqt", (10.0, 10.0, 10.0), (20.0, 20.0, 20.0)
        ... )
        >>> dock("CCO")                          # doctest: +SKIP
        -4.7
    """
    engine = VinaDockingEngine(
        receptor_pdbqt=receptor_pdbqt,
        box_center=box_center,
        box_size=box_size,
        **kwargs,
    )
    return engine.__call__
