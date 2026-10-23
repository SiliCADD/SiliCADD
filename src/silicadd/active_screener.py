"""
Active Learning Virtual Screening (ALVS) benchmark.

This module implements an iterative active learning workflow for virtual
screening of a molecular library:

1. Featurization -- Morgan fingerprints (radius 2, 2048 bits) via RDKit.
2. Surrogate model -- ``RandomForestRegressor`` trained on successfully docked
   molecules; per-tree predictions provide the uncertainty estimate.
3. Active loop -- dock a random seed batch, (re)train the surrogate, select the
   next batch of undocked molecules with the chosen strategy, dock it, repeat
   until the docking budget is exhausted or no candidates remain.

Three selection strategies are supported:

* ``random``      -- uniformly sample from the undocked candidates.
* ``greedy``      -- take the candidates with the best (most negative) predicted
  score.
* ``uncertainty`` -- rank by an acquisition score that balances predicted
  affinity against model uncertainty, weighted by ``exploration_weight``.

Score convention
----------------
Docking scores follow the usual convention where **lower (more negative) is
better** (binding energy / affinity). The same convention is used for
``predicted_score`` so that rankings are directly comparable across all three
strategies. Every ranking in this module -- strategy selection, global top-k
construction and recovery accounting -- uses the same direction.

Uncertainty heuristic
---------------------
``prediction_uncertainty`` is the standard deviation of the individual tree
predictions of the random forest, i.e. the disagreement between the trees.
This is an **uncertainty heuristic**, not a calibrated predictive uncertainty:
it is uncalibrated, unquantified, and not a probability. It must not be
interpreted as a confidence interval or a likelihood.

Effort accounting
-----------------
Three distinct quantities are tracked, and are never conflated:

``docking_attempts``
    Number of molecules selected for docking. Each attempt consumes one unit
    of the docking budget, whether it came from the cache, failed, or
    succeeded.
``true_docking_calls``
    Number of *docking function invocations* -- the real computational cost.
    Failed attempts count (the work was done); cache hits do not (no work was
    done).
``successful_dockings``
    Number of molecules for which a finite score was obtained, from either a
    fresh docking or the cache.

Robustness
----------
Invalid SMILES, docking exceptions, non-finite scores, duplicate molecules and
empty candidate pools are all handled without aborting the run. A molecule is
only ever used as a *training label* when the docking function returned a
finite score for it; predicted scores are never used as true labels.

Molecular identity
------------------
Every molecule is identified by its RDKit canonical SMILES. Reference scores,
the candidate library, duplicate detection, cache lookups and top-k membership
all use the canonical form, so two different SMILES strings for the same
molecule can never be counted as two molecules.

Evaluation oracle
-----------------
``benchmark_strategies`` establishes the global top-k by exhaustive docking (or
from a supplied reference dataset). The reference must cover the *complete*
eligible library; when it does not, the global top-k recovery metrics are
reported as unavailable rather than being computed from a partial set. Those
reference scores are used for evaluation only and are never passed to a
selection strategy.

Interface changes
-----------------
This version adds fields and parameters; nothing was removed.

* ``ActiveScreener`` and the ``run_active_loop`` function accept
  ``docking_config``, which identifies the docking setup the scores belong to.
  A ``cache_path`` now requires it.
* The per-molecule result DataFrame gained ``canonical_smiles`` and
  ``docking_call`` columns. ``docking_call`` is the index of the docking
  function invocation that produced the score, and is ``NaN`` when the score
  came from the cache.
* The benchmark DataFrame gained ``docking_attempts``,
  ``successful_dockings``, ``recovery_metrics_valid``,
  ``reference_complete``, ``reference_source``, ``reference_status`` and
  ``top_k_size``. The previous ``top_k_measurable`` and ``top_k_note`` fields
  are retained as aliases.
* ``DockingCache`` uses schema version 2.0, which keys entries by docking
  configuration as well as canonical SMILES. Version 1.0 files are reported and
  ignored.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors

from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

# Cache schema version -- bump when the on-disk layout changes.
#
# Version 2.0 keys every entry by (docking configuration, canonical SMILES).
# Version 1.0 files have no configuration identity and are therefore never
# reused: they are reported and ignored rather than silently accepted.
CACHE_VERSION = "2.0"
LEGACY_CACHE_VERSIONS = ("1.0",)

# Default number of trees in the surrogate model.
DEFAULT_N_ESTIMATORS = 100


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return the canonical SMILES string for ``smiles``.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES, or ``None`` if the input cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def featurize_smiles(
    smiles_list: Sequence[str],
    radius: int = 2,
    nbits: int = 2048,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[str]], List[bool]]:
    """Featurize SMILES strings with Morgan fingerprints.

    Invalid SMILES are handled gracefully: the corresponding entries are
    ``None`` / ``False`` instead of raising.

    Args:
        smiles_list: SMILES strings to featurize.
        radius: Morgan fingerprint radius (default 2).
        nbits: Number of fingerprint bits (default 2048).

    Returns:
        Tuple of:

        * ``fingerprints`` -- one ``np.ndarray`` of shape ``(nbits,)`` per input
          SMILES, ``None`` where the SMILES was invalid.
        * ``canonical`` -- canonical SMILES per input, ``None`` where invalid.
        * ``valid`` -- validity flag per input SMILES.

    Example:
        >>> fps, canonical, valid = featurize_smiles(["CCO", "nonsense"])
        >>> valid
        [True, False]
        >>> fps[0].shape
        (2048,)
    """
    fingerprints: List[Optional[np.ndarray]] = []
    canonical: List[Optional[str]] = []
    valid: List[bool] = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("Invalid SMILES encountered: %s", smiles)
            fingerprints.append(None)
            canonical.append(None)
            valid.append(False)
            continue

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nbits
        )
        np_fp = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, np_fp)

        fingerprints.append(np_fp)
        canonical.append(Chem.MolToSmiles(mol))
        valid.append(True)

    return fingerprints, canonical, valid


def stack_fingerprints(
    fingerprints: Sequence[Optional[np.ndarray]],
) -> np.ndarray:
    """Stack a sequence of fingerprints into a 2D feature matrix.

    Rows with ``None`` fingerprints are dropped, so the returned matrix may be
    shorter than the input sequence.
    """
    rows = [fp for fp in fingerprints if fp is not None]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(rows)


#: Validation status of a molecule that was docked successfully.
STATUS_OK = "ok"
#: Validation status of a molecule whose docking was attempted but failed.
STATUS_FAILED = "failed"
#: Validation status of a SMILES string that could not be parsed by RDKit.
STATUS_INVALID = "invalid"
#: Validation status of a SMILES string that duplicates an earlier molecule.
STATUS_DUPLICATE = "duplicate"
#: Validation status of a valid molecule that was never selected for docking.
STATUS_NOT_DOCKED = "not_docked"


@dataclass
class DockingRecord:
    """Tracking record for a single molecule in the screening run.

    Attributes:
        smiles: SMILES string as supplied by the caller.
        canonical_smiles: Canonical SMILES; the molecule's identity and the
            cache key. Equals ``smiles`` for molecules that could not be
            parsed.
        fingerprint: Morgan fingerprint, ``None`` for invalid molecules.
        predicted_score: Score predicted by the surrogate model. Populated for
            every featurizable molecule once a model exists, including
            molecules that were never docked. ``NaN`` when no model was
            available.
        true_score: Actual docking score. ``NaN`` when the molecule was never
            docked or the docking failed.
        prediction_uncertainty: Standard deviation across the individual trees
            of the surrogate model -- an uncertainty *heuristic*, not a
            calibrated uncertainty. ``NaN`` when no model was available.
        iteration: Active learning iteration in which the molecule was docked.
        strategy: Strategy that selected the molecule.
        validation_status: One of ``ok`` (docked successfully), ``failed``
            (docking attempted and failed), ``invalid`` (SMILES could not be
            parsed), ``duplicate`` (same molecule as an earlier entry) or
            ``not_docked`` (valid molecule that was never selected).
        docking_call: 1-based index of the docking function invocation that
            produced ``true_score``. ``None`` when the score came from the
            cache (no invocation was performed).
        attempt_index: 1-based position of this molecule in the sequence of
            docking attempts, regardless of outcome.
    """

    smiles: str
    canonical_smiles: str
    fingerprint: Optional[np.ndarray] = None
    predicted_score: float = math.nan
    true_score: float = math.nan
    prediction_uncertainty: float = math.nan
    iteration: Optional[int] = None
    strategy: Optional[str] = None
    validation_status: str = STATUS_NOT_DOCKED
    docking_call: Optional[int] = None
    attempt_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the public (DataFrame-facing) representation of the record."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "predicted_score": self.predicted_score,
            "true_score": self.true_score,
            "prediction_uncertainty": self.prediction_uncertainty,
            "iteration": self.iteration,
            "strategy": self.strategy,
            "validation_status": self.validation_status,
            "docking_call": self.docking_call,
        }


@dataclass
class EligibleLibrary:
    """The set of molecules that may be docked, and their identities.

    A molecule is *eligible* when its SMILES parses and it is the first
    occurrence of its canonical form within the library. Invalid SMILES are
    excluded entirely; duplicate representations of an already-seen molecule
    are excluded so that a molecule is never counted -- or docked -- twice.

    Attributes:
        smiles: Original SMILES of each eligible molecule, in library order.
        canonical: Canonical SMILES of each eligible molecule, parallel to
            ``smiles``.
        invalid_count: Number of SMILES that could not be parsed.
        duplicate_count: Number of SMILES that duplicated an earlier molecule.
    """

    smiles: List[str] = field(default_factory=list)
    canonical: List[str] = field(default_factory=list)
    invalid_count: int = 0
    duplicate_count: int = 0

    @property
    def canonical_set(self) -> Set[str]:
        """The set of canonical SMILES of the eligible molecules."""
        return set(self.canonical)

    def __len__(self) -> int:
        return len(self.smiles)


def build_eligible_library(smiles_list: Sequence[str]) -> EligibleLibrary:
    """Build the eligible library: valid, deduplicated molecules in order.

    Args:
        smiles_list: The raw molecular library.

    Returns:
        An :class:`EligibleLibrary` describing the dockable molecules.
    """
    library = EligibleLibrary()
    seen: Set[str] = set()

    for smiles in smiles_list:
        canonical = canonical_smiles(smiles)
        if canonical is None:
            library.invalid_count += 1
            continue
        if canonical in seen:
            library.duplicate_count += 1
            continue
        seen.add(canonical)
        library.smiles.append(smiles)
        library.canonical.append(canonical)

    return library


def compute_docking_config_id(settings: Mapping[str, Any]) -> str:
    """Compute a stable identifier for a docking configuration.

    The identifier is a truncated SHA-256 digest of the canonical JSON
    serialisation of ``settings``. It is stable across processes and runs, so
    it can be used to partition a persistent docking cache by configuration.

    A docking configuration should describe every setting that materially
    affects the score. The recommended keys are:

    ``target``
        Identifier of the protein / receptor (e.g. PDB id and chain).
    ``engine``
        Docking engine name and version (e.g. ``"vina"``, ``"vina-1.2.5"``).
    ``scoring``
        Scoring function used (e.g. ``"vina"``, ``"ad4"``).
    ``exhaustiveness``
        Number of search runs performed by the engine.
    ``ligand_prep``
        Ligand preparation settings (protonation state, tautomer handling,
        charge model, ...).
    ``protein_prep``
        Protein preparation settings (protonation state, missing residues,
        waters, ...).
    ``box``
        Search box definition (centre and size), if it is not fixed by the
        target.

    Args:
        settings: Mapping of configuration key to value.

    Returns:
        A 32 character hexadecimal identifier.

    Raises:
        ValueError: If ``settings`` is empty.

    Example:
        >>> a = compute_docking_config_id({"engine": "vina", "exhaustiveness": 8})
        >>> b = compute_docking_config_id({"exhaustiveness": 8, "engine": "vina"})
        >>> a == b
        True
    """
    if not settings:
        raise ValueError(
            "docking configuration settings must not be empty; provide the "
            "settings that materially affect the docking score"
        )

    payload = json.dumps(
        _json_safe(settings), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _json_safe(value: Any) -> Any:
    """Convert ``value`` into a JSON-serialisable, order-stable structure."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class DockingCache:
    """Cache of true docking scores, keyed by canonical SMILES *and* configuration.

    A docking score is only meaningful for a specific docking setup: the same
    ligand against a different target, with a different engine version or a
    different protonation model, is a different physical quantity. The cache
    therefore keys every entry by ``(config_id, canonical_smiles)`` so that
    scores can never leak between incompatible configurations.

    The persistent file format (version 2.0) is::

        {
          "version": "2.0",
          "configurations": {
            "<config_id>": {
              "<canonical smiles>": {"score": -8.1, "timestamp": 1710000000.0}
            }
          }
        }

    Legacy version 1.0 files have no configuration identity. They are reported
    and ignored, never silently interpreted as configuration-compatible.

    Attributes:
        path: Optional path of the JSON file backing the cache.
        config_id: Identifier of the docking configuration these entries
            belong to. ``None`` disables persistence entirely.
        data: Mapping of canonical SMILES to cached payload for ``config_id``.
    """

    path: Optional[Union[str, Path]] = None
    config_id: Optional[str] = None
    data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    version: str = CACHE_VERSION
    _loaded: bool = False

    def load(self) -> None:
        """Load cached entries for this configuration from disk.

        Cache files with an incompatible schema version, and files that hold
        no entries for this configuration, are reported and skipped. No legacy
        entry is ever reused.
        """
        if self._loaded:
            return
        self._loaded = True

        if not self.path:
            return

        path = Path(self.path)
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read cache file %s: %s", path, exc)
            return

        if not isinstance(payload, dict):
            logger.warning("Ignoring cache %s: expected a JSON object", path)
            return

        version = payload.get("version")
        if version != self.version:
            if version in LEGACY_CACHE_VERSIONS:
                logger.warning(
                    "Ignoring legacy cache %s (version %s): its entries carry no "
                    "docking-configuration identity and cannot be safely "
                    "reused. Re-dock to rebuild the cache.",
                    path,
                    version,
                )
            else:
                logger.warning(
                    "Ignoring cache %s: version %s does not match %s",
                    path,
                    version,
                    self.version,
                )
            return

        configurations = payload.get("configurations")
        if not isinstance(configurations, dict):
            logger.warning("Ignoring cache %s: malformed 'configurations'", path)
            return

        if self.config_id is None:
            logger.warning(
                "Ignoring cache %s: no docking configuration id was supplied, so "
                "its entries cannot be matched to a setup",
                path,
            )
            return

        section = configurations.get(self.config_id)
        if not isinstance(section, dict):
            logger.info(
                "Cache %s holds no entries for configuration %s",
                path,
                self.config_id,
            )
            return

        for smiles, entry in section.items():
            if not isinstance(entry, dict) or "score" not in entry:
                logger.debug("Skipping malformed cache entry for %s", smiles)
                continue
            score = entry["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                logger.debug("Skipping non-numeric cached score for %s", smiles)
                continue
            if not math.isfinite(score):
                logger.debug("Skipping non-finite cached score for %s", smiles)
                continue
            self.data[smiles] = entry

        logger.info(
            "Loaded %d cached docking results for configuration %s",
            len(self.data),
            self.config_id,
        )

    def get(self, canonical_smiles: str) -> Optional[float]:
        """Return the cached score for ``canonical_smiles``, if any."""
        if not self._loaded:
            self.load()
        entry = self.data.get(canonical_smiles)
        if entry is None:
            return None
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        if not math.isfinite(score):
            return None
        return float(score)

    def set(self, canonical_smiles: str, score: float) -> None:
        """Store a successful docking score for ``canonical_smiles``."""
        if not self._loaded:
            self.load()
        self.data[canonical_smiles] = {
            "score": float(score),
            "timestamp": time.time(),
        }

    def save(self) -> None:
        """Persist the cache to disk atomically.

        Existing entries for other configurations are preserved: the file is
        rewritten with this configuration's section merged into the existing
        ``configurations`` mapping.
        """
        if not self.path or self.config_id is None:
            return

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        configurations: Dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if (
                    isinstance(payload, dict)
                    and payload.get("version") == self.version
                    and isinstance(payload.get("configurations"), dict)
                ):
                    configurations = dict(payload["configurations"])
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Could not merge into existing cache %s: %s", path, exc
                )

        configurations[self.config_id] = self.data
        payload = {"version": self.version, "configurations": configurations}

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)


def predict_with_uncertainty(
    model: RandomForestRegressor, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with a random forest and estimate uncertainty from the trees.

    The uncertainty is the standard deviation of the individual tree
    predictions, i.e. the ensemble disagreement.

    This is an **uncertainty heuristic**, not a calibrated predictive
    uncertainty. It is uncalibrated, it carries no probabilistic guarantee,
    and it must not be read as a confidence interval or a likelihood. It is
    used only to rank candidates against each other.

    Args:
        model: Fitted ``RandomForestRegressor``.
        X: Feature matrix of shape ``(n_samples, n_features)``.

    Returns:
        Tuple ``(mean_predictions, uncertainties)``.
    """
    if not hasattr(model, "estimators_"):
        raise ValueError("Model has no 'estimators_' attribute")

    tree_predictions = np.vstack(
        [tree.predict(X) for tree in model.estimators_]
    )
    return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


def _minmax_normalise(values: np.ndarray) -> np.ndarray:
    """Scale ``values`` into ``[0, 1]``; a constant vector maps to zeros."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    low = float(finite.min())
    high = float(finite.max())
    if math.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    normalised = (values - low) / (high - low)
    return np.nan_to_num(normalised, nan=0.0)


class ActiveScreener:
    """Iterative active learning virtual screener.

    The screener runs a docking loop in which a surrogate model progressively
    learns which molecules are worth docking:

    1. A random seed batch is docked.
    2. The surrogate model is trained on every successfully docked molecule.
    3. A batch of previously undocked molecules is selected with the configured
       strategy.
    4. The batch is docked, the model is retrained, and the loop repeats until
       the docking budget is exhausted or no candidates remain.

    Args:
        smiles_list: Molecular library as SMILES strings.
        seed_size: Number of molecules docked in the initial random seed batch.
        batch_size: Number of molecules selected per active learning iteration.
        docking_budget: Maximum number of docking operations.
        strategy: One of ``random``, ``greedy`` or ``uncertainty``.
        docking_function: Callable taking a SMILES string and returning a
            floating point score (lower is better).
        random_state: Seed controlling the random selections, so that runs are
            reproducible.
        exploration_weight: Weight of the uncertainty term in the
            ``uncertainty`` acquisition score. ``0.0`` is pure exploitation,
            ``1.0`` is pure exploration.
        cache_path: Optional path to a JSON file caching true docking results.
            Requires ``docking_config`` to be set: a persistent cache is keyed
            by docking configuration, because a score is only meaningful for
            the setup that produced it.
        docking_config: Identity of the docking setup, either a stable string
            id or a mapping of settings (hashed into a stable id). See
            :func:`compute_docking_config_id` for the recommended settings.
        n_estimators: Number of trees in the surrogate random forest.
        verbose: Whether to log progress.

    Raises:
        ValueError: If any argument is invalid, or if ``cache_path`` is given
            without a ``docking_config``.

    Example:
        >>> smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        >>> screener = ActiveScreener(
        ...     smiles_list=smiles,
        ...     seed_size=1,
        ...     batch_size=1,
        ...     docking_budget=2,
        ...     strategy="greedy",
        ...     docking_function=lambda s: -float(len(s)),
        ...     random_state=42,
        ... )
        >>> results = screener.run_active_loop()
        >>> sorted(results.columns)
        ['iteration', 'predicted_score', 'prediction_uncertainty', 'smiles', 'strategy', 'true_score', 'validation_status']
    """

    VALID_STRATEGIES = ("random", "greedy", "uncertainty")

    def __init__(
        self,
        smiles_list: Sequence[str],
        seed_size: int,
        batch_size: int,
        docking_budget: int,
        strategy: str,
        docking_function: Callable[[str], float],
        random_state: int = 42,
        exploration_weight: float = 0.5,
        cache_path: Optional[Union[str, Path]] = None,
        docking_config: Optional[Union[str, Mapping[str, Any]]] = None,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        verbose: bool = False,
    ) -> None:
        if not smiles_list:
            raise ValueError("smiles_list must not be empty")
        if seed_size <= 0:
            raise ValueError(f"seed_size must be positive, got {seed_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if docking_budget <= 0:
            raise ValueError(
                f"docking_budget must be positive, got {docking_budget}"
            )
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.VALID_STRATEGIES}, got {strategy!r}"
            )
        if not 0.0 <= exploration_weight <= 1.0:
            raise ValueError(
                "exploration_weight must lie in [0.0, 1.0], got "
                f"{exploration_weight}"
            )
        if docking_budget < seed_size:
            raise ValueError(
                f"docking_budget ({docking_budget}) must be at least seed_size "
                f"({seed_size})"
            )

        # The docking configuration identifies the setup a score belongs to.
        # Without it a persistent cache could hand a score from one setup to
        # another, so persistence is refused rather than silently unsafe.
        if docking_config is None:
            config_id = None
        elif isinstance(docking_config, str):
            config_id = docking_config
            if not config_id:
                raise ValueError("docking_config must not be an empty string")
        elif isinstance(docking_config, Mapping):
            config_id = compute_docking_config_id(docking_config)
        else:
            raise ValueError(
                "docking_config must be a string id or a mapping of settings, "
                f"got {type(docking_config).__name__}"
            )

        if cache_path is not None and config_id is None:
            raise ValueError(
                "cache_path requires a docking_config: a persistent cache is "
                "keyed by docking configuration, and reusing scores across "
                "incompatible setups (different target, engine version, "
                "protonation model, ...) would be invalid. Pass "
                "docking_config=<id or mapping of settings>, or drop "
                "cache_path to run without persistence."
            )

        self.smiles_list = list(smiles_list)
        self.seed_size = seed_size
        self.batch_size = batch_size
        self.docking_budget = docking_budget
        self.strategy = strategy
        self.docking_function = docking_function
        self.random_state = random_state
        self.exploration_weight = exploration_weight
        self.n_estimators = n_estimators
        self.verbose = verbose

        self.rng = random.Random(random_state)
        self.config_id = config_id
        self.cache = DockingCache(path=cache_path, config_id=config_id)
        self.cache.load()

        # Featurize the library, dropping duplicate molecules and invalid ones.
        self.records: List[DockingRecord] = []
        self._build_records()

        # Indices of records that may be docked (valid and not duplicates).
        self.candidate_indices: List[int] = [
            index
            for index, record in enumerate(self.records)
            if record.validation_status == STATUS_NOT_DOCKED
        ]
        self.remaining_indices: List[int] = list(self.candidate_indices)

        # The seed batch cannot be filled if the library holds too few valid,
        # unique molecules.
        if len(self.candidate_indices) < seed_size:
            raise ValueError(
                f"Not enough valid, unique SMILES ({len(self.candidate_indices)}) "
                f"for the requested seed size ({seed_size})"
            )

        self.model: Optional[RandomForestRegressor] = None

        # Effort accounting -- three distinct quantities, never conflated.
        self.docking_calls = 0      # docking attempts (budget slots consumed)
        self.true_docking_calls = 0  # docking function invocations (real cost)
        self.successful_dockings = 0  # finite scores obtained
        self.iteration = 0
        self.history: List[Dict[str, Any]] = []
        # Ordered log of every docking attempt, in the order they happened.
        self.attempt_log: List[Dict[str, Any]] = []

    # -- setup -----------------------------------------------------------

    def _build_records(self) -> None:
        """Create one record per library entry, flagging invalid and duplicate ones.

        Identity is the canonical SMILES. A duplicate representation of an
        already-seen molecule keeps a record (so the caller can see it) but is
        never docked and never enters the candidate pool, so a molecule is
        never counted twice.
        """
        fingerprints, canonical, valid = featurize_smiles(self.smiles_list)
        seen: Set[str] = set()

        for smiles, fp, canon, is_valid in zip(
            self.smiles_list, fingerprints, canonical, valid
        ):
            if not is_valid:
                self.records.append(
                    DockingRecord(
                        smiles=smiles,
                        canonical_smiles=smiles,
                        fingerprint=None,
                        validation_status=STATUS_INVALID,
                    )
                )
                continue

            assert canon is not None
            if canon in seen:
                # Duplicate molecule: the first occurrence is docked, this one
                # is recorded as a duplicate and never docked.
                logger.debug("Duplicate molecule skipped: %s", smiles)
                self.records.append(
                    DockingRecord(
                        smiles=smiles,
                        canonical_smiles=canon,
                        fingerprint=fp,
                        validation_status=STATUS_DUPLICATE,
                    )
                )
                continue

            seen.add(canon)
            self.records.append(
                DockingRecord(
                    smiles=smiles,
                    canonical_smiles=canon,
                    fingerprint=fp,
                    validation_status=STATUS_NOT_DOCKED,
                )
            )

    # -- surrogate model -------------------------------------------------

    def train_surrogate_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestRegressor:
        """Train the random forest surrogate on true docking scores.

        Only true docking scores are ever used as training labels; predicted
        scores are never used as labels.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: True docking scores, shape ``(n_samples,)``.

        Returns:
            The fitted model.

        Raises:
            ValueError: If the shapes are inconsistent or the sample count is
                too small to fit a forest.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1D, got shape {y_train.shape}")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train has {X_train.shape[0]} rows but y_train has "
                f"{y_train.shape[0]} entries"
            )
        if X_train.shape[0] < 2:
            raise ValueError(
                "At least 2 docked molecules are required to train the "
                f"surrogate model, got {X_train.shape[0]}"
            )

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.model = model
        return model

    def _retrain_surrogate_model(self) -> bool:
        """Retrain the surrogate on every successfully docked molecule.

        Returns:
            ``True`` if a model was trained, ``False`` if there was not enough
            data yet.
        """
        X_train: List[np.ndarray] = []
        y_train: List[float] = []

        for record in self.records:
            if (
                record.validation_status == STATUS_OK
                and record.fingerprint is not None
                and math.isfinite(record.true_score)
            ):
                X_train.append(record.fingerprint)
                y_train.append(record.true_score)

        if len(X_train) < 2:
            logger.debug(
                "Not enough successful dockings (%d) to train the model",
                len(X_train),
            )
            self.model = None
            return False

        self.train_surrogate_model(np.vstack(X_train), np.asarray(y_train))
        logger.debug("Retrained surrogate model on %d molecules", len(X_train))
        return True

    def _predict_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict scores and uncertainties for every record.

        Returns:
            Tuple ``(predictions, uncertainties)`` aligned with
            ``self.records``. Both are ``NaN`` where no model is available.
        """
        n = len(self.records)
        if self.model is None:
            nan_vec = np.full(n, np.nan, dtype=float)
            return nan_vec, nan_vec.copy()

        X = np.vstack(
            [
                record.fingerprint
                for record in self.records
                if record.fingerprint is not None
            ]
        )
        predictions, uncertainties = predict_with_uncertainty(self.model, X)

        # Scatter the predictions back onto the full record list.
        full_predictions = np.full(n, np.nan, dtype=float)
        full_uncertainties = np.full(n, np.nan, dtype=float)
        position = 0
        for index, record in enumerate(self.records):
            if record.fingerprint is not None:
                full_predictions[index] = predictions[position]
                full_uncertainties[index] = uncertainties[position]
                position += 1

        return full_predictions, full_uncertainties

    # -- selection strategies --------------------------------------------

    def _select_batch_random(self, pool: Sequence[int]) -> List[int]:
        """Select a batch uniformly at random from ``pool``."""
        if not pool:
            return []
        size = min(self.batch_size, len(pool))
        return self.rng.sample(list(pool), size)

    def _select_batch_greedy(
        self, pool: Sequence[int], predictions: np.ndarray
    ) -> List[int]:
        """Select the candidates with the best predicted scores.

        Lower (more negative) predictions are better, so the candidates are
        sorted ascending by predicted score.
        """
        if not pool:
            return []
        # Candidates whose prediction is not finite cannot be ranked, so they
        # are pushed to the end deterministically rather than compared with
        # NaN (which would make the ordering undefined).
        ranked = sorted(
            pool,
            key=lambda index: (
                not math.isfinite(predictions[index]),
                predictions[index] if math.isfinite(predictions[index]) else 0.0,
                index,
            ),
        )
        return list(ranked[: min(self.batch_size, len(pool))])

    def _select_batch_uncertainty(
        self,
        pool: Sequence[int],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> List[int]:
        """Select candidates by an acquisition score.

        The acquisition score combines two min-max normalised terms:

        * exploitation -- ``-predicted_score`` (higher is better, since lower
          scores are better);
        * exploration -- the prediction uncertainty (higher is better).

        The two are blended with ``exploration_weight``.
        """
        if not pool:
            return []

        pool_predictions = predictions[list(pool)]
        pool_uncertainties = uncertainties[list(pool)]

        # A candidate with a non-finite prediction or uncertainty cannot be
        # scored, so it is ranked last rather than compared with NaN.
        usable = np.isfinite(pool_predictions) & np.isfinite(pool_uncertainties)

        exploitation = np.zeros(len(pool), dtype=float)
        exploration = np.zeros(len(pool), dtype=float)
        if usable.any():
            exploitation[usable] = _minmax_normalise(-pool_predictions[usable])
            exploration[usable] = _minmax_normalise(pool_uncertainties[usable])

        acquisition = (
            (1.0 - self.exploration_weight) * exploitation
            + self.exploration_weight * exploration
        )
        acquisition[~usable] = -np.inf

        # Sort by descending acquisition; the index breaks ties deterministically.
        ranked = sorted(
            range(len(pool)),
            key=lambda position: (-acquisition[position], pool[position]),
        )
        return [pool[position] for position in ranked[: min(self.batch_size, len(pool))]]

    def _select_batch(
        self,
        pool: Sequence[int],
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> List[int]:
        """Dispatch to the configured selection strategy."""
        if self.strategy == "random":
            return self._select_batch_random(pool)
        if self.strategy == "greedy":
            return self._select_batch_greedy(pool, predictions)
        return self._select_batch_uncertainty(pool, predictions, uncertainties)

    # -- docking ---------------------------------------------------------

    def _dock(self, record: DockingRecord) -> Tuple[Optional[float], str, bool]:
        """Dock a single molecule, using the cache when possible.

        Returns:
            Tuple ``(score, status, cached)`` where ``score`` is ``None`` for
            failures, ``status`` is ``ok`` or ``failed``, and ``cached`` is
            ``True`` when the score came from the cache and therefore cost no
            docking function invocation.
        """
        cached = self.cache.get(record.canonical_smiles)
        if cached is not None:
            logger.debug("Cache hit for %s", record.canonical_smiles)
            return cached, STATUS_OK, True

        # A real docking operation: it costs an invocation even if it fails.
        self.true_docking_calls += 1

        try:
            score = self.docking_function(record.smiles)
        except Exception as exc:  # noqa: BLE001 - docking failures are expected
            logger.warning(
                "Docking failed for %s: %s", record.smiles, exc
            )
            return None, STATUS_FAILED, False

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            logger.warning(
                "Docking returned a non-numeric score for %s: %r",
                record.smiles,
                score,
            )
            return None, STATUS_FAILED, False

        score = float(score)
        if not math.isfinite(score):
            logger.warning(
                "Docking returned a non-finite score for %s: %r",
                record.smiles,
                score,
            )
            return None, STATUS_FAILED, False

        self.cache.set(record.canonical_smiles, score)
        return score, STATUS_OK, False

    # -- main loop -------------------------------------------------------

    def run_active_loop(self) -> pd.DataFrame:
        """Run the active learning loop and return the per-molecule results.

        The loop docks a random seed batch, trains the surrogate model, selects
        and docks the next batch, retrains, and repeats until the docking
        budget is exhausted or no candidates remain.

        Returns:
            DataFrame with one row per molecule in the library and the columns
            ``smiles``, ``predicted_score``, ``true_score``,
            ``prediction_uncertainty``, ``iteration``, ``strategy`` and
            ``validation_status``. Undocked compounds have
            ``true_score = NaN``.
        """
        self.rng.seed(self.random_state)
        self.docking_calls = 0
        self.true_docking_calls = 0
        self.successful_dockings = 0
        self.iteration = 0
        self.history = []
        self.attempt_log = []

        # --- seed phase ------------------------------------------------
        seed_pool = list(self.candidate_indices)
        size = min(self.seed_size, len(seed_pool), self.docking_budget)
        seed_indices = self.rng.sample(seed_pool, size) if seed_pool else []

        for index in seed_indices:
            self._attempt(index, iteration=0)

        self._retrain_surrogate_model()
        self._record_history(iteration=0, selected=seed_indices)

        # --- active loop -----------------------------------------------
        while self.remaining_indices and self.docking_calls < self.docking_budget:
            self.iteration += 1

            predictions, uncertainties = self._predict_candidates()
            selected = self._select_batch(
                self.remaining_indices, predictions, uncertainties
            )

            if not selected:
                logger.info("No candidates selected; ending the loop")
                break

            attempts_before = self.docking_calls

            for index in selected:
                if self.docking_calls >= self.docking_budget:
                    logger.info(
                        "Docking budget exhausted (%d/%d)",
                        self.docking_calls,
                        self.docking_budget,
                    )
                    break

                self._attempt(index, iteration=self.iteration)

            self._retrain_surrogate_model()
            self._record_history(iteration=self.iteration, selected=selected)

            # Guard against a stalled loop: every iteration must either dock
            # something or terminate.
            if self.docking_calls == attempts_before:
                logger.info(
                    "No docking progress in iteration %d; ending the loop",
                    self.iteration,
                )
                break

        # Fill in predictions and uncertainties for every featurizable record
        # using the final model. These are model estimates, never true labels.
        # Valid molecules that were never docked still receive a prediction --
        # that is the whole point of the surrogate -- while ``true_score``
        # stays NaN. Molecules that could not be featurized keep NaN.
        predictions, uncertainties = self._predict_candidates()
        for index, record in enumerate(self.records):
            if record.fingerprint is None:
                # Not featurizable: no prediction can be made.
                continue
            record.predicted_score = float(predictions[index])
            record.prediction_uncertainty = float(uncertainties[index])

        self.cache.save()
        return self.get_results()

    def _attempt(self, index: int, iteration: int) -> None:
        """Dock one molecule and record the attempt.

        The attempt consumes one unit of docking budget. The attempt is
        appended to :attr:`attempt_log` in the order it happened, together
        with the number of docking function invocations performed so far
        (``effort``), so that recovery metrics can be computed against the
        real order rather than the iteration order.
        """
        record = self.records[index]

        score, status, cached = self._dock(record)

        self.docking_calls += 1
        if status == STATUS_OK:
            self.successful_dockings += 1

        record.true_score = score if score is not None else math.nan
        record.iteration = iteration
        record.strategy = self.strategy
        record.validation_status = status
        record.docking_call = None if cached else self.true_docking_calls
        record.attempt_index = self.docking_calls

        if index in self.remaining_indices:
            self.remaining_indices.remove(index)

        self.attempt_log.append(
            {
                "attempt_index": self.docking_calls,
                "record_index": index,
                "smiles": record.smiles,
                "canonical_smiles": record.canonical_smiles,
                "iteration": iteration,
                "status": status,
                "cached": cached,
                "docking_call": record.docking_call,
                "effort": self.true_docking_calls,
            }
        )

        if self.verbose:
            logger.info(
                "Iteration %d dock %s -> %s (status=%s, cached=%s)",
                iteration,
                record.smiles,
                record.true_score,
                status,
                cached,
            )

    def _record_history(
        self, iteration: int, selected: Sequence[int]
    ) -> None:
        """Append a per-iteration progress entry to the run history."""
        docked_scores = [
            record.true_score
            for record in self.records
            if record.validation_status == STATUS_OK
            and math.isfinite(record.true_score)
        ]
        self.history.append(
            {
                "iteration": iteration,
                "selected": len(selected),
                "cumulative_budget_used": self.docking_calls,
                "docking_function_calls": self.true_docking_calls,
                "successful_dockings": self.successful_dockings,
                "best_true_score": min(docked_scores) if docked_scores else math.nan,
                "n_docked_ok": len(docked_scores),
                "n_remaining": len(self.remaining_indices),
            }
        )

    # -- outputs ---------------------------------------------------------

    def get_results(self) -> pd.DataFrame:
        """Return the per-molecule results as a DataFrame.

        The columns are:

        * ``smiles`` -- the SMILES as supplied by the caller.
        * ``canonical_smiles`` -- the molecule's identity.
        * ``predicted_score`` -- the surrogate model's estimate. Populated for
          every featurizable molecule once a model exists, including
          molecules that were never docked.
        * ``true_score`` -- the actual docking score; ``NaN`` for molecules
          that were never docked or whose docking failed.
        * ``prediction_uncertainty`` -- the tree-disagreement uncertainty
          heuristic; ``NaN`` when no model was available.
        * ``iteration`` -- the iteration in which the molecule was docked.
        * ``strategy`` -- the strategy that selected it.
        * ``validation_status`` -- ``ok``, ``failed``, ``invalid``,
          ``duplicate`` or ``not_docked``.
        * ``docking_call`` -- the index of the docking function invocation
          that produced ``true_score``; ``NaN`` for a cached score.
        """
        frame = pd.DataFrame([record.to_dict() for record in self.records])
        column_order = [
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
        frame = frame.reindex(columns=column_order)

        # ``docking_call`` is an integer index that is missing for cached
        # scores and for molecules that were never docked. A nullable integer
        # keeps the column numeric and represents the missing values as NA.
        frame["docking_call"] = frame["docking_call"].astype("Int64")
        frame["iteration"] = frame["iteration"].astype("Int64")

        return frame

    def get_attempt_log(self) -> pd.DataFrame:
        """Return the ordered log of docking attempts.

        One row per attempt, in the order the attempts happened. The
        ``effort`` column is the number of docking function invocations
        performed up to and including that attempt, which is the unit in
        which recovery effort is measured.
        """
        columns = [
            "attempt_index",
            "record_index",
            "smiles",
            "canonical_smiles",
            "iteration",
            "status",
            "cached",
            "docking_call",
            "effort",
        ]
        return pd.DataFrame(self.attempt_log, columns=columns)

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics for the run."""
        statuses = pd.Series(
            [record.validation_status for record in self.records]
        ).value_counts()
        return {
            "total_molecules": len(self.smiles_list),
            "unique_molecules": len(self.candidate_indices),
            "invalid_molecules": int(statuses.get(STATUS_INVALID, 0)),
            "duplicate_molecules": int(statuses.get(STATUS_DUPLICATE, 0)),
            "docked_ok": int(statuses.get(STATUS_OK, 0)),
            "docking_failures": int(statuses.get(STATUS_FAILED, 0)),
            "not_docked": int(statuses.get(STATUS_NOT_DOCKED, 0)),
            # Docking attempts: budget slots consumed.
            "docking_calls": self.docking_calls,
            "docking_attempts": self.docking_calls,
            # Docking function invocations: the real computational cost.
            "true_docking_calls": self.true_docking_calls,
            # Molecules for which a finite score was obtained.
            "successful_dockings": self.successful_dockings,
            "docking_budget": self.docking_budget,
            "iterations": self.iteration,
            "strategy": self.strategy,
            "cache_config_id": self.config_id,
        }

    def get_history(self) -> pd.DataFrame:
        """Return the per-iteration progress history as a DataFrame."""
        return pd.DataFrame(self.history)


def run_active_loop(
    smiles_list: Sequence[str],
    seed_size: int,
    batch_size: int,
    docking_budget: int,
    strategy: str,
    docking_function: Callable[[str], float],
    random_state: int = 42,
    exploration_weight: float = 0.5,
    cache_path: Optional[Union[str, Path]] = None,
    docking_config: Optional[Union[str, Mapping[str, Any]]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run an active learning virtual screening loop.

    Convenience wrapper around :class:`ActiveScreener` for one-shot use.

    Args:
        smiles_list: Molecular library as SMILES strings.
        seed_size: Number of molecules in the initial random seed batch.
        batch_size: Number of molecules selected per iteration.
        docking_budget: Maximum number of docking operations.
        strategy: ``random``, ``greedy`` or ``uncertainty``.
        docking_function: Callable mapping a SMILES string to a score.
        random_state: Seed for reproducible runs.
        exploration_weight: Uncertainty weight in the acquisition score.
        cache_path: Optional path to a JSON docking-result cache. Requires
            ``docking_config`` to be set.
        docking_config: Identity of the docking setup, either a stable string
            id or a mapping of settings (hashed into an id). Required when
            ``cache_path`` is given, because scores are only meaningful for a
            specific setup.
        n_estimators: Number of trees in the surrogate model.
        verbose: Whether to log progress.

    Returns:
        DataFrame with the columns ``smiles``, ``canonical_smiles``,
        ``predicted_score``, ``true_score``, ``prediction_uncertainty``,
        ``iteration``, ``strategy``, ``validation_status`` and
        ``docking_call``.

    Example:
        >>> results = run_active_loop(
        ...     ["CCO", "c1ccccc1"],
        ...     seed_size=1,
        ...     batch_size=1,
        ...     docking_budget=2,
        ...     strategy="random",
        ...     docking_function=lambda s: -1.0,
        ... )
        >>> len(results)
        2
    """
    screener = ActiveScreener(
        smiles_list=smiles_list,
        seed_size=seed_size,
        batch_size=batch_size,
        docking_budget=docking_budget,
        strategy=strategy,
        docking_function=docking_function,
        random_state=random_state,
        exploration_weight=exploration_weight,
        cache_path=cache_path,
        docking_config=docking_config,
        n_estimators=n_estimators,
        verbose=verbose,
    )
    return screener.run_active_loop()


@dataclass
class TopKReference:
    """The global top-k ground truth used for evaluation.

    Attributes:
        available: Whether a global top-k could be established at all.
        complete: Whether the reference covers the *complete* eligible
            library. A partial reference cannot define a global top-k.
        recovery_metrics_valid: Whether the top-k recovery metrics
            (``recall_top_k``, ``fraction_recovered`` and the calls-to-recovery
            figures) may be reported. This is only true for a complete
            reference.
        source: ``"reference"``, ``"exhaustive"`` or ``"none"``.
        top_k: The ``(canonical_smiles, score)`` pairs of the global top-k,
            ordered best (most negative) first.
        status: Human readable explanation of what happened.
        covered: Number of eligible library molecules with a reference score.
        eligible: Number of eligible library molecules.
    """

    available: bool = False
    complete: bool = False
    recovery_metrics_valid: bool = False
    source: str = "none"
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    status: str = "no reference was established"
    covered: int = 0
    eligible: int = 0

    @property
    def top_k_smiles(self) -> Set[str]:
        """The canonical SMILES of the global top-k molecules."""
        return {smiles for smiles, _ in self.top_k}

    @property
    def top_k_size(self) -> int:
        """The number of molecules in the global top-k."""
        return len(self.top_k)


def canonicalize_reference_scores(
    reference_scores: Mapping[str, float]
) -> Dict[str, float]:
    """Canonicalize the keys of a reference score mapping.

    Two different SMILES strings may describe the same molecule, so reference
    scores are keyed by canonical SMILES. If two entries canonicalize to the
    same molecule but carry different scores, that is a genuine conflict in
    the reference data and is reported as an error rather than silently
    resolved.

    Args:
        reference_scores: Mapping of SMILES to true docking score.

    Returns:
        Mapping of canonical SMILES to score.

    Raises:
        ValueError: If two reference entries canonicalize to the same molecule
            with conflicting scores, or if a score is not a finite number.
    """
    canonical_scores: Dict[str, float] = {}

    for smiles, score in reference_scores.items():
        canonical = canonical_smiles(smiles)
        if canonical is None:
            raise ValueError(
                f"reference dataset contains an invalid SMILES that cannot be "
                f"canonicalised: {smiles!r}"
            )

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(
                f"reference score for {smiles!r} is not a number: {score!r}"
            )

        score = float(score)
        if not math.isfinite(score):
            raise ValueError(
                f"reference score for {smiles!r} is not finite: {score!r}"
            )

        if canonical in canonical_scores:
            if not math.isclose(canonical_scores[canonical], score):
                raise ValueError(
                    "conflicting reference scores for the same molecule: "
                    f"{canonical!r} has both "
                    f"{canonical_scores[canonical]!r} and {score!r}. The "
                    "reference dataset is ambiguous and cannot be used."
                )
            # Identical score from two representations: not a conflict.
            continue

        canonical_scores[canonical] = score

    return canonical_scores


def _establish_global_top_k(
    smiles_list: Sequence[str],
    k: int,
    docking_function: Callable[[str], float],
    reference_scores: Optional[Mapping[str, float]] = None,
) -> TopKReference:
    """Establish the global top-k ground truth used for evaluation.

    Either an independently generated reference dataset is used, or the whole
    eligible library is docked exhaustively.

    The reference must cover the *complete* eligible library -- the valid,
    deduplicated molecules of the library. A reference that covers only part
    of the library does not define a global top-k: the uncovered molecules
    might have scored better than the covered ones. In that case the top-k is
    reported as unavailable rather than computed from the partial set.

    Args:
        smiles_list: The molecular library.
        k: Size of the top-k set.
        docking_function: Callable mapping a SMILES string to a score.
        reference_scores: Optional mapping of SMILES to true scores, generated
            independently of this benchmark run.

    Returns:
        A :class:`TopKReference` describing the ground truth and whether the
        recovery metrics are valid.

    Raises:
        ValueError: If the reference dataset contains invalid SMILES,
            non-finite scores, or conflicting scores for the same molecule.
    """
    library = build_eligible_library(smiles_list)

    if not library.smiles:
        return TopKReference(
            status=(
                "the library contains no valid, unique molecules, so a global "
                "top-k cannot be established"
            ),
            eligible=0,
        )

    if reference_scores is not None:
        canonical_scores = canonicalize_reference_scores(reference_scores)

        missing = [
            canonical
            for canonical in library.canonical
            if canonical not in canonical_scores
        ]
        covered = len(library.canonical) - len(missing)

        if missing:
            return TopKReference(
                available=False,
                complete=False,
                recovery_metrics_valid=False,
                source="reference",
                status=(
                    f"the reference dataset covers {covered} of "
                    f"{len(library.canonical)} eligible library molecules, so "
                    "it does not define a global top-k and top-k recovery "
                    "cannot be measured"
                ),
                covered=covered,
                eligible=len(library.canonical),
            )

        # Restrict to the eligible library: a global top-k is a statement
        # about the library, not about molecules outside it.
        scores = {
            canonical: canonical_scores[canonical]
            for canonical in library.canonical
        }
        ordered = sorted(scores.items(), key=lambda item: item[1])
        return TopKReference(
            available=True,
            complete=True,
            recovery_metrics_valid=True,
            source="reference",
            top_k=ordered[:k],
            status=(
                "the reference dataset covers the complete eligible library "
                f"({len(library.canonical)} molecules)"
            ),
            covered=covered,
            eligible=len(library.canonical),
        )

    # Exhaustive docking of the entire eligible library.
    exhaustive: Dict[str, float] = {}
    failures: List[str] = []

    for smiles, canonical in zip(library.smiles, library.canonical):
        try:
            score = docking_function(smiles)
        except Exception as exc:  # noqa: BLE001 - docking failures are expected
            logger.warning("Reference docking failed for %s: %s", smiles, exc)
            failures.append(smiles)
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            failures.append(smiles)
            continue
        score = float(score)
        if not math.isfinite(score):
            failures.append(smiles)
            continue
        exhaustive[canonical] = score

    if failures:
        return TopKReference(
            available=False,
            complete=False,
            recovery_metrics_valid=False,
            source="exhaustive",
            status=(
                f"exhaustive reference docking failed for {len(failures)} of "
                f"{len(library.smiles)} eligible molecules, so it does not "
                "define a global top-k and top-k recovery cannot be measured"
            ),
            covered=len(exhaustive),
            eligible=len(library.smiles),
        )

    ordered = sorted(exhaustive.items(), key=lambda item: item[1])
    return TopKReference(
        available=True,
        complete=True,
        recovery_metrics_valid=True,
        source="exhaustive",
        top_k=ordered[:k],
        status=(
            "the whole eligible library was docked exhaustively "
            f"({len(library.smiles)} molecules)"
        ),
        covered=len(exhaustive),
        eligible=len(library.smiles),
    )


def _calls_to_recover_fraction(
    attempts: Sequence[Tuple[str, int]],
    top_k_smiles: Set[str],
    fraction: float,
) -> Optional[int]:
    """Docking effort needed to recover ``fraction`` of the global top-k.

    Args:
        attempts: The *successful* docking attempts in the order they
            happened, as ``(canonical_smiles, effort)`` pairs. ``effort`` is the
            number of docking function invocations performed up to and
            including that attempt, so failed attempts that happened earlier
            are still charged to the effort.
        top_k_smiles: The canonical SMILES of the global top-k molecules.
        fraction: Target fraction, e.g. ``0.5``.

    Returns:
        The docking effort at which the target was first reached, or ``None``
        when the target was never reached.
    """
    if not top_k_smiles:
        return None

    target = math.ceil(fraction * len(top_k_smiles))
    if target <= 0:
        return 0

    recovered = 0
    for canonical, effort in attempts:
        if canonical in top_k_smiles:
            recovered += 1
            if recovered >= target:
                return int(effort)
    return None


def benchmark_strategies(
    smiles_list: Sequence[str],
    k: int,
    seed: int = 42,
    strategies: Sequence[str] = ("random", "greedy", "uncertainty"),
    budgets: Optional[Sequence[int]] = None,
    docking_function: Optional[Callable[[str], float]] = None,
    reference_scores: Optional[Mapping[str, float]] = None,
    exploration_weight: float = 0.5,
    seed_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare active learning strategies on the same library.

    Each strategy is run with the same random seed over a range of docking
    budgets, and the results are compared against the global top-k ground
    truth.

    The ground truth is established either from ``reference_scores`` (an
    independently generated reference dataset) or by exhaustively docking the
    whole eligible library. It is used **only** for evaluation: the scores are
    never passed to a selection strategy, and the strategies never see which
    molecules are in the top-k. The benchmark deliberately runs without a
    persistent cache, so no strategy can inherit reference scores through the
    cache.

    Validity of the recovery metrics
    --------------------------------
    A global top-k can only be established when the reference covers the
    **complete eligible library** (the valid, deduplicated molecules). When it
    does not, the recovery metrics are reported as unavailable -- they are
    ``NaN`` and ``recovery_metrics_valid`` is ``False`` -- rather than being
    computed from a partial set, which would be misleading. Runtime, budget
    usage and successful docking counts are reported either way.

    Args:
        smiles_list: Molecular library as SMILES strings.
        k: Size of the global top-k set.
        seed: Random seed, used identically for every strategy so the
            comparison is fair and reproducible.
        strategies: Strategies to compare.
        budgets: Docking budgets to evaluate. Defaults to a spread of values
            between 10% and 100% of the library size.
        docking_function: Callable mapping a SMILES string to a score.
        reference_scores: Optional independently generated reference scores.
        exploration_weight: Uncertainty weight for the ``uncertainty``
            strategy.
        seed_size: Seed batch size. Defaults to ``max(1, 10% of library)``.
        batch_size: Batch size. Defaults to ``seed_size``.
        verbose: Whether to log progress.

    Returns:
        DataFrame with one row per (strategy, budget) pair and the columns:

        * ``strategy``
        * ``budget``
        * ``docking_attempts`` -- molecules selected for docking (budget slots
          consumed)
        * ``true_docking_calls`` -- docking function invocations, i.e. the
          real computational effort. Failed attempts count; cache hits do not.
        * ``successful_dockings`` -- molecules for which a finite score was
          obtained
        * ``cumulative_budget_used`` -- budget slots consumed (same as
          ``docking_attempts``)
        * ``best_true_score`` -- best (most negative) true score found
        * ``recall_top_k`` -- fraction of the global top-k recovered
        * ``fraction_recovered`` -- fraction of docked molecules that are
          top-k members
        * ``docking_calls_to_50pct`` / ``_80pct`` / ``_100pct`` -- docking
          effort required to recover 50% / 80% / 100% of the global top-k.
          ``NaN`` when the target was not reached, or when the recovery
          metrics are invalid.
        * ``runtime_seconds``
        * ``recovery_metrics_valid`` -- whether the top-k recovery metrics
          are meaningful for this run
        * ``reference_complete`` -- whether the reference covered the complete
          eligible library
        * ``reference_source`` -- ``"reference"``, ``"exhaustive"`` or
          ``"none"``
        * ``reference_status`` -- human readable explanation of the reference
          coverage
        * ``top_k_size`` -- size of the global top-k that was used
        * ``top_k_measurable`` -- retained alias of
          ``recovery_metrics_valid``
        * ``top_k_note`` -- retained alias of ``reference_status``

    Raises:
        ValueError: If the arguments are invalid, or if the reference dataset
            is ambiguous (invalid SMILES, non-finite scores, or conflicting
            scores for the same canonical molecule).

    Example:
        >>> results = benchmark_strategies(
        ...     smiles_list=["CCO", "c1ccccc1", "CC(=O)O", "C1CCCCC1"],
        ...     k=2,
        ...     seed=42,
        ...     strategies=["random", "greedy"],
        ...     budgets=[2, 3],
        ...     docking_function=lambda s: -float(len(s)),
        ...     verbose=False,
        ... )
        >>> len(results)
        4
    """
    if not smiles_list:
        raise ValueError("smiles_list must not be empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not strategies:
        raise ValueError("strategies must not be empty")
    for strategy in strategies:
        if strategy not in ActiveScreener.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected one of "
                f"{ActiveScreener.VALID_STRATEGIES}"
            )
    if not 0.0 <= exploration_weight <= 1.0:
        raise ValueError(
            "exploration_weight must lie in [0.0, 1.0], got "
            f"{exploration_weight}"
        )

    if docking_function is None:
        raise ValueError(
            "docking_function is required to establish the global top-k "
            "ground truth"
        )

    library_size = len(smiles_list)
    if budgets is None:
        budgets = sorted(
            {max(1, library_size * fraction // 10) for fraction in range(1, 11)}
        )
    budgets = [int(budget) for budget in budgets]
    if not budgets:
        raise ValueError("budgets must not be empty")

    if seed_size is None:
        seed_size = max(1, min(library_size, library_size // 10))
    if batch_size is None:
        batch_size = seed_size

    reference = _establish_global_top_k(
        smiles_list, k, docking_function, reference_scores
    )

    if reference.recovery_metrics_valid:
        top_k_smiles = reference.top_k_smiles
        logger.info(
            "Global top-%d (%s): %s",
            reference.top_k_size,
            reference.source,
            [smiles for smiles, _ in reference.top_k],
        )
    else:
        top_k_smiles = set()
        logger.warning(
            "Global top-k ground truth unavailable: %s. Top-k recovery "
            "metrics will be reported as unavailable.",
            reference.status,
        )

    rows: List[Dict[str, Any]] = []

    for strategy in strategies:
        for budget in budgets:
            # The seed batch must fit inside the budget.
            effective_seed = min(seed_size, budget)
            effective_batch = min(batch_size, max(1, budget))

            # No cache_path: the benchmark must never let a strategy inherit
            # reference scores through the cache.
            screener = ActiveScreener(
                smiles_list=smiles_list,
                seed_size=effective_seed,
                batch_size=effective_batch,
                docking_budget=budget,
                strategy=strategy,
                docking_function=docking_function,
                random_state=seed,
                exploration_weight=exploration_weight,
                verbose=False,
            )

            start = time.time()
            results = screener.run_active_loop()
            runtime = time.time() - start

            docked = results[results["validation_status"] == STATUS_OK]
            true_scores = docked["true_score"].dropna().tolist()
            best_true_score = min(true_scores) if true_scores else math.nan

            # The real order of docking attempts, taken from the screener's
            # attempt log rather than from the iteration column: several
            # molecules share an iteration, and sorting by iteration alone
            # would not preserve the sequence in which they were docked.
            attempt_log = screener.get_attempt_log()
            attempts = [
                (row.canonical_smiles, int(row.effort), row.status)
                for row in attempt_log.itertuples()
            ]

            # Only successful attempts recover a top-k molecule; a failed
            # attempt produces no score and therefore recovers nothing, but it
            # still consumed real docking effort.
            successful_attempts = [
                (canonical, effort)
                for canonical, effort, status in attempts
                if status == STATUS_OK
            ]

            recovered = sum(
                1 for canonical, _ in successful_attempts
                if canonical in top_k_smiles
            )
            successful = len(docked)

            if top_k_smiles:
                recall_top_k = recovered / len(top_k_smiles)
                fraction_recovered = (
                    recovered / successful if successful else math.nan
                )
                calls_50 = _calls_to_recover_fraction(
                    successful_attempts, top_k_smiles, 0.50
                )
                calls_80 = _calls_to_recover_fraction(
                    successful_attempts, top_k_smiles, 0.80
                )
                calls_100 = _calls_to_recover_fraction(
                    successful_attempts, top_k_smiles, 1.00
                )
            else:
                # The recovery metrics are not valid without a complete
                # reference; report them as unavailable rather than as zero.
                recall_top_k = math.nan
                fraction_recovered = math.nan
                calls_50 = calls_80 = calls_100 = None

            rows.append(
                {
                    "strategy": strategy,
                    "budget": budget,
                    "docking_attempts": int(screener.docking_calls),
                    "true_docking_calls": int(screener.true_docking_calls),
                    "successful_dockings": int(screener.successful_dockings),
                    "cumulative_budget_used": int(screener.docking_calls),
                    "best_true_score": best_true_score,
                    "recall_top_k": recall_top_k,
                    "fraction_recovered": fraction_recovered,
                    "docking_calls_to_50pct": (
                        calls_50 if calls_50 is not None else math.nan
                    ),
                    "docking_calls_to_80pct": (
                        calls_80 if calls_80 is not None else math.nan
                    ),
                    "docking_calls_to_100pct": (
                        calls_100 if calls_100 is not None else math.nan
                    ),
                    "runtime_seconds": runtime,
                    "recovery_metrics_valid": bool(reference.recovery_metrics_valid),
                    "reference_complete": bool(reference.complete),
                    "reference_source": reference.source,
                    "reference_status": reference.status,
                    "top_k_size": reference.top_k_size,
                    # Retained from the previous interface: these are aliases
                    # of the clearer fields above.
                    "top_k_measurable": bool(reference.recovery_metrics_valid),
                    "top_k_note": reference.status,
                }
            )

            if verbose:
                logger.info(
                    "%s @ budget %d: %d attempts, %d invocations, best=%.4f, "
                    "recall=%.3f",
                    strategy,
                    budget,
                    screener.docking_calls,
                    screener.true_docking_calls,
                    best_true_score,
                    recall_top_k,
                )

    return pd.DataFrame(rows)
