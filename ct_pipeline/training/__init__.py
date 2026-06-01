"""Public CT training package API.

The training package is split by responsibility:

- ``config`` owns CLI registration and validated runtime defaults.
- ``bootstrap`` owns CT training context construction and Phase 1 field setup.
- ``runtime`` owns CT prediction/loss construction.
- ``objectives`` owns loss dataclasses and mode/schedule predicates.
- ``control`` owns optimizer/checkpoint control helpers used by the train loop.
- ``preview`` owns CT preview rendering and export artifacts.
- ``session`` owns process/session utilities such as NVML and command logging.
- ``mutations`` owns densify/reseed/prune mechanics as a subpackage.
- ``grid_cache`` owns native grid-cache lifecycle.
- ``reporting`` owns metric/report serialization.

Import stable orchestration objects from here; import private helpers from their
own modules only inside tests or focused diagnostics.
"""

from .config import build_parser, validate_ct_training_args
from .grid_cache import CTGridCacheManager
from .bootstrap import CTTrainingBootstrap, prepare_ct_training_bootstrap
from .objectives import CTLossTerms
from .runtime import compute_ct_loss_terms

__all__ = [
    "CTGridCacheManager",
    "CTLossTerms",
    "CTTrainingBootstrap",
    "build_parser",
    "compute_ct_loss_terms",
    "prepare_ct_training_bootstrap",
    "validate_ct_training_args",
]
