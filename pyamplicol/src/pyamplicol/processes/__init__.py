# SPDX-License-Identifier: 0BSD
"""Model-neutral process syntax and canonical process IR.

Built-in-SM catalogue enumeration and selection live under
``pyamplicol.models.builtin`` and are intentionally not re-exported here.
"""

from .core_syntax import (
    OrderTuple,
    ParticleName,
    ProcessTuple,
    canonical_process_key,
    expand_process_variants,
    split_process_set,
)

__all__ = [
    "OrderTuple",
    "ParticleName",
    "ProcessTuple",
    "canonical_process_key",
    "expand_process_variants",
    "split_process_set",
]
