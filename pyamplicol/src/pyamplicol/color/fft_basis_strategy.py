# SPDX-License-Identifier: 0BSD
"""Choose an exact FFT colour representation without inventing amplitude identities.

DDM half-ladders apply only when the model supplies their algebraic certificate.
Fundamental chains are a different representation, already natural for quark
amplitudes.  Products of those chains must not be renamed DDM: their coefficients
are colour-flow amplitudes, not the distinct-flavour Melia/JO primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..processes.ir import CanonicalProcessIR


@dataclass(frozen=True)
class FFTBasisStrategy:
    """Selected metric and its truthful mathematical description.

    ``actual_basis`` selects the existing metric implementation; ``name`` states
    the colour tensors used.  ``tensor_count`` counts the complete tensor list,
    not its rank at a particular Nc or its kinematically nonzero amplitudes.
    It excludes the construction-only traversal aliases of open-line products.
    """

    actual_basis: Literal["trace", "adjoint"]
    name: Literal[
        "ddm", "fundamental-chain", "fundamental-chain-products", "singlet", "trace"
    ]
    reason: str
    permutation_degree: int
    tensor_count: int | None


def best_general_fft_basis(
    process: CanonicalProcessIR,
    *,
    ddm_proven: bool,
) -> FFTBasisStrategy:
    """Select a certified reduction, otherwise retain the exact existing basis.

    This inexpensive decision uses only canonical colour roles and an already
    computed model certificate.  It deliberately does not infer tree-level
    Kleiss--Kuijf identities from external adjoint particles alone.  Colourless
    external particles do not change the permutation group or the tensor count.
    """

    gluons = len(process.adjoint_labels)
    fundamentals = len(process.fundamental_labels)
    antifundamentals = len(process.antifundamental_labels)
    if fundamentals != antifundamentals:
        return FFTBasisStrategy(
            "trace",
            "trace",
            "No balanced fundamental-chain domain; ordinary colour validation applies.",
            gluons,
            None,
        )
    if fundamentals:
        # Allocate a permutation of m adjoints to k ordered open strings, then
        # choose the k! antifundamental routings.  Different block traversals
        # are construction aliases, not additional colour tensors.
        tensor_count = (
            math.factorial(fundamentals)
            * math.factorial(gluons)
            * math.comb(gluons + fundamentals - 1, fundamentals - 1)
        )
        return FFTBasisStrategy(
            "trace",
            "fundamental-chain" if fundamentals == 1 else "fundamental-chain-products",
            (
                "One open fundamental chain already fixes both quark endpoints; "
                "there is no trace-to-DDM ordering reduction."
                if fundamentals == 1
                else "Retain exact products of fundamental chains and all endpoint "
                "routings; distinct-flavour DDM generalizations are not identities "
                "of arbitrary colour-flow amplitudes."
            ),
            gluons,
            tensor_count,
        )
    if not gluons:
        return FFTBasisStrategy(
            "trace",
            "singlet",
            "The colour-singlet domain has one trivial colour tensor.",
            0,
            1,
        )
    if ddm_proven and gluons >= 2:
        return FFTBasisStrategy(
            "adjoint",
            "ddm",
            (
                "The certified two-adjoint metric is the single delta tensor."
                if gluons == 2
                else "The model certifies the connected structure-constant "
                "tree identities."
            ),
            gluons - 2,
            math.factorial(gluons - 2),
        )
    return FFTBasisStrategy(
        "trace",
        "trace",
        (
            "A single adjoint leg leaves no two-anchor DDM domain."
            if gluons < 2
            else "No DDM amplitude identity is certified; retain the exact trace basis."
        ),
        max(0, gluons - 1),
        math.factorial(max(0, gluons - 1)),
    )
