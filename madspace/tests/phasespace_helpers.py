"""Sampling and kinematics shared by the phase-space cut tests.

pytest puts this directory on sys.path (rootdir-relative imports, no
__init__.py), so the test modules import it by name.
"""

import torch
from madspace_testing import generator, rand


def sample(mapping, seed, n, conditions=()):
    """Map n uniform points through a PhaseSpaceMapping.

    Returns the momenta and the weight (Jacobian times cuts) as tensors, as the
    mapping gives them: non-finite values are left in, see finite_weight.
    conditions are passed on unchanged, e.g. a permutation index per point.
    """
    r = rand((n, mapping.random_dim()), generator=generator(seed))
    p_ext, _x1, _x2, det = mapping.map_forward([r], list(conditions))
    return p_ext, det


def finite_weight(p, det):
    """The weight with every point whose weight or momenta are not finite set
    to 0, i.e. what an integral over the batch should sum."""
    finite = torch.isfinite(det) & torch.isfinite(p).all(dim=2).all(dim=1)
    return torch.where(finite, det, 0.0)


def invariant_mass(p, *indices):
    """Invariant mass of the sum of the momenta at the given positions."""
    total = sum(p[:, i, :] for i in indices)
    m2 = total[:, 0] ** 2 - torch.sum(total[:, 1:] ** 2, dim=1)
    return torch.sqrt(torch.clamp(m2, min=0.0))
