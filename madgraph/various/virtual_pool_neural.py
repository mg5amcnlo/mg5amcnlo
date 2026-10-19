"""Optional frozen conditional corrections to pooled virtual polynomials.

Only native sampled/full-branch observations are fitted. Native probabilities
already include helicity multiplicities and are used exactly once in a mean.
SciPy is optional for training and supplies the Python antenna reference
evaluator. Native worker inference is implemented in Fortran.
"""
import hashlib
import itertools
import math
import os
import time
from collections import namedtuple

FORMAT_VERSION = 1
MAX_HEADS = 64
MAX_INPUTS = 128
MAX_WIDTH = 64
MAX_NATIVE_HELICITIES = 65536
MAX_PROBABILITY_ELEMENTS = 16000000
MAX_FEATURE_ELEMENTS = 40000000
MAX_ACTIVATION_ELEMENTS = 12000000
MAX_CONFIGURATIONS = 4
MAX_OPTIMIZER_ITERATIONS = 4000
MIN_HEAD_ROWS = 60
FRACTIONS = (.5, 2./3., .8)
DEFAULT_CONFIGURATIONS = (dict(widths=(64,), ridge=.001, alpha=1., iterations=400),)
LINEAR_CONFIGURATION = dict(family='linear', widths=(1,), ridge=.001, alpha=1., iterations=1000)
MAX_LINEAR_DESIGN_ELEMENTS = 8000000
MAX_DENSE_LINEAR_PARAMETERS = 512
LINEAR_RESIDUAL_L1 = .02
LINEAR_RESIDUAL_RMS = .05
PHYSICS_LINEAR_RESIDUAL_L1 = .005
PHYSICS_LINEAR_RESIDUAL_RMS = .01
_GROUPED_MIN_HEADS = 32
_GROUPED_MAX_DENSITY = .25

_ObjectiveData = namedtuple('_ObjectiveData', 'affine mixture grouped')
_HeadGroups = namedtuple('_HeadGroups', 'single full_rows full_mixing')


def _dot(a, b):
    return a[..., 0]*b[..., 0] - (a[..., 1:]*b[..., 1:]).sum(axis=-1)


def antenna_remainder(momenta, incoming, a, b, c):
    """Real crossed R(u,v) dilogarithms appearing in finite one-loop antennae.

    This is a basis function, not a complete antenna or finite amplitude.
    Only descriptors with three massless coloured legs are constructed.
    """
    import numpy as np
    from scipy.special import spence
    p = np.asarray(momenta, dtype=float)
    crossing = np.where(incoming, -1., 1.)
    total = p[:, np.flatnonzero(incoming)].sum(axis=1)
    shat = _dot(total, total)
    triple = (p[:, a]*crossing[a]+p[:, b]*crossing[b]+p[:, c]*crossing[c])
    denominator = _dot(triple, triple)
    valid = np.abs(denominator) > 1.e-12*shat
    denominator = np.where(valid, denominator, 1.)
    u = 2*_dot(p[:, a], p[:, b])*crossing[a]*crossing[b]/denominator
    v = 2*_dot(p[:, b], p[:, c])*crossing[b]*crossing[c]/denominator
    valid &= (u != 0) & (v != 0)
    u, v = np.where(valid, u, 1.).astype(complex), np.where(valid, v, 1.).astype(complex)
    value = (spence(u)+spence(v)+np.log(u)*np.log(v)-np.pi**2/6).real
    return np.where(valid, value, 0.), valid


def orientation_features(momenta, incoming):
    """Catalog-7 signed normalized epsilon coordinates in native leg order."""
    import numpy as np
    p = np.asarray(momenta, dtype=float)
    inc = np.flatnonzero(incoming)
    if len(inc) not in (1, 2):
        raise ValueError('orientation requires one or two incoming legs')
    anchors = inc if len(inc) == 2 else (inc[0], np.flatnonzero(np.asarray(incoming) == 0)[0])
    total = p[:, inc].sum(axis=1)
    shat = _dot(total, total)
    if np.any(shat <= 0):
        raise ValueError('invalid orientation incoming invariant')
    scaled = p/np.sqrt(shat)[:, None, None]
    a, b = scaled[:, anchors[0]], scaled[:, anchors[1]]
    remaining = [i for i in range(p.shape[1]) if i not in anchors]
    columns = []
    triple = lambda u, v, w: np.einsum('ni,ni->n', u, np.cross(v, w))
    for i, j in itertools.combinations(remaining, 2):
        c, d = scaled[:, i], scaled[:, j]
        epsilon = (a[:, 0]*triple(b[:, 1:], c[:, 1:], d[:, 1:])
                   -b[:, 0]*triple(a[:, 1:], c[:, 1:], d[:, 1:])
                   +c[:, 0]*triple(a[:, 1:], b[:, 1:], d[:, 1:])
                   -d[:, 0]*triple(a[:, 1:], b[:, 1:], c[:, 1:]))
        denominator = np.sqrt(np.maximum(0., 16*_dot(a, c)*_dot(b, c)*_dot(a, d)*_dot(b, d)))
        value = np.divide(2*epsilon, denominator, out=np.zeros(len(p)), where=denominator > 1.e-15)
        columns.append(np.clip(value, -1., 1.))
    return np.column_stack(columns) if columns else np.zeros((len(p), 0))


def canonical_momenta(momenta, incoming, return_valid=False):
    """Deterministic incoming-CM frame, also defined for one-body decays."""
    import numpy as np
    p = np.asarray(momenta, dtype=float)
    inc = np.flatnonzero(incoming)
    outgoing = np.flatnonzero(np.asarray(incoming) == 0)
    total = p[:, inc].sum(axis=1)
    s = _dot(total, total)
    if np.any(s <= 0) or np.any(total[:, 0] <= 0):
        raise ValueError('invalid timelike incoming momentum')
    q = np.sqrt(s)
    beta = total[:, 1:]/total[:, :1]
    gamma = total[:, 0]/q
    bp = np.einsum('ni,nji->nj', beta, p[:, :, 1:])
    # Algebraically (gamma-1)/beta**2, with no cancellation near rest; the
    # native evaluator uses the same expression.
    factor = gamma*gamma/(gamma+1)
    spatial = p[:, :, 1:] + (factor[:, None]*bp-gamma[:, None]*p[:, :, 0])[:, :, None]*beta[:, None, :]
    energy = gamma[:, None]*(p[:, :, 0]-bp)
    anchor = inc[0] if len(inc) == 2 else outgoing[0]
    z = spatial[:, anchor].copy()
    zn = np.linalg.norm(z, axis=1)
    valid = zn > 1.e-14*q
    z /= np.maximum(zn, 1.e-300)[:, None]
    x = np.zeros_like(z)
    chosen = np.zeros(len(p), dtype=bool)
    for leg in outgoing:
        candidate = spatial[:, leg] - np.einsum('ni,ni->n', spatial[:, leg], z)[:, None]*z
        norm = np.linalg.norm(candidate, axis=1)
        use = (~chosen) & (norm > 1.e-12*q)
        x[use] = candidate[use]/norm[use, None]
        chosen |= use
    valid &= chosen
    y = np.cross(z, x)
    result = np.zeros_like(p)
    result[:, :, 0] = 4*energy/q[:, None]-1
    for component, axis in enumerate((x, y, z), 1):
        result[:, :, component] = 2*np.einsum('nji,ni->nj', spatial, axis)/q[:, None]
    result[:, inc] = 0
    # Collinear configurations have no unique transverse frame. Zero auxiliary
    # inputs keep inference finite; exact residual correction remains active.
    result[~valid] = 0
    return (result, valid) if return_valid else result


def input_values(descriptors, features, momenta, q, incoming, return_outside=False):
    """Evaluate typed portable inputs; indices in descriptors are one-based."""
    import numpy as np
    from .virtual_pool_polynomial import coordinates
    x, p, probabilities = (np.asarray(v, dtype=float) for v in (features, momenta, q))
    n = len(x)
    columns = []
    outside = np.zeros(n, dtype=bool)
    canonical = None
    total = p[:, np.flatnonzero(incoming)].sum(axis=1)
    shat = _dot(total, total)
    crossing = np.where(np.asarray(incoming), -1., 1.)
    for kind, a, b, c, lower, upper in descriptors:
        if kind in (1, 2, 7, 9, 10):
            value = x[:, a-1]
        elif kind == 3:
            if canonical is None:
                canonical, valid = canonical_momenta(p, incoming, return_valid=True)
                outside |= ~valid
            value = canonical[:, a-1, b]
        elif kind in (4, 5):
            legs = np.asarray((a, b, c))-1
            momentum = (p[:, legs]*crossing[legs][None, :, None]).sum(axis=1)
            invariant = _dot(momentum, momentum)
            value = (np.log(np.maximum(np.abs(invariant)/shat, 1.e-15)) if kind == 4
                     else np.sign(invariant))
        elif kind == 6:
            value = 2*probabilities[:, a-1]-1
        elif kind == 8:
            value, valid = antenna_remainder(p, incoming, a-1, b-1, c-1)
            outside |= ~valid
        else:
            raise ValueError('unknown neural input descriptor')
        if kind in (1, 4, 7, 8, 9, 10):
            scaled, tail = coordinates(value[:, None], [lower], [upper])
            value = (2*(value-lower)/(upper-lower)-1 if kind in (8, 9, 10) and upper-lower > 1.e-12
                     else scaled[:, 0])
            outside |= tail
            if kind in (7, 10):
                value = value*value
        columns.append(value)
    result = np.column_stack(columns) if columns else np.zeros((n, 0))
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite neural inputs')
    return (result, outside) if return_outside else result


def descriptors_for(metadata, features, momenta, q, train, heads, bank='scalar'):
    import numpy as np
    from .virtual_pool import scalar_feature_count, threshold_pairs
    if bank not in ('scalar', 'canonical', 'composite', 'compact'):
        raise ValueError('unknown neural input bank')
    ns = scalar_feature_count(metadata) if metadata['catalog'] in (7, 8, 9) else metadata['nf']
    lo, hi = features[train].min(axis=0), features[train].max(axis=0)
    raw = metadata['catalog'] == 9 and bank == 'compact'
    descriptors = [(9 if raw else 1, i+1, 0, 0, float(lo[i]), float(hi[i])) for i in range(ns)
                   if hi[i]-lo[i] > 1.e-12]
    descriptors += [(2, i+1, 0, 0, -1., 1.) for i in range(ns, metadata['nf'])]
    if bank == 'compact':
        # The affine layer supplies the constant. No probability coordinates
        # or polynomial cross terms: the same compact bank serves every head.
        # One-loop Coulomb terms are linear in 1/v. Squaring this and the
        # colour-dressed singular inputs would invent stronger singularities.
        square_ns = ns-5-len(threshold_pairs(metadata)) if raw else ns
        descriptors += [(10 if raw else 7, i+1, 0, 0, float(lo[i]), float(hi[i])) for i in range(square_ns)
                        if hi[i]-lo[i] > 1.e-12]
        if raw:
            legs = [i for i, (m, colour) in enumerate(zip(metadata['masses'], metadata['colours']))
                    if m == 0 and abs(colour) > 1]
            for i, j, k in itertools.combinations(legs, 3):
                for a, b, c in ((i, j, k), (j, i, k), (i, k, j)):
                    value, valid = antenna_remainder(momenta, metadata['incoming'], a, b, c)
                    if np.all(valid[train]) and np.ptp(value[train]) > 1.e-12:
                        descriptors.append((8, a+1, b+1, c+1, float(value[train].min()), float(value[train].max())))
        if len(descriptors) > MAX_INPUTS:
            raise ValueError('compact input budget exceeded')
        return descriptors
    extra = len(heads)
    if bank in ('canonical', 'composite'):
        extra += 4*sum(not flag for flag in metadata['incoming'])
    if bank == 'composite':
        nlegs = metadata['nlegs']
        triples = nlegs*(nlegs-1)*(nlegs-2)//6
        if metadata['nlegs'] == 6:
            triples //= 2
        extra += 2*triples
    if len(descriptors)+extra > MAX_INPUTS:
        raise ValueError('neural input budget exceeded')
    if bank in ('canonical', 'composite'):
        descriptors += [(3, i+1, c, 0, -1., 1.) for i, flag in enumerate(metadata['incoming'])
                        if not flag for c in range(4)]
    if bank == 'composite':
        for legs in itertools.combinations(range(1, metadata['nlegs']+1), 3):
            complement = tuple(i for i in range(1, metadata['nlegs']+1) if i not in legs)
            if len(complement) == 3 and complement < legs:
                continue
            descriptor = (4,) + legs + (0., 1.)
            # Evaluate the raw logarithm, before fitting its bounds.
            crossing = np.where(np.asarray(metadata['incoming']), -1., 1.)
            indices = np.asarray(legs)-1
            total = momenta[:, np.flatnonzero(metadata['incoming'])].sum(axis=1)
            v = (momenta[:, indices]*crossing[indices][None, :, None]).sum(axis=1)
            value = np.log(np.maximum(np.abs(_dot(v, v))/_dot(total, total), 1.e-15))
            descriptors.append(descriptor[:4] + (float(value[train].min()), float(value[train].max())))
            descriptors.append((5,) + legs + (-1., 1.))
    descriptors += [(6, int(h), 0, 0, -1., 1.) for h in heads]
    if len(descriptors) > MAX_INPUTS:
        raise ValueError('neural input budget exceeded')
    return descriptors


def helicity_permutation(metadata, permutation):
    """Map original native IDs to IDs after a metadata-preserving leg swap."""
    n = metadata['nlegs']
    identity = list(range(1, n+1))
    if (len(permutation) != n or sorted(permutation) != identity or
            any(int(i) != i for i in permutation)):
        raise ValueError('invalid native leg permutation')
    moved = [i for i, j in enumerate(permutation) if j != i+1]
    if moved:
        if len(moved) != 2:
            raise ValueError('only one outgoing transposition is supported')
        a, b = moved
        if (metadata['incoming'][a] or metadata['incoming'][b] or
                metadata['pdgs'][a] != metadata['pdgs'][b] or
                metadata['masses'][a] != metadata['masses'][b]):
            raise ValueError('leg permutation changes external particle metadata')
    lookup = {tuple(h[2:]): h[0] for h in metadata['helicities']}
    if len(lookup) != metadata['nhel']:
        raise ValueError('duplicate native helicity labels')
    mapping = []
    for h in metadata['helicities']:
        transformed = tuple(h[2+i-1] for i in permutation)
        actual = lookup.get(transformed)
        if actual is None or metadata['helicities'][actual-1][1] != h[1]:
            raise ValueError('native helicity map/multiplicity is not permutation-closed')
        mapping.append(actual)
    return mapping


def permutation_candidate(metadata):
    """First supported identical-outgoing pair; no particle-specific rules."""
    identity = list(range(1, metadata['nlegs']+1))
    for a, b in itertools.combinations(range(metadata['nlegs']), 2):
        if (metadata['incoming'][a] or metadata['incoming'][b] or
                metadata['pdgs'][a] != metadata['pdgs'][b] or
                metadata['masses'][a] != metadata['masses'][b]):
            continue
        candidate = identity.copy(); candidate[a], candidate[b] = candidate[b], candidate[a]
        try:
            helicity_permutation(metadata, candidate)
        except ValueError:
            continue
        return candidate
    return None


def _check_head_closure(entry, mapping):
    if not entry.get('enabled', True):
        return
    flags = dict(zip(entry['heads'], entry['trained']))
    if any(mapping[h-1] not in flags or flags[mapping[h-1]] != flag for h, flag in flags.items()):
        raise ValueError('trained neural heads are not permutation-closed')


def permuted_features(metadata, features, momenta, permutation):
    import numpy as np
    from .virtual_pool import scalar_feature_count, threshold_features, threshold_pairs
    x = np.asarray(features).copy()
    p = np.asarray(momenta)[:, np.asarray(permutation)-1]
    pairs = list(itertools.combinations(range(metadata['nlegs']), 2))
    indices = {pair: index+1 for index, pair in enumerate(pairs)}
    for index, (a, b) in enumerate(pairs, 1):
        pair = tuple(sorted((permutation[a]-1, permutation[b]-1)))
        x[:, index] = np.asarray(features)[:, indices[pair]]
    if metadata['catalog'] in (8, 9):
        nt = len(threshold_pairs(metadata))
        index = scalar_feature_count(metadata)-nt-1-(4 if metadata['catalog'] == 9 else 0)
        x[:, index:index+nt] = threshold_features(metadata, p)
    if metadata['catalog'] in (7, 8, 9):
        x[:, scalar_feature_count(metadata):] = orientation_features(p, metadata['incoming'])
    return x, p


def _shapes(ninputs, nheads, widths):
    if len(widths) not in (1, 2) or any(not 1 <= w <= MAX_WIDTH for w in widths):
        raise ValueError('unsupported neural architecture')
    result = []
    previous = ninputs
    for width in widths:
        result.extend(((previous, width), (width,)))
        previous = width
    return result + [(previous, nheads), (nheads,), (ninputs+1, nheads)]


def unpack(theta, ninputs, nheads, widths):
    import numpy as np
    shapes = _shapes(ninputs, nheads, widths)
    parts, offset = [], 0
    for shape in shapes:
        size = shape[0]*(shape[1] if len(shape) == 2 else 1)
        parts.append(np.asarray(theta[offset:offset+size]).reshape(shape))
        offset += size
    if offset != len(theta):
        raise ValueError('incorrect neural parameter count')
    return parts


def forward(entry, inputs):
    import numpy as np
    parts = unpack(entry['theta'], inputs.shape[1], len(entry['heads']), entry['widths'])
    hidden = inputs
    for layer in range(len(entry['widths'])):
        hidden = np.tanh(hidden@parts[2*layer] + parts[2*layer+1])
    output = hidden@parts[-3] + parts[-2] + np.column_stack((inputs, np.ones(len(inputs))))@parts[-1]
    return output*np.asarray(entry['trained'], dtype=float)[None, :]


def _grouped_output(hidden, affine, parameters, mixture, born, target, weight, ridge):
    """Evaluate selected heads and full native mixtures, with output derivatives."""
    import numpy as np
    output, bias, skip = parameters
    groups, full, full_mixing = mixture
    prediction = np.zeros(len(hidden))
    blocks = [(head, rows, probability, hidden[rows], affine[rows])
              for head, rows, probability in groups]
    for head, rows, probability, block_hidden, block_affine in blocks:
        prediction[rows] = probability*(block_hidden@output[:, head] + bias[head] +
                                        block_affine@skip[:, head])
    full_hidden, full_affine = hidden[full], affine[full]
    prediction[full] = np.sum(full_mixing*(full_hidden@output + bias + full_affine@skip), axis=1)
    residual = born*prediction-target
    derivative = weight*residual*born
    output_gradient = ridge*output
    bias_gradient = ridge*bias
    skip_gradient = ridge*skip
    back = np.zeros_like(hidden)
    for head, rows, probability, block_hidden, block_affine in blocks:
        dc = derivative[rows]*probability
        output_gradient[:, head] += block_hidden.T@dc
        bias_gradient[head] += dc.sum()
        skip_gradient[:, head] += block_affine.T@dc
        back[rows] = dc[:, None]*output[:, head]
    dc = derivative[full, None]*full_mixing
    output_gradient += full_hidden.T@dc
    bias_gradient += dc.sum(axis=0)
    skip_gradient += full_affine.T@dc
    back[full] = dc@output.T
    return residual, (output_gradient, bias_gradient, skip_gradient), back


def objective(theta, inputs, mixing, born, target, weight, widths, ridge, trained,
              prepared=None):
    """Direct weighted amplitude residual and analytic gradient (one/two layers)."""
    import numpy as np
    parts = unpack(theta, inputs.shape[1], mixing.shape[1], widths)
    activations = [inputs]
    for layer in range(len(widths)):
        activations.append(np.tanh(activations[-1]@parts[2*layer] + parts[2*layer+1]))
    if prepared is None:
        affine = np.column_stack((inputs, np.ones(len(inputs))))
        effective = mixing*np.asarray(trained)[None, :]
        grouped = False
    else:
        affine, effective, grouped = prepared
    gradients = [None]*len(parts)
    if grouped:
        residual, output_gradients, back = _grouped_output(
            activations[-1], affine, parts[-3:], effective, born, target, weight, ridge)
        gradients[-3:] = output_gradients
    else:
        allheads = activations[-1]@parts[-3] + parts[-2] + affine@parts[-1]
        residual = born*np.sum(allheads*effective, axis=1)-target
        dc = (weight*residual*born)[:, None]*effective
        gradients[-3] = activations[-1].T@dc + ridge*parts[-3]
        gradients[-2] = dc.sum(axis=0) + ridge*parts[-2]
        gradients[-1] = affine.T@dc + ridge*parts[-1]
        back = dc@parts[-3].T
    for layer in reversed(range(len(widths))):
        dt = back*(1-activations[layer+1]**2)
        gradients[2*layer] = activations[layer].T@dt
        gradients[2*layer+1] = dt.sum(axis=0)
        back = dt@parts[2*layer].T
    penalty = sum(np.sum(v*v) for v in parts[-3:])
    return (.5*float(weight@(residual*residual)) + .5*ridge*penalty,
            np.concatenate([g.ravel() for g in gradients]))


def _prepare_objective(inputs, mixing, trained):
    """Prepare arrays which are constant across all optimiser evaluations."""
    import numpy as np
    affine = np.column_stack((inputs, np.ones(len(inputs))))
    effective = mixing*np.asarray(trained)[None, :]
    grouped = (effective.shape[1] >= _GROUPED_MIN_HEADS and
               np.count_nonzero(effective) < _GROUPED_MAX_DENSITY*effective.size)
    if grouped:
        counts = np.count_nonzero(effective, axis=1)
        groups = []
        for head in range(effective.shape[1]):
            rows = np.flatnonzero((counts == 1) & (effective[:, head] != 0))
            if len(rows):
                groups.append((head, rows, effective[rows, head]))
        full = np.flatnonzero(counts > 1)
        effective = _HeadGroups(groups, full, effective[full])
    return _ObjectiveData(affine, effective, grouped)


def fit_linear(inputs, mixing, born, target, weight, heads, trained, configuration):
    """Coupled ridge fit to V-B*f, including native full-sum constraints.

    Selected branches produce one sparse block; full branches couple all
    represented heads with their native q. Zero/signed Born rows are retained.
    A zero hidden layer keeps the existing portable inference format.
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import lsmr
    from .virtual_pool_polynomial import _ridge_solve
    if (inputs.ndim != 2 or not 1 <= inputs.shape[1] <= MAX_INPUTS or
            not 1 <= len(heads) <= MAX_HEADS or mixing.shape != (len(inputs), len(heads)) or
            len(trained) != len(heads) or not 0 < configuration['ridge'] <= 1 or
            not 0 <= configuration['alpha'] <= 2 or
            not 1 <= configuration['iterations'] <= MAX_OPTIMIZER_ITERATIONS):
        raise ValueError('invalid linear fitting dimensions/objective')
    active = np.flatnonzero(trained)
    if not len(active):
        raise ValueError('no trained linear heads')
    w = weight/weight.sum()
    bscale = float(w@np.abs(born))
    if not bscale:
        raise ValueError('zero linear Born scale')
    carrier = born/bscale
    noise = np.maximum(np.abs(carrier), 1.e-12)**configuration['alpha']
    vscale = math.sqrt(float(w@(target*target/noise)))
    if not vscale or not math.isfinite(vscale):
        raise ValueError('zero/nonfinite linear target scale')
    affine = np.column_stack((inputs, np.ones(len(inputs))))
    row, head = np.nonzero(mixing[:, active])
    ni, nh = affine.shape[1], len(active)
    if len(row)*ni > MAX_LINEAR_DESIGN_ELEMENTS:
        raise ValueError('linear design budget exceeded')
    root = np.sqrt(w/noise)
    values = affine[row]*(root[row]*carrier[row]*mixing[row, active[head]])[:, None]
    columns = head[:, None]*ni+np.arange(ni)
    design = sparse.csr_matrix((values.ravel(),
                                (np.repeat(row, ni), columns.ravel())),
                               shape=(len(inputs), ni*nh))
    response = root*(target/vscale)
    design.eliminate_zeros()
    # Zero-Born and absent-head rows contribute a constant to the objective.
    # Keep them in the scales and validation, but remove that constant from
    # the solve's stopping norm; it can otherwise hide an inaccurate solution.
    response[design.getnnz(axis=1) == 0] = 0.
    column_scale = np.ones(ni*nh)
    if configuration.get('column_scale', False):
        column_scale = np.sqrt(np.asarray(design.multiply(design).sum(axis=0)).ravel())
        column_scale[column_scale <= max(float(column_scale.max())*1.e-10, 1.e-20)] = 1.
        design = design @ sparse.diags(1/column_scale)
    if ni*nh <= MAX_DENSE_LINEAR_PARAMETERS:
        coefficients = _ridge_solve((design.T@design).toarray(), design.T@response,
                                    configuration['ridge'])
        iterations, status = 1, 'positive_definite_solve'
    else:
        solution = lsmr(design, response, damp=math.sqrt(configuration['ridge']),
                        atol=1.e-8, btol=1.e-8, maxiter=configuration['iterations'])
        coefficients, status, iterations = solution[:3]
        if status not in (0, 1, 2, 4, 5):
            raise ValueError('linear solve did not converge: %s' % status)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError('nonfinite linear coefficients')
    # Only the affine skip is used; the width-one tanh and output are zero.
    parts = [np.zeros(shape) for shape in _shapes(inputs.shape[1], len(heads), (1,))]
    parts[-1][:, active] = (coefficients/column_scale).reshape(nh, ni).T
    return dict(family='linear', widths=[1], heads=list(map(int, heads)),
                trained=list(map(int, trained)), theta=np.concatenate([p.ravel() for p in parts]).tolist(),
                born_scale=bscale, target_scale=vscale, optimizer_evaluations=0,
                optimizer_iterations=0, solver_iterations=int(iterations), solver_status=str(status))


def fit(inputs, mixing, born, target, weight, heads, trained, configuration):
    if configuration.get('family') == 'linear':
        return fit_linear(inputs, mixing, born, target, weight, heads, trained, configuration)
    import numpy as np
    from scipy.optimize import minimize
    widths = tuple(configuration['widths'])
    if (not 1 <= inputs.shape[1] <= MAX_INPUTS or not 1 <= len(heads) <= MAX_HEADS or
            mixing.shape != (len(inputs), len(heads)) or len(trained) != len(heads)):
        raise ValueError('neural fitting dimension budget exceeded')
    if (not 1 <= configuration['iterations'] <= MAX_OPTIMIZER_ITERATIONS or
            not 0 < configuration['ridge'] <= 1 or not 0 <= configuration['alpha'] <= 2):
        raise ValueError('invalid neural optimization budget/objective')
    _shapes(inputs.shape[1], len(heads), widths)
    if len(inputs)*sum(widths) > MAX_ACTIVATION_ELEMENTS:
        raise ValueError('neural activation budget exceeded')
    w = weight/weight.sum()
    bscale = float(w@np.abs(born))
    if not bscale:
        raise ValueError('zero neural Born scale')
    carrier = born/bscale
    noise = np.maximum(np.abs(carrier), 1.e-12)**configuration['alpha']
    vscale = math.sqrt(float(w@(target*target/noise)))
    if not vscale or not math.isfinite(vscale):
        raise ValueError('zero/nonfinite neural target scale')
    shapes = _shapes(inputs.shape[1], mixing.shape[1], widths)
    rng = np.random.RandomState(98273)
    parts = [np.zeros(shape) for shape in shapes]
    for layer in range(len(widths)):
        parts[2*layer] = rng.normal(size=shapes[2*layer])/math.sqrt(shapes[2*layer][0])
    theta = np.concatenate([p.ravel() for p in parts])
    prepared = _prepare_objective(inputs, mixing, trained)
    args = (inputs, mixing, carrier, target/vscale, w/noise, widths,
            configuration['ridge'], trained, prepared)
    result = minimize(objective, theta, args=args, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=configuration['iterations'], maxcor=20,
                                   ftol=1.e-12, gtol=1.e-8, maxls=30))
    if not np.all(np.isfinite(result.x)) or not math.isfinite(result.fun):
        raise ValueError('nonfinite neural optimization')
    return dict(widths=list(widths), heads=list(map(int, heads)), trained=list(map(int, trained)),
                theta=result.x.tolist(), born_scale=bscale, target_scale=vscale,
                optimizer_evaluations=int(result.nfev),
                optimizer_iterations=int(result.nit), optimizer_status=str(result.message))


def predict_correction(entry, metadata, features, momenta, q, born, selected=None, full=None,
                       common=None):
    """Return selected and native-q-mean corrections in the units of ``born``."""
    import numpy as np
    b = np.asarray(born, dtype=float)
    if not entry.get('enabled', True):
        return np.zeros_like(b), np.zeros_like(b)
    probabilities = np.asarray(q, dtype=float)
    if (b.ndim != 1 or probabilities.shape != (len(b), metadata['nhel']) or
            not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or
            np.any(probabilities > 1) or not np.all(np.isfinite(b))):
        raise ValueError('invalid neural inference probabilities/carriers')
    total = probabilities.sum(axis=1)
    known = total > 0
    if np.any(np.abs(total[known]-1) > 1.e-9):
        raise ValueError('native inference probabilities do not sum to one')
    heads = np.asarray(entry['heads'], dtype=int)
    outputs = np.zeros((len(b), len(heads)))
    if np.any(known):
        x = input_values(entry['descriptors'], np.asarray(features)[known],
                         np.asarray(momenta)[known], probabilities[known], metadata['incoming'])
        outputs[known] = forward(entry, x)*(b[known]*entry['target_scale']/entry['born_scale'])[:, None]
    if entry.get('replaces_common', False):
        if common is None or np.asarray(common).shape != b.shape or not np.all(np.isfinite(common)):
            raise ValueError('absolute heads require the common prediction')
        outputs[known] -= (np.asarray(common)[known, None]*
                           np.asarray(entry['trained'], dtype=float)[None, :])
        outputs *= entry.get('replacement_blend', 1.)
    mean = np.sum(probabilities[:, heads-1]*outputs, axis=1)
    if selected is None:
        return mean.copy(), mean
    selected = np.asarray(selected)
    answer = np.zeros(len(b))
    for index, actual in enumerate(heads):
        mask = selected == actual
        answer[mask] = outputs[mask, index]
    full = selected == -1 if full is None else np.asarray(full, dtype=bool)
    answer[full] = mean[full]
    if not np.all(np.isfinite(answer)) or not np.all(np.isfinite(mean)):
        raise ValueError('nonfinite neural prediction')
    return answer, mean


def _complete_prediction(entry, common, metadata, features, momenta, q, born, mint,
                         selected=None, full=None, permutations=None):
    import numpy as np
    from .virtual_pool_polynomial import predict as common_predict
    identity = list(range(1, metadata['nlegs']+1))
    permutations = permutations or [identity]
    selected_total = np.zeros(len(born))
    mean_total = np.zeros(len(born))
    identity_common = None
    for permutation in permutations:
        permutation = list(permutation)
        if permutation == identity:
            x, p, probabilities, actual = features, momenta, q, selected
        else:
            mapping = np.asarray(helicity_permutation(metadata, permutation), dtype=int)
            _check_head_closure(entry, mapping)
            x, p = permuted_features(metadata, features, momenta, permutation)
            probabilities = np.asarray(q)[:, np.argsort(mapping)]
            actual = None if selected is None else np.asarray(selected).copy()
            if actual is not None:
                active = actual > 0
                actual[active] = mapping[actual[active].astype(int)-1]
        shared = common_predict(common, np.asarray(x), np.asarray(born), np.asarray(mint))
        if identity_common is None:
            identity_common = shared
        if entry.get('replaces_common', False):
            # Compose absolute heads directly: subtracting a large MINT value
            # and adding it back can erase a perfectly finite head prediction.
            absolute = dict(entry, replaces_common=False)
            selected_value, mean_value = predict_correction(absolute, metadata, x, p, probabilities,
                                                            born, actual, full)
            blend = entry.get('replacement_blend', 1.)
            missing = np.ones(metadata['nhel'], dtype=bool)
            trained = np.asarray(entry['heads'])[np.asarray(entry['trained'], dtype=bool)]
            missing[trained-1] = False
            mean_common = 1-blend+blend*np.asarray(probabilities)[:, missing].sum(axis=1)
            if actual is None:
                selected_common = mean_common
            else:
                selected_common = np.where(np.isin(actual, trained), 1-blend, 1.)
                full_rows = np.asarray(actual) == -1 if full is None else np.asarray(full, dtype=bool)
                selected_common = np.where(full_rows, mean_common, selected_common)
            selected_total += selected_common*shared+blend*selected_value
            mean_total += mean_common*shared+blend*mean_value
        else:
            selected_delta, mean_delta = predict_correction(entry, metadata, x, p, probabilities,
                                                            born, actual, full)
            selected_total += shared+selected_delta
            mean_total += shared+mean_delta
    known = np.asarray(q).sum(axis=1) > 0
    return (np.where(known, selected_total/len(permutations), identity_common),
            np.where(known, mean_total/len(permutations), identity_common))


def predict_model(model, common_model, order, features, momenta, q, born, mint,
                  selected=None, full=None):
    """Complete common+conditional prediction, including selected symmetries."""
    if not 1 <= order <= model['metadata']['norders']:
        raise ValueError('invalid conditional model order')
    return _complete_prediction(model['orders'][order-1], common_model['orders'][order-1],
        model['metadata'], features, momenta, q, born, mint, selected, full,
        model.get('permutations'))


def _validate(model):
    import numpy as np
    metadata = model['metadata']
    if (not 0 <= metadata['nhel'] <= MAX_NATIVE_HELICITIES or
            len(metadata['helicities']) != metadata['nhel']):
        raise ValueError('invalid neural native map')
    for actual, record in enumerate(metadata['helicities'], 1):
        if len(record) != metadata['nlegs']+2 or record[0] != actual or record[1] < 0:
            raise ValueError('invalid neural complete native map')
    digest = model['base_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid base-model SHA256')
    if len(model['orders']) != metadata['norders']:
        raise ValueError('incorrect neural order count')
    permutations = model.get('permutations')
    mappings = []
    if permutations is not None:
        if (len(permutations) not in (1, 2) or
                list(permutations[0]) != list(range(1, metadata['nlegs']+1)) or
                (len(permutations) == 2 and list(permutations[1]) == list(permutations[0]))):
            raise ValueError('invalid neural permutation table')
        for permutation in permutations:
            mappings.append(helicity_permutation(metadata, permutation))
    for number, entry in enumerate(model['orders'], 1):
        if entry['order'] != number:
            raise ValueError('unordered neural coupling order')
        if not entry.get('enabled', False):
            continue
        descriptors, heads = entry['descriptors'], entry['heads']
        if (not 1 <= len(descriptors) <= MAX_INPUTS or not 1 <= len(heads) <= MAX_HEADS or
                len(set(heads)) != len(heads) or any(not 1 <= h <= metadata['nhel'] for h in heads) or
                len(entry['trained']) != len(heads) or any(v not in (0, 1) for v in entry['trained']) or
                any(not metadata['helicities'][h-1][1] for h in heads) or
                not math.isfinite(entry['born_scale']) or entry['born_scale'] <= 0 or
                not math.isfinite(entry['target_scale']) or entry['target_scale'] <= 0):
            raise ValueError('invalid neural order dimensions/scales')
        if entry.get('replaces_common', False) not in (0, 1):
            raise ValueError('invalid conditional head interpretation')
        if entry.get('replaces_common', False) and not 0 <= entry.get('replacement_blend', 1.) <= 1:
            raise ValueError('invalid absolute-head blend')
        for descriptor in descriptors:
            if len(descriptor) != 6:
                raise ValueError('invalid neural descriptor width')
            kind, a, b, c, lo, hi = descriptor
            if any(int(v) != v for v in (kind, a, b, c)) or not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
                raise ValueError('invalid neural descriptor metadata')
            if (kind not in range(1, 11) or
                    (kind in (1, 2, 7, 9, 10) and (not 1 <= a <= metadata['nf'] or b or c)) or
                    (kind == 3 and (not 1 <= a <= metadata['nlegs'] or b not in range(4) or c or
                                    metadata['incoming'][a-1])) or
                    (kind in (4, 5, 8) and (len({a, b, c}) != 3 or min(a, b, c) < 1 or max(a, b, c) > metadata['nlegs'])) or
                    (kind == 6 and (not 1 <= a <= metadata['nhel'] or b or c))):
                raise ValueError('invalid neural descriptor indices')
        unpack(entry['theta'], len(descriptors), len(heads), entry['widths'])
        if not np.all(np.isfinite(entry['theta'])):
            raise ValueError('nonfinite neural coefficients')
        # Validate IDs before indexing the native permutation map, so malformed
        # companions fail with ValueError rather than an accidental IndexError.
        for mapping in mappings:
            _check_head_closure(entry, mapping)


def model_text(model):
    """Portable text codec shared with the native reader; matrices input-major."""
    _validate(model)
    metadata = model['metadata']
    floats = lambda values: ' '.join('%.17e' % v for v in values)
    kinds = {d[0] for entry in model['orders'] for d in entry.get('descriptors', [])}
    if any(kind >= 8 for kind in kinds):
        version = 4
    elif 7 in kinds or any(entry.get('replaces_common', False) for entry in model['orders']):
        version = 3
    elif len(model.get('permutations', [])) > 1:
        version = 2
    else:
        version = 1
    lines = ['VIRTUAL_POOL_NEURAL %d' % version,
             ' '.join(str(metadata[k]) for k in ('catalog', 'nf', 'norders', 'nlegs', 'nx', 'nhel')),
             ' '.join(map(str, metadata['pdgs'])), floats(metadata['masses']),
             ' '.join(map(str, metadata['incoming']))]
    if metadata['catalog'] == 9:
        lines.append(' '.join(map(str, metadata['colours'])))
    lines.append(model['base_sha256'])
    lines += [' '.join(map(str, h)) for h in metadata['helicities']]
    if version >= 2:
        permutations = model.get('permutations', [list(range(1, metadata['nlegs']+1))])
        lines.append(str(len(permutations)))
        lines += [' '.join(map(str, p)) for p in permutations]
    for entry in model['orders']:
        if not entry.get('enabled', False):
            line = '%d 0 0 0 0 0 1 1' % entry['order']
            if version >= 3:
                line += ' -1'
            lines.append(line)
            continue
        widths = entry['widths']
        line = '%d 1 %d %d %d %d %.17e %.17e' % (
            entry['order'], len(entry['descriptors']), len(entry['heads']), widths[0],
            widths[1] if len(widths) == 2 else 0, entry['born_scale'], entry['target_scale'])
        if version >= 3:
            replacement = entry.get('replacement_blend', 1.) if entry.get('replaces_common', False) else -1.
            line += ' %.17e' % replacement
        lines.append(line)
        lines += ['%d %d %d %d %.17e %.17e' % tuple(d) for d in entry['descriptors']]
        lines += ['%d %d' % pair for pair in zip(entry['heads'], entry['trained'])]
        for part in unpack(entry['theta'], len(entry['descriptors']), len(entry['heads']), widths):
            lines.extend(floats(row) for row in (part if part.ndim == 2 else [part]))
    return '\n'.join(lines)+'\n'


def load_model(path, base_path=None):
    from .virtual_pool import _floats, _header
    import io
    with open(path) as stream:
        header = stream.readline().split()
        if len(header) != 2 or header[0] != 'VIRTUAL_POOL_NEURAL' or header[1] not in ('1', '2', '3', '4'):
            raise ValueError('unsupported neural companion format')
        version = int(header[1])
        dims = stream.readline().split()
        if len(dims) != 6:
            raise ValueError('invalid neural dimensions')
        particle_lines = [stream.readline() for _ in range(3)]
        if int(dims[0]) == 9:
            particle_lines.append(stream.readline())
        common_header = 'VIRTUAL_POOL_MODEL 3\n'+' '.join(dims[:5])+'\n'+''.join(particle_lines)
        metadata = _header(io.StringIO(common_header), 'VIRTUAL_POOL_MODEL')
        metadata['nhel'] = int(dims[5])
        if not 0 <= metadata['nhel'] <= MAX_NATIVE_HELICITIES:
            raise ValueError('invalid neural native map size')
        digest = stream.readline().strip()
        metadata['helicities'] = tuple(tuple(int(v) for v in stream.readline().split())
                                       for _ in range(metadata['nhel']))
        permutations = None
        if version >= 2:
            fields = stream.readline().split()
            if len(fields) != 1 or int(fields[0]) not in (1, 2):
                raise ValueError('invalid neural permutation count')
            permutations = [tuple(int(v) for v in stream.readline().split()) for _ in range(int(fields[0]))]
        orders = []
        for number in range(1, metadata['norders']+1):
            fields = stream.readline().split()
            if len(fields) != (9 if version >= 3 else 8):
                raise ValueError('invalid neural order header')
            order, enabled, ni, nh, w1, w2 = (int(v) for v in fields[:6])
            bs, vs = _floats(' '.join(fields[6:8]))
            replacement = _floats(fields[8])[0] if version >= 3 else -1.
            if not (replacement == -1 or 0 <= replacement <= 1) or (not enabled and replacement != -1):
                raise ValueError('invalid conditional head interpretation')
            if order != number or enabled not in (0, 1):
                raise ValueError('invalid neural order number/enabled flag')
            entry = dict(order=order, enabled=bool(enabled), born_scale=bs, target_scale=vs)
            if replacement >= 0:
                entry.update(replaces_common=True, replacement_blend=replacement)
            if not enabled:
                if any((ni, nh, w1, w2)) or bs != 1 or vs != 1:
                    raise ValueError('invalid disabled neural order')
                orders.append(entry)
                continue
            if not 1 <= ni <= MAX_INPUTS or not 1 <= nh <= MAX_HEADS or not 1 <= w1 <= MAX_WIDTH or not 0 <= w2 <= MAX_WIDTH:
                raise ValueError('neural resource cap exceeded')
            descriptors = []
            for _ in range(ni):
                fields = stream.readline().split()
                if len(fields) != 6:
                    raise ValueError('invalid neural descriptor')
                descriptor = tuple(int(v) for v in fields[:4])+_floats(' '.join(fields[4:]))
                if descriptor[0] == 7 and version < 3:
                    raise ValueError('squared input requires companion format 3')
                if descriptor[0] >= 8 and version < 4:
                    raise ValueError('physics input requires companion format 4')
                descriptors.append(descriptor)
            heads = [tuple(int(v) for v in stream.readline().split()) for _ in range(nh)]
            if any(len(h) != 2 for h in heads):
                raise ValueError('invalid neural head record')
            widths = [w1]+([w2] if w2 else [])
            theta = []
            for shape in _shapes(ni, nh, widths):
                for _ in range(shape[0] if len(shape) == 2 else 1):
                    row = _floats(stream.readline())
                    if len(row) != shape[-1]:
                        raise ValueError('invalid neural matrix row')
                    theta.extend(row)
            entry.update(descriptors=descriptors, heads=[h[0] for h in heads], trained=[h[1] for h in heads], widths=widths, theta=theta)
            orders.append(entry)
        if stream.read().strip():
            raise ValueError('unexpected neural trailing data')
    model = dict(metadata=metadata, base_sha256=digest, orders=orders)
    if permutations is not None:
        model['permutations'] = permutations
    _validate(model)
    if base_path is not None:
        with open(base_path, 'rb') as stream:
            actual = hashlib.sha256(stream.read()).hexdigest()
        if actual != digest:
            raise ValueError('neural companion/base-model checksum mismatch')
        from .virtual_pool import load_model as load_base
        base = load_base(base_path)['metadata']
        if any(base[key] != metadata[key] for key in
               ('catalog', 'nf', 'norders', 'nlegs', 'nx', 'pdgs', 'masses', 'incoming')):
            raise ValueError('neural companion/base-model metadata mismatch')
        if base.get('colours') != metadata.get('colours'):
            raise ValueError('neural companion/base-model colour mismatch')
    return model


def chronological_mask(rows, fraction):
    """Split each worker/channel chronologically without separating a sample."""
    import numpy as np
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row['directory'], row['channel']), []).append(index)
    mask = np.zeros(len(rows), dtype=bool)
    for indices in groups.values():
        keys = sorted({(rows[i]['iteration'], rows[i]['sample']) for i in indices})
        early = set(keys[:int(len(keys)*fraction)])
        for i in indices:
            mask[i] = (rows[i]['iteration'], rows[i]['sample']) in early
    return mask


def _mixing(rows, heads):
    import numpy as np
    q = np.asarray([r['q'] for r in rows])
    selected = np.asarray([r['selected_helicity'] for r in rows])
    full = np.asarray([r['full_branch'] for r in rows])
    mixing = np.zeros((len(rows), len(heads)))
    for column, head in enumerate(heads):
        mixing[:, column] = selected == head
    mixing[full] = q[full][:, np.asarray(heads)-1]
    return mixing


def _head_mask(rows, heads, train):
    import numpy as np
    return np.asarray([sum(bool(train[i]) and
                          ((r['full_branch'] and r['q'][h-1] > 0) or r['selected_helicity'] == h)
                          for i, r in enumerate(rows)) >= MIN_HEAD_ROWS for h in heads])


def _refit_base(rows, metadata, order_entry, diagnostics, train):
    import numpy as np
    from . import virtual_pool_polynomial as polynomial
    from .virtual_pool import _folds
    mint = polynomial.mint_predictions(rows, train)
    # Select the common recipe inside this earlier subset as well as refitting
    # its coefficients. Reusing the full-capture choice would let later labels
    # influence an outer fold through architecture/blend selection.
    earlier = [row for row, include in zip(rows, train) if include]
    arguments = dict(strategy='compact') if diagnostics.get('strategy') == 'compact' else {}
    fitted, report = polynomial.fit_order(earlier, _folds(earlier, 3), metadata,
                                         order_entry['order'], **arguments)
    x, b = (np.asarray([r[key] for r in rows]) for key in ('features', 'born'))
    return polynomial.predict(fitted, x, b, mint), report.get('selected'), fitted['blend'], fitted, mint


def _metrics(y, prediction, weight):
    import numpy as np
    scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(prediction))), 1.e-100)
    w = weight/weight.sum()
    residual = (y-prediction)/scale
    target = y/scale
    l1, second = float(w@np.abs(residual)), float(w@(residual*residual))
    true_l1, true_second = float(w@np.abs(target)), float(w@(target*target))
    return dict(l1=l1*scale, rms=math.sqrt(second)*scale,
                maximum=float(np.max(np.abs(residual)))*scale,
                relative_l1=l1/true_l1 if true_l1 else None,
                relative_rms=math.sqrt(second/true_second) if true_second else None)


def _forward_validation(metrics, gains, errors):
    """Use the same promotion guards for conditional fits and symmetry averages."""
    import numpy as np
    baseline_l1 = float(np.mean([m['baseline']['l1'] for m in metrics]))
    gain = float(np.mean(gains))
    # Forward validation sets overlap. Keep the largest paired error rather
    # than treating the folds as independent estimates of the improvement.
    error = max(errors)
    stable = all(m['l1'] < m['baseline']['l1'] and
                 m['rms'] <= 1.05*m['baseline']['rms'] for m in metrics)
    variance = np.mean([m['rms']**2 for m in metrics]) <= np.mean([
        m['baseline']['rms']**2 for m in metrics])
    accepted = stable and variance and gain > max(.02*baseline_l1, 2*error)
    return dict(eligible=bool(accepted), mean_l1_gain=gain, paired_gain_error=error)


def _validate_symmetry(rows, metadata, x, p, q, b, y, w, folds, models, blend, permutation):
    """Compare a complete-prediction average on the existing forward holds."""
    import numpy as np
    selected = np.asarray([r['selected_helicity'] for r in rows])
    full = np.asarray([r['full_branch'] for r in rows])
    identity = list(range(1, metadata['nlegs']+1))
    result = dict(permutation=permutation, eligible=False, folds=[])
    gains, errors = [], []
    try:
        mapping = helicity_permutation(metadata, permutation)
        for fold, fitted in zip(folds, models):
            entry = dict(fitted, descriptors=fold['descriptors'], enabled=True)
            if entry.get('replaces_common', False):
                entry['replacement_blend'] = blend
            else:
                entry['target_scale'] *= blend
            _check_head_closure(entry, mapping)
            va = fold['valid']
            args = (entry, fold['common_model'], metadata, x[va], p[va], q[va], b[va],
                    fold['mint'][va], selected[va], full[va])
            original, _ = _complete_prediction(*args)
            averaged, _ = _complete_prediction(*args, permutations=[identity, permutation])
            metric = _metrics(y[va], averaged, w[va])
            metric['baseline'] = _metrics(y[va], original, w[va])
            result['folds'].append(metric)
            ww = w[va]/w[va].sum()
            improvement = np.abs(y[va]-original)-np.abs(y[va]-averaged)
            gains.append(float(ww@improvement))
            errors.append(float(np.linalg.norm(ww*improvement)))
    except (ValueError, FloatingPointError):
        result['reason'] = 'unsupported_permutation_or_prediction'
        return result
    result.update(_forward_validation(result['folds'], gains, errors))
    result['reason'] = ('selected_by_three_forward_folds' if result['eligible']
                        else 'forward_validation_rejected')
    return result


def train_order(rows, metadata, order_entry, diagnostics, configurations=None, bank='scalar', permutation=None):
    """Three forward fits select a correction; coefficients never see later rows.

    The common polynomial recipe is selected within each earlier subset by its
    own guarded CV, with a fixed compact ridge recipe in production. Every
    coefficient, input bound and head eligibility uses only the earlier part
    of each neural fold. No diagnostic fullsum labels are generated or inferred.
    Full native branches use the q-mean in the loss.
    """
    import numpy as np
    from . import virtual_pool_polynomial as polynomial
    all_order_rows = [r for r in rows if r['order'] == order_entry['order']]
    known = np.asarray([bool(r.get('q')) and sum(r['q']) > 0 for r in all_order_rows])
    subset = [r for r, keep in zip(all_order_rows, known) if keep]
    report = dict(order=order_entry['order'], candidates=[], native_rows=len(subset),
                  skipped_unknown_native_probabilities=len(all_order_rows)-len(subset),
                  additional_loop_calls=0, validation_fractions=list(FRACTIONS))
    disabled = dict(order=order_entry['order'], enabled=False, born_scale=1., target_scale=1.)
    if metadata['catalog'] == 9 and polynomial.physics_sufficient(diagnostics.get('selected')):
        return disabled, dict(report, reason='coupling_carried_common_residual_small')
    if len(subset) < 4*MIN_HEAD_ROWS or not all('q' in r for r in subset):
        return disabled, dict(report, reason='insufficient_conditional_samples')
    if (len(subset)*metadata['nhel'] > MAX_PROBABILITY_ELEMENTS or
            len(subset)*metadata['nf'] > MAX_FEATURE_ELEMENTS):
        return disabled, dict(report, reason='neural_input_storage_budget_exceeded')
    x, b, y, w = (np.asarray([r[key] for r in subset], dtype=float)
                  for key in ('features', 'born', 'virtual', 'weight'))
    p = np.asarray([r['momenta'] for r in subset]).reshape(len(subset), metadata['nlegs'], 4)
    q = np.asarray([r['q'] for r in subset])
    heads = [h[0] for h in metadata['helicities'] if h[1] > 0]
    first = chronological_mask(all_order_rows, FRACTIONS[0])[known]
    counts = {h: sum(first[i] and (r['selected_helicity'] == h or
                    (r['full_branch'] and r['q'][h-1] > 0)) for i, r in enumerate(subset)) for h in heads}
    heads = sorted(sorted(heads, key=lambda h: (-counts[h], h))[:MAX_HEADS])
    trained_heads = _head_mask(subset, heads, first)
    if not heads or not np.any(trained_heads):
        return disabled, dict(report, reason='insufficient_native_head_samples')
    mixing = _mixing(subset, heads)
    represented = mixing@trained_heads.astype(float)
    folds = []
    try:
        for fraction in FRACTIONS:
            train_all = chronological_mask(all_order_rows, fraction)
            train = train_all[known]
            valid = ~train
            if train.sum() < 2*MIN_HEAD_ROWS or valid.sum() < MIN_HEAD_ROWS:
                return disabled, dict(report, reason='insufficient_forward_samples')
            baseline_all, common_recipe, common_blend, common_model, mint_all = _refit_base(
                all_order_rows, metadata, order_entry, diagnostics, train_all)
            baseline = baseline_all[known]
            folds.append(dict(train=train, valid=valid, baseline=baseline,
                              trained=trained_heads,
                              common_model=common_model, mint=mint_all[known],
                              common_recipe=common_recipe, common_blend=common_blend,
                              baseline_metrics=_metrics(y[valid], baseline[valid], w[valid])))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return disabled, dict(report, reason='forward_base_or_input_failed', detail=str(error))
    eligible = []
    candidate_models = {}
    candidate_predictions = {}
    automatic = configurations is None
    stop_l1, stop_rms = ((PHYSICS_LINEAR_RESIDUAL_L1, PHYSICS_LINEAR_RESIDUAL_RMS)
                         if metadata['catalog'] == 9 else (LINEAR_RESIDUAL_L1, LINEAR_RESIDUAL_RMS))
    configurations = (LINEAR_CONFIGURATION,)+DEFAULT_CONFIGURATIONS if automatic else configurations
    if not 1 <= len(configurations) <= MAX_CONFIGURATIONS:
        return disabled, dict(report, reason='neural_recipe_budget_exceeded')
    for configuration in configurations:
        if metadata['catalog'] == 9 and configuration.get('family') == 'linear':
            configuration = dict(configuration, column_scale=True, ridge=1.e-4)
        if automatic and configuration.get('family') != 'linear' and eligible:
            linear = min(eligible, key=lambda c: c['mean_l1'])
            if all(m['relative_l1'] is not None and m['relative_rms'] is not None and
                   m['relative_l1'] <= stop_l1 and
                   m['relative_rms'] <= stop_rms for m in linear['folds']):
                report['neural_screen'] = 'compact_linear_residual_small'
                break
            report['neural_screen'] = 'remaining_forward_residual'
        predictions = []
        models = []
        input_bank = 'compact' if configuration.get('family') == 'linear' else bank
        try:
            for fold in folds:
                tr = fold['train']
                descriptors = descriptors_for(metadata, x, p, q, tr, heads, input_bank)
                inputs = input_values(descriptors, x, p, q, metadata['incoming'])
                if not inputs.shape[1]:
                    raise ValueError('no conditional inputs')
                absolute = configuration.get('family') == 'linear'
                target = (y-fold['baseline']*(1-represented) if absolute else y-fold['baseline'])
                model = fit(inputs[tr], mixing[tr], b[tr], target[tr], w[tr],
                            heads, fold['trained'], configuration)
                if absolute:
                    model['replaces_common'] = True
                model['descriptors'] = descriptors
                correction = (b*model['target_scale']/model['born_scale'])*np.sum(
                    mixing*forward(model, inputs), axis=1)
                if absolute:
                    correction -= fold['baseline']*represented
                predictions.append(correction)
                models.append(model)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            report['candidates'].append(dict(configuration=configuration, eligible=False,
                                            reason='fit_failed', detail=str(error)))
            continue
        for blend in polynomial.BLENDS:
            metrics, gains, errors = [], [], []
            for fold, correction in zip(folds, predictions):
                va = fold['valid']
                prediction = fold['baseline'][va]+blend*correction[va]
                metric = _metrics(y[va], prediction, w[va])
                metric['baseline'] = fold['baseline_metrics']
                metric['common_recipe'] = fold['common_recipe']
                metric['common_blend'] = fold['common_blend']
                metrics.append(metric)
                gain = np.abs(y[va]-fold['baseline'][va])-np.abs(y[va]-prediction)
                ww = w[va]/w[va].sum()
                gains.append(float(ww@gain))
                errors.append(float(np.linalg.norm(ww*gain)))
            candidate = dict(configuration=dict(configuration), blend=blend,
                             folds=metrics, mean_l1=float(np.mean([m['l1'] for m in metrics])),
                             **_forward_validation(metrics, gains, errors))
            report['candidates'].append(candidate)
            if candidate['eligible']:
                eligible.append(candidate)
                candidate_models[id(candidate)] = models
                candidate_predictions[id(candidate)] = predictions
    if not eligible:
        return disabled, dict(report, reason='forward_validation_prefers_common_model')
    if automatic:
        linear = [c for c in eligible if c['configuration'].get('family') == 'linear']
        if linear:
            reference = min(linear, key=lambda c: c['mean_l1'])
            guarded = list(linear)
            for candidate in eligible:
                if candidate['configuration'].get('family') == 'linear':
                    continue
                metrics, gains, errors = [], [], []
                for fold, old, new, metric, baseline in zip(
                        folds, candidate_predictions[id(reference)], candidate_predictions[id(candidate)],
                        candidate['folds'], reference['folds']):
                    va = fold['valid']
                    ww = w[va]/w[va].sum()
                    old_residual = y[va]-fold['baseline'][va]-reference['blend']*old[va]
                    new_residual = y[va]-fold['baseline'][va]-candidate['blend']*new[va]
                    delta = np.abs(old_residual)-np.abs(new_residual)
                    metrics.append(dict(metric, baseline=baseline))
                    gains.append(float(ww@delta))
                    errors.append(float(np.linalg.norm(ww*delta)))
                comparison = _forward_validation(metrics, gains, errors)
                candidate['versus_linear'] = comparison
                if comparison['eligible']:
                    guarded.append(candidate)
            eligible = guarded
    selected = min(eligible, key=lambda c: c['mean_l1'])
    if automatic:
        simpler = [c for c in eligible if c['configuration'].get('family') == 'linear' and
                   c['mean_l1'] <= 1.05*selected['mean_l1'] and
                   all(a['rms'] <= 1.05*b['rms'] for a, b in zip(c['folds'], selected['folds']))]
        if simpler:
            selected = min(simpler, key=lambda c: c['mean_l1'])
    for fold, fitted in zip(folds, candidate_models[id(selected)]):
        fold['descriptors'] = fitted['descriptors']
    if permutation is not None:
        report['symmetry'] = _validate_symmetry(subset, metadata, x, p, q, b, y, w,
            folds, candidate_models[id(selected)], selected['blend'], permutation)
    allrows = np.ones(len(subset), dtype=bool)
    mint = polynomial.mint_predictions(all_order_rows, np.ones(len(all_order_rows), dtype=bool))[known]
    baseline = polynomial.predict(order_entry, x, b, mint)
    try:
        selected_bank = 'compact' if selected['configuration'].get('family') == 'linear' else bank
        descriptors = descriptors_for(metadata, x, p, q, allrows, heads, selected_bank)
        inputs = input_values(descriptors, x, p, q, metadata['incoming'])
        # Keep the validated architecture fixed. Extra all-data observations
        # must not activate a rare head absent from the earliest fitting fold.
        trained = trained_heads
        absolute = selected['configuration'].get('family') == 'linear'
        target = y-baseline*(1-represented) if absolute else y-baseline
        result = fit(inputs, mixing, b, target, w, heads, trained, selected['configuration'])
        if absolute:
            result['replaces_common'] = True
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return disabled, dict(report, reason='final_fit_failed', detail=str(error))
    result.update(order=order_entry['order'], enabled=True, descriptors=descriptors)
    if result.get('replaces_common', False):
        result['replacement_blend'] = selected['blend']
    else:
        result['target_scale'] *= selected['blend']
    trained_native_heads = [h for h, flag in zip(heads, trained) if flag]
    report.update(reason='selected_by_three_forward_folds', selected=selected,
                  trained_heads=trained_native_heads,
                  untrained_native_heads=[h[0] for h in metadata['helicities'] if h[1] and
                                          h[0] not in trained_native_heads],
                  parameters=len(result['theta']))
    return result, report


def train(rows, metadata, orders, diagnostics, base_path, output_path, configurations=None, bank='scalar', symmetry=True):
    """Write an optional companion only if at least one order passes validation."""
    from .virtual_pool import _atomic_write
    started = time.monotonic()
    cpu_started = time.process_time()
    report = dict(model_written=False, enabled=False, format_version=FORMAT_VERSION,
                  orders=[], additional_loop_calls=0, label_policy='native selected/full branches only')
    def finished(reason=None):
        if reason is not None:
            report['reason'] = reason
        report['training_seconds'] = time.monotonic()-started
        report['training_cpu_seconds'] = time.process_time()-cpu_started
        return report
    if os.path.exists(output_path):
        os.unlink(output_path)
    if metadata.get('version') not in (3, 4) or not metadata.get('nhel'):
        return finished('native_helicity_metadata_unavailable')
    try:
        import numpy
        import scipy.optimize
    except ImportError:
        return finished('numpy_or_scipy_unavailable')
    fitted = []
    permutation = permutation_candidate(metadata) if symmetry else None
    for order, diagnostic in zip(orders, diagnostics):
        model, detail = train_order(rows, metadata, order, diagnostic, configurations, bank, permutation)
        fitted.append(model)
        report['orders'].append(detail)
    if any(entry['enabled'] for entry in fitted):
        with open(base_path, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        model = dict(metadata=metadata, base_sha256=digest, orders=fitted)
        if permutation is not None:
            active = {r['order'] for r in rows if r['virtual'] != 0.}
            chosen = bool(active) and all(detail.get('symmetry', {}).get('eligible', False)
                for detail in report['orders'] if detail['order'] in active)
            report['symmetry'] = dict(permutation=permutation, selected=chosen,
                reason='all_active_orders_passed' if chosen else 'active_order_validation_rejected')
            if chosen:
                model['permutations'] = [list(range(1, metadata['nlegs']+1)), permutation]
                report['format_version'] = 2
        payload = model_text(model)
        report['format_version'] = int(payload.splitlines()[0].split()[1])
        _atomic_write(output_path, payload)
        report.update(model_written=True, enabled=True, reason='validated_conditional_correction',
                      base_sha256=digest, model_sha256=hashlib.sha256(payload.encode('ascii')).hexdigest())
    else:
        report['reason'] = 'all_orders_rejected'
    return finished()
