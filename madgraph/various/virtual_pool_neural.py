"""Optional frozen conditional corrections to pooled virtual polynomials.

Only native sampled/full-branch observations are fitted. Native probabilities
already include helicity multiplicities and are used exactly once in a mean.
SciPy is an optional training dependency; inference needs only NumPy.
"""
import hashlib
import itertools
import math
import os
import time

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


def _dot(a, b):
    return a[..., 0]*b[..., 0] - (a[..., 1:]*b[..., 1:]).sum(axis=-1)


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
        if kind in (1, 2):
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
        else:
            raise ValueError('unknown neural input descriptor')
        if kind in (1, 4):
            scaled, tail = coordinates(value[:, None], [lower], [upper])
            value = scaled[:, 0]
            outside |= tail
        columns.append(value)
    result = np.column_stack(columns) if columns else np.zeros((n, 0))
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite neural inputs')
    return (result, outside) if return_outside else result


def descriptors_for(metadata, features, momenta, q, train, heads, bank='scalar'):
    import numpy as np
    from .virtual_pool import scalar_feature_count
    if bank not in ('scalar', 'canonical', 'composite'):
        raise ValueError('unknown neural input bank')
    ns = scalar_feature_count(metadata) if metadata['catalog'] == 7 else metadata['nf']
    lo, hi = features[train].min(axis=0), features[train].max(axis=0)
    descriptors = [(1, i+1, 0, 0, float(lo[i]), float(hi[i])) for i in range(ns)
                   if hi[i]-lo[i] > 1.e-12]
    descriptors += [(2, i+1, 0, 0, -1., 1.) for i in range(ns, metadata['nf'])]
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
    from .virtual_pool import scalar_feature_count
    x = np.asarray(features).copy()
    p = np.asarray(momenta)[:, np.asarray(permutation)-1]
    pairs = list(itertools.combinations(range(metadata['nlegs']), 2))
    indices = {pair: index+1 for index, pair in enumerate(pairs)}
    for index, (a, b) in enumerate(pairs, 1):
        pair = tuple(sorted((permutation[a]-1, permutation[b]-1)))
        x[:, index] = np.asarray(features)[:, indices[pair]]
    if metadata['catalog'] == 7:
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


def objective(theta, inputs, mixing, born, target, weight, widths, ridge, trained):
    """Direct weighted amplitude residual and analytic gradient (one/two layers)."""
    import numpy as np
    parts = unpack(theta, inputs.shape[1], mixing.shape[1], widths)
    activations = [inputs]
    for layer in range(len(widths)):
        activations.append(np.tanh(activations[-1]@parts[2*layer] + parts[2*layer+1]))
    affine = np.column_stack((inputs, np.ones(len(inputs))))
    allheads = activations[-1]@parts[-3] + parts[-2] + affine@parts[-1]
    effective = mixing*np.asarray(trained)[None, :]
    residual = born*np.sum(allheads*effective, axis=1)-target
    dc = (weight*residual*born)[:, None]*effective
    gradients = [None]*len(parts)
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


def fit(inputs, mixing, born, target, weight, heads, trained, configuration):
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
    args = (inputs, mixing, carrier, target/vscale, w/noise, widths, configuration['ridge'], trained)
    result = minimize(objective, theta, args=args, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=configuration['iterations'], maxcor=20,
                                   ftol=1.e-12, gtol=1.e-8, maxls=30))
    if not np.all(np.isfinite(result.x)) or not math.isfinite(result.fun):
        raise ValueError('nonfinite neural optimization')
    return dict(widths=list(widths), heads=list(map(int, heads)), trained=list(map(int, trained)),
                theta=result.x.tolist(), born_scale=bscale, target_scale=vscale,
                optimizer_iterations=int(result.nit), optimizer_status=str(result.message))


def predict_correction(entry, metadata, features, momenta, q, born, selected=None, full=None):
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
    permutations = permutations or [list(range(1, metadata['nlegs']+1))]
    selected_total = np.zeros(len(born)); mean_total = np.zeros(len(born))
    identity_common = None
    for permutation in permutations:
        permutation = list(permutation)
        if permutation == list(range(1, metadata['nlegs']+1)):
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
        for descriptor in descriptors:
            if len(descriptor) != 6:
                raise ValueError('invalid neural descriptor width')
            kind, a, b, c, lo, hi = descriptor
            if any(int(v) != v for v in (kind, a, b, c)) or not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
                raise ValueError('invalid neural descriptor metadata')
            if (kind not in range(1, 7) or
                    (kind in (1, 2) and (not 1 <= a <= metadata['nf'] or b or c)) or
                    (kind == 3 and (not 1 <= a <= metadata['nlegs'] or b not in range(4) or c or
                                    metadata['incoming'][a-1])) or
                    (kind in (4, 5) and (len({a, b, c}) != 3 or min(a, b, c) < 1 or max(a, b, c) > metadata['nlegs'])) or
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
    version = 2 if len(model.get('permutations', [])) > 1 else 1
    lines = ['VIRTUAL_POOL_NEURAL %d' % version,
             ' '.join(str(metadata[k]) for k in ('catalog', 'nf', 'norders', 'nlegs', 'nx', 'nhel')),
             ' '.join(map(str, metadata['pdgs'])), floats(metadata['masses']),
             ' '.join(map(str, metadata['incoming'])), model['base_sha256']]
    lines += [' '.join(map(str, h)) for h in metadata['helicities']]
    if version == 2:
        lines.append(str(len(model['permutations'])))
        lines += [' '.join(map(str, p)) for p in model['permutations']]
    for entry in model['orders']:
        if not entry.get('enabled', False):
            lines.append('%d 0 0 0 0 0 1 1' % entry['order'])
            continue
        widths = entry['widths']
        lines.append('%d 1 %d %d %d %d %.17e %.17e' % (
            entry['order'], len(entry['descriptors']), len(entry['heads']), widths[0],
            widths[1] if len(widths) == 2 else 0, entry['born_scale'], entry['target_scale']))
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
        if len(header) != 2 or header[0] != 'VIRTUAL_POOL_NEURAL' or header[1] not in ('1', '2'):
            raise ValueError('unsupported neural companion format')
        version = int(header[1])
        dims = stream.readline().split()
        if len(dims) != 6:
            raise ValueError('invalid neural dimensions')
        particle_lines = [stream.readline() for _ in range(3)]
        metadata = _header(io.StringIO('VIRTUAL_POOL_MODEL 3\n'+' '.join(dims[:5])+'\n'+''.join(particle_lines)), 'VIRTUAL_POOL_MODEL')
        metadata['nhel'] = int(dims[5])
        if not 0 <= metadata['nhel'] <= MAX_NATIVE_HELICITIES:
            raise ValueError('invalid neural native map size')
        digest = stream.readline().strip()
        metadata['helicities'] = tuple(tuple(int(v) for v in stream.readline().split())
                                       for _ in range(metadata['nhel']))
        permutations = None
        if version == 2:
            fields = stream.readline().split()
            if len(fields) != 1 or int(fields[0]) not in (1, 2):
                raise ValueError('invalid neural permutation count')
            permutations = [tuple(int(v) for v in stream.readline().split()) for _ in range(int(fields[0]))]
        orders = []
        for number in range(1, metadata['norders']+1):
            fields = stream.readline().split()
            if len(fields) != 8:
                raise ValueError('invalid neural order header')
            order, enabled, ni, nh, w1, w2 = (int(v) for v in fields[:6])
            bs, vs = _floats(' '.join(fields[6:]))
            if order != number or enabled not in (0, 1):
                raise ValueError('invalid neural order number/enabled flag')
            entry = dict(order=order, enabled=bool(enabled), born_scale=bs, target_scale=vs)
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
                descriptors.append(tuple(int(v) for v in fields[:4])+_floats(' '.join(fields[4:])))
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
    fitted, report = polynomial.fit_order(earlier, _folds(earlier, 3), metadata,
                                         order_entry['order'])
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
    metrics = result['folds']
    baseline_l1 = float(np.mean([m['baseline']['l1'] for m in metrics]))
    gain, error = float(np.mean(gains)), max(errors)
    stable = all(m['l1'] < m['baseline']['l1'] and
                 m['rms'] <= 1.05*m['baseline']['rms'] for m in metrics)
    variance = np.mean([m['rms']**2 for m in metrics]) <= np.mean([m['baseline']['rms']**2 for m in metrics])
    accepted = stable and variance and gain > max(.02*baseline_l1, 2*error)
    result.update(eligible=bool(accepted), mean_l1_gain=gain, paired_gain_error=error,
                  reason='selected_by_three_forward_folds' if accepted else 'forward_validation_rejected')
    return result


def train_order(rows, metadata, order_entry, diagnostics, configurations=None, bank='scalar', permutation=None):
    """Three forward fits select a correction; coefficients never see later rows.

    The common polynomial recipe is selected within each earlier subset by its
    own guarded CV. Every coefficient, input bound and head eligibility uses
    only the earlier part of each neural fold. No diagnostic fullsum labels are
    generated or inferred. Full native branches use the q-mean in the loss.
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
            descriptors = descriptors_for(metadata, x, p, q, train, heads, bank)
            inputs = input_values(descriptors, x, p, q, metadata['incoming'])
            if not inputs.shape[1]:
                return disabled, dict(report, reason='no_neural_inputs')
            folds.append(dict(train=train, valid=valid, baseline=baseline,
                              descriptors=descriptors, inputs=inputs,
                              trained=trained_heads,
                              common_model=common_model, mint=mint_all[known],
                              common_recipe=common_recipe, common_blend=common_blend,
                              baseline_metrics=_metrics(y[valid], baseline[valid], w[valid])))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return disabled, dict(report, reason='forward_base_or_input_failed', detail=str(error))
    eligible = []
    candidate_models = {}
    configurations = DEFAULT_CONFIGURATIONS if configurations is None else configurations
    if not 1 <= len(configurations) <= MAX_CONFIGURATIONS:
        return disabled, dict(report, reason='neural_recipe_budget_exceeded')
    for configuration in configurations:
        predictions = []
        models = []
        try:
            for fold in folds:
                tr = fold['train']
                model = fit(fold['inputs'][tr], mixing[tr], b[tr],
                            y[tr]-fold['baseline'][tr], w[tr], heads, fold['trained'], configuration)
                correction = (b*model['target_scale']/model['born_scale'])*np.sum(
                    mixing*forward(model, fold['inputs']), axis=1)
                predictions.append(correction)
                models.append(model)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            report['candidates'].append(dict(configuration=configuration, eligible=False,
                                            reason='fit_failed', detail=str(error)))
            continue
        for blend in (.25, .5, .75, 1.):
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
            base_l1 = np.mean([m['baseline']['l1'] for m in metrics])
            gain = float(np.mean(gains))
            # Forward validation sets overlap. Do not count them as independent
            # replicates to shrink uncertainty: retain the largest paired SE.
            error = max(errors)
            stable = all(m['l1'] < m['baseline']['l1'] and
                         m['rms'] <= 1.05*m['baseline']['rms'] for m in metrics)
            aggregate_rms = np.mean([m['rms']**2 for m in metrics]) <= np.mean([
                m['baseline']['rms']**2 for m in metrics])
            accept = stable and aggregate_rms and gain > max(.02*base_l1, 2*error)
            candidate = dict(configuration=dict(configuration), blend=blend,
                             folds=metrics, mean_l1=float(np.mean([m['l1'] for m in metrics])),
                             mean_l1_gain=gain, paired_gain_error=error, eligible=bool(accept))
            report['candidates'].append(candidate)
            if accept:
                eligible.append(candidate)
                candidate_models[id(candidate)] = models
    if not eligible:
        return disabled, dict(report, reason='forward_validation_prefers_common_model')
    selected = min(eligible, key=lambda c: c['mean_l1'])
    if permutation is not None:
        report['symmetry'] = _validate_symmetry(subset, metadata, x, p, q, b, y, w,
            folds, candidate_models[id(selected)], selected['blend'], permutation)
    allrows = np.ones(len(subset), dtype=bool)
    mint = polynomial.mint_predictions(all_order_rows, np.ones(len(all_order_rows), dtype=bool))[known]
    baseline = polynomial.predict(order_entry, x, b, mint)
    try:
        descriptors = descriptors_for(metadata, x, p, q, allrows, heads, bank)
        inputs = input_values(descriptors, x, p, q, metadata['incoming'])
        # Keep the validated architecture fixed. Extra all-data observations
        # must not activate a rare head absent from the earliest fitting fold.
        trained = trained_heads
        result = fit(inputs, mixing, b, y-baseline, w, heads, trained, selected['configuration'])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return disabled, dict(report, reason='final_fit_failed', detail=str(error))
    result.update(order=order_entry['order'], enabled=True, descriptors=descriptors)
    result['target_scale'] *= selected['blend']
    report.update(reason='selected_by_three_forward_folds', selected=selected,
                  trained_heads=[h for h, flag in zip(heads, trained) if flag],
                  untrained_native_heads=[h[0] for h in metadata['helicities'] if h[1] and
                                          h[0] not in [head for head, flag in zip(heads, trained) if flag]],
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
    if metadata.get('version') != 3 or not metadata.get('nhel'):
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
        _atomic_write(output_path, payload)
        report.update(model_written=True, enabled=True, reason='validated_conditional_correction',
                      base_sha256=digest, model_sha256=hashlib.sha256(payload.encode('ascii')).hexdigest())
    else:
        report['reason'] = 'all_orders_rejected'
    return finished()
