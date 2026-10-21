"""Refine the native virtual surrogate using committed pooled observations.

The physical surrogate is the first fit. Small channel deviations share its
coefficients, and a compact polynomial only learns the remaining residual.
Selection refits this entire sequence on earlier whole iterations. Captured
contemporaneous subtractions are used for reconstruction, never as labels.
"""
from .data import (
    MAX_SURROGATE_FEATURES as MAX_FEATURES,
    MAX_SURROGATE_CHANNELS as MAX_CHANNELS,
)
from .validation import (
    FORWARD_FRACTIONS as FRACTIONS, chronological_mask,
    residual_metrics as _metrics, point_key, validation_cache,
)

MIN_ROWS = 60
MAX_DESIGN_ELEMENTS = 40000000
SHARED_RIDGE = 1.e-8
CHANNEL_RIDGE = .03


def _arrays(rows):
    import numpy as np
    return tuple(np.asarray([r[key] for r in rows], dtype=float)
                 for key in ('surrogate', 'born', 'virtual', 'weight'))


def _solve(matrix, target, weight, ridge):
    """Fit absolute weighted virtuals; no target is divided by its Born."""
    import numpy as np
    w = weight/weight.sum()
    scale = np.sqrt(w @ (matrix*matrix))
    active = scale > max(float(scale.max())*1.e-10, 1.e-100)
    result = np.zeros(matrix.shape[1])
    if not np.any(active):
        return result
    a = matrix[:, active]/scale[active]
    ys = max(float(w @ np.abs(target)), 1.e-100)
    gram = a.T @ (w[:, None]*a)
    gram.flat[::len(gram)+1] += ridge
    result[active] = np.linalg.solve(gram, a.T @ (w*target/ys))*ys/scale[active]
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite pooled surrogate coefficients')
    return result


def fit_basis(rows):
    """Shared physical coefficients, with ridge-shrunk channel deviations."""
    import numpy as np
    x, b, y, w = _arrays(rows)
    usable = np.asarray([r['surrogate_valid'] for r in rows], dtype=bool)
    if (x.ndim != 2 or not 4 <= x.shape[1] <= MAX_FEATURES or
            usable.sum() < max(MIN_ROWS, 3*x.shape[1])):
        raise ValueError('insufficient surrogate observations')
    if x.size > MAX_DESIGN_ELEMENTS:
        raise ValueError('pooled surrogate design budget exceeded')
    channels = sorted({r['channel'] for r in rows})
    if len(channels) > MAX_CHANNELS:
        raise ValueError('pooled surrogate channel budget exceeded')
    matrix = b[:, None]*x
    shared = _solve(matrix[usable], y[usable], w[usable], SHARED_RIDGE)
    residual = y-matrix@shared
    entries = []
    for channel in channels:
        selected = usable & np.asarray([r['channel'] == channel for r in rows])
        offset = np.zeros(x.shape[1])
        if selected.sum() >= max(MIN_ROWS, 3*x.shape[1]):
            offset = _solve(matrix[selected], residual[selected], w[selected], CHANNEL_RIDGE)
        entries.append(dict(channel=channel, coefficient=offset.tolist(), blend=1.))
    return dict(nfeatures=x.shape[1], shared=shared.tolist(), channels=entries)


def basis_prediction(entry, surrogate, born, channels):
    import numpy as np
    x, b, channel = (np.asarray(a) for a in (surrogate, born, channels))
    if x.shape != (len(b), entry['nfeatures']) or channel.shape != b.shape:
        raise ValueError('invalid surrogate prediction dimensions')
    value = x @ np.asarray(entry['shared'])
    blend = np.zeros(len(b))
    for item in entry['channels']:
        selected = channel == item['channel']
        value[selected] += x[selected] @ np.asarray(item['coefficient'])
        blend[selected] = item['blend']
    return b*value, blend


def native_predictions(rows, include):
    """Rebuild a frozen local surrogate and its MINT fallback without leakage.

    Each worker/channel is kept separate. The local physical fit and the grid
    both exclude held-out labels, including the decision between them. Native
    deployment retains its actual saved iteration fit as the fallback.
    """
    import numpy as np
    from .polynomial import mint_predictions
    answer = mint_predictions(rows, include)
    x, b, y, w = _arrays(rows)
    # Reconstruct the native basis for the local comparator: its colour
    # links contain g**2 and its Born carrier has no extra loop coupling.
    x = x.copy()
    x[:, 1:4] *= np.asarray([r['surrogate_gs2'] for r in rows])[:, None]
    b = np.asarray([r.get('mint_born', r['born']) for r in rows])
    matrix = b[:, None]*x
    histories = {}
    for i, row in enumerate(rows):
        histories.setdefault((row['directory'], row['channel']), []).append(i)
    for indices in histories.values():
        indices = np.asarray(indices)
        train = indices[np.asarray(include)[indices] &
                        np.asarray([rows[i]['surrogate_valid'] for i in indices], dtype=bool)]
        if len(train) < max(50, 3*x.shape[1]):
            continue
        coefficient = _solve(matrix[train], y[train], w[train], SHARED_RIDGE)
        prediction = matrix[indices] @ coefficient
        fit_error = y[train]-matrix[train]@coefficient
        grid_error = y[train]-answer[train]
        if w[train] @ (fit_error*fit_error) < w[train] @ (grid_error*grid_error):
            good = np.asarray([rows[i]['surrogate_valid'] for i in indices], dtype=bool)
            answer[indices[good]] = prediction[good]
    return answer


def _recipe(rows, metadata, order, recipe):
    import numpy as np
    from . import polynomial
    from .data import scalar_feature_count, threshold_pairs
    rows = [row for row in rows if row['surrogate_valid']]
    basis = fit_basis(rows)
    x, b, y, w = _arrays(rows)
    base, _ = basis_prediction(basis, x, b, [r['channel'] for r in rows])
    entry = polynomial.empty_order(order, metadata['nf'], 'surrogate_only')
    features = np.asarray([r['features'] for r in rows])
    ns = scalar_feature_count(metadata)
    lower, upper = features.min(axis=0), features.max(axis=0)
    lower[ns:], upper[ns:] = -1., 1.
    entry.update(lower=lower.tolist(), upper=upper.tolist())
    if recipe != 'surrogate':
        linear_only = tuple(range(ns-5-len(threshold_pairs(metadata)), ns-1))
        degree, family = (1, 'full') if recipe == 'linear' else (2, 'compact')
        _, count = polynomial.basis_size(features, degree, ns, family, linear_only)
        if len(rows) < polynomial.ROWS_PER_TERM*count:
            raise ValueError('insufficient residual observations per coefficient')
        residual = y-base
        if np.any(residual):
            entry = polynomial.fit(features, b, residual, w, degree,
                polynomial.COMPACT_RIDGE, family, ns, linear_only=linear_only)
    entry.update(order=order, blend=1., surrogate=basis, unclipped=True)
    return entry


def prediction(entry, rows, native):
    import numpy as np
    from .polynomial import predict
    return predict(entry, np.asarray([r['features'] for r in rows]),
                   np.asarray([r['born'] for r in rows]), np.asarray(native),
                   surrogate=np.asarray([r['surrogate'] for r in rows]),
                   channels=np.asarray([r['channel'] for r in rows]),
                   supported=np.asarray([r['surrogate_valid'] for r in rows], dtype=bool))


def _guard(y, old, new, weight, fixed_order):
    import numpy as np
    before, after = _metrics(y, old, weight), _metrics(y, new, weight)
    w = weight/weight.sum()
    scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(old))),
                float(np.max(np.abs(new))), 1.e-100)
    if fixed_order:
        delta = ((y-old)/scale)**2-((y-new)/scale)**2
        baseline = (before['rms']/scale)**2
    else:
        delta = np.abs((y-old)/scale)-np.abs((y-new)/scale)
        baseline = before['l1']/scale
    gain, error = float(w @ delta), float(np.linalg.norm(w*delta))
    return dict(baseline=before, candidate=after, gain=gain, gain_error=error,
                eligible=bool(gain > max(.02*baseline, 2*error) and
                              after['rms'] <= 1.05*before['rms'] and
                              after['l1'] <= 1.05*before['l1']))


def guard_joint_events(rows, common, conditional=None):
    """Check summed orders, folded S flavours and the selected exact residual.

    H-event weights do not depend on the approximation and cancel in this
    paired comparison. Inverse loop probability gives the expected residual
    negative weight without including an additional helicity factor.
    """
    import numpy as np
    orders = sorted({r['order'] for r in rows})
    report = dict(eligible=True, folds=[], objective='negative folded S plus separate virtual events')
    for index, fraction in enumerate(FRACTIONS):
        points = {}
        for order in orders:
            subset = [r for r in rows if r['order'] == order]
            early = chronological_mask(subset, fraction)
            cached = common.get(order, [])
            records = cached[index] if len(cached) > index else {}
            heads = (conditional or {}).get(order, [])
            head_records = heads[index] if len(heads) > index else {}
            missing = any(not early[i] and point_key(row) not in records for i, row in enumerate(subset))
            native = native_predictions(subset, early) if missing else None
            for i, row in enumerate(subset):
                if early[i]:
                    continue
                key = point_key(row)
                if key in records:
                    old, selected, mean = records[key]
                else:
                    old = selected = mean = native[i]
                if key in head_records:
                    _, selected, mean = head_records[key]
                entry = points.setdefault(key, dict(row=row, old=0., selected=0., mean=0., virtual=0.))
                entry['old'] += old
                entry['selected'] += selected
                entry['mean'] += mean
                entry['virtual'] += row['virtual']
        before, after, weights = [], [], []
        for entry in points.values():
            row = entry['row']
            soft, flavour = row.get('event_soft', ()), row.get('event_flavour', ())
            if not flavour:
                if entry['selected'] != entry['old'] or entry['mean'] != entry['old']:
                    return dict(report, eligible=False, reason='changed_point_without_supported_event_groups')
                continue
            soft, flavour = np.asarray(soft), np.asarray(flavour)
            def negative(actual, mean):
                return float(np.maximum(-(soft+flavour*mean), 0).sum()+
                             np.maximum(-flavour*(entry['virtual']-actual), 0).sum())
            before.append(negative(entry['old'], entry['old']))
            after.append(negative(entry['selected'], entry['mean']))
            weights.append(row['weight'])
        if not weights:
            return dict(report, eligible=False, reason='insufficient_forward_event_groups')
        weight = np.asarray(weights)
        weight /= weight.sum()
        baseline, candidate = float(weight @ before), float(weight @ after)
        error = float(np.linalg.norm(weight*(np.asarray(before)-np.asarray(after))))
        eligible = candidate <= baseline+1.e-12*max(baseline, candidate, 1.e-100)
        report['folds'].append(dict(fraction=fraction, samples=len(weights),
                                   baseline_negative=baseline, candidate_negative=candidate,
                                   paired_error=error, eligible=bool(eligible)))
        report['eligible'] = bool(report['eligible'] and eligible)
    report['reason'] = 'passed' if report['eligible'] else 'negative_event_weight_increased'
    return report


def fit_order(rows, metadata, order, fixed_order=False, validation=None):
    """Select a complete baseline/refinement on three forward iteration splits."""
    import numpy as np
    from . import polynomial
    subset = [r for r in rows if r['order'] == order]
    report = dict(order=order, training_rows=len(subset), candidates=[],
                  strategy='surrogate_residual', validation_fractions=list(FRACTIONS),
                  fixed_order=bool(fixed_order),
                  comparator='local surrogate and MINT refitted on earlier whole iterations')
    fallback = lambda reason: (polynomial.empty_order(order, metadata['nf'], reason), report)
    if len(subset) < 2*MIN_ROWS:
        return fallback('insufficient_surrogate_history')
    _, _, y, w = _arrays(subset)
    if not np.any(y) or not any(r['born'] for r in subset):
        return fallback('zero_born_or_virtual_order')
    folds = []
    for fraction in FRACTIONS:
        train = chronological_mask(subset, fraction)
        valid = ~train
        if train.sum() < MIN_ROWS or valid.sum() < 10:
            return fallback('insufficient_distinct_iteration_history')
        try:
            native = native_predictions(subset, train)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return fallback('local_surrogate_refit_failed')
        folds.append(dict(train=train, valid=valid, native=native,
                          rows=[r for r, take in zip(subset, train) if take]))
    if validation is not None:
        validation[order] = validation_cache(subset, folds)
    eligible = []
    saved = {}
    for recipe in ('surrogate', 'linear', 'quadratic'):
        proposals = []
        try:
            for fold in folds:
                model = _recipe(fold['rows'], metadata, order, recipe)
                proposals.append(prediction(model, subset, fold['native']))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            report['candidates'].append(dict(recipe=recipe, eligible=False, reason=str(error)))
            continue
        for blend in polynomial.BLENDS:
            guards = []
            for fold, proposed in zip(folds, proposals):
                va = fold['valid']
                trial = fold['native']+blend*(proposed-fold['native'])
                guards.append(_guard(y[va], fold['native'][va], trial[va], w[va], fixed_order))
            candidate = dict(recipe=recipe, blend=blend, folds=guards,
                             eligible=all(g['eligible'] for g in guards),
                             mean_l1=float(np.mean([g['candidate']['l1'] for g in guards])),
                             mean_rms=float(np.mean([g['candidate']['rms'] for g in guards])))
            # Aggregate RMS cannot increase even though individual folds have
            # a small tolerance for the uncertainty of their error estimates.
            candidate['eligible'] = bool(candidate['eligible'] and
                sum(g['candidate']['rms']**2 for g in guards) <=
                sum(g['baseline']['rms']**2 for g in guards))
            report['candidates'].append(candidate)
            if candidate['eligible']:
                eligible.append(candidate)
                saved[id(candidate)] = proposals
        if eligible:
            best = min(eligible, key=lambda c: c['mean_rms'] if fixed_order else c['mean_l1'])
            target = max(float((w/w.sum()) @ np.abs(y)), 1.e-100)
            if best['mean_l1']/target <= .002 and best['mean_rms']/target <= .01:
                report['richer_screen'] = 'surrogate_residual_small'
                break
    if not eligible:
        return fallback('forward_validation_prefers_local_surrogate')
    objective = 'mean_rms' if fixed_order else 'mean_l1'
    selected = min(eligible, key=lambda c: c[objective])
    # Keep the earliest model within 2% of the best validated score.
    selected = next(c for c in eligible if c[objective] <= 1.02*selected[objective] and
                    c['mean_rms'] <= 1.05*selected['mean_rms'])
    try:
        entry = _recipe(subset, metadata, order, selected['recipe'])
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return fallback('final_surrogate_refinement_failed')
    entry.update(blend=selected['blend'], reason='selected_by_forward_surrogate_validation')
    # Channels with a measured degradation keep their native frozen fit.
    # The shared model remains available for the validated channels only.
    for channel in entry['surrogate']['channels']:
        supported = True
        observed = False
        for fold, proposed in zip(folds, saved[id(selected)]):
            mask = fold['valid'] & np.asarray([r['channel'] == channel['channel'] for r in subset])
            if mask.sum() < 10:
                supported = False
                continue
            observed = True
            trial = fold['native']+entry['blend']*(proposed-fold['native'])
            before, after = (_metrics(y[mask], a[mask], w[mask]) for a in (fold['native'], trial))
            supported = supported and after['rms'] <= 1.05*before['rms'] and after['l1'] <= before['l1']
        channel['blend'] = 1. if supported and observed else 0.
    if not any(c['blend'] for c in entry['surrogate']['channels']):
        return fallback('channel_validation_prefers_local_surrogate')
    report.update(selected=selected, actual_terms=len(entry['terms']),
                  shared_surrogate_features=entry['surrogate']['nfeatures'],
                  channel_deviation_ridge=CHANNEL_RIDGE,
                  channels=[dict(channel=c['channel'], enabled=bool(c['blend']))
                            for c in entry['surrogate']['channels']])
    if validation is not None:
        active = {c['channel']: c['blend'] for c in entry['surrogate']['channels']}
        channel_blend = np.asarray([active.get(r['channel'], 0.) for r in subset])
        promoted = [fold['native']+entry['blend']*channel_blend*(proposed-fold['native'])
                    for fold, proposed in zip(folds, saved[id(selected)])]
        validation[order] = validation_cache(subset, folds, promoted)
    return entry, report
