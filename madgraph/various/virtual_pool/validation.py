"""Shared point grouping, chronological splits and residual diagnostics."""
import math

FORWARD_FRACTIONS = (.5, 2./3., .8)


def chronological_mask(rows, fraction):
    """Keep whole iterations; histories shorter than four only train fits."""
    import numpy as np
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row['directory'], row['channel']), []).append(index)
    mask = np.zeros(len(rows), dtype=bool)
    for indices in groups.values():
        levels = sorted({rows[i]['iteration'] for i in indices})
        count = len(levels)
        if len(levels) >= len(FORWARD_FRACTIONS)+1:
            rank = FORWARD_FRACTIONS.index(fraction)
            count = max(rank+1, min(len(levels)-len(FORWARD_FRACTIONS)+rank,
                                   int(math.ceil(fraction*len(levels)))))
        cutoff = levels[count-1]
        mask[indices] = [rows[i]['iteration'] <= cutoff for i in indices]
    return mask


def residual_metrics(y, prediction, weight):
    """Physical weighted residual norms, target-relative errors and negatives."""
    import numpy as np
    scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(prediction))), 1.e-100)
    w = weight/weight.sum()
    residual = (y-prediction)/scale
    target = y/scale
    l1, second = float(w @ np.abs(residual)), float(w @ (residual*residual))
    target_l1, target_second = float(w @ np.abs(target)), float(w @ (target*target))
    return dict(l1=l1*scale, rms=math.sqrt(second)*scale,
                maximum=float(np.max(np.abs(residual)))*scale,
                negative=float(w @ np.maximum(-residual, 0))*scale,
                relative_l1=l1/target_l1 if target_l1 else None,
                relative_rms=math.sqrt(second/target_second) if target_second else None)


def point_key(row):
    return row['directory'], row['channel'], row['iteration'], row['sample']


def validation_cache(rows, folds, selected=None, means=None):
    """Temporary predictions for the joint order/flavour promotion guard."""
    cache = []
    for index, fold in enumerate(folds):
        old = fold['native']
        actual = old if selected is None else selected[index]
        mean = actual if means is None else means[index]
        cache.append({point_key(row): (float(old[i]), float(actual[i]), float(mean[i]))
                      for i, row in enumerate(rows) if fold['valid'][i]})
    return cache


def retain_native_predictions(validation):
    """Discard rejected refinements while keeping their matched comparators."""
    return {order: [{key: (native, native, native) for key, (native, _, _) in fold.items()}
                    for fold in cache] for order, cache in validation.items()}
