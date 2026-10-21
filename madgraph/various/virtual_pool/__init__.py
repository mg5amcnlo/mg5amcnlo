"""Train and freeze virtual refinements at the subprocess integration barrier.

The public data/codec helpers remain available here for existing callers.
Workers and controllers without NumPy retain the native surrogate fallback.
"""
import json
import os
import time

from .data import (
    SAMPLES, ITERATIONS, REPORT_FORMAT_VERSION, MODEL_FORMAT_VERSION,
    CAPTURE_FORMAT_VERSION, FEATURE_CATALOG, scalar_feature_count,
    threshold_pairs, threshold_features, _physics_rows, _floats, _header,
    _read_iterations, _read_iteration_history, _native_selection, _read_job,
    collect_samples, _atomic_write, _model_text, load_model,
)

__all__ = ['train_pool', 'load_model', 'predict', 'collect_samples']


def _fit_orders(rows, metadata, fixed_order, report):
    """Fit the common approximation and apply the joint event guard once."""
    from . import polynomial, surrogate
    validation = {}
    orders = []
    for order in range(1, metadata['norders']+1):
        entry, diagnostics = surrogate.fit_order(
            rows, metadata, order, fixed_order, validation if not fixed_order else None)
        orders.append(entry)
        diagnostics.update(blend=entry['blend'], reason=entry['reason'])
        report['orders'].append(diagnostics)

    if not fixed_order and any(o['blend'] for o in orders):
        report['event_guard'] = surrogate.guard_joint_events(rows, validation)
        if not report['event_guard']['eligible']:
            orders = [polynomial.empty_order(o['order'], metadata['nf'], 'joint_event_guard')
                      for o in orders]
            for diagnostic in report['orders']:
                diagnostic.update(blend=0., reason='joint_event_guard', joint_disabled=True)
            from .validation import retain_native_predictions
            validation = retain_native_predictions(validation)
    return orders, validation


def _fit_conditional(rows, metadata, orders, diagnostics, output_model, output_neural,
                     fixed_order, validation):
    """Freeze conditional outputs only after their complete event guard passes."""
    from . import helicity

    head_validation = {}
    result = helicity.train(
        rows, metadata, orders, diagnostics, output_model, output_neural,
        validation=head_validation if not fixed_order else None)
    if not fixed_order and result['enabled']:
        from .surrogate import guard_joint_events
        result['event_guard'] = guard_joint_events(rows, validation, head_validation)
        if not result['event_guard']['eligible']:
            if os.path.exists(output_neural):
                os.unlink(output_neural)
            result['rejected_model_sha256'] = result.pop('model_sha256', None)
            result.update(enabled=False, model_written=False, reason='joint_event_guard')
    return result


def train_pool(job_dirs, output_model, report_path=None, job_groups=None,
               neural=False, output_neural=None, fixed_order=False):
    """Fit a frozen subprocess model from committed first-stage observations.

    Missing NumPy or captures retain the native surrogate/grid fallback.
    Malformed captures raise ValueError. A written model with zero blends is
    a frozen fallback; an absent model keeps native adaptation active.
    """
    from . import polynomial

    started, cpu_started = time.monotonic(), time.process_time()
    if neural:
        output_neural = output_neural or os.path.join(os.path.dirname(os.path.abspath(output_model)),
                                                      'virtual_pool_neural.dat')
        if os.path.exists(output_neural):
            os.unlink(output_neural)
    metadata, rows, jobs = collect_samples(job_dirs, job_groups)
    report = dict(version=REPORT_FORMAT_VERSION, status='skipped', enabled=False, model_written=False,
                  family='surrogate_pooled_residual', accepted_rows=len(rows), jobs=jobs, orders=[],
                  accepted_samples=len({(r['directory'], r['channel'], r['iteration'], r['sample'])
                                        for r in rows}),
                  objective=('physical residual variance' if fixed_order else
                             'physical residual L1 with RMS guards'),
                  validation='three chronological whole-iteration splits; refit baseline and correction',
                  fallback='saved native iteration surrogate and MINT',
                  tail_margin=polynomial.TAIL_MARGIN, max_coefficients=polynomial.MAX_TERMS,
                  minimum_rows_per_coefficient=polynomial.ROWS_PER_TERM,
                  helicity_sampler='unchanged native Born importance sampler',
                  call_probability='adapt to measured residual; exact loop outside trained domain',
                  boundary_prediction='physical function continuation; exact residual correction')
    reason = 'no_sample_headers' if metadata is None else None
    if metadata is not None:
        report['metadata'] = metadata
        try:
            import numpy
        except ImportError:
            report.update(status='fallback')
            reason = 'numpy_unavailable'
    if reason:
        report['reason'] = reason
        if neural:
            report['neural'] = dict(model_written=False, enabled=False, reason=reason)
    else:
        metadata, rows, report['physics_inputs'] = _physics_rows(metadata, rows)
        report['metadata'] = metadata
        orders, validation = _fit_orders(rows, metadata, fixed_order, report)
        _atomic_write(output_model, _model_text(metadata, orders))
        report.update(model_written=True, model_format_version=MODEL_FORMAT_VERSION,
                      enabled=any(o['blend'] > 0 for o in orders))
        if neural:
            report['neural'] = _fit_conditional(rows, metadata, orders, report['orders'],
                output_model, output_neural, fixed_order, validation)
            report['enabled'] = report['enabled'] or report['neural']['enabled']
        report['status'] = 'trained' if report['enabled'] else 'fallback'
    if not report['model_written']:
        report['call_probability'] = 'adaptive native surrogate/grid residual'
    report['training_seconds'] = time.monotonic()-started
    report['training_cpu_seconds'] = time.process_time()-cpu_started
    if report_path is not None:
        _atomic_write(report_path, json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return report


def predict(model, order, features, born, mint, surrogate=None, channels=None, supported=None):
    """Evaluate the frozen convex blend with the same normalization as inputs."""
    import numpy as np
    from .polynomial import predict as polynomial_predict

    if not 1 <= order <= model['metadata']['norders']:
        raise ValueError('invalid pooled model order')
    entry = model['orders'][order - 1]
    x, born, mint = (np.asarray(values, dtype=float) for values in (features, born, mint))
    if (born.ndim != 1 or x.shape != (len(born), model['metadata']['nf']) or mint.shape != born.shape
            or not all(np.all(np.isfinite(v)) for v in (x, born, mint))):
        raise ValueError('invalid pooled prediction inputs')
    return polynomial_predict(entry, x, born, mint, surrogate=surrogate, channels=channels, supported=supported)
