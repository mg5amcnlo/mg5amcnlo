"""Percentages from madspace's systematics summary.

The summary madspace writes -- ``SystematicsCalculator::summary()``, stored in
``Events/<run>/info.json`` under "systematics" -- carries cross sections and
absolute uncertainties. Two places turn those into the percentages a person
reads:

* the run's own Systematics box
  (:meth:`launch.MadgraphProcess.log_systematics_summary`);
* the parameter-scan summary row (:meth:`launch.MadgraphProcess.get_result`).

They must not disagree -- the same run printing one number in its box and a
different one in the scan table would be worse than either convention on its
own -- so the arithmetic lives here once and both format what it returns.

The conventions, which are the part worth getting right:

* **scale** is measured against the nominal cross section, up as
  ``(max - nominal) / |nominal|`` and down as ``(nominal - min) / |nominal|``.
  Both come back as positive magnitudes; the caller supplies the sign. This
  matches what systematics.py wrote, so a scan column does not shift when a run
  moves from the legacy path to the native one. The absolute value is for an
  interference, whose nominal can be negative: the band stays [min, max] and
  the percentages stay positive.
* **PDF** uncertainties are absolute in the summary, and are measured against
  the set's own ``central`` (in absolute value, as for the scale) -- for a
  replicas set that is the replica mean, not the nominal member. An entry with no ``central`` had none that could be
  computed, and is skipped rather than measured against something else.

This module deliberately imports nothing: it is pure arithmetic over a dict, so
anything that reads a summary can use it without dragging in dependencies.
"""

from __future__ import absolute_import


def nominal_cross_section(summary):
    """The nominal cross section, or None if the run produced no events.

    ``summary()`` omits it when ``event_count`` is zero.
    """

    if not summary:
        return None
    value = (summary.get('nominal') or {}).get('cross_section')
    return value or None


def scale_percentages(summary, group='scale'):
    """``(up, down)`` percent for the scale envelope, or None.

    Positive magnitudes: ``up`` is how far above the nominal the envelope
    reaches, ``down`` how far below. ``group`` names the envelope in the
    summary: 'scale', or 'emission' (see emission_percentages).
    """

    nominal = nominal_cross_section(summary)
    if nominal is None:
        return None
    band = summary.get(group)
    if not band:
        return None
    low, high = band.get('min'), band.get('max')
    if low is None or high is None:
        return None
    return ((high - nominal) / abs(nominal) * 100.,
            (nominal - low) / abs(nominal) * 100.)


def emission_percentages(summary):
    """``(up, down)`` percent for the MLM emission scale envelope (madevent's
    alpsfact), or None, with the conventions of scale_percentages."""

    return scale_percentages(summary, 'emission')


def pdf_percentages(summary):
    """``[(entry, up, down), ...]``, one per PDF set that has an uncertainty.

    Sets whose uncertainty could not be computed -- a one-member set asked for
    its ``errorset``, say -- are left out entirely rather than reported as
    zero.
    """

    if nominal_cross_section(summary) is None:
        return []
    out = []
    for entry in summary.get('pdf') or []:
        up, down = entry.get('uncertainty_up'), entry.get('uncertainty_down')
        central = entry.get('central')
        if up is None or down is None or not central:
            continue
        out.append((entry, up / abs(central) * 100., down / abs(central) * 100.))
    return out
