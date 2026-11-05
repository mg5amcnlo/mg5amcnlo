"""Render the event-sample histograms of a run, bands included.

``EventHistograms::to_json()`` (info.json, "event_histograms") already holds
the nominal distribution, the scale envelope and the PDF band, all computed
from the final unweighted sample. Drawing them here rather than going through
an HwU file and madgraph/various/histograms.py buys two things: a run produces
its plots without any extra step, and the PDF band is the one madspace
computed from the member weights -- histograms.py has to rebuild it and needs
the LHAPDF python module to do so, which is not always installed.

The y axis is the differential cross section dsigma/dx, the same convention as
MadBoard's browser plots: the cross section in a bin divided by the bin width,
so the area under the curve is the cross section. (The HwU file of a run,
hwu_output.py, keeps the cross section per bin instead, as HwU files do.) The
under/overflow bins that the madspace arrays carry are dropped, as both of
those do.

matplotlib is imported lazily and only here: a run without it still writes
every number to info.json, it just draws nothing.
"""

import logging
import os

# matplotlib narrates every font it embeds in a PDF at DEBUG level
# ("Embedding font FontPath(...)", "Writing TrueType font"), which is a dozen
# lines per plot on the run's stdout as soon as anything -- MG5's debug mode,
# a user's logging.basicConfig -- lets DEBUG records through the root logger.
# None of it is ever what an MG7 run wants to read, so those loggers are
# pinned once, when matplotlib is first imported.
QUIET_LOGGERS = ('matplotlib', 'PIL')

# how much of the y range has to be covered before the axis goes logarithmic:
# a pt spectrum falls over orders of magnitude and is unreadable linear, while
# an eta or a weight distribution is flat-ish and unreadable logarithmic
LOG_SCALE_RATIO = 50.0


class BackendMissing(Exception):
    """matplotlib could not be imported, so nothing can be drawn."""


def _pyplot():
    # before the import: matplotlib narrates its own start-up (data path,
    # CONFIGDIR, the backend it picked) while it is being imported, so a
    # logger silenced afterwards is silenced four lines too late. A Logger
    # exists independently of the module it belongs to.
    for name in QUIET_LOGGERS:
        quiet = logging.getLogger(name)
        if quiet.level < logging.WARNING:
            quiet.setLevel(logging.WARNING)
    try:
        import matplotlib
    except ImportError as error:
        raise BackendMissing(str(error))
    # no display is attached to a batch run, and the import of pyplot picks
    # its backend once and for all
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def available():
    """True when the histograms of a run can be drawn."""
    try:
        _pyplot()
    except BackendMissing:
        return False
    return True


def _inner(values, count):
    """Drop the under/overflow entries madspace puts around the bins."""
    values = list(values or [])
    if len(values) == count + 2:
        return values[1:-1]
    return (values + [0.] * count)[:count]


def _scaled(values, count, width):
    """The bins of a column as a differential cross section."""
    if not width:
        return [0.] * count
    return [value / width for value in _inner(values, count)]


def _stepped(values):
    """Repeat the last bin so that a 'post' step covers the last edge."""
    return list(values) + [values[-1] if values else 0.]


def _bands(histogram, count, width):
    """[(label, low, high, colour), ...] for the uncertainty bands."""
    bands = []
    envelope = histogram.get('scale_envelope')
    if envelope:
        bands.append((
            'scale', _scaled(envelope.get('low'), count, width),
            _scaled(envelope.get('high'), count, width), 'tab:orange'))
    for group in histogram.get('pdf_uncertainty') or []:
        central = _scaled(group.get('central'), count, width)
        down = _scaled(group.get('uncertainty_down'), count, width)
        up = _scaled(group.get('uncertainty_up'), count, width)
        if not central:
            continue
        label = 'PDF (%s)' % group.get('pdf_set', 'pdf')
        bands.append((label, [c - d for c, d in zip(central, down)],
                      [c + u for c, u in zip(central, up)], 'tab:blue'))
    return bands


def _y_scale(values):
    """'log' when the spectrum falls far enough for a linear axis to hide it.

    Empty bins are ignored rather than forcing a linear axis: the tail of a pt
    spectrum runs out of events long before it stops being interesting, and a
    logarithmic axis simply leaves those bins out. A negative bin (an
    interference) does force it: a logarithmic axis would hide it.
    """
    if any(v < 0 for v in values):
        return 'linear'
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return 'linear'
    return 'log' if max(positive) / min(positive) > LOG_SCALE_RATIO else 'linear'


def _file_name(name):
    """A file name for a histogram: the observable names are made of the
    [multiparticles] groups and '-', so only a path separator can appear."""
    safe = name.replace(os.sep, '_').replace('/', '_').strip()
    return '%s.pdf' % (safe or 'histogram')


def render_one(plt, histogram, path, title_suffix=''):
    """Draw one histogram to `path`: the distribution with its statistical
    errors on top, the bands relative to it underneath."""

    count = int(histogram.get('bin_count') or 0)
    low = float(histogram.get('min', 0.))
    high = float(histogram.get('max', 0.))
    width = (high - low) / count if count else 0.
    if not count or not width:
        return False

    edges = [low + index * width for index in range(count + 1)]
    values = _scaled(histogram.get('bin_values'), count, width)
    errors = _scaled(histogram.get('bin_errors'), count, width)
    bands = _bands(histogram, count, width)
    name = histogram.get('name', '')

    figure, (main, ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(6.4, 5.6),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.07})

    for label, band_low, band_high, colour in bands:
        main.fill_between(edges, _stepped(band_low), _stepped(band_high),
                          step='post', alpha=0.3, color=colour,
                          linewidth=0, label=label)
    main.step(edges, _stepped(values), where='post', color='black',
              linewidth=1.2, label='central')
    centres = [low + (index + 0.5) * width for index in range(count)]
    main.errorbar(centres, values, yerr=errors, fmt='none', ecolor='black',
                  elinewidth=0.8, capsize=0)

    main.set_ylabel(r'$d\sigma/dx$  [pb/unit]')
    main.set_title('%s%s' % (name, title_suffix))
    main.set_yscale(_y_scale(values))
    main.legend(fontsize='small', frameon=False)
    main.grid(alpha=0.2)

    def relative(column):
        # an empty bin has no ratio to show: NaN leaves a gap rather than
        # drawing a band sitting on 1 that nothing measured
        return [c / v if v else float('nan') for c, v in zip(column, values)]

    for label, band_low, band_high, colour in bands:
        ratio.fill_between(edges, _stepped(relative(band_low)),
                           _stepped(relative(band_high)), step='post',
                           alpha=0.3, color=colour, linewidth=0)
    ratio.axhline(1., color='black', linewidth=1.)
    ratio.set_ylabel('ratio')
    ratio.set_xlabel(name)
    ratio.set_xlim(low, high)
    ratio.grid(alpha=0.2)

    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)
    return True


def render(event_histograms, out_dir, title_suffix=''):
    """Draw every histogram of a run into `out_dir`, one file each. Returns
    the file names written, in the order of the histograms. Raises
    BackendMissing when matplotlib is not installed."""

    histograms = [h for h in event_histograms or [] if h.get('bin_count')]
    if not histograms:
        return []
    plt = _pyplot()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for histogram in histograms:
        name = _file_name(histogram.get('name', ''))
        if render_one(plt, histogram, os.path.join(out_dir, name), title_suffix):
            written.append(name)
    return written
