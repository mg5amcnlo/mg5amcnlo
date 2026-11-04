"""The event-sample histograms in the HwU format aMC@NLO writes.

``EventHistograms::to_json()`` (stored in ``Events/<run>/info.json`` under
"event_histograms") already holds everything a plot needs: the nominal
distribution, one distribution per systematics variation, and the scale/PDF
bands built from them. HwU is the format the rest of MadGraph reads -- an
aMC@NLO run writes ``MADatNLO.HwU`` and ``madgraph/various/histograms.py``
plots and compares those -- so writing the same thing here is what lets an
mg7 run be overlaid on an NLO one without a conversion step in between.

Three conventions of the format are worth stating, because they are where a
converter goes wrong:

* HwU bins hold the **cross section in the bin** (sigma per bin), not
  dsigma/dx: ``sum(central)`` is the cross section. That is what aMC@NLO's
  ``HwU_output`` writes, what histograms.py labels its y axis with ("sigma per
  bin") and what its ``--rebin`` assumes when it adds bins up. The madspace
  histograms already hold the cross section in the bin, so the values are
  written unchanged. (The plots of plots.py and of MadBoard divide by the bin
  width and show dsigma/dx; only this file keeps the HwU convention.)
* HwU has **no under/overflow bins**. The madspace arrays carry them (first and
  last entry), and they are dropped, exactly as an aMC@NLO analysis does. The
  sum of an HwU histogram is therefore the cross section inside the range, not
  the total one.
* The scale columns are labelled exactly as in an aMC@NLO ``MADatNLO.HwU``
  ("dyn=-1 muR= 0.500 muF= 2.000"), and the central scale gets its own column
  as there, so the two files carry the same set of scale columns:
  histograms.py stops when the files it reads together have different weight
  columns. That column also puts the central value into the envelope
  histograms.py rebuilds, as madspace's own envelope does. The PDF columns are
  labelled "PDF= 331901", the form ``HwU.weight_label_PDF`` reads as one PDF
  band; an LO and an NLO run use different PDF sets, so an overlay of the two
  is drawn with ``--no_pdf`` when either of them carries PDF columns.

histograms.py draws two histograms in one plot when their titles and bin edges
agree exactly: the histogram name of the run card is the title here, so an
overlay on an NLO run needs an aMC@NLO analysis that books that name with the
same bins, and the edges are written with the 7 significant digits aMC@NLO
prints them with.

This module imports nothing: it is a pure transformation of the two dicts
madspace writes, so anything holding them can produce an HwU file.
"""


def _float(value):
    """One HwU number: signed scientific notation, as aMC@NLO writes it."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.
    if value != value:  # NaN, which no HwU reader expects in a bin
        value = 0.
    return '%+.7e' % value


def _edge(value):
    """One bin edge, rounded to the 7 significant digits of aMC@NLO's
    HwU_output (Fortran e14.7): histograms.py compares edges exactly, and
    300 + 700/60 has to read 311.6667 in both files."""
    return _float(float('%.6e' % value))


def scale_label(mur, muf, dyn=-1):
    """The label of a scale column, as in aMC@NLO's MADatNLO.HwU.

    That file is rewritten by histograms.py, whose HwU.get_formatted_header
    prints a dynamical-scale variation as 'dyn=%i muR=%6.3f muF=%6.3f'; dyn=-1
    is the run's own scale choice.
    """
    return 'dyn=%d muR=%6.3f muF=%6.3f' % (-1 if dyn is None else dyn, mur, muf)


def weight_labels(summary):
    """{weight id: HwU column label} from madspace's systematics summary.

    The group an id belongs to has the last word, not the variation's own
    fields: the central member of a PDF set carries mur = muf = 1 and no
    pdf_index, and labelling it as a scale variation would both hide it from
    the PDF band and duplicate the nominal column.
    """

    labels = {}
    if not summary:
        return labels
    variations = {}
    for entry in summary.get('variations', []):
        if 'id' in entry:
            variations[entry['id']] = entry

    for wgt_id in (summary.get('scale') or {}).get('weight_ids', []):
        entry = variations.get(wgt_id, {})
        labels[wgt_id] = scale_label(entry.get('mur', 1.), entry.get('muf', 1.),
                                     entry.get('dyn', -1))

    for group in summary.get('pdf') or []:
        for wgt_id in group.get('weight_ids', []):
            entry = variations.get(wgt_id, {})
            lhaid = entry.get('pdf_lhaid', group.get('pdf_lhaid'))
            member = entry.get('pdf_member', 0)
            if lhaid is None:
                continue
            # the LHAPDF id of the member itself, which is what an HwU
            # "PDF= <id>" column names
            labels[wgt_id] = 'PDF= %d' % (int(lhaid) + int(member))

    return labels


def central_scale_label(summary, labels):
    """The label of the central-scale column, None when there is none to add.

    madspace's scale weights are the variations only, while aMC@NLO lists
    muR = muF = 1 among its scale columns; the central value is added under
    that label whenever the run has scale variations and none of them is it.
    """
    scale_ids = ((summary or {}).get('scale') or {}).get('weight_ids', [])
    if not scale_ids:
        return None
    label = scale_label(1., 1.)
    if label in labels.values():
        return None
    return label


def _columns(histogram, labels, central_label=None):
    """(column label, values) for one histogram, under/overflow still in."""

    nominal = histogram.get('bin_values') or []
    columns = [('central value', nominal),
               ('dy', histogram.get('bin_errors') or [])]

    envelope = histogram.get('scale_envelope')
    if envelope:
        columns.append(('delta_mu_min @aux', envelope.get('low') or []))
        columns.append(('delta_mu_max @aux', envelope.get('high') or []))

    for group in histogram.get('pdf_uncertainty') or []:
        central = group.get('central') or []
        down = group.get('uncertainty_down') or []
        up = group.get('uncertainty_up') or []
        if not central:
            continue
        columns.append(('delta_pdf_min @aux',
                        [c - d for c, d in zip(central, down)]))
        columns.append(('delta_pdf_max @aux',
                        [c + u for c, u in zip(central, up)]))
        break  # the format has one PDF band; further sets stay as raw columns

    if central_label:
        columns.append((central_label, nominal))

    for index, entry in enumerate(histogram.get('weights') or []):
        label = labels.get(entry.get('id'), 'wgt_%d' % (index + 1))
        columns.append((label, entry.get('bin_values') or []))

    return columns


def to_hwu(event_histograms, summary=None, type_label='LO'):
    """Render madspace's event-sample histograms as HwU text.

    `event_histograms` is the "event_histograms" list of info.json and
    `summary` its "systematics" dict (None when the run had none).
    `type_label` is the HwU TYPE tag, which is how histograms.py tells the
    curves of two runs apart when it draws them together.
    """

    histograms = [h for h in event_histograms or [] if h.get('bin_count')]
    if not histograms:
        return ''
    labels = weight_labels(summary)
    central_label = central_scale_label(summary, labels)

    header = ['xmin', 'xmax']
    header += [label for label, _ in
               _columns(histograms[0], labels, central_label)]
    lines = ['##& ' + ' & '.join(header), '']

    for histogram in histograms:
        bin_count = int(histogram['bin_count'])
        low = float(histogram['min'])
        width = (float(histogram['max']) - low) / bin_count
        columns = _columns(histogram, labels, central_label)
        lines.append('<histogram> %d "%s |X_AXIS@LIN |Y_AXIS@LOG |TYPE@%s"'
                     % (bin_count, histogram.get('name', ''), type_label))
        for index in range(bin_count):
            row = [_edge(low + index * width), _edge(low + (index + 1) * width)]
            for _label, values in columns:
                # index + 1: the madspace arrays start with the underflow bin;
                # the value is the cross section in the bin, as HwU wants it
                value = values[index + 1] if index + 1 < len(values) else 0.
                row.append(_float(value))
            lines.append('  ' + '   '.join(row))
        lines.append('<\\histogram>')
        lines.append('')

    return '\n'.join(lines)
