"""Parsing/preprocessing for MadtRex PDF reweight cards.

This is a sibling of reweight_card.py: same "launch"/"set NAME VALUE" lexical
shape, but a completely different, PDF-reweighting-specific vocabulary of
recognized names (see PDF_OPTION_NAMES) -- kept as its own module rather than
folded into reweight_card.py's grammar since the two modes are otherwise
unrelated (see madtrex.py's run_reweighting()/run_pdf_reweighting(), and the
module docstring there for why PDF and parameter reweighting are two
independent passes to begin with).

- "set PDF_SET <name>" (or "set BEAM1_PDF_SET <name>" + "set BEAM2_PDF_SET
  <name>" together, for asymmetric beams -- eg ep colliders, or proton-ion
  running where one beam uses a nuclear PDF set) chooses which
  Driver.load_pdf_set()'d set a launch reweights to. Before the card's first
  "launch", this is the *baseline*: which PDF set (and, by extension, scale
  choice -- see below) the input sample's own weight is taken to already
  correspond to. Unlike reweight_card.py's FPTYPE baseline, there is no
  hard-set fallback default here: MadtRex does not currently record which
  PDF set an existing sample was generated with anywhere on the sample
  itself (a future hook might sniff the LHE file or other metadata for
  this). This module's own resolve() raises if the baseline PDF set is
  still unset by the time it's called -- but madtrex.py's
  run_pdf_reweighting() fills it in first from the originating run's own
  run_card.toml ([beam] pdf) when one is found next to the events being
  reweighted, so an explicit PDF_SET is only actually required when no such
  run_card is available (a bare card handed to a standalone/relocated
  sample, or one deliberately reweighting to a *different* baseline than
  the run actually used). A launch that doesn't set its own PDF_SET/
  BEAM1_PDF_SET/BEAM2_PDF_SET inherits the baseline's (see resolve());
- "set MUF <value>"/"set MUR <value>" (an absolute scale, in GeV, the same
  for every event) and "set MUF_FACTOR <value>"/"set MUR_FACTOR <value>" (a
  multiplier of each event's own recorded scale -- the default, factor 1.0,
  ie "the event's own nominal scale") -- see madtrex.ScaleChoice,
  which these map onto directly. Settable per launch (defaulting to factor
  1.0, *not* inherited from the baseline's own scale choice -- a launch with
  no MUF/MUR line reweights at the nominal scale, the common case);
- "change alpha_s_power <int>", "change verbose <bool>", "change
  scale_variation_output <none|envelope|all>": card-wide PdfDriver options,
  exactly analogous to reweight_card.py's DRIVER_OPTION_NAMES -- preamble-
  only (before the first "launch"), since they configure the whole
  PdfDriver, not any one hypothesis. Their former spellings "set
  ALPHA_S_POWER ..." etc. are still accepted, with a deprecation warning;
- "change helicity ..." (or its former spellings "set HELICITY_REWEIGHTING"/
  "set USE_HELICITY") is an error here: PDF reweighting rescales by PDF
  ratios and never evaluates a matrix element, so it belongs to a parameter
  reweight card (or a generic card's "change mode LO" section), as do the
  other parameter-reweighting directives (change model/process/fptype,
  "set BLOCK ID VALUE", ...);
- "change mode pdf" and "change backend madtrex|auto" are accepted; the
  directives only the classic reweighting module understands (see
  generic_reweight_card.py) are errors;
- scan:[...] (or scanN:[...] to vary several together) values on a launch's
  own MUF/MUR/MUF_FACTOR/MUR_FACTOR lines, expanded into explicit per-point
  launches via the same eval-based mechanism and group-index semantics as
  reweight_card.py's own scan:[...] (see expand_scans()/_expand_launch()).
  Scoped to the four scale directives only -- PDF_SET/BEAM1_PDF_SET/
  BEAM2_PDF_SET are not scannable this way (list a separate "launch" per PDF
  set instead), matching how reweight_card.py's own "change model" is a
  distinct, non-scannable mechanism from its scan:[...] physics values.
"""

import itertools
import re
from dataclasses import dataclass, field

try:
    from . import generic_reweight_card as _vocab
except ImportError:  # loaded as a plain module (gridpack bin/, tests)
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    import generic_reweight_card as _vocab

# Recognized "set <NAME> <value>" names, lowercase, matched case-
# insensitively against the card. PDF_SET_NAMES are valid both before the
# first "launch" (the baseline) and within one (a per-launch override);
# SCALE_NAMES are valid within a launch (or before the first one, to set the
# baseline's own scale choice -- rarely needed, defaults to nominal);
# CARD_OPTION_NAMES ("change <NAME> <value>", formerly "set") are
# preamble-only.
PDF_SET_NAMES = frozenset({"pdf_set", "beam1_pdf_set", "beam2_pdf_set"})
SCALE_NAMES = frozenset({"muf", "mur", "muf_factor", "mur_factor"})
CARD_OPTION_NAMES = frozenset({"alpha_s_power", "verbose", "scale_variation_output"})
PDF_OPTION_NAMES = PDF_SET_NAMES | SCALE_NAMES | CARD_OPTION_NAMES
# Parameter-reweighting "change" options and former "set" names: errors here.
_PARAMETER_CHANGE_OPTIONS = frozenset({
    "helicity", "model", "process", "fptype", "max_batch", "max_deferred_iterations",
    "streaming", "stream_batch_size", "nb_core", "min_batch",
})
_PARAMETER_SET_NAMES = frozenset({"helicity_reweighting", "use_helicity"}) | (
    _PARAMETER_CHANGE_OPTIONS - {"nb_core", "min_batch"})

_LAUNCH_RE = re.compile(r"^\s*launch\b(?P<rest>.*)$", re.I)
_NAME_RE = re.compile(r"rwgt_name\s*=\s*(?P<name>\S+)", re.I)
_OPTION_RE = re.compile(r"^\s*(?:set|change)\s+(?P<name>\S+)\s+(?P<value>.+?)\s*$", re.I)
_SCAN_RE = re.compile(r"scan\s*(?P<group>\d*)\s*:\s*(?P<values>.*)$", re.I)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "t", ".true."})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "f", ".false."})
_SCALE_VARIATION_OUTPUTS = frozenset({"none", "envelope", "all"})


def _parse_bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    raise ValueError(f"cannot parse '{name} {value}': expected a boolean (eg true/false)")


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"cannot parse '{name} {value}': expected an integer") from error


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"cannot parse '{name} {value}': expected a number") from error


def _resolve_scale(name: str, value: str) -> tuple[str, "PdfScale"]:
    """Parses one already-lowercased MUF/MUR/MUF_FACTOR/MUR_FACTOR (name,
    value) pair into which field it targets ("muF" or "muR") and the
    PdfScale it resolves to. Shared by the baseline's eager resolution
    (parse_pdf_reweight_card()) and a launch's post-scan-expansion
    resolution (_finalize_scales()), so both go through identical parsing."""
    if name == "muf":
        return "muF", PdfScale(_parse_float(name, value), absolute=True)
    if name == "mur":
        return "muR", PdfScale(_parse_float(name, value), absolute=True)
    if name == "muf_factor":
        return "muF", PdfScale(_parse_float(name, value), absolute=False)
    return "muR", PdfScale(_parse_float(name, value), absolute=False)  # mur_factor


@dataclass
class PdfScale:
    """Mirrors madtrex.ScaleChoice (kept as plain data here so this
    module stays usable without a madspace install -- same rationale as
    reweight_card.DriverOptions): value, and whether it's absolute (GeV,
    MUF/MUR) or a multiplicative factor of the event's own recorded scale
    (MUF_FACTOR/MUR_FACTOR, the default -- factor 1.0 is "the event's own
    nominal scale")."""
    value: float = 1.0
    absolute: bool = False


@dataclass
class PdfLaunch:
    name: str | None
    pdf_set_name: str | None = None
    beam1_pdf_set_name: str | None = None
    beam2_pdf_set_name: str | None = None
    # Raw (name, value) pairs from this launch's own "set MUF/MUR/
    # MUF_FACTOR/MUR_FACTOR <value>" lines, in card order -- kept unparsed
    # (rather than resolved into PdfScale directly, unlike pdf_set_name
    # above) so expand_scans() can substitute concrete scan:[...] values
    # before _finalize_scales() actually parses them into muF_scale/
    # muR_scale (see resolve()). The last entry for a given name wins,
    # matching every other "set" directive in this card.
    scale_sets: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PdfBaseline:
    pdf_set_name: str | None = None
    beam1_pdf_set_name: str | None = None
    beam2_pdf_set_name: str | None = None
    muF_scale: PdfScale = field(default_factory=PdfScale)
    muR_scale: PdfScale = field(default_factory=PdfScale)


@dataclass
class ParsedPdfCard:
    baseline: PdfBaseline = field(default_factory=PdfBaseline)
    alpha_s_power: int | None = None
    verbose: bool | None = None
    # "none"/"envelope"/"all" (lowercase) if set; see
    # madtrex.ScaleVariationOutput, which this maps onto directly.
    scale_variation_output: str | None = None
    launches: list[PdfLaunch] = field(default_factory=list)


@dataclass
class ResolvedPdfLaunch:
    name: str | None
    # One name (same PDF set both beams) or two (beam1, beam2) -- pass as
    # *pdf_set_names to PdfDriver.add_hypothesis()/set_baseline(), whose
    # matching overload is selected by the resulting argument count.
    pdf_set_names: tuple[str] | tuple[str, str]
    muF_scale: PdfScale
    muR_scale: PdfScale


@dataclass
class ResolvedPdfCard:
    baseline_pdf_set_names: tuple[str] | tuple[str, str]
    baseline_muF_scale: PdfScale
    baseline_muR_scale: PdfScale
    alpha_s_power: int | None
    verbose: bool | None
    scale_variation_output: str | None
    launches: list[ResolvedPdfLaunch]


def parse_pdf_reweight_card(text: str) -> ParsedPdfCard:
    """Splits text into a ParsedPdfCard: a baseline (PDF_SET/BEAM1_PDF_SET+
    BEAM2_PDF_SET/MUF/MUR/MUF_FACTOR/MUR_FACTOR lines found before the first
    "launch") plus a flat list of launches (unlike reweight_card.py's
    ModelGroup/FptypeGroup, there is no further grouping here: a single
    PdfDriver handles every launch in one pass, since -- unlike a different
    UFO model or a different FPTYPE -- a different PDF set or scale choice
    needs no separate library at all, just a different PdfSet/ScaleChoice
    passed to the same already-loaded PdfDriver).

    Lines are matched loosely (case-insensitive, "#" strips trailing
    comments), same conventions as reweight_card.py.

    Raises if a recognized card-wide option (ALPHA_S_POWER/VERBOSE/
    SCALE_VARIATION_OUTPUT) appears anywhere other than before the first
    "launch", or if a recognized line's value can't be parsed as whatever
    type it expects. Does *not* raise here if no baseline PDF_SET was ever
    given -- see resolve(), which is where that's actually needed and thus
    where it's checked."""
    baseline = PdfBaseline()
    alpha_s_power: int | None = None
    verbose: bool | None = None
    scale_variation_output: str | None = None
    launches: list[PdfLaunch] = []
    current_launch: PdfLaunch | None = None

    for raw_line in text.splitlines():
        line = _vocab.strip_comment(raw_line)
        if not line.strip():
            continue

        launch_match = _LAUNCH_RE.match(line)
        if launch_match:
            name_match = _NAME_RE.search(launch_match.group("rest"))
            current_launch = PdfLaunch(name=name_match.group("name") if name_match else None)
            launches.append(current_launch)
            continue

        _vocab.check_not_classic_only(line, raw_line)
        words = line.split()
        keyword = words[0].lower()
        name = words[1].lower() if len(words) > 1 else ""
        if keyword == "change":
            if name == "backend":
                _vocab.check_backend_line(words, raw_line)
                continue
            if name == "mode":
                if len(words) > 2 and words[2].lower() == "pdf":
                    continue
                raise ValueError(
                    f"{raw_line.strip()!r} in a PDF reweight card: parameter reweighting "
                    f"goes in a 'change mode LO' section of the card"
                )
        if (keyword == "change" and name in _PARAMETER_CHANGE_OPTIONS) or (
            keyword == "set" and name in _PARAMETER_SET_NAMES
        ):
            raise ValueError(
                f"PDF reweight card sets {words[1]!r}: {raw_line!r}; this only applies "
                f"to parameter (matrix-element) reweighting -- set it in a parameter "
                f"reweight card, or in the 'change mode LO' section of a generic card"
            )
        if keyword == "set" and len(words) >= 4:
            raise ValueError(
                f"PDF reweight card sets a model parameter: {raw_line!r}; parameter "
                f"(matrix-element) reweighting goes in a parameter reweight card, or in "
                f"the 'change mode LO' section of a generic card"
            )
        option_match = _OPTION_RE.match(line) if keyword in ("set", "change") else None
        if keyword == "change" and name not in CARD_OPTION_NAMES:
            _vocab.logger.warning("PDF reweight card: ignoring unknown option %r", raw_line.strip())
            continue
        if not option_match or option_match.group("name").strip().lower() not in PDF_OPTION_NAMES:
            continue  # not a recognized line: silently ignored, same tolerant philosophy as reweight_card.py
        name = option_match.group("name").strip().lower()
        value = option_match.group("value").strip()
        if keyword == "set" and name in CARD_OPTION_NAMES:
            _vocab.deprecated(f"set {option_match.group('name')}", f"change {name}")

        if name in CARD_OPTION_NAMES:
            if current_launch is not None:
                raise ValueError(
                    f"PDF reweight card sets card-wide option '{name}' after the first "
                    f"'launch': {raw_line!r}; ALPHA_S_POWER/VERBOSE/SCALE_VARIATION_OUTPUT "
                    f"configure the whole PdfDriver and can only be set in the card's "
                    f"preamble, before any 'launch'"
                )
            if name == "alpha_s_power":
                alpha_s_power = _parse_int(name, value)
            elif name == "verbose":
                verbose = _parse_bool(name, value)
            else:  # scale_variation_output
                v = value.strip().lower()
                if v not in _SCALE_VARIATION_OUTPUTS:
                    raise ValueError(
                        f"cannot parse '{name} {value}': expected one of "
                        f"{sorted(_SCALE_VARIATION_OUTPUTS)}"
                    )
                scale_variation_output = v
            continue

        target = current_launch if current_launch is not None else baseline
        if name == "pdf_set":
            target.pdf_set_name = value
        elif name == "beam1_pdf_set":
            target.beam1_pdf_set_name = value
        elif name == "beam2_pdf_set":
            target.beam2_pdf_set_name = value
        elif current_launch is not None:
            # A launch's own MUF/MUR/MUF_FACTOR/MUR_FACTOR: kept raw so
            # expand_scans() gets a chance at any scan:[...] value first --
            # see PdfLaunch.scale_sets and _finalize_scales() (called from
            # resolve(), after expansion).
            current_launch.scale_sets.append((name, value))
        else:
            # The baseline's own scale choice: resolved eagerly, same as
            # pdf_set_name -- not scannable (see the module docstring).
            field_name, scale = _resolve_scale(name, value)
            setattr(baseline, f"{field_name}_scale", scale)

    return ParsedPdfCard(
        baseline=baseline,
        alpha_s_power=alpha_s_power,
        verbose=verbose,
        scale_variation_output=scale_variation_output,
        launches=launches,
    )


def _resolve_pdf_set_names(
    pdf_set_name: str | None,
    beam1_pdf_set_name: str | None,
    beam2_pdf_set_name: str | None,
    context: str,
) -> tuple[str] | tuple[str, str]:
    """Resolves PDF_SET/BEAM1_PDF_SET/BEAM2_PDF_SET into either a single-name
    tuple (same set both beams) or a two-name tuple (beam1, beam2), matching
    PdfDriver.set_baseline()/add_hypothesis()'s two overloads. context names
    where this came from (eg "baseline", "launch 'foo'"), used only in error
    messages."""
    if pdf_set_name is not None and (beam1_pdf_set_name is not None or beam2_pdf_set_name is not None):
        raise ValueError(
            f"{context}: both PDF_SET and BEAM1_PDF_SET/BEAM2_PDF_SET are set; use one or "
            f"the other"
        )
    if beam1_pdf_set_name is not None or beam2_pdf_set_name is not None:
        if beam1_pdf_set_name is None or beam2_pdf_set_name is None:
            raise ValueError(f"{context}: BEAM1_PDF_SET and BEAM2_PDF_SET must both be set together")
        return (beam1_pdf_set_name, beam2_pdf_set_name)
    if pdf_set_name is None:
        raise ValueError(f"{context}: no PDF_SET (or BEAM1_PDF_SET/BEAM2_PDF_SET) is set")
    return (pdf_set_name,)


def _finalize_scales(scale_sets: list[tuple[str, str]]) -> tuple[PdfScale, PdfScale]:
    """Resolves a launch's raw scale_sets (post scan-expansion) into its
    final (muF_scale, muR_scale) -- both default to factor(1.0) (nominal)
    if scale_sets sets neither. The last MUF/MUF_FACTOR entry wins for
    muF_scale, independently for MUR/MUR_FACTOR and muR_scale."""
    muF_scale, muR_scale = PdfScale(), PdfScale()
    for name, value in scale_sets:
        field_name, scale = _resolve_scale(name, value)
        if field_name == "muF":
            muF_scale = scale
        else:
            muR_scale = scale
    return muF_scale, muR_scale


def _expand_launch(launch: PdfLaunch, base_index: int) -> list[PdfLaunch]:
    """Expands one PdfLaunch's scan:[...] MUF/MUR/MUF_FACTOR/MUR_FACTOR
    values (if any) into the Cartesian product of concrete-valued
    PdfLaunches -- same scanN:/scan: group-index semantics as
    reweight_card._expand_launch() (scanN: entries sharing a group index
    vary together; ungrouped scan: entries each vary independently).
    PDF_SET/BEAM1_PDF_SET/BEAM2_PDF_SET are copied through unchanged onto
    every expanded point -- they're resolved eagerly at parse time (see the
    module docstring) and never scannable here."""
    groups: dict[object, list[int]] = {}
    values_by_index: dict[int, list] = {}
    for i, (_, raw_value) in enumerate(launch.scale_sets):
        value = raw_value.strip()
        if not value.lower().startswith("scan"):
            continue
        scan_match = _SCAN_RE.match(value)
        if not scan_match:
            continue
        try:
            values_by_index[i] = eval(scan_match.group("values"))
        except SyntaxError as error:
            raise ValueError(f"invalid scan definition {raw_value!r}: {error}") from error
        group = scan_match.group("group")
        key = group if group else object()  # ungrouped: unique key, varies independently
        groups.setdefault(key, []).append(i)

    if not groups:
        return [launch]

    keys = list(groups.keys())
    lengths = [range(len(values_by_index[groups[key][0]])) for key in keys]
    name = launch.name or f"pdf_rwgt_{base_index}"
    expanded = []
    for point, positions in enumerate(itertools.product(*lengths)):
        scale_sets = list(launch.scale_sets)
        for key, pos in zip(keys, positions):
            for i in groups[key]:
                set_name, _ = launch.scale_sets[i]
                scale_sets[i] = (set_name, repr(values_by_index[i][pos]))
        expanded.append(
            PdfLaunch(
                name=f"{name}_{point + 1}",
                pdf_set_name=launch.pdf_set_name,
                beam1_pdf_set_name=launch.beam1_pdf_set_name,
                beam2_pdf_set_name=launch.beam2_pdf_set_name,
                scale_sets=scale_sets,
            )
        )
    return expanded


def expand_scans(card: ParsedPdfCard) -> ParsedPdfCard:
    """Returns a copy of card with every launch's scan:[...] MUF/MUR/
    MUF_FACTOR/MUR_FACTOR values resolved into explicit per-point launch
    blocks (see _expand_launch()); a no-op for cards with no scans. Mirrors
    reweight_card.expand_scans() -- call this before resolve()."""
    launches = []
    for i, launch in enumerate(card.launches):
        launches.extend(_expand_launch(launch, i + 1))
    return ParsedPdfCard(
        baseline=card.baseline,
        alpha_s_power=card.alpha_s_power,
        verbose=card.verbose,
        scale_variation_output=card.scale_variation_output,
        launches=launches,
    )


def resolve(card: ParsedPdfCard) -> ResolvedPdfCard:
    """Resolves card into a ResolvedPdfCard ready to drive PdfDriver calls:
    the baseline's PDF set name(s) (raising if none was ever given -- see
    the module docstring), and each launch's own, with any launch that set
    none of PDF_SET/BEAM1_PDF_SET/BEAM2_PDF_SET inheriting the baseline's.
    Call after expand_scans() if the card might contain scan:[...] values --
    resolve() itself does not expand them, it just finalizes whatever
    scale_sets each launch already has (see _finalize_scales())."""
    baseline_names = _resolve_pdf_set_names(
        card.baseline.pdf_set_name,
        card.baseline.beam1_pdf_set_name,
        card.baseline.beam2_pdf_set_name,
        "PDF reweight card baseline",
    )
    resolved_launches = []
    for i, launch in enumerate(card.launches):
        context = f"PDF reweight card launch {launch.name!r}" if launch.name else f"PDF reweight card launch #{i + 1}"
        if launch.pdf_set_name is None and launch.beam1_pdf_set_name is None and launch.beam2_pdf_set_name is None:
            names = baseline_names
        else:
            names = _resolve_pdf_set_names(
                launch.pdf_set_name, launch.beam1_pdf_set_name, launch.beam2_pdf_set_name, context
            )
        muF_scale, muR_scale = _finalize_scales(launch.scale_sets)
        resolved_launches.append(ResolvedPdfLaunch(launch.name, names, muF_scale, muR_scale))
    return ResolvedPdfCard(
        baseline_pdf_set_names=baseline_names,
        baseline_muF_scale=card.baseline.muF_scale,
        baseline_muR_scale=card.baseline.muR_scale,
        alpha_s_power=card.alpha_s_power,
        verbose=card.verbose,
        scale_variation_output=card.scale_variation_output,
        launches=resolved_launches,
    )
