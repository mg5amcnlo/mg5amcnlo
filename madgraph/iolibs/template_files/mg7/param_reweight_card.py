"""Parsing/preprocessing for MadtRex parameter (matrix-element) reweight
cards -- see pdf_reweight_card.py for the separate, independent PDF
reweighting card, and generic_reweight_card.py for the dispatcher that lets
one card mix both.

madtrex.ParamHandler.read_rwgt_card() (madtrex/include/madtrex/
param_handler.hpp) only understands the literal "# launch / # set BLOCK_NAME
PARAM_ID PARAM_VALUE" grammar parsed by REX::tea::rwgt_slha::parse_rwgt_card
(Rex/src/teaRex.cc) -- unrecognized lines (eg "change model ...") are silently
ignored there, and scan:[...] value expressions fail to parse as a plain
double. This module extracts the extra directives MadtRex needs to act on
before handing a card to that low-level parser:

- any number of "change model <spec>" lines, each paired with a "change
  process <definition>" line, requesting the launches that follow be run
  against a different UFO model, regenerated and compiled on demand (see
  madtrex.py's regenerate_process_in_model()) -- MG7 does not persist the
  original "generate <process>" command anywhere on disk, so it must be given
  again explicitly whenever the model changes. Launches are grouped by their
  (model, process) target regardless of how they're interleaved in the card,
  so "change model" may appear more than once (switching back and forth, or
  requesting several distinct alternate models) without duplicating work.
  Such launches are still normalised against the *original* model's matrix
  element (the one the events were generated with): their weights are the
  new model's |M|^2 over the original |M|^2 (see Driver.
  set_normaliser_process_directory() in madtrex.hpp);
- "change fptype <value>" lines (formerly "set FPTYPE <value>", still
  accepted with a deprecation warning), requesting the launch they belong to be
  evaluated against a differently-precision-compiled matrix element library
  (see MADMATRIX_LIB in madmatrix.mk / Driver.set_fptype_priority() in
  madtrex.hpp) -- settable per launch, exactly like a physics "set BLOCK ID
  VALUE" line (nothing carries over between launches). This is always the
  *target* fptype to reweight TO -- normalisation is a separate, always-on
  concern: every launch must be normalised against the sample's own actual
  generation-time fptype, never against whichever fptype it's being
  reweighted to (this is a fundamental assumption of how reweighting works
  here -- see Driver.set_normaliser_fptype_priority() in madtrex.hpp/.cpp,
  and madtrex.py's run_parameter_reweighting()). A "set FPTYPE ..."
  appearing before the card's very first "launch" is special: it is taken as
  the *default target* fptype, used by every launch that doesn't specify its
  own "set FPTYPE" (see ParsedCard.baseline_fptype/group_by_fptype()). When
  a card sets no FPTYPE at all (neither a baseline nor any per-launch
  override), the default target -- like the always-on normaliser -- resolves
  to the process directory's own run_card.toml [run] fptype (madtrex.py's
  run_parameter_reweighting(), via group_by_fptype()'s default_fptype
  parameter), falling back further to DEFAULT_FPTYPE only if that can't be
  read either; this means a plain, non-FPTYPE-changing reweight (eg a bare
  "set MT 200") correctly stays at whatever precision the sample was already
  generated at, rather than silently drifting to a hardcoded default;
- "change <option> <value>" lines for "helicity" and DRIVER_OPTION_NAMES,
  configuring one of Driver's options (helicity reweighting, warning
  verbosity, event-batch splitting, deferred-iteration flushing, streaming)
  for every Driver madtrex.py builds from this card. "change helicity
  <bool>" -- the classic reweighting module's option, with its default --
  chooses between reweighting each event at its own helicity (True, the
  default: the matrix elements are evaluated for the helicities in the
  event's LHE SPINUP column) and helicity-summed matrix elements (False).
  Helicity reweighting needs every event to carry helicity information
  (SPINUP -1/0/+1 for every particle) and is an error otherwise; set it
  False for a sample without it. The former spellings "set
  helicity_reweighting"/"set use_helicity" and "set <option> <value>" are
  still accepted, with a deprecation warning (see DEPRECATED_SET_NAMES).
  Unlike
  "change fptype", these are *not* per-launch (a Driver's own options are all
  fixed before it ever loads/runs anything, uniformly for whatever launches
  end up sharing that Driver -- see run_reweighting()), so they may only
  appear before the card's very first "launch", exactly like "change
  model"/the FPTYPE baseline; see ParsedCard.driver_options. Sizes/counts
  (MAX_BATCH, MAX_DEFERRED_ITERATIONS, STREAM_BATCH_SIZE) are REX::npos
  ("off"/"no cap") by default on the C++ side; any positive integer here
  both sets the value *and* implies the option is wanted "on" (see
  DriverOptions.streaming's own doc comment for STREAM_BATCH_SIZE's
  particular case: it has no boolean of its own on the C++ side, only
  STREAMING does, so setting just a batch size still needs to flip that on);
- "set <NAME> <value>" lines naming a parameter by its param_card name (eg
  "set MT 175", as the classic reweighting module allows), resolved to
  "set BLOCK ID VALUE" against the param_card the launch runs with (see
  resolve_parameter_names());
- scan:[...] values in "set" lines, expanded into explicit per-value launch
  blocks via the same eval-based mechanism and group-index semantics as
  models.check_param_card.ParamCardIterator.iterate().

The card shares its vocabulary with the classic reweighting module (see
generic_reweight_card.py): "change backend madtrex|auto" and "change mode
LO" are accepted, and the classic-only directives ("change output", "change
mode NLO", density-matrix options, ...) are errors here.
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

# Hard-set for now (see the module docstring); should become a modifiable
# global default once MadtRex/madevent.py grow one of their own.
DEFAULT_FPTYPE = "m"

# Recognized "change <NAME> <value>" preamble Driver options (see
# DriverOptions) besides the shared "change helicity", lowercase, matched
# case-insensitively against the card. MadtRex-only, as is "change fptype".
# The threading ones (THREAD_OPTION_NAMES) are newer than the "set" spellings
# and have none:
# - "change nb_core <N|auto>": MadtRex's worker threads. auto (the default)
#   uses as many as the event generation: the [run] cpu_thread_pool_size of
#   the run card the sample was generated with, all available cores when it
#   is -1 (see madtrex_driver.reweighting_threads());
# - "change min_batch <N>": the fewest events per batch when the sample is
#   split into batches across the threads (DEFAULT_MIN_BATCH by default; see
#   Driver.set_min_batch()). A subprocess with more events than
#   max(min_batch, events / threads) is split into batches of that size, so a
#   sample dominated by one subprocess still keeps every thread busy, while a
#   small one is not spread so thinly that a batch costs more to set up (its
#   own matrix element instance, its own sorting) than it saves. An explicit
#   "change max_batch" takes precedence.
THREAD_OPTION_NAMES = frozenset({"nb_core", "min_batch"})
DRIVER_OPTION_NAMES = frozenset({
    "verbose", "max_batch", "max_deferred_iterations", "streaming", "stream_batch_size",
}) | THREAD_OPTION_NAMES
# "change" options only MadtRex understands.
MADTREX_CHANGE_OPTIONS = DRIVER_OPTION_NAMES | {"fptype"}
# Former "set <NAME> <value>" spellings -> the "change" option they now are.
DEPRECATED_SET_NAMES = {
    "helicity_reweighting": "helicity", "use_helicity": "helicity", "fptype": "fptype",
    **{name: name for name in DRIVER_OPTION_NAMES - THREAD_OPTION_NAMES},
}
# The default of "change min_batch". Splitting a batch off costs a few
# milliseconds (a matrix element instance reloaded from its library and
# param_card, about 2.5 ms for p p > t t~ j j on a laptop, and the batch's
# own setup), which a batch of 1000 events amortises even for a simple
# process (2 -> 2 at its own helicity: about a microsecond per event and
# launch); below that, threads gain little over running the subprocesses as
# they are.
DEFAULT_MIN_BATCH = 1000
# PDF-reweighting names (pdf_reweight_card.py), errors in a parameter card.
_PDF_NAMES = frozenset({
    "pdf_set", "beam1_pdf_set", "beam2_pdf_set", "muf", "mur", "muf_factor",
    "mur_factor", "alpha_s_power", "scale_variation_output",
})

_MODEL_RE = re.compile(r"^\s*change\s+model\s+(?P<spec>.+?)\s*$", re.I)
_PROCESS_RE = re.compile(r"^\s*change\s+process\s+(?P<definition>.+?)\s*$", re.I)
_LAUNCH_RE = re.compile(r"^\s*launch\b(?P<rest>.*)$", re.I)
_NAME_RE = re.compile(r"rwgt_name\s*=\s*(?P<name>\S+)", re.I)
# "set [param_card] <block> <id> <value>", <id> one integer or several
# (multi-index blocks: "set NMIX 1 2 0.5"); and "set [param_card] <name>
# <value>" for a parameter named by its param_card name. <value> may hold
# spaces (scan:[1, 2]).
_SET_RE = re.compile(
    r"^\s*set\s+(?:param_card\s+)?(?P<block>\S+)\s+(?P<id>[+-]?\d+(?:\s+[+-]?\d+)*)"
    r"\s+(?P<value>.+?)\s*$",
    re.I,
)
_NAMED_SET_RE = re.compile(r"^\s*set\s+(?:param_card\s+)?(?P<name>\S+)\s+(?P<value>.+?)\s*$", re.I)
_SCAN_RE = re.compile(r"scan\s*(?P<group>\d*)\s*:\s*(?P<values>.*)$", re.I)
# as the classic module (banner.ConfigFile.format_variable), plus yes/on/no/off
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "t", ".true."})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "f", ".false."})


def _parse_bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    raise ValueError(f"cannot parse '{name} {value}': expected a boolean (eg True/False)")


def _parse_optional_size(name: str, value: str) -> int | None:
    """Parses a MAX_BATCH/MAX_DEFERRED_ITERATIONS/STREAM_BATCH_SIZE value:
    any positive integer sets it (and implies the option is wanted "on");
    "0"/"none"/"off"/"false" (case-insensitively) mean "leave it at its
    REX::npos ("off"/"no cap") default", represented here as None."""
    v = value.strip().lower()
    if v in _FALSE_VALUES or v == "none":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"cannot parse '{name} {value}': expected a non-negative integer, or "
            f"none/off to leave it unset"
        ) from error
    if parsed < 0:
        raise ValueError(f"'{name} {value}': must not be negative")
    return parsed or None


@dataclass
class DriverOptions:
    """Card-wide (not per-launch) Driver options -- see DRIVER_OPTION_NAMES
    and the module docstring. Every field is None ("don't touch, leave
    Driver's own default") unless the card actually set it; apply() is what
    turns a resolved instance into the matching Driver calls."""
    helicity_reweighting: bool | None = None
    verbose: bool | None = None
    max_batch: int | None = None
    max_deferred_iterations: int | None = None
    # streaming=None with stream_batch_size set still means "on": see
    # apply_options(). streaming=False always means "off", regardless of
    # stream_batch_size (matches Driver.set_streaming(False)'s own C++
    # semantics, which ignores its own batch_size argument when disabling).
    streaming: bool | None = None
    stream_batch_size: int | None = None
    # None: the generation's thread count / DEFAULT_MIN_BATCH (set by
    # madtrex_driver before apply())
    nb_core: int | None = None
    min_batch: int | None = None

    def apply(self, driver) -> None:
        """Applies whichever fields aren't None to driver (a
        madtrex.Driver) via its matching set_*() method. Not typed
        as madtrex.Driver directly so this module stays free of a
        madspace/pybind11 import of its own; see madtrex.py's own use of
        this for the concrete type."""
        if self.helicity_reweighting is not None:
            driver.set_helicity_reweighting(self.helicity_reweighting)
        if self.verbose is not None:
            driver.set_verbose(self.verbose)
        if self.max_batch is not None:
            driver.set_max_batch(self.max_batch)
        if self.max_deferred_iterations is not None:
            driver.set_max_deferred_iterations(self.max_deferred_iterations)
        if self.nb_core is not None:
            driver.set_threads(self.nb_core)
        if self.min_batch is not None:
            driver.set_min_batch(self.min_batch)
        streaming = self.streaming
        if streaming is None and self.stream_batch_size is not None:
            streaming = True  # a batch size with no explicit STREAMING line still means "on"
        if streaming is True:
            if self.stream_batch_size is not None:
                driver.set_streaming(True, self.stream_batch_size)
            else:
                driver.set_streaming(True)
        elif streaming is False:
            driver.set_streaming(False)


@dataclass
class LaunchBlock:
    name: str | None
    # (block, id, value); (name, None, value) for a parameter given by its
    # param_card name, until resolve_parameter_names()
    sets: list[tuple[str, str | None, str]] = field(default_factory=list)
    # None: use the card's (or DEFAULT_FPTYPE, if that's also unset) baseline
    # -- see group_by_fptype().
    fptype: str | None = None


@dataclass
class ModelGroup:
    # model_spec is None for launches that never followed a "change model"
    # line, ie ones meant to run against the original process directory as-is.
    model_spec: str | None
    process_definition: str | None
    launches: list[LaunchBlock] = field(default_factory=list)


@dataclass
class ParsedCard:
    # One entry per distinct (model_spec, process_definition) target
    # actually used by a launch, in order of first appearance.
    groups: list[ModelGroup]
    # The "set FPTYPE ..." found before the card's first "launch" line, if
    # any; see group_by_fptype().
    baseline_fptype: str | None = None
    # Resolved from every "set <DRIVER_OPTION_NAME> ..." line found before
    # the card's first "launch"; see DriverOptions.
    driver_options: DriverOptions = field(default_factory=DriverOptions)


@dataclass
class FptypeGroup:
    # Mirrors ModelGroup, further split by resolved (not None) fptype -- see
    # group_by_fptype(). model_spec/process_definition carry the same
    # meaning as on ModelGroup.
    model_spec: str | None
    process_definition: str | None
    fptype: str
    launches: list[LaunchBlock] = field(default_factory=list)


def _set_option(
    name: str,
    value: str,
    raw_line: str,
    after_launch: bool,
    current_group: "ModelGroup | None",
    baseline_fptype: str | None,
    driver_options: DriverOptions,
) -> str | None:
    """Applies "change <name> <value>" (name one of "fptype", "helicity",
    DRIVER_OPTION_NAMES) and returns the possibly updated baseline fptype."""
    if name == "fptype":
        if not after_launch:
            # Nothing has launched anywhere in the card yet: this is the
            # baseline (see the module docstring / group_by_fptype()),
            # not a per-launch override.
            if baseline_fptype is not None and baseline_fptype != value:
                raise ValueError(
                    f"reweight card sets a baseline FPTYPE more than once with "
                    f"different values ('{baseline_fptype}' and '{value}')"
                )
            return value
        current_group.launches[-1].fptype = value
        return baseline_fptype
    if after_launch:
        raise ValueError(
            f"reweight card sets driver option '{name}' after the first "
            f"'launch': {raw_line!r}; driver options apply to a whole "
            f"reweighting pass (see DriverOptions) and can only be set in the "
            f"card's preamble, before any 'launch'"
        )
    if name == "helicity":
        driver_options.helicity_reweighting = _parse_bool(name, value)
    elif name in ("verbose", "streaming"):
        setattr(driver_options, name, _parse_bool(name, value))
    elif name == "nb_core" and value.strip().lower() == "auto":
        driver_options.nb_core = None
    else:
        setattr(driver_options, name, _parse_optional_size(name, value))
    return baseline_fptype


def parse_reweight_card(text: str) -> ParsedCard:
    """Splits text into per-(model, process)-target groups of "launch"/"set"
    blocks. Lines are matched loosely (case-insensitive, "#" strips trailing
    comments) to stay tolerant of the same card conventions the legacy
    reweight_card.dat format allows (eg "set param_card BLOCK ID VALUE
    # orig: ...").

    Raises if a "launch" follows a "change model" with no paired "change
    process" yet (MG7 has no other way to learn the process definition), or if
    the same model is later paired with a different process definition (each
    model may be regenerated for at most one process per card), if a "change
    fptype ..." appears before the first "launch" more than once with
    different values, if "change helicity"/"change <DRIVER_OPTION_NAME> ..."
    appears anywhere other than before the first "launch", if such a line's
    value can't be parsed as whatever type that option expects, or for a
    line only the classic reweighting module understands (see
    generic_reweight_card.classic_only_reason()) or that belongs to PDF
    reweighting."""
    groups: dict[str | None, ModelGroup] = {}
    order: list[str | None] = []
    current_model_spec: str | None = None
    current_process_definition: str | None = None
    current_group: ModelGroup | None = None
    baseline_fptype: str | None = None
    driver_options = DriverOptions()

    for raw_line in text.splitlines():
        line = _vocab.strip_comment(raw_line)
        if not line.strip():
            continue
        _vocab.check_not_classic_only(line, raw_line)

        model_match = _MODEL_RE.match(line)
        if model_match:
            current_model_spec = model_match.group("spec").strip()
            continue

        process_match = _PROCESS_RE.match(line)
        if process_match:
            current_process_definition = process_match.group("definition").strip()
            continue

        launch_match = _LAUNCH_RE.match(line)
        if launch_match:
            if current_model_spec is not None and current_process_definition is None:
                raise ValueError(
                    f"reweight card requests 'change model {current_model_spec}' but "
                    f"has no 'change process <definition>' line before the next "
                    f"'launch': MG7 does not persist the original process definition "
                    f"anywhere on disk, so it must be given again explicitly to "
                    f"regenerate it in the new model"
                )
            group = groups.get(current_model_spec)
            if group is None:
                group = ModelGroup(current_model_spec, current_process_definition)
                groups[current_model_spec] = group
                order.append(current_model_spec)
            elif current_model_spec is not None and group.process_definition != current_process_definition:
                raise ValueError(
                    f"reweight card requests model '{current_model_spec}' with two "
                    f"different process definitions ('{group.process_definition}' and "
                    f"'{current_process_definition}'); MadtRex only supports "
                    f"regenerating a single process per model per card"
                )
            name_match = _NAME_RE.search(launch_match.group("rest"))
            group.launches.append(LaunchBlock(name=name_match.group("name") if name_match else None))
            current_group = group
            continue

        words = line.split()
        name = words[1].lower() if len(words) > 1 else ""
        after_launch = current_group is not None and bool(current_group.launches)

        if words[0].lower() == "change":
            if name == "backend":
                _vocab.check_backend_line(words, raw_line)
                continue
            if name == "mode":
                mode = words[2].lower() if len(words) > 2 else ""
                if mode == "lo":
                    continue
                raise ValueError(
                    f"{raw_line.strip()!r} in a parameter reweight card: only 'change mode "
                    f"LO' belongs here; PDF reweighting goes in a 'change mode pdf' section "
                    f"of the card"
                )
            if name in _PDF_NAMES:
                raise ValueError(
                    f"{raw_line.strip()!r} is a PDF reweighting option: put it in a "
                    f"'change mode pdf' section of the card"
                )
            if name == "helicity" or name in MADTREX_CHANGE_OPTIONS:
                if len(words) != 3:
                    raise ValueError(f"cannot parse {raw_line.strip()!r}: expected 'change {name} <value>'")
                baseline_fptype = _set_option(
                    name, words[2], raw_line, after_launch, current_group, baseline_fptype,
                    driver_options,
                )
                continue
            _vocab.logger.warning("reweight card: ignoring unknown option %r", raw_line.strip())
            continue

        if words[0].lower() == "set":
            if name == "mode":
                raise ValueError(
                    f"{raw_line.strip()!r}: card sections are read by the generic reweight "
                    f"card reader (generic_reweight_card.py), not inside a parameter card"
                )
            if name in _PDF_NAMES and len(words) == 3:
                raise ValueError(
                    f"{raw_line.strip()!r} is a PDF reweighting setting: put it in a "
                    f"'change mode pdf' section of the card"
                )
            if name in DEPRECATED_SET_NAMES and len(words) == 3:
                _vocab.deprecated(f"set {words[1]}", f"change {DEPRECATED_SET_NAMES[name]}")
                baseline_fptype = _set_option(
                    DEPRECATED_SET_NAMES[name], words[2], raw_line, after_launch,
                    current_group, baseline_fptype, driver_options,
                )
                continue
            set_match = _SET_RE.match(line)
            named_match = None if set_match else _NAMED_SET_RE.match(line)
            if set_match or named_match:
                if not after_launch:
                    raise ValueError(f"reweight card 'set' line appears before any 'launch': {raw_line!r}")
                if set_match:
                    entry = (set_match.group("block"), " ".join(set_match.group("id").split()),
                             set_match.group("value"))
                else:
                    entry = (named_match.group("name"), None, named_match.group("value"))
                current_group.launches[-1].sets.append(entry)
            continue

    return ParsedCard(
        groups=[groups[key] for key in order],
        baseline_fptype=baseline_fptype,
        driver_options=driver_options,
    )


def _expand_launch(block: LaunchBlock, base_index: int) -> list[LaunchBlock]:
    """Expands one LaunchBlock's scan:[...] values (if any) into the Cartesian
    product of concrete-valued LaunchBlocks. "set" lines sharing the same
    scanN: group index vary together; ungrouped ("scan:") entries each vary
    independently -- mirrors models.check_param_card.ParamCardIterator.iterate()."""
    groups: dict[object, list[int]] = {}
    values_by_index: dict[int, list] = {}
    for i, (_, _, raw_value) in enumerate(block.sets):
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
        return [block]

    keys = list(groups.keys())
    lengths = [range(len(values_by_index[groups[key][0]])) for key in keys]
    name = block.name or f"rwgt_{base_index}"
    expanded = []
    for point, positions in enumerate(itertools.product(*lengths)):
        sets = list(block.sets)
        for key, pos in zip(keys, positions):
            for i in groups[key]:
                block_name, param_id, _ = block.sets[i]
                sets[i] = (block_name, param_id, repr(values_by_index[i][pos]))
        expanded.append(LaunchBlock(name=f"{name}_{point + 1}", sets=sets, fptype=block.fptype))
    return expanded


def expand_scans(card: ParsedCard) -> ParsedCard:
    """Returns a copy of card with every launch's scan:[...] values resolved
    into explicit per-point launch blocks, within each model group
    independently; a no-op for cards with no scans."""
    groups = []
    for group in card.groups:
        launches = []
        for i, block in enumerate(group.launches):
            launches.extend(_expand_launch(block, i + 1))
        groups.append(ModelGroup(group.model_spec, group.process_definition, launches))
    return ParsedCard(groups=groups, baseline_fptype=card.baseline_fptype, driver_options=card.driver_options)


def group_by_fptype(card: ParsedCard, default_fptype: str = DEFAULT_FPTYPE) -> list[FptypeGroup]:
    """Splits each of card.groups' launches further by their own resolved
    *target* fptype (LaunchBlock.fptype, defaulting to card.baseline_fptype,
    itself defaulting to default_fptype if the card set neither) -- launches
    targeting a different fptype need an independently-compiled matrix
    element library (see Driver.set_fptype_priority() in madtrex.hpp), so
    unlike ordinary physics "set" values (handled within one Driver/
    WareHouse pass via SLHA iterators) they can't share their model group's
    single pass. Groups are kept in order of first appearance of each
    (model_spec, process_definition, fptype) combination, merging
    non-contiguous launches for the same combination together -- the same
    tolerance for interleaving ModelGroup's own (model, process) grouping
    already has.

    default_fptype is the module constant DEFAULT_FPTYPE unless the caller
    overrides it -- madtrex.py's run_parameter_reweighting() always passes
    the process directory's own run_card.toml [run] fptype instead (falling
    back to DEFAULT_FPTYPE only if that can't be read), so a card that sets
    no FPTYPE at all resolves every launch's target to the sample's own
    actual generation-time precision -- a no-op fptype-wise -- rather than a
    hardcoded value that may not match. This is purely about each launch's
    *target*; the separate, always-on normaliser-fptype concern (see the
    module docstring) is resolved independently in madtrex.py, not here."""
    baseline = card.baseline_fptype or default_fptype
    result: dict[tuple[str | None, str | None, str], FptypeGroup] = {}
    order: list[tuple[str | None, str | None, str]] = []
    for group in card.groups:
        for launch in group.launches:
            fptype = launch.fptype or baseline
            key = (group.model_spec, group.process_definition, fptype)
            fgroup = result.get(key)
            if fgroup is None:
                fgroup = FptypeGroup(group.model_spec, group.process_definition, fptype)
                result[key] = fgroup
                order.append(key)
            fgroup.launches.append(launch)
    return [result[key] for key in order]


def read_param_card_names(text: str) -> dict[str, tuple[str, str]]:
    """{name: (block, id)} for the parameters of the param_card text, named
    by the comment of their line as the classic reweighting module does
    (eg "  6 1.730000e+02 # MT" in BLOCK MASS, "DECAY 6 1.49 # WT"), names
    lowercase. An entry with several indices (eg "  1  2 5.0e-01 # NMIX12")
    gets them all, space separated ("1 2")."""
    names: dict[str, tuple[str, str]] = {}
    block = None
    for line in text.splitlines():
        content, _, comment = line.partition("#")
        words = content.split()
        if not words:
            continue
        keyword = words[0].lower()
        if keyword == "block":
            block = words[1] if len(words) > 1 else None
            continue
        if keyword == "decay":
            block = None
            if len(words) >= 3 and comment.split():
                names.setdefault(comment.split()[0].lower(), ("DECAY", words[1]))
            continue
        if block is not None and len(words) >= 2 and comment.split():
            names.setdefault(comment.split()[0].lower(), (block, " ".join(words[:-1])))
    return names


def has_parameter_names(launches: list[LaunchBlock]) -> bool:
    """Some launch sets a parameter by its param_card name."""
    return any(param_id is None for block in launches for _, param_id, _ in block.sets)


def resolve_parameter_names(
    launches: list[LaunchBlock], param_card_text: str
) -> list[LaunchBlock]:
    """launches with every parameter given by its param_card name ("set MT
    175") replaced by its "set BLOCK ID VALUE" form, looked up in the
    param_card text (see read_param_card_names()). Raises for a name the
    param_card does not have."""
    names = read_param_card_names(param_card_text)
    resolved = []
    for block in launches:
        sets = []
        for name, param_id, value in block.sets:
            if param_id is None:
                found = names.get(name.lower())
                if found is None:
                    found = names.get(f"mdl_{name.lower()}")
                if found is None:
                    raise ValueError(
                        f"reweight card: no parameter named {name!r} in the param_card; "
                        f"use 'set BLOCK ID VALUE'"
                    )
                name, param_id = found
            sets.append((name, param_id, value))
        resolved.append(LaunchBlock(name=block.name, sets=sets, fptype=block.fptype))
    return resolved


def render_launches(launches: list[LaunchBlock]) -> str:
    """Renders one group's launches back into the literal "launch"/
    "set BLOCK ID VALUE" text ParamHandler.read_rwgt_card() understands -- no
    "change model"/"change process" lines, no "change fptype ..." lines (that's
    consumed entirely at the Python level -- see group_by_fptype() and its
    use in madtrex.py -- and never has any meaning to REX::tea::rwgt_slha's
    own SLHA-only grammar), and no scan:[...] values, all of which must
    already be resolved (see expand_scans() and the group handling in
    madtrex.py)."""
    lines = []
    for block in launches:
        lines.append(f"launch --rwgt_name={block.name}" if block.name else "launch")
        for block_name, param_id, value in block.sets:
            if param_id is None:
                raise ValueError(
                    f"parameter {block_name!r} is given by name: resolve it first "
                    f"(resolve_parameter_names())"
                )
            lines.append(f"set {block_name} {param_id} {value}")
    return "\n".join(lines) + "\n"
