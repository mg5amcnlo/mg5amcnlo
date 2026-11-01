import argparse
import contextlib
import gc
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
import glob
import shutil
import json
import subprocess
import tempfile
import re
import logging
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
import resource

# Install madspace on first use, then import it. The bootstrap lives in its own
# module (which must not import madspace) so that an in-process caller can run
# it *before* importing this one -- see bootstrap.ensure_madspace.
from madgraph.iolibs.template_files.mg7.bootstrap import (
    ensure_madspace as _ensure_madspace,
    drop_install_path as _drop_install_path,
    MG_ROOT as _MG_ROOT,
    MADSPACE_DIR as _MADSPACE_DIR,
    INSTALL_DIR as _INSTALL_DIR,
    MADTREX_DIR as _MADTREX_DIR,
    MADTREX_INSTALL_DIR as _MADTREX_INSTALL_DIR,
)

_ensure_madspace()

import madspace as ms

# madspace is imported; stop its install directory from shadowing the caller's
# yaml/packaging/... for the rest of what is now MG5's own session.
_drop_install_path()
from madgraph.models.check_param_card import ParamCard, group_auto_widths
from madgraph.models.model_reader import Alphas_Runner
from madgraph.iolibs.template_files.mg7 import hwu_output
from madgraph.iolibs.template_files.mg7 import plots
from madgraph.iolibs.template_files.mg7 import systematics_summary
from madgraph.iolibs.template_files.mg7 import npy_to_lhe
from madgraph.iolibs.template_files.mg7.custom_function_loader import load_custom_function
from madgraph.various.banner import RunCardMG7
from madgraph.various import misc
from madgraph.interface.extended_cmd import Cmd

_source_hash = subprocess.run(
    [sys.executable, str(_MADSPACE_DIR / "source_hash.py")],
    capture_output=True, text=True, check=True,
).stdout.strip()
if _source_hash != ms.SOURCE_HASH:
    print()
    print(
        "\033[1m\033[31mWARNING\033[39m: madspace source and installed binaries "
        "are not compatible (source hash mismatch) — consider recompiling "
        "madspace (e.g. `install madspace -y`)\033[0m"
    )
    print()

logger = logging.getLogger("madgraph7")

# From this many subprocess directories to compile, the standard headers are
# precompiled once ('make PCH=1', see madmatrix.mk): each file of a small process
# spends most of its time parsing them. Below, building the precompiled header
# costs about what it saves.
PCH_MIN_SUBPROCESSES = 4

LOG_LEVEL_MAP = {
    ms.Logger.LogLevel.level_debug: logging.DEBUG,
    ms.Logger.LogLevel.level_info: logging.INFO,
    ms.Logger.LogLevel.level_warning: logging.WARNING,
    ms.Logger.LogLevel.level_error: logging.ERROR,
}
def ms_log_handler(level: ms.Logger.LogLevel, message: str):
    logger.log(LOG_LEVEL_MAP[level], message)
ms.Logger.set_log_handler(ms_log_handler)


def get_start_time():
    return time.time(), time.process_time()


def format_time(t: int, centi: bool = False):
    hours, t = divmod(t, 3600)
    minutes, seconds = divmod(t, 60)
    if centi:
        return f"{int(hours):02}:{int(minutes):02}:{seconds:02.2f}"
    else:
        return f"{int(hours):02}:{int(minutes):02}:{seconds:02.0f}"


def resolve_verbosity(verbosity: str) -> str:
    """Resolve the run_card "auto" verbosity to "pretty"/"log" depending on
    whether stdout is attached to a terminal; other values pass through
    unchanged."""
    if verbosity == "auto":
        return "pretty" if sys.stdout.isatty() else "log"
    return verbosity


def resolve_seed(seed: int) -> int:
    """Resolve the run_card "seed": -1 draws a fresh 64-bit seed via
    os.urandom, any other value is used as-is."""
    if seed == -1:
        return int.from_bytes(os.urandom(8), "big")
    return seed


def device_type_of(device_name: str) -> str:
    """The device type of one 'device' run_card entry (the optional ':<index>'
    suffix is stripped)."""
    return device_name.split(":")[0]


def backend_of(device_name: str, cpu_mode: str) -> str:
    """The build BACKEND that serves one 'device' run_card entry.

    ``cpu_mode`` describes the SIMD width of the CPU code and is therefore
    meaningless for the cuda/hip devices: those build the backend named after
    the device itself.
    """
    device_type = device_type_of(device_name)
    return cpu_mode if device_type == "cpu" else device_type


_PRECISION_2_FPTYPE = {"all64": "d", "all32": "f", "color32": "m", "denom64": "v"}


def fptype_of(precision: str) -> str:
    return _PRECISION_2_FPTYPE.get(str(precision).lower(), str(precision).lower())


def resolve_auto_backend(build_path: str) -> str:
    """Ask the matrix-element Makefile to resolve the ``auto`` cpu_mode.

    Given the produced shared library will have its name taken from the resolved
    backend name, we need to make sure the detection is taking place correctly
    and catch any possible error.
    """
    command = ["make", "-n", "BACKEND=auto", "detect-backend"]
    try:
        result = subprocess.run(
            command,
            cwd=build_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not run make to resolve cpu_mode='auto' in '{build_path}': {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part.strip()
        )
        detail = f"\nmake output:\n{output}" if output else ""
        raise RuntimeError(
            f"Could not resolve cpu_mode='auto' in '{build_path}': "
            f"Exit status {exc.returncode}.{detail}"
        ) from exc

    match = re.search(
        r"^BACKEND=(\S+) \(was auto\)$", result.stdout, re.MULTILINE
    )
    if match is None or match.group(1) == "auto":
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        detail = f"\nmake output:\n{output}" if output else ""
        raise RuntimeError(
            f"Could not resolve cpu_mode='auto' in '{build_path}': "
            f"make failed to report a backend.{detail}"
        )
    return match.group(1)


def me_parameters(run_card):
    """Run-time parameters of the matrix-element libraries, passed to each
    instance through umami_set_parameter by ms.Context.load_matrix_element:
    the window of the $-excluded propagators is the run card bw_cutoff (the
    one the phase space uses around the resonances too), and with
    [generation] interference_helicity = "summed" the helicity sum as the
    |M|^2 of an interference (see interference_helicities_summed). That one is
    only sent when asked for: a library made before it existed refuses an
    unknown parameter."""
    parameters = {"bwcutoff": float(run_card["phasespace"]["bw_cutoff"])}
    if interference_helicities_summed(run_card):
        parameters["interference_helicity_summed"] = 1.
    return parameters


def mlm_pdf_ladder(run_card, mlm_clustering, is_decay=False, leptonic=False):
    """Whether an MLM sample's beam densities walk the clustering ladder, the
    pdf half of madevent's rewgt: under scale_scheme "madevent", and under
    "clustering_mean" with clustering_mean_pdf_ladder. The Python side of
    EnergyScale::mlm_pdf_reweighting, which decides the same for the kernel;
    the systematics then vary the ladder as madevent's do."""
    beam = run_card["beam"]
    return bool(
        mlm_clustering and not is_decay and not leptonic
        and beam["pdf_reweighting"]
        and (beam["scale_scheme"] == "madevent"
             or beam["clustering_mean_pdf_ladder"]))


def interference_helicities_summed(run_card):
    """Whether [generation] interference_helicity is "summed".

    For an interference (squared split orders dropping a component), a helicity
    can contribute negatively. "exact", the default, draws the helicity of each
    event on |T_i| and gives the event sign(T_i) * sum_j |T_j|, as madevent does:
    the helicity of the event is exact, at the cost of a larger sum|w| (about 2x
    the events for the same precision). "summed" keeps the helicity sum sum_j T_j
    as the weight (lower variance), and the helicity then means nothing: it is
    written as 9 in the LHE (lhe_helicities)."""
    return run_card["generation"].get("interference_helicity", "exact") == "summed"


def lhe_helicities(meta, run_card):
    """The helicity table the LHE completer writes for a subprocess: its own,
    or 9 for every particle of an interference subprocess whose |M|^2 is the
    helicity sum (interference_helicities_summed)."""
    helicities = meta["helicities"]
    if meta.get("interference") and interference_helicities_summed(run_card):
        return [[9] * len(row) for row in helicities]
    return helicities


def lhe_weight_info(status):
    """(IDWTUP, XMAXUP) for the LHE <init> block of a run with this generator
    status. The events are unweighted to +-sigma_abs, keeping the sign of the
    integrand, and XMAXUP is that unit weight rather than the signed cross
    section. A sample that can hold negative weights -- an interference --
    declares it with IDWTUP = -4, as madevent does: the cross section is then
    the mean of the signed event weights. With -3 a shower takes it as
    |XSECUP| times the mean sign instead (Pythia8 does), which is not the
    cross section, and +3 is for positive weights only. gridpack.py applies the
    same rule inline, since it must not import this module."""
    negative = status.mean < 0 or status.mean_abs > abs(status.mean) * (1 + 1e-12)
    return (-4 if negative else 3), status.mean_abs


@dataclass
class Channel:
    phasespace_mapping: ms.PhaseSpaceMapping
    adaptive_mapping: ms.Flow | ms.VegasMapping
    discrete_sym: ms.DiscreteSampler | ms.DiscreteFlow | None
    discrete_flavor: ms.DiscreteSampler | ms.DiscreteFlow | None
    channel_weight_indices: list[int] | None
    name: str
    active_flavors: list[int]
    event_generator: ms.ChannelEventGenerator | None = None


@dataclass
class PhaseSpace:
    mode: Literal["multichannel", "flat", "both"]
    channels: list[Channel]
    symfact: list[int | None]
    first_chan_weight_remap: list[list[int]] = field(default_factory=list)
    first_remapped_chan_count: int = 0
    second_chan_weight_remap: list[int] = field(default_factory=list)
    second_remapped_chan_count: int = 0
    prop_chan_weights: ms.PropagatorChannelWeights | None = None
    subchan_weights: ms.SubchannelWeights | None = None
    cwnet: ms.ChannelWeightNetwork | None = None


class MultiChannelData(NamedTuple):
    amp2_remaps: list[list[int]]
    symfact: list[int | None]
    topologies: list[list[ms.Topology]]
    permutations: list[list[list[int]]]
    channel_indices: list[list[int]]
    channel_weight_indices: list[list[list[int]]]
    diagram_indices: list[list[int]]
    diagram_color_indices: list[list[list[int]]]
    diagram_propagator_pdgs: list[list[list[int]]]
    active_flavors: list[list[list[int]]]
    qcd_s_channel_count: list[int]


@dataclass
class CutItem:
    observable_kwargs: dict
    min: float
    max: float
    mode: str


@dataclass
class HistItem:
    observable_kwargs: dict
    min: float
    max: float
    bin_count: int
    # the distribution of the event weight itself (the reserved "weight" key
    # of [histograms]); it is read off the event record, not computed from
    # the momenta, so it has no observable
    name: str = ""
    from_weight: bool = False


class MadgraphProcess:
    # write each event's squared matrix element into the events (the
    # "matrix_element" column of events.npy), for MadtRex to normalise its
    # reweighting with instead of evaluating it again; set before survey()
    store_matrix_elements = False

    def __init__(self, extra_make_args=(), split_flavors=False):
        # appended to the make command that builds the matrix elements
        self.extra_make_args = list(extra_make_args)
        # one subprocess per flavour of a decay (split_decay_flavors)
        self.split_flavors = split_flavors
        self.madnis_trained = False
        self.lhe_completer = None
        self.load_cards()
        self.init_backend()
        self.init_event_dir()
        self.init_context()
        self.init_custom_functions()
        self.init_cuts()
        self.init_histograms()
        self.init_generator_config()
        self.init_beam()
        self.init_subprocesses()

    def load_cards(self) -> None:
        self.run_card = RunCardMG7(os.path.join("Cards", "run_card.toml"))
        misc.configure_gzip(self.run_card.gzip_configuration())
        # Resolved once so every generator built during this run shares the same
        # seed; the concrete value (even if randomly drawn) is recorded in each
        # generator's info.json.
        self.run_seed = resolve_seed(self.run_card["run"]["seed"])
        self.param_card_path = os.path.join("Cards", "param_card.dat")
        self.param_card = ParamCard(self.param_card_path)
        with open(os.path.join("SubProcesses", "subprocesses.json")) as f:
            self.subprocess_data = json.load(f)
        merge_subprocesses = self.run_card["phasespace"]["merge_subprocesses"]
        if merge_subprocesses and self.run_card["beam"]["dynamical_scale_choice"] == "mlm":
            # A merged subprocess bundles matrix elements whose legs differ -
            # q q~ > l l g next to q g > l l q - while the MLM clustering is
            # built once per subprocess from a single assignment of flavours to
            # legs. It would cluster the gluon-initiated elements as if their
            # beam were a quark, and pick the history per diagram from the
            # weights of one element only.
            logger.warning(
                "merge_subprocesses is not supported with dynamical_scale_choice "
                "= mlm; the subprocesses are kept separate"
            )
            merge_subprocesses = False
        if merge_subprocesses:
            with open(os.path.join("SubProcesses", "merged_subprocesses.json")) as f:
                self.merged_subprocess_data = json.load(f)
        else:
            self.merged_subprocess_data = None

        self.init_decay_mode()
        self.init_me_frame()
        if interference_helicities_summed(self.run_card):
            count = sum(1 for meta in self.subprocess_data if meta.get("interference"))
            if count:
                logger.info(
                    "interference_helicity = summed: the weight of the %d interference "
                    "subprocess(es) is the helicity sum, and their helicities are "
                    "written as 9 in the LHE", count)

    def init_me_frame(self) -> None:
        """Resolve the run card's me_frame into the list of external particles
        whose momentum sum defines the frame the matrix element is evaluated in.

        Empty (the default) means no boost: the matrix element sees the momenta
        in the frame they are generated in. Unlike madevent, which hands its
        matrix element momenta that are already in the partonic centre of mass,
        madspace generates them in the lab frame, so asking for madevent's own
        frame ([1, 2]) does mean an actual boost here. It changes nothing for a
        Lorentz invariant matrix element (only the rounding), but it is what
        defines the polarisation axes of a polarised one.

        The particle numbers are checked here rather than only in madspace,
        which would not see them until the integrands are built -- long after
        the matrix-element libraries have been compiled. Every subprocess is
        checked against its own particles: an output can mix multiplicities
        (p p > z j and p p > z j j, say, for MLM merging), and madspace builds
        one matrix element per subprocess, each refusing a particle it does not
        have. That is stricter than madevent, which ignores the particles a
        process does not have and so changes the frame from one multiplicity
        to the next without a word.
        """
        me_frame = list(self.run_card["run"]["me_frame"])
        pdgs = [clean_pids(meta["incoming"]) + clean_pids(meta["outgoing"])
                for meta in self.subprocess_data]
        counts = sorted({len(pdg_list) for pdg_list in pdgs})
        for meta, pdg_list in zip(self.subprocess_data, pdgs):
            for index in me_frame:
                if 1 <= index <= len(pdg_list):
                    continue
                process = "%s > %s" % (
                    " ".join(str(pid) for pid in meta["incoming"]),
                    " ".join(str(pid) for pid in meta["outgoing"]),
                )
                mixed = (
                    f"; this output mixes subprocesses with "
                    f"{', '.join(str(count) for count in counts)} external "
                    "particles, and me_frame has to name particles every one "
                    "of them has"
                    if len(counts) > 1 else ""
                )
                raise ValueError(
                    f"me_frame particle {index} out of range: subprocess "
                    f"{process} has {len(pdg_list)} external particles{mixed}"
                )
        if len(me_frame) == 1:
            # A massless particle has no rest frame: its momentum squares to
            # zero, so the boost divides by it and every momentum comes back as
            # a NaN, silently. Name the mistake instead.
            massless = [pdg_list[me_frame[0] - 1] for pdg_list in pdgs
                        if self.get_mass(pdg_list[me_frame[0] - 1]) == 0.]
            if massless:
                raise ValueError(
                    f"me_frame = {me_frame} asks for the rest frame of particle "
                    f"{me_frame[0]}, which is massless (pdg "
                    f"{sorted(set(massless))}) and has none. Name a massive "
                    "particle, or several particles whose sum is massive."
                )
        self.me_frame = me_frame
        self.incoming_count = 1 if self.is_decay else 2

    def init_decay_mode(self) -> None:
        """Decide whether this directory is a decay (1 -> n) or a collision.

        Read off the exported process rather than the run card: the two cannot
        then disagree. MadSpin generates its decay matrix elements this way.
        """
        incoming_counts = {
            len(clean_pids(meta["incoming"])) for meta in self.subprocess_data
        }
        if incoming_counts - {1, 2}:
            raise ValueError(
                f"processes with {sorted(incoming_counts)} incoming particles "
                "are not supported"
            )
        if len(incoming_counts) > 1:
            raise ValueError(
                "cannot mix decays and collisions in one output directory"
            )
        self.is_decay = incoming_counts == {1}
        if not self.is_decay:
            self.decaying_mass = None
            single = [meta["outgoing"] for meta in self.subprocess_data
                      if len(clean_pids(meta["outgoing"])) == 1]
            if single and self.run_card["beam"]["leptonic"]:
                raise ValueError(
                    f"2 -> 1 processes ({single[0]} ...) need beams with a PDF "
                    "(leptonic = false in [beam]): with fixed beam energies the "
                    "one-particle phase space is a delta function in s_hat"
                )
            return

        masses = {
            self.get_mass(clean_pids(meta["incoming"])[0])
            for meta in self.subprocess_data
        }
        if len(masses) > 1:
            raise ValueError(
                f"decaying particles have different masses: {sorted(masses)}"
            )
        self.decaying_mass = masses.pop()
        if self.decaying_mass <= 0.0:
            raise ValueError("the decaying particle must have a non-zero mass")
        self.drop_closed_channels()
        if getattr(self, "split_flavors", False):
            self.split_decay_flavors()

    def split_decay_flavors(self) -> None:
        """Integrate every flavour of a decay subprocess on its own.

        A subprocess can hold several flavours with different matrix elements
        (z > d d~ g and z > u u~ g in one P1_z_QQxg): its integrand samples
        them, and its result is their sum. The widths of compute_decay_widths
        need each of them, so each flavour becomes a subprocess of its own,
        with the same matrix-element library and flavour index and the
        diagrams active for it (see _single_flavor_metas).
        """
        if self.merged_subprocess_data is not None:
            raise ValueError(
                "split_flavors needs merge_subprocesses = false: "
                "merged_subprocesses.json indexes the unsplit subprocesses"
            )
        self.subprocess_data = [
            single
            for meta in self.subprocess_data
            for single in _single_flavor_metas(meta)
        ]

    def drop_closed_channels(self) -> None:
        """Remove subprocesses the decaying particle is too light to produce.

        A multiparticle decay definition enumerates every vertex the model
        allows, closed ones included: "t > b w+, w+ > all all" yields
        t > b t b~ (and b W+ Z, b W+ h, ...), which need more mass than the top
        has. Their partial width is exactly zero, but the phase-space mapping
        has no physical point to hand back -- the invariant's lower bound ends
        up above its upper bound -- so it produces NaN momenta and poisons the
        whole integral. Drop them here instead.
        """
        kept, dropped = [], []
        for meta in self.subprocess_data:
            total = sum(
                self.get_mass(pid) for pid in clean_pids(meta["outgoing"])
            )
            if total < self.decaying_mass:
                kept.append(meta)
            else:
                dropped.append((meta["outgoing"], total))
        if dropped and self.merged_subprocess_data is not None:
            # merged_subprocesses.json indexes the *unfiltered* subprocess
            # list, so dropping entries here would silently shift every index
            # it holds. Refuse rather than mis-map.
            raise ValueError(
                "merge_subprocesses is not supported for a decay directory "
                "with kinematically closed channels"
            )
        if dropped:
            for outgoing, total in dropped:
                logger.info(
                    "skipping closed decay channel -> %s (needs %.4g GeV, "
                    "the decaying particle has %.4g GeV)",
                    outgoing, total, self.decaying_mass,
                )
        if not kept:
            raise ValueError(
                "every decay channel is kinematically closed: the decaying "
                f"particle's mass is {self.decaying_mass} GeV"
            )
        self.subprocess_data = kept

    def init_backend(self) -> None:
        ms.set_simd_mode(self.run_card["run"]["madspace_cpu_mode"])

    def init_event_dir(self) -> None:
        output_dir = self.run_card["gridpack"]["output_dir"]
        temp_dir = self.run_card["gridpack"]["temp_output_dir"]
        if output_dir:
            self.run_path = os.path.abspath(os.path.expanduser(output_dir))
            os.makedirs(self.run_path, exist_ok=True)
        else:
            run_name = self.run_card["run"]["run_name"]
            os.makedirs("Events", exist_ok=True)
            run_dir_prefix = os.path.join("Events", f"{run_name}_")
            existing_run_dirs = glob.glob(f"{run_dir_prefix}*")
            run_index = 1
            for run_dir in existing_run_dirs:
                run_index_str = run_dir[len(run_dir_prefix):]
                if run_index_str.isnumeric():
                    run_index = max(run_index, int(run_index_str) + 1)
            while True:
                try:
                    self.run_path = f"{run_dir_prefix}{run_index:02d}"
                    os.mkdir(self.run_path)
                    break
                except FileExistsError:
                    run_index += 1
        # temporary npy files of the channel generators
        self.auto_temp_path = temp_dir == "auto"
        if self.auto_temp_path:
            self.temp_path = tempfile.mkdtemp(prefix="mg7_events_")
        elif temp_dir:
            self.temp_path = os.path.abspath(os.path.expanduser(temp_dir))
            os.makedirs(self.temp_path, exist_ok=True)
        else:
            self.temp_path = self.run_path
        # Absolute on purpose. StatusFile is a C++ object that finishes its
        # write from its destructor (rename info.json.tmp -> info.json), and
        # that destructor runs whenever Python gets round to collecting the
        # process object. When the launcher runs in MG5's process, an
        # interrupted run is collected *after* the launch command has restored
        # the working directory, and a relative path would then resolve
        # somewhere else -- the rename throws a C++ filesystem_error out of a
        # destructor, which is an immediate abort.
        self.status_file = ms.StatusFile(
            os.path.abspath(os.path.join(self.run_path, "info.json")))

    def init_context(self) -> None:
        device_names = self.run_card["run"]["device"]
        self.contexts = []
        self.device_types = []
        self.devices = []
        self.pool_sizes = []
        for device_name in device_names:
            if ":" in device_name:
                device_type, device_index_str = device_name.split(":")
                device_index = int(device_index_str)
            else:
                device_type = device_name
                device_index = 0
            self.device_types.append(device_type)
            if device_type == "cuda":
                device = ms.cuda_device(device_index)
                pool_size = self.run_card["run"]["gpu_thread_pool_size"]
            elif device_type == "hip":
                device = ms.hip_device(device_index)
                pool_size = self.run_card["run"]["gpu_thread_pool_size"]
            else:
                device = ms.cpu_device()
                pool_size = self.run_card["run"]["cpu_thread_pool_size"]
            self.devices.append(device)
            self.pool_sizes.append(pool_size)
            self.contexts.append(ms.Context(device=device, thread_count=pool_size))

    CUSTOM_FUNCTION_KINDS = ("cut", "scale", "pdf", "bias")

    def init_custom_functions(self) -> None:
        """Load the functions of [custom_functions], building the libraries
        first. They are loaded before the matrix elements, so that they have the
        same index in every context and in a gridpack made from this run."""
        self.custom_functions = {}
        self.custom_function_paths = []
        for kind in self.CUSTOM_FUNCTION_KINDS:
            path = self.run_card["custom_functions"][kind]
            if not path:
                continue
            build_dir = f"custom_{kind}"
            if path.endswith(".so") and os.path.isfile(os.path.join(build_dir, "Makefile")):
                self.build_custom_function(build_dir)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"[custom_functions] {kind} = {path!r}: file not found. "
                    "Create a .so library or a .py file with bin/init_custom_func."
                )
            parameters = self.run_card.custom_parameters(kind)
            apis = load_custom_function(
                ms, self.contexts, self.device_types, kind, path, parameters
            )
            if len({api.index() for api in apis}) != 1:
                raise RuntimeError(f"custom {kind} function loaded at different indices")
            self.custom_functions[kind] = apis[0]
            self.custom_function_paths.append([kind, path, parameters])
            logger.info("Using the custom %s function %s", kind, path)

    @staticmethod
    def build_custom_function(build_dir: str) -> None:
        """Run make in a custom_<kind> directory, logging to compile.log."""
        log_path = os.path.join(build_dir, "compile.log")
        with open(log_path, "w") as log:
            returncode = subprocess.call(
                ["make"], cwd=build_dir, stdout=log, stderr=subprocess.STDOUT
            )
        if returncode != 0:
            raise RuntimeError(
                f"Compilation of {build_dir} failed with exit status {returncode}; "
                f"see {log_path} for details"
            )

    def custom_function(self, kind: str, particle_count: int, pid_options: list):
        """The ms.CustomFunction of [custom_functions] <kind>, or None."""
        api = self.custom_functions.get(kind)
        if api is None:
            return None
        function_type = getattr(ms.CustomFunction, kind)
        return ms.CustomFunction(api, function_type, particle_count, pid_options)

    def parse_observable(self, name: str, order_observable: str) -> dict:
        parts = name.split("-")
        sum_momenta = False
        sum_observable = False
        ordered = False
        multiparticles = self.run_card["multiparticles"]

        if len(parts) == 0:
            raise ValueError("Invalid observable name")
        elif len(parts) == 1:
            # event-level observables
            obs_name = parts[0]
            select_pids = []
        else:
            if parts[-1] == "sum":
                sum_observable = True
                obs_name = parts[-2]
                selection = parts[:-2]
            elif parts[-2] == "sum":
                sum_momenta = True
                obs_name = parts[-1]
                selection = parts[:-2]
            else:
                obs_name = parts[-1]
                selection = parts[:-1]
            select_pids = []
            order_indices = []
            for mp_name in selection:
                mp_parts = mp_name.split("_")
                if mp_parts[-1].isnumeric():
                    order_indices.append(int(mp_parts[-1]))
                    select_pids.append(multiparticles["_".join(mp_parts[:-1])])
                    ordered = True
                else:
                    order_indices.append(0)
                    select_pids.append(multiparticles[mp_name])

        return dict(
            observable=obs_name,
            select_pids=select_pids,
            sum_momenta=sum_momenta,
            sum_observable=sum_observable,
            order_observable=order_observable if ordered else None,
            order_indices=order_indices if ordered else [],
            ignore_incoming=True,
            name=name,
        )

    def apply_auto_ptj_mjj(self, cuts: dict) -> None:
        """madevent's auto_ptj_mjj, from Template/LO/SubProcesses/setcuts.f.

        A merging cut already bounds every matrix-element jet from below, so
        raising the jet pt and pair-mass cuts to xqcut removes no phase space
        that survives the merging anyway while making the integration much more
        efficient. The jet dR cuts go the other way and are dropped:
        xqcut supersedes them, and a dR cut would carve a hole out of exactly
        the region the parton shower is supposed to fill, leaving a gap in the
        jet rates at the merging scale.

        With the option off, madevent still drops the dR cuts, and only zeroes
        a pt or pair-mass cut that sits above xqcut - it never tightens.
        """
        if self.is_decay or self.run_card["beam"]["dynamical_scale_choice"] != "mlm":
            return
        xqcut = self.run_card["phasespace"]["xqcut"]
        if xqcut <= 0:
            return
        for key in ("jet-delta_r", "jet-lepton-delta_r"):
            if cuts.get(key, {}).get("min", 0.0) > 0.0:
                cuts[key]["min"] = 0.0
        auto = self.run_card["phasespace"]["auto_ptj_mjj"]
        # Both halves of madevent's rule matter, and for different reasons.
        #
        # ptj reaches the phase space only for jets that are leaf children of
        # the root: a jet inside a composite s-channel node - two jets off one
        # gluon, say - has its pt_min reset to zero in PhaseSpaceMapping, and
        # that node's invariant is then sampled from threshold upward knowing
        # nothing about the cut. mjj is what bounds those nodes, because a
        # pairwise invariant-mass minimum is carried into the propagator's own
        # e_min and so into the s_min the sampler uses.
        #
        # Leaving mjj out is what makes the highest multiplicity misbehave at a
        # tight xqcut: almost every point it proposes is vetoed, and the few
        # survivors sit against a boundary it cannot see.
        for key in ("jet-pt", "jet-pair_mass"):
            if auto:
                cuts.setdefault(key, {})["min"] = xqcut
            elif cuts.get(key, {}).get("min", 0.0) > xqcut:
                cuts[key]["min"] = 0.0

    def init_cuts(self) -> None:
        inf = float("inf")
        order_observable = self.run_card["cuts"].get("order_by", "pt")
        cuts = {
            key: dict(values)
            for key, values in self.run_card["cuts"].items()
            if key != "order_by"
        }
        self.apply_auto_ptj_mjj(cuts)
        self.cut_data = [
            CutItem(
                observable_kwargs=self.parse_observable(key, order_observable),
                min=values.get("min", -inf),
                max=values.get("max", inf),
                mode=values.get("mode", "all"),
            )
            for key, values in cuts.items()
        ]

    # [histograms] key that means "the distribution of the event weight",
    # normalised to the unit weight sigma_abs, rather than an observable of the
    # momenta: an unweighted sample is a spike at 1 (and at -1 for its negative
    # weights) and a partially unweighted one shows its spread.
    weight_histogram_key = "weight"

    def init_histograms(self) -> None:
        order_observable = self.run_card["histograms"].get("order_by", "pt")
        self.hist_data = []
        for key, values in self.run_card["histograms"].items():
            if key == "order_by":
                continue
            from_weight = key == self.weight_histogram_key
            self.hist_data.append(HistItem(
                observable_kwargs=(
                    None if from_weight
                    else self.parse_observable(key, order_observable)),
                min=values["min"],
                max=values["max"],
                bin_count=values["bin_count"],
                name=key,
                from_weight=from_weight,
            ))

    def ensure_pdf_set(self, pdf_set: str) -> misc.LhapdfPaths:
        """Make sure the requested LHAPDF set is available, downloading it into
        the first writable PDF directory if needed, and return the (possibly
        updated) LHAPDF resolution. Nothing is downloaded when the set is
        already present. LHAPDF itself is not needed: madspace reads the grid
        files directly, so without a usable lhapdf-config (or when 'lhapdf
        install' fails) the set is fetched straight from the LHAPDF server."""
        paths = lhapdf_paths()
        if paths.find_set(pdf_set):
            return paths
        if not paths.download_path:
            logger.debug("no writable PDF directory: cannot download %s", pdf_set)
            return paths
        logger.info("PDF set %s not found; downloading into %s",
                    pdf_set, paths.download_path)
        if paths.config:
            try:
                from madgraph.interface.common_run_interface import CommonRunCmd
                os.makedirs(paths.download_path, exist_ok=True)
                CommonRunCmd.install_lhapdf_pdfset_static(
                    paths.config, paths.download_path, pdf_set)
            except Exception as err:
                logger.debug("lhapdf could not install %s: %s", pdf_set, err)
        if not os.path.isdir(os.path.join(paths.download_path, pdf_set)):
            try:
                misc.download_lhapdf_set(pdf_set, paths.download_path)
            except Exception as err:
                logger.warning("Could not download PDF set %s: %s", pdf_set, err)
                return paths
        global _LHAPDF
        _LHAPDF = paths.with_data_path(paths.download_path)
        # subprocesses (systematics, MadSpin, reweight) find it here too
        os.environ["LHAPDF_DATA_PATH"] = os.pathsep.join(_LHAPDF.data_paths)
        return _LHAPDF

    def _no_pdf_message(self, pdf_set: str) -> str:
        """Explain a missing PDF set in terms of the settings that fix it."""
        searched = os.pathsep.join(self.lhapdf.data_paths) or \
            "(no PDF set directory could be located)"
        if self.lhapdf.config:
            found = "using lhapdf-config: %s" % self.lhapdf.config
        else:
            found = "no lhapdf-config found (not needed for the download)"
        if self.lhapdf.download_path:
            download = ("the download from %s into %s failed (see the "
                        "warning above)" % (misc.LHAPDF_SETS_URL,
                                            self.lhapdf.download_path))
        else:
            download = "no writable directory to download the set into"
        return (
            "LHAPDF set %r, requested by [beam] pdf1/pdf2 in Cards/run_card.toml, "
            "was not found.\n"
            "  searched: %s\n"
            "  %s\n"
            "  %s\n"
            "Fix this in one of the following ways:\n"
            "  * check the set name against %s/pdfsets.index and your network\n"
            "  * install LHAPDF for this MadGraph:  MG7> install lhapdf6\n"
            "  * point MadGraph at an existing one:\n"
            "        MG7> set lhapdf /path/to/lhapdf-config\n"
            "        MG7> save options\n"
            "  * or set it for this directory only, by adding\n"
            "        lhapdf = /path/to/lhapdf-config\n"
            "    to %s\n"
            "  * or set $LHAPDF_DATA_PATH to a directory containing %r."
            % (pdf_set, searched, found, download, misc.LHAPDF_SETS_URL,
               os.path.join("Cards", "me5_configuration.txt"), pdf_set))

    def init_beam(self) -> None:
        beam_args = self.run_card["beam"]

        # built lazily by build_systematics (needs the subprocess data)
        self.systematics = None
        self.systematics_data = None
        self.systematics_context = None
        self.event_histograms = None
        self.event_histograms_context = None

        if self.is_decay:
            # No beams: the total energy is the decaying particle's mass, and
            # "leptonic" is what the mappings call "no parton luminosity".
            self.e_cm = self.decaying_mass
            self.beam_energies = (float(self.e_cm), 0.0)
            self.beam_rapidity = 0.0
            self.leptonic = True
        else:
            ebeam1 = float(beam_args["ebeam1"])
            ebeam2 = float(beam_args["ebeam2"])
            self.beam_energies = (ebeam1, ebeam2)
            # The phase space is generated in the beams' centre-of-mass frame
            # (massless beams: s = 4 E1 E2) and boosted by this rapidity into
            # the lab frame, where the events are written and the cuts applied.
            self.e_cm = 2.0 * math.sqrt(ebeam1 * ebeam2)
            self.beam_rapidity = 0.5 * math.log(ebeam1 / ebeam2)
            self.leptonic = beam_args["leptonic"]

        dynamical_scales = {
            "transverse_energy": ms.EnergyScale.transverse_energy,
            "transverse_mass": ms.EnergyScale.transverse_mass,
            "half_transverse_mass": ms.EnergyScale.half_transverse_mass,
            "partonic_energy": ms.EnergyScale.partonic_energy,
        }
        # "mlm" is not one of the closed-form scale choices: it needs the diagram
        # topologies of each subprocess, so the EnergyScale is built later, in
        # MadgraphSubprocess.init_madspace, from an MLMClustering instance.
        if beam_args["dynamical_scale_choice"] == "mlm":
            self.mlm_clustering = True
            self.scale_kwargs = {}
        elif beam_args["dynamical_scale_choice"] in dynamical_scales:
            self.mlm_clustering = False
            self.scale_kwargs = dict(
                dynamical_scale_type=dynamical_scales[
                    beam_args["dynamical_scale_choice"]
                ],
                ren_scale_fixed=beam_args["fixed_ren_scale"],
                fact_scale_fixed=beam_args["fixed_fact_scale"],
                ren_scale=beam_args["ren_scale"],
                fact_scale1=beam_args["fact_scale1"],
                fact_scale2=beam_args["fact_scale2"],
                scale_factor=float(beam_args.get("scale_factor", 1.)),
            )
        else:
            raise ValueError("Unknown dynamical scale choice")
        if self.is_decay and self.custom_functions.pop("scale", None) is not None:
            logger.warning("custom scale ignored: a decay uses the decaying mass")
        if self.is_decay and self.custom_functions.pop("pdf", None) is not None:
            logger.warning("custom pdf ignored: a decay has no parton densities")
        if self.mlm_clustering and not self.is_decay:
            # The merging needs the scales of its own clustering history: a
            # custom scale or a factor on it would move the merging scale away
            # from the one the shower is matched to.
            if self.custom_functions.pop("scale", None) is not None:
                logger.warning(
                    "custom scale ignored under dynamical_scale_choice = mlm")
            if float(beam_args.get("scale_factor", 1.)) != 1.:
                logger.warning(
                    "scale_factor ignored under dynamical_scale_choice = mlm")
        if self.is_decay:
            # One scale is available for a decay -- the decaying mass -- so use
            # it, fixed, whatever the card asks for. That includes "mlm": a
            # decay has no beams to cluster into, so there is no merging to do.
            if self.mlm_clustering:
                logger.warning(
                    "dynamical_scale_choice = mlm has no meaning for a decay; "
                    "using the decaying mass as a fixed scale instead"
                )
                self.mlm_clustering = False
            self.scale_kwargs.setdefault(
                "dynamical_scale_type", ms.EnergyScale.half_transverse_mass
            )
            self.scale_kwargs.update(
                ren_scale_fixed=True,
                fact_scale_fixed=True,
                ren_scale=self.decaying_mass,
                fact_scale1=self.decaying_mass,
                fact_scale2=self.decaying_mass,
                scale_factor=1.,
            )
            self.pdf_grid = None
            self.pdf_grid2 = None
            self.lhapdf = None
            self.pdf_dir = None
            self.pdf_dir2 = None
            self.fixed_alphas_info = self.write_fixed_alphas_info()
            self.alphas_grid = ms.AlphaSGrid(self.fixed_alphas_info)
            for context in self.contexts:
                self.alphas_grid.initialize_globals(context)
            self.running_coupling = ms.RunningCoupling(self.alphas_grid)
            return

        pdf_set = beam_args["pdf1"]
        self.pdf_set = pdf_set
        self.pdf_set2 = beam_args["pdf2"]
        self.fixed_alphas_info = None
        self.lhapdf = self.ensure_pdf_set(pdf_set)
        self.pdf_dir = self.lhapdf.find_set(pdf_set)
        if self.pdf_dir is None:
            raise RuntimeError(self._no_pdf_message(pdf_set))
        # A second grid only when the beams differ (and have a PDF at all);
        # otherwise beam 2 shares the first one.
        self.pdf_grid2 = None
        self.pdf_dir2 = self.pdf_dir
        if self.pdf_set2 != pdf_set and not self.leptonic:
            self.pdf_dir2 = self.ensure_pdf_set(self.pdf_set2).find_set(self.pdf_set2)
            if self.pdf_dir2 is None:
                raise RuntimeError(self._no_pdf_message(self.pdf_set2))
            if "pdf" not in self.custom_functions:
                self.pdf_grid2 = ms.PdfGrid(os.path.join(
                    self.pdf_dir2, self.pdf_set2, f"{self.pdf_set2}_0000.dat"))
                logger.info("PDF sets: %s (beam 1), %s (beam 2); alpha_s from %s",
                            pdf_set, self.pdf_set2, pdf_set)
        self.alphas_grid = ms.AlphaSGrid(os.path.join(self.pdf_dir, pdf_set, f"{pdf_set}.info"))
        if "pdf" in self.custom_functions:
            # alpha_s still comes from the PDF set of the run card
            self.pdf_grid = ms.CustomPdf(self.custom_functions["pdf"])
        else:
            self.pdf_grid = ms.PdfGrid(
                os.path.join(self.pdf_dir, pdf_set, f"{pdf_set}_0000.dat")
            )
        # The grid only covers a band in Q. Asking for a density above its
        # highest Q reads past the end of the interpolation and comes back as a
        # NaN, so the scale is capped there. A dynamical scale rarely gets near
        # it, but the madevent MLM scheme, which multiplies four clustering
        # scales together under a fourth root, can. A custom PDF brings no grid
        # to read the range from, so it is left uncapped.
        grids = [grid for grid in (self.pdf_grid, self.pdf_grid2)
                 if isinstance(grid, ms.PdfGrid)]
        self.max_scale = min(max(grid.q) for grid in grids) if grids else 0.0
        # the bottom of the same range, where the densities freeze
        self.min_pdf_scale = max(min(grid.q) for grid in grids) if grids else 0.0
        for context in self.contexts:
            if not isinstance(self.pdf_grid, ms.CustomPdf):
                self.pdf_grid.initialize_globals(context)
            if self.pdf_grid2 is not None:
                # the prefix ms.Integrand reads the second beam's grid from
                # (Integrand::pdf2_prefix)
                self.pdf_grid2.initialize_globals(context, "beam2")
            self.alphas_grid.initialize_globals(context)
        self.running_coupling = ms.RunningCoupling(self.alphas_grid)

    # ------------------------------------------------------------------
    # scale / PDF systematics
    # ------------------------------------------------------------------
    def systematics_enabled(self) -> bool:
        """The native scale/PDF weights are requested ([systematics] enable, or
        the legacy [generation] systematics alias)."""
        try:
            enabled = bool(self.run_card["systematics"]["enable"])
        except Exception:
            enabled = False
        if not enabled:
            try:
                enabled = bool(self.run_card["generation"]["systematics"])
            except Exception:
                enabled = False
        if enabled and "pdf" in getattr(self, "custom_functions", {}):
            # the variations re-evaluate the PDF set, which a custom PDF replaces
            if not getattr(self, "_warned_custom_pdf_systematics", False):
                logger.warning("systematics are disabled with a custom pdf")
                self._warned_custom_pdf_systematics = True
            return False
        return enabled

    @staticmethod
    def pdf_set_info(info_path: str) -> dict:
        """Read SetIndex, NumMembers, ErrorType and SetDesc from an LHAPDF
        .info file (missing keys get neutral defaults)."""
        info = {"SetIndex": 0, "NumMembers": 1, "ErrorType": "", "SetDesc": ""}
        try:
            with open(info_path) as f:
                for line in f:
                    key, sep, value = line.partition(":")
                    key = key.strip()
                    if not sep or key not in info:
                        continue
                    value = value.strip().strip('"').strip("'")
                    if key in ("SetIndex", "NumMembers"):
                        info[key] = int(value)
                    else:
                        info[key] = value
        except OSError as err:
            logger.warning("could not read %s: %s", info_path, err)
        return info

    def resolve_pdf_set_name(self, entry: str) -> tuple[str, int | None]:
        """Turn a [systematics] pdf entry ("<set>", "<lhaid>", "<set>@<member>",
        "<lhaid>@<member>") into (set name, member or None) using the LHAPDF
        pdfsets.index when an id is given."""
        member = None
        if "@" in entry:
            entry, member_str = entry.split("@", 1)
            member = int(member_str)
        entry = entry.strip()
        if not entry.isdigit():
            return entry, member
        lhaid = int(entry)
        index_path = os.path.join(self.pdf_dir or "", "pdfsets.index")
        sets = []
        try:
            with open(index_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        sets.append((int(parts[0]), parts[1]))
        except OSError:
            raise ValueError(
                "cannot resolve LHAPDF id %s: no %s" % (lhaid, index_path))
        for set_id, name in sets:
            if set_id == lhaid:
                return name, member
        # an id inside a set's member range
        sets.sort()
        for (set_id, name), (next_id, _) in zip(sets, sets[1:] + [(None, None)]):
            if set_id < lhaid and (next_id is None or lhaid < next_id):
                return name, lhaid - set_id if member is None else member
        raise ValueError("cannot resolve LHAPDF id %s" % lhaid)

    def resolve_pdf_variations(self) -> list:
        """Expand the [systematics] pdf entries into ms.PdfMemberSpec objects
        (downloading the sets when needed). The nominal member itself is never
        listed: the calculator adds it as the central weight of its group."""
        entries = self.run_card["systematics"]["pdf"]
        if isinstance(entries, str):
            entries = [entries]
        specs = []
        seen = set()
        for raw in entries:
            entry = str(raw).strip()
            if not entry or entry.lower() in ("none", "no", "false"):
                continue
            if entry.lower() == "central":
                continue
            if entry.lower() == "errorset":
                set_name, member = self.pdf_set, None
            else:
                set_name, member = self.resolve_pdf_set_name(entry)
            variation_lhapdf = self.ensure_pdf_set(set_name)
            variation_dir = variation_lhapdf.find_set(set_name)
            if variation_dir is None:
                logger.warning("PDF set %s not available; its variation is skipped", set_name)
                continue
            info_path = os.path.join(variation_dir, set_name, f"{set_name}.info")
            if not os.path.exists(info_path):
                logger.warning("PDF set %s not available; its variation is skipped", set_name)
                continue
            info = self.pdf_set_info(info_path)
            is_nominal = set_name == self.pdf_set
            if member is not None:
                members = [member]
            else:
                members = list(range(info["NumMembers"]))
                if is_nominal:
                    members = members[1:]
                    if not members:
                        logger.warning(
                            "PDF set %s has a single member: no PDF uncertainty can be "
                            "computed from it. Use a set with error members in "
                            "[systematics] pdf (e.g. a PDF4LHC or NNPDF error set).",
                            set_name)
            for m in members:
                if is_nominal and m == 0:
                    continue
                if (set_name, m) in seen:
                    continue
                seen.add((set_name, m))
                grid_file = os.path.join(variation_dir, set_name, f"{set_name}_{m:04d}.dat")
                if not os.path.exists(grid_file):
                    logger.warning("PDF member file %s not found; skipped", grid_file)
                    continue
                specs.append(ms.PdfMemberSpec(
                    set_name=set_name, set_lhaid=info["SetIndex"], member=m,
                    grid_file=grid_file, info_file=info_path,
                    error_type=info["ErrorType"], description=info["SetDesc"],
                ))
        return specs

    def build_systematics_args(self) -> list:
        """One ms.SubprocessSystArgs per (unmerged) subprocess: the alpha_s power
        of |M|^2 and the beam parton ids of every flavor index, indexed like the
        LHECompleter (subprocess_index / flavor_index of the combined events)."""
        args = []
        for meta in self.subprocess_data:
            beam_pdgs = [
                [int(flavor["options"][0][0]), int(flavor["options"][0][1])]
                for flavor in meta["flavors"]
            ]
            args.append(ms.SubprocessSystArgs(
                qcd_power=int(meta.get("qcd_power", -1)), beam_pdgs=beam_pdgs))
        return args

    def build_systematics(self):
        """Create the ms.SystematicsCalculator from the [systematics] section, or
        None when disabled. Also keeps the JSON description needed by gridpacks."""
        if not self.systematics_enabled():
            self.systematics = None
            return None
        syst = self.run_card["systematics"]
        config = ms.SystematicsConfig()
        config.mur = [float(v) for v in syst["mur"]]
        config.muf = [float(v) for v in syst["muf"]]
        # the emission scale of the MLM alpha_s vertices, which only an MLM
        # sample with per-vertex alpha_s has
        if (self.mlm_clustering and not self.is_decay
                and self.run_card["beam"]["alphas_reweighting"] != "none"):
            config.alps = [float(v) for v in syst["alps"]]
        else:
            config.alps = [1.0]
        # the pdf reweighting ladder of an MLM sample, varied as madevent's
        # systematics vary it; a varied scale is kept inside the PDF grid,
        # where LHAPDF freezes it for madevent
        config.pdf_ladder = mlm_pdf_ladder(
            self.run_card, self.mlm_clustering, self.is_decay, self.leptonic)
        if config.pdf_ladder:
            config.ladder_min_scale = float(self.min_pdf_scale)
            config.ladder_max_scale = float(self.max_scale)
        config.together = bool(syst["together"])
        config.dyn_scales = self.resolve_dynamical_scales()
        # alternative dynamical scales get the same factor as the nominal one
        config.scale_factor = float(self.run_card["beam"].get("scale_factor", 1.))
        config.write_inputs = bool(syst["write_inputs"])
        config.has_pdf = not self.leptonic
        # Without parton luminosity -- a decay, or leptonic beams -- there is no
        # nominal PDF *set* to describe and no PDF variation to compute, so
        # has_pdf gates the nominal-set metadata and the member list below.
        # The alpha_s grid is a separate matter: the mu_R variations always need
        # one, so its .info file has to be recorded whatever has_pdf says.
        # init_beam() skips the PDF lookup for a decay only -- leptonic beams
        # still resolve a set and take alpha_s from it -- so the path follows
        # self.pdf_dir, and falls back to the constant grid written for a decay.
        if self.pdf_dir is not None:
            info_path = os.path.join(self.pdf_dir, self.pdf_set, f"{self.pdf_set}.info")
        else:
            info_path = self.fixed_alphas_info
        if config.has_pdf:
            info = self.pdf_set_info(info_path)
            config.nominal_set_name = self.pdf_set
            config.nominal_lhaid = info["SetIndex"]
            config.nominal_error_type = info["ErrorType"]
            config.nominal_description = info["SetDesc"]
            if self.pdf_grid2 is None:
                config.pdf_members = self.resolve_pdf_variations()
            else:
                logger.warning(
                    "systematics: the beams use different PDF sets (%s, %s), so "
                    "no PDF variation is computed; the scale variations use both "
                    "sets", self.pdf_set, self.pdf_set2)
                config.pdf_members = []
        else:
            config.pdf_members = []
        args = self.build_systematics_args()
        # the PDFs, alpha_s and (for mixed-order subprocesses) the matrix
        # elements are evaluated with the batched madspace functions on this
        # CPU context
        self.systematics_context = ms.Context(device=ms.cpu_device(), thread_count=1)
        matrix_elements, flavor_remap, me_backend = self.build_systematics_matrix_elements(
            args, self.systematics_context)
        self.systematics = ms.SystematicsCalculator(
            config, args, self.pdf_grid, self.alphas_grid,
            self.systematics_context, matrix_elements, flavor_remap,
            nominal_pdf2=self.pdf_grid2 if config.has_pdf else None)
        for warning in self.systematics.warnings:
            logger.warning("systematics: %s", warning)
        self.systematics_data = {
            "config": json.loads(config.to_json()),
            "subproc_args": [json.loads(a.to_json()) for a in args],
            "nominal_grid_file": os.path.join(
                self.pdf_dir, self.pdf_set, f"{self.pdf_set}_0000.dat")
                if config.has_pdf else None,
            "nominal_grid_file2": os.path.join(
                self.pdf_dir2, self.pdf_set2, f"{self.pdf_set2}_0000.dat")
                if config.has_pdf and self.pdf_grid2 is not None else None,
            "nominal_info_file": info_path,
            # matrix elements re-evaluated for the mixed-order subprocesses
            "me_paths": [meta["me_path"] for meta in self.subprocess_data],
            "me_backend": me_backend,
            "flavor_remap": flavor_remap,
            "me_frame": self.me_frame,
            "incoming_count": self.incoming_count,
        }
        return self.systematics

    _DYN_SCALE_CODES = {
        "transverse_energy": 1, "transverse_mass": 2,
        "half_transverse_mass": 3, "partonic_energy": 4,
    }

    def resolve_dynamical_scales(self) -> list:
        """[systematics] dynamical_scale entries (names or systematics.py codes
        1-4) as codes, without the choice the events were generated with."""
        try:
            entries = self.run_card["systematics"]["dynamical_scale"]
        except Exception:
            return []
        if isinstance(entries, str):
            entries = [entries]
        beam = self.run_card["beam"]
        generated = None
        # a custom scale replaces dynamical_scale_choice, so no variation is redundant
        custom_scale = "scale" in getattr(self, "custom_functions", {})
        if not (beam["fixed_ren_scale"] and beam["fixed_fact_scale"] or custom_scale):
            generated = self._DYN_SCALE_CODES.get(beam["dynamical_scale_choice"])
        codes = []
        for raw in entries:
            entry = str(raw).strip().lower()
            if not entry:
                continue
            if entry.isdigit():
                code = int(entry)
            elif entry in self._DYN_SCALE_CODES:
                code = self._DYN_SCALE_CODES[entry]
            else:
                raise ValueError("unknown dynamical scale choice %r in [systematics] "
                                 "dynamical_scale" % raw)
            if code not in (1, 2, 3, 4):
                raise ValueError("dynamical scale code %s must be 1-4" % code)
            if code == generated or code in codes:
                continue
            codes.append(code)
        return codes

    def build_systematics_matrix_elements(self, args, context):
        """For the subprocesses whose |M|^2 mixes several powers of alpha_s
        (qcd_power = -1), load the matrix element into the systematics CPU
        context so that the renormalisation scale variations can re-evaluate it
        at the varied alpha_s. Returns ([MatrixElement or None per subprocess],
        flavor remap, backend name); empty when nothing is needed or no CPU
        library exists."""
        need = [i for i, a in enumerate(args) if a.qcd_power < 0]
        empty = ([], [], None)
        if not need or self.run_card["run"]["dummy_matrix_element"]:
            return empty
        cpu_backends = [b for b in self.backends if not b.startswith(("cuda", "hip"))]
        if not cpu_backends:
            logger.warning("systematics: no CPU matrix-element library available, the "
                           "renormalisation scale variations of the mixed-order "
                           "subprocesses cannot be computed")
            return empty
        backend = cpu_backends[0]
        matrix_elements = []
        flavor_remap = []
        for i, meta in enumerate(self.subprocess_data):
            flavor_remap.append([int(f["index"]) for f in meta["flavors"]])
            if i not in need:
                matrix_elements.append(None)
                continue
            api = context.load_matrix_element(
                meta["me_path"].format(device=backend, fptype=self.fptype),
                self.param_card_path,
                me_parameters(self.run_card))
            matrix_elements.append(ms.MatrixElement(
                api,
                [ms.MatrixElement.momenta_in, ms.MatrixElement.alpha_s_in,
                 ms.MatrixElement.flavor_in],
                [ms.MatrixElement.matrix_element_out],
                False,
                self.me_frame,
                self.incoming_count,
            ))
        logger.info("systematics: re-evaluating the matrix element of %d mixed-order "
                    "subprocess(es) for the renormalisation scale variations", len(need))
        return matrix_elements, flavor_remap, backend

    def build_event_histograms(self):
        """ms.EventHistograms for the [histograms] observables, filled with the
        written events and all their variation weights (info.json
        "event_histograms"); None without histograms."""
        if not self.hist_data \
                or not self.run_card["run"]["postprocessing_histograms"]:
            self.event_histograms = None
            return None
        context = ms.Context(device=ms.cpu_device(), thread_count=1)
        specs = [
            ms.EventHistogramSpec(
                name=item.name, min=item.min, max=item.max,
                bin_count=item.bin_count, from_weight=item.from_weight)
            for item in self.hist_data
        ]
        from_momenta = [item for item in self.hist_data if not item.from_weight]
        observables = []
        for meta in self.subprocess_data:
            if not from_momenta:
                # nothing to evaluate on the momenta (only the weight is
                # histogrammed): no observable runtime to build
                observables.append(None)
                continue
            all_pids = clean_pids(meta["incoming"]) + clean_pids(meta["outgoing"])
            values = ms.ObservableValues([
                build_observable(
                    observable_pids(meta, item.observable_kwargs),
                    len(meta["incoming"]), item.observable_kwargs
                )
                for item in from_momenta
            ])
            observables.append(ms.SubprocessObservables(values, len(all_pids)))
        # the weight histograms are drawn in units of the unit weight sigma_abs
        # (the cross section of a sample without negative weights)
        self.event_histograms_context = context
        self.event_histograms = ms.EventHistograms(
            context, specs, observables,
            reference_weight=self._mean_event_weight())
        return self.event_histograms

    def _mean_event_weight(self) -> float:
        """The unit weight of the unweighted events, sigma_abs, used as the
        unit of the weight histograms: the cross section of a positive sample,
        and for one with negative weights (an interference) what puts them at
        +-1 rather than dividing by a signed, possibly small, total. 0 (the raw
        weight) when it is not available."""
        try:
            return float(self.event_generator.status().mean_abs)
        except Exception as error:
            logger.debug("no cross section for the weight histograms: %s", error)
            return 0.

    def log_systematics_summary(self) -> None:
        """Print the scale/PDF uncertainties on the total cross section (the
        per-variation cross sections are in events.weights.json / info.json)."""
        if self.systematics is None or self.systematics.weight_count == 0:
            return
        summary = json.loads(self.systematics.summary())
        # the percentages come from systematics_summary, which get_result()
        # reads too: the box and the scan row must not disagree, so neither
        # computes them itself
        xsec = systematics_summary.nominal_cross_section(summary)
        if xsec is None:
            return
        def format_variation(up, down):
            # right-justify the signed numbers (not the sign alone) so the %
            # values line up across rows, with extra spacing between the two
            return f"{'+%.3g' % up:>6}%   {'-%.3g' % down:>6}%"

        scale_count = sum(1 for v in self.systematics.variations
                          if v.is_scale or v.is_emission)
        member_count = len(self.systematics.members)
        rows = [("Variations per event:",
                 f"{self.systematics.weight_count} ({scale_count} scale, {member_count} PDF members)")]
        for pdf in summary.get("pdf", []):
            rows.append(("PDF set:", f"{pdf['pdf_set']}, {pdf['error_type']}"))
        rows.append(("Original cross-section:", f"{xsec} pb"))
        scale = systematics_summary.scale_percentages(summary)
        if scale is not None:
            rows.append(("Scale variation:", format_variation(*scale)))
        emission = systematics_summary.emission_percentages(summary)
        if emission is not None:
            rows.append(("Emission scale variation:", format_variation(*emission)))
        for _entry, up, down in systematics_summary.pdf_percentages(summary):
            rows.append(("PDF variation:", format_variation(up, down)))
        if self.event_generator_config.verbosity == ms.Verbosity.pretty:
            box = ms.PrettyBox("Systematics", len(rows), [24, 0])
            box.set_column(0, [label for label, _ in rows])
            box.set_column(1, [value for _, value in rows])
            box.print_first()
        else:
            for label, value in rows:
                logger.info("systematics, %s %s", label, value)

    # the scale below which the alpha_s of a decay is frozen (write_fixed_alphas_info)
    ALPHAS_FREEZE_SCALE = 1.0

    def write_fixed_alphas_info(self) -> str:
        """Write a minimal LHAPDF ``.info`` with the alpha_s of a decay.

        A decay has no beams, so there is no PDF set to take alpha_s from --
        and demanding one (possibly downloading it) just to evaluate a coupling
        would be absurd. The renormalisation scale of a decay is fixed at the
        decaying mass, so only alpha_s(M) matters. Like the Fortran decay
        matrix elements (Source/alfas_functions.f, ``nloop = 2``) it is the
        2-loop running from the alpha_s(m_Z) of the param card up to M -- not
        alpha_s(m_Z) itself, which overshoots alpha_s(125 GeV) by 4.7% (and
        h > g g, which goes like alpha_s^2, by 9.7%).

        The grid carries the running over the whole scale range and has a node
        exactly at the decaying mass, so the interpolation returns alpha_s(M)
        untouched. Below ALPHAS_FREEZE_SCALE (or wherever the running stops
        being defined, closer to the Landau pole) alpha_s is frozen at its last
        value: a light decaying particle (the muon, ...) has no QCD to speak
        of, and its widths must not fail on a coupling they do not use.
        """
        # SMINPUTS entry 3 is alpha_s(m_Z), the same value the Fortran
        # parameter setup feeds to G.
        alpha_s_mz = float(self.param_card.get_value("sminputs", 3))
        runner = Alphas_Runner(alpha_s_mz, 2, cmass=1.42, bmass=4.7)
        mass = float(self.decaying_mass)
        path = os.path.join(self.run_path, "fixed_alphas.info")
        # AlphaSGrid interpolates in log(q^2) across the grid, so give it a
        # comfortable number of nodes rather than the bare minimum of three:
        # 1 GeV to 100 TeV, and the decaying mass.
        nodes = [10 ** (i / 4.0) for i in range(0, 21)]
        q_values = sorted(
            [q for q in nodes if abs(q / mass - 1) > 1e-9] + [mass]
        )

        def running(q):
            try:
                value = runner(q)
            except (ArithmeticError, ValueError):
                return None
            return value if math.isfinite(value) and value > 0 else None

        # from the top down, freezing at the first scale where the running is
        # not used (below ALPHAS_FREEZE_SCALE) or not defined
        values = {}
        frozen = None
        for q in reversed(q_values):
            value = None
            if frozen is None and q >= self.ALPHAS_FREEZE_SCALE:
                value = running(q)
            if value is None:
                if frozen is None:
                    if not values:
                        raise ValueError(
                            f"alpha_s({q:g} GeV) cannot be computed from "
                            f"alpha_s(m_Z) = {alpha_s_mz:g}"
                        )
                    frozen = min(values)
                value = values[frozen]
            values[q] = value
        alpha_s_values = [values[q] for q in q_values]
        if frozen is not None and mass < frozen:
            logger.info(
                "decay mode: alpha_s frozen below %g GeV at alpha_s(%g GeV) = "
                "%.6g (decaying mass %g GeV)", frozen, frozen, values[frozen],
                mass,
            )
        with open(path, "w") as f:
            f.write("SetDesc: 2-loop running alpha_s of a decay (no beams)\n")
            f.write(
                "AlphaS_Qs: [%s]\n" % ", ".join(f"{q:.12g}" for q in q_values)
            )
            f.write(
                "AlphaS_Vals: [%s]\n"
                % ", ".join(f"{a:.12g}" for a in alpha_s_values)
            )
        logger.info(
            "decay mode: alpha_s(mu = %g GeV) = %.6g (2-loop running from "
            "alpha_s(m_Z) = %g, as madevent)",
            mass, values[mass], alpha_s_mz,
        )
        return path

    def needs_systematics_matrix_elements(self) -> bool:
        """The systematics will re-evaluate a matrix element per scale point.

        True when the native weights are on and some subprocess mixes alpha_s
        powers (qcd_power < 0), i.e. when build_systematics_matrix_elements()
        loads a matrix element. Conservative: it does not check that a CPU
        backend exists, so it can be True where that call ends up loading
        nothing.
        """
        if not self.systematics_enabled():
            return False
        if self.run_card["run"]["dummy_matrix_element"]:
            return False
        return any(int(meta.get("qcd_power", -1)) < 0
                   for meta in self.subprocess_data)

    def init_generator_config(self) -> None:
        run_args = self.run_card["run"]
        gen_args = self.run_card["generation"]
        vegas_args = self.run_card["vegas"]
        cfg = ms.GeneratorConfig()
        cfg.target_count = gen_args["events"]
        cfg.vegas_damping = vegas_args["damping"]
        cfg.max_overweight_truncation = gen_args["max_overweight_truncation"]
        cfg.freeze_max_weight_after = gen_args["freeze_max_weight_after"]
        cfg.start_batch_size = vegas_args["start_batch_size"]
        cfg.max_batch_size = vegas_args["max_batch_size"]
        cfg.survey_min_iters = gen_args["survey_min_iters"]
        cfg.survey_max_iters = gen_args["survey_max_iters"]
        cfg.survey_target_precision = gen_args["survey_target_precision"]
        cfg.optimization_patience = vegas_args["optimization_patience"]
        cfg.optimization_threshold = vegas_args["optimization_threshold"]
        cfg.cpu_batch_size = gen_args["cpu_batch_size"]
        cfg.gpu_batch_size = gen_args["gpu_batch_size"]
        cfg.verbosity = resolve_verbosity(run_args["verbosity"])
        cfg.combine_thread_count = run_args["combine_thread_pool_size"]
        if self.needs_systematics_matrix_elements() and cfg.combine_thread_count != 1:
            # SystematicsCalculator::compute() runs inside the combine workers,
            # and a matrix element's per-thread process instances live in a
            # ThreadResource indexed by ThreadPool::thread_index() -- the
            # *calling* thread's index. A parallel combine pool would therefore
            # need one process instance per worker; building N of them to index
            # them safely is not worth it, the more so as
            # SystematicsCalculator::matrix_elements() already serialises every
            # evaluation on its own mutex, so that parallelism was not being
            # used anyway. Run the combine stage single-threaded instead.
            logger.info(
                "systematics: the combine stage runs single-threaded because a "
                "subprocess needs its matrix element re-evaluated for the "
                "renormalisation scale variations")
            cfg.combine_thread_count = 1
        cfg.cut_efficiency_threshold = gen_args["cut_efficiency_threshold"]
        cfg.max_cut_repetitions = gen_args["max_cut_repetitions"]
        self.event_generator_config = cfg
        self.event_generator = None

    def init_subprocesses(self) -> None:
        self.backends = self.compile_matrix_elements()
        self.subprocesses = []
        if self.merged_subprocess_data is None:
            for subproc_id, meta in enumerate(self.subprocess_data):
                self.subprocesses.append(MadgraphSubprocess(self, meta, subproc_id))
        else:
            for subproc_id, meta in enumerate(self.merged_subprocess_data):
                self.subprocesses.append(
                    MadgraphSubprocess(self, meta, subproc_id, self.subprocess_data)
                )

    def compile_matrix_elements(self) -> list[str]:
        """Build the matrix-element library of every subprocess, and return the
        build backend of each requested device (in the same order).

        A 'cpu' device builds the backend named by the 'cpu_mode' entry; the
        cuda/hip devices ignore cpu_mode and build their own backend. A
        cpu_mode of 'auto' is resolved to the actual SIMD backend chosen on
        this machine.

        SubProcesses/makefile is a dispatcher over the P* directories, so a
        single 'make -j N' there builds all the subprocesses at once with one
        shared pool of N jobs: no subprocess is built with N jobs while the
        others wait, and none is limited to N/#subprocesses jobs either.
        """
        device_names = self.run_card["run"]["device"]
        if not isinstance(device_names, list):
            device_names = [device_names]
        cpu_mode = self.run_card["run"]["cpu_mode"]
        backends = [backend_of(name, cpu_mode) for name in device_names]
        # FPTYPE is part of the library names ({fptype} in me_path)
        fptype = self.fptype = fptype_of(self.run_card["run"]["precision"])
        if not self.subprocess_data:
            return backends

        first_proc_path = self.subprocess_data[0]["path"]
        subproc_path = os.path.dirname(first_proc_path)

        # Resolve cpu_mode='auto' once (the build rules pick the best SIMD
        # backend available here), so all subprocesses agree on the library names.
        auto_backend = None
        if "auto" in backends:
            auto_backend = resolve_auto_backend(first_proc_path)
            logger.info("cpu_mode 'auto' resolved as '%s'", auto_backend)
        resolved = [
            auto_backend if backend == "auto" else backend
            for backend in backends
        ]

        nb_core = self.run_card["run"]["cpu_thread_pool_size"]
        if not nb_core or nb_core < 0:
            nb_core = os.cpu_count() or 1

        log_path = os.path.join(self.run_path, "compile_subprocesses.log")
        for backend in resolved:
            # by library: the flavours of a split subprocess share one
            missing = list({
                meta["me_path"]: meta for meta in self.subprocess_data
                if not os.path.isfile(
                    meta["me_path"].format(device=backend, fptype=fptype)
                )
            }.values())
            if not missing:
                continue
            logger.info(
                f"Start compilation of SubProcesses for device '{backend}' "
                f"({len(missing)} subprocess(es), {nb_core} parallel job(s)), "
                f"see log detail in {log_path}"
            )
            start_time = time.time()
            make_args = [f"BACKEND={backend}", f"FPTYPE={fptype}", "USEBUILDDIR=1"]
            make_args += self.extra_make_args
            if (len(missing) >= PCH_MIN_SUBPROCESSES
                    and not any(arg.startswith("PCH=") for arg in make_args)):
                make_args.append("PCH=1")
            self.make_subprocesses(subproc_path, make_args, nb_core, log_path)
            logger.info(
                f"Compilation of SubProcesses done in {time.time() - start_time:.1f} s"
            )
        return resolved

    @staticmethod
    def make_subprocesses(
        subproc_path: str, args: list[str], nb_core: int, log_path: str
    ) -> None:
        """Run 'make -j nb_core' in SubProcesses/, appending the full build
        output to log_path (one run per device, so the file is appended to)."""
        command = ["make", f"-j{nb_core}"] + args
        with open(log_path, "a") as log:
            log.write(f"\n$ cd {subproc_path} && {' '.join(command)}\n")
            log.flush()
            returncode = subprocess.call(
                command, cwd=subproc_path, stdout=log, stderr=subprocess.STDOUT
            )
        if returncode != 0:
            raise RuntimeError(
                f"Compilation of the SubProcesses failed with exit status "
                f"{returncode}; see {log_path} for details"
            )

    def build_event_generator(self, phasespaces: list[PhaseSpace]) -> ms.EventGenerator:
        channel_generators = []
        for i, (subproc, phasespace) in enumerate(zip(self.subprocesses, phasespaces)):
            for integrand, channel in zip(
                subproc.build_integrands(phasespace), phasespace.channels
            ):
                if channel.event_generator is None:
                    channel.event_generator = ms.ChannelEventGenerator(
                        contexts=self.contexts,
                        integrand=integrand,
                        event_file=os.path.join(self.temp_path, f"events.{i}.{channel.name}.npy"),
                        weight_file=os.path.join(self.temp_path, f"weights.{i}.{channel.name}.npy"),
                        config=self.event_generator_config,
                        subprocess_index=i,
                        name=f"{i}.{channel.name}",
                        histograms=subproc.histograms
                    )
                channel_generators.append(channel.event_generator)

        event_generator = ms.EventGenerator(
            contexts=self.contexts,
            channels=channel_generators,
            status_file=self.status_file,
            config=self.event_generator_config,
            seed=self.run_seed,
        )
        unused_globals = (
            set(self.contexts[0].global_names()) - event_generator.used_globals()
        )
        for context in self.contexts:
            for global_name in unused_globals:
                context.delete_global(global_name)
        return event_generator

    def survey_phasespaces(
        self, phasespaces: list[PhaseSpace | None], survey_pass: int = 0
    ) -> ms.EventGenerator | None:
        ps_filtered = [ps for ps in phasespaces if ps is not None]
        if len(ps_filtered) == 0:
            return None
        event_generator = self.build_event_generator(ps_filtered)
        event_generator.survey(survey_pass)
        return event_generator

    def survey(self) -> None:
        # survey_pass distinguishes the survey() calls below: "both" mode can
        # re-survey a channel carried over unchanged from the multichannel pass
        # into the final (simplified) pass, and both passes schedule jobs on the
        # same underlying ChannelEventGenerator. The explicit pass index keeps each
        # pass's job seeds independent of the other passes' job counts, rather than
        # depending on call history.
        phasespace_mode = self.run_card["phasespace"]["mode"]
        if self.is_decay and phasespace_mode != "multichannel":
            # The flat mapping is built from a synthetic two-incoming diagram,
            # which a decay has no counterpart for -- and 'both'/'auto' end up
            # simplifying towards it. Decays have few channels, so the
            # multichannel phase space is the right one regardless.
            logger.info(
                "decay mode: using the multichannel phase space instead of "
                "'%s'", phasespace_mode
            )
            phasespace_mode = "multichannel"
        if phasespace_mode in ["multichannel", "both", "auto"]:
            self.phasespaces = [
                subproc.build_multichannel_phasespace()
                for subproc in self.subprocesses
            ]
            self.event_generator = self.survey_phasespaces(self.phasespaces, 0)
        elif phasespace_mode == "flat":
            self.phasespaces = [
                subproc.build_flat_phasespace()
                for subproc in self.subprocesses
            ]
            self.event_generator = self.survey_phasespaces(self.phasespaces, 0)
        else:
            raise ValueError("Unknown phasespace mode")

        channel_status = self.event_generator.channel_status()
        chan_offset = 0
        madnis_enabled = False
        for subproc, ps in zip(self.subprocesses, self.phasespaces):
            mean = 0.
            variance = 0.
            count_opt = 0
            for status in channel_status[chan_offset:chan_offset + len(ps.channels)]:
                mean += status.mean_abs
                variance += status.error_abs**2
                count_opt += status.count_opt
            # A subprocess whose matrix element is identically zero (an FCNC
            # channel with every Wilson coefficient at zero, say) integrates to
            # exactly zero, so there is no relative spread to speak of. Report 0
            # rather than dividing by it: that asks for the smallest networks and
            # leaves MadNIS off, which is what an empty subprocess wants.
            rsd = (variance * count_opt)**0.5 / mean if mean else 0.
            subproc.set_madnis_auto_settings(
                rsd,
                min(
                    (channel.phasespace_mapping.random_dim() for channel in ps.channels),
                    default=0,
                ),
            )
            chan_offset += len(ps.channels)
            if subproc.madnis_settings["enable"]:
                madnis_enabled = True
        if not (
            phasespace_mode == "both" or (phasespace_mode == "auto" and madnis_enabled)
        ):
            return

        phasespaces_multi = self.phasespaces
        cross_sections = []
        index = 0
        for phasespace in phasespaces_multi:
            channel_count = len(phasespace.channels)
            # rank the channels by the integral of |w|: a channel whose signed
            # integral cancels (an interference) can still carry most of the
            # variance, and folding it into the flat channel would blow it up
            cross_sections.append([
                status.mean_abs
                for status in channel_status[index:index + channel_count]
            ])
            index += channel_count

        self.phasespaces = [
            subproc.simplify_phasespace(ps_multi, cross_secs)
            for subproc, ps_multi, cross_secs in zip(
                self.subprocesses, phasespaces_multi, cross_sections
            )
        ]
        if any(
            ps_multi is not ps_both
            for ps_multi, ps_both in zip(phasespaces_multi, self.phasespaces)
        ):
            # distinct survey_pass: a channel carried over unchanged from the
            # multichannel pass (pass 0) into this resurvey must not share its
            # seed stream with that earlier pass.
            self.event_generator = self.survey_phasespaces(self.phasespaces, 1)

    def train_madnis(self) -> None:
        madnis_args = self.run_card["madnis"]
        if not any(subproc.madnis_settings["enable"] for subproc in self.subprocesses):
            return

        gen_args = self.run_card["generation"]
        run_args = self.run_card["run"]

        verbosity = resolve_verbosity(run_args["verbosity"])
        madnis_phasespaces = []
        trained_phasespaces = []
        training_args = []
        self.event_generator = None
        for subproc, phasespace in zip(self.subprocesses, self.phasespaces):
            if not subproc.madnis_supported:
                # no flow for a one-dimensional phase space: keep its VEGAS
                # channels, and their event generators, from the survey
                madnis_phasespaces.append(phasespace)
                continue
            for channel in phasespace.channels:
                channel.event_generator = None

            config = ms.MadnisConfig()
            config.learning_rate = subproc.madnis_settings["lr"]
            config.batches = subproc.madnis_settings["train_batches"]
            config.log_interval = madnis_args["log_interval"]
            config.integration_history_length = madnis_args["integration_history_length"]
            config.channel_dropping_interval = madnis_args["channel_dropping_interval"]
            config.channel_dropping_threshold = madnis_args["channel_dropping_threshold"]
            config.cpu_generator_batch_size = gen_args["cpu_batch_size"]
            config.gpu_generator_batch_size = gen_args["gpu_batch_size"]
            config.gpu_generator_batch_granularity = madnis_args["gpu_generator_batch_granularity"]
            config.generator_target_size_factor = madnis_args["generator_target_size_factor"]
            config.batch_size_offset = madnis_args["batch_size_offset"]
            config.batch_size_per_channel = subproc.madnis_settings["batch_size_per_channel"]
            config.uniform_channel_ratio = madnis_args["uniform_channel_ratio"]
            config.lr_schedule = madnis_args["lr_scheduler"]
            config.adam_beta1 = madnis_args["adam_beta1"]
            config.adam_beta2 = madnis_args["adam_beta2"]
            config.adam_eps = madnis_args["adam_eps"]
            config.adam_weight_decay = madnis_args["adam_weight_decay"]
            config.grad_clip_threshold = madnis_args["grad_clip_threshold"]
            config.buffer_capacity = madnis_args["buffer_capacity"]
            config.minimum_buffer_size = madnis_args["minimum_buffer_size"]
            config.buffered_steps_fraction = madnis_args["buffered_steps_fraction"]
            config.buffer_skip_batches = madnis_args["buffer_skip_batches"]
            config.buffer_unweighting_quantile = madnis_args["buffer_unweighting_quantile"]
            config.fixed_cwnet_fraction = subproc.madnis_settings["fixed_cwnet_fraction"]
            config.softclip_threshold = madnis_args["softclip_threshold"]
            config.compressed_channel_weight_count = madnis_args["compressed_channel_weight_count"]
            phasespace = subproc.build_madnis(phasespace)
            madnis_phasespaces.append(phasespace)
            trained_phasespaces.append(phasespace)
            training_args.append(
                ms.TrainingArgs(
                    config=config,
                    integrands=subproc.build_integrands(
                        phasespace,
                        madnis_training=True,
                        drop_cuts_and_rescale=madnis_args["drop_zero_integrands"]
                    ),
                    cwnet=phasespace.cwnet,
                )
            )

        gen_context = self.contexts[0]
        opt_context = ms.Context(
            device=self.devices[0], thread_count=self.pool_sizes[0]
        )
        opt_context.copy_globals_from(gen_context)

        madnis_training = ms.MultiMadnisTraining(
            generator_context=gen_context,
            optimizer_context=opt_context,
            training_args=training_args,
            verbosity=verbosity,
            status_file=self.status_file,
            # Reuses the run's resolved seed (also used by build_event_generator()).
            # Only the single-channel CPU sample-generation path is currently seeded
            # -- buffered training and GPU multi-channel batches are still
            # non-deterministic.
            seed=self.run_seed,
        )
        madnis_training.train()
        for phasespace, active_channels in zip(
            trained_phasespaces, madnis_training.active_channels()
        ):
            phasespace.channels = [
                phasespace.channels[index] for index in active_channels
            ]
        del madnis_training
        del opt_context
        self.phasespaces = madnis_phasespaces
        for context in self.contexts[1:]:
            context.copy_globals_from(self.contexts[0])
        self.event_generator = self.build_event_generator(madnis_phasespaces)
        self.madnis_trained = True

    def update_madnis_status_single(
        self, batch: int, batch_target: int, loss: float, lr: float, channel_count: int
    ) -> None:
        now = time.time()
        if batch + 1 < batch_target:
            if now - self.last_update_time < 0.1:
                return
            self.last_update_time = now
            progress_bar = ms.format_progress((batch + 1) / batch_target, 52)
            time_diff = now - self.madnis_wall_time
            time_str = f"{format_time(time_diff)}"
        else:
            progress_bar = ""
            wall_diff = now - self.madnis_wall_time
            cpu_diff = time.process_time() - self.madnis_cpu_time
            time_str = (
                f"{format_time(wall_diff, centi=True)} wall, "
                f"{format_time(cpu_diff, centi=True)} cpu"
            )
        batch_str = f"{batch + 1} / {batch_target}"
        self.madnis_box.set_column(1, [
            f"{batch_str:<15} {progress_bar}",
            f"{loss:>.4f}",
            f"{channel_count}",
            time_str
        ])
        self.madnis_box.print_update()

    def update_madnis_status_multi(
        self,
        subproc_id: int,
        batch: int,
        batch_target: int,
        loss: float,
        lr: float,
        channel_count: int
    ) -> None:
        now = time.time()
        subproc_count = len(self.subprocesses)
        if batch + 1 < batch_target:
            if now - self.last_update_time < 0.1:
                return
            self.last_update_time = now
            progress_bar = ms.format_progress((batch + 1) / batch_target, 34)
            progress_bar_all = ms.format_progress(
                (subproc_id * batch_target + batch + 1) / (subproc_count * batch_target),
                52
            )
            time_str = f"{format_time(now - self.madnis_wall_time)}"
            subproc_str = f"{subproc_id} / {subproc_count}"
        elif subproc_id < subproc_count - 1:
            progress_bar = ""
            progress_bar_all = ms.format_progress(
                ((subproc_id + 1) * batch_target + 1) / (subproc_count * batch_target),
                52
            )
            time_str = f"{format_time(now - self.madnis_wall_time)}"
            subproc_str = f"{subproc_id} / {subproc_count}"
        else:
            progress_bar = ""
            progress_bar_all = ""
            wall_diff = now - self.madnis_wall_time
            cpu_diff = time.process_time() - self.madnis_cpu_time
            time_str = (
                f"{format_time(wall_diff, centi=True)} wall, "
                f"{format_time(cpu_diff, centi=True)} cpu"
            )
            subproc_str = f"{subproc_count} / {subproc_count}"
        batch_str = f"{batch + 1} / {batch_target}"
        self.madnis_upper_box.set_column(1, [
            f"{subproc_str:<15} {progress_bar_all}",
            time_str,
        ])
        self.madnis_lower_box.set_row(subproc_id + 1, [
            f"{subproc_id}",
            f"{loss:>.4f}",
            f"{channel_count}",
            f"{batch_str:<15} {progress_bar}",
        ])
        self.madnis_upper_box.print_update()
        self.madnis_lower_box.print_update()

    @property
    def gridpack_run_mode(self) -> str:
        """How this run ends: a gridpack run may skip the events (see the
        [gridpack] run_mode run card option); any other run is a regular one."""
        if not self.run_card["gridpack"]["save_gridpack"]:
            return "regular"
        return self.run_card["gridpack"]["run_mode"]

    def finish_without_events(self) -> None:
        """The end of a gridpack run that writes no events: only what the
        gridpack needs is prepared, then it is stored."""
        if self.gridpack_run_mode == "minimal":
            # a trained MadNIS needs nothing more, a VEGAS grid has to converge
            if not self.madnis_trained:
                self.event_generator.optimize()
        else:
            self.event_generator.fix_max_weights()
        self.build_systematics()
        self.save_gridpack()

    def generate_events(self, output_format=None) -> None:
        """Generate the events and write them in output_format, by default the
        run_card's. run_single() passes "lhe_npy" when MadtRex reweights an
        "lhe" run: the LHE file is then made from the reweighted npy events
        (see lhe_via_binary())."""
        if self.gridpack_run_mode != "regular":
            self.finish_without_events()
            return
        start_time = get_start_time()
        self.event_generator.generate()
        if output_format is None:
            output_format = self.run_card["run"]["output_format"]
        systematics = self.build_systematics()
        histograms = self.build_event_histograms()
        if output_format == "compact_npy":
            self.lhe_completer = None
            self.events_path = os.path.join(self.run_path, "events.npy")
            self.event_generator.combine_to_compact_npy(
                self.events_path, systematics, histograms
            )
            self.write_lhe_header(systematics)
            self.save_lhe_completer()
        elif output_format == "lhe_npy":
            self.lhe_completer = self.build_lhe_completer()
            self.events_path = os.path.join(self.run_path, "events.npy")
            self.event_generator.combine_to_lhe_npy(
                self.events_path, self.lhe_completer,
                systematics, histograms
            )
            self.write_lhe_header(systematics)
        elif output_format == "lhe":
            self.lhe_completer = self.build_lhe_completer()
            lhe_path = os.path.join(self.run_path, "events.lhe")
            self.event_generator.combine_to_lhe(
                lhe_path, self.lhe_completer,
                self.build_lhe_meta(), systematics, histograms
            )
            # Ship the LHE compressed by default. These files are large and
            # very compressible, madevent has always stored its events
            # gzipped, and every consumer here already accepts either form
            # (see _find_event_file). misc.gzip replaces events.lhe with
            # events.lhe.gz, and switches to an external multithreaded tool
            # above 256 MB.
            misc.gzip(lhe_path)
            self.events_path = _find_event_file(self.run_path)
        else:
            raise ValueError("Unknown output format")
        self.write_hwu(histograms)
        self.make_plots(histograms)
        if systematics is not None:
            self.write_systematics_sidecar()
            self.log_systematics_summary()
        self.save_gridpack()

    def save_lhe_completer(self) -> None:
        """Keep the LHE completer next to compact_npy events, so npy_to_lhe
        can complete them into an LHE file later (a shower or MadSpin requested
        after the run). Never fatal to the run."""
        try:
            self.build_lhe_completer().save(
                os.path.join(self.run_path, npy_to_lhe.COMPLETER_FILE))
        except Exception as err:
            logger.warning("could not save the LHE completer; these events "
                           "cannot be converted to LHE later: %s", err)

    hwu_file_name = "MADatLO.HwU"
    plot_dir_name = "plots"

    def make_plots(self, histograms) -> None:
        """Draw the event-sample histograms into Events/<run>/plots, with the
        scale and PDF bands info.json carries, and say where they landed.

        This does not go through the HwU file: the bands are already computed
        per bin by madspace, so drawing them directly is both fewer steps and
        more than histograms.py can do without the LHAPDF python module.
        """
        if histograms is None or not self.run_card["run"]["make_plots"]:
            return
        out_dir = os.path.join(self.run_path, self.plot_dir_name)
        try:
            data = json.loads(histograms.to_json(self.systematics))
            written = plots.render(data, out_dir)
        except plots.BackendMissing as error:
            logger.info("matplotlib is not available (%s): no plots. The "
                        "distributions are in %s",
                        error, os.path.join(self.run_path, "info.json"))
            return
        except Exception as error:
            logger.warning("could not draw the histograms: %s", error)
            return
        if not written:
            return
        logger.info("%d plot(s) stored in %s", len(written), out_dir)

    def write_hwu(self, histograms) -> None:
        """Write the event-sample histograms next to the events in the HwU
        format (MADatLO.HwU), the one an aMC@NLO run writes as MADatNLO.HwU
        and madgraph/various/histograms.py reads: an mg7 distribution can then
        be overlaid on an NLO one without converting anything in between. The
        numbers are the ones info.json carries, bands included, so this is off
        by default -- it is a second copy in another format, for the runs that
        are going to be plotted."""
        if histograms is None or not self.run_card["run"]["write_hwu"]:
            return
        try:
            data = json.loads(histograms.to_json(self.systematics))
            summary = (json.loads(self.systematics.summary())
                       if self.systematics is not None else None)
            text = hwu_output.to_hwu(data, summary)
        except Exception as error:
            logger.warning("could not write the HwU histograms: %s", error)
            return
        if not text:
            return
        with open(os.path.join(self.run_path, self.hwu_file_name), "w") as f:
            f.write(text)

    def write_systematics_sidecar(self) -> None:
        """Describe the variation weights next to the event file
        (events.weights.json): ids, scale factors, PDF set/member and the
        <initrwgt> text, so npy consumers get the same metadata as the LHE
        header, plus the per-variation cross sections."""
        if self.systematics is None:
            return
        data = json.loads(self.systematics.summary())
        data["initrwgt"] = self.systematics.initrwgt()
        data["columns"] = ["rwgt_%d" % i for i in self.systematics.weight_ids]
        with open(os.path.join(self.run_path, "events.weights.json"), "w") as f:
            json.dump(data, f, indent=1)

    @staticmethod
    def _histogram_mean(hist_min, hist_max, bin_values):
        """Cross-section-weighted mean of a histogrammed observable over its
        range. ``bin_values`` carries the underflow and overflow bins first and
        last, as both madspace histogram kinds do; they have no position and
        are left out."""
        values = list(bin_values)[1:-1]
        total = sum(values)
        if not values or total == 0:
            return None
        width = (hist_max - hist_min) / len(values)
        return sum(v * (hist_min + (i + 0.5) * width)
                   for i, v in enumerate(values)) / total

    def _histogram_means(self) -> dict:
        """name -> mean of every [histograms] entry, from the post-processing
        histograms (the final event sample) when they were filled, otherwise
        from the weighted integration histograms."""
        if self.event_histograms is not None:
            data = json.loads(self.event_histograms.to_json(self.systematics))
            hists = [(h["name"], h["min"], h["max"], h["bin_values"])
                     for h in data]
        else:
            hists = [(h.name, h.min, h.max, h.bin_values)
                     for h in self.event_generator.histograms()]
        means = {}
        for name, hist_min, hist_max, values in hists:
            mean = self._histogram_mean(hist_min, hist_max, values)
            if mean is not None:
                means[name] = mean
        return means

    def get_result(self) -> dict:
        """Return the run result: cross-section (pb) with MC error, the number
        of (unweighted) events, and the mean of every observable declared in the
        [histograms] section. Used to build the scan summary."""
        status = self.event_generator.status()
        result = {'cross(pb)': status.mean, 'error(pb)': status.error,
                  'nb_event': status.count_unweighted}
        try:
            if self.systematics is not None and self.systematics.weight_count:
                summary = json.loads(self.systematics.summary())
                scale = systematics_summary.scale_percentages(summary)
                if scale is not None:
                    result['scale_up(%)'], result['scale_down(%)'] = scale
                pdf = systematics_summary.pdf_percentages(summary)
                if pdf:
                    # one column pair; the nominal set's entry comes first
                    _entry, up, down = pdf[0]
                    result['pdf_up(%)'], result['pdf_down(%)'] = up, down
        except Exception as err:
            logger.warning("could not extract the systematics summary: %s", err)
        try:
            for name, mean in self._histogram_means().items():
                result['<%s>' % name] = mean
        except Exception as err:
            logger.warning("could not extract observable means: %s", err)
        return result

    def _beam_info(self):
        """Return (beam_pdg_ids, beam_energies) for the LHE <init> block.

        `incoming` holds the *partonic* initial state (e.g. gluons), not the
        beam particle, so hadronic beams are protons (2212); leptonic beams are
        the incoming leptons themselves."""
        if self.is_decay:
            # No beams. LHE has no way to say that, so report the decaying
            # particle at rest as a single "beam"; the second slot is empty.
            data = self.subprocess_data[0]
            pdg = clean_pids(data["incoming"])[0]
            return [pdg, 0], [float(self.e_cm), 0.0]
        if not self.leptonic:
            # hadronic collider: proton beams (p-pbar is not distinguished)
            return [2212, 2212], list(self.beam_energies)
        data = self.subprocess_data[0]
        incoming = data["incoming"]
        flavor0 = data["flavors"][0]["options"][0]
        init_pdgs = flavor0[:len(incoming)]
        beam_pdgs = [pdg if abs(code) in (81, 82) else code
                     for code, pdg in zip(incoming, init_pdgs)]
        return beam_pdgs, list(self.beam_energies)

    def _lhapdf_id(self, beam=1):
        """Central LHAPDF id of the PDF set of `beam` (read from its .info
        SetIndex), or -1 for a leptonic beam (no PDF)."""
        if self.leptonic or not self.pdf_dir:
            return -1
        pdf_set = self.pdf_set if beam == 1 else self.pdf_set2
        pdf_dir = self.pdf_dir if beam == 1 else self.pdf_dir2
        info = os.path.join(pdf_dir, pdf_set, "%s.info" % pdf_set)
        try:
            for line in open(info):
                if line.strip().startswith("SetIndex:"):
                    return int(line.split(":", 1)[1].strip())
        except Exception as err:
            logger.warning("could not read LHAPDF id from %s: %s", info, err)
        return -1

    def build_lhe_meta(self, systematics=None):
        """Build the LHE header/<init> metadata: the param_card (<slha>) and the
        run_card.toml (<MG7RunCard>) headers plus the beam/PDF/cross-section info
        needed by downstream tools (systematics, MadSpin, ...).

        `combine_to_lhe` injects the <initrwgt> header itself, so callers that
        go through it should leave `systematics` unset; pass it only when
        building meta for a writer that bypasses that injection (e.g.
        write_lhe_header)."""
        beam_pdgs, energies = self._beam_info()
        pdf_group = -1 if self.leptonic else 0
        status = self.event_generator.status()
        xsec, err = status.mean, status.error
        weight_mode, max_weight = lhe_weight_info(status)
        with open(self.param_card_path) as f:
            param_text = f.read()
        with open(os.path.join("Cards", "run_card.toml")) as f:
            run_text = f.read()
        headers = []
        # generation commands (model + process): read by MadSpin/reweight/...
        proc_card = os.path.join("Cards", "proc_card_mg5.dat")
        if os.path.exists(proc_card):
            with open(proc_card) as f:
                headers.append(ms.LHEHeader(name="MG5ProcCard", content=f.read()))
        headers.append(ms.LHEHeader(name="slha", content=param_text))
        headers.append(ms.LHEHeader(name="MG7RunCard", content=run_text))
        # The resolved seed (even when the run_card requested a random one via
        # seed = -1), so the run can be reproduced from the LHE file alone.
        headers.append(ms.LHEHeader(name="MG7Seed", content=str(self.run_seed)))
        if systematics is not None and systematics.weight_ids:
            headers.append(ms.LHEHeader(
                name="initrwgt", content=systematics.initrwgt(), escape_content=False
            ))
        return ms.LHEMeta(
            beam1_pdg_id=beam_pdgs[0], beam2_pdg_id=beam_pdgs[1],
            beam1_energy=energies[0], beam2_energy=energies[1],
            beam1_pdf_authors=pdf_group, beam2_pdf_authors=pdf_group,
            beam1_pdf_id=self._lhapdf_id(1), beam2_pdf_id=self._lhapdf_id(2),
            weight_mode=weight_mode,
            # positional: the pybind arg name for max_weight is non-kwarg-safe
            processes=[ms.LHEProcess(xsec, err, max_weight, 1)],
            headers=headers,
        )

    def write_lhe_header(self, systematics) -> None:
        """Write header.lhe next to events.npy: the <header>/<init> blocks
        (run card, param card, beam/PDF, cross section) that the npy formats
        otherwise drop, with no events."""
        header_path = os.path.join(self.run_path, "header.lhe")
        writer = ms.LHEFileWriter(header_path, self.build_lhe_meta(systematics))
        del writer  # closes the file (writes the closing tag)

    def build_lhe_completer(self):
        all_mcdata = (
            [subproc.build_multi_channel_data() for subproc in self.subprocesses]
            if self.merged_subprocess_data is None else
            [build_multi_channel_data(meta, self) for meta in self.subprocess_data]
        )
        subproc_args = [
            ms.SubprocArgs(
                topologies = [topo[0] for topo in mcdata.topologies],
                permutations = mcdata.permutations,
                diagram_indices = mcdata.diagram_indices,
                diagram_color_indices = mcdata.diagram_color_indices,
                diagram_propagator_pdgs = mcdata.diagram_propagator_pdgs,
                color_flows = meta["color_flows"],
                pdg_color_types = {
                    int(key): value
                    for key, value in meta["pdg_color_types"].items()
                },
                helicities = lhe_helicities(meta, self.run_card),
                pdg_ids = [flavor["options"] for flavor in meta["flavors"]],
            )
            for mcdata, meta in zip(all_mcdata, self.subprocess_data)
        ]
        return ms.LHECompleter(
            subproc_args=subproc_args,
            bw_cutoff=self.run_card["phasespace"]["bw_cutoff"]
        )

    def save_gridpack(self) -> None:
        if not self.run_card["gridpack"]["save_gridpack"]:
            return

        gridpack_path = os.path.join(self.run_path, "gridpack")
        data_path = os.path.join(gridpack_path, "data")
        events_path = os.path.join(gridpack_path, "Events")
        os.mkdir(gridpack_path)
        os.mkdir(data_path)
        os.mkdir(events_path)
        self.contexts[0].save_globals(os.path.join(data_path, "globals.tar"))

        include_estimates = self.gridpack_run_mode == "fix_max_weight"
        # one json object per channel, joined by hand so that the channel
        # json is not parsed and dumped again
        channel_entries = [
            f"{json.dumps(channel.status().name)}:{channel.to_json(include_estimates)}"
            for channel in self.event_generator.channels()
        ]
        with open(os.path.join(data_path, "channels.json"), "w") as f:
            f.write("{" + ",".join(channel_entries) + "}")

        lib_path = os.path.join(gridpack_path, "lib")
        if self.run_card["gridpack"]["include_source"]:
            os.mkdir(lib_path)
            shutil.copytree("src", os.path.join(gridpack_path, "src"))
            shutil.copytree("SubProcesses", os.path.join(gridpack_path, "SubProcesses"))
            shutil.copytree("backend", os.path.join(gridpack_path, "backend"))
        else:
            shutil.copytree("lib", lib_path)
        # the gridpack refers to its own copies, by paths relative to it
        gridpack_custom_paths = []
        for kind, path, parameters in self.custom_function_paths:
            rel_path = os.path.normpath(path)
            if os.path.isabs(rel_path) or rel_path.startswith(".."):
                rel_path = os.path.join("custom_functions", kind, os.path.basename(path))
            target = os.path.join(gridpack_path, rel_path)
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy(path, target)
            gridpack_custom_paths.append([kind, rel_path, parameters])

        if self.run_card["gridpack"]["include_madspace_source"]:
            shutil.copytree(
                _MADSPACE_DIR,
                os.path.join(gridpack_path, "madspace"),
                ignore=shutil.ignore_patterns("build", "install"),
            )

        if self.run_card["gridpack"]["include_madspace"]:
            shutil.copytree(
                _INSTALL_DIR / "madspace",
                os.path.join(gridpack_path, "madspace", "install", "madspace"),
                ignore=shutil.ignore_patterns("include"),
            )

        # MadtRex (event reweighting) is optional and installed separately on
        # top of madspace (madtrex/install.py), so a missing install is
        # skipped rather than an error.
        if self.run_card["gridpack"]["include_madtrex_source"]:
            shutil.copytree(
                _MADTREX_DIR,
                os.path.join(gridpack_path, "madtrex"),
                ignore=shutil.ignore_patterns("build", "install", "__pycache__"),
            )

        # madspace/madtrex sources build with their pyproject.toml, which takes
        # the version from the MadGraph VERSION file one level above them
        if (self.run_card["gridpack"]["include_madspace_source"]
                or self.run_card["gridpack"]["include_madtrex_source"]):
            shutil.copy(_MG_ROOT / "VERSION", os.path.join(gridpack_path, "VERSION"))

        include_madtrex = (
            self.run_card["gridpack"]["include_madtrex"]
            and (_MADTREX_INSTALL_DIR / "madtrex").is_dir()
        )
        if include_madtrex:
            # only the runtime pieces: madtrex itself, the bundled Rex fork
            # (pyrex, plus librex/libtearex in lib/), not the build tools
            # install.py also put there
            for name in ("madtrex", "pyrex", "lib"):
                if (_MADTREX_INSTALL_DIR / name).is_dir():
                    shutil.copytree(
                        _MADTREX_INSTALL_DIR / name,
                        os.path.join(gridpack_path, "madtrex", "install", name),
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
        elif self.run_card["gridpack"]["include_madtrex"]:
            logger.info(
                "madtrex is not installed (python madtrex/install.py), so it is "
                "not included in the gridpack%s",
                "; bin/reweight builds it from the included sources on first use"
                if self.run_card["gridpack"]["include_madtrex_source"] else "")
        # bin/reweight (below) with an installed madtrex, or with its sources,
        # which bin/reweight builds when the gridpack has no (usable) install
        with_reweight = include_madtrex or self.run_card["gridpack"]["include_madtrex_source"]
        if with_reweight:
            # the standalone reweighting driver finds the matrix elements
            # through subprocesses.json, which the gridpack otherwise only has
            # with include_source
            gridpack_subprocesses_json = os.path.join(
                gridpack_path, "SubProcesses", "subprocesses.json")
            if not os.path.exists(gridpack_subprocesses_json):
                os.makedirs(os.path.dirname(gridpack_subprocesses_json), exist_ok=True)
                shutil.copy(os.path.join("SubProcesses", "subprocesses.json"),
                            gridpack_subprocesses_json)

        matrix_elements = []
        for subproc in self.subprocess_data:
            me_path = subproc["me_path"]
            matrix_elements.append(me_path)

        cards_path = os.path.join(gridpack_path, "Cards")
        os.mkdir(cards_path)
        shutil.copy(os.path.join("Cards", "param_card.dat"), cards_path)
        # Full run card with a header noting it is read-only in gridpack context.
        import io as _io
        _buf = _io.StringIO()
        self.run_card.write(_buf)
        _header = (
            "# This is the run card used to generate this gridpack.\n"
            "# Modifying this file will have no effect on gridpack execution.\n"
            "# To change event-generation settings, edit grid_run_card.toml.\n\n"
        )
        with open(os.path.join(cards_path, "run_card.toml"), 'w') as _f:
            _f.write(_header + _buf.getvalue())
        # Minimal card containing only the settings used by generate_events.
        # The gridpack ships the libraries that were actually built, and their
        # names carry the resolved backend, so the card it runs from has to name
        # that backend as well: a cpu_mode of 'auto' would send the gridpack
        # looking for a ..._auto.so that was never built. The full run_card.toml
        # written just above keeps 'auto', since that is what was asked for.
        device_names = self.run_card["run"]["device"]
        if not isinstance(device_names, list):
            device_names = [device_names]
        resolved_cpu_mode = None
        for name, backend in zip(device_names, getattr(self, "backends", [])):
            if name.split(":")[0] == "cpu":
                resolved_cpu_mode = backend
                break
        previous_cpu_mode = self.run_card["run"]["cpu_mode"]
        if resolved_cpu_mode is not None:
            self.run_card["run"]["cpu_mode"] = resolved_cpu_mode
        # A gridpack is run to feed a shower or detector chain, which reads
        # LHE, so it writes LHE by default whatever format this run used
        # (bin/generate_events --output_format still selects an npy output).
        previous_output_format = self.run_card["run"]["output_format"]
        self.run_card["run"]["output_format"] = "lhe"
        # the directories of this run mean nothing on the machine running the gridpack
        previous_dirs = (self.run_card["gridpack"]["temp_output_dir"],
                         self.run_card["gridpack"]["output_dir"])
        self.run_card["gridpack"]["temp_output_dir"] = ""
        self.run_card["gridpack"]["output_dir"] = ""
        try:
            self.run_card.write_gridpack_card(
                os.path.join(cards_path, "grid_run_card.toml"))
        finally:
            self.run_card["run"]["cpu_mode"] = previous_cpu_mode
            self.run_card["run"]["output_format"] = previous_output_format
            (self.run_card["gridpack"]["temp_output_dir"],
             self.run_card["gridpack"]["output_dir"]) = previous_dirs

        bin_path = os.path.join(gridpack_path, "bin")
        os.mkdir(bin_path)
        gen_events_file = os.path.join(bin_path, "generate_events")
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "gridpack.py"), gen_events_file
        )
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "custom_function_loader.py"),
            os.path.join(bin_path, "custom_function_loader.py"),
        )
        os.chmod(gen_events_file, 0o755)
        if with_reweight:
            # MadtRex reweighting of the gridpack's events, with the modules
            # parsing its reweight cards
            reweight_file = os.path.join(bin_path, "reweight")
            shutil.copy(
                os.path.join(os.path.dirname(__file__), "madtrex_driver.py"), reweight_file
            )
            os.chmod(reweight_file, 0o755)
            for name in ("param_reweight_card.py", "pdf_reweight_card.py",
                         "generic_reweight_card.py"):
                shutil.copy(os.path.join(os.path.dirname(__file__), name),
                            os.path.join(bin_path, name))

        data = {
            "matrix_elements": matrix_elements,
            # loaded before the matrix elements, as in the run
            "custom_functions": gridpack_custom_paths,
            "me_parameters": me_parameters(self.run_card),
            "source_hash": ms.SOURCE_HASH,
        }
        with open(os.path.join(data_path, "data.json"), "w") as f:
            json.dump(data, f)

        if self.lhe_completer is None:
            self.lhe_completer = self.build_lhe_completer()
        self.lhe_completer.save(os.path.join(data_path, "lhe.json"))
        self.save_gridpack_lhe_meta(os.path.join(data_path, "lhe_meta.json"))
        if self.systematics_data is not None:
            systematics_data = dict(self.systematics_data)
            # A decay takes alpha_s from a constant grid written into the run
            # directory rather than from an LHAPDF set, so the gridpack cannot
            # look that file up: ship it and point at the shipped copy.
            if self.fixed_alphas_info and \
                    systematics_data["nominal_info_file"] == self.fixed_alphas_info:
                name = os.path.basename(self.fixed_alphas_info)
                shutil.copy(self.fixed_alphas_info, os.path.join(data_path, name))
                systematics_data["nominal_info_file"] = os.path.join("data", name)
            with open(os.path.join(data_path, "systematics.json"), "w") as f:
                json.dump(systematics_data, f)
        if self.run_card["gridpack"]["compress"]:
            archive_path = gridpack_path + ".tar.gz"
            use_pigz = self.run_card["run"]["use_pigz"]
            pigz = (shutil.which("pigz") if use_pigz is not False else None)
            if pigz:
                # multi-threaded gzip
                with open(archive_path, "wb") as f:
                    tar = subprocess.Popen(
                        ["tar", "-cf", "-", "-C", self.run_path, "gridpack"],
                        stdout=subprocess.PIPE,
                    )
                    gz_command = [pigz, "-c"]
                    cpu_threads = self.run_card["run"]["cpu_thread_pool_size"]
                    if cpu_threads > 0:
                        gz_command += ["-p", str(cpu_threads)]
                    gz = subprocess.run(gz_command, stdin=tar.stdout, stdout=f)
                    tar.stdout.close()
                    if tar.wait() != 0 or gz.returncode != 0:
                        raise RuntimeError("compressing the gridpack failed")
            else:
                shutil.make_archive(gridpack_path, "gztar", self.run_path, "gridpack")
            shutil.rmtree(gridpack_path)

    def save_gridpack_lhe_meta(self, path) -> None:
        """The run-independent part of the LHE header (cards, beams, PDF), for
        the gridpack to write complete LHE headers: it adds the cross section
        and seed of its own run, which is why those two are left out here."""
        meta = self.build_lhe_meta()
        data = {key: getattr(meta, key) for key in (
            "beam1_pdg_id", "beam2_pdg_id", "beam1_energy", "beam2_energy",
            "beam1_pdf_authors", "beam2_pdf_authors", "beam1_pdf_id",
            "beam2_pdf_id", "weight_mode")}
        data["headers"] = [
            {"name": h.name, "content": h.content,
             "escape_content": h.escape_content}
            for h in meta.headers if h.name != "MG7Seed"]
        with open(path, "w") as f:
            json.dump(data, f)

    def get_mass(self, pid: int) -> float:
        return self.param_card.get_value("mass", pid)

    def get_width(self, pid: int) -> float:
        return self.param_card.get_value("width", pid)


# Flavor-merged legs carry a group id instead of a pdg. Every member of a group
# shares the same mass -- that is what makes them mergeable -- so any member is
# a valid representative for the kinematics.
_MERGED_PID_REPRESENTATIVE = {
    81: 1,   # light quarks   d u s c
    82: 11,  # charged leptons e mu
    83: 12,  # neutrinos      ve vm vt
}


def clean_pids(pids: list[int]) -> list[int]:
    pids_out = []
    for pid in pids:
        pid = abs(pid)
        if pid in _MERGED_PID_REPRESENTATIVE:
            pid = _MERGED_PID_REPRESENTATIVE[pid]
        elif 81 <= pid <= 99:
            # Reserved for flavor merging. Anything outside this window is a
            # real pdg (BSM models use codes in the millions), so let it pass.
            raise ValueError(
                f"unknown flavor-merged particle id {pid}; add its "
                "representative to _MERGED_PID_REPRESENTATIVE"
            )
        pids_out.append(pid)
    return pids_out


# observables that tell a particle from its antiparticle, and so need the signed
# pdg ids of the actual flavours rather than clean_pids' unsigned representatives
_SIGNED_OBSERVABLES = {"sfos_pair_mass"}


def _flavor_options(meta: dict, unmerged_meta: dict | None = None) -> list[list[int]]:
    """Every flavour assignment (signed pdg ids, incoming then outgoing) a
    subprocess evaluates with the same momenta."""
    if "flavors" in meta:
        return [
            option for flavor in meta["flavors"] for option in flavor["options"]
        ]
    return [
        option
        for subproc in meta.get("subprocesses", [])
        for option in _flavor_options(unmerged_meta[subproc])
    ]


def observable_pids(
    meta: dict, observable_kwargs: dict, unmerged_meta: dict | None = None
) -> list[int]:
    """The pdg ids an observable of this subprocess is built with.

    Most observables only select particles by group, which the unsigned
    representatives of clean_pids do. One that pairs a particle with its
    antiparticle needs the real signed flavours: those of the first flavour
    assignment, provided every assignment of the subprocess forms the same
    pairs -- the cut is applied once to momenta that all of them share.
    """
    all_pids = clean_pids(meta["incoming"]) + clean_pids(meta["outgoing"])
    if observable_kwargs.get("observable") not in _SIGNED_OBSERVABLES:
        return all_pids
    options = _flavor_options(meta, unmerged_meta)
    if not options:
        raise ValueError(
            f"{observable_kwargs.get('name')}: no flavour information for "
            f"subprocess {meta['incoming']} > {meta['outgoing']}"
        )
    n_in = len(meta["incoming"])
    groups = [set(group) for group in observable_kwargs["select_pids"]]
    if len(groups) == 1:
        groups = groups * 2

    def sfos_pairs(pids):
        out = range(n_in, len(pids))
        return frozenset(
            frozenset((i, j))
            for i in out if pids[i] in groups[0]
            for j in out if pids[j] in groups[1]
            if i != j and pids[i] != 0 and pids[i] == -pids[j]
        )

    reference = sfos_pairs(options[0])
    if any(sfos_pairs(option) != reference for option in options[1:]):
        raise ValueError(
            f"{observable_kwargs.get('name')}: the flavours grouped into subprocess "
            f"{meta['incoming']} > {meta['outgoing']} pair the particles "
            "differently, so a same-flavour opposite-sign cut cannot be applied "
            "to their common momenta; regenerate the process after "
            "'set apply_flavor_grouping False'"
        )
    return list(options[0])


def build_observable(pids: list[int], n_in: int, observable_kwargs: dict):
    """ms.Observable of a subprocess with `n_in` incoming particles.

    madspace takes the first two momenta to be the beams (ignore_incoming), so
    for a 1 -> N decay it would drop the first decay product from every
    selection. There the selection runs over all the momenta instead, with the
    incoming particle set to pdg id 0, which is in no group.
    """
    if n_in == 2:
        return ms.Observable(pids, **observable_kwargs)
    pids = [0] * n_in + list(pids[n_in:])
    return ms.Observable(pids, **dict(observable_kwargs, ignore_incoming=False))


def decay_products(meta: dict, unmerged_meta: dict | None = None) -> set[int]:
    """Outgoing positions (0-based, beams excluded) of the particles that come
    from an on-shell (decay-chain) propagator, as MadEvent's check_decay marks
    them. Every channel of a subprocess shares its decay chains, so the first
    channel tells, read through its first diagram's permutation."""
    if not meta["channels"]:
        return set()
    channel = meta["channels"][0]
    topo_channel = channel
    if "vertices" not in channel:
        topo_channel = unmerged_meta[channel["subprocess"]]["channels"][channel["channel"]]
    on_shell = topo_channel.get("on_shell_propagators", [])
    if not on_shell:
        return set()
    vertices = topo_channel["vertices"]
    # the closing vertex creates no propagator
    created = {vertex[-1]: vertex[:-1] for vertex in vertices[:-1]}

    def legs(name):
        if name in created:
            return set().union(*(legs(child) for child in created[name]))
        return {name}

    n_in = len(meta["incoming"])
    n_out = len(meta["outgoing"])
    products = set()
    for prop in on_shell:
        external = legs(f"p{prop}")
        incoming = {leg for leg in external if leg.startswith("i")}
        outgoing = {int(leg[1:]) for leg in external if leg.startswith("o")}
        if not incoming:
            products |= outgoing
        elif len(incoming) == n_in:
            # built from the beams: the decay is everything else
            products |= set(range(n_out)) - outgoing
    # event[i] = topology[permutation[i]]
    permutation = channel["diagrams"][0]["permutation"]
    return {
        i - n_in for i in range(n_in, len(permutation))
        if permutation[i] - n_in in products
    }


def pid_is_qcd(pid: int):
    return abs(pid) in [21, 1, 2, 3, 4, 5, 6, 81]


def build_topologies(
    incoming_masses: list[float],
    outgoing_masses: list[float],
    channel: dict,
    process: MadgraphProcess
) -> list[ms.Topology]:
    propagators = []
    for i, (pid, signed_pid) in enumerate(zip(
        clean_pids(channel["propagators"]), channel["propagators"]
    )):
        mass = process.get_mass(pid)
        width = process.get_width(pid)
        bw_cutoff = process.run_card["phasespace"]["bw_cutoff"]
        if i in channel["on_shell_propagators"]:
            e_min = mass - bw_cutoff * width
            e_max = mass + bw_cutoff * width
        else:
            e_min = 0
            e_max = 0
        # $-excluded propagator: the matrix element vanishes within bw_cutoff
        # widths of the pole, so do not pile the channel's points up there
        # (a flat density in that window instead of the Breit-Wigner peak).
        # Channels written before the exporter recorded them have none.
        flat_window = 0.
        if i in channel.get("dollar_propagators", ()) and width > 0:
            flat_window = float(bw_cutoff)
        propagators.append(ms.Propagator(
            mass=mass,
            width=width,
            integration_order=0,
            e_min=e_min,
            e_max=e_max,
            pdg_id=signed_pid,
            flat_window=flat_window,
        ))
    vertices = channel["vertices"]
    diag = ms.Diagram(
        incoming_masses, outgoing_masses, propagators, vertices
    )
    return ms.Topology.topologies(diag)


def build_multi_channel_data(
    meta: dict, process: MadgraphProcess, unmerged_meta: dict | None = None
) -> MultiChannelData:
    incoming_masses = [
        process.get_mass(pid) for pid in clean_pids(meta["incoming"])
    ]
    outgoing_masses = [
        process.get_mass(pid) for pid in clean_pids(meta["outgoing"])
    ]

    if unmerged_meta is None:
        diagram_count = meta["diagram_count"]
        amp2_remaps = [[-1] * diagram_count]
    else:
        amp2_remaps = [
            [-1] * unmerged_meta[subproc]["diagram_count"]
            for subproc in meta["subprocesses"]
        ]
    symfact = []
    topologies = []
    permutations = []
    channel_indices = []
    channel_weight_indices = []
    diagram_indices = []
    diagram_color_indices = []
    diagram_propagator_pdgs = []
    active_flavors = []
    channel_index = 0
    qcd_s_channel_count = []

    for channel in meta["channels"]:
        if unmerged_meta is None:
            topo_channel = channel
        else:
            topo_subproc = channel["subprocess"]
            topo_channel_index = channel["channel"]
            topo_channel = unmerged_meta[topo_subproc]["channels"][topo_channel_index]
        chan_topologies = build_topologies(
            incoming_masses, outgoing_masses, topo_channel, process
        )
        topo_count = len(chan_topologies)
        if topo_count == 0:
            continue

        topo = chan_topologies[0]
        s_chan_count = [0] * len(topo.decays)
        non_qcd = [False] * len(topo.decays)
        pdg_ids = [decay.pdg_id for decay in topo.decays]
        for i, pid in zip(topo.outgoing_indices, meta["outgoing"]):
            pdg_ids[i] = pid
        for decay in reversed(topo.decays):
            if len(decay.child_indices) == 0:
                continue
            if decay.index == 0 and topo.t_propagator_count > 0:
                s_chan_count[0] = sum(s_chan_count[i] for i in decay.child_indices)
                break

            is_qcd = pid_is_qcd(pdg_ids[decay.index]) and all(
                pid_is_qcd(pdg_ids[i]) for i in decay.child_indices
            )
            is_non_qcd = decay.on_shell or not is_qcd or any(
                non_qcd[i] for i in decay.child_indices
            )
            non_qcd[decay.index] = is_non_qcd
            s_chan_count[decay.index] = (
                0 if is_non_qcd else sum(s_chan_count[i] for i in decay.child_indices) + 1
            )
        qcd_s_channel_count.append(s_chan_count[0])

        diagrams = channel["diagrams"]
        chan_permutations = [d["permutation"] for d in diagrams]
        if unmerged_meta is None:
            amp2_remaps[0][diagrams[0]["diagram"]] = channel_index
        else:
            for amp2_remap, diag in zip(amp2_remaps, diagrams[0]["diagram"]):
                if diag != -1:
                    amp2_remap[diag] = channel_index

        channel_index_first = channel_index
        symfact_index_first = len(symfact)
        channel_index += 1
        symfact.extend([None] * topo_count)
        for d in diagrams[1:]:
            if unmerged_meta is None:
                amp2_remaps[0][d["diagram"]] = channel_index
            else:
                for amp2_remap, diag in zip(amp2_remaps, d["diagram"]):
                    if diag != -1:
                        amp2_remap[diag] = channel_index
            channel_index += 1
            symfact.extend(range(symfact_index_first, symfact_index_first + topo_count))

        topologies.append(chan_topologies)
        permutations.append(chan_permutations)
        channel_indices.append(list(range(channel_index_first, channel_index)))
        channel_weight_indices.append([
            [
                symfact_index_first + topo_index + i * topo_count
                for i in range(len(chan_permutations))
            ]
            for topo_index in range(topo_count)
        ])
        diagram_indices.append([d["diagram"] for d in diagrams])
        if unmerged_meta is None:
            diagram_color_indices.append([d["active_colors"] for d in diagrams])
            diagram_propagator_pdgs.append(
                [d["propagator_pdgs"] for d in diagrams]
            )
        active_flavors.append([d["active_flavors"] for d in diagrams])

    return MultiChannelData(
        amp2_remaps,
        symfact,
        topologies,
        permutations,
        channel_indices,
        channel_weight_indices,
        diagram_indices,
        diagram_color_indices,
        diagram_propagator_pdgs,
        active_flavors,
        qcd_s_channel_count,
    )


class MadgraphSubprocess:
    def __init__(
        self,
        process: MadgraphProcess,
        meta: dict,
        subproc_id: int,
        unmerged_meta: dict | None = None
    ):
        self.process = process
        self.meta = meta
        self.subproc_id = subproc_id
        # set from the survey by set_madnis_auto_settings
        self.madnis_supported = True
        self.multi_channel_data = None

        self.unmerged_meta = None
        if unmerged_meta is None:
            api_path_formats = [self.meta["me_path"]]
        else:
            api_path_formats = []
            for subproc in self.meta["subprocesses"]:
                submeta = unmerged_meta[subproc]
                api_path_formats.append(submeta["me_path"])
            if len(api_path_formats) == 1:
                self.meta = submeta
            else:
                self.unmerged_meta = unmerged_meta

        # The libraries were all built up front by MadgraphProcess.compile_matrix_elements
        all_api_paths = [
            [
                api_path_format.format(device=backend, fptype=self.process.fptype)
                for backend in self.process.backends
            ]
            for api_path_format in api_path_formats
        ]

        self.incoming_masses = [
            self.process.get_mass(pid) for pid in clean_pids(self.meta["incoming"])
        ]
        self.outgoing_masses = [
            self.process.get_mass(pid) for pid in clean_pids(self.meta["outgoing"])
        ]
        self.particle_count = len(self.incoming_masses) + len(self.outgoing_masses)
        # MadEvent's cut_decays = F: the decay products of on-shell propagators
        # are left out of every cut selection (a pair cut then needs both
        # members to be cut), while the histograms still see them
        if self.process.run_card["phasespace"]["cut_decays"]:
            uncut = set()
        else:
            uncut = decay_products(self.meta, self.unmerged_meta)
        n_in = len(self.incoming_masses)

        def cut_pids(observable_kwargs):
            pids = observable_pids(self.meta, observable_kwargs, self.unmerged_meta)
            # pdg id 0 is in no group, so the particle is never selected
            return [0 if i - n_in in uncut else pid for i, pid in enumerate(pids)]

        # custom cuts and scales see the PDG ids of the first flavor, like the
        # built-in cuts, which select by flavor-independent groups
        representative_pids = [self.flavor_pids()[0]]
        custom_cut = self.process.custom_function(
            "cut", self.particle_count, representative_pids
        )
        if len(self.process.cut_data) > 0:
            self.cuts = ms.Cuts([
                ms.CutItem(
                    observable=build_observable(
                        cut_pids(cut_item.observable_kwargs),
                        n_in, cut_item.observable_kwargs
                    ),
                    min=cut_item.min,
                    max=cut_item.max,
                    mode=cut_item.mode,
                )
                for cut_item in self.process.cut_data
            ], custom_cut)
        elif custom_cut is not None:
            self.cuts = ms.Cuts(self.particle_count, custom_cut)
        else:
            self.cuts = None
        # A flavor standing for both beam orientations draws the orientation
        # before the phase-space mapping, so boost, cuts and PDFs all see it.
        self.mirror_beams = not self.process.is_decay and self.has_mirrored_flavors()
        # the integration histograms are functions of the momenta; the weight
        # distribution is a property of the final event sample, so it is only
        # filled by MadgraphProcess.build_event_histograms at combine time
        # They cost an observable evaluation per integration point, and the
        # plots/HwU are drawn from the post-processing ones instead, so they
        # are only filled on request ([run] weighted_histograms).
        momentum_hists = [item for item in self.process.hist_data
                          if not item.from_weight] \
            if self.process.run_card["run"]["weighted_histograms"] else []
        self.histograms = (
            ms.ObservableHistograms([
                ms.HistItem(
                    observable=build_observable(
                        observable_pids(
                            self.meta, hist_item.observable_kwargs, self.unmerged_meta
                        ),
                        n_in, hist_item.observable_kwargs
                    ),
                    min=hist_item.min,
                    max=hist_item.max,
                    bin_count=hist_item.bin_count,
                )
                for hist_item in momentum_hists
            ])
            if momentum_hists
            else None
        )

        # The scale range exists to keep the densities inside the PDF grid, so
        # a run without beam densities - a lepton collider or a decay - has
        # none: a floor there would only throw away events whose scale is
        # legitimately low, a decay of a particle lighter than min_scale say.
        if self.process.leptonic:
            min_scale = max_scale = 0.0
        else:
            min_scale = self.process.run_card["beam"]["min_scale"]
            max_scale = self.process.max_scale
        if self.process.mlm_clustering:
            mc_data = self.build_multi_channel_data()
            all_pids = (clean_pids(self.meta["incoming"])
                        + clean_pids(self.meta["outgoing"]))
            self.scale = ms.EnergyScale(
                ms.MLMClustering(
                    topologies=[topo[0] for topo in mc_data.topologies],
                    permutations=mc_data.permutations,
                    diagram_indices=mc_data.diagram_indices,
                    cm_energy=self.process.e_cm,
                    jet_scale_scheme=getattr(
                        ms.MLMClustering.JetScaleScheme,
                        self.process.run_card["beam"]["jet_scale_scheme"],
                    ),
                    scale_scheme=getattr(
                        ms.MLMClustering.ScaleScheme,
                        self.process.run_card["beam"]["scale_scheme"],
                    ),
                    pdg_color_types={
                        int(key): value
                        for key, value in self.meta["pdg_color_types"].items()
                    },
                    xqcut=self.process.run_card["phasespace"]["xqcut"],
                    bw_cutoff=self.process.run_card["phasespace"]["bw_cutoff"],
                    jet_radius=self.process.run_card["beam"]["jet_radius"],
                    hadronic=not self.process.leptonic,
                    external_pdg_ids=all_pids,
                    max_jet_flavor=self.process.run_card["beam"]["max_jet_flavor"],
                    parton_line_scheme=getattr(
                        ms.MLMClustering.PartonLineScheme,
                        self.process.run_card["beam"]["parton_line_scheme"],
                    ),
                    alphas_scheme=getattr(
                        ms.MLMClustering.AlphasScheme,
                        self.process.run_card["beam"]["alphas_reweighting"],
                    ),
                    pdf_reweighting=self.process.run_card["beam"][
                        "pdf_reweighting"
                    ],
                    clustering_measure=getattr(
                        ms.MLMClustering.ClusteringMeasure,
                        self.process.run_card["beam"]["clustering_measure"],
                    ),
                    clustering_history=getattr(
                        ms.MLMClustering.ClusteringHistory,
                        self.process.run_card["beam"]["clustering_history"],
                    ),
                    clustering_mean_pdf_ladder=self.process.run_card["beam"][
                        "clustering_mean_pdf_ladder"
                    ],
                ),
                min_scale=min_scale,
                max_scale=max_scale,
            )
        else:
            self.scale = ms.EnergyScale(
                particle_count=self.particle_count,
                custom_scale=self.process.custom_function(
                    "scale", self.particle_count, representative_pids
                ),
                min_scale=min_scale,
                max_scale=max_scale,
                **self.process.scale_kwargs,
            )

        if self.process.run_card["run"]["dummy_matrix_element"]:
            self.matrix_elements = [None] * len(all_api_paths)
        else:
            self.matrix_elements = []
            for api_paths in all_api_paths:
                for context, api_path in zip(self.process.contexts, api_paths):
                    mat = context.load_matrix_element(
                        api_path, self.process.param_card_path,
                        me_parameters(self.process.run_card)
                    )
                self.matrix_elements.append(mat)

    def flavor_pids(self) -> list[list[int]]:
        """PDG ids of the external particles for every flavor of the
        integrand, in the order of its flavor index."""
        pids = []
        for flav in self.meta["flavors"]:
            if self.unmerged_meta is not None:
                subproc_index = self.meta["subprocesses"][flav["subprocess"]]
                flav = self.unmerged_meta[subproc_index]["flavors"][flav["flavor"]]
            pids.append(flav["options"][0])
        return pids

    def build_multi_channel_data(self) -> MultiChannelData:
        if self.multi_channel_data is not None:
            return self.multi_channel_data
        self.multi_channel_data = build_multi_channel_data(
            self.meta, self.process, self.unmerged_meta
        )
        return self.multi_channel_data

    def build_multichannel_phasespace(self) -> PhaseSpace:
        mcdata = self.build_multi_channel_data()
        channel_count = sum(len(topos) for topos in mcdata.topologies)
        drop_threshold = self.process.run_card["phasespace"]["drop_qcd_s_channel"]
        if drop_threshold >= 0 and channel_count > drop_threshold:
            mcdata = self.drop_qcd_s_channels(mcdata)
        mcdata = self.drop_empty_channels(mcdata)

        channels = []
        t_channel_mode = self.t_channel_mode(
            self.process.run_card["phasespace"]["t_channel"]
        )
        for channel_id, (chan_topologies, chan_permutations, chan_indices, active_flavors) in enumerate(zip(
            mcdata.topologies, mcdata.permutations, mcdata.channel_weight_indices,
            mcdata.active_flavors
        )):
            topo_count = len(chan_topologies)
            for topo_index, (topo, indices) in enumerate(zip(chan_topologies, chan_indices)):
                mapping = ms.PhaseSpaceMapping(
                    chan_topologies[0],
                    self.process.e_cm,
                    t_channel_mode=t_channel_mode,
                    cuts=self.cuts,
                    invariant_power=self.process.run_card["phasespace"]["invariant_power"],
                    permutations=chan_permutations,
                    leptonic=self.process.leptonic,
                    beam_rapidity=self.process.beam_rapidity,
                    mirror_beams=self.mirror_beams,
                )
                prefix = f"subproc{self.subproc_id}.channel{channel_id}"
                if topo_count > 1:
                    prefix += f".subchan{topo_index}"
                discrete_sym, discrete_flavor = self.build_discrete(
                    len(chan_permutations), len(self.meta["flavors"]), prefix
                )
                channels.append(Channel(
                    phasespace_mapping = mapping,
                    adaptive_mapping = self.build_vegas(mapping, prefix),
                    discrete_sym = discrete_sym,
                    discrete_flavor = discrete_flavor,
                    channel_weight_indices = indices,
                    name = f"{channel_id}",
                    active_flavors = active_flavors,
                ))

        remapped_chan_count = sum(
            len(indices) for indices in mcdata.channel_indices
        )
        if self.process.run_card["phasespace"]["sde_strategy"] == "denominators":
            prop_chan_weights = ms.PropagatorChannelWeights(
                [topo[0] for topo in mcdata.topologies], mcdata.permutations,
                mcdata.channel_indices
            )
            chan_weight_remap = []
        else:
            prop_chan_weights = None
            chan_weight_remap = [
                [
                    len(mcdata.symfact) if remap == -1 else remap
                    for remap in amp2_remap
                ]
                for amp2_remap in mcdata.amp2_remaps
            ]

        if any(len(topos) > 1 for topos in mcdata.topologies):
            subchan_weights = ms.SubchannelWeights(
                mcdata.topologies, mcdata.permutations, mcdata.channel_indices
            )
        else:
            subchan_weights = None

        return PhaseSpace(
            mode="multichannel",
            channels=channels,
            first_chan_weight_remap=chan_weight_remap,
            first_remapped_chan_count=remapped_chan_count,
            symfact=mcdata.symfact,
            prop_chan_weights=prop_chan_weights,
            subchan_weights=subchan_weights,
        )

    def has_mirrored_flavors(self) -> bool:
        """Does any flavor of this subprocess stand for both beam orientations?
        Resolved the same way build_integrands resolves flavor_mirror."""
        for flav in self.meta["flavors"]:
            if self.unmerged_meta is not None:
                subproc_index = self.meta["subprocesses"][flav["subprocess"]]
                flav = self.unmerged_meta[subproc_index]["flavors"][flav["flavor"]]
            if flav["mirror"]:
                return True
        return False

    def build_flat_phasespace(self) -> PhaseSpace:
        mapping = ms.PhaseSpaceMapping(
            self.incoming_masses + self.outgoing_masses,
            self.process.e_cm,
            mode=self.t_channel_mode(self.process.run_card["phasespace"]["flat_mode"]),
            cuts=self.cuts,
            leptonic=self.process.leptonic,
            beam_rapidity=self.process.beam_rapidity,
            mirror_beams=self.mirror_beams,
        )
        prefix = f"subproc{self.subproc_id}.flat"
        discrete_sym, discrete_flavor = self.build_discrete(
            1, len(self.meta["flavors"]), prefix
        )
        channel = Channel(
            phasespace_mapping = mapping,
            adaptive_mapping = self.build_vegas(mapping, prefix),
            discrete_sym = discrete_sym,
            discrete_flavor = discrete_flavor,
            channel_weight_indices = [0],
            name = "F",
            active_flavors = [],
        )
        if self.unmerged_meta is None:
            remap = [list(range(self.meta["diagram_count"]))]
        else:
            remap = [
                list(range(self.unmerged_meta[subproc]["diagram_count"]))
                for subproc in self.meta["subprocesses"]
            ]
        return PhaseSpace(
            mode="flat",
            channels=[channel],
            first_chan_weight_remap=remap,
            first_remapped_chan_count=1,
            symfact=[None],
        )

    def simplify_phasespace(
        self,
        multi_phasespace: PhaseSpace,
        cross_sections: list[float]
    ) -> PhaseSpace | None:
        assert multi_phasespace.mode == "multichannel"

        threshold = 1 - self.process.run_card["phasespace"]["combine_channel_threshold"]
        kept_channels = []
        tot_cs = sum(cross_sections)
        cum_cs = 0.
        seen_active_flavors = set()
        #seen_resonances = set()
        for index, (cs, chan) in sorted(
            enumerate(zip(cross_sections, multi_phasespace.channels)),
            key=lambda pair: pair[1][0],
            reverse=True
        ):
            cum_cs += cs
            has_unseen_flavors = False
            has_unseen_resonances = False
            for flavs in chan.active_flavors:
                for flav in flavs:
                    if flav not in seen_active_flavors:
                        has_unseen_flavors = True
                        seen_active_flavors.add(flav)
            #for resonance in chan.resonances:
            #    if resonance not in seen_resonances:
            #        has_unseen_resonances = True
            #        seen_resonances.add(flav)
            if has_unseen_flavors or has_unseen_resonances or not tot_cs \
                    or cum_cs / tot_cs < threshold:
                kept_channels.append(index)
        if len(kept_channels) >= len(cross_sections) - 1:
            return multi_phasespace

        channels = []
        channel_map = {}
        symfact = []
        for old_chan_index in kept_channels:
            channel = multi_phasespace.channels[old_chan_index]
            perm_count = max(1, channel.phasespace_mapping.channel_count())
            channel_index = len(symfact)
            symfact.append(None)
            symfact.extend([channel_index] * (perm_count - 1))
            channel_map.update({
                old_index: new_index
                for new_index, old_index in enumerate(
                    channel.channel_weight_indices, start=channel_index
                )
            })
            channels.append(Channel(
                phasespace_mapping = channel.phasespace_mapping,
                adaptive_mapping = channel.adaptive_mapping,
                discrete_sym = channel.discrete_sym,
                discrete_flavor = channel.discrete_flavor,
                channel_weight_indices = list(range(
                    channel_index, channel_index + perm_count
                )),
                name = channel.name,
                active_flavors = channel.active_flavors,
                event_generator = channel.event_generator,
            ))

        flat_phasespace = self.build_flat_phasespace()
        flat_channel = flat_phasespace.channels[0]
        channels.append(Channel(
            phasespace_mapping = flat_channel.phasespace_mapping,
            adaptive_mapping = flat_channel.adaptive_mapping,
            discrete_sym = flat_channel.discrete_sym,
            discrete_flavor = flat_channel.discrete_flavor,
            channel_weight_indices = [len(symfact)],
            name = flat_channel.name,
            active_flavors = flat_channel.active_flavors,
        ))
        flat_index = len(symfact)
        symfact.append(None)
        channel_map[len(multi_phasespace.symfact)] = len(symfact)
        if multi_phasespace.subchan_weights is None and len(multi_phasespace.first_chan_weight_remap) > 0:
            first_chan_weight_remap = [
                [
                    channel_map.get(remap, flat_index)
                    for remap in cw_remap
                ]
                for cw_remap in multi_phasespace.first_chan_weight_remap
            ]
            first_remapped_chan_count = len(symfact)
            second_chan_weight_remap = []
            second_remapped_chan_count = 0
        else:
            first_chan_weight_remap = multi_phasespace.first_chan_weight_remap
            first_remapped_chan_count = multi_phasespace.first_remapped_chan_count
            chan_count = multi_phasespace.first_remapped_chan_count if multi_phasespace.subchan_weights is None else multi_phasespace.subchan_weights.channel_count()
            second_chan_weight_remap = [
                channel_map.get(i, flat_index)
                for i in range(chan_count)
            ]
            second_remapped_chan_count = len(symfact)

        return PhaseSpace(
            mode="both",
            channels=channels,
            first_chan_weight_remap=first_chan_weight_remap,
            first_remapped_chan_count=first_remapped_chan_count,
            second_chan_weight_remap=second_chan_weight_remap,
            second_remapped_chan_count=second_remapped_chan_count,
            symfact=symfact,
            prop_chan_weights=multi_phasespace.prop_chan_weights,
            subchan_weights=multi_phasespace.subchan_weights,
        )

    def drop_qcd_s_channels(self, mcdata: MultiChannelData) -> MultiChannelData:
        """Drop channels with non-resonant QCD s-channel propagators, to reduce the
        channel count for processes with many diagrams. Channels are grouped by the
        number of QCD s-channels, allowing for more if necessary to map out all flavor
        indices. Channel weights belonging to a dropped channel are left unmapped."""
        groups_by_s_count = {}
        for index, count in enumerate(mcdata.qcd_s_channel_count):
            groups_by_s_count.setdefault(count, []).append(index)

        covered_flavors = set()
        kept_groups = set()
        for s_count in sorted(groups_by_s_count):
            s_count_groups = groups_by_s_count[s_count]
            if s_count == 0:
                selected = s_count_groups
            else:
                selected = [
                    index
                    for index in s_count_groups
                    if any(
                        flav not in covered_flavors
                        for flavs in mcdata.active_flavors[index]
                        for flav in flavs
                    )
                ]
            kept_groups.update(selected)
            for index in selected:
                for flavs in mcdata.active_flavors[index]:
                    covered_flavors.update(flavs)

        return self.keep_channel_groups(mcdata, sorted(kept_groups))

    def drop_empty_channels(self, mcdata: MultiChannelData) -> MultiChannelData:
        """Drop the channels whose phase space the cuts exclude entirely: a pair
        mass cut whose floor lies at or above the top of an on-shell window of
        the channel (ms.PhaseSpaceMapping.empty). Every point such a channel
        generates fails the cuts, so it could only ever fail to find one.

        If that leaves a flavour with no channel at all, its cross section is
        zero -- typically a cut that excludes the window of a decay chain every
        diagram shares -- and the run cannot sample it: that is reported as an
        error naming the cause, rather than as the survey finding no point
        passing the cuts."""
        if self.cuts is None:
            return mcdata
        t_channel_mode = self.t_channel_mode(
            self.process.run_card["phasespace"]["t_channel"]
        )
        empty = [
            ms.PhaseSpaceMapping(
                chan_topologies[0],
                self.process.e_cm,
                t_channel_mode=t_channel_mode,
                cuts=self.cuts,
                invariant_power=self.process.run_card["phasespace"]["invariant_power"],
                permutations=chan_permutations,
                leptonic=self.process.leptonic,
            ).empty()
            for chan_topologies, chan_permutations in zip(
                mcdata.topologies, mcdata.permutations
            )
        ]
        if not any(empty):
            return mcdata
        kept_groups = [index for index, is_empty in enumerate(empty) if not is_empty]
        covered_flavors = {
            flav
            for index in kept_groups
            for flavs in mcdata.active_flavors[index]
            for flav in flavs
        }
        uncovered = sorted({
            flav
            for index, is_empty in enumerate(empty) if is_empty
            for flavs in mcdata.active_flavors[index]
            for flav in flavs
        } - covered_flavors)
        if uncovered:
            raise ValueError(
                f"subprocess {self.meta['incoming']} > {self.meta['outgoing']}: "
                "the pair mass cuts exclude the whole on-shell window "
                f"(bw_cutoff = {self.process.run_card['phasespace']['bw_cutoff']}) "
                f"of every channel of flavour(s) {uncovered}, so their cross "
                "section is zero; loosen the cut or remove that decay"
            )
        logger.info(
            "subprocess %d: %d channel(s) dropped, the cuts exclude the "
            "on-shell window they sample", self.subproc_id, len(empty) - len(kept_groups)
        )
        return self.keep_channel_groups(mcdata, kept_groups)

    def keep_channel_groups(
        self, mcdata: MultiChannelData, kept_groups: list[int]
    ) -> MultiChannelData:
        """Rebuild the multi-channel data with only the channel groups (indices
        into mcdata.topologies) in kept_groups. Channel weights belonging to a
        dropped channel are left unmapped."""
        if len(kept_groups) == len(mcdata.topologies):
            return mcdata

        amp2_remaps = [[-1] * len(remap) for remap in mcdata.amp2_remaps]
        symfact = []
        topologies = []
        permutations = []
        channel_indices = []
        channel_weight_indices = []
        diagram_indices = []
        diagram_color_indices = []
        diagram_propagator_pdgs = []
        active_flavors = []
        qcd_s_channel_count = []
        channel_index = 0

        for group in kept_groups:
            chan_topologies = mcdata.topologies[group]
            chan_permutations = mcdata.permutations[group]
            chan_diagram_indices = mcdata.diagram_indices[group]
            topo_count = len(chan_topologies)

            channel_index_first = channel_index
            symfact_index_first = len(symfact)
            for i, diag in enumerate(chan_diagram_indices):
                if self.unmerged_meta is None:
                    amp2_remaps[0][diag] = channel_index
                else:
                    for amp2_remap, d in zip(amp2_remaps, diag):
                        if d != -1:
                            amp2_remap[d] = channel_index
                if i == 0:
                    symfact.extend([None] * topo_count)
                else:
                    symfact.extend(range(symfact_index_first, symfact_index_first + topo_count))
                channel_index += 1

            topologies.append(chan_topologies)
            permutations.append(chan_permutations)
            channel_indices.append(list(range(channel_index_first, channel_index)))
            channel_weight_indices.append([
                [
                    symfact_index_first + topo_index + i * topo_count
                    for i in range(len(chan_permutations))
                ]
                for topo_index in range(topo_count)
            ])
            diagram_indices.append(chan_diagram_indices)
            if mcdata.diagram_color_indices:
                diagram_color_indices.append(mcdata.diagram_color_indices[group])
            if mcdata.diagram_propagator_pdgs:
                diagram_propagator_pdgs.append(
                    mcdata.diagram_propagator_pdgs[group]
                )
            active_flavors.append(mcdata.active_flavors[group])
            qcd_s_channel_count.append(mcdata.qcd_s_channel_count[group])

        return MultiChannelData(
            amp2_remaps,
            symfact,
            topologies,
            permutations,
            channel_indices,
            channel_weight_indices,
            diagram_indices,
            diagram_color_indices,
            diagram_propagator_pdgs,
            active_flavors,
            qcd_s_channel_count,
        )

    def set_madnis_auto_settings(self, rsd: float, random_dim: int):
        """Choose the MadNIS settings of this subprocess from the relative
        spread rsd of the survey. random_dim is the smallest number of random
        numbers of its channels: a normalizing flow needs at least two, so a
        one-dimensional phase space (a 2 -> 1 process) never uses MadNIS."""
        madnis_args = self.process.run_card["madnis"]
        n_out = len(self.meta["outgoing"])
        n_events = self.process.run_card["generation"]["events"]
        is_gridpack = self.process.run_card["gridpack"]["save_gridpack"]
        train_batches = madnis_args["train_batches"]
        hidden_dim = min(max(int((7 * rsd) / 32) * 32 + 64, 64), 256)
        flow_layers = 3 if rsd < 32 else 4
        lr = min(max((10000 - train_batches) / 8000 * 7e-4 + 3e-4, 3e-4), 1e-3)
        if n_out <= 2:
            enable = False
        elif n_out == 3:
            enable = rsd > 10. or n_events > 1000000 or is_gridpack
        else:
            enable = True
        fixed_cwnet_fraction = max(0.33, 1.0 - 10000. / train_batches)
        batch_size_per_channel = min(max(int((7 * rsd) / 32) * 32 + 64, 128), 512)

        self.madnis_settings = {
            "enable": enable,
            "flow_layers": flow_layers,
            "flow_hidden_dim": hidden_dim,
            "discrete_hidden_dim": hidden_dim,
            "cwnet_hidden_dim": hidden_dim,
            "train_batches": train_batches,
            "lr": lr,
            "fixed_cwnet_fraction": fixed_cwnet_fraction,
            "batch_size_per_channel": batch_size_per_channel,
        }
        for key, value in self.madnis_settings.items():
            run_card_value = madnis_args[key]
            if run_card_value != "auto":
                self.madnis_settings[key] = run_card_value
        self.madnis_supported = random_dim >= 2
        if not self.madnis_supported and self.madnis_settings["enable"]:
            logger.warning(
                "MadNIS disabled for subprocess %s: its phase space has %d "
                "random dimension(s), a normalizing flow needs at least 2",
                self.subproc_id, random_dim,
            )
            self.madnis_settings["enable"] = False

    def build_madnis(self, phasespace: PhaseSpace) -> PhaseSpace:
        madnis_args = self.process.run_card["madnis"]
        # Shared across all networks below: initialize_globals() derives an
        # independent, non-colliding stream per tensor from this one base seed.
        seed = self.process.run_seed
        channels = []
        for channel_id, channel in enumerate(phasespace.channels):
            prefix = f"subproc{self.subproc_id}.channel{channel_id}"
            cond_dim = 0

            flow_dim = channel.phasespace_mapping.random_dim()
            flow = ms.Flow(
                input_dim=flow_dim,
                condition_dim=cond_dim,
                prefix=prefix,
                bin_count=madnis_args["flow_spline_bins"],
                subnet_hidden_dim=self.madnis_settings["flow_hidden_dim"],
                subnet_layers=self.madnis_settings["flow_layers"],
                subnet_activation=self.activation(madnis_args["flow_activation"]),
                invert_spline=madnis_args["flow_invert_spline"],
                subnet_precision=self.nn_precision(),
            )
            if channel.adaptive_mapping is None:
                flow.initialize_globals(self.process.contexts[0], seed)
            else:
                flow.initialize_from_vegas(
                    self.process.contexts[0], channel.adaptive_mapping.grid_name(),
                    seed
                )
            cond_dim += flow_dim

            # discrete_sym runs after the adaptive map, so it can condition on its latent.
            discrete_sym = channel.discrete_sym
            if discrete_sym is not None:
                perm_count = channel.phasespace_mapping.channel_count()
                discrete_sym = ms.DiscreteFlow(
                    option_counts=[perm_count],
                    prefix=f"{prefix}.discrete_flow_sym",
                    dims_with_prior=[],
                    condition_dim=cond_dim,
                    subnet_hidden_dim=self.madnis_settings["discrete_hidden_dim"],
                    subnet_layers=madnis_args["discrete_layers"],
                    subnet_activation=self.activation(madnis_args["discrete_activation"]),
                    subnet_precision=self.nn_precision(),
                )
                discrete_sym.initialize_globals(self.process.contexts[0], seed)
                cond_dim += perm_count

            discrete_flavor = channel.discrete_flavor
            if discrete_flavor is not None:
                discrete_flavor = ms.DiscreteFlow(
                    option_counts=[len(self.meta["flavors"])],
                    prefix=f"{prefix}.discrete_flow_flavor",
                    dims_with_prior=[0],
                    condition_dim=cond_dim,
                    subnet_hidden_dim=self.madnis_settings["discrete_hidden_dim"],
                    subnet_layers=madnis_args["discrete_layers"],
                    subnet_activation=self.activation(madnis_args["discrete_activation"]),
                    subnet_precision=self.nn_precision(),
                )
                discrete_flavor.initialize_globals(self.process.contexts[0], seed)

            channels.append(Channel(
                phasespace_mapping = channel.phasespace_mapping,
                adaptive_mapping = flow,
                discrete_sym = discrete_sym,
                discrete_flavor = discrete_flavor,
                channel_weight_indices = channel.channel_weight_indices,
                name = channel.name,
                active_flavors = channel.active_flavors,
            ))

        return PhaseSpace(
            mode="both",
            channels=channels,
            first_chan_weight_remap=phasespace.first_chan_weight_remap,
            first_remapped_chan_count=phasespace.first_remapped_chan_count,
            second_chan_weight_remap=phasespace.second_chan_weight_remap,
            second_remapped_chan_count=phasespace.second_remapped_chan_count,
            symfact=phasespace.symfact,
            cwnet=self.build_cwnet(len(phasespace.symfact)),
            prop_chan_weights=phasespace.prop_chan_weights,
            subchan_weights=phasespace.subchan_weights,
        )

    def build_vegas(self, mapping: ms.PhaseSpaceMapping, prefix: str) -> ms.VegasMapping:
        if not self.process.run_card["vegas"]["enable"]:
            return None

        vegas = ms.VegasMapping(
            mapping.random_dim(),
            self.process.run_card["vegas"]["bins"],
            prefix,
        )
        for context in self.process.contexts:
            vegas.initialize_globals(context)
        return vegas

    def build_discrete(
        self, permutation_count: int, flavor_count: int, prefix: str
    ) -> tuple[ms.DiscreteSampler | None, ms.DiscreteSampler | None]:
        is_adaptive = self.process.run_card["phasespace"]["adaptive_symmetry_sampling"]
        if is_adaptive and permutation_count > 1:
            discrete_sym = ms.DiscreteSampler(
                [permutation_count], f"{prefix}.discrete_sym"
            )
            for context in self.process.contexts:
                discrete_sym.initialize_globals(context)
        else:
            discrete_sym = None

        if flavor_count > 1:
            # the flavor probabilities are conditioned on the PDF prior; a
            # leptonic process (or a decay) has none, and the Integrand then
            # passes no condition: declaring one anyway fails to build
            # ("keys and values must have the same size"), e.g. z > q q~ with
            # both u- and d-type rows
            discrete_flavor = ms.DiscreteSampler(
                [flavor_count], f"{prefix}.discrete_flavor",
                [] if self.process.leptonic else [0]
            )
            for context in self.process.contexts:
                discrete_flavor.initialize_globals(context)
        else:
            discrete_flavor = None

        return discrete_sym, discrete_flavor

    def build_cwnet(self, channel_count: int) -> ms.ChannelWeightNetwork:
        #if channel_count == 1:
        #    return None
        madnis_args = self.process.run_card["madnis"]
        cwnet = ms.ChannelWeightNetwork(
            channel_count=channel_count,
            particle_count=self.particle_count,
            hidden_dim=self.madnis_settings["cwnet_hidden_dim"],
            layers=madnis_args["cwnet_layers"],
            activation=self.activation(madnis_args["cwnet_activation"]),
            prefix=f"subproc{self.subproc_id}.cwnet",
            precision=self.nn_precision(),
        )
        cwnet.initialize_globals(
            self.process.contexts[0], self.process.run_seed
        )
        return cwnet

    def t_channel_mode(self, name: str) -> ms.PhaseSpaceMapping.TChannelMode:
        modes = {
            "propagator": ms.PhaseSpaceMapping.propagator,
            "rambo": ms.PhaseSpaceMapping.rambo,
            "chili": ms.PhaseSpaceMapping.chili,
        }
        if name in modes:
            return modes[name]
        else:
            raise ValueError(f"Invalid t-channel mode '{name}'")

    def nn_precision(self) -> ms.DataType:
        # one precision for all networks: the optimizer trains all of them together
        # the run card only allows float32 and float64
        return getattr(ms.DataType, self.process.run_card["madnis"]["nn_float_precision"])

    def activation(self, name: str) -> ms.MLP.Activation:
        activations = {
            "relu": ms.MLP.relu,
            "leaky_relu": ms.MLP.leaky_relu,
            "elu": ms.MLP.elu,
            "gelu": ms.MLP.gelu,
            "sigmoid": ms.MLP.sigmoid,
            "softplus": ms.MLP.softplus,
        }
        if name in activations:
            return activations[name]
        else:
            raise ValueError(f"Invalid activation function '{name}'")

    def build_integrands(
        self,
        phasespace: PhaseSpace,
        madnis_training: bool = False,
        drop_cuts_and_rescale: bool = False
    ) -> list[ms.Integrand]:
        flavors = []
        flavor_remap = []
        flavor_factors = []
        flavor_mirror = []
        flavor_diff_xs_indices = []
        flavor_subproc_indices = []
        flavor_per_subproc_remap = []

        for flav in self.meta["flavors"]:
            if self.unmerged_meta is not None:
                diff_xs_index = flav["subprocess"]
                subproc_index = self.meta["subprocesses"][diff_xs_index]
                ps_flavor = flav["flavor"]
                flavor_diff_xs_indices.append(diff_xs_index)
                flavor_subproc_indices.append(subproc_index)
                flavor_per_subproc_remap.append(ps_flavor)
                flav = self.unmerged_meta[subproc_index]["flavors"][ps_flavor]
            flavors.append(flav["options"][0])
            flavor_remap.append(flav["index"])
            flavor_factors.append(len(flav["options"]))
            flavor_mirror.append(flav["mirror"])

        cross_sections = []
        for matrix_element in self.matrix_elements:
            if matrix_element:
                mat = ms.MatrixElement(
                    matrix_element,
                    ms.Integrand.matrix_element_inputs,
                    ms.Integrand.matrix_element_outputs,
                    True,
                    self.process.me_frame,
                    self.process.incoming_count,
                )
            else:
                #TODO: not working in merged mode
                mat = ms.MatrixElement(
                    0xBADCAFE,
                    self.particle_count,
                    ms.Integrand.matrix_element_inputs,
                    ms.Integrand.matrix_element_outputs,
                    self.meta["diagram_count"],
                    True,
                    self.process.me_frame,
                    self.process.incoming_count,
                )
            pdf_grid = None if self.process.leptonic else self.process.pdf_grid
            pdf_arg = None if self.process.leptonic else ms.CachedPdf()
            cross_sections.append(
                ms.DifferentialCrossSection(
                    matrix_element=mat,
                    # For a decay this is the decaying particle's mass, and the
                    # flux becomes 1/(2M): the result is a partial width in
                    # GeV, not a cross section in pb.
                    cm_energy=self.process.e_cm,
                    running_coupling=None,
                    energy_scale=ms.CachedScale(),
                    pid_options=[],
                    pdf1=pdf_arg,
                    pdf2=pdf_arg,
                    input_momentum_fraction=True,
                    decay=self.process.is_decay,
                    matrix_element_output=self.process.store_matrix_elements,
                )
            )
        partial_weights = self.process.systematics_enabled()
        madnis_args = self.process.run_card["madnis"]
        custom_bias = self.process.custom_function("bias", self.particle_count, flavors)
        # With an MLM history picked per diagram, the cross section at the
        # scales of that history is the first matrix-element evaluation rescaled
        # by alpha_s and PDF ratios when every diagram has the same power of
        # alpha_s (-1: mixed powers, evaluate the matrix element again)
        mlm_history_qcd_power = int(self.meta.get("qcd_power", -1))
        integrands = []
        for channel in phasespace.channels:
            integrands.append(ms.Integrand(
                channel.phasespace_mapping,
                cross_sections,
                channel.adaptive_mapping,
                channel.discrete_sym,
                channel.discrete_flavor,
                flavors,
                pdf_grid,
                self.process.running_coupling,
                self.scale,
                phasespace.prop_chan_weights,
                phasespace.subchan_weights,
                phasespace.cwnet,
                phasespace.first_chan_weight_remap,
                phasespace.first_remapped_chan_count,
                phasespace.second_chan_weight_remap,
                phasespace.second_remapped_chan_count,
                madnis_training,
                drop_cuts_and_rescale,
                partial_weights,
                channel.channel_weight_indices,
                channel.active_flavors,
                flavor_remap,
                flavor_factors,
                flavor_mirror,
                flavor_diff_xs_indices,
                flavor_subproc_indices,
                flavor_per_subproc_remap,
                madnis_args["compressed_channel_weight_count"],
                custom_bias,
                None if self.process.leptonic else self.process.pdf_grid2,
                mlm_history_qcd_power,
            ))
        #print(integrands[1].function())
        #for i in integrands: print(i.function())
        return integrands


_ROOTED_OPTIONS = ('lhapdf', 'lhapdf_py3', 'lhapdf_py2', 'heptools_install_dir')

def load_mg5_options(me_dir=None) -> dict:
    """Read the tool paths from the MadGraph configuration, so the launcher
    knows where LHAPDF and the optional programs (Pythia8/Delphes/MadSpin/
    reweight/analysis) live.

    Files are read least- to most-specific, each overriding the previous, which
    is the same layering CommonRunCmd.set_configuration uses: the card inside
    this process directory has the last word, then this installation, then the
    per-user file. Relative values are resolved against the root of the file
    they came from.

    ``me_dir`` is the process directory whose Cards/me5_configuration.txt has
    the last word. It defaults to the working directory, which is right for
    bin/generate_events (it chdirs there first) but not for an in-process
    caller, which has not moved yet.
    """

    import madgraph
    mg5dir = os.path.dirname(os.path.dirname(os.path.abspath(madgraph.__file__)))
    me_dir = os.getcwd() if me_dir is None else me_dir

    options = {
        'pythia-pgs_path': None, 'pythia8_path': None,
        'madanalysis5_path': None, 'exrootanalysis_path': None, 'delphes_path': None,
        'rivet_path': None, 'contur_path': None, 'f2py_compiler': None,
        'lhapdf': None, 'lhapdf_py3': None, 'lhapdf_py2': None, 'timeout': 0,
        'heptools_install_dir': None, 'hepmc3_path': None,
    }
    config_files = []
    base_config = misc.base_config_file()
    if base_config:
        config_files.append((base_config, mg5dir))
    user_config = misc.user_config_file()
    if user_config:
        config_files.append((user_config, mg5dir))
    config_files.append((misc.install_config_file(mg5dir), mg5dir))
    config_files.append((os.path.join(me_dir, 'Cards', 'me5_configuration.txt'),
                         me_dir))
    for cfg, root in config_files:
        if not os.path.exists(cfg):
            continue
        with open(cfg) as fsock:
            for line in fsock:
                line = line.split('#', 1)[0]
                if '=' not in line:
                    continue
                name, value = (x.strip() for x in line.split('=', 1))
                if name not in options or value in ('', 'None'):
                    continue
                if (name.endswith('_path') or name in _ROOTED_OPTIONS) and \
                        not os.path.isabs(value) and \
                        (os.sep in value or value.startswith('.')):
                    value = os.path.normpath(os.path.join(root, value))
                options[name] = value
    options['mg5_path'] = mg5dir  # enables MadSpin/reweight
    return options


_LHAPDF = None

def lhapdf_paths(refresh: bool = False) -> misc.LhapdfPaths:
    """The LHAPDF installation this run should use, resolved once.

    Uses the same helper as the 'launch' command, so driving a process
    directory straight through bin/generate_events behaves identically to
    launching it from MadGraph.
    """

    global _LHAPDF
    if _LHAPDF is None or refresh:
        _LHAPDF = misc.resolve_lhapdf(load_mg5_options())
        # Export what we found: the real LHAPDF library, systematics and the
        # MadSpin/reweight subprocesses read these from the environment.
        if _LHAPDF.data_paths:
            os.environ["LHAPDF_DATA_PATH"] = os.pathsep.join(_LHAPDF.data_paths)
        if _LHAPDF.config:
            os.environ["MADGRAPH_LHAPDF_CONFIG"] = _LHAPDF.config
    return _LHAPDF


def lhapdf_index_paths() -> list:
    """The PDF directories to look an lhaid up in. Without LHAPDF there may be
    no pdfsets.index anywhere yet, so fetch one into the download directory."""

    paths = lhapdf_paths()
    data_paths = list(paths.data_paths)
    if not any(os.path.exists(os.path.join(p, "pdfsets.index")) for p in data_paths) \
            and paths.download_path \
            and misc.download_lhapdf_index(paths.download_path):
        data_paths.append(paths.download_path)
    return data_paths


class MG7Cmd(Cmd):
    """The mg7 run interface: the command object that drives a launch.

    It plays two roles, and they are the same object on purpose.

    1. It is the *mother* of the merged switch/card question (the thing
       ``AskRunEditCard`` calls back into for ``keep_cards``/``do_open``/
       ``compute_widths``).
    2. It is the child command interface MG5 registers via
       ``define_child_cmd_interface`` when you type ``launch`` on an mg7 output.

    Role 2 is what makes role 1 behave. ``Cmd.ask`` resolves a scripted answer
    through ``self.check_answer_in_input_file``, where ``self`` is the mother --
    so once MG5 hands us its ``inputfile`` and registers itself as our
    ``mother``, the question reads the very lines that follow ``launch`` in the
    user's command file, and any line it does not recognise goes back up to MG5
    through ``store_line`` instead of being silently consumed. This is exactly
    how a madevent output already behaves; before this class existed the mg7
    branch shelled out to ``bin/generate_events`` and had to guess which
    following lines belonged to the run, which quietly swallowed every ``set``.
    """

    prompt = 'MG7> '

    def __init__(self, me_dir='.', options=None, *args, **opts):
        super().__init__(*args, **opts)
        self.me_dir = os.path.abspath(me_dir) if me_dir != '.' else '.'
        # The rest of the launcher is written against the process directory as
        # the working directory, so do_generate_events chdirs there; but at
        # construction time MG5 has not moved, hence the explicit me_dir.
        self.options = load_mg5_options(
            self.me_dir if self.me_dir != '.' else None)
        if options:
            # MG5's own resolved tool paths win: they are the ones the user
            # configured in the session that is doing the launching.
            self.options.update({k: v for k, v in options.items()
                                 if k in self.options and v is not None})
            # and so does its error policy: with crash_on_error a failing
            # launch has to stop the script with a non-zero exit code, as a
            # madevent launch does, instead of returning to the prompt.
            if 'crash_on_error' in options:
                self.options['crash_on_error'] = options['crash_on_error']
        self.plugin_path = []
        self.proc_characteristics = {'grouped_matrix': False, 'limitations': []}
        self._model = None  # lazily imported by get_model()

    # ------------------------------------------------------------------
    # the contract the card question expects from its mother
    # ------------------------------------------------------------------
    def keep_cards(self, need_card=[], ignore=[]):
        from madgraph.interface.common_run_interface import CommonRunCmd
        return CommonRunCmd.keep_cards(self, need_card, ignore)

    def do_open(self, line):
        from madgraph.interface.common_run_interface import CommonRunCmd
        CommonRunCmd.do_open(self, line)

    def check_open(self, args):
        from madgraph.interface.common_run_interface import CommonRunCmd
        CommonRunCmd.check_open(self, args)

    def get_model(self):
        """The model this process was generated with (an ``import_ufo`` model).

        This is the hook the card question goes through for everything that
        needs the model rather than the card: 'update dependent' (which resets
        the masses/widths the model computes from the free parameters) and the
        generic auto-width handler both call ``mother_interface.get_model()``.
        Without it the launch question could only warn that it had failed to
        update the dependent parameters, and MadSpin/the shower then read
        whatever inconsistent values the param_card still held.

        None when the process recorded no model (an output made before
        SubProcesses/model.txt existed) -- the caller warns in that case.
        """
        if self._model is None:
            me_dir = self.me_dir if self.me_dir != '.' else None
            # False (not None) so a model that cannot be imported is not
            # retried at every question.
            self._model = load_process_model(me_dir) or False
        return self._model or None

    def do_compute_widths(self, line):
        # The interactive card editor delegates 'auto' width computation to
        # the mother interface. Reuse the runtime helper (madgraph subprocess
        # + the model stored at output time). ``line`` looks like
        # "<pdgs> --path=<param_card> [--nlo]"; we only need the card path.
        m = re.search(r'--path=(\S+)', line or "")
        path = m.group(1) if m else os.path.join("Cards", "param_card.dat")
        compute_auto_widths(path)
        # return an empty mapping: the caller iterates out.items() for the
        # small-width treatment, which mg7 does not apply.
        return {}

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def do_generate_events(self, line):
        """generate_events [-f] [-n NAME] [--laststep=parton] [--reweight=CARD]
                           [--reweight-backend=classic|madtrex|auto]

        Run the mg7 generation and the selected post-processing tools.
        --reweight=CARD is a shortcut for switching reweighting on with CARD
        as Cards/reweight_card.dat; --reweight-backend chooses who reweights
        (see madgraph/various/reweight_backend.py)."""

        opts = self._parse_run_options(line)
        cwd = os.getcwd()
        target = self.me_dir if self.me_dir != '.' else cwd
        if opts['reweight_card']:
            opts['reweight_card'] = os.path.abspath(opts['reweight_card'])
        try:
            os.chdir(target)
            _setup_logging()
            if opts['name']:
                self._set_run_name(opts['name'])
            if opts['reweight_card']:
                install_reweight_card(opts['reweight_card'])
            switch = {}
            if not opts['force']:
                switch = ask_edit_cards(mother=self)
            switch = self._apply_laststep(switch, opts)
            if opts['reweight_card'] and _off(switch.get('reweight')):
                switch = dict(switch, reweight='ON')
            reweight = prepare_reweighting(switch, opts['reweight_backend'],
                                           self.options)
            force_lhe_output_if_needed(switch, reweight)
            _raise_open_file_limit()
            run_generation(switch, reweight)
        finally:
            os.chdir(cwd)

    # ``launch`` is the name MG5 users type; keep it working here too.
    do_launch = do_generate_events

    def do_reweight(self, line):
        """reweight RUN_NAME [--reweight=CARD] [--reweight-backend=classic|madtrex|auto]

        Reweight the events of an earlier run with Cards/reweight_card.dat
        (or CARD, which then becomes Cards/reweight_card.dat), as
        generate_events does after the generation: the weights are added to
        the run's event file, after the weights already there."""
        args = self.split_arg(line)
        run_name, card, backend = None, None, None
        for arg in args:
            if arg.startswith('--reweight='):
                card = os.path.abspath(arg.split('=', 1)[1])
            elif arg.startswith('--reweight-backend='):
                backend = arg.split('=', 1)[1]
            elif arg.startswith('-'):
                logger.warning("'%s' is not an option of reweight and is ignored", arg)
            else:
                run_name = arg
        cwd = os.getcwd()
        target = self.me_dir if self.me_dir != '.' else cwd
        try:
            os.chdir(target)
            _setup_logging()
            if run_name is None:
                runs = list_runs()
                raise self.InvalidCmd(
                    'reweight: give the run to reweight%s'
                    % (' (%s)' % ', '.join(runs) if runs else ': there is no run yet'))
            try:
                run = ExistingRun(run_name)
            except ValueError as error:
                raise self.InvalidCmd('reweight: %s' % error)
            if card:
                install_reweight_card(card)
            if not os.path.exists(REWEIGHT_CARD):
                raise self.InvalidCmd('reweight: %s is missing' % REWEIGHT_CARD)
            decision = prepare_reweighting({'reweight': 'ON'}, backend, self.options)
            if decision is None:
                return
            run_reweighting(run, REWEIGHT_CARD, decision=decision)
        finally:
            os.chdir(cwd)

    def complete_reweight(self, text, line, begidx, endidx):
        me_dir = self.me_dir if self.me_dir != '.' else os.getcwd()
        options = ['--reweight=', '--reweight-backend=']
        return self.list_completion(text, list_runs(os.path.join(me_dir, 'Events'))
                                    + options, line)

    def do_npy_to_lhe(self, line):
        """npy_to_lhe [RUN_NAME] [-o OUTPUT] [--seed N] [--no-gzip]

        Write the LHE file (events.lhe.gz) of a run that produced npy events,
        e.g. to shower it. Without a run name, the most recent such run."""
        import shlex
        me_dir = self.me_dir if self.me_dir != '.' else os.getcwd()
        try:
            npy_to_lhe.main(shlex.split(line), me_dir=me_dir)
        except SystemExit as error:
            if error.code not in (None, 0):
                logger.error(str(error.code))

    def do_quit(self, line):
        """Leave the mg7 run interface."""
        return super().do_quit(line)

    do_exit = do_quit

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _parse_run_options(self, line):
        """Understand the subset of MG5's `launch` options that mean something
        for an mg7 run, and say so out loud for the ones that do not."""
        out = {'force': False, 'name': '', 'laststep': '', 'reweight_card': None,
               'reweight_backend': None}
        args = self.split_arg(line)
        index = 0
        while index < len(args):
            arg = args[index]
            index += 1
            if arg in ('-f', '--force'):
                out['force'] = True
            elif arg in ('-n', '--name') and index < len(args):
                out['name'] = args[index]
                index += 1
            elif arg.startswith(('-n=', '--name=')):
                out['name'] = arg.split('=', 1)[1]
            elif arg in ('-s', '--laststep') and index < len(args):
                out['laststep'] = args[index]
                index += 1
            elif arg.startswith(('-s=', '--laststep=')):
                out['laststep'] = arg.split('=', 1)[1]
            elif arg == '--reweight' and index < len(args):
                out['reweight_card'] = args[index]
                index += 1
            elif arg.startswith('--reweight='):
                out['reweight_card'] = arg.split('=', 1)[1]
            elif arg == '--reweight-backend' and index < len(args):
                out['reweight_backend'] = args[index]
                index += 1
            elif arg.startswith('--reweight-backend='):
                out['reweight_backend'] = arg.split('=', 1)[1]
            elif arg.startswith('-'):
                logger.warning(
                    "'%s' is not supported by the mg7 output and is ignored; "
                    "the equivalent settings live in Cards/run_card.toml", arg)
            elif arg and not out['name']:
                out['name'] = arg
        return out

    def _set_run_name(self, name):
        """Honour `launch -n NAME`: the mg7 run directory is
        Events/<run.run_name>_NN, so the name is a run_card setting."""
        path = os.path.join("Cards", "run_card.toml")
        run_card = RunCardMG7(path, consistency=False)
        if run_card["run"]["run_name"] != name:
            run_card["run"]["run_name"] = name
            run_card.write(path)
            logger.info("run name set to '%s'", name)

    def _apply_laststep(self, switch, opts):
        """`--laststep=parton` means: generate the events, run nothing on them.

        The mg7 tool selection is the switch dict the question returns, so
        stopping at parton level is switching those entries off.
        """
        if opts['laststep'] not in ('parton', 'auto', ''):
            logger.warning(
                "--laststep=%s is not supported by the mg7 output; "
                "select the programs to run in the launch question instead",
                opts['laststep'])
            return switch
        if opts['laststep'] != 'parton' or not switch:
            return switch
        switch = dict(switch)
        for key in ('shower', 'detector', 'analysis', 'madspin', 'reweight'):
            if not _off(switch.get(key, 'OFF')):
                logger.info("--laststep=parton: not running %s", key)
                switch[key] = 'OFF'
        return switch


def build_selector_cmd(mother=None):
    """Build the (monkey-patched) mother command + merged switch/card selector
    used by the mg7 output.  Returns the selector *class* and a mother instance
    understood by AskRun/AskforEditCard.

    ``mother`` lets an in-process caller supply its own :class:`MG7Cmd` -- the
    one MG5 registered as its child -- so that the question resolves scripted
    answers against MG5's own command file (see :class:`MG7Cmd`). When it is
    omitted a standalone instance is built, which is what ``bin/generate_events``
    gets.
    """

    from madgraph.interface.madevent_interface import AskRunEditCard

    from madgraph.various import banner as _banner_mod
    from madgraph.various import misc as _misc

    class MG7Selector(AskRunEditCard):
        """Merged switch/card question for the mg7 output.

        The mg7 run_card is a TOML file, so define_paths/init_run/do_set are
        overridden *on this class* (not globally on AskforEditCard) to treat it
        as such. Scoping them here is important: MadSpin/reweight spawn their own
        legacy madevent runs in the same process, and a global patch would break
        their (run_card.dat based) card editing."""

        # param_card + the TOML run_card are always offered
        always_cards = ['param_card.dat', 'run_card.toml']
        optional_cards = []

        def define_paths(self, **opt):
            super().define_paths(**opt)
            self.paths["run"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
            self.paths["run_card.toml"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
            # the TOML run_card uses its own default file (concrete defaults
            # written at output time); this powers "set <param> default".
            self.paths["run_default"] = os.path.join(self.me_dir, "Cards", "run_card_default.toml")

        def init_run(self, cards):
            # Make sure the run_card is loaded as a RunCardMG7 (the generic
            # editor does not recognise run_card.toml), else "set <param>" fails.
            out = super().init_run(cards)
            if not isinstance(getattr(self, "run_card", None), RunCardMG7):
                toml_path = self.paths.get("run") or os.path.join(
                    self.me_dir, "Cards", "run_card.toml")
                if os.path.exists(toml_path):
                    try:
                        # allow_scan so a run_card holding scan:[...] values loads
                        with _misc.TMP_variable(_banner_mod.RunCard, "allow_scan", True):
                            self.run_card = RunCardMG7(toml_path, consistency="warning")
                        self.run_set = list(self.run_card.keys())
                    except Exception as err:
                        logger.warning("could not load %s: %s", toml_path, err)
            if isinstance(getattr(self, "run_card", None), RunCardMG7):
                self.run_card.allow_scan = True
            return getattr(self, "run_set", out)

        def check_card_consistency(self):
            # madevent settings given per beam ("set fixed_fac_scale1 ...")
            # that the mg7 card could not follow: said once, at the end, so
            # "set fixed_fac_scale1 T" then "set fixed_fac_scale2 T" stays
            # quiet. Collected first: lo_set_warnings also forgets them, which
            # must happen even if the generic checks below raise.
            warnings = []
            if isinstance(getattr(self, "run_card", None), RunCardMG7):
                warnings = self.run_card.lo_set_warnings()
            try:
                super().check_card_consistency()
            finally:
                for text in warnings:
                    logger.warning(text)

        def do_set(self, line, *args, **kwargs):
            # madevent-style shortcuts (lhc/lep/fixed_scale/no_parton_cut), cut
            # editing, energy units and arithmetic/mass expressions on the TOML
            # run_card, intercepted before delegating to the generic editor.
            targs = self.split_arg(line)
            run_card = getattr(self, "run_card", None)
            if isinstance(run_card, RunCardMG7) and targs:
                start = 1 if targs[0] == "run_card" else 0
                if len(targs) > start:
                    name = targs[start]
                    nlow = name.lower()
                    rest = " ".join(targs[start + 1:]).split("#")[0].strip()
                    masses = run_card.get_mass_shortcuts(getattr(self, "param_card", None))

                    if nlow in ("no_parton_cut", "nocut", "no_cut"):
                        run_card.remove_all_cut()
                        logger.info("removing all cuts from the run_card.toml")
                        self.modified_card.add("run")
                        return
                    # [histograms] is a list of observables, not a parameter,
                    # so "set histograms ..." can only mean the whole section.
                    if nlow in ("histograms", "histogram", "plots"):
                        value = rest.lower()
                        if value in ("off", "none", "no", "false", "0"):
                            run_card.remove_all_histograms()
                            logger.info(
                                "removing all histograms from the run_card.toml")
                            self.modified_card.add("run")
                            return
                        if value in ("default", "on"):
                            if run_card.restore_default_histograms(
                                    self.paths.get("run_default")):
                                logger.info(
                                    "restored the %d histograms written at "
                                    "output time",
                                    len(run_card["histograms"]))
                                self.modified_card.add("run")
                            else:
                                self.invalid_set(
                                    "no run_card_default.toml to restore the "
                                    "histograms from")
                            return
                        self.invalid_set(
                            "'set histograms %s': only OFF (remove them all) "
                            "and default (put the generated ones back) are "
                            "understood; edit the [histograms] section of "
                            "Cards/run_card.toml for anything else"
                            % (rest or "(no value)"))
                        return
                    if nlow in ("lhc", "lep", "ilc", "lcc") and rest:
                        try:
                            ecm = run_card.set_collider(nlow, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set %s collider: e_cm = %s GeV", nlow, ecm)
                        self.modified_card.add("run")
                        return
                    if nlow == "fixed_scale" and rest:
                        try:
                            val = run_card.set_fixed_scale(rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set fixed scales to %s GeV", val)
                        self.modified_card.add("run")
                        return

                    # the generic editor would say "Ambiguous key 'pdf'"
                    # without naming the per-beam entries
                    if nlow == "pdf" and rest:
                        self.invalid_set("Ambiguous key 'pdf' — use beam.pdf (both beams), "
                                         "beam.pdf1, beam.pdf2 or systematics.pdf")
                        return
                    # custom function parameters are not run_card keys, so the
                    # generic editor rejects them
                    if rest and nlow.startswith("custom_functions.") and nlow.count(".") == 2:
                        # the parameter name is matched as written by the
                        # custom function: keep its case
                        try:
                            run_card.set(name, run_card.evaluate(rest, masses), user=True)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (name, rest, error))
                            return
                        logger.info("modify parameter %s of the run_card.toml to %s", name, rest)
                        self.modified_card.add("run")
                        return

                    if rest and run_card.is_cut_name(name):
                        try:
                            cut, bound, val = run_card.set_cut(name, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            # an unknown bound, a value that is not a number,
                            # or a unit on a cut that is not an energy
                            self.invalid_set("invalid 'set %s %s': %s" % (name, rest, error))
                            return
                        logger.info("modify cut %s.%s of the run_card.toml to %s", cut, bound, val)
                        self.modified_card.add("run")
                        return

                    # legacy madevent cut names ("set ptj 20", "set mmll 200",
                    # "set etaj -1", ...) -> the [cuts] entry the LO->MG7
                    # run_card conversion maps them to (_LO_CUT_MAP), with the
                    # same "no cut" sentinels (minimum <= 0, maximum < 0).
                    if rest and nlow in run_card._LO_CUT_MAP:
                        if rest.lower().startswith("scan"):
                            self.invalid_set("invalid 'set %s %s': the [cuts] entries of "
                                             "the mg7 run_card.toml cannot be scanned"
                                             % (nlow, rest))
                            return
                        try:
                            cut, bound, val = run_card.set_lo_cut(nlow, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        if val is None:
                            logger.info("set %s (mg7 cut %s.%s) of the run_card.toml: "
                                        "no cut", nlow, cut, bound)
                        else:
                            logger.info("set %s (mg7 cut %s.%s) of the run_card.toml to %s",
                                        nlow, cut, bound, val)
                        self.modified_card.add("run")
                        return

                    # madevent parameters that are not a plain rename: the
                    # value is translated ("set dynamical_scale_choice 3") or
                    # several of them make one mg7 entry (fixed_fac_scale1 +
                    # fixed_fac_scale2 -> beam.fixed_fact_scale). mg7 values
                    # ("set sde_strategy denominators") come back as None and
                    # go on to the generic editor.
                    if rest and nlow in run_card._LO_SET_PARAMS:
                        try:
                            done = run_card.set_lo_param(nlow, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        if done is not None:
                            changes, warnings = done
                            for key, val in changes.items():
                                if val is None:
                                    logger.info("set %s: removed %s from the run_card.toml",
                                                nlow, key)
                                else:
                                    logger.info("set %s (mg7 %s) of the run_card.toml to %s",
                                                nlow, key, val)
                            for text in warnings:
                                logger.warning(text)
                            if changes:
                                self.modified_card.add("run")
                            elif not warnings:
                                logger.info("set %s %s: nothing to change in the run_card.toml",
                                            nlow, rest)
                            return

                    # 'iseed' is the madevent spelling of the mg7 run.seed,
                    # but the two disagree on how to ask for a random seed:
                    # madevent uses 0, mg7 uses -1 (0 being a perfectly good
                    # fixed seed there). Translate here rather than in
                    # _LO_SCALAR_MAP, which also drives the LO -> MG7 run_card
                    # conversion and would silently pin those runs to seed 0.
                    if rest and nlow == 'iseed':
                        try:
                            value = int(run_card.evaluate(rest, masses, energy=False,
                                                          name=nlow))
                        except (TypeError, ValueError, _banner_mod.InvalidCmd):
                            value = None
                        if value is None:
                            self.invalid_set("invalid 'set iseed %s': not an integer" % rest)
                            return
                        seed = -1 if value == 0 else value
                        run_card.set('run.seed', seed, user=True)
                        logger.info("set iseed (mg7 run.seed) of the run_card.toml to %s%s",
                                    seed, " (random)" if seed == -1 else "")
                        self.modified_card.add("run")
                        return

                    # madevent beam settings (ebeam, lpp, pdlabel, lhaid, per
                    # beam or for both) onto the per-beam [beam] entries
                    if rest and nlow in run_card.lo_beam_params:
                        data_paths = lhapdf_index_paths() if nlow == "lhaid" else ()
                        try:
                            done = run_card.set_lo_beam_param(nlow, rest, masses, data_paths)
                        except _banner_mod.InvalidRunCard as error:
                            text = "'set %s %s': %s" % (name, rest, error)
                            # an unknown lhaid is reported, not rejected, in every mode
                            if nlow == "lhaid" and isinstance(error, _banner_mod.UnknownPDFSet):
                                logger.warning("ignoring " + text)
                            else:
                                self.invalid_set("invalid " + text)
                            return
                        logger.info("set %s %s: %s", nlow, rest, done)
                        self.modified_card.add("run")
                        return

                    # quantities the card stores per beam: set both beams
                    if rest and nlow in ("e_cm", "beam.e_cm", "beam.pdf"):
                        try:
                            value = rest if nlow == "beam.pdf" else run_card.evaluate(rest, masses)
                            run_card.set(nlow, value, user=True)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set %s of the run_card.toml to %s (both beams)",
                                    nlow, run_card["beam." + nlow.split(".")[-1]])
                        self.modified_card.add("run")
                        return

                    # legacy madevent run_card names -> mg7 "section.key", so a
                    # madevent-style launch script ("set nevents 500", "set
                    # use_syst F", "set bwcutoff 10", ...) edits the mg7
                    # run_card.toml verbatim. RunCardMG7._LO_SCALAR_MAP is the
                    # same rename table used by the LO->MG7 run_card conversion.
                    if rest and nlow in run_card._LO_SCALAR_MAP \
                            and nlow not in [k.lower() for k in run_card.keys()]:
                        target = run_card._LO_SCALAR_MAP[nlow]
                        try:
                            run_card.set(target, rest, user=True)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set %s (mg7 %s) of the run_card.toml to %s",
                                    nlow, target, run_card[target])
                        self.modified_card.add("run")
                        return

                    if rest and nlow in [k.lower() for k in run_card.keys()]:
                        current = run_card[nlow]
                        if isinstance(current, (int, float)) and not isinstance(current, bool):
                            try:
                                resolved = run_card.evaluate(
                                    rest, masses, name=nlow,
                                    energy=nlow in run_card.energy_parameter)
                            except _banner_mod.InvalidCmd as error:
                                self.invalid_set("invalid 'set %s %s': %s" % (name, rest, error))
                                return
                            if not isinstance(resolved, str):
                                prefix = "run_card " if start == 1 else ""
                                line = "%s%s %s" % (prefix, name, resolved)
            return super().do_set(line, *args, **kwargs)

    return MG7Selector, (mother if mother is not None else MG7Cmd())


def ask_edit_cards(mother=None) -> dict:
    """Single (MadDM-style) question letting the user both pick which programs
    to run after generation and edit the associated cards.  Returns the switch
    dict describing the selected tools.

    ``mother`` is the :class:`MG7Cmd` asking the question. Passing the instance
    MG5 registered as its child is what lets a scripted ``launch`` answer this
    question from MG5's own command file.
    """

    selector_class, mother = build_selector_cmd(mother)
    # path_msg is what makes Cmd.check_answer_in_input_file accept a bare path as
    # an answer (its "elif path:" branch), so that a scripted launch can hand the
    # question a card/banner path -- as the question itself advertises -- and have
    # it replace the corresponding card. Without it the path is rejected ("This
    # answer is not valid for current question") and the default is used instead.
    try:
        switch, question = mother.ask('', '0', [], path_msg='enter path',
                                      ask_class=selector_class,
                                      mode='auto', line_args=[], force=False,
                                      return_instance=True)
    except BaseException:
        # interrupted (e.g. ctrl-C) before the prune below could run: prune
        # here too, using the instance ask() stashed pre-interrupt, or the
        # eagerly-materialised tool cards get misread as "left ON" next time.
        instance = getattr(mother, '_last_ask_instance', None)
        if instance is not None and hasattr(instance, 'active_cards'):
            try:
                mother.keep_cards(instance.active_cards())
            except Exception as error:
                logger.debug('could not prune tool cards after aborted launch question: %s', error)
        raise
    switch = dict(switch)
    prune_unselected_tool_cards(mother, question, switch)
    return switch


def prune_unselected_tool_cards(mother, question, switch) -> None:
    """Hide the cards of the tools that were NOT selected in the question.

    The question materialises every candidate tool card (copying the
    *_default.dat) so that it can offer them all for edition; madevent's
    ask_run_configuration prunes them again afterwards (keep_cards on the
    selected cards), but the mg7 launcher had no such cleanup. The materialised
    cards therefore persisted, and a later re-launch of the same output saw them
    all present and defaulted every switch to ON (set_default_<tool> keys off
    card presence). Prune here too, using the question's own switch->card map, so
    a re-launch reflects the previous selection instead of turning everything on.
    """
    try:
        keep = list(question.always_cards)
        for spec in question.switch_cards:
            if spec['on'](switch):
                keep.append(spec['card'])
        mother.keep_cards(keep)
    except Exception as error:
        logger.debug('could not prune unselected tool cards: %s', error)


def _find_event_file(run_path):
    """Locate the LHE event file produced by generate_events (if any)."""
    for name in ("events.lhe", "events.lhe.gz"):
        path = os.path.join(run_path, name)
        if os.path.exists(path):
            return path
    return None


def _find_or_convert_event_file(run_path, log):
    """The run's LHE event file, converting its npy events when that is all
    the run wrote (see npy_to_lhe)."""
    lhe_path = _find_event_file(run_path)
    if lhe_path is None and npy_to_lhe.can_convert(run_path):
        log.info("No LHE event file in %s: converting the npy events.", run_path)
        try:
            lhe_path = npy_to_lhe.convert(run_path)
        except Exception as error:
            _report_failure(log, "npy->LHE conversion", error, run_path)
    return lhe_path


def _report_failure(log, what, error, directory=None):
    """Log a post-processing failure and write the full traceback to a file
    (whose path is printed) so the problem can be investigated."""
    import traceback
    if not directory or not os.path.isdir(directory):
        directory = os.getcwd()
    path = os.path.join(directory, "%s_crash.log" % what.replace(" ", "_"))
    try:
        with open(path, "w") as fsock:
            fsock.write(traceback.format_exc())
    except Exception:
        path = None
    log.warning("%s failed: %s", what, error)
    if path:
        log.warning("full traceback written to: %s", path)


def _off(value) -> bool:
    """True when a switch value means the tool was not selected."""
    return value in (None, "OFF", "Not Avail.", "Not Avail. (numpy missing)")


def _has_effective_handler(lg) -> bool:
    """True when ``lg`` (or an ancestor it propagates to) already has a handler,
    i.e. some other party is already displaying these records."""
    while lg:
        if lg.handlers:
            return True
        if not lg.propagate:
            return False
        lg = lg.parent
    return False


_TOOL_LOGGING_READY = False


def _setup_logging():
    """Make the reused madevent tool drivers' progress visible on screen.

    The drivers already narrate what they do through logger.info /
    update_status ("Running Pythia8 [arXiv:...]", "Splitting .lhe event
    file...", the live Idle/Running/Completed job counters, "Running
    MadSpin", ...), but the standalone mg7 launcher never attaches a handler
    to the 'madgraph'/'madevent' loggers, so all of it is swallowed. Attach a
    colored INFO console handler to them (idempotent)."""
    global _TOOL_LOGGING_READY
    if _TOOL_LOGGING_READY:
        return
    if _has_effective_handler(logging.getLogger("madgraph")):
        # Someone already configured logging for us -- in practice MG5, which
        # now runs the launch in its own process. Its configuration is
        # authoritative (it owns stdout_level and the tutorial logger); adding
        # a second handler here would print every line twice.
        _TOOL_LOGGING_READY = True
        return
    try:
        import madgraph.interface.coloring_logging  # registers ColorFormatter
        formatter = logging.ColorFormatter("%(message)s")
    except Exception:
        formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler._mg7_tool_handler = True
    for name in ("madgraph", "madevent", "cmdprint", "madgraph7"):
        lg = logging.getLogger(name)
        if not any(getattr(h, "_mg7_tool_handler", False) for h in lg.handlers):
            lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    _TOOL_LOGGING_READY = True


def selected_tools(switch) -> list:
    """The post-processing drivers run_selected_tools runs for this switch."""
    # MadAnalysis5 hadron level analyses the shower/detector output, so it only
    # makes sense when a shower ran (mirrors madevent's card gating:
    # analysis == 'MadAnalysis5' and shower != 'OFF').
    ma5 = switch.get("analysis") == "MadAnalysis5"
    showered = not _off(switch.get("shower"))
    # reweighting is not one of them: run_single runs it (run_reweighting)
    return [t for t, on in (
        ("MadSpin", not _off(switch.get("madspin"))),
        ("MadAnalysis5 (parton level)", ma5),
        ("Pythia8 shower", switch.get("shower") == "Pythia8"),
        ("Delphes", switch.get("detector") == "Delphes"),
        ("MadAnalysis5 (hadron level)", ma5 and showered),
        ("Rivet", switch.get("analysis") == "Rivet"),
    ) if on]


def run_selected_tools(switch, process) -> None:
    """Run the optional post-processing programs selected in the merged
    question on the generated events.

    All the tools are driven through :class:`MG7RunCmd`, a thin adapter around
    the standard madevent run interface (``MadEventCmd``). This means the mg7
    output reuses the *exact* madevent tool drivers -- in particular Pythia8 is
    showered in parallel over the cluster/multicore backend, identically to a
    madevent run -- instead of re-implementing each tool here. The command
    sequence mirrors ``MadEventCmd.do_launch`` (MadSpin, MA5 parton, shower,
    Delphes, MA5 hadron, Rivet); each tool checks its own card, so a tool is
    invoked only when its switch is on. Reweighting, which comes first in
    that sequence, already ran (run_reweighting, from run_single).
    """
    log = logging.getLogger("madevent")

    tools = selected_tools(switch)
    if not tools:
        # A switch that is not off does not necessarily select a driver here:
        # "Not Avail." is not off, analysis = ExRoot has no mg7 driver, and a
        # shower switch set to something other than Pythia8 selects nothing
        # either. Without this a plain generate/output/launch announced a
        # post-processing step with nothing after the colon, and paid for
        # building the run interface (or an npy->LHE conversion) to do nothing.
        return

    lhe_path = _find_or_convert_event_file(process.run_path, log)
    if lhe_path is None:
        log.warning("No LHE event file in %s; cannot run %s.",
                    process.run_path, ", ".join(tools))
        return

    run_name = os.path.basename(os.path.dirname(os.path.abspath(lhe_path)))
    run_dir = os.path.dirname(os.path.abspath(lhe_path))

    ma5 = switch.get("analysis") == "MadAnalysis5"
    showered = not _off(switch.get("shower"))

    log.info("")
    log.info("Post-processing the generated events with: %s", ", ".join(tools))

    try:
        from madgraph.iolibs.template_files.mg7.run_interface import MG7RunCmd
        cmd = MG7RunCmd(os.getcwd(), load_mg5_options(), run_name, lhe_path)
        cmd.set_run_name(run_name, None, "parton")
    except Exception as error:
        _report_failure(log, "post-processing setup", error, run_dir)
        return

    def run(tool, command):
        bar = "=" * 60
        log.info("")
        log.info(bar)
        log.info("  post-processing step: %s", tool)
        log.info(bar)
        start = time.time()
        try:
            cmd.exec_cmd(command, postcmd=False, printcmd=False)
        except Exception as error:
            _report_failure(log, tool, error, run_dir)
        else:
            log.info("  -> %s done (%.1fs)", tool, time.time() - start)

    # Same order as MadEventCmd.do_launch. The commands take no run name and so
    # act on cmd.run_name: decay_events (MadSpin) repoints cmd.run_name to the
    # "<run>_decayed_i" run, so the shower and the analyses that follow act on
    # the decayed events -- exactly as a madevent run does. Each driver still
    # guards on its own card.
    if not _off(switch.get("madspin")):
        run("MadSpin", "decay_events -from_cards")
    if ma5:
        run("MadAnalysis5 (parton)", "madanalysis5_parton --no_default")
    if switch.get("shower") == "Pythia8":
        run("Pythia8 shower", "shower --no_default")
    if switch.get("detector") == "Delphes":
        run("Delphes", "delphes --no_default")
    # hadron-level MA5 needs the shower/detector output: only run it if a shower
    # was requested (otherwise there is no hadron-level event file to analyse).
    if ma5 and showered:
        run("MadAnalysis5 (hadron)", "madanalysis5_hadron --no_default")
    if switch.get("analysis") == "Rivet":
        run("Rivet", "rivet --no_default")

    # Finalize the run exactly as MadEventCmd.do_launch does after the
    # shower/detector/analysis tools: store_result() processes the deferred
    # "to_store" actions -- in particular it gzips the Pythia8 HepMC to
    # <tag>_pythia8_events.hepmc.gz (the pythia8_card default is HEPMCoutput:file
    # = hepmc.gz). The deferred Rivet job reads exactly that .gz path, so this
    # must run before the Rivet/Contur postprocessor below.
    try:
        cmd.store_result()
    except Exception as error:
        _report_failure(log, "store_result", error, run_dir)

    if switch.get("analysis") == "Rivet":
        # do_rivet only *prepared* the run (wrote Events/<run>/run_rivet.sh) and
        # deferred execution to the postprocessor -- the rivet_card default has
        # run_rivet_later = True, so with --no_default it logged "Skipping Rivet
        # for now, passing it to postprocessor" and appended the run to
        # cmd.postprocessing_dirs. MadEventCmd.do_launch runs that postprocessor
        # at the end of a run; do the same here so the .yoda (and, when
        # run_contur = True, the Contur limits) are actually produced.
        bar = "=" * 60
        log.info("")
        log.info(bar)
        log.info("  post-processing step: Rivet/Contur postprocessing")
        log.info(bar)
        start = time.time()
        try:
            cmd.postprocessing()
        except Exception as error:
            _report_failure(log, "Rivet-Contur postprocessing", error, run_dir)
        else:
            log.info("  -> Rivet/Contur postprocessing done (%.1fs)",
                     time.time() - start)


def _add_time_of_flight(lhe_path, threshold, param_card_path, log):
    """Add invariant-lifetime (vtim) information to the LHE events, drawing each
    unstable particle's decay length from its width (in mm). Mirrors
    common_run_interface.CommonRunCmd.do_add_time_of_flight but reads the widths
    from the local param_card instead of the LHE banner."""
    import random
    try:
        import madgraph.various.lhe_parser as lhe_parser
    except ImportError:
        import internal.lhe_parser as lhe_parser
    from madgraph.models import check_param_card as param_card_mod
    from madgraph.various import misc as _misc
    import madgraph.iolibs.files as files

    need_zip = lhe_path.endswith('.gz')
    if need_zip:
        _misc.gunzip(lhe_path)
        lhe_path = lhe_path[:-3]

    param_card = param_card_mod.ParamCard(param_card_path)
    cst = 6.58211915e-25   # hbar in GeV s
    c = 299792458000       # speed of light in mm/s
    log.info('Adding time of flight information on %s', lhe_path)
    lhe = lhe_parser.EventFile(lhe_path)
    out = open('%s_2vertex.lhe' % lhe_path, 'w')
    out.write(lhe.banner)
    for event in lhe:
        for particle in event:
            # default=0 -> particles without a decay entry are treated as stable
            width = param_card['decay'].get((abs(particle.pid),), 0.).value
            if width:
                vtim = c * random.expovariate(width / cst)
                if vtim > threshold:
                    particle.vtim = vtim
        out.write(str(event))
    out.write('</LesHouchesEvents>\n')
    out.close()
    lhe.close()
    files.mv('%s_2vertex.lhe' % lhe_path, lhe_path)
    if need_zip:
        _misc.gzip(lhe_path)


def _lhapdf_config_path():
    """Path to lhapdf-config so systematics can import the python lhapdf
    module (required to compute PDF/scale variations)."""
    return lhapdf_paths().config


def _run_systematics(lhe_path, cfg, log):
    """Run systematics.py (scale/PDF variations) on the LHE file, reusing the
    exact worker madevent uses (systematics.call_systematics)."""
    try:
        import madgraph.various.systematics as systematics
    except ImportError:
        import internal.systematics as systematics

    def fmt(vals):
        return ','.join(str(v) for v in vals)

    opts = []
    if cfg.get('systematics_mur'):
        opts.append('--mur=%s' % fmt(cfg['systematics_mur']))
    if cfg.get('systematics_muf'):
        opts.append('--muf=%s' % fmt(cfg['systematics_muf']))
    if cfg.get('systematics_pdf'):
        opts.append('--pdf=%s' % fmt(cfg['systematics_pdf']))
    extra = cfg.get('systematics_str_options', '')
    if extra:
        opts.extend(extra.split())
    # The mg7 LHE has no <mgrwt> block, so systematics cannot read the per-event
    # LO reweighting info. For a process with a single alpha_s power (stored in
    # SubProcesses/proc_characteristics at output time) it can be reconstructed
    # from the events: pass that power as --lo_nqcd. -1 means the QCD power is
    # not uniform, so the reconstruction is not applicable.
    if not any(o.startswith('--lo_nqcd') for o in opts):
        try:
            from madgraph.various import banner as _banner_mod
        except ImportError:
            import internal.banner as _banner_mod
        pc_path = os.path.join('SubProcesses', 'proc_characteristics')
        if os.path.exists(pc_path):
            try:
                nqcd = int(_banner_mod.ProcCharacteristic(pc_path)['single_qcd_order'])
            except Exception:
                nqcd = -1
            if nqcd >= 0:
                opts.append('--lo_nqcd=%d' % nqcd)
    # tell systematics where to find lhapdf (so it can link the python module)
    if not any(o.startswith('--lhapdf_config') for o in opts):
        lhapdf_config = _lhapdf_config_path()
        if lhapdf_config:
            opts.append('--lhapdf_config=%s' % lhapdf_config)

    log.info('Running systematics on %s %s', lhe_path, ' '.join(opts))
    systematics.call_systematics([lhe_path, lhe_path] + opts,
                                 log=lambda x: log.info(str(x)))


def run_lhe_postprocessing(process) -> None:
    """Run the LHE-level post-processings configured in the run_card
    [postprocessing] section on the generated event file (displaced-vertex
    time-of-flight and systematics). Only applies when an LHE file exists."""
    lhe_path = _find_event_file(process.run_path)
    if lhe_path is None:
        return
    # process.run_card is a RunCardMG7; the section view exposes .get(key, def)
    try:
        cfg = process.run_card["postprocessing"]
    except Exception:
        return
    log = logging.getLogger('madevent')

    if cfg.get('systematics') and getattr(process, 'is_decay', False):
        # Scale and PDF variations are a beam quantity. A decay has neither, so
        # systematics can only fail here ("not supported for pdlabel=none") --
        # and it is not free: MadSpin reruns the launcher for every pool refill.
        log.info("decay mode: skipping systematics (no beams to vary)")
    elif cfg.get('systematics'):
        native = getattr(process, 'systematics', None)
        if native is not None and native.weight_count > 0:
            log.warning("[postprocessing] systematics is ignored: the scale/PDF "
                        "weights were already computed by madspace ([systematics] "
                        "section); set systematics.enable = false to use the legacy "
                        "systematics.py path instead.")
        else:
            try:
                _run_systematics(lhe_path, cfg, log)
            except Exception as error:
                _report_failure(log, "systematics computation", error,
                                os.path.dirname(lhe_path))

    tof = cfg.get('time_of_flight', -1.0)
    try:
        if tof is not None and float(tof) >= 0:
            _add_time_of_flight(lhe_path, float(tof), process.param_card_path, log)
    except Exception as error:
        _report_failure(log, "add_time_of_flight", error,
                        os.path.dirname(lhe_path))


_model_hash_warned = set()
_model_cache = {}


def read_stored_model(me_dir=None) -> "str | None":
    """The model this process was generated with, as recorded at output time in
    ``SubProcesses/model.txt``: a UFO path carrying its restriction (e.g.
    ``.../models/sm-no_b_mass``) or a model name -- in either case something
    ``import model`` understands. Returns None when the process was written
    without that information.

    Warns (once per directory) when the model on disk no longer matches the
    hash stored on the second line, i.e. it changed since the output was made.
    """
    base = me_dir or os.getcwd()
    model_file = os.path.join(base, "SubProcesses", "model.txt")
    if not os.path.exists(model_file):
        return None
    with open(model_file) as f:
        lines = f.read().splitlines()
    model = lines[0].strip() if lines else ""
    stored_hash = lines[1].strip() if len(lines) > 1 else ""
    if not model:
        logger.warning("SubProcesses/model.txt is empty; the model of this "
                       "process is unknown.")
        return None

    # verify the model on disk still matches the one used at output time. The
    # reference may carry a restriction suffix ('sm-no_b_mass'); the hash was
    # taken on the UFO directory itself ('sm').
    model_dir = model if os.path.isdir(model) else model.rsplit('-', 1)[0]
    if stored_hash and os.path.isdir(model_dir) \
            and model_file not in _model_hash_warned:
        current_hash = misc.hash_model_files(model_dir)
        if current_hash and current_hash != stored_hash:
            _model_hash_warned.add(model_file)
            logger.warning(
                "The model at %s has changed since this process was generated "
                "(hash mismatch); anything computed from it now (the 'auto' "
                "widths, the dependent masses/widths of the param_card) may be "
                "inconsistent with the matrix element.", model_dir)
    return model


def _proc_characteristic(me_dir=None) -> dict:
    """SubProcesses/proc_characteristics as a plain dict (empty when the file is
    missing or unreadable). Plain, because ProcCharacteristic is a ConfigFile,
    whose get() is the parameter accessor and takes no default."""
    from madgraph.various import banner as _banner_mod
    path = os.path.join(me_dir or os.getcwd(), 'SubProcesses',
                        'proc_characteristics')
    if os.path.exists(path):
        try:
            return dict(_banner_mod.ProcCharacteristic(path))
        except Exception as error:
            logger.debug("could not read %s: %s", path, error)
    return {}


def load_process_model(me_dir=None):
    """Import the UFO model this process was generated with, as
    :func:`madgraph.models.import_ufo.import_model` builds it -- the object every
    model-level consumer of the param_card expects (see
    :meth:`MG7Cmd.get_model`). None when the process recorded no model, or when
    that model no longer imports.

    Cached per (reference, complex-mass-scheme): importing a model is slow and
    the card question asks for it again at every 'update dependent'."""
    ref = read_stored_model(me_dir)
    if not ref:
        return None
    cms = _proc_characteristic(me_dir).get('complex_mass_scheme', False)
    key = (ref, bool(cms))
    if key not in _model_cache:
        import madgraph.models.import_ufo as import_ufo
        try:
            with misc.MuteLogger(['madgraph.model'], [50]):
                _model_cache[key] = import_ufo.import_model(
                    ref, complex_mass_scheme=bool(cms))
        except Exception as error:
            # 'update dependent' calls this under a SIGALRM whose handler
            # raises a TimeOutError of its own (a class defined inside the
            # caller, so it can only be recognised by name). Let it through:
            # the caller then says the model took too long to load and how to
            # force it, instead of reporting a model that does not import.
            if type(error).__name__ == 'TimeOutError':
                raise
            logger.debug("could not import the model %s: %s", ref, error)
            _model_cache[key] = None
    return _model_cache[key]


def compute_auto_widths(param_card_path=os.path.join("Cards", "param_card.dat")) -> None:
    """Fill any width set to ``auto`` in the param_card, using madgraph and the
    model stored at output time (``SubProcesses/model.txt``), and write the
    result back into the card. A no-op when the card has no ``auto`` width.

    Called before every generation, so a single run and each scan point (whose
    param_card is rewritten just before ``run_single``) both get model-computed
    widths."""
    try:
        with open(param_card_path) as f:
            text = f.read()
    except OSError:
        return
    # "DECAY <pdg> auto", with the options of compute_widths that may follow
    # ("auto@NLO", "auto --nlo --body_decay=3"), as in
    # common_run_interface.static_check_param_card: one compute_widths per set of
    # options
    groups = group_auto_widths(text)
    if not groups:
        return
    pdgs = [pdg for group in groups.values() for pdg in group]

    model = read_stored_model()
    if not model:
        logger.warning(
            "The param_card requests 'auto' width(s) for %s but the model of "
            "this process is unknown; leaving them as-is.", " ".join(pdgs))
        return

    mg5 = str(_MG_ROOT / "bin" / "madgraph")
    if not os.path.exists(mg5):
        logger.warning("Cannot find madgraph at %s; 'auto' widths not computed.", mg5)
        return

    import tempfile
    cmds = "import model %s\n" % model
    for options, group in groups.items():
        cmds += "compute_widths %s --path=%s %s\n" % (
            " ".join(group), os.path.abspath(param_card_path), " ".join(options))
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as fh:
        fh.write(cmds)
        cmdfile = fh.name
    try:
        logger.info("Computing 'auto' width(s) for %s ...", " ".join(pdgs))
        proc = subprocess.run([mg5, cmdfile])
        if proc.returncode != 0:
            logger.warning(
                "compute_widths returned a non-zero exit code; the param_card "
                "may still contain 'auto' entries.")
    finally:
        try:
            os.remove(cmdfile)
        except OSError:
            pass


def compute_decay_widths(
    me_dir: str = ".", precision: float = 0.01, nb_core: int | None = None,
    extra_make_args=(),
) -> list[dict]:
    """Integrate the decays of an mg7 decay directory and return their widths.

    This is the madspace counterpart of madevent's ``survey decay``: only the
    survey runs (no unweighting, no events, no systematics), and it stops once
    every channel is below ``precision`` relative error, like the
    ``--precision_channel`` of ``compute_widths``.

    ``me_dir`` is an output directory of 1 -> n processes of one decaying
    particle. It is meant to run in a process of its own (``bin/generate_events
    --decay-widths=FILE``, which is how ``compute_widths`` calls it): the
    madspace objects write their status file from a destructor, and the
    caller removes the temporary directory as soon as the widths are back.
    The result holds one entry per decay and flavour assignment::

        {"incoming": 25, "outgoing": [5, -5], "width": 4.3e-3, "error": 1e-9}

    (signed pdg ids, width in GeV). The flavours of a subprocess (different
    matrix elements, e.g. the u u~ g and d d~ g of z > q q~ g) are integrated
    separately (MadgraphProcess.split_decay_flavors); the assignments of one
    flavour share its matrix element (they are the same function of the
    momenta, or they would not be grouped) and split its width evenly. A
    decay the decaying particle is too light for has no entry.
    """
    cwd = os.getcwd()
    os.chdir(me_dir)
    process = None
    try:
        card_path = os.path.join("Cards", "run_card.toml")
        card = RunCardMG7(card_path, consistency=False)
        card["generation"]["survey_target_precision"] = precision
        card["generation"]["survey_max_iters"] = 30
        # one survey result per decay, not per group of mergeable ones
        card["phasespace"]["merge_subprocesses"] = False
        card["run"]["verbosity"] = "silent"
        if nb_core:
            card["run"]["cpu_thread_pool_size"] = int(nb_core)
        card.write(card_path)

        # one survey result per flavour, not per subprocess
        process = MadgraphProcess(extra_make_args=extra_make_args,
                                  split_flavors=True)
        if not process.is_decay:
            raise ValueError("compute_decay_widths needs a decay (1 -> n) output")
        process.survey()

        channel_status = process.event_generator.channel_status()
        results = []
        offset = 0
        for subproc, phasespace in zip(process.subprocesses, process.phasespaces):
            statuses = channel_status[offset:offset + len(phasespace.channels)]
            offset += len(phasespace.channels)
            width = sum(status.mean for status in statuses)
            error = sum(status.error ** 2 for status in statuses) ** 0.5
            options = _decay_flavor_options(subproc.meta)
            for option in options:
                results.append({
                    "incoming": option[0],
                    "outgoing": list(option[1:]),
                    "width": width / len(options),
                    "error": error / len(options),
                })
        return results
    finally:
        if process is not None:
            # StatusFile finishes info.json from its destructor: run it before
            # the caller removes the directory (see init_event_dir)
            _release_channel_generators(process)
            del process
            gc.collect()
        os.chdir(cwd)


def _single_flavor_metas(meta: dict) -> list[dict]:
    """The subprocess ``meta`` as one subprocess per flavour (see
    MadgraphProcess.split_decay_flavors).

    Each copy keeps the flavour (its matrix-element index and its assignments)
    and, in every channel, the diagrams that flavour is active in, renumbered
    to its only flavour, 0. A channel left without diagrams is dropped, and a
    flavour left without channels (no diagram at all) has no subprocess. A
    subprocess of one flavour is returned as it is.
    """
    flavors = meta["flavors"]
    if len(flavors) == 1:
        return [meta]
    singles = []
    for position, flavor in enumerate(flavors):
        channels = []
        for channel in meta["channels"]:
            diagrams = [
                dict(diagram, active_flavors=[0])
                for diagram in channel["diagrams"]
                if position in diagram["active_flavors"]
            ]
            if diagrams:
                channels.append(dict(channel, diagrams=diagrams))
        if channels:
            singles.append(dict(meta, flavors=[flavor], channels=channels))
    return singles


def _decay_flavor_options(meta: dict) -> list[list[int]]:
    """The flavour assignments of a subprocess of compute_decay_widths, which
    share its width evenly.

    That is right for the assignments of one flavour, which have one matrix
    element (d d~ and s s~ of a massless s, the three neutrinos). Different
    flavours have different matrix elements (the u u~ and d d~ of a
    flavour-grouped subprocess) and the survey of a subprocess only gives
    their sum: compute_decay_widths splits them first (split_flavors), and a
    subprocess still holding several is an error, not an even split that
    would get the branching ratios wrong.
    """
    flavors = meta["flavors"]
    if len(flavors) != 1:
        raise ValueError(
            "compute_decay_widths: the subprocess %s has %d flavours with "
            "different matrix elements, the survey cannot give the width of "
            "each" % (meta.get("path", "?"), len(flavors))
        )
    return [list(option) for option in flavors[0]["options"]]


def _release_channel_generators(process) -> None:
    """Drop the per-channel event/weight generators so their intermediate
    .npy files are deleted right away. madspace only frees them when the
    owning Python objects are collected, and process.event_generator /
    process.phasespaces sit in a reference cycle that plain refcounting
    never breaks -- only an explicit gc.collect() does, promptly, instead of
    leaving it to whenever the cyclic GC next runs on its own."""
    process.event_generator = None
    process.phasespaces = None
    gc.collect()
    if getattr(process, "auto_temp_path", False):
        shutil.rmtree(process.temp_path, ignore_errors=True)


REWEIGHT_CARD = os.path.join("Cards", "reweight_card.dat")


class ExistingRun:
    """The run directory Events/<run_name> of an earlier generation, with
    what run_reweighting() needs of a MadgraphProcess: run_path, events_path
    (the file the generation wrote: events.npy for the npy formats, else
    events.lhe[.gz]) and run_card["run"]["output_format"] (from the files)."""

    def __init__(self, run_name, events_dir="Events"):
        self.run_path = os.path.abspath(os.path.join(events_dir, run_name))
        if not os.path.isdir(self.run_path):
            runs = list_runs(events_dir)
            raise ValueError("no run '%s' in %s%s" % (
                run_name, events_dir,
                "; runs: %s" % ", ".join(runs) if runs else ""))
        npy = os.path.join(self.run_path, npy_to_lhe.EVENT_FILE)
        if os.path.isfile(npy):
            self.events_path = npy
            compact = os.path.isfile(os.path.join(self.run_path,
                                                  npy_to_lhe.COMPLETER_FILE))
            output_format = "compact_npy" if compact else "lhe_npy"
        else:
            self.events_path = _find_event_file(self.run_path)
            output_format = "lhe"
        if self.events_path is None:
            raise ValueError("no event file (events.npy, events.lhe[.gz]) in %s"
                             % self.run_path)
        self.run_card = {"run": {"output_format": output_format}}


def list_runs(events_dir="Events") -> list:
    """The runs of events_dir with an event file, sorted by name."""
    if not os.path.isdir(events_dir):
        return []
    return sorted(
        name for name in os.listdir(events_dir)
        if any(os.path.isfile(os.path.join(events_dir, name, f)) for f in
               (npy_to_lhe.EVENT_FILE, "events.lhe", "events.lhe.gz")))


def install_reweight_card(path) -> None:
    """``--reweight=CARD``: CARD becomes Cards/reweight_card.dat. A different
    card already there is kept as reweight_card.dat.bak."""
    if os.path.exists(REWEIGHT_CARD):
        if os.path.samefile(path, REWEIGHT_CARD):
            return
        with open(path) as new, open(REWEIGHT_CARD) as old:
            if new.read() == old.read():
                return
        shutil.copy(REWEIGHT_CARD, REWEIGHT_CARD + ".bak")
        logger.info("--reweight: %s saved as %s.bak", REWEIGHT_CARD, REWEIGHT_CARD)
    shutil.copy(path, REWEIGHT_CARD)
    logger.info("--reweight: using %s as %s", path, REWEIGHT_CARD)


# reweight switch values naming a backend (ON is auto)
_SWITCH_BACKENDS = {"classic": "classic", "density": "classic", "madtrex": "madtrex"}


def prepare_reweighting(switch, cli_backend=None, options=None):
    """The reweight_backend.Decision for this run, or None when it does not
    reweight: the reweight switch is off, or Cards/reweight_card.dat is
    missing or has no "launch". Chosen before the generation, so that a card
    that cannot run fails before the events are generated, and the event
    format the backend needs is known (force_lhe_output_if_needed).

    Precedence: cli_backend (--reweight-backend), then a switch value naming
    a backend (classic/density/madtrex; ON is auto), then the card's
    "change backend" line; auto otherwise."""
    from madgraph.various import reweight_backend as rb

    value = (switch or {}).get("reweight")
    if _off(value) or not os.path.exists(REWEIGHT_CARD):
        return None
    with open(REWEIGHT_CARD) as f:
        traits = rb.classify_card(f.read())
    if not traits.launches:
        logger.info("reweighting is on but %s has no 'launch': nothing to reweight",
                    REWEIGHT_CARD)
        return None
    plugin = value if value not in ("ON", "classic", "madtrex", "density") else None
    # a plugin is a classic reweighting module
    switch_backend = "classic" if plugin else _SWITCH_BACKENDS.get(value)
    requested = [("--reweight-backend", cli_backend),
                 ("the reweight switch (%s)" % value, switch_backend)]
    if value == "density":
        traits.density = True
    if options is None:
        options = load_mg5_options()
    env = rb.probe_environment(os.getcwd(), options)
    decision = rb.select_backend(requested, traits, env, plugin=plugin)
    for text in decision.warnings:
        logger.warning(text)
    return decision


@contextlib.contextmanager
def _stripped_reweight_card(path):
    """path without its "change backend" line for the duration (the classic
    module reads the card itself); restored afterwards."""
    from madgraph.various import reweight_backend as rb
    with open(path) as f:
        text = f.read()
    stripped = rb.strip_backend_line(text)
    if stripped == text:
        yield
        return
    with open(path, "w") as f:
        f.write(stripped)
    try:
        yield
    finally:
        with open(path, "w") as f:
            f.write(text)


# the <weightgroup name=...> each backend writes into the LHE header
REWEIGHT_WEIGHT_GROUPS = {"classic": "mg_reweighting", "madtrex": "tearex_reweighting"}


def write_reweighting_record(run_path, decision, card_path, events=None) -> str:
    """Record who reweighted the run, and why, in <run>/reweighting.json:
    backend, how it was chosen (source, reasons, warnings), the card kind,
    the event file the weights went into (events), the LHE weight group it
    wrote and the card itself. Returns the path."""
    with open(card_path) as f:
        card = f.read()
    record = {
        "backend": decision.backend,
        "source": decision.source,
        "reasons": [r.replace("{card}", os.path.basename(card_path))
                    for r in decision.reasons],
        "warnings": list(decision.warnings),
        "card_kind": decision.card_kind,
        "plugin": decision.plugin,
        "weight_group": REWEIGHT_WEIGHT_GROUPS[decision.backend],
        "events": events,
        "card": card,
    }
    path = os.path.join(run_path, "reweighting.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    return path


def drop_stored_matrix_elements(events_path) -> None:
    """Removes the "matrix_element" column (MadgraphProcess.
    store_matrix_elements) from the npy events at events_path, if they have
    one: it is only there for MadtRex's normalisation, not part of the
    run's output."""
    import numpy as np
    from numpy.lib import format as npy_format

    if not events_path.endswith(".npy") or not os.path.isfile(events_path):
        return
    events = np.load(events_path, mmap_mode="r", max_header_size=1 << 30)
    if "matrix_element" not in (events.dtype.names or ()):
        return
    names = [name for name in events.dtype.names if name != "matrix_element"]
    descr = [(name, events.dtype.fields[name][0]) for name in names]
    tmp = events_path + ".tmp"
    out = npy_format.open_memmap(tmp, mode="w+", dtype=np.dtype(descr), shape=events.shape)
    chunk = 100000
    for start in range(0, len(events), chunk):
        stop = min(start + chunk, len(events))
        for name in names:
            out[name][start:stop] = events[name][start:stop]
    out.flush()
    del out, events
    os.replace(tmp, events_path)


def run_reweighting(process, card_path=REWEIGHT_CARD, requested=None, decision=None):
    """Reweight the events of process with card_path, once, by the backend
    of decision (prepare_reweighting()), or of select_backend(requested)
    when no decision is given. The classic module runs through MG7RunCmd
    ("reweight -from_cards", on the LHE file); MadtRex through
    madtrex.run_reweighting(), on the events as written (lhe or npy)."""
    from madgraph.various import reweight_backend as rb

    log = logging.getLogger("madevent")
    if decision is None:
        with open(card_path) as f:
            traits = rb.classify_card(f.read())
        env = rb.probe_environment(os.getcwd(), load_mg5_options())
        decision = rb.select_backend(requested, traits, env)
    log.info(decision.describe(os.path.basename(card_path)))
    run_dir = process.run_path
    # both backends add their weights to the run's own event file (in place,
    # after the systematics weights): events.npy or events.lhe[.gz]

    if decision.backend == "madtrex":
        if process.run_card["run"]["output_format"] == "compact_npy":
            raise ValueError(
                "MadtRex reweighting requires run_card [run] output_format to be "
                "'lhe' or 'lhe_npy': 'compact_npy' discards the per-event flavor/"
                "helicity information MadtRex needs to match events to the "
                "matrix element library")
        from madgraph.iolibs.template_files.mg7.madtrex import (
            run_reweighting as madtrex_reweighting)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                card = os.path.join(tmp, "reweight_card.dat")
                with open(card_path) as f, open(card, "w") as out:
                    out.write(rb.strip_backend_line(f.read()))
                # the stored matrix elements leave events.npy with the
                # reweighting's last weight merge
                madtrex_reweighting(card, events_path=process.events_path,
                                    card_kind=decision.card_kind,
                                    process_directory=os.getcwd(),
                                    drop_stored_matrix_elements=True)
        finally:
            # and here only when that merge did not happen (a failure)
            drop_stored_matrix_elements(process.events_path)
        tool_copy = os.path.join(run_dir, "unweighted_events.lhe.gz")
        if not process.events_path.endswith(".npy") and os.path.exists(tool_copy):
            from madgraph.iolibs.template_files.mg7.run_interface import sync_lhe_copy
            sync_lhe_copy(process.events_path, tool_copy)
        log.info("reweighting weights written to %s", process.events_path)
        write_reweighting_record(run_dir, decision, card_path, process.events_path)
        return decision

    lhe_path = _find_or_convert_event_file(run_dir, log)
    if lhe_path is None:
        raise RuntimeError("No LHE event file in %s: the classic reweighting "
                           "module needs one" % run_dir)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(lhe_path)))
    from madgraph.iolibs.template_files.mg7.run_interface import MG7RunCmd
    cmd = MG7RunCmd(os.getcwd(), load_mg5_options(), run_name, lhe_path)
    cmd.set_run_name(run_name, None, "parton")
    command = "reweight --backend=classic --mode=%s -from_cards" % (
        "density" if decision.card_kind == "density" else "ON")
    if decision.plugin:
        command += " --plugin=%s" % decision.plugin
    with _stripped_reweight_card(card_path):
        cmd.exec_cmd(command, postcmd=False, printcmd=False)
    try:
        cmd.store_result()
    except Exception as error:
        _report_failure(log, "store_result", error, run_dir)
    # the classic module's weights into the run's event file (lhe_path)
    cmd.sync_reweighted_events()
    log.info("reweighting weights written to %s", lhe_path)
    write_reweighting_record(run_dir, decision, card_path, lhe_path)
    return decision


def lhe_via_binary(process, reweight) -> bool:
    """Whether an output_format = "lhe" run that MadtRex reweights is
    generated as npy ("lhe_npy") and turned into the LHE file it asked for
    only after the reweighting (deliver_lhe_from_binary()): MadtRex reads
    and extends the npy events much faster than an LHE file, which it would
    decompress, parse and rewrite in full. Not when an LHE-only step runs
    before the reweighting: the run_card [postprocessing] time_of_flight or
    legacy systematics (see run_lhe_postprocessing())."""
    if getattr(reweight, "backend", None) != "madtrex":
        return False
    if process.gridpack_run_mode != "regular":
        return False
    if process.run_card["run"]["output_format"] != "lhe":
        return False
    cfg = process.run_card["postprocessing"]
    try:
        if float(cfg.get("time_of_flight", -1.0)) >= 0:
            return False
    except (TypeError, ValueError):
        return False
    if cfg.get("systematics") and not process.run_card["systematics"]["enable"]:
        return False
    return True


def deliver_lhe_from_binary(process) -> None:
    """The LHE file of a run lhe_via_binary() generated as npy: npy_to_lhe
    writes Events/<run>/events.lhe.gz from events.npy and header.lhe, with
    their weights (the systematics', MadtRex's), and it replaces them as the
    run's event file (process.events_path, and reweighting.json's events).
    When the conversion fails the npy events are kept, and reported."""
    log = logging.getLogger("madevent")
    npy_path = process.events_path
    process.lhe_from_binary = False
    try:
        lhe_path = npy_to_lhe.convert(npy_path)
    except Exception as error:
        _report_failure(log, "npy->LHE conversion", error, process.run_path)
        log.warning("the events of %s are kept as %s", os.path.basename(process.run_path),
                    npy_path)
        return
    os.remove(npy_path)
    os.remove(os.path.join(process.run_path, npy_to_lhe.HEADER_FILE))
    process.events_path = lhe_path
    record = os.path.join(process.run_path, "reweighting.json")
    if os.path.isfile(record):
        with open(record) as f:
            content = json.load(f)
        if content.get("events") == npy_path:
            content["events"] = lhe_path
            with open(record, "w") as f:
                json.dump(content, f, indent=1)


def run_single(switch=None, reweight=None) -> "MadgraphProcess":
    """Run a single generation and return the process (for its result).
    reweight is the reweight_backend.Decision of prepare_reweighting(), or
    None: reweighting runs once, after the run_card post-processing
    (systematics) and before the post-processing tools."""
    compute_auto_widths()
    process = MadgraphProcess()
    process.store_matrix_elements = getattr(reweight, "backend", None) == "madtrex"
    process.survey()
    process.train_madnis()
    process.lhe_from_binary = lhe_via_binary(process, reweight)
    if process.lhe_from_binary:
        logging.getLogger("madevent").info(
            "MadtRex reweights this run: the events are written as npy, and as "
            "the requested LHE file once reweighted")
    process.generate_events("lhe_npy" if process.lhe_from_binary else None)
    if process.gridpack_run_mode != "regular":
        return process
    # run_card-driven LHE post-processing (displaced vertex + systematics)
    run_lhe_postprocessing(process)
    # reweighting, after the systematics, by the backend chosen beforehand.
    # Like the post-processing tools, a failure is reported without losing
    # the rest of the run (MadSpin, shower, analyses): the event file is only
    # ever replaced atomically, so the events are kept, without the weights.
    if reweight is not None:
        try:
            run_reweighting(process, REWEIGHT_CARD, decision=reweight)
        except Exception as error:
            log = logging.getLogger("madevent")
            _report_failure(log, "reweighting", error, process.run_path)
            log.warning(
                "the events of %s are kept without the reweighting weights; to "
                "reweight them once the problem is fixed: 'launch %s -i' in "
                "MadGraph, then 'reweight %s'", os.path.basename(process.run_path),
                os.getcwd(), os.path.basename(process.run_path))
    # the LHE file of an "lhe" run MadtRex reweighted as npy, with the
    # reweighting weights, or without them when the reweighting failed
    if process.lhe_from_binary:
        deliver_lhe_from_binary(process)
    # run the post-processing tools (Pythia8/Delphes/MadSpin/analysis)
    # selected in the merged question above on the generated events.
    if switch:
        run_selected_tools(switch, process)
    return process


def detect_run_scan(run_card_path):
    """Return a banner.RunCardIterator if the run_card contains scan:[...]
    values, else None."""
    from madgraph.various import banner as banner_mod
    from madgraph.various import misc as _misc
    with _misc.TMP_variable(banner_mod.RunCard, 'allow_scan', True):
        rc = banner_mod.RunCard(run_card_path, consistency=False)
    if getattr(rc, 'scan_set', None):
        return banner_mod.RunCardIterator(run_card_path)
    return None


def detect_param_scan(param_card_path):
    """Return a ParamCardIterator if the param_card contains scan:[...] values,
    else None."""
    if not os.path.exists(param_card_path):
        return None
    from madgraph.models import check_param_card as param_card_mod
    it = param_card_mod.ParamCardIterator(param_card_path)
    for block in it.order:
        for param in block:
            if isinstance(param.value, str) and param.value.strip().lower().startswith('scan'):
                return it
    return None


def run_scan(iterator, card_path, switch=None, reweight=None) -> None:
    """Iterate over all scan points, running a full generation for each and
    accumulating the results, then write the scan summary. Works for both the
    run_card (RunCardIterator) and the param_card (ParamCardIterator); their
    interface (__iter__/write/store_entry/get_next_name/write_summary) is the
    same. The scan card is restored afterwards. reweight, if given, is
    forwarded to run_single() so every scan point is reweighted in turn."""
    import tomllib
    with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
        run_name = tomllib.load(f).get("run", {}).get("run_name", "run")

    from madgraph.models import check_param_card as param_card_mod
    is_param_scan = isinstance(iterator, param_card_mod.ParamCardIterator)

    backup = card_path + ".scan_bak"
    shutil.copy(card_path, backup)
    try:
        for i, point in enumerate(iterator):
            point.write(card_path)
            logger.info("=== scan point %d ===", i + 1)
            process = run_single(switch, reweight)
            # use the run directory the process actually created, so the
            # per-point params.dat written by write_summary has a home
            name = os.path.basename(process.run_path)
            if is_param_scan:
                # pass the (possibly auto-width-updated) card so the summary
                # records the model-computed width for each 'auto' entry
                iterator.store_entry(name, process.get_result(),
                                     param_card_path=card_path)
            else:
                iterator.store_entry(name, process.get_result())
            _release_channel_generators(process)
        os.makedirs("Events", exist_ok=True)
        summary = os.path.join("Events", "scan_%s.txt" % run_name)
        iterator.write_summary(summary)
        logger.info("scan results written to %s", summary)
    finally:
        shutil.move(backup, card_path)


def run_generation(switch=None, reweight=None) -> None:
    """Run the generation, expanding a scan over the run_card or the param_card
    when one is present (scanning both simultaneously is not allowed).
    reweight (see prepare_reweighting()), if given, is forwarded to
    run_single()/run_scan() to reweight after each generation."""
    run_card_path = os.path.join("Cards", "run_card.toml")
    param_card_path = os.path.join("Cards", "param_card.dat")
    run_iter = detect_run_scan(run_card_path)
    param_iter = detect_param_scan(param_card_path)
    if run_iter and param_iter:
        raise RuntimeError(
            "Scanning simultaneously over the run_card and the param_card is "
            "not allowed. Please keep the scan:[...] entries in only one card.")
    if run_iter:
        run_scan(run_iter, run_card_path, switch, reweight)
    elif param_iter:
        run_scan(param_iter, param_card_path, switch, reweight)
    else:
        _release_channel_generators(run_single(switch, reweight))


def _lhe_needed_by(switch, card, reweight=None) -> "str | None":
    """Why the run needs an LHE event file, or None when it does not: every
    post-processing tool run_selected_tools drives, the classic reweighting
    module and the run_card [postprocessing] steps read the LHE file. MadtRex
    also reads the npy events. ``card`` is the raw TOML content of the
    run_card, ``reweight`` the prepare_reweighting() decision."""
    if reweight is not None and reweight.backend == "classic":
        return "reweighting (classic)"
    tools = selected_tools(switch) if switch else []
    if tools:
        return tools[0]
    cfg = card.get("postprocessing", {})
    try:
        if float(cfg.get("time_of_flight", -1.0)) >= 0:
            return "time_of_flight"
    except (TypeError, ValueError):
        pass
    # the legacy systematics.py path only runs when madspace did not compute
    # the weights itself (see run_lhe_postprocessing)
    if cfg.get("systematics") \
            and not card.get("systematics", {}).get("enable", True):
        return "systematics"
    return None


def force_lhe_output_if_needed(switch, reweight=None) -> None:
    """The events are written as npy by default; switch the run_card to the
    LHE format when a selected post-processing needs an LHE file. The card is
    rewritten rather than refused, so that asking for a shower in the launch
    question is enough -- the user does not have to know that it constrains
    output_format.

    The card is read as raw TOML and only its output_format line is edited: a
    run_card scan ("scan:[...]" values) is not a valid RunCardMG7 and must
    survive untouched for run_scan."""
    import tomllib
    path = os.path.join("Cards", "run_card.toml")
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        card = tomllib.load(f)
    output_format = card.get("run", {}).get(
        "output_format", RunCardMG7()["run"]["output_format"])
    if output_format == "lhe":
        return
    reason = _lhe_needed_by(switch, card, reweight)
    if reason is None:
        return
    with open(path) as f:
        text = f.read()
    text, count = re.subn(r'(?m)^(\s*output_format\s*=\s*)("[^"]*"|\'[^\']*\'|\S+)',
                          r'\g<1>"lhe"', text, count=1)
    if not count:
        text, count = re.subn(r'(?m)^\[run\][^\n]*\n', '\\g<0>output_format = "lhe"\n',
                              text, count=1)
    if not count:
        text += '\n[run]\noutput_format = "lhe"\n'
    with open(path, "w") as f:
        f.write(text)
    logging.getLogger("madevent").info(
        "output_format set to 'lhe' (required by %s).", reason)


def _raise_open_file_limit() -> None:
    """Remove the soft limit on the number of open files: it can be quite low
    on some systems and the event-combination step opens one file per channel."""
    soft_lim, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))
    except (ValueError, OSError) as error:
        logger.debug("could not raise the open-file limit: %s", error)


def _stdin_is_redirected() -> bool:
    """True when stdin is a pipe, a socket or a file, i.e. neither a terminal
    nor a character device such as /dev/null (batch jobs)."""
    import stat
    try:
        return not stat.S_ISCHR(os.fstat(sys.stdin.fileno()).st_mode)
    except (OSError, ValueError, AttributeError):
        return False


def _stdin_has_input() -> bool:
    """True when something is waiting on stdin. A forced (-f) run never reads
    stdin, so a closed or idle pipe (CI runners, agents) is harmless there,
    but piped answers would be silently dropped."""
    import select
    try:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return False
        return bool(os.read(sys.stdin.fileno(), 1))
    except (OSError, ValueError, AttributeError):
        return False


def main() -> None:
    """``bin/generate_events`` entry point.

    It runs the same :class:`MG7Cmd` command MG5's in-process ``launch`` uses,
    so the two entry points cannot drift apart; the only difference is that
    nothing here supplies a ``mother``/``inputfile``, so the question falls back
    to reading this process's own stdin.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_false", dest="ask_edit_cards")
    parser.add_argument(
        "--decay-widths", metavar="JSON", default=None,
        help="only integrate the 1 -> n decays of this directory and write "
             "their widths (GeV) as JSON, see compute_decay_widths")
    parser.add_argument(
        "--precision", type=float, default=0.01,
        help="relative precision aimed at for every channel (--decay-widths)")
    parser.add_argument(
        "--nb-core", type=int, default=None,
        help="number of threads and compilation jobs (--decay-widths)")
    parser.add_argument(
        "--make-arg", action="append", default=[],
        help="extra argument of the matrix element build (--decay-widths)")
    parser.add_argument(
        "--reweight", dest="reweight_card", default=None,
        help="reweight the events after generation with this reweight card "
             "(it becomes Cards/reweight_card.dat and switches reweighting on)",
    )
    parser.add_argument(
        "--reweight-backend", dest="reweight_backend", default=None,
        choices=("classic", "madtrex", "auto"),
        help="who reweights: the classic module, MadtRex, or auto (the default: "
             "MadtRex when it can run the card)",
    )
    parser.add_argument(
        "--allow-stdin", action="store_true", default=False,
        help="read the answers to the questions piped or redirected into stdin, "
             "as an interactive session: invalid \"set\" commands then only warn")
    args = parser.parse_args()

    # The answers to the questions are read from stdin as in an interactive
    # session, where a command that cannot be applied (e.g. an invalid "set")
    # is only a warning: refuse that mode unless asked for. --decay-widths and
    # a forced run ask nothing; the latter is still refused when answers are
    # waiting on stdin, which it would silently drop.
    if not args.allow_stdin and not args.decay_widths and _stdin_is_redirected() \
            and (args.ask_edit_cards or _stdin_has_input()):
        sys.exit("Reading the answers to the questions from stdin is not supported: "
                 "write the line 'launch %s' and the answers to a file and run "
                 "'%s PATH' instead, so that an invalid command stops the run.\n"
                 "Use --allow-stdin to read stdin as an interactive session (invalid "
                 "commands then only warn)."
                 % (os.getcwd(), _MG_ROOT / 'bin' / 'madgraph'))

    if args.decay_widths:
        results = compute_decay_widths(
            ".", precision=args.precision, nb_core=args.nb_core,
            extra_make_args=args.make_arg)
        with open(args.decay_widths, "w") as f:
            json.dump(results, f)
        return

    line = "generate_events" if args.ask_edit_cards else "generate_events -f"
    if args.reweight_card:
        line += " --reweight=%s" % args.reweight_card
    if args.reweight_backend:
        line += " --reweight-backend=%s" % args.reweight_backend
    MG7Cmd().run_cmd(line)
