"""The run of one process directory: cards, integration, events and output."""

import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

from madgraph.mg7.common import backend_of, fptype_of, interference_helicities_summed, lhe_helicities, lhe_weight_info, logger, me_parameters, mlm_pdf_ladder, ms, resolve_auto_backend, resolve_seed, resolve_verbosity
from madgraph.various import misc
from madgraph.various.banner import RunCardMG7
from madgraph.models.check_param_card import ParamCard
from madgraph.models.model_reader import Alphas_Runner
from madgraph.mg7 import hwu_output
from madgraph.mg7 import npy_to_lhe
from madgraph.mg7 import options
from madgraph.mg7 import plots
from madgraph.mg7 import systematics_summary
from madgraph.mg7.bootstrap import INSTALL_DIR as _INSTALL_DIR
from madgraph.mg7.bootstrap import MADSPACE_DIR as _MADSPACE_DIR
from madgraph.mg7.bootstrap import MADTREX_DIR as _MADTREX_DIR
from madgraph.mg7.bootstrap import MADTREX_INSTALL_DIR as _MADTREX_INSTALL_DIR
from madgraph.mg7.bootstrap import MG_ROOT as _MG_ROOT
from madgraph.iolibs.template_files.mg7.custom_function_loader import load_custom_function
from madgraph.mg7.observables import CutItem, HistItem, build_observable, clean_pids, observable_pids
from madgraph.mg7.options import lhapdf_paths
from madgraph.mg7.postprocess import _find_event_file
from madgraph.mg7.subprocess_amplicol import AmplicolSubprocess, amplicol_subproc_args, is_amplicol
from madgraph.mg7.subprocess_madmatrix import MadgraphSubprocess, PhaseSpace, build_multi_channel_data

# From this many subprocess directories to compile, the standard headers are
# precompiled once ('make PCH=1', see madmatrix.mk): each file of a small process
# spends most of its time parsing them. Below, building the precompiled header
# costs about what it saves.
PCH_MIN_SUBPROCESSES = 4


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
        self.init_surrogate()
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
        # AmpliCol libraries, shared by all channels of their process
        self.amplicol_apis = {}
        # [path format, init path, parameters] of every load_matrix_element, in order
        self.loaded_matrix_elements = []
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

    def load_matrix_element(self, path_format: str, init_path: str, parameters: dict):
        """Load a matrix-element library into every context, recorded for the
        gridpack to load the same ones at the same indices. The init path is
        the param card, or the artifact of an AmpliCol library."""
        for context, backend in zip(self.contexts, self.backends):
            api = context.load_matrix_element(
                path_format.format(device=backend, fptype=self.fptype),
                os.path.abspath(init_path),
                parameters
            )
        self.loaded_matrix_elements.append([path_format, init_path, parameters])
        return api

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

    def init_surrogate(self) -> None:
        """Check the [surrogate] settings. The surrogates are loaded per
        subprocess, by MadgraphSubprocess."""
        settings = self.run_card["surrogate"]
        self.surrogate = None
        if not settings["function"]:
            return
        path = settings["function"]
        if any(is_amplicol(meta) for meta in self.subprocess_data):
            raise RuntimeError(
                "[surrogate] is not supported with [amplicol] subprocesses: "
                "set [surrogate] function = \"\" or use the madmatrix backend"
            )
        # a .so path with {subprocess} or {device} is checked when it is loaded
        if "{" not in path and not os.path.isfile(path):
            raise RuntimeError(f"[surrogate] function = {path!r}: file not found")
        if self.run_card["phasespace"]["sde_strategy"] != "denominators":
            raise RuntimeError(
                "[surrogate] needs [phasespace] sde_strategy = \"denominators\", "
                "as the surrogate does not return channel weights"
            )
        if self.run_card["run"]["dummy_matrix_element"]:
            raise RuntimeError("[surrogate] cannot be used with dummy_matrix_element")
        if self.run_card["gridpack"]["save_gridpack"]:
            raise RuntimeError("[surrogate] does not support gridpacks yet")
        self.surrogate = dict(settings)
        if not self.surrogate["param_card"]:
            self.surrogate["param_card"] = self.param_card_path
        logger.info(
            "Using the surrogate %s, %s with the full matrix element",
            path,
            "unweighting" if settings["mode"] == "unweight" else "reweighting",
        )

    def weighting_groups(self, subprocess_channels: list[list[int]]) -> list:
        """One ms.WeightingGroup per subprocess with weighting steps, over its
        channels, given by their indices in the event generator."""
        groups = []
        for subproc, channel_indices in zip(self.subprocesses, subprocess_channels):
            steps = subproc.weighting_steps()
            if not (channel_indices and steps):
                continue
            group = ms.WeightingGroup(channel_indices, steps)
            # [generation] sets the last unweighting, the first_pass_ ones the earlier
            first_pass = subproc.first_pass_settings()
            if first_pass is not None:
                group.first_pass_max_overweight_truncation = (
                    first_pass["first_pass_max_overweight_truncation"]
                )
                group.first_pass_freeze_max_weight_after = (
                    first_pass["first_pass_freeze_max_weight_after"]
                )
            groups.append(group)
        return groups

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
        options._LHAPDF = paths.with_data_path(paths.download_path)
        # subprocesses (systematics, MadSpin, reweight) find it here too
        os.environ["LHAPDF_DATA_PATH"] = os.pathsep.join(options._LHAPDF.data_paths)
        return options._LHAPDF

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
                subproc_class = (
                    AmplicolSubprocess if is_amplicol(meta) else MadgraphSubprocess
                )
                self.subprocesses.append(subproc_class(self, meta, subproc_id))
        else:
            for subproc_id, meta in enumerate(self.merged_subprocess_data):
                # an AmpliCol subprocess is never merged with another one
                first = self.subprocess_data[meta["subprocesses"][0]]
                subproc_class = (
                    AmplicolSubprocess if is_amplicol(first) else MadgraphSubprocess
                )
                self.subprocesses.append(
                    subproc_class(self, meta, subproc_id, self.subprocess_data)
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
        self.compile_amplicol()
        madmatrix_data = [
            meta for meta in self.subprocess_data if not is_amplicol(meta)
        ]
        if not madmatrix_data:
            return backends

        first_proc_path = madmatrix_data[0]["path"]
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
        # --hel_recycling is a CPU build: on a cuda/hip device the same output
        # compiles the plain matrix element (see backend/gpu/SigmaKin.cc)
        recycled = [
            os.path.basename(meta["path"]) for meta in madmatrix_data
            if os.path.isfile(os.path.join(meta["path"], "HelRecycling.inc"))
        ]
        gpu_backends = [b for b in resolved if b.startswith(("cuda", "hip"))]
        if recycled and gpu_backends:
            logger.info(
                "--hel_recycling is CPU only: the %s build of %s uses the plain "
                "matrix element", "/".join(gpu_backends), ", ".join(recycled)
            )
        # On a CPU the recycled build itself falls back to the plain matrix
        # element when its tables are too big for the SIMD width (see
        # hrUseRecycling in backend/{cpu,simd}/SigmaKin.cc, which prints so when
        # it runs). The limit and the row-by-row color sum are the hidden
        # run-card entries hel_recycling_max_mb / hel_recycling_row_color_sum
        # (else HRMAXMB / HRNOBLAS from the environment), passed to make.
        hr_args = self.hel_recycling_make_args()
        if recycled and len(gpu_backends) < len(resolved):
            limit = hr_args[0].split("=", 1)[1]
            logger.info(
                "--hel_recycling: %s use the recycled matrix element on the cpu "
                "unless its tables exceed %s MB per SIMD page (run.hel_recycling_max_mb, "
                "0 for no limit)", ", ".join(recycled),
                {"": "10 (default)", "0": "<no limit>"}.get(limit, limit)
            )
        hr_flags = ":".join(arg.split("=", 1)[1] for arg in hr_args)
        for backend in resolved:
            # by library: the flavours of a split subprocess share one; a library
            # built with other recycling settings is rebuilt too: its SigmaKin.o
            # records them in build.<backend>/.hrflags (madmatrix.mk)
            missing = list({
                meta["me_path"]: meta for meta in madmatrix_data
                if not os.path.isfile(
                    meta["me_path"].format(device=backend, fptype=fptype)
                )
                or self.hel_recycling_flags_changed(meta["path"], backend, hr_flags)
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
            make_args += self.extra_make_args + hr_args
            if (len(missing) >= PCH_MIN_SUBPROCESSES
                    and not any(arg.startswith("PCH=") for arg in make_args)):
                make_args.append("PCH=1")
            self.make_subprocesses(subproc_path, make_args, nb_core, log_path)
            logger.info(
                f"Compilation of SubProcesses done in {time.time() - start_time:.1f} s"
            )
        return resolved

    def compile_amplicol(self) -> None:
        """Build the libraries of the [amplicol] processes that are missing."""
        paths = sorted({
            meta["path"] for meta in self.subprocess_data
            if is_amplicol(meta) and not (
                os.path.isfile(meta["me_path"]) and os.path.isfile(meta["ratio_path"])
            )
        })
        log_path = os.path.join(self.run_path, "compile_amplicol.log")
        for check in sorted({
            os.path.join(os.path.dirname(os.path.normpath(path)), "check-pyamplicol")
            for path in paths
        }):
            # older outputs have no check script
            if os.path.isfile(check):
                self.check_pyamplicol(check)
        # as for the madmatrix libraries (the sub-makes share the jobs)
        nb_core = self.run_card["run"]["cpu_thread_pool_size"]
        if not nb_core or nb_core < 0:
            nb_core = os.cpu_count() or 1
        for path in paths:
            logger.info(f"Compiling the AmpliCol matrix elements in {path}, see {log_path}")
            self.make_subprocesses(path, [], nb_core, log_path)

    def hel_recycling_make_args(self) -> list[str]:
        """HRMAXMB=... and HRNOBLAS=... for make (madmatrix.mk), from the hidden
        run-card entries, else from the environment; empty values are the
        build's defaults."""
        run = self.run_card["run"]
        max_mb = run.get("hel_recycling_max_mb", -1)
        max_mb = str(max_mb) if max_mb is not None and max_mb >= 0 \
            else os.environ.get("HRMAXMB", "")
        no_blas = "1" if run.get("hel_recycling_row_color_sum", False) \
            else os.environ.get("HRNOBLAS", "")
        return [f"HRMAXMB={max_mb}", f"HRNOBLAS={no_blas}"]

    @staticmethod
    def hel_recycling_flags_changed(proc_path: str, backend: str, hr_flags: str) -> bool:
        """Whether a --hel_recycling subprocess was built for this backend with
        other recycling settings than hr_flags (see .hrflags in madmatrix.mk)."""
        if backend.startswith(("cuda", "hip")):
            return False
        if not os.path.isfile(os.path.join(proc_path, "HelRecycling.inc")):
            return False
        marker = os.path.join(proc_path, f"build.{backend}", ".hrflags")
        try:
            with open(marker) as fsock:
                return fsock.read().strip() != hr_flags
        except OSError:
            return False

    @staticmethod
    def check_pyamplicol(script: str) -> None:
        """Run SubProcesses/check-pyamplicol, which fails if the installed
        pyamplicol is not the one the output was made with."""
        result = subprocess.run([script], capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr.strip() or result.stdout.strip()
                       or f"{script} failed with exit status {result.returncode}")
            raise RuntimeError(message)

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
        subprocess_channels = [[] for _ in self.subprocesses]
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
                subprocess_channels[i].append(len(channel_generators))
                channel_generators.append(channel.event_generator)

        groups = self.weighting_groups(subprocess_channels)
        # the groups of the last generator built, the one a gridpack stores
        self.weighting_groups_json = ms.weighting_groups_to_json(groups)
        event_generator = ms.EventGenerator(
            contexts=self.contexts,
            channels=channel_generators,
            status_file=self.status_file,
            config=self.event_generator_config,
            seed=self.run_seed,
            groups=groups,
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
        all_mcdata = [
            None if is_amplicol(meta)
            else self.subprocesses[index].build_multi_channel_data()
            if self.merged_subprocess_data is None
            else build_multi_channel_data(meta, self)
            for index, meta in enumerate(self.subprocess_data)
        ]
        subproc_args = [
            amplicol_subproc_args(meta, self) if mcdata is None else ms.SubprocArgs(
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
        with open(os.path.join(data_path, "groups.json"), "w") as f:
            f.write(self.weighting_groups_json)

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

        # the AmpliCol artifacts are read at run time; the ratio library finds
        # them next to its own, otherwise empty, reweight directory
        for artifact_dir in sorted({
            os.path.dirname(meta["artifact"])
            for meta in self.subprocess_data if is_amplicol(meta)
        }):
            for name in ("lc", "full"):
                shutil.copytree(
                    os.path.join(artifact_dir, name),
                    os.path.join(gridpack_path, artifact_dir, name),
                )
            os.makedirs(os.path.join(gridpack_path, artifact_dir, "reweight"))

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
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "iolibs", "template_files", "mg7")
        shutil.copy(os.path.join(template_dir, "gridpack.py"), gen_events_file)
        shutil.copy(os.path.join(template_dir, "custom_function_loader.py"), bin_path)
        os.chmod(gen_events_file, 0o755)
        shutil.copy(os.path.join(template_dir, "gridpack_setup.py"), bin_path)
        if with_reweight:
            # MadtRex reweighting of the gridpack's events, with the modules
            # parsing its reweight cards
            reweight_file = os.path.join(bin_path, "reweight")
            shutil.copy(os.path.join(template_dir, "madtrex_driver.py"), reweight_file)
            os.chmod(reweight_file, 0o755)
            for name in ("param_reweight_card.py", "pdf_reweight_card.py",
                         "generic_reweight_card.py"):
                shutil.copy(os.path.join(template_dir, name),
                            os.path.join(bin_path, name))
        # streaming needs the fixed maximum weights of this mode
        if self.gridpack_run_mode == "fix_max_weight":
            shutil.copy(os.path.join(template_dir, "event_stream.py"), bin_path)

        data = {
            "matrix_elements": self.loaded_matrix_elements,
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
