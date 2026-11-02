"""Subprocesses with AmpliCol matrix elements ('[amplicol]' processes).

Each subprocess is one UMAMI channel of an AmpliCol provider: colour orderings
integrated at leading colour over one colour-ordered phase-space mapping, its
first phase-space channel. A colour ordering with partners (where the colour
singlets attach to the colour lines, the s-channel resonances it maps out) is
copied into the subprocess of each partner, and every copy weighted by its
mapping's density over the partners' densities, as in AmpliCol. Resonances that
cannot go on shell with the param card masses and widths are dropped at launch.
A second pass re- or
unweights the events to full colour with the ratio of the full-colour to the
leading-colour matrix element summed over all colour orderings, at the
helicity of the event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from madgraph.mg7.common import lhe_helicities, ms
from madgraph.mg7.observables import clean_pids
from madgraph.mg7.subprocess_madmatrix import (
    Channel,
    MadgraphSubprocess,
    PhaseSpace,
)

if TYPE_CHECKING:
    from madgraph.mg7.madgraph_process import MadgraphProcess


def is_amplicol(meta: dict) -> bool:
    return meta.get("backend") == "amplicol"


def amplicol_parameters(meta: dict, process: MadgraphProcess) -> dict[str, float]:
    """The values of the mutable model parameters in the param card."""
    parameters = {}
    for name, (block, code) in meta["parameters"].items():
        try:
            parameters[name] = float(
                process.param_card.get_value(block.lower(), tuple(code))
            )
        except (KeyError, ValueError):
            continue
    return parameters


def chain_topology(incoming_masses: list[float], outgoing_masses: list[float]):
    """A t-channel chain without propagators, for the LHE completion."""
    n_out = len(outgoing_masses)
    vertices = [["i0", "p0", "o0"]]
    vertices += [[f"p{i - 1}", f"p{i}", f"o{i}"] for i in range(1, n_out - 1)]
    vertices.append(["i1", f"p{n_out - 2}", f"o{n_out - 1}"])
    propagators = [ms.Propagator() for _ in range(n_out - 1)]
    return ms.Topology(
        ms.Diagram(incoming_masses, outgoing_masses, propagators, vertices)
    )


def channel_topology(
    incoming_masses: list[float],
    outgoing_masses: list[float],
    order: list[int],
    resonances: list[tuple[int, list[int]]],
    process: MadgraphProcess,
) -> tuple[ms.Topology, list[int], int]:
    """The topology of a colour-ordered phase-space channel, a t-channel chain
    through the final-state legs and the resonances (pdg, legs) that decay to
    them, its colour order over the children of that chain, and the number of
    chain propagators (which come first among the propagators)."""
    n_in = len(incoming_masses)
    legs = [frozenset(r_legs) for _, r_legs in resonances]
    top = [i for i, r in enumerate(legs) if not any(r < other for other in legs)]
    children = []
    for slot in order:
        if slot < n_in:
            continue
        child = next((("r", i) for i in top if slot in legs[i]), ("o", slot))
        if child not in children:
            children.append(child)
        elif children[-1] != child:
            raise RuntimeError(f"the resonance legs of {order} are not contiguous")
    chain_count = len(children) - 1

    def ref(child):
        kind, index = child
        return f"o{index - n_in}" if kind == "o" else f"p{chain_count + index}"

    refs = [ref(child) for child in children]
    if len(children) == 1:
        vertices = [["i0", "i1", refs[0]]]
    else:
        vertices = [["i0", "p0", refs[0]]]
        vertices += [[f"p{i - 1}", f"p{i}", refs[i]] for i in range(1, chain_count)]
        vertices.append(["i1", f"p{chain_count - 1}", refs[-1]])
    position = {slot: i for i, slot in enumerate(order)}
    for index, r_legs in enumerate(legs):
        inner = [
            i for i, other in enumerate(legs)
            if other < r_legs and not any(other < mid < r_legs for mid in legs)
        ]
        covered = set().union(*(legs[i] for i in inner))
        decay = [(min(position[s] for s in legs[i]), ref(("r", i))) for i in inner]
        decay += [(position[s], ref(("o", s))) for s in r_legs - covered]
        vertices.append([f"p{chain_count + index}", *(r for _, r in sorted(decay))])
    propagators = [ms.Propagator() for _ in range(chain_count)]
    for pdg, _ in resonances:
        propagators.append(ms.Propagator(
            mass=process.get_mass(abs(pdg)),
            width=process.get_width(abs(pdg)),
            pdg_id=pdg,
        ))
    topology = ms.Topology(
        ms.Diagram(incoming_masses, outgoing_masses, propagators, vertices)
    )
    color_order = []
    for slot in order:
        node = slot if slot < n_in else n_in + children.index(
            next((("r", i) for i in top if slot in legs[i]), ("o", slot))
        )
        if node not in color_order:
            color_order.append(node)
    return topology, color_order, chain_count


def channel_topologies(meta: dict, process: MadgraphProcess) -> list[tuple]:
    """channel_topology of every phase-space channel of a subprocess."""
    incoming_masses = [process.get_mass(pid) for pid in clean_pids(meta["incoming"])]
    outgoing_masses = [process.get_mass(pid) for pid in clean_pids(meta["outgoing"])]
    return [
        channel_topology(
            incoming_masses,
            outgoing_masses,
            channel["order"],
            [(r["pdg"], r["legs"]) for r in channel["resonances"]],
            process,
        )
        for channel in meta["phasespace_channels"]
    ]


def active_resonances(meta: dict, get_mass, get_width) -> dict:
    """The subprocess meta without the s-channel resonances that cannot go on
    shell: a resonance is active if its width is positive and its mass above
    the masses of its decay legs. Phase-space channels that then agree are kept
    apart: each is the channel of a copy of the flavours, in this subprocess or
    another one, and its density counts once in the weights of every copy."""
    leg_masses = [
        get_mass(pid) for pid in clean_pids(meta["incoming"] + meta["outgoing"])
    ]

    def active(resonance):
        pdg = abs(resonance["pdg"])
        return get_width(pdg) > 0 and get_mass(pdg) > sum(
            leg_masses[leg] for leg in resonance["legs"]
        )

    for resonance in meta.get("colored_resonances", []):
        if active(resonance):
            raise RuntimeError(
                f"the coloured s-channel resonance {resonance['pdg']} can go on "
                "shell, which [amplicol] phase-space mappings do not follow yet; "
                "generate the process with on-shell decays or with the madmatrix "
                "backend"
            )
    kept = [
        [i for i, r in enumerate(channel["resonances"]) if active(r)]
        for channel in meta["phasespace_channels"]
    ]
    channels = [
        {**channel, "resonances": [channel["resonances"][i] for i in keep]}
        for channel, keep in zip(meta["phasespace_channels"], kept)
    ]
    flavors = [
        {
            **flavor,
            "resonance_pdgs": [
                [pdgs[i] for i in kept[index]]
                for index, pdgs in zip(flavor["channels"], flavor["resonance_pdgs"])
            ],
        }
        for flavor in meta["flavors"]
    ]
    return {**meta, "phasespace_channels": channels, "flavors": flavors}


def lhe_diagrams(meta: dict) -> tuple[list[tuple[int, tuple]], list[list[float]]]:
    """The diagrams that the LHE writer takes the resonances of an event from,
    one per phase-space channel and resonance charges, and per flavour a mask
    of those it can be written with. None if there are no resonances."""
    if not any(channel["resonances"] for channel in meta["phasespace_channels"]):
        return [], []
    diagrams = []
    for flavor in meta["flavors"]:
        for channel, pdgs in zip(flavor["channels"], flavor["resonance_pdgs"]):
            if (channel, tuple(pdgs)) not in diagrams:
                diagrams.append((channel, tuple(pdgs)))
    masks = [
        [
            float(diagram in zip(flavor["channels"], map(tuple, flavor["resonance_pdgs"])))
            for diagram in diagrams
        ]
        for flavor in meta["flavors"]
    ]
    return diagrams, masks


def amplicol_subproc_args(meta: dict, process: MadgraphProcess) -> ms.SubprocArgs:
    """LHE completion of an AmpliCol subprocess: its diagrams are the
    phase-space channels, with the resonances they map out."""
    meta = active_resonances(meta, process.get_mass, process.get_width)
    diagrams, _ = lhe_diagrams(meta)
    if diagrams:
        topologies = channel_topologies(meta, process)
        particle_count = len(meta["incoming"]) + len(meta["outgoing"])
        colors = sorted({flavor["color_index"] for flavor in meta["flavors"]})
        by_channel = [
            [(d, pdgs) for d, (c, pdgs) in enumerate(diagrams) if c == channel]
            for channel in range(len(topologies))
        ]
        # the chain propagators come first: one less than the root's children
        chain_counts = [count for _, _, count in topologies]
        layout = dict(
            topologies=[topology for topology, _, _ in topologies],
            permutations=[
                [list(range(particle_count))] * len(channel) for channel in by_channel
            ],
            diagram_indices=[[d for d, _ in channel] for channel in by_channel],
            diagram_color_indices=[[colors] * len(channel) for channel in by_channel],
            diagram_propagator_pdgs=[
                [[0] * count + list(pdgs) for _, pdgs in channel]
                for channel, count in zip(by_channel, chain_counts)
            ],
        )
    else:
        layout = dict(
            topologies=[chain_topology(
                [process.get_mass(pid) for pid in clean_pids(meta["incoming"])],
                [process.get_mass(pid) for pid in clean_pids(meta["outgoing"])],
            )],
            permutations=[[]],
            diagram_indices=[[]],
            diagram_color_indices=[[]],
        )
    return ms.SubprocArgs(
        **layout,
        color_flows=meta["color_flows"],
        pdg_color_types={
            int(key): value for key, value in meta["pdg_color_types"].items()
        },
        helicities=lhe_helicities(meta, process.run_card),
        pdg_ids=[flavor["options"] for flavor in meta["flavors"]],
        pdg_id_weights=[flavor.get("weights", []) for flavor in meta["flavors"]],
    )


class AmplicolSubprocess(MadgraphSubprocess):
    def __init__(self, process, meta, subproc_id, unmerged_meta=None):
        if process.mlm_clustering:
            # the clustering needs Feynman-diagram topologies, which AmpliCol lacks
            raise RuntimeError(
                "dynamical_scale_choice = mlm is not supported for [amplicol] "
                f"subprocesses ({meta['incoming']} > {meta['outgoing']}): "
                "use a closed-form scale choice"
            )
        super().__init__(process, meta, subproc_id, unmerged_meta)
        self.meta = active_resonances(self.meta, process.get_mass, process.get_width)
        self.check_flavor_blind_cuts()

    def check_flavor_blind_cuts(self):
        """AmpliCol groups the flavours and identical-particle orderings that
        share a matrix element into one channel, so its cuts, built for the
        first flavour, must select the same particles for all of them."""
        options = [
            clean_pids(pdgs) for flavor in self.meta["flavors"]
            for pdgs in flavor["options"]
        ]
        for item in self.process.cut_data:
            for group in item.observable_kwargs["select_pids"]:
                if len({tuple(pid in group for pid in pdgs) for pdgs in options}) > 1:
                    raise RuntimeError(
                        f"the cut {item.observable_kwargs['name']} does not select "
                        f"the same particles in all flavours of the [amplicol] "
                        f"subprocess {self.meta['incoming']} > {self.meta['outgoing']}"
                    )

    def load_matrix_elements(self, api_path_formats, member_metas):
        run_args = self.process.run_card["run"]
        if any(device != "cpu" for device in self.process.device_types):
            raise RuntimeError("[amplicol] processes are only supported on CPUs")
        if run_args["dummy_matrix_element"]:
            raise RuntimeError("[amplicol] processes need their matrix elements")
        gridpack_args = self.process.run_card["gridpack"]
        if gridpack_args["save_gridpack"] and gridpack_args["include_source"]:
            raise RuntimeError(
                "gridpacks of [amplicol] processes cannot include the sources"
            )
        if self.meta["qcd_power"] < 0 and self.process.systematics_enabled():
            raise RuntimeError(
                "the scale and PDF systematics of [amplicol] processes that mix "
                "coupling orders are not supported yet: disable [systematics]"
            )
        self.supports_alpha_s = self.meta.get("supports_alpha_s", True)
        if not self.supports_alpha_s and not self.process.run_card["beam"]["fixed_ren_scale"]:
            raise RuntimeError(
                "the AmpliCol matrix elements of this model cannot take the running "
                "alpha_s, as the model has no SMINPUTS 3 parameter: use a fixed "
                "renormalization scale"
            )
        parameters = amplicol_parameters(self.meta, self.process)
        self.channel_index = self.meta["umami_channel"]
        self.matrix_elements = [
            self.load(self.meta["me_path"], self.meta["artifact"], parameters)
        ]
        self.surrogates = []
        self.ratio_matrix_element = None
        if self.process.run_card["amplicol"]["color_accuracy"] != "lc":
            self.ratio_matrix_element = self.load(
                self.meta["ratio_path"], self.meta["ratio_artifact"], parameters
            )

    def load(self, path: str, artifact: str, parameters: dict):
        # loaded once for all channels; each load reads the whole pyamplicol artifact
        apis = self.process.amplicol_apis
        if path not in apis:
            apis[path] = self.process.load_matrix_element(path, artifact, parameters)
        return apis[path]

    def build_multi_channel_data(self):
        return None

    def build_multichannel_phasespace(self) -> PhaseSpace:
        """The first phase-space channel of the UMAMI channel, weighted by the
        density of its mapping over those of each flavour's partners, the other
        phase-space channels, which are sampled by the subprocesses of the
        flavour's copies."""
        flavors = self.meta["flavors"]
        particle_count = len(self.incoming_masses) + len(self.outgoing_masses)
        mappings = []
        for topology, color_order, _ in channel_topologies(self.meta, self.process):
            mappings.append(ms.PhaseSpaceMapping(
                topology,
                self.process.e_cm,
                leptonic=self.process.leptonic,
                invariant_power=self.process.run_card["phasespace"]["invariant_power"],
                t_channel_mode=ms.PhaseSpaceMapping.color_ordered,
                cuts=self.cuts,
                permutations=[list(range(particle_count))],
                color_ordered=ms.ColorOrderedMapping.Options(color_order),
                beam_rapidity=self.process.beam_rapidity,
                mirror_beams=self.mirror_beams,
            ))
        if any(flavor["channels"][0] != 0 for flavor in flavors):
            raise RuntimeError(
                "the [amplicol] flavours of a subprocess must sample its first "
                "phase-space channel: regenerate the output"
            )
        prefix = f"subproc{self.subproc_id}.amplicol0"
        discrete_sym, discrete_flavor = self.build_discrete(1, len(flavors), prefix)
        channel = Channel(
            phasespace_mapping=mappings[0],
            adaptive_mapping=self.build_vegas(mappings[0], prefix),
            discrete_sym=discrete_sym,
            discrete_flavor=discrete_flavor,
            channel_weight_indices=[0],
            name="C0",
            active_flavors=[list(range(len(flavors)))],
        )
        diagrams, diagram_masks = lhe_diagrams(self.meta)
        weights = None
        if len(mappings) > 1 or diagrams:
            weights = ms.MappingChannelWeights(
                mappings[:1],
                [
                    [float(index in flavor["channels"]) for index in range(len(mappings))]
                    for flavor in flavors
                ],
                [channel for channel, _ in diagrams],
                diagram_masks,
                partner_mappings=mappings[1:],
            )
        return PhaseSpace(
            mode="multichannel",
            channels=[channel],
            symfact=[None],
            mapping_chan_weights=weights,
        )

    def build_flat_phasespace(self) -> PhaseSpace:
        return self.build_multichannel_phasespace()

    def simplify_phasespace(self, multi_phasespace, cross_sections) -> PhaseSpace:
        return multi_phasespace

    def integrand_matrix_elements(self) -> list:
        # without an alpha_s input, the model value of the fixed scale is used
        inputs = [
            ms.MatrixElement.momenta_in,
            *([ms.MatrixElement.alpha_s_in] if self.supports_alpha_s else []),
            ms.MatrixElement.flavor_in,
            ms.MatrixElement.channel_in,
            ms.MatrixElement.random_helicity_in,
        ]
        outputs = [
            ms.MatrixElement.matrix_element_out,
            ms.MatrixElement.color_index_out,
            ms.MatrixElement.helicity_index_out,
        ]
        return [ms.MatrixElement(
            self.matrix_elements[0].index(), self.particle_count, inputs, outputs, 1,
            True, self.process.me_frame, self.process.incoming_count,
            self.channel_index,
        )]

    def has_weighting_steps(self) -> bool:
        return self.ratio_matrix_element is not None

    def weighting_steps(self) -> list:
        """The second pass to full colour."""
        if self.ratio_matrix_element is None:
            return []
        ratio = ms.MatrixElement(
            self.ratio_matrix_element.index(),
            self.particle_count,
            [ms.MatrixElement.momenta_in, ms.MatrixElement.flavor_in,
             ms.MatrixElement.channel_in, ms.MatrixElement.helicity_in],
            [ms.MatrixElement.matrix_element_ratio_out],
            1,
            False,
            self.process.me_frame,
            self.process.incoming_count,
            self.channel_index,
        )
        gen_args = self.process.run_card["generation"]
        return [ms.WeightingStep(
            [ratio],
            mode=self.process.run_card["amplicol"]["mode"],
            max_overweight_truncation=gen_args["max_overweight_truncation"],
            freeze_max_weight_after=gen_args["freeze_max_weight_after"],
        )]

    def first_pass_settings(self) -> dict | None:
        amplicol = self.process.run_card["amplicol"]
        if self.ratio_matrix_element is None or amplicol["mode"] != "unweight":
            return None
        return amplicol
