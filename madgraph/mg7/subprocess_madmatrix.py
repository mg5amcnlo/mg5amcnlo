"""Subprocesses with madmatrix matrix elements and MG diagram based phase space."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TYPE_CHECKING

from madgraph.mg7.common import logger, me_parameters, ms
from madgraph.iolibs.template_files.mg7.custom_function_loader import load_surrogate
from madgraph.mg7.observables import build_observable, clean_pids, decay_products, observable_pids, pid_is_qcd

if TYPE_CHECKING:
    from madgraph.mg7.madgraph_process import MadgraphProcess

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
    mapping_chan_weights: ms.MappingChannelWeights | None = None


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
            member_metas = [self.meta]
        else:
            api_path_formats = []
            member_metas = []
            for subproc in self.meta["subprocesses"]:
                submeta = unmerged_meta[subproc]
                api_path_formats.append(submeta["me_path"])
                member_metas.append(submeta)
            if len(api_path_formats) == 1:
                self.meta = submeta
            else:
                self.unmerged_meta = unmerged_meta

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
            self.scale = self.build_mlm_scale(min_scale, max_scale)
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

        # the libraries were all built up front by MadgraphProcess.compile_matrix_elements
        self.load_matrix_elements(api_path_formats, member_metas)

    def build_mlm_scale(self, min_scale: float, max_scale: float):
        """The scales of the MLM clustering history of each event."""
        mc_data = self.build_multi_channel_data()
        beam = self.process.run_card["beam"]
        all_pids = (clean_pids(self.meta["incoming"])
                    + clean_pids(self.meta["outgoing"]))
        return ms.EnergyScale(
            ms.MLMClustering(
                topologies=[topo[0] for topo in mc_data.topologies],
                permutations=mc_data.permutations,
                diagram_indices=mc_data.diagram_indices,
                cm_energy=self.process.e_cm,
                jet_scale_scheme=getattr(
                    ms.MLMClustering.JetScaleScheme, beam["jet_scale_scheme"]),
                scale_scheme=getattr(
                    ms.MLMClustering.ScaleScheme, beam["scale_scheme"]),
                pdg_color_types={
                    int(key): value
                    for key, value in self.meta["pdg_color_types"].items()
                },
                xqcut=self.process.run_card["phasespace"]["xqcut"],
                bw_cutoff=self.process.run_card["phasespace"]["bw_cutoff"],
                jet_radius=beam["jet_radius"],
                hadronic=not self.process.leptonic,
                external_pdg_ids=all_pids,
                max_jet_flavor=beam["max_jet_flavor"],
                parton_line_scheme=getattr(
                    ms.MLMClustering.PartonLineScheme, beam["parton_line_scheme"]),
                alphas_scheme=getattr(
                    ms.MLMClustering.AlphasScheme, beam["alphas_reweighting"]),
                pdf_reweighting=beam["pdf_reweighting"],
                clustering_measure=getattr(
                    ms.MLMClustering.ClusteringMeasure, beam["clustering_measure"]),
                clustering_history=getattr(
                    ms.MLMClustering.ClusteringHistory, beam["clustering_history"]),
                clustering_mean_pdf_ladder=beam["clustering_mean_pdf_ladder"],
            ),
            min_scale=min_scale,
            max_scale=max_scale,
        )

    def load_matrix_elements(self, api_path_formats, member_metas):
        """The matrix element and [surrogate] of every member subprocess."""
        if self.process.run_card["run"]["dummy_matrix_element"]:
            self.matrix_elements = [None] * len(api_path_formats)
        else:
            self.matrix_elements = [
                self.process.load_matrix_element(
                    path_format, self.process.param_card_path,
                    me_parameters(self.process.run_card)
                )
                for path_format in api_path_formats
            ]
        self.surrogates = []
        if self.process.surrogate is not None:
            for path_format, member_meta in zip(api_path_formats, member_metas):
                self.surrogates.append(self.load_surrogate(path_format, member_meta))

    def load_surrogate(self, me_path_format: str, member_meta: dict):
        """The [surrogate] of one matrix element of the subprocess."""
        name = re.sub(
            r"^libmadmatrix_(.*)_\{device\}\.so$", r"\1", os.path.basename(me_path_format)
        )
        # PDG ids for each flavor index of the matrix element
        flavors = member_meta["flavors"]
        pdg_ids = [flavors[0]["options"][0]] * (max(f["index"] for f in flavors) + 1)
        for flavor in flavors:
            pdg_ids[flavor["index"]] = flavor["options"][0]
        settings = self.process.surrogate
        apis = load_surrogate(
            ms,
            self.process.contexts,
            self.process.backends,
            self.process.device_types,
            settings["function"],
            name,
            pdg_ids,
            settings["param_card"],
            me_parameters(self.process.run_card),
        )
        if len({api.index() for api in apis}) != 1:
            raise RuntimeError("surrogate loaded at different indices")
        return apis[0]

    def has_weighting_steps(self) -> bool:
        return bool(self.surrogates)

    def weighting_steps(self) -> list:
        """The steps after the first unweighting: with a [surrogate], the
        second pass with the full matrix element."""
        if not self.surrogates:
            return []
        gen_args = self.process.run_card["generation"]
        return [ms.WeightingStep(
            self.step_matrix_elements,
            mode=self.process.surrogate["mode"],
            max_overweight_truncation=gen_args["max_overweight_truncation"],
            freeze_max_weight_after=gen_args["freeze_max_weight_after"],
        )]

    def first_pass_settings(self) -> dict | None:
        """The run card section with the first_pass_ unweighting settings, if
        the events are unweighted again after the first unweighting."""
        if not self.surrogates or self.process.surrogate["mode"] != "unweight":
            return None
        return self.process.surrogate

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
            mapping_chan_weights=phasespace.mapping_chan_weights,
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

    def integrand_matrix_elements(self) -> list:
        """The matrix element of every member subprocess in the integrand. Sets
        the step_matrix_elements of the second pass of a [surrogate]."""
        matrix_elements = []
        self.step_matrix_elements = []
        for me_index, matrix_element in enumerate(self.matrix_elements):
            if self.surrogates:
                # the surrogate in the integrand, the full matrix element in
                # the second pass, with the random numbers passed in
                mat = ms.MatrixElement(
                    self.surrogates[me_index].index(),
                    self.particle_count,
                    [ms.MatrixElement.momenta_in, ms.MatrixElement.alpha_s_in,
                     ms.MatrixElement.flavor_in],
                    [ms.MatrixElement.matrix_element_out],
                    1,
                    False,
                    self.process.me_frame,
                    self.process.incoming_count,
                )
                self.step_matrix_elements.append(ms.MatrixElement(
                    matrix_element,
                    [ms.MatrixElement.momenta_in, ms.MatrixElement.alpha_s_in,
                     ms.MatrixElement.flavor_in, ms.MatrixElement.random_color_in,
                     ms.MatrixElement.random_helicity_in,
                     ms.MatrixElement.random_diagram_in],
                    [ms.MatrixElement.matrix_element_out,
                     ms.MatrixElement.color_index_out,
                     ms.MatrixElement.helicity_index_out,
                     ms.MatrixElement.diagram_index_out],
                    False,
                    self.process.me_frame,
                    self.process.incoming_count,
                ))
            elif matrix_element:
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
            matrix_elements.append(mat)
        return matrix_elements

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
            flavor_factors.append(flav.get("factor", len(flav["options"])))
            flavor_mirror.append(flav["mirror"])

        cross_sections = []
        for mat in self.integrand_matrix_elements():
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
                    # also the events' raw |M|^2, which MadtRex normalises with
                    return_matrix_element=(
                        self.has_weighting_steps() or self.process.store_matrix_elements
                    ),
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
                phasespace.mapping_chan_weights,
                mlm_history_qcd_power,
            ))
        #print(integrands[1].function())
        #for i in integrands: print(i.function())
        return integrands
