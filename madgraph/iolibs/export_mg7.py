import json
import logging
import os
from collections import defaultdict

from madgraph.various.diagram_symmetry import find_symmetry, IdentifySGConfigTag
from madgraph.iolibs import export_cpp
from madgraph.iolibs.group_subprocs import IdentifyConfigTag
from madgraph.core.diagram_generation import DiagramTag

logger = logging.getLogger('madgraph.export_mg7')

class IdentifyTopologyTag(IdentifyConfigTag):
    """ Like IndentifyConfigTag, but ignores spin and color """

    @staticmethod
    def link_from_leg(leg, model):
        # the parent link grew a third element (the bound state) in 3.8.0,
        # so take the leg data and its number positionally
        link = super(
            IdentifyTopologyTag, IdentifyTopologyTag
        ).link_from_leg(leg, model)[0]
        (leg_num1, _, mass, width, _), leg_num2 = link[0], link[1]
        return [((leg_num1, mass, width), leg_num2)]

    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        vertex = super(IdentifyTopologyTag, IdentifyTopologyTag).vertex_id_from_vertex(
            vertex, last_vertex, model, ninitial
        )
        if len(vertex) == 1:
            return ((0,),)
        (_, mass, width), _ = vertex
        return ((mass, width), 0)


class IdentifySGTopologyTag(IdentifySGConfigTag):
    """ Like IndentifySGConfigTag, but ignores spin, color and charge """

    @staticmethod
    def link_from_leg(leg, model):
        link = super(
            IdentifySGTopologyTag, IdentifySGTopologyTag
        ).link_from_leg(leg, model)[0]
        (state, _, _, _, mass, width), leg_num = link[0], link[1]
        return [((state, mass, width), leg_num)]

    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        vertex = super(IdentifySGTopologyTag, IdentifySGTopologyTag).vertex_id_from_vertex(
            vertex, last_vertex, model, ninitial
        )
        if vertex == (0,):
            return (0,)
        (_, mass, width, qcd, onshell), = vertex
        return ((mass, width, qcd, onshell),)


class OneProcessExporterMG7(export_cpp.OneProcessExporterCPP):

    def __init__(self, matrix_element, cpp_helas_call_writer, merge_same_topologies=True):
        super().__init__(matrix_element, cpp_helas_call_writer)
        self.matrix_element = matrix_element
        self.name = f"P{matrix_element.get('processes')[0].shell_string()}"
        self.model = self.matrix_element.get("processes")[0].get("model")
        self.amplitude = self.matrix_element.get("base_amplitude")
        if merge_same_topologies:
            self.sym_indices, self.sym_perms, _ = find_symmetry(
                self.matrix_element,
                lambda diag: IdentifySGTopologyTag(diag, self.model),
                skip_identical_check=True,
            )
        else:
            self.sym_indices, self.sym_perms, _ = find_symmetry(
                self.matrix_element, lambda diag: IdentifySGConfigTag(diag, self.model)
            )

        self.diagrams = self.amplitude.get("diagrams")
        self.helas_diagrams = self.matrix_element.get("diagrams")
        self.all_flavors, self.all_flavors_pdgs = self.matrix_element.get_external_flavors_with_iden(return_pdgs=True)
        self.all_flavors = [list(flavors) for flavors in self.all_flavors]
        self.all_flavors_pdgs = [list(pdgs) for pdgs in self.all_flavors_pdgs]
        self.expand_flavors_over_processes()
        self.process = self.amplitude.get("process")
        self.legs = self.process.get("legs_with_decays")
        self.color_basis = self.matrix_element.get("color_basis")
        # The basis a color flow is picked among: always the trace one, which
        # is the color basis itself unless the color sum runs on the DDM basis.
        # Everything indexing a color flow -- the color_flows table, the
        # active_colors masks, icolamp -- has to use this one and not the
        # (smaller) basis of the color sum.
        self.color_flow_basis = self.color_basis.get_flow_basis() \
                                if self.color_basis else self.color_basis
        self.set_subprocess_class()
        self.set_topology()
        self.set_flavor_indices()
        self.set_active_flavors()
        self.set_channels_colors_map()

    def generate_process_files(self):
        super().generate_process_files()

    def set_subprocess_class(self):
        is_parts = [
            self.model.get_particle(l.get("id"))
            for l in self.process.get("legs")
            if not l.get("state")
        ]
        fs_parts = [
            self.model.get_particle(l.get("id"))
            for l in self.process.get("legs")
            if l.get("state")
        ]
        self.subprocess_class = (
            tuple(
                (p.get("mass"), l.get("onshell"))
                for (p, l) in zip(is_parts + fs_parts, self.process.get("legs"))
            ),
            self.process.get("id"),
        )

    def set_topology(self):
        """Name every external leg i<k>/o<k> and record the initial/final pdgs.

        Two initial legs for a collision, one for a decay (``t > b w+, ...``,
        which MadSpin hands over as a single flattened matrix element). Legs are
        numbered 1..n with the initial state first, so the outgoing offset is
        the number of initial legs.
        """
        self.edge_names = {}
        self.n_initial = sum(1 for leg in self.legs if not leg.get("state"))
        self.incoming = [None] * self.n_initial
        self.outgoing = [None] * (len(self.legs) - self.n_initial)
        for leg in self.legs:
            number = leg.get("number")
            if leg.get("state"):
                index = number - self.n_initial - 1
                self.edge_names[number] = f"o{index}"
                self.outgoing[index] = leg.get("id")
            else:
                self.edge_names[number] = f"i{number - 1}"
                self.incoming[number - 1] = leg.get("id")
        if any(pdg is None for pdg in self.incoming + self.outgoing):
            raise AssertionError(
                "external legs of %s are not numbered 1..%d with the initial "
                "state first: %s" % (
                    self.name, len(self.legs),
                    [(leg.get("number"), leg.get("state")) for leg in self.legs])
            )

    def expand_flavors_over_processes(self):
        """Add the flavors that live in the *processes* mapped onto this matrix
        element rather than in its merged legs.

        get_external_flavors_with_iden only expands merged legs (pdg 81/82/...),
        i.e. the apply_flavor_grouping=True case. With grouping off there are no
        merged legs and MG5 instead maps every flavor-equivalent process onto a
        single matrix element -- u u~ > e+ e-, u u~ > mu+ mu-, c c~ > e+ e- and
        c c~ > mu+ mu- all share one -- so asking only for the merged expansion
        returns the representative alone and the other channels never make it
        into subprocesses.json (p p > l+ l- came out at 538 pb instead of
        1336 pb, i.e. 4 of 16 channels). madevent walks both sources in
        get_leshouche_lines; use the same shared enumeration here.
        """
        combinations = self.matrix_element.get_flavor_pdg_combinations(self.model)
        # Merged legs: get_external_flavors_with_iden already enumerated
        # everything, and re-expanding here would double count.
        if any(has_merged for _, has_merged in combinations):
            return
        pdg_lists = [pdgs for pdg_lists, _ in combinations for pdgs in pdg_lists]
        if len(pdg_lists) <= 1:
            return
        # Without merged legs every leg trivially takes flavor index 1, so all
        # these processes share the single coupling class and its flavor-index
        # tuple; keep all_flavors aligned with all_flavors_pdgs.
        if len(self.all_flavors) != 1 or len(self.all_flavors[0]) != 1:
            return
        self.all_flavors_pdgs = [pdg_lists]
        self.all_flavors = [self.all_flavors[0] * len(pdg_lists)]

    def set_flavor_indices(self):
        # Flavor combinations are grouped by their initial state: the launcher
        # picks one initial state (PDF-weighted), then a final state within it.
        # A decay has a single initial leg to group on, not a beam pair.
        self.all_flavors_same_initial = []
        self.all_flavors_indices = []
        for i, flavors in enumerate(self.all_flavors_pdgs):
            flv_dict = defaultdict(list)
            for flv in flavors:
                flv_dict[tuple(flv[:self.n_initial])].append(flv)
            indices = []
            for flv in flv_dict.values():
                indices.append(len(self.all_flavors_same_initial))
                self.all_flavors_same_initial.append((i, flv))
            self.all_flavors_indices.append(indices)

    def set_active_flavors(self):
        # Per-diagram flavor validity is precomputed in the diagram flavor store
        # (populate_flavor_validity, triggered via get_external_flavors_with_iden
        # in __init__), so this is a pure read through HelasDiagram.has_flavor.
        self.active_flavors = [[] for d in self.diagrams]
        for indices, flavors in zip(self.all_flavors_indices, self.all_flavors):
            flavor = tuple(flavors[0])
            for active_flavors, diag in zip(
                self.active_flavors, self.matrix_element.get('diagrams')
            ):
                if diag.has_flavor(flavor):
                    active_flavors.extend(indices)

    def diagram_edge_leg_sets(self, diagram, sym_perm=None):
        """For each internal line of `diagram`, in vertex-list order, the
        frozenset of external edge names behind it -- a vertex-order-
        independent identity, unlike diagram.get("vertices") position.
        `sym_perm` translates this diagram's leg numbers to the
        representative's; leave None for the representative itself."""
        def canonical_name(leg_number):
            if sym_perm is not None:
                leg_number = sym_perm[leg_number - 1] + 1
            return self.edge_names[leg_number]

        diagram_edge_names = {}
        edge_leg_sets = {name: frozenset((name,)) for name in self.edge_names.values()}
        leg_sets = []
        diag_vertices = diagram.get("vertices")
        for i_vert, vertex in enumerate(diag_vertices):
            legs = vertex.get("legs")
            input_names = [
                diagram_edge_names.get(leg.get("number"))
                or canonical_name(leg.get("number"))
                for leg in legs[:-1]
            ]
            downstream = frozenset().union(*(edge_leg_sets[name] for name in input_names))
            if i_vert == len(diag_vertices) - 1:
                # Closing vertex: its last leg is a pre-existing external edge,
                # not a new internal line.
                continue
            prop_name = f"p{len(leg_sets)}"
            diagram_edge_names[legs[-1].get("number")] = prop_name
            edge_leg_sets[prop_name] = downstream
            leg_sets.append(downstream)
        return leg_sets

    def propagator_pdg(self, leg, leg_set):
        """Signed pdg id of the internal line `leg`, oriented the way the
        phase-space topology reads it: flowing away from the initial state.

        Madgraph records, on the leg a vertex creates, the pdg of the line
        flowing into the legs that were combined to make it -- `leg_set`.
        That is already the decay orientation while those are all final
        state, but a line holding *every* initial leg is the one madspace
        roots the other way round: its decay products are the complementary
        legs, so what belongs in the LHE is the anti-particle. Without this
        the s-channel W+ of `p p > e+ ve` is written as a W- decaying to
        e+ ve. A leg set holding only some of the initial legs is a
        t-channel, which never becomes a decay and is left as madgraph put
        it.

        Only colour singlets actually depend on this: for a coloured line
        lhe_output.cpp's compute_decay_color infers the orientation from the
        colour flow and flips the pdg back itself.
        """
        part = self.model.get_particle(leg.get("id"))
        if part.get("self_antipart"):
            return part.get("pdg_code")
        sign = 1 if part.get("is_part") else -1
        if sum(1 for name in leg_set if name.startswith("i")) == self.n_initial:
            sign = -sign
        return sign * part.get("pdg_code")

    def diagram_propagator_pdgs(self, diagram, channel_leg_sets, sym_perm):
        """Signed pdg id of each internal line of `diagram`, reordered to
        match `channel_leg_sets` (the order used for
        Topology::Decay::flat_propagator_index) rather than this diagram's
        own vertex order, which need not agree even for a diagram merged
        into the channel by merge_same_topologies."""
        diag_vertices = diagram.get("vertices")
        leg_sets = self.diagram_edge_leg_sets(diagram, sym_perm)
        pdg_by_leg_set = {}
        for i_vert, vertex in enumerate(diag_vertices[:-1]):
            legs = vertex.get("legs")
            pdg_by_leg_set[leg_sets[i_vert]] = self.propagator_pdg(
                legs[-1], leg_sets[i_vert]
            )
        return [pdg_by_leg_set[leg_set] for leg_set in channel_leg_sets]

    def set_channels_colors_map(self):
        if self.color_basis:
            # active_colors ends up in the icolamp mask, which is walked over
            # the color flows, so it must be indexed on the flow basis
            flow_basis = self.color_flow_basis
            diag_jamps = defaultdict(list)
            # Only leading-Nc jamps are planar-compatible with a diagram's own
            # topology; like export_v4's get_icolamp_lines, drop the rest.
            max_Nc = max(
                v[4] - v[5]
                for val in flow_basis.values()
                for v in val
            )
            for ijamp, col_basis_elem in enumerate(sorted(flow_basis.keys())):
                for diag_tuple in flow_basis[col_basis_elem]:
                    if diag_tuple[4] - diag_tuple[5] == max_Nc:
                        diag_jamps[diag_tuple[0]].append(ijamp)

        self.channels = []
        # Index-aligned with self.channels; kept separate (not serialized --
        # frozensets aren't JSON-able) and only needed transiently to reorder
        # merged diagrams' propagator_pdgs, see diagram_propagator_pdgs.
        channel_leg_sets = []
        channel_indices = []
        self.diagram_tags = []
        for diagram_index, (sym_index, sym_perm) in enumerate(zip(self.sym_indices, self.sym_perms)):
            if sym_index == 0:
                channel_indices.append(-1)
                continue

            active_colors = diag_jamps[diagram_index] if self.color_basis else [0]
            active_flavors = self.active_flavors[diagram_index]
            if len(active_flavors) == 0:
                raise RuntimeError(
                    f"no valid flavor configurations found for diagram {diagram_index+1}"
                )
            diagram = self.diagrams[diagram_index]
            if sym_index < 0:
                chan_index = channel_indices[-sym_index - 1]
                self.diagram_tags[chan_index].append(
                    IdentifyTopologyTag(diagram, self.model),
                )
                self.channels[chan_index]["diagrams"].append(
                    {
                        "diagram": diagram_index,
                        "permutation": sym_perm,
                        "active_flavors": active_flavors,
                        "active_colors": active_colors,
                        "propagator_pdgs": self.diagram_propagator_pdgs(
                            diagram, channel_leg_sets[chan_index], sym_perm
                        ),
                    }
                )
                channel_indices.append(-1)
                continue

            vertices = []
            propagators = []
            on_shell_propagators = []
            # $-excluded s-channels: the matrix element vanishes within
            # bwcutoff widths of the pole, so the phase space flattens its
            # Breit-Wigner there (see build_topologies in launch.py)
            dollar_propagators = []
            diagram_edge_names = dict(self.edge_names)
            diag_vertices = diagram.get("vertices")
            # Index-aligned with `propagators`: both are filled in vertex-list
            # order and both skip the closing vertex, which is the last one.
            leg_sets = self.diagram_edge_leg_sets(diagram)
            for i_vert, vertex in enumerate(diag_vertices):
                legs = vertex.get("legs")
                # Last amplitude vertex does not create new edges
                vertex_props = [diagram_edge_names[leg.get("number")] for leg in legs[:-1]]

                if i_vert == len(diag_vertices) - 1:
                    vertex_props.append(diagram_edge_names[legs[-1].get("number")])
                else:
                    prop_index = len(propagators)
                    prop_name = f"p{prop_index}"
                    diagram_edge_names[legs[-1].get("number")] = prop_name
                    vertex_props.append(prop_name)
                    propagators.append(
                        self.propagator_pdg(legs[-1], leg_sets[prop_index])
                    )
                    if legs[-1].get("onshell"):
                        on_shell_propagators.append(prop_index)
                    elif legs[-1].get("onshell") is False:
                        dollar_propagators.append(prop_index)
                vertices.append(vertex_props)

            chan_index = len(self.channels)
            self.diagram_tags.append([IdentifyTopologyTag(diagram, self.model)])
            channel_indices.append(chan_index)
            channel_leg_sets.append(leg_sets)
            self.channels.append(
                {
                    "propagators": propagators,
                    "vertices": vertices,
                    "on_shell_propagators": on_shell_propagators,
                    "dollar_propagators": dollar_propagators,
                    "diagrams": [
                        {
                            "diagram": diagram_index,
                            "permutation": sym_perm,
                            "active_flavors": active_flavors,
                            "active_colors": active_colors,
                            "propagator_pdgs": propagators,
                        }
                    ],
                }
            )

        self.multi_channel_map = {}
        self.active_color_map = []
        i = 0
        for channel in self.channels:
            for diag in channel["diagrams"]:
                diagram_index = diag["diagram"]
                active_colors = diag["active_colors"]
                self.multi_channel_map[i] = [diagram_index]
                self.active_color_map.append(active_colors)
                i += 1

    def get_subprocess_info(self, proc_dir, lib_me_path):
        n_external, n_initial = self.matrix_element.get_nexternal_ninitial()
        if self.color_basis:
            # First build a color representation dictionary
            repr_dict = {}
            legs = self.process.get_legs_with_decays()
            for leg in legs:
                repr_dict[leg.get("number")] = self.model.get_particle(
                    leg.get("id")
                ).get_color() * (-1) ** (1 + leg.get("state"))
            # Get the list of color flows. This is about color flows, so
            # always the trace basis, even when the color sum runs on the DDM
            # one.
            color_flow_dicts = self.color_flow_basis.\
                               color_flow_decomposition(repr_dict, n_initial)
            # And output them properly
            color_flows = [
                [[color_flow_dict[leg.get("number")][i] for i in [0, 1]] for leg in legs]
                for color_flow_dict in color_flow_dicts
            ]
        else:
            color_flows = [[[0, 0]] * n_external]

        # We need the both particle and antiparticle wf_ids, since the identity
        # depends on the direction of the wf.
        wf_ids = set(
            wf_id
            for d in self.matrix_element.get("diagrams")
            for wf in d.get("wavefunctions")
            for wf_id in [wf.get_pdg_code(), wf.get_anti_pdg_code()]
        )
        leg_ids = set(
            leg_id
            for p in self.matrix_element.get("processes")
            for leg in p.get_legs_with_decays()
            for leg_id in [leg.get("id"), self.model.get_particle(leg.get("id")).get_anti_pdg_code()]
        )
        pdg_color_types = {}
        for part_id in sorted(list(wf_ids.union(leg_ids))):
            pdg_color_types[part_id] = self.model.get_particle(part_id).get_color()
            if abs(part_id) in self.model["merged_particles"]:
                for pdg in self.model["merged_particles"][abs(part_id)]:
                    sign = -1 if part_id < 0 else 1
                    pdg_color_types[sign * pdg] = sign * self.model.get_particle(part_id).get_color()

        has_mirror_all = self.matrix_element.get("has_mirror_process")
        # Whether the beam-swapped initial state is part of the process must be
        # derived from the process definition (per-beam multiparticle content),
        # exactly as madevent's write_mirrorprocs does -- not from the pdg of the
        # matrix-element legs. With flavor merging both initial legs of
        # "u q > u q" (q = u d) carry the same merged pdg (81), yet leg 1 is
        # fixed to u, so "d u > u d" is not part of the process and mirroring the
        # u d flavor would double count it.
        # A decay has a single initial leg, so there is no beam swap to mirror.
        same_initial_multiparticle = self.n_initial == 2 and \
            self.matrix_element.get("processes")[0].has_same_initial_multiparticle()
        flavors = [
            {
                "index": index,
                "options": options,
                "mirror": self.n_initial == 2 and (has_mirror_all or (
                    same_initial_multiparticle and options[0][0] != options[0][1]
                ))
            }
            for index, options in self.all_flavors_same_initial
        ]

        # power of alpha_s in |M|^2 (the QCD coupling order of the amplitude),
        # -1 when it differs between diagrams. The systematics computation uses
        # it to rescale |M|^2 for renormalisation scale variations; never let
        # its computation break the output.
        qcd_orders = set()
        try:
            for diagram in self.helas_diagrams:
                qcd_orders.add(diagram.calculate_orders().get('QCD', 0))
        except Exception as error:
            logger.debug('could not determine the QCD order: %s', error)
            qcd_orders.add(None)
        qcd_power = (qcd_orders.pop()
                     if len(qcd_orders) == 1 and None not in qcd_orders else -1)

        # an interference: squared split orders that drop a component, so that a
        # helicity can contribute negatively (madmatrix select_helicity_signed);
        # with [generation] interference_helicity = "summed" the |M|^2 is the
        # helicity sum and the LHE helicities of this subprocess are written as 9
        interference = False
        try:
            from madgraph.iolibs.export_v4 import split_order_tables
            so = split_order_tables(self.matrix_element)
            interference = bool(so) and so['nampso'] > 1 and not all(so['chosen'])
        except Exception as error:
            logger.debug('could not determine the split orders: %s', error)

        return (
            {
                "incoming": self.incoming,
                "outgoing": self.outgoing,
                "channels": self.channels,
                "me_path": lib_me_path,
                "path": proc_dir,
                "flavors": flavors,
                "qcd_power": qcd_power,
                "color_flows": color_flows,
                "pdg_color_types": pdg_color_types,
                "diagram_count": len(self.diagrams),
                "helicities": list(self.matrix_element.get_helicity_matrix()),
                "interference": interference,
            },
            self.diagram_tags,
            self.subprocess_class,
        )
