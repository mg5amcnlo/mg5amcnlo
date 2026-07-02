import json
import os
from collections import defaultdict

from madgraph.various.diagram_symmetry import find_symmetry, IdentifySGConfigTag
from madgraph.iolibs import export_cpp

class OneProcessExporterMG7(export_cpp.OneProcessExporterCPP):

    def __init__(self, matrix_element, cpp_helas_call_writer):
        super().__init__(matrix_element, cpp_helas_call_writer)
        self.matrix_element = matrix_element
        self.name = f"P{matrix_element.get('processes')[0].shell_string()}"
        self.model = self.matrix_element.get("processes")[0].get("model")
        self.amplitude = self.matrix_element.get("base_amplitude")
        self.sym_indices, self.sym_perms, _ = find_symmetry(
            self.matrix_element, lambda diag: IdentifySGConfigTag(diag, self.model)
        )
        self.diagrams = self.amplitude.get("diagrams")
        self.helas_diagrams = self.matrix_element.get("diagrams")
        self.all_flavors, self.all_flavors_pdgs = self.matrix_element.get_external_flavors_with_iden(return_pdgs=True)
        self.process = self.amplitude.get("process")
        self.legs = self.process.get("legs_with_decays")
        self.color_basis = self.matrix_element.get("color_basis")
        self.set_topology()
        self.set_flavor_indices()
        self.set_channels_colors_map()

    def generate_process_files(self):
        super().generate_process_files()

    def set_topology(self):
        self.edge_names = {}
        self.incoming = [None] * 2
        self.outgoing = [None] * (len(self.legs) - 2)
        for leg in self.legs:
            number = leg.get("number")
            if leg.get("state"):
                self.edge_names[number] = f"o{number - 3}"
                self.outgoing[number - 3] = leg.get("id")
            else:
                self.edge_names[number] = f"i{number - 1}"
                self.incoming[number - 1] = leg.get("id")

    def set_flavor_indices(self):
        self.all_flavors_same_initial = []
        self.all_flavors_indices = []
        for i, flavors in enumerate(self.all_flavors_pdgs):
            flv_dict = defaultdict(list)
            for flv in flavors:
                flv_dict[(flv[0], flv[1])].append(flv)
            indices = []
            for flv in flv_dict.values():
                indices.append(len(self.all_flavors_same_initial))
                self.all_flavors_same_initial.append((i, flv))
            self.all_flavors_indices.append(indices)

    def set_channels_colors_map(self):
        if self.color_basis:
            diag_jamps = defaultdict(list)
            for ijamp, col_basis_elem in enumerate(sorted(self.color_basis.keys())):
                for diag_tuple in self.color_basis[col_basis_elem]:
                    diag_jamps[diag_tuple[0]].append(ijamp)

        sym_indices, sym_perms, _ = find_symmetry(
            self.matrix_element, lambda diag: IdentifySGConfigTag(diag, self.model)
        )

        self.channels = []
        self.channel_indices = []
        for diagram_index, (sym_index, sym_perm) in enumerate(zip(sym_indices, sym_perms)):
            if sym_index == 0:
                self.channel_indices.append(-1)
                continue

            active_colors = diag_jamps[diagram_index] if self.color_basis else [0]
            if sym_index < 0:
                self.channels[self.channel_indices[-sym_index - 1]]["diagrams"].append(
                    {
                        "diagram": diagram_index,
                        "permutation": sym_perm,
                        "active_colors": active_colors,
                    }
                )
                self.channel_indices.append(-1)
                continue

            helas_diagram = self.helas_diagrams[diagram_index]
            active_flavors = [
                flav_id
                for indices, flavors in zip(self.all_flavors_indices, self.all_flavors)
                if helas_diagram.check_flavor([flv for flv in flavors[0]], self.model)
                for flav_id in indices
            ]

            diagram = self.diagrams[diagram_index]
            vertices = []
            propagators = []
            on_shell_propagators = []
            diagram_edge_names = dict(self.edge_names)
            diag_vertices = diagram.get("vertices")
            for i_vert, vertex in enumerate(diag_vertices):
                legs = vertex.get("legs")
                # Last amplitude vertex does not create new edges
                vertex_props = [diagram_edge_names[leg.get("number")] for leg in legs[:-1]]

                final_part = self.model.get_particle(legs[-1].get("id"))
                if i_vert == len(diag_vertices) - 1:
                    vertex_props.append(diagram_edge_names[legs[-1].get("number")])
                else:
                    prop_index = len(propagators)
                    prop_name = f"p{prop_index}"
                    diagram_edge_names[legs[-1].get("number")] = prop_name
                    vertex_props.append(prop_name)
                    propagators.append(final_part.get_pdg_code())
                    if legs[-1].get("onshell"):
                        on_shell_propagators.append(prop_index)
                vertices.append(vertex_props)

            self.channel_indices.append(len(self.channels))
            self.channels.append(
                {
                    "propagators": propagators,
                    "vertices": vertices,
                    "on_shell_propagators": on_shell_propagators,
                    "active_flavors": active_flavors,
                    "diagrams": [
                        {
                            "diagram": diagram_index,
                            "permutation": sym_perm,
                            "active_colors": active_colors,
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
            # First build a color representation dictionnary
            repr_dict = {}
            legs = self.process.get_legs_with_decays()
            for leg in legs:
                repr_dict[leg.get("number")] = self.model.get_particle(
                    leg.get("id")
                ).get_color() * (-1) ** (1 + leg.get("state"))
            # Get the list of color flows
            color_flow_dicts = self.color_basis.color_flow_decomposition(repr_dict, n_initial)
            # And output them properly
            color_flows = [[
                [[color_flow_dict[leg.get("number")][i] for i in [0, 1]] for leg in legs]
                for color_flow_dict in color_flow_dicts
            ]] * len(self.all_flavors_same_initial) #TODO: this is wrong for multiple flavors!!!
        else:
            color_flows = [[[[0, 0]] * n_external]] * len(self.all_flavors_same_initial) #TODO: this is wrong for multiple flavors!!!

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
        same_initial_multiparticle = self.incoming[0] == self.incoming[1]
        flavors = [
            {
                "index": index,
                "options": options,
                "mirror": has_mirror_all or (
                    same_initial_multiparticle and options[0][0] != options[0][1]
                )
            }
            for index, options in self.all_flavors_same_initial
        ]
        return {
            "incoming": self.incoming,
            "outgoing": self.outgoing,
            "channels": self.channels,
            "me_path": lib_me_path,
            "path": proc_dir,
            "flavors": flavors,
            "color_flows": color_flows,
            "pdg_color_types": pdg_color_types,
            "diagram_count": len(self.diagrams),
            "helicities": list(self.matrix_element.get_helicity_matrix()),
        }
