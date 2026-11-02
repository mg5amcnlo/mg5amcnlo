# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction
from itertools import permutations
from types import SimpleNamespace

import pytest

from pyamplicol.artifacts.umami import (
    FLAVOR_BLIND_ASSUMPTION,
    UmamiProcessInput,
    _identical_orbit,
    build_umami_metadata,
    umami_bundle_payloads,
)
from pyamplicol.artifacts.umami_otf import (
    build_umami_on_the_fly_physics,
    restore_umami_on_the_fly_input,
)
from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    GroupingProcess,
    GroupMember,
    ProcessGroupMap,
)
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def model():
    # Deliberately non-SM particle/parameter names and PDGs.
    return {
        "ir": {
            "particles": [
                {"pdg_code": 990, "color": 8, "mass": "ZERO"},
                {"pdg_code": 991, "color": 3, "mass": "ZERO"},
                {"pdg_code": -991, "color": -3, "mass": "ZERO"},
                {"pdg_code": 992, "color": 3, "mass": "ZERO"},
                {"pdg_code": -992, "color": -3, "mass": "ZERO"},
                {"pdg_code": 995, "color": 1, "mass": "mass_s"},
            ],
            "parameters": [
                {
                    "name": "strong_input",
                    "nature": "external",
                    "parameter_type": "real",
                    "lhablock": "SMINPUTS",
                    "lhacode": [3],
                    "value": [0.12, 0],
                },
                {"name": "mass_s", "nature": "external", "value": [7, 0]},
            ],
        },
        "parameter_defaults": {"strong_input": [0.12, 0], "mass_s": [7, 0]},
    }


def process(
    identifier="p", pdgs=(990, 990, 990, 990), accuracy="lc", words=((1, 2, 3, 4),)
):
    helicities = [
        {
            "id": "h:all-plus",
            "values": [1] * len(pdgs),
            "representative_id": "h:all-plus",
            "coefficient": 1,
            "structural_zero": False,
        }
    ]
    colors = [
        {
            "id": "flow:" + ",".join(map(str, word)),
            "word": list(word),
            "representative_id": "flow:" + ",".join(map(str, word)),
            "coefficient": 1,
        }
        for word in words
    ]
    if accuracy != "lc":
        colors = [{"id": "color:contracted", "kind": "contracted-color"}]
    physics = {
        "external_particles": [
            {
                "pdg": pdg,
                "label": i + 1,
                "index": i,
                "role": "initial" if i < 2 else "final",
            }
            for i, pdg in enumerate(pdgs)
        ],
        "helicities": helicities,
        "color_components": colors,
        "coverage": {
            "helicities": "selected",
            "color": "complete" if accuracy == "lc" else "contracted",
        },
        "model_parameters": [
            {
                "name": "strong_input",
                "kind": "external",
                "default_real": 0.12,
                "default_imaginary": 0,
                "mutable": True,
            }
        ],
        "extensions": {},
    }
    return UmamiProcessInput(identifier, identifier, accuracy, tuple(pdgs), physics)


def export(*processes, compiled_model=None):
    return build_umami_metadata(
        processes=processes, compiled_model=compiled_model or model()
    )


def entries(document):
    return [p for channel in document["channels"] for p in channel["processes"]]


def with_replay():
    p = process(words=((1, 2, 3, 4), (1, 2, 4, 3)))
    sectors = (
        {"id": 0, "kind": "single-trace", "word_labels": [1, 2, 3, 4]},
        {"id": 1, "kind": "single-trace", "word_labels": [1, 2, 4, 3]},
    )
    p.physics["extensions"]["lc_topology_replay"] = {
        "partitions": [
            {
                "representative_sector_id": 0,
                "active_sector_ids": [0, 1],
                "label_permutations": [
                    [[1, 1], [2, 2], [3, 3], [4, 4]],
                    [[1, 1], [2, 2], [3, 4], [4, 3]],
                ],
                "replay_weights": [
                    2,
                    2,
                ],  # Folded runtime weights are not physical-flow multiplicities.
                "proof": {"status": "proven"},
            }
        ]
    }
    return replace(p, color_sectors=sectors)


def test_generic_gluon_tables_have_physical_pdgs_and_complete_permutations():
    (data,) = export(process())
    assert data["pdg_ids"] == [[990, 990, 990, 990]]
    assert data["provider"]["masses"] == [0] * 4
    assert data["provider"]["alpha_s_parameter"] == "strong_input"
    assert data["provider"]["supports_alpha_s"]
    assert sorted(data["channels"][0]["phasespace_order"]) == list(range(4))
    assert data["color_flows"][0] == [[504, 501], [501, 502], [503, 502], [504, 503]]
    assert entries(data)[0]["matrix_elements"] == [{"pdg_ids": 0, "factor": 1}]
    assert data["grouping"] == {
        "assumptions": [
            "identical-final-particle permutation-invariant observables and cuts",
            "final-state flavour-blind observables and cuts",
        ],
        "physical_contributions": 1,
        "exported_contributions": 1,
    }


def test_quark_chain_and_singlet_derive_from_model_not_particle_names():
    (data,) = export(process(pdgs=(991, -991, 995, 990), words=((2, 4, 1),)))
    assert data["provider"]["masses"] == [0, 0, 7, 0]
    assert data["color_flows"] == [[[502, 0], [0, 501], [0, 0], [502, 501]]]
    assert data["color_orders"] == [[1, 3, 0, 2]]


def test_two_quark_pairs_use_supplied_sector_lines():
    p = process(pdgs=(991, -991, 992, -992, 990), words=((2, 5, 4, 3, 1),))
    sector = {
        "id": 0,
        "kind": "open-lines",
        "word_labels": [2, 5, 4, 3, 1],
        "open_color_lines": [
            {"fundamental_label": 3, "adjoint_labels": [], "antifundamental_label": 1},
            {"fundamental_label": 2, "adjoint_labels": [5], "antifundamental_label": 4},
        ],
    }
    (data,) = export(replace(p, color_sectors=(sector,)))
    assert data["color_flows"] == [[[501, 0], [0, 502], [501, 0], [0, 503], [503, 502]]]
    sector["open_color_lines"][0]["antifundamental_label"] = 4
    with pytest.raises(ValueError, match="cover"):
        export(replace(p, color_sectors=(sector,)))


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
def test_contracted_color_has_no_fake_flows(accuracy):
    (data,) = export(process(accuracy=accuracy))
    assert data["color_flows"] == []
    assert entries(data)[0]["color_flows"] is None
    assert entries(data)[0]["runtime"]["color_id"] is None
    payloads = umami_bundle_payloads(
        processes=[process(accuracy=accuracy)], compiled_model=model()
    )
    header = next(p for p in payloads if p.path.endswith("provider_p0.h"))
    assert b"#define UMAMI_HAS_COLOR_FLOW 0" in header.content


def test_fixed_metadata_and_overlapping_selections_split_providers():
    docs = export(
        process("a"),
        process("b"),
        process("c", accuracy="full"),
        process("d", pdgs=(991, -991, 995, 990), words=((2, 4, 1),)),
    )
    assert len(docs) == 4
    payloads = {
        p.path: p
        for p in umami_bundle_payloads(
            processes=[process("a"), process("b")], compiled_model=model()
        )
    }
    index = json.loads(payloads["API/umami/metadata.json"].content)
    assert index["kind"] == "pyamplicol-umami-provider-index"
    assert len(index["providers"]) == 2
    assert b"UMAMI_SINGLE_PROVIDER := 0" in payloads["API/umami/providers.mk"].content


def test_permuted_duplicate_final_states_are_not_summed_in_one_provider():
    a = process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),))
    b = process("b", pdgs=(990, 990, -991, 991), words=((4, 1, 2, 3),))
    assert len(export(a, b)) == 2


def test_distinct_restricted_helicity_domains_require_separate_providers():
    a = process("a", pdgs=(991, -991, 990, 990), words=((2, 3, 4, 1),))
    b = process("b", pdgs=(992, -992, 990, 990), words=((2, 3, 4, 1),))
    b.physics["helicities"][0]["values"] = [-1, 1, 1, 1]
    assert len(export(a, b)) == 2


def test_groups_identical_legs_and_preserves_physical_runtime_mappings():
    (data,) = export(with_replay())
    (row,) = entries(data)
    assert row["runtime"]["factor"] == 1
    assert row["runtime"]["color_id"] == "flow:1,2,3,4"
    assert row["runtime"]["permutation"] == [0, 1, 2, 3]
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]
    assert len(row["members"]) == 2
    assert row["members"][1]["runtime"]["permutation"] == [0, 1, 3, 2]
    assert row["members"][1]["runtime"]["color_id"] == "flow:1,2,3,4"
    assert {member["color_flows"] for member in row["members"]} == {0, 1}
    assert data["grouping"]["physical_contributions"] == 2
    assert data["grouping"]["exported_contributions"] == 1


def test_incoming_replay_swap_is_not_used_for_identical_leg_grouping():
    p = with_replay()
    p.physics["extensions"]["lc_topology_replay"]["partitions"][0][
        "label_permutations"
    ][1] = [[1, 2], [2, 1], [3, 3], [4, 4]]
    (data,) = export(p)
    (row,) = entries(data)
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]
    assert row["members"][1]["runtime"]["permutation"] == [0, 1, 3, 2]


def test_noninvariant_restricted_helicity_is_not_grouped():
    p = with_replay()
    p.physics["helicities"][0]["values"] = [1, 1, -1, 1]
    (data,) = export(p)
    assert len(entries(data)) == 2
    assert entries(data)[1]["runtime"]["color_id"] == "flow:1,2,4,3"
    # Both orders share one channel, the second with its final legs relabelled.
    assert len(data["channels"]) == 1
    assert entries(data)[1]["runtime"]["permutation"] == [0, 1, 3, 2]
    assert entries(data)[1]["members"][0]["runtime"]["permutation"] == [0, 1, 2, 3]


def test_orders_differing_by_equal_mass_final_labels_share_a_channel():
    """One phase-space map per cyclic pattern of incoming legs and final masses."""
    extended = model()
    extended["ir"]["particles"] += [
        {"pdg_code": 993, "color": 8, "mass": "mass_s"},
        {"pdg_code": 994, "color": 8, "mass": "ZERO"},
    ]
    words = ((1, 2, 3, 4), (1, 2, 4, 3))
    (data,) = export(process(pdgs=(990, 990, 994, 990), words=words),
                     compiled_model=extended)
    assert len(data["channels"]) == 1
    first, second = entries(data)
    pdgs = [data["pdg_ids"][e["matrix_elements"][0]["pdg_ids"]] for e in (first, second)]
    assert pdgs == [[990, 990, 994, 990], [990, 990, 990, 994]]
    assert second["runtime"]["permutation"] == [0, 1, 3, 2]
    assert data["color_orders"][second["color_order"]] == [0, 1, 2, 3]
    # The member keeps the physical labels of its process.
    assert second["members"][0]["pdgs"] == [990, 990, 994, 990]
    assert second["members"][0]["runtime"]["permutation"] == [0, 1, 2, 3]
    # Final legs of different masses are never exchanged.
    (data,) = export(process(pdgs=(990, 990, 993, 990), words=words),
                     compiled_model=extended)
    assert len(data["channels"]) == 2


def test_permutation_invariant_restricted_helicity_set_is_grouped():
    p = with_replay()
    p.physics["helicities"][0]["values"] = [1, 1, -1, 1]
    opposite = copy.deepcopy(p.physics["helicities"][0])
    opposite.update(
        id="h:opposite", representative_id="h:opposite", values=[1, 1, 1, -1]
    )
    p.physics["helicities"].append(opposite)
    (data,) = export(p)
    (row,) = entries(data)
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]


def test_unproven_replay_is_not_used():
    p = with_replay()
    p.physics["extensions"]["lc_topology_replay"]["partitions"][0]["proof"] = None
    p.physics["extensions"]["lc_topology_replay"]["partitions"][0][
        "label_permutations"
    ][1] = [[1, 2], [2, 1], [3, 3], [4, 4]]
    (data,) = export(p)
    (row,) = entries(data)
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]
    assert row["members"][1]["runtime"]["permutation"] == [0, 1, 3, 2]


def test_identical_adjoint_finals_reduce_24_oriented_flows_to_four():
    words = tuple((1, *tail) for tail in permutations((2, 3, 4, 5)))
    p = process(pdgs=(990,) * 5, words=words)
    (data,) = export(p)
    rows = entries(data)
    assert len(rows) == 4
    assert data["grouping"]["physical_contributions"] == 24
    assert data["grouping"]["exported_contributions"] == 4
    assert len(data["color_flows"]) == 24
    assert all(row["matrix_elements"] == [{"pdg_ids": 0, "factor": 6}] for row in rows)
    assert all(len(row["members"]) == 6 for row in rows)
    assert all(row["runtime"]["factor"] == 1 for row in rows)
    assert {member["color_id"] for row in rows for member in row["members"]} == {
        color["id"] for color in p.physics["color_components"]
    }
    assert {member["color_flows"] for row in rows for member in row["members"]} == set(
        range(24)
    )
    assert all(
        member["runtime"]["permutation"][:2] == [0, 1]
        for row in rows
        for member in row["members"]
    )


def test_identical_grouping_does_not_fold_trace_reflection_or_swap_beams():
    p = process(pdgs=(990,) * 5, words=((1, 2, 3, 4, 5), (1, 5, 4, 3, 2)))
    p.physics["color_components"][1]["representative_id"] = "flow:1,2,3,4,5"
    (data,) = export(p)
    assert len(entries(data)) == 2
    assert all(
        row["matrix_elements"] == [{"pdg_ids": 0, "factor": 1}] for row in entries(data)
    )


def test_identical_trace_grouping_allows_cyclic_reanchoring():
    m = model()
    m["ir"]["particles"].append({"pdg_code": 993, "color": 8, "mass": "ZERO"})
    p = process(pdgs=(995, 995, 990, 993, 990), words=((3, 4, 5), (3, 5, 4)))
    (data,) = export(p, compiled_model=m)
    (row,) = entries(data)
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]
    assert len(row["members"]) == 2
    # Swap only the two identical adjoints, then re-anchor the directed trace.
    assert row["members"][1]["runtime"]["permutation"] == [0, 1, 4, 3, 2]


def test_identical_open_chains_allow_cyclic_block_rotation():
    p = process(
        pdgs=(991, 991, 991, 991, 990),
        words=((3, 5, 1, 4, 2), (3, 2, 4, 5, 1)),
    )
    (data,) = export(p)
    (row,) = entries(data)
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]
    assert len(row["members"]) == 2
    # The two outgoing quarks exchange; each directed chain and both beams
    # remain intact, while the complete chain blocks rotate cyclically.
    assert row["members"][1]["runtime"]["permutation"] == [0, 1, 3, 2, 4]


@pytest.mark.parametrize("pdgs", [(990,) * 5, (995, 995, 990, 990, 990)])
def test_identical_orbit_matches_full_key_minimum(pdgs):
    # Colourless beams make every re-anchoring of the trace tie on legs and
    # colour edges, so the helicity domain alone picks the canonical order.
    def full_key_minimum(order, flow, incoming, helicities, word_size):
        def key(candidate):
            tags = {}
            edges = [
                tags.setdefault(tag, len(tags) + 1) if tag else 0
                for slot in candidate
                for tag in flow[slot]
            ]
            return (
                tuple(
                    ("incoming", slot) if slot < incoming else ("final", pdgs[slot])
                    for slot in candidate
                ),
                tuple(edges),
                tuple(
                    sorted(
                        tuple(h["values"][slot] for slot in candidate)
                        for h in helicities
                    )
                ),
            )

        candidates = [
            order[i:word_size] + order[:i] + order[word_size:] for i in range(word_size)
        ]
        return min((key(candidate), candidate) for candidate in candidates)

    coloured = [slot for slot, pdg in enumerate(pdgs) if pdg == 990]
    domains = [
        [[1] * 5],
        [[1, 1, -1, 1, -1], [1, -1, 1, 1, -1]],
        [[-1, 1, 1, -1, 1], [1, 1, -1, -1, 1], [1, -1, 1, 1, 1]],
    ]
    for word in permutations(coloured):
        flow = [[0, 0] for _ in pdgs]
        for i, slot in enumerate(word):
            flow[slot] = [501 + i, 501 + (i - 1) % len(word)]
        order = [*word, *(slot for slot in range(len(pdgs)) if slot not in word)]
        for domain in domains:
            helicities = [{"values": values} for values in domain]
            assert _identical_orbit(
                order, flow, pdgs, 2, helicities, len(word)
            ) == full_key_minimum(order, flow, 2, helicities, len(word))


def test_equal_mass_distinct_final_species_are_not_identical():
    m = model()
    m["ir"]["particles"].append({"pdg_code": 993, "color": 8, "mass": "ZERO"})
    p = process(pdgs=(990, 990, 990, 993), words=((1, 2, 3, 4), (1, 2, 4, 3)))
    (data,) = export(p, compiled_model=m)
    assert len(entries(data)) == 2


def test_identical_grouping_preserves_open_chain_topology():
    p = process(
        pdgs=(991, -991, 991, -991, 990),
        words=((2, 5, 4, 3, 1), (2, 5, 1, 3, 4)),
    )
    sectors = (
        {
            "id": 0,
            "kind": "open-lines",
            "word_labels": [2, 5, 4, 3, 1],
            "open_color_lines": [
                {
                    "fundamental_label": 2,
                    "adjoint_labels": [5],
                    "antifundamental_label": 4,
                },
                {
                    "fundamental_label": 3,
                    "adjoint_labels": [],
                    "antifundamental_label": 1,
                },
            ],
        },
        {
            "id": 1,
            "kind": "open-lines",
            "word_labels": [2, 5, 1, 3, 4],
            "open_color_lines": [
                {
                    "fundamental_label": 2,
                    "adjoint_labels": [5],
                    "antifundamental_label": 1,
                },
                {
                    "fundamental_label": 3,
                    "adjoint_labels": [],
                    "antifundamental_label": 4,
                },
            ],
        },
    )
    (data,) = export(replace(p, color_sectors=sectors))
    assert len(entries(data)) == 2


def test_structural_crossflavour_equality_keeps_incoming_pdf_flavours():
    a = replace(
        process("a", pdgs=(991, -991, 990, 990), words=((2, 3, 4, 1),)),
        structural_keys={"flow:2,3,4,1": "certified-key"},
    )
    b = replace(
        process("b", pdgs=(992, -992, 990, 990), words=((2, 3, 4, 1),)),
        structural_keys={"flow:2,3,4,1": "certified-key"},
    )
    (data,) = export(a, b)
    rows = entries(data)
    assert len(rows) == 2
    assert rows[1]["runtime"]["process_id"] == "a"
    assert data["pdg_ids"] == [[991, -991, 990, 990], [992, -992, 990, 990]]
    assert [row["matrix_elements"] for row in rows] == [
        [{"pdg_ids": 0, "factor": 1}],
        [{"pdg_ids": 1, "factor": 1}],
    ]


def test_groups_certified_distinct_final_flavours():
    a = replace(
        process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),)),
        structural_keys={"flow:3,1,2,4": "certified-key"},
    )
    b = replace(
        process("b", pdgs=(990, 990, 992, -992), words=((3, 1, 2, 4),)),
        structural_keys={"flow:3,1,2,4": "certified-key"},
    )
    (data,) = export(a, b)
    assert len(entries(data)) == 1
    assert sum(len(row["members"]) for row in entries(data)) == 2
    assert entries(data)[0]["matrix_elements"] == [
        {"pdg_ids": 0, "factor": 1},
        {"pdg_ids": 1, "factor": 1},
    ]


def test_equal_defaults_are_not_an_equivalence_and_aliases_not_multiplicity():
    a = process("a", pdgs=(991, -991, 990, 990), words=((2, 3, 4, 1),))
    b = process("b", pdgs=(992, -992, 990, 990), words=((2, 3, 4, 1),))
    a = replace(
        a,
        aliases=(
            {
                "id": "alias",
                "external_pdgs": list(a.external_pdgs),
                "external_permutation": [1, 0, 2, 3],
            },
        ),
    )
    (data,) = export(a, b)
    assert len(entries(data)) == 2
    assert entries(data)[1]["runtime"]["process_id"] == "b"
    assert len(data["aliases"]) == 1
    assert data["grouping"]["physical_contributions"] == 2


def test_alpha_s_not_guessed_from_parameter_name():
    m = model()
    m["ir"]["parameters"][0]["lhablock"] = "UNRELATED"
    (data,) = export(process(), compiled_model=m)
    assert data["provider"]["alpha_s_parameter"] is None
    assert not data["provider"]["supports_alpha_s"]


def test_payloads_have_sdk_roles_and_generated_c_matches_json():
    payloads = {
        p.path: p
        for p in umami_bundle_payloads(
            processes=[with_replay()], compiled_model=model()
        )
    }
    data = json.loads(payloads["API/umami/metadata.json"].content)
    header = payloads["API/umami/provider_p0.h"].content.decode()
    assert "#define UMAMI_HAS_ALPHA_S 1" in header
    assert "static const UmamiProvider umami_provider" in header
    assert "umami_c0_f0_permutation[] = {0, 1, 2, 3}" in header
    assert "umami_c1_f0_permutation" not in header
    assert entries(data)[0]["members"][1]["runtime"]["permutation"] == [0, 1, 3, 2]
    assert data["provider"]["library"] == "libumami.so"
    assert payloads["API/umami/metadata.json"].role == "sdk-metadata"


def test_provider_reuses_existing_runtime_artifact_identity():
    payloads = {
        p.path: p
        for p in umami_bundle_payloads(
            processes=[process()],
            compiled_model=model(),
            artifact_id="existing-id",
        )
    }
    data = json.loads(payloads["API/umami/metadata.json"].content)
    assert data["provider"]["artifact_id"] == "existing-id"
    assert (
        b'#define UMAMI_ARTIFACT_ID "existing-id"'
        in payloads["API/umami/provider_p0.h"].content
    )


def test_mass_parameter_semantics_not_only_current_value_split_providers():
    m = model()
    m["ir"]["particles"].append({"pdg_code": 996, "color": 1, "mass": "other_mass"})
    m["ir"]["parameters"].append(
        {"name": "other_mass", "nature": "external", "value": [7, 0]}
    )
    a = process("a", pdgs=(991, -991, 995, 990), words=((2, 4, 1),))
    b = process("b", pdgs=(991, -991, 996, 990), words=((2, 4, 1),))
    assert len(export(a, b, compiled_model=m)) == 2


def test_derived_mass_uses_already_resolved_public_default():
    m = model()
    m["parameter_defaults"].pop("mass_s")
    m["ir"]["parameters"][1] = {
        "name": "mass_s",
        "nature": "internal",
        "parameter_type": "real",
        "expression": "sqrt(strong_input)",
        "resolved_expression": "sqrt(strong_input)",
    }
    p = process(pdgs=(991, -991, 995, 990), words=((2, 4, 1),))
    p.physics["model_parameters"].append(
        {
            "name": "mass_s",
            "kind": "derived",
            "default_real": 0.12**0.5,
            "default_imaginary": 0.0,
            "mutable": False,
        }
    )
    (data,) = export(p, compiled_model=m)
    assert data["provider"]["masses"] == [0, 0, 0.12**0.5, 0]
    assert data["provider"]["mass_parameters"][2] == "sqrt(strong_input)"


@pytest.mark.parametrize("representative", ["h:all-plus", "h:all-minus"])
def test_only_certified_helicity_equivalences_are_reused(representative):
    p = process()
    h = copy.deepcopy(p.physics["helicities"][0])
    h.update(id="h:all-minus", values=[-1] * 4, representative_id=representative)
    p.physics["helicities"].append(h)
    (data,) = export(p)
    assert entries(data)[0]["helicities"] == (
        [[0, 1]] if representative == "h:all-plus" else [[0], [1]]
    )
    assert entries(data)[0]["helicity_mappings"] == [
        {
            "helicity": index,
            "representative_id": "h:all-plus" if index == 0 else representative,
            "coefficient": 1,
            "structural_zero": False,
        }
        for index in range(2)
    ]


@pytest.mark.parametrize("accuracy,flows", [("lc", 6), ("full", 1)])
@pytest.mark.parametrize("domain", [(-1, 1), (1, -1)])
def test_otf_sdk_axes_do_not_change_compact_public_metadata(accuracy, flows, domain):
    p = process(accuracy=accuracy)
    compact = {
        k: v
        for k, v in p.physics.items()
        if k not in {"helicities", "color_components", "coverage", "extensions"}
    }
    process_ir = CanonicalProcessIR(
        process="x x > x x",
        key="x",
        color_accuracy=accuracy,
        legs=tuple(
            ProcessLegIR(
                label=i + 1,
                side="initial" if i < 2 else "final",
                particle="x",
                outgoing_particle="x",
                pdg=990,
                outgoing_pdg=990,
                statistics="bosonic",
                wavefunction_family="vector",
                color_role="adjoint",
                source_orientation="self-conjugate",
            )
            for i in range(4)
        ),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )
    seed = SimpleNamespace(
        external_sources=tuple(
            SimpleNamespace(
                public_label=i + 1,
                source_states=tuple(SimpleNamespace(public_helicity=h) for h in domain),
            )
            for i in reversed(range(4))
        )
    )
    resolved, sectors = build_umami_on_the_fly_physics(compact, process_ir, seed)
    assert "helicities" not in compact
    assert len(resolved["helicities"]) == 16
    assert resolved["helicities"][0]["id"] == "h:" + ",".join([f"{domain[0]:+d}"] * 4)
    assert resolved["helicities"][1]["values"] == [domain[0]] * 3 + [domain[1]]
    assert len(resolved["color_components"]) == flows
    (data,) = export(replace(p, physics=resolved, color_sectors=sectors))
    assert len(entries(data)) == (3 if accuracy == "lc" else 1)
    assert data["grouping"]["physical_contributions"] == flows
    restored = restore_umami_on_the_fly_input(replace(p, physics=compact), [data])
    assert export(restored) == (data,)
    assert "helicities" not in compact


def test_otf_append_recovers_grouped_physical_color_members():
    p = with_replay()
    p.physics["helicities"][0].update(
        id="h:+1,+1,+1,+1", representative_id="h:+1,+1,+1,+1"
    )
    (data,) = export(p)
    restored = restore_umami_on_the_fly_input(p, [data])
    assert [color["id"] for color in restored.physics["color_components"]] == [
        "flow:1,2,3,4",
        "flow:1,2,4,3",
    ]
    assert restored.color_flows == {
        member["color_id"]: data["color_flows"][member["color_flows"]]
        for member in entries(data)[0]["members"]
    }
    assert export(restored) == (data,)


def test_writer_append_recovers_old_compact_otf_sdk(tmp_path, monkeypatch):
    from pyamplicol.generation.artifact_writer import _write_umami_bundle

    p = process()
    p.physics["helicities"][0]["id"] = "h:+1,+1,+1,+1"
    (data,) = export(p)
    compact = {
        k: v
        for k, v in p.physics.items()
        if k not in {"helicities", "color_components"}
    }
    compact["kind"] = "pyamplicol-on-the-fly-public-metadata"
    files = {"API/umami/metadata.json": data, "processes/p/physics.json": compact}
    for path, value in files.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value), encoding="utf-8")
    written = {}
    builder = SimpleNamespace(
        staged_path=lambda path: tmp_path / path,
        payload_records=lambda: [],
        add_bytes=lambda path, content, **kwargs: written.setdefault(path, content),
    )
    _write_umami_bundle(
        builder,
        processes=(),
        process_records=[
            {
                "id": "p",
                "expression": "p",
                "color_accuracy": "lc",
                "external_pdgs": list(p.external_pdgs),
                "physics_path": "processes/p/physics.json",
            }
        ],
        compiled_model=SimpleNamespace(to_dict=model),
    )
    recovered = json.loads(written["API/umami/metadata.json"])
    assert recovered["provider"]["helicity_count"] == 1
    assert recovered["color_flows"] == data["color_flows"]
    assert len(entries(recovered)) == 1


def blind_member(identifier, pdgs, permutation=None, *, conjugated=False):
    return GroupMember(
        process_id=identifier,
        expression=identifier,
        pdgs=tuple(pdgs),
        initial_count=2,
        permutation=tuple(range(len(pdgs)) if permutation is None else permutation),
        conjugated=conjugated,
        basis=GroupingBasis.FLAVOR_BLIND,
        generated=False,
    )


def physical_flows(data, process_id):
    """Colour ID -> LHE tags, renumbered by first use, of one physical process."""

    def canonical(flow):
        names = {}
        return [
            [names.setdefault(tag, len(names) + 1) if tag else 0 for tag in pair]
            for pair in flow
        ]

    return {
        member["color_id"]: canonical(data["color_flows"][member["color_flows"]])
        for entry in entries(data)
        for member in entry["members"]
        if member["process_id"] == process_id
    }


def quark_gluon(identifier="a", pdgs=(991, 990, 991, 990)):
    # Every helicity of the incoming legs: a set invariant under their swap.
    p = process(identifier, pdgs=pdgs, words=((3, 2, 4, 1), (3, 4, 2, 1)))
    template = p.physics["helicities"][0]
    p.physics["helicities"][:] = [
        {**template, "id": f"h{i}", "representative_id": f"h{i}", "values": values}
        for i, values in enumerate(
            [[1, 1, 1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]]
        )
    ]
    return p


def test_flavor_blind_member_with_equal_incoming_flavours_is_an_extra_row():
    a = process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),))
    (data,) = export(
        replace(a, group_members=(blind_member("m", (990, 990, 992, -992)),))
    )
    (row,) = entries(data)
    assert [p["id"] for p in data["runtime_processes"]] == ["a"]
    assert data["pdg_ids"] == [[990, 990, 991, -991], [990, 990, 992, -992]]
    assert row["matrix_elements"] == [
        {"pdg_ids": 0, "factor": 1},
        {"pdg_ids": 1, "factor": 1},
    ]
    representative, member = row["members"]
    assert member["process_id"] == "m"
    assert member["pdgs"] == [990, 990, 992, -992]
    assert member["color_id"] == "flow:3,1,2,4"
    assert member["runtime"] == representative["runtime"]
    assert data["grouping"]["physical_contributions"] == 2
    assert data["grouping"]["exported_contributions"] == 1
    assert data["grouping"]["assumptions"][-1] == FLAVOR_BLIND_ASSUMPTION
    assert FLAVOR_BLIND_ASSUMPTION not in export(a)[0]["grouping"]["assumptions"]


def test_flavor_blind_member_with_swapped_incoming_legs_maps_axes():
    # "g u > u g" is "u g > u g" with the incoming legs exchanged.
    a = quark_gluon()
    member = blind_member("m", (990, 991, 991, 990), (1, 0, 2, 3))
    (data,) = export(replace(a, group_members=(member,)))
    assert [p["id"] for p in data["runtime_processes"]] == ["a"]
    rows = entries(data)
    # Different incoming flavours remain separate selectors.
    assert all(
        len({m["process_id"] for m in row["members"]}) == 1
        and len(row["matrix_elements"]) == 1
        for row in rows
    )
    members = [m for row in rows for m in row["members"] if m["process_id"] == "m"]
    assert len(members) == 2
    assert data["grouping"]["physical_contributions"] == 4
    runtime_values = [h["values"] for h in a.physics["helicities"]]
    for physical in members:
        runtime = physical["runtime"]
        assert runtime["process_id"] == "a" and runtime["process_index"] == 0
        assert runtime["permutation"] == [1, 0, 2, 3]
        assert [data["helicities"][i] for i in runtime["helicity_export"]] == [
            [v[1], v[0], *v[2:]] for v in runtime_values
        ]
    # The relabelled tags are those of the member exported on its own.
    reference = process("m", pdgs=member.pdgs, words=((3, 1, 4, 2), (3, 4, 1, 2)))
    assert physical_flows(data, "m") == physical_flows(export(reference)[0], "m")


def test_conjugated_flavor_blind_member_reverses_colour_flows():
    a = quark_gluon()
    member = blind_member("c", (-991, 990, -991, 990), conjugated=True)
    (data,) = export(replace(a, group_members=(member,)))
    members = [
        m for row in entries(data) for m in row["members"] if m["process_id"] == "c"
    ]
    assert {m["color_id"]: m["runtime"]["color_id"] for m in members} == {
        "flow:1,4,2,3": "flow:3,2,4,1",
        "flow:1,2,4,3": "flow:3,4,2,1",
    }
    reference = process("c", pdgs=member.pdgs, words=((1, 4, 2, 3), (1, 2, 4, 3)))
    assert physical_flows(data, "c") == physical_flows(export(reference)[0], "c")
    assert all(m["runtime"]["permutation"] == [0, 1, 2, 3] for m in members)


def test_member_moving_unequal_masses_reuses_runtime_in_its_own_provider():
    a = process("a", pdgs=(995, 990, 995, 990), words=((2, 4),))
    member = blind_member("m", (990, 995, 995, 990), (1, 0, 2, 3))
    first, second = export(replace(a, group_members=(member,)))
    assert first["provider"]["masses"] == [7, 0, 7, 0]
    assert second["provider"]["masses"] == [0, 7, 7, 0]
    assert FLAVOR_BLIND_ASSUMPTION not in first["grouping"]["assumptions"]
    # No runtime row of its own: the second provider loads the representative.
    assert [p["id"] for p in second["runtime_processes"]] == ["a"]
    assert {m["process_id"] for row in entries(second) for m in row["members"]} == {
        "m"
    }
    payloads = {
        p.path: p.content
        for p in umami_bundle_payloads(
            processes=[replace(a, group_members=(member,))], compiled_model=model()
        )
    }
    header = payloads["API/umami/provider_p1.h"].decode()
    assert '"a"' in header and '"m"' not in header


def test_otf_append_ignores_providers_loading_a_runtime_only_for_members():
    a = process("a", pdgs=(995, 990, 995, 990), words=((2, 4),))
    a.physics["helicities"][0].update(
        id="h:+1,+1,+1,+1", representative_id="h:+1,+1,+1,+1"
    )
    member = blind_member("m", (990, 995, 995, 990), (1, 0, 2, 3))
    a = replace(a, group_members=(member,))
    documents = export(a)
    assert len(documents) == 2
    assert export(restore_umami_on_the_fly_input(a, documents)) == documents


def test_flavours_with_different_runtime_permutations_are_not_merged():
    a = process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),))
    member = blind_member("m", (990, 990, 992, -992), (1, 0, 2, 3))
    (data,) = export(replace(a, group_members=(member,)))
    rows = entries(data)
    assert len(rows) == 2
    assert [len(row["matrix_elements"]) for row in rows] == [1, 1]
    assert rows[0]["runtime"]["permutation"] != rows[1]["runtime"]["permutation"]


def test_member_factor_multiplies_the_representative_runtime():
    # ME_member(p) = factor * ME_representative(q): a selector of its own.
    a = process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),))
    member = replace(blind_member("m", (990, 990, 992, -992)), factor=Fraction(1, 2))
    (data,) = export(replace(a, group_members=(member,)))
    assert [row["runtime"]["factor"] for row in entries(data)] == [1, 0.5]
    assert [row["matrix_elements"] for row in entries(data)] == [
        [{"pdg_ids": 0, "factor": 1}],
        [{"pdg_ids": 1, "factor": 1}],
    ]


def test_writer_exports_only_ungenerated_group_members(tmp_path):
    from pyamplicol.generation.artifact_writer import _write_umami_bundle

    a = process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),))
    b = process("b", pdgs=(992, -992, 990, 990), words=((2, 3, 4, 1),))
    groups = ProcessGroupMap()
    groups.add_group(
        [
            GroupMember.identity(GroupingProcess("a", "a", a.external_pdgs, 2)),
            replace(
                GroupMember.identity(GroupingProcess("b", "b", b.external_pdgs, 2)),
                basis=GroupingBasis.NUMERICAL,
            ),
            blind_member("m", (990, 990, 992, -992)),
        ]
    )
    records = []
    for p in (a, b):
        path = f"processes/{p.process_id}/physics.json"
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(json.dumps(p.physics), encoding="utf-8")
        records.append(
            {
                "id": p.process_id,
                "expression": p.expression,
                "color_accuracy": "lc",
                "external_pdgs": list(p.external_pdgs),
                "physics_path": path,
            }
        )
    written = {}
    builder = SimpleNamespace(
        staged_path=lambda path: tmp_path / path,
        payload_records=lambda: [],
        add_bytes=lambda path, content, **kwargs: written.setdefault(path, content),
    )
    _write_umami_bundle(
        builder,
        processes=(),
        process_records=records,
        compiled_model=SimpleNamespace(to_dict=model),
        process_groups=groups,
    )
    data = json.loads(written["API/umami/metadata.json"])
    assert [p["id"] for p in data["runtime_processes"]] == ["a", "b"]
    runtimes = {
        m["process_id"]: m["runtime"]["process_id"]
        for row in entries(data)
        for m in row["members"]
    }
    assert runtimes == {"a": "a", "b": "b", "m": "a"}
