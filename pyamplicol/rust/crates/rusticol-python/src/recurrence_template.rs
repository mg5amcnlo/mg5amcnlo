// SPDX-License-Identifier: 0BSD

//! Private Python boundary for model-wide recurrence-template input v1.
//!
//! The Python producer owns and freezes canonical NumPy columns. This module
//! checks that envelope before copying it into the strongly typed core input.
//! Prepared-kernel IDs can additionally be checked against an inventory from
//! an already authenticated prepared pack. Central registration must provide
//! that inventory before the validated template input is used for lowering.

use std::collections::{BTreeMap, BTreeSet};

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rusticol_core::recurrence::template::{
    CatalogHeaderRow, ClosureRow, ColorContractionRow, ColorNcTermRow, ContactOrbitCertificateRow,
    ContactOrbitStepRow, CouplingOrderTermRow, CurrentStateRow, DigestCatalogRow,
    EvaluatorBindingRow, ExactFactorRow, IndexedRangeRow, LCColorTransitionWitnessRow, MISSING_U32,
    OwnedRecurrenceTemplateInput, ParameterRow, PropagatorRow, QuantumFlowRow,
    QuantumNumberFlowTermRow, RECURRENCE_TEMPLATE_INPUT_ABI,
    RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION, RuntimeHelicityContractRow,
    RuntimeHelicityEmbeddingRow, RuntimeHelicityProjectionRow, RuntimeHelicityVariantRow,
    SourceRow, SymmetryProofRow, TransitionRow,
};
use rusticol_core::recurrence::{CheckedTableRange, SemanticDigest, checked_usize};
use rusticol_core::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};

use crate::python_error;

const RESULT_KIND: &str = "pyamplicol-recurrence-template-validation-result";
const RESULT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrimitiveKind {
    U8,
    U32,
    U64,
    I32,
}

impl PrimitiveKind {
    const fn dtype(self) -> &'static str {
        match self {
            Self::U8 => "|u1",
            Self::U32 => "<u4",
            Self::U64 => "<u8",
            Self::I32 => "<i4",
        }
    }
}

#[derive(Clone, Copy)]
struct ColumnSpec {
    name: &'static str,
    kind: PrimitiveKind,
    tail_shape: &'static [usize],
}

#[derive(Clone, Copy)]
struct TableSpec {
    name: &'static str,
    columns: &'static [ColumnSpec],
}

const fn column(name: &'static str, kind: PrimitiveKind) -> ColumnSpec {
    ColumnSpec {
        name,
        kind,
        tail_shape: &[],
    }
}

const fn shaped_column(
    name: &'static str,
    kind: PrimitiveKind,
    tail_shape: &'static [usize],
) -> ColumnSpec {
    ColumnSpec {
        name,
        kind,
        tail_shape,
    }
}

const TABLE_SPECS: &[TableSpec] = &[
    TableSpec {
        name: "catalog_header",
        columns: &[
            column("schema_version", PrimitiveKind::U32),
            column("abi_string_id", PrimitiveKind::U32),
            column("canonicalization_abi_string_id", PrimitiveKind::U32),
            column("exact_scalar_abi_string_id", PrimitiveKind::U32),
            column("compiled_model_digest_id", PrimitiveKind::U32),
            column("prepared_kernel_pack_digest_id", PrimitiveKind::U32),
            column("catalog_digest_id", PrimitiveKind::U32),
            column("parameter_count", PrimitiveKind::U32),
            column("current_state_count", PrimitiveKind::U32),
            column("source_count", PrimitiveKind::U32),
            column("quantum_flow_count", PrimitiveKind::U32),
            column("contact_orbit_certificate_count", PrimitiveKind::U32),
            column("contact_orbit_step_count", PrimitiveKind::U32),
            column("transition_count", PrimitiveKind::U32),
            column("propagator_count", PrimitiveKind::U32),
            column("closure_count", PrimitiveKind::U32),
            column("color_contraction_count", PrimitiveKind::U32),
            column("symmetry_proof_count", PrimitiveKind::U32),
            column("runtime_helicity_contract_count", PrimitiveKind::U32),
            column("evaluator_binding_count", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "closures",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("input_state_sequence_id", PrimitiveKind::U32),
            column("result_state_template_id", PrimitiveKind::U32),
            column("evaluator_binding_id", PrimitiveKind::U32),
            column("canonical_input_order_sequence_id", PrimitiveKind::U32),
            column("coupling_parameter_sequence_id", PrimitiveKind::U32),
            column("coupling_order_set_id", PrimitiveKind::U32),
            column("eligible_quantum_flow_sequence_id", PrimitiveKind::U32),
            column("color_contraction_template_id", PrimitiveKind::U32),
            column("binding_coupling_factor_id", PrimitiveKind::U32),
            column("exact_factor_id", PrimitiveKind::U32),
            column("output_factor_source", PrimitiveKind::U8),
            column("equivalence_class_string_id", PrimitiveKind::U32),
            column("input_exchange_factor_id", PrimitiveKind::U32),
            column("projection_string_id", PrimitiveKind::U32),
            column("component_coefficient_sequence_id", PrimitiveKind::U32),
            column("chirality_relation_string_id", PrimitiveKind::U32),
            column("metric_signature_string_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "color_contractions",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("rule_kind_string_id", PrimitiveKind::U32),
            column("input_representation_sequence_id", PrimitiveKind::U32),
            column("has_output_representation", PrimitiveKind::U8),
            column("output_representation", PrimitiveKind::I32),
            column("ordered_open_string_arity", PrimitiveKind::U32),
            column("exact_coefficient_factor_id", PrimitiveKind::U32),
            column("witness_start", PrimitiveKind::U64),
            column("witness_count", PrimitiveKind::U64),
            column("nc_term_start", PrimitiveKind::U64),
            column("nc_term_count", PrimitiveKind::U64),
            column("expression_digest_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "color_nc_terms",
        columns: &[
            column("color_contraction_id", PrimitiveKind::U32),
            column("exponent", PrimitiveKind::I32),
            column("factor_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "contact_orbit_certificates",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("algorithm_string_id", PrimitiveKind::U32),
            column("algorithm_version", PrimitiveKind::U32),
            column("term_id", PrimitiveKind::U32),
            column("vertex_string_id", PrimitiveKind::U32),
            column("particle_string_sequence_id", PrimitiveKind::U32),
            column("evaluator_class_string_id", PrimitiveKind::U32),
            column("physical_leg_equivalence_sequence_id", PrimitiveKind::U32),
            column("reconstruction_factor_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "contact_orbit_steps",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("certificate_id", PrimitiveKind::U32),
            column("stage", PrimitiveKind::U8),
            column("result_leg", PrimitiveKind::U8),
            column("left_covered_leg_sequence_id", PrimitiveKind::U32),
            column("right_covered_leg_sequence_id", PrimitiveKind::U32),
            column("source_particle_leg_sequence_id", PrimitiveKind::U32),
            column("reconstruction_factor_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "coupling_order_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "coupling_order_terms",
        columns: &[
            column("set_id", PrimitiveKind::U32),
            column("name_string_id", PrimitiveKind::U32),
            column("power", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "current_states",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("particle_id", PrimitiveKind::I32),
            column("anti_particle_id", PrimitiveKind::I32),
            column("species_string_id", PrimitiveKind::U32),
            column("orientation", PrimitiveKind::U8),
            column("statistics", PrimitiveKind::U8),
            column("color_representation", PrimitiveKind::I32),
            column("basis_string_id", PrimitiveKind::U32),
            column("tensor_ordering_sequence_id", PrimitiveKind::U32),
            column("dimension", PrimitiveKind::U32),
            column("chirality", PrimitiveKind::I32),
            column("lc_color_shape_string_id", PrimitiveKind::U32),
            column("auxiliary_kind_string_id", PrimitiveKind::U32),
            column("mass_parameter_id", PrimitiveKind::U32),
            column("width_parameter_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "digest_catalog",
        columns: &[
            column("id", PrimitiveKind::U32),
            shaped_column("value", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "evaluator_bindings",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("resolver_key_string_id", PrimitiveKind::U32),
            column("prepared_kernel_id", PrimitiveKind::U32),
            column("contract_kind", PrimitiveKind::U8),
            column("callable_signature_digest_id", PrimitiveKind::U32),
            column("input_state_sequence_id", PrimitiveKind::U32),
            column("output_state_template_id", PrimitiveKind::U32),
            column("input_layout_sequence_id", PrimitiveKind::U32),
            column("output_layout_sequence_id", PrimitiveKind::U32),
            column("exact_expression_digest_sequence_id", PrimitiveKind::U32),
            column("semantic_template_sequence_id", PrimitiveKind::U32),
            column("callable_kind", PrimitiveKind::U8),
            column("runtime_template_string_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "exact_factors",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("real_numerator_string_id", PrimitiveKind::U32),
            column("real_denominator_string_id", PrimitiveKind::U32),
            column("imag_numerator_string_id", PrimitiveKind::U32),
            column("imag_denominator_string_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "flavour_flow_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "flavour_flow_values",
        columns: &[column("value", PrimitiveKind::I32)],
    },
    TableSpec {
        name: "i32_sequence_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "i32_sequence_values",
        columns: &[column("value", PrimitiveKind::I32)],
    },
    TableSpec {
        name: "lc_color_transition_witnesses",
        columns: &[
            column("color_contraction_id", PrimitiveKind::U32),
            column("ordinal", PrimitiveKind::U32),
            column("left_shape_string_id", PrimitiveKind::U32),
            column("right_shape_string_id", PrimitiveKind::U32),
            column("input_permutation", PrimitiveKind::U8),
            column("reverse_parent_mask", PrimitiveKind::U8),
            column("component_operation", PrimitiveKind::U8),
            column("result_component_kind", PrimitiveKind::U8),
            column("result_component_role", PrimitiveKind::U8),
            column("result_shape_string_id", PrimitiveKind::U32),
            column("exact_factor_id", PrimitiveKind::U32),
            column("proof_digest_id", PrimitiveKind::U32),
            column("input_port_pairing_sequence_id", PrimitiveKind::U32),
            column("result_port_binding_sequence_id", PrimitiveKind::U32),
            column("provenance_sequence_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "parameters",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("name_string_id", PrimitiveKind::U32),
            column("kind", PrimitiveKind::U8),
            column("value_type", PrimitiveKind::U8),
            column("mutable", PrimitiveKind::U8),
            column("default_factor_id", PrimitiveKind::U32),
            column("exact_expression_digest_id", PrimitiveKind::U32),
            column("dependency_sequence_id", PrimitiveKind::U32),
            column("prepared_parameter_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "propagators",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("state_template_id", PrimitiveKind::U32),
            column("applies_propagator", PrimitiveKind::U8),
            column("evaluator_binding_id", PrimitiveKind::U32),
            column("numerator_expression_digest_id", PrimitiveKind::U32),
            column("denominator_expression_digest_id", PrimitiveKind::U32),
            column("mass_parameter_id", PrimitiveKind::U32),
            column("width_parameter_id", PrimitiveKind::U32),
            column("gauge_string_id", PrimitiveKind::U32),
            column("linearity_proof_template_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "quantum_flows",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("input_state_sequence_id", PrimitiveKind::U32),
            column("input_spin_sequence_id", PrimitiveKind::U32),
            column("input_flavour_sequence_id", PrimitiveKind::U32),
            column("input_quantum_sequence_id", PrimitiveKind::U32),
            column("flavour_flow_operation_string_id", PrimitiveKind::U32),
            column(
                "quantum_number_flow_operation_string_id",
                PrimitiveKind::U32,
            ),
            column("coupling_order_set_id", PrimitiveKind::U32),
            column("result_state_template_id", PrimitiveKind::U32),
            column("result_spin_state", PrimitiveKind::I32),
            column("result_flavour_flow_id", PrimitiveKind::U32),
            column("result_quantum_number_flow_id", PrimitiveKind::U32),
            column("exact_coupling_factor_id", PrimitiveKind::U32),
            column("predicate_digest_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "quantum_number_flow_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "quantum_number_flow_terms",
        columns: &[
            column("flow_id", PrimitiveKind::U32),
            column("name_string_id", PrimitiveKind::U32),
            column("expression_string_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "runtime_helicity_contracts",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("full_state_template_id", PrimitiveKind::U32),
            column("variant_offset", PrimitiveKind::U32),
            column("variant_count", PrimitiveKind::U32),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            column("proof_digest_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "runtime_helicity_embeddings",
        columns: &[
            column("variant_id", PrimitiveKind::U32),
            column("full_component", PrimitiveKind::U32),
            column("source_component", PrimitiveKind::U32),
            column("factor_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "runtime_helicity_projections",
        columns: &[
            column("variant_id", PrimitiveKind::U32),
            column("source_component", PrimitiveKind::U32),
            column("full_component", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "runtime_helicity_variants",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("contract_id", PrimitiveKind::U32),
            column("source_template_id", PrimitiveKind::U32),
            column("source_state_template_id", PrimitiveKind::U32),
            column("embedding_offset", PrimitiveKind::U32),
            column("embedding_count", PrimitiveKind::U32),
            column("projection_offset", PrimitiveKind::U32),
            column("projection_count", PrimitiveKind::U32),
            column("proof_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "sources",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("state_template_id", PrimitiveKind::U32),
            column("crossing_string_id", PrimitiveKind::U32),
            column("wavefunction_family_string_id", PrimitiveKind::U32),
            column("helicity", PrimitiveKind::I32),
            column("spin_state", PrimitiveKind::I32),
            column("flavour_flow_id", PrimitiveKind::U32),
            column("quantum_number_flow_id", PrimitiveKind::U32),
            column("lc_color_seed_operation", PrimitiveKind::U8),
            column("lc_color_seed_shape_string_id", PrimitiveKind::U32),
            column("lc_color_seed_component_kind", PrimitiveKind::U8),
            column("lc_color_seed_component_role", PrimitiveKind::U8),
            column("lc_color_seed_proof_digest_id", PrimitiveKind::U32),
            column("lc_color_seed_provenance_sequence_id", PrimitiveKind::U32),
            column("wavefunction_expression_digest_id", PrimitiveKind::U32),
            column("evaluator_binding_id", PrimitiveKind::U32),
            column("mass_parameter_id", PrimitiveKind::U32),
            column("width_parameter_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "string_bytes",
        columns: &[column("value", PrimitiveKind::U8)],
    },
    TableSpec {
        name: "string_ranges",
        columns: &[
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "symmetry_proofs",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            column("subject_template_sequence_id", PrimitiveKind::U32),
            column("input_permutation_sequence_id", PrimitiveKind::U32),
            column("exact_phase_factor_id", PrimitiveKind::U32),
            column("expression_digest_sequence_id", PrimitiveKind::U32),
            column("witness_digest_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "transitions",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("template_string_id", PrimitiveKind::U32),
            column("input_state_sequence_id", PrimitiveKind::U32),
            column("result_state_template_id", PrimitiveKind::U32),
            column("quantum_flow_template_id", PrimitiveKind::U32),
            column("evaluator_binding_id", PrimitiveKind::U32),
            column("canonical_input_order_sequence_id", PrimitiveKind::U32),
            column("momentum_convention_sequence_id", PrimitiveKind::U32),
            column("coupling_parameter_sequence_id", PrimitiveKind::U32),
            column("coupling_order_set_id", PrimitiveKind::U32),
            column("color_contraction_template_id", PrimitiveKind::U32),
            column("binding_coupling_factor_id", PrimitiveKind::U32),
            column("exact_factor_id", PrimitiveKind::U32),
            column("output_factor_source", PrimitiveKind::U8),
            column("equivalence_class_string_id", PrimitiveKind::U32),
            column("input_exchange_factor_id", PrimitiveKind::U32),
            column("output_projection_string_id", PrimitiveKind::U32),
            column("contact_orbit_step_sequence_id", PrimitiveKind::U32),
            column(
                "contact_orbit_step_semantic_digest_sequence_id",
                PrimitiveKind::U32,
            ),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "u32_sequence_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "u32_sequence_values",
        columns: &[column("value", PrimitiveKind::U32)],
    },
];

#[derive(Clone)]
enum OwnedValues {
    U8(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
}

impl OwnedValues {
    fn raw_bytes(&self) -> &[u8] {
        fn bytes<T>(values: &[T]) -> &[u8] {
            // Multi-byte inputs are explicitly little-endian and big-endian
            // hosts are rejected before NumPy extraction.
            unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            }
        }
        match self {
            Self::U8(values) => values,
            Self::U32(values) => bytes(values),
            Self::U64(values) => bytes(values),
            Self::I32(values) => bytes(values),
        }
    }

    fn as_u8(&self, context: &str) -> RusticolResult<&[u8]> {
        match self {
            Self::U8(values) => Ok(values),
            _ => Err(wrong_type(context, "u8")),
        }
    }

    fn as_u32(&self, context: &str) -> RusticolResult<&[u32]> {
        match self {
            Self::U32(values) => Ok(values),
            _ => Err(wrong_type(context, "u32")),
        }
    }

    fn as_u64(&self, context: &str) -> RusticolResult<&[u64]> {
        match self {
            Self::U64(values) => Ok(values),
            _ => Err(wrong_type(context, "u64")),
        }
    }

    fn as_i32(&self, context: &str) -> RusticolResult<&[i32]> {
        match self {
            Self::I32(values) => Ok(values),
            _ => Err(wrong_type(context, "i32")),
        }
    }
}

#[derive(Clone)]
struct OwnedColumn {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    values: OwnedValues,
}

#[derive(Clone)]
struct OwnedTable {
    name: String,
    row_count: u64,
    columns: Vec<OwnedColumn>,
    column_by_name: BTreeMap<String, usize>,
}

impl OwnedTable {
    fn row_count(&self) -> RusticolResult<usize> {
        checked_usize(self.row_count, &format!("{} row count", self.name))
    }

    fn column(&self, name: &str) -> RusticolResult<&OwnedColumn> {
        let index = self.column_by_name.get(name).ok_or_else(|| {
            invalid(format!(
                "recurrence template table {:?} has no column {name:?}",
                self.name
            ))
        })?;
        Ok(&self.columns[*index])
    }

    fn u8(&self, column: &str) -> RusticolResult<&[u8]> {
        self.column(column)?
            .values
            .as_u8(&format!("{}.{column}", self.name))
    }

    fn u32(&self, column: &str) -> RusticolResult<&[u32]> {
        self.column(column)?
            .values
            .as_u32(&format!("{}.{column}", self.name))
    }

    fn u64(&self, column: &str) -> RusticolResult<&[u64]> {
        self.column(column)?
            .values
            .as_u64(&format!("{}.{column}", self.name))
    }

    fn i32(&self, column: &str) -> RusticolResult<&[i32]> {
        self.column(column)?
            .values
            .as_i32(&format!("{}.{column}", self.name))
    }
}

struct DecodedInput {
    abi: String,
    declared_digest: String,
    canonical_digest_property: String,
    catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
    tables: Vec<OwnedTable>,
    table_by_name: BTreeMap<String, usize>,
}

impl DecodedInput {
    fn table(&self, name: &str) -> RusticolResult<&OwnedTable> {
        let index = self
            .table_by_name
            .get(name)
            .ok_or_else(|| invalid(format!("recurrence template input has no table {name:?}")))?;
        Ok(&self.tables[*index])
    }

    fn canonical_digest(&self) -> RusticolResult<String> {
        let mut digest = Sha256::new();
        hash_text(&mut digest, &self.abi)?;
        digest.update(self.catalog_digest.as_bytes());
        digest.update(self.compiled_model_digest.as_bytes());
        digest.update(self.prepared_kernel_pack_digest.as_bytes());
        digest.update(
            u64::try_from(self.tables.len())
                .map_err(|_| invalid("recurrence template table count exceeds u64"))?
                .to_le_bytes(),
        );
        for table in &self.tables {
            hash_text(&mut digest, &table.name)?;
            digest.update(table.row_count.to_le_bytes());
            digest.update(
                u32::try_from(table.columns.len())
                    .map_err(|_| invalid("recurrence template column count exceeds u32"))?
                    .to_le_bytes(),
            );
            for column in &table.columns {
                hash_text(&mut digest, &column.name)?;
                hash_text(&mut digest, column.dtype)?;
                digest.update(
                    u8::try_from(column.shape.len())
                        .map_err(|_| invalid("recurrence template column rank exceeds u8"))?
                        .to_le_bytes(),
                );
                for dimension in &column.shape {
                    digest.update(dimension.to_le_bytes());
                }
                digest.update(column.values.raw_bytes());
            }
        }
        Ok(hex_digest(digest.finalize()))
    }

    fn into_core(self) -> RusticolResult<OwnedRecurrenceTemplateInput> {
        let catalog_header = decode_rows(self.table("catalog_header")?, |table, row| {
            Ok(CatalogHeaderRow {
                schema_version: table.u32("schema_version")?[row],
                abi_string_id: table.u32("abi_string_id")?[row],
                canonicalization_abi_string_id: table.u32("canonicalization_abi_string_id")?[row],
                exact_scalar_abi_string_id: table.u32("exact_scalar_abi_string_id")?[row],
                compiled_model_digest_id: table.u32("compiled_model_digest_id")?[row],
                prepared_kernel_pack_digest_id: table.u32("prepared_kernel_pack_digest_id")?[row],
                catalog_digest_id: table.u32("catalog_digest_id")?[row],
                parameter_count: table.u32("parameter_count")?[row],
                current_state_count: table.u32("current_state_count")?[row],
                source_count: table.u32("source_count")?[row],
                quantum_flow_count: table.u32("quantum_flow_count")?[row],
                contact_orbit_certificate_count: table.u32("contact_orbit_certificate_count")?[row],
                contact_orbit_step_count: table.u32("contact_orbit_step_count")?[row],
                transition_count: table.u32("transition_count")?[row],
                propagator_count: table.u32("propagator_count")?[row],
                closure_count: table.u32("closure_count")?[row],
                color_contraction_count: table.u32("color_contraction_count")?[row],
                symmetry_proof_count: table.u32("symmetry_proof_count")?[row],
                runtime_helicity_contract_count: table.u32("runtime_helicity_contract_count")?[row],
                evaluator_binding_count: table.u32("evaluator_binding_count")?[row],
            })
        })?;
        let coupling_order_ranges = indexed_ranges(self.table("coupling_order_ranges")?)?;
        let coupling_order_terms =
            decode_rows(self.table("coupling_order_terms")?, |table, row| {
                Ok(CouplingOrderTermRow {
                    set_id: table.u32("set_id")?[row],
                    name_string_id: table.u32("name_string_id")?[row],
                    power: table.u32("power")?[row],
                })
            })?;
        let contact_orbit_certificates =
            decode_rows(self.table("contact_orbit_certificates")?, |table, row| {
                Ok(ContactOrbitCertificateRow {
                    id: table.u32("id")?[row],
                    template_string_id: table.u32("template_string_id")?[row],
                    algorithm_string_id: table.u32("algorithm_string_id")?[row],
                    algorithm_version: table.u32("algorithm_version")?[row],
                    term_id: table.u32("term_id")?[row],
                    vertex_string_id: table.u32("vertex_string_id")?[row],
                    particle_string_sequence_id: table.u32("particle_string_sequence_id")?[row],
                    evaluator_class_string_id: table.u32("evaluator_class_string_id")?[row],
                    physical_leg_equivalence_sequence_id: table
                        .u32("physical_leg_equivalence_sequence_id")?[row],
                    reconstruction_factor_id: table.u32("reconstruction_factor_id")?[row],
                    semantic_digest_id: table.u32("semantic_digest_id")?[row],
                })
            })?;
        let contact_orbit_steps = decode_rows(self.table("contact_orbit_steps")?, |table, row| {
            Ok(ContactOrbitStepRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                certificate_id: table.u32("certificate_id")?[row],
                stage: table.u8("stage")?[row],
                result_leg: table.u8("result_leg")?[row],
                left_covered_leg_sequence_id: table.u32("left_covered_leg_sequence_id")?[row],
                right_covered_leg_sequence_id: table.u32("right_covered_leg_sequence_id")?[row],
                source_particle_leg_sequence_id: table.u32("source_particle_leg_sequence_id")?[row],
                reconstruction_factor_id: table.u32("reconstruction_factor_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let current_states = decode_rows(self.table("current_states")?, |table, row| {
            Ok(CurrentStateRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                particle_id: table.i32("particle_id")?[row],
                anti_particle_id: table.i32("anti_particle_id")?[row],
                species_string_id: table.u32("species_string_id")?[row],
                orientation: table.u8("orientation")?[row],
                statistics: table.u8("statistics")?[row],
                color_representation: table.i32("color_representation")?[row],
                basis_string_id: table.u32("basis_string_id")?[row],
                tensor_ordering_sequence_id: table.u32("tensor_ordering_sequence_id")?[row],
                dimension: table.u32("dimension")?[row],
                chirality: table.i32("chirality")?[row],
                lc_color_shape_string_id: table.u32("lc_color_shape_string_id")?[row],
                auxiliary_kind_string_id: table.u32("auxiliary_kind_string_id")?[row],
                mass_parameter_id: table.u32("mass_parameter_id")?[row],
                width_parameter_id: table.u32("width_parameter_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let digest_catalog = decode_digest_catalog(self.table("digest_catalog")?)?;
        let evaluator_bindings = decode_rows(self.table("evaluator_bindings")?, |table, row| {
            Ok(EvaluatorBindingRow {
                id: table.u32("id")?[row],
                resolver_key_string_id: table.u32("resolver_key_string_id")?[row],
                prepared_kernel_id: table.u32("prepared_kernel_id")?[row],
                contract_kind: table.u8("contract_kind")?[row],
                callable_signature_digest_id: table.u32("callable_signature_digest_id")?[row],
                input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
                output_state_template_id: table.u32("output_state_template_id")?[row],
                input_layout_sequence_id: table.u32("input_layout_sequence_id")?[row],
                output_layout_sequence_id: table.u32("output_layout_sequence_id")?[row],
                exact_expression_digest_sequence_id: table
                    .u32("exact_expression_digest_sequence_id")?[row],
                semantic_template_sequence_id: table.u32("semantic_template_sequence_id")?[row],
                callable_kind: table.u8("callable_kind")?[row],
                runtime_template_string_id: table.u32("runtime_template_string_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let exact_factors = decode_rows(self.table("exact_factors")?, |table, row| {
            Ok(ExactFactorRow {
                id: table.u32("id")?[row],
                real_numerator_string_id: table.u32("real_numerator_string_id")?[row],
                real_denominator_string_id: table.u32("real_denominator_string_id")?[row],
                imag_numerator_string_id: table.u32("imag_numerator_string_id")?[row],
                imag_denominator_string_id: table.u32("imag_denominator_string_id")?[row],
            })
        })?;
        let flavour_flow_ranges = indexed_ranges(self.table("flavour_flow_ranges")?)?;
        let flavour_flow_values = self.table("flavour_flow_values")?.i32("value")?.to_vec();
        let i32_sequence_ranges = indexed_ranges(self.table("i32_sequence_ranges")?)?;
        let i32_sequence_values = self.table("i32_sequence_values")?.i32("value")?.to_vec();
        let parameters = decode_rows(self.table("parameters")?, |table, row| {
            Ok(ParameterRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                name_string_id: table.u32("name_string_id")?[row],
                kind: table.u8("kind")?[row],
                value_type: table.u8("value_type")?[row],
                mutable: table.u8("mutable")?[row],
                default_factor_id: table.u32("default_factor_id")?[row],
                exact_expression_digest_id: table.u32("exact_expression_digest_id")?[row],
                dependency_sequence_id: table.u32("dependency_sequence_id")?[row],
                prepared_parameter_id: table.u32("prepared_parameter_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let propagators = decode_rows(self.table("propagators")?, |table, row| {
            Ok(PropagatorRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                state_template_id: table.u32("state_template_id")?[row],
                applies_propagator: table.u8("applies_propagator")?[row],
                evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
                numerator_expression_digest_id: table.u32("numerator_expression_digest_id")?[row],
                denominator_expression_digest_id: table.u32("denominator_expression_digest_id")?
                    [row],
                mass_parameter_id: table.u32("mass_parameter_id")?[row],
                width_parameter_id: table.u32("width_parameter_id")?[row],
                gauge_string_id: table.u32("gauge_string_id")?[row],
                linearity_proof_template_id: table.u32("linearity_proof_template_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let quantum_flows = decode_rows(self.table("quantum_flows")?, |table, row| {
            Ok(QuantumFlowRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
                input_spin_sequence_id: table.u32("input_spin_sequence_id")?[row],
                input_flavour_sequence_id: table.u32("input_flavour_sequence_id")?[row],
                input_quantum_sequence_id: table.u32("input_quantum_sequence_id")?[row],
                flavour_flow_operation_string_id: table.u32("flavour_flow_operation_string_id")?
                    [row],
                quantum_number_flow_operation_string_id: table
                    .u32("quantum_number_flow_operation_string_id")?[row],
                coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
                result_state_template_id: table.u32("result_state_template_id")?[row],
                result_spin_state: table.i32("result_spin_state")?[row],
                result_flavour_flow_id: table.u32("result_flavour_flow_id")?[row],
                result_quantum_number_flow_id: table.u32("result_quantum_number_flow_id")?[row],
                exact_coupling_factor_id: table.u32("exact_coupling_factor_id")?[row],
                predicate_digest_id: table.u32("predicate_digest_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let quantum_number_flow_ranges = indexed_ranges(self.table("quantum_number_flow_ranges")?)?;
        let quantum_number_flow_terms =
            decode_rows(self.table("quantum_number_flow_terms")?, |table, row| {
                Ok(QuantumNumberFlowTermRow {
                    flow_id: table.u32("flow_id")?[row],
                    name_string_id: table.u32("name_string_id")?[row],
                    expression_string_id: table.u32("expression_string_id")?[row],
                })
            })?;
        let runtime_helicity_contracts =
            decode_rows(self.table("runtime_helicity_contracts")?, |table, row| {
                Ok(RuntimeHelicityContractRow {
                    id: table.u32("id")?[row],
                    template_string_id: table.u32("template_string_id")?[row],
                    full_state_template_id: table.u32("full_state_template_id")?[row],
                    variant_range: CheckedTableRange::new(
                        table.u32("variant_offset")?[row].into(),
                        table.u32("variant_count")?[row].into(),
                    ),
                    proof_algorithm_string_id: table.u32("proof_algorithm_string_id")?[row],
                    proof_digest_id: table.u32("proof_digest_id")?[row],
                    semantic_digest_id: table.u32("semantic_digest_id")?[row],
                })
            })?;
        let runtime_helicity_embeddings =
            decode_rows(self.table("runtime_helicity_embeddings")?, |table, row| {
                Ok(RuntimeHelicityEmbeddingRow {
                    variant_id: table.u32("variant_id")?[row],
                    full_component: table.u32("full_component")?[row],
                    source_component: table.u32("source_component")?[row],
                    factor_id: table.u32("factor_id")?[row],
                })
            })?;
        let runtime_helicity_projections =
            decode_rows(self.table("runtime_helicity_projections")?, |table, row| {
                Ok(RuntimeHelicityProjectionRow {
                    variant_id: table.u32("variant_id")?[row],
                    source_component: table.u32("source_component")?[row],
                    full_component: table.u32("full_component")?[row],
                })
            })?;
        let runtime_helicity_variants =
            decode_rows(self.table("runtime_helicity_variants")?, |table, row| {
                Ok(RuntimeHelicityVariantRow {
                    id: table.u32("id")?[row],
                    contract_id: table.u32("contract_id")?[row],
                    source_template_id: table.u32("source_template_id")?[row],
                    source_state_template_id: table.u32("source_state_template_id")?[row],
                    embedding_range: CheckedTableRange::new(
                        table.u32("embedding_offset")?[row].into(),
                        table.u32("embedding_count")?[row].into(),
                    ),
                    projection_range: CheckedTableRange::new(
                        table.u32("projection_offset")?[row].into(),
                        table.u32("projection_count")?[row].into(),
                    ),
                    proof_digest_id: table.u32("proof_digest_id")?[row],
                })
            })?;
        let sources = decode_rows(self.table("sources")?, |table, row| {
            Ok(SourceRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                state_template_id: table.u32("state_template_id")?[row],
                crossing_string_id: table.u32("crossing_string_id")?[row],
                wavefunction_family_string_id: table.u32("wavefunction_family_string_id")?[row],
                helicity: table.i32("helicity")?[row],
                spin_state: table.i32("spin_state")?[row],
                flavour_flow_id: table.u32("flavour_flow_id")?[row],
                quantum_number_flow_id: table.u32("quantum_number_flow_id")?[row],
                lc_color_seed_operation: table.u8("lc_color_seed_operation")?[row],
                lc_color_seed_shape_string_id: table.u32("lc_color_seed_shape_string_id")?[row],
                lc_color_seed_component_kind: table.u8("lc_color_seed_component_kind")?[row],
                lc_color_seed_component_role: table.u8("lc_color_seed_component_role")?[row],
                lc_color_seed_proof_digest_id: table.u32("lc_color_seed_proof_digest_id")?[row],
                lc_color_seed_provenance_sequence_id: table
                    .u32("lc_color_seed_provenance_sequence_id")?[row],
                wavefunction_expression_digest_id: table
                    .u32("wavefunction_expression_digest_id")?[row],
                evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
                mass_parameter_id: table.u32("mass_parameter_id")?[row],
                width_parameter_id: table.u32("width_parameter_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let string_ranges = plain_ranges(self.table("string_ranges")?)?;
        let string_bytes = self.table("string_bytes")?.u8("value")?.to_vec();
        let symmetry_proofs = decode_rows(self.table("symmetry_proofs")?, |table, row| {
            Ok(SymmetryProofRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                proof_algorithm_string_id: table.u32("proof_algorithm_string_id")?[row],
                subject_template_sequence_id: table.u32("subject_template_sequence_id")?[row],
                input_permutation_sequence_id: table.u32("input_permutation_sequence_id")?[row],
                exact_phase_factor_id: table.u32("exact_phase_factor_id")?[row],
                expression_digest_sequence_id: table.u32("expression_digest_sequence_id")?[row],
                witness_digest_id: table.u32("witness_digest_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let transitions = decode_rows(self.table("transitions")?, |table, row| {
            Ok(TransitionRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
                result_state_template_id: table.u32("result_state_template_id")?[row],
                quantum_flow_template_id: table.u32("quantum_flow_template_id")?[row],
                evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
                canonical_input_order_sequence_id: table
                    .u32("canonical_input_order_sequence_id")?[row],
                momentum_convention_sequence_id: table.u32("momentum_convention_sequence_id")?[row],
                coupling_parameter_sequence_id: table.u32("coupling_parameter_sequence_id")?[row],
                coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
                color_contraction_template_id: table.u32("color_contraction_template_id")?[row],
                binding_coupling_factor_id: table.u32("binding_coupling_factor_id")?[row],
                exact_factor_id: table.u32("exact_factor_id")?[row],
                output_factor_source: table.u8("output_factor_source")?[row],
                equivalence_class_string_id: table.u32("equivalence_class_string_id")?[row],
                input_exchange_factor_id: table.u32("input_exchange_factor_id")?[row],
                output_projection_string_id: table.u32("output_projection_string_id")?[row],
                contact_orbit_step_sequence_id: table.u32("contact_orbit_step_sequence_id")?[row],
                contact_orbit_step_semantic_digest_sequence_id: table
                    .u32("contact_orbit_step_semantic_digest_sequence_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let closures = decode_rows(self.table("closures")?, |table, row| {
            Ok(ClosureRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
                result_state_template_id: table.u32("result_state_template_id")?[row],
                evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
                canonical_input_order_sequence_id: table
                    .u32("canonical_input_order_sequence_id")?[row],
                coupling_parameter_sequence_id: table.u32("coupling_parameter_sequence_id")?[row],
                coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
                eligible_quantum_flow_sequence_id: table
                    .u32("eligible_quantum_flow_sequence_id")?[row],
                color_contraction_template_id: table.u32("color_contraction_template_id")?[row],
                binding_coupling_factor_id: table.u32("binding_coupling_factor_id")?[row],
                exact_factor_id: table.u32("exact_factor_id")?[row],
                output_factor_source: table.u8("output_factor_source")?[row],
                equivalence_class_string_id: table.u32("equivalence_class_string_id")?[row],
                input_exchange_factor_id: table.u32("input_exchange_factor_id")?[row],
                projection_string_id: table.u32("projection_string_id")?[row],
                component_coefficient_sequence_id: table
                    .u32("component_coefficient_sequence_id")?[row],
                chirality_relation_string_id: table.u32("chirality_relation_string_id")?[row],
                metric_signature_string_id: table.u32("metric_signature_string_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let color_contractions = decode_rows(self.table("color_contractions")?, |table, row| {
            Ok(ColorContractionRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                rule_kind_string_id: table.u32("rule_kind_string_id")?[row],
                input_representation_sequence_id: table.u32("input_representation_sequence_id")?
                    [row],
                has_output_representation: table.u8("has_output_representation")?[row],
                output_representation: table.i32("output_representation")?[row],
                ordered_open_string_arity: table.u32("ordered_open_string_arity")?[row],
                exact_coefficient_factor_id: table.u32("exact_coefficient_factor_id")?[row],
                witness_start: table.u64("witness_start")?[row],
                witness_count: table.u64("witness_count")?[row],
                nc_term_start: table.u64("nc_term_start")?[row],
                nc_term_count: table.u64("nc_term_count")?[row],
                expression_digest_id: table.u32("expression_digest_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
        let lc_color_transition_witnesses = decode_rows(
            self.table("lc_color_transition_witnesses")?,
            |table, row| {
                Ok(LCColorTransitionWitnessRow {
                    color_contraction_id: table.u32("color_contraction_id")?[row],
                    ordinal: table.u32("ordinal")?[row],
                    left_shape_string_id: table.u32("left_shape_string_id")?[row],
                    right_shape_string_id: table.u32("right_shape_string_id")?[row],
                    input_permutation: table.u8("input_permutation")?[row],
                    reverse_parent_mask: table.u8("reverse_parent_mask")?[row],
                    component_operation: table.u8("component_operation")?[row],
                    result_component_kind: table.u8("result_component_kind")?[row],
                    result_component_role: table.u8("result_component_role")?[row],
                    result_shape_string_id: table.u32("result_shape_string_id")?[row],
                    exact_factor_id: table.u32("exact_factor_id")?[row],
                    proof_digest_id: table.u32("proof_digest_id")?[row],
                    input_port_pairing_sequence_id: table.u32("input_port_pairing_sequence_id")?
                        [row],
                    result_port_binding_sequence_id: table
                        .u32("result_port_binding_sequence_id")?[row],
                    provenance_sequence_id: table.u32("provenance_sequence_id")?[row],
                })
            },
        )?;
        let color_nc_terms = decode_rows(self.table("color_nc_terms")?, |table, row| {
            Ok(ColorNcTermRow {
                color_contraction_id: table.u32("color_contraction_id")?[row],
                exponent: table.i32("exponent")?[row],
                factor_id: table.u32("factor_id")?[row],
            })
        })?;
        let u32_sequence_ranges = indexed_ranges(self.table("u32_sequence_ranges")?)?;
        let u32_sequence_values = self.table("u32_sequence_values")?.u32("value")?.to_vec();

        Ok(OwnedRecurrenceTemplateInput {
            input_abi: self.abi,
            catalog_digest: self.catalog_digest,
            compiled_model_digest: self.compiled_model_digest,
            prepared_kernel_pack_digest: self.prepared_kernel_pack_digest,
            catalog_header,
            coupling_order_ranges,
            coupling_order_terms,
            contact_orbit_certificates,
            contact_orbit_steps,
            current_states,
            digest_catalog,
            evaluator_bindings,
            exact_factors,
            flavour_flow_ranges,
            flavour_flow_values,
            i32_sequence_ranges,
            i32_sequence_values,
            parameters,
            propagators,
            quantum_flows,
            quantum_number_flow_ranges,
            quantum_number_flow_terms,
            runtime_helicity_contracts,
            runtime_helicity_variants,
            runtime_helicity_embeddings,
            runtime_helicity_projections,
            sources,
            string_ranges,
            string_bytes,
            symmetry_proofs,
            transitions,
            closures,
            color_contractions,
            lc_color_transition_witnesses,
            color_nc_terms,
            u32_sequence_ranges,
            u32_sequence_values,
        })
    }
}

struct InventoryValidation {
    referenced_count: usize,
    authenticated_count: Option<usize>,
    verified: bool,
}

struct NativeValidationResult {
    input_digest: String,
    catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
    parameter_count: u32,
    current_state_count: u32,
    source_count: u32,
    quantum_flow_count: u32,
    contact_orbit_certificate_count: u32,
    contact_orbit_step_count: u32,
    transition_count: u32,
    propagator_count: u32,
    closure_count: u32,
    color_contraction_count: u32,
    lc_color_transition_witness_count: u32,
    symmetry_proof_count: u32,
    runtime_helicity_contract_count: u32,
    evaluator_binding_count: u32,
    prepared_kernel_count: u32,
    inventory: InventoryValidation,
}

#[pyfunction(signature = (template_input, authenticated_prepared_kernel_ids=None))]
pub(crate) fn _validate_recurrence_template_input_v1(
    py: Python<'_>,
    template_input: &Bound<'_, PyAny>,
    authenticated_prepared_kernel_ids: Option<Vec<u32>>,
) -> PyResult<Py<PyAny>> {
    let decoded = parse_input(template_input)?;
    let native = py
        .detach(move || validate_input(decoded, authenticated_prepared_kernel_ids))
        .map_err(python_error)?;
    result_mapping(py, native)
}

fn parse_input(input: &Bound<'_, PyAny>) -> PyResult<DecodedInput> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "recurrence template input v1 requires a little-endian target",
        ));
    }
    let abi = required_string(input, "abi")?;
    if abi != RECURRENCE_TEMPLATE_INPUT_ABI {
        return Err(PyValueError::new_err(format!(
            "unsupported recurrence template input ABI {abi:?}"
        )));
    }
    let declared_digest = required_sha256(input, "digest", "template input digest")?;
    let canonical_digest_property =
        required_sha256(input, "canonical_digest", "template input canonical digest")?;
    if declared_digest != canonical_digest_property {
        return Err(PyValueError::new_err(
            "recurrence template input digest and canonical_digest disagree",
        ));
    }
    let catalog_digest = semantic_digest_attribute(input, "catalog_digest")?;
    let compiled_model_digest = semantic_digest_attribute(input, "compiled_model_digest")?;
    let prepared_kernel_pack_digest =
        semantic_digest_attribute(input, "prepared_kernel_pack_digest")?;

    let table_objects = iterable_attribute(input, "tables", "recurrence template tables")?;
    if table_objects.len() != TABLE_SPECS.len() {
        return Err(PyValueError::new_err(format!(
            "recurrence template table inventory has {} tables, expected {}",
            table_objects.len(),
            TABLE_SPECS.len()
        )));
    }

    let mut tables = Vec::with_capacity(TABLE_SPECS.len());
    let mut table_by_name = BTreeMap::new();
    for (table_object, spec) in table_objects.into_iter().zip(TABLE_SPECS) {
        let table_name = required_nonempty_string(&table_object, "name", "table name")?;
        if table_name != spec.name {
            return Err(PyValueError::new_err(format!(
                "recurrence template table inventory mismatch: found {table_name:?}, expected {:?}",
                spec.name
            )));
        }
        let row_count = table_object
            .getattr("row_count")?
            .extract::<u64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "recurrence template table {table_name:?} row_count must be u64"
                ))
            })?;
        let column_objects = iterable_attribute(
            &table_object,
            "columns",
            &format!("recurrence template table {table_name:?} columns"),
        )?;
        if column_objects.len() != spec.columns.len() {
            return Err(PyValueError::new_err(format!(
                "recurrence template table {table_name:?} has {} columns, expected {}",
                column_objects.len(),
                spec.columns.len()
            )));
        }

        let mut columns = Vec::with_capacity(spec.columns.len());
        let mut column_by_name = BTreeMap::new();
        for (column_object, column_spec) in column_objects.into_iter().zip(spec.columns) {
            let column_name = required_nonempty_string(&column_object, "name", "column name")?;
            if column_name != column_spec.name {
                return Err(PyValueError::new_err(format!(
                    "recurrence template table {table_name:?} column mismatch: found {column_name:?}, expected {:?}",
                    column_spec.name
                )));
            }
            let context = format!("{table_name}.{column_name}");
            let values_object = column_object.getattr("values")?;
            let dtype = values_object
                .getattr("dtype")?
                .getattr("str")?
                .extract::<String>()?;
            if dtype != column_spec.kind.dtype() {
                return Err(PyValueError::new_err(format!(
                    "{context} has dtype {dtype:?}, expected {:?}",
                    column_spec.kind.dtype()
                )));
            }
            let flags = values_object.getattr("flags")?;
            if flags.getattr("writeable")?.extract::<bool>()? {
                return Err(PyValueError::new_err(format!(
                    "{context} must be read-only"
                )));
            }
            if !flags.getattr("owndata")?.extract::<bool>()? {
                return Err(PyValueError::new_err(format!(
                    "{context} must own its storage"
                )));
            }
            let (values, shape) = extract_owned_values(&values_object, column_spec.kind, &context)?;
            if shape.first().copied() != Some(row_count) {
                return Err(PyValueError::new_err(format!(
                    "{context} first dimension does not match row_count {row_count}"
                )));
            }
            let actual_tail = shape
                .iter()
                .skip(1)
                .map(|value| usize::try_from(*value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| PyValueError::new_err(format!("{context} shape exceeds usize")))?;
            if actual_tail != column_spec.tail_shape {
                return Err(PyValueError::new_err(format!(
                    "{context} has tail shape {actual_tail:?}, expected {:?}",
                    column_spec.tail_shape
                )));
            }
            column_by_name.insert(column_name.clone(), columns.len());
            columns.push(OwnedColumn {
                name: column_name,
                dtype: column_spec.kind.dtype(),
                shape,
                values,
            });
        }
        table_by_name.insert(table_name.clone(), tables.len());
        tables.push(OwnedTable {
            name: table_name,
            row_count,
            columns,
            column_by_name,
        });
    }

    Ok(DecodedInput {
        abi,
        declared_digest,
        canonical_digest_property,
        catalog_digest,
        compiled_model_digest,
        prepared_kernel_pack_digest,
        tables,
        table_by_name,
    })
}

fn extract_owned_values(
    value: &Bound<'_, PyAny>,
    kind: PrimitiveKind,
    context: &str,
) -> PyResult<(OwnedValues, Vec<u64>)> {
    macro_rules! extract {
        ($ty:ty, $variant:ident) => {{
            let array = value
                .extract::<PyReadonlyArrayDyn<'_, $ty>>()
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{context} is not a NumPy array with dtype {:?}",
                        kind.dtype()
                    ))
                })?;
            if !array.is_c_contiguous() {
                return Err(PyValueError::new_err(format!(
                    "{context} must be C-contiguous"
                )));
            }
            let shape = array
                .shape()
                .iter()
                .map(|value| {
                    u64::try_from(*value)
                        .map_err(|_| PyValueError::new_err(format!("{context} shape exceeds u64")))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let values = array
                .as_slice()
                .map_err(|error| {
                    PyValueError::new_err(format!("{context} must be contiguous: {error}"))
                })?
                .to_vec();
            (OwnedValues::$variant(values), shape)
        }};
    }
    Ok(match kind {
        PrimitiveKind::U8 => extract!(u8, U8),
        PrimitiveKind::U32 => extract!(u32, U32),
        PrimitiveKind::U64 => extract!(u64, U64),
        PrimitiveKind::I32 => extract!(i32, I32),
    })
}

fn validate_input(
    decoded: DecodedInput,
    authenticated_prepared_kernel_ids: Option<Vec<u32>>,
) -> RusticolResult<NativeValidationResult> {
    let actual_digest = decoded.canonical_digest()?;
    if actual_digest != decoded.declared_digest
        || actual_digest != decoded.canonical_digest_property
    {
        return Err(invalid(format!(
            "recurrence template input digest mismatch: declared {}, found {actual_digest}",
            decoded.declared_digest
        )));
    }
    let core = decoded.into_core()?;
    let inventory =
        validate_prepared_kernel_inventory(&core, authenticated_prepared_kernel_ids.as_deref())?;
    let validated = core.validate()?;
    let summary = validated.summary();
    if summary.prepared_kernel_count as usize != inventory.referenced_count {
        return Err(invalid(format!(
            "core prepared-kernel count {} disagrees with decoded inventory count {}",
            summary.prepared_kernel_count, inventory.referenced_count
        )));
    }
    Ok(NativeValidationResult {
        input_digest: actual_digest,
        catalog_digest: summary.catalog_digest,
        compiled_model_digest: summary.compiled_model_digest,
        prepared_kernel_pack_digest: summary.prepared_kernel_pack_digest,
        parameter_count: summary.parameter_count,
        current_state_count: summary.current_state_count,
        source_count: summary.source_count,
        quantum_flow_count: summary.quantum_flow_count,
        contact_orbit_certificate_count: summary.contact_orbit_certificate_count,
        contact_orbit_step_count: summary.contact_orbit_step_count,
        transition_count: summary.transition_count,
        propagator_count: summary.propagator_count,
        closure_count: summary.closure_count,
        color_contraction_count: summary.color_contraction_count,
        lc_color_transition_witness_count: summary.lc_color_transition_witness_count,
        symmetry_proof_count: summary.symmetry_proof_count,
        runtime_helicity_contract_count: summary.runtime_helicity_contract_count,
        evaluator_binding_count: summary.evaluator_binding_count,
        prepared_kernel_count: summary.prepared_kernel_count,
        inventory,
    })
}

fn validate_prepared_kernel_inventory(
    input: &OwnedRecurrenceTemplateInput,
    authenticated_ids: Option<&[u32]>,
) -> RusticolResult<InventoryValidation> {
    let referenced = input
        .evaluator_bindings
        .iter()
        .filter_map(|binding| {
            (binding.prepared_kernel_id != MISSING_U32).then_some(binding.prepared_kernel_id)
        })
        .collect::<BTreeSet<_>>();
    let Some(authenticated_ids) = authenticated_ids else {
        return Ok(InventoryValidation {
            referenced_count: referenced.len(),
            authenticated_count: None,
            verified: false,
        });
    };
    let authenticated = authenticated_ids.iter().copied().collect::<BTreeSet<_>>();
    if authenticated.len() != authenticated_ids.len() {
        return Err(invalid(
            "authenticated prepared-kernel ID inventory contains duplicates",
        ));
    }
    if authenticated.contains(&MISSING_U32) {
        return Err(invalid(
            "authenticated prepared-kernel ID inventory contains the missing-ID sentinel",
        ));
    }
    let missing = referenced
        .difference(&authenticated)
        .copied()
        .take(16)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(invalid(format!(
            "recurrence template references prepared-kernel IDs absent from the authenticated pack inventory: {missing:?}"
        )));
    }
    Ok(InventoryValidation {
        referenced_count: referenced.len(),
        authenticated_count: Some(authenticated.len()),
        verified: true,
    })
}

fn result_mapping(py: Python<'_>, native: NativeValidationResult) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("kind", RESULT_KIND)?;
    result.set_item("schema_version", RESULT_SCHEMA_VERSION)?;
    result.set_item("validation_status", "validated")?;
    result.set_item("template_input_abi", RECURRENCE_TEMPLATE_INPUT_ABI)?;
    result.set_item(
        "template_input_schema_version",
        RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
    )?;
    result.set_item("template_input_sha256", native.input_digest)?;
    result.set_item("catalog_digest", native.catalog_digest.to_string())?;
    result.set_item(
        "compiled_model_digest",
        native.compiled_model_digest.to_string(),
    )?;
    result.set_item(
        "prepared_kernel_pack_digest",
        native.prepared_kernel_pack_digest.to_string(),
    )?;
    result.set_item(
        "prepared_kernel_inventory_verified",
        native.inventory.verified,
    )?;
    result.set_item(
        "prepared_kernel_inventory_count",
        native.inventory.authenticated_count,
    )?;
    if !native.inventory.verified {
        result.set_item(
            "prepared_kernel_inventory_requirement",
            "central registration must compare referenced IDs with the authenticated prepared pack",
        )?;
    }

    let counts = PyDict::new(py);
    counts.set_item("parameters", native.parameter_count)?;
    counts.set_item("current_states", native.current_state_count)?;
    counts.set_item("sources", native.source_count)?;
    counts.set_item("quantum_flows", native.quantum_flow_count)?;
    counts.set_item(
        "contact_orbit_certificates",
        native.contact_orbit_certificate_count,
    )?;
    counts.set_item("contact_orbit_steps", native.contact_orbit_step_count)?;
    counts.set_item("transitions", native.transition_count)?;
    counts.set_item("propagators", native.propagator_count)?;
    counts.set_item("closures", native.closure_count)?;
    counts.set_item("color_contractions", native.color_contraction_count)?;
    counts.set_item(
        "lc_color_transition_witnesses",
        native.lc_color_transition_witness_count,
    )?;
    counts.set_item("symmetry_proofs", native.symmetry_proof_count)?;
    counts.set_item(
        "runtime_helicity_contracts",
        native.runtime_helicity_contract_count,
    )?;
    counts.set_item("evaluator_bindings", native.evaluator_binding_count)?;
    counts.set_item("prepared_kernels", native.prepared_kernel_count)?;
    counts.set_item(
        "referenced_prepared_kernels",
        native.inventory.referenced_count,
    )?;
    result.set_item("counts", counts)?;
    Ok(result.into_any().unbind())
}

fn decode_rows<T>(
    table: &OwnedTable,
    mut decode: impl FnMut(&OwnedTable, usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = table.row_count()?;
    (0..row_count).map(|row| decode(table, row)).collect()
}

fn indexed_ranges(table: &OwnedTable) -> RusticolResult<Vec<IndexedRangeRow>> {
    decode_rows(table, |table, row| {
        Ok(IndexedRangeRow {
            id: table.u32("id")?[row],
            range: CheckedTableRange::new(table.u64("start")?[row], table.u64("count")?[row]),
        })
    })
}

fn plain_ranges(table: &OwnedTable) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_rows(table, |table, row| {
        Ok(CheckedTableRange::new(
            table.u64("start")?[row],
            table.u64("count")?[row],
        ))
    })
}

fn decode_digest_catalog(table: &OwnedTable) -> RusticolResult<Vec<DigestCatalogRow>> {
    let ids = table.u32("id")?;
    let values = table.u8("value")?;
    let row_count = table.row_count()?;
    let expected = row_count
        .checked_mul(32)
        .ok_or_else(|| invalid("digest catalog byte count exceeds usize"))?;
    if values.len() != expected {
        return Err(invalid(format!(
            "digest_catalog.value has {} bytes, expected {expected}",
            values.len()
        )));
    }
    (0..row_count)
        .map(|row| {
            let start = row * 32;
            let mut value = [0_u8; 32];
            value.copy_from_slice(&values[start..start + 32]);
            Ok(DigestCatalogRow {
                id: ids[row],
                value,
            })
        })
        .collect()
}

fn semantic_digest_attribute(
    input: &Bound<'_, PyAny>,
    attribute: &str,
) -> PyResult<SemanticDigest> {
    let value = required_sha256(input, attribute, attribute)?;
    let bytes = decode_sha256(&value, attribute)?;
    SemanticDigest::new(bytes).map_err(python_error)
}

fn iterable_attribute<'py>(
    value: &Bound<'py, PyAny>,
    attribute: &str,
    context: &str,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    value
        .getattr(attribute)?
        .try_iter()
        .map_err(|_| PyTypeError::new_err(format!("{context} must be iterable")))?
        .collect()
}

fn required_string(value: &Bound<'_, PyAny>, attribute: &str) -> PyResult<String> {
    value.getattr(attribute)?.extract::<String>().map_err(|_| {
        PyTypeError::new_err(format!(
            "recurrence template input {attribute} must be a string"
        ))
    })
}

fn required_nonempty_string(
    value: &Bound<'_, PyAny>,
    attribute: &str,
    context: &str,
) -> PyResult<String> {
    let result = required_string(value, attribute)?;
    if result.is_empty() {
        return Err(PyValueError::new_err(format!(
            "recurrence template {context} must not be empty"
        )));
    }
    Ok(result)
}

fn required_sha256(value: &Bound<'_, PyAny>, attribute: &str, context: &str) -> PyResult<String> {
    let result = required_string(value, attribute)?;
    validate_sha256_text(&result, context)?;
    Ok(result)
}

fn validate_sha256_text(value: &str, context: &str) -> PyResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(PyValueError::new_err(format!(
            "{context} must be a lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn decode_sha256(value: &str, context: &str) -> PyResult<[u8; 32]> {
    validate_sha256_text(value, context)?;
    let mut result = [0_u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| PyValueError::new_err(format!("{context} is not hexadecimal")))?;
    }
    Ok(result)
}

fn hash_text(digest: &mut Sha256, value: &str) -> RusticolResult<()> {
    digest.update(
        u64::try_from(value.len())
            .map_err(|_| invalid("recurrence template digest text length exceeds u64"))?
            .to_le_bytes(),
    );
    digest.update(value.as_bytes());
    Ok(())
}

fn hex_digest(value: impl AsRef<[u8]>) -> String {
    let mut result = String::with_capacity(value.as_ref().len() * 2);
    for byte in value.as_ref() {
        use std::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn wrong_type(context: &str, expected: &str) -> RusticolError {
    invalid(format!(
        "{context} is not a {expected} recurrence template column"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_inventory_is_canonical() {
        assert!(
            TABLE_SPECS
                .windows(2)
                .all(|pair| pair[0].name < pair[1].name)
        );
        for table in TABLE_SPECS {
            assert!(
                table
                    .columns
                    .iter()
                    .map(|column| column.name)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == table.columns.len()
            );
        }
    }

    #[test]
    fn sha256_decoder_is_strict() {
        let value = "01".repeat(32);
        assert_eq!(decode_sha256(&value, "test").unwrap(), [1; 32]);
        assert!(decode_sha256(&"A1".repeat(32), "test").is_err());
        assert!(decode_sha256("00", "test").is_err());
    }
}
