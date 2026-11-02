// SPDX-License-Identifier: 0BSD

//! Private Python boundary for Direct-Arena recurrence lowering.
//!
//! Python supplies authenticated fixed-width builder and prepared-template
//! columns. Rust constructs the compact recurrence, lowers its direct arena
//! plan, and publishes the plan-v2 PACBIN without exposing an intermediate
//! packet schedule.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use rusticol_core::recurrence::process;
use rusticol_core::recurrence::template;
use rusticol_core::recurrence::{
    AuthenticatedRecurrenceBuilderInput, CLOSURE_TARGET_DOMAIN_PROOF_KIND, CheckedTableRange,
    DIRECT_NONE_U32, DirectExecutorRole, DirectRecurrencePlan, DirectRecurrenceRuntimeOptions,
    DirectSelectorWorkSummary, PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog,
    RECURRENCE_BUILDER_INPUT_ABI, RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
    RECURRENCE_DIRECT_PLAN_ABI, RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI, RECURRENCE_DIRECT_SCHEDULE_MEMBER,
    RECURRENCE_DIRECT_TEMPLATE_ABI, RECURRENCE_LC_COLOR_CAPABILITY, RecurrenceBuildProgress,
    RecurrenceGenerationTelemetry, RecurrenceRelationDiscoveryMode,
    RecurrenceRelationDiscoveryOptions, RecurrenceRelationDiscoveryReport, RecurrenceStrategy,
    SemanticDigest, authenticate_recurrence_numerical_relation_provenance, checked_usize,
    lower_recurrence_direct_plan_v2, lower_recurrence_direct_plan_v2_with_helicity_dispatch,
    lower_recurrence_direct_plan_v2_with_relation_discovery,
    write_recurrence_direct_plan_pacbin_with_projection_certificate,
    write_recurrence_helicity_dispatch_v1_atomic,
};
#[cfg(feature = "on-the-fly-test-support")]
use rusticol_core::recurrence::{
    ConstructionTransitionDiagnosticRowV1, ExactComplexRational, OnTheFlyQueryFamilyCensusV1,
    OnTheFlyTestSupportReportV1, on_the_fly_query_family_census_v1,
    on_the_fly_test_support_probe_v1,
};
#[cfg(feature = "on-the-fly-test-support")]
use rusticol_core::{
    NativeOnTheFlyArtifactProbeV1, NativeOnTheFlyExecutionDiagnosticV1, NativeRuntime,
};
use rusticol_core::{
    NativeRecurrenceExactExecutor, NativeRecurrenceExactFactor, NativeRecurrenceExactSections,
    RusticolError, RusticolResult,
};
use serde_json::{Map as JsonMap, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use crate::python_error;
use crate::recurrence_numerical_evidence::{
    authenticated_runtime_parameter_contract, parse_numerical_relation_evidence,
};

const RUNTIME_CONTAINER_KIND: &str = "pyamplicol-recurrence-runtime-container";
const RUNTIME_CONTAINER_SCHEMA_VERSION: u32 = 1;
const STORAGE_ABI: &str = "pacbin-v1";
const DIRECT_BUILDER_INPUT_ABI: &str = "pyamplicol-recurrence-builder-input-v2";
const DIRECT_LOWERING_RESULT_KIND: &str = "pyamplicol-recurrence-direct-lowering-result";
const DIRECT_LOWERING_RESULT_SCHEMA_VERSION: u32 = 4;
const RECURRENCE_HELICITY_DISPATCH_ABI: &str = "pyamplicol-recurrence-helicity-dispatch-v1";
const DIRECT_CANONICALIZATION_ABI: &str = "pyamplicol-canonical-json-v1";
const DIRECT_BACKEND_ABI: &str = "rusticol.recurrence-direct-backend.v1";
const DIRECT_PAYLOAD_BINDING_ABI: &str = "pyamplicol-recurrence-plane-binding-v2";
const DIRECT_IDENTITY_FINALIZER: &str = "rusticol.identity-finalize-in-place.v1";

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

    fn from_dtype(value: &str) -> Option<Self> {
        match value {
            "|u1" => Some(Self::U8),
            "<u4" => Some(Self::U32),
            "<u8" => Some(Self::U64),
            "<i4" => Some(Self::I32),
            _ => None,
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
        name: "bitset_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
            column("bit_count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "bitset_words",
        columns: &[column("value", PrimitiveKind::U64)],
    },
    TableSpec {
        name: "coupling_limits",
        columns: &[
            column("name_string_id", PrimitiveKind::U32),
            column("minimum", PrimitiveKind::U32),
            column("maximum", PrimitiveKind::U32),
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
        name: "external_legs",
        columns: &[
            column("source_slot", PrimitiveKind::U32),
            column("public_label", PrimitiveKind::U32),
            column("physical_pdg", PrimitiveKind::I32),
            column("outgoing_pdg", PrimitiveKind::I32),
            column("is_initial", PrimitiveKind::U8),
            column("is_fermionic", PrimitiveKind::U8),
            column("source_state_start", PrimitiveKind::U64),
            column("source_state_count", PrimitiveKind::U64),
            column("momentum_mask_id", PrimitiveKind::U32),
            column("support_mask_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "header",
        columns: &[
            column("schema_version", PrimitiveKind::U32),
            column("abi_string_id", PrimitiveKind::U32),
            column("process_id_string_id", PrimitiveKind::U32),
            column("layout", PrimitiveKind::U8),
            column("selected_flow_mode", PrimitiveKind::U8),
            column("selected_source_mode", PrimitiveKind::U8),
            column("external_leg_count", PrimitiveKind::U32),
            column("physical_sector_count", PrimitiveKind::U32),
            column("public_flow_count", PrimitiveKind::U32),
            column("replay_partition_count", PrimitiveKind::U32),
            column("coupling_limit_count", PrimitiveKind::U32),
            column("parameter_projection_count", PrimitiveKind::U32),
            column("process_support_mask_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "header_digests",
        columns: &[
            column("role_string_id", PrimitiveKind::U32),
            column("digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "lc_open_strings",
        columns: &[
            column("sector_id", PrimitiveKind::U32),
            column("ordinal", PrimitiveKind::U32),
            column("fundamental_source_slot", PrimitiveKind::U32),
            column("antifundamental_source_slot", PrimitiveKind::U32),
            column("adjoint_sequence_id", PrimitiveKind::U32),
            column("singlet_sequence_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "normalization",
        columns: &[
            column("factor_id", PrimitiveKind::U32),
            column("convention_string_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "parameter_projection",
        columns: &[
            column("runtime_slot", PrimitiveKind::U32),
            column("runtime_name_string_id", PrimitiveKind::U32),
            column("parameter_template_id", PrimitiveKind::U32),
            column("prepared_parameter_id", PrimitiveKind::U32),
            column("component", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "physical_lc_sectors",
        columns: &[
            column("sector_id", PrimitiveKind::U32),
            column("public_id_string_id", PrimitiveKind::U32),
            column("kind", PrimitiveKind::U8),
            column("closure_source_slot", PrimitiveKind::U32),
            column("closure_proof_algorithm_string_id", PrimitiveKind::U32),
            column("closure_proof_digest_id", PrimitiveKind::U32),
            column("open_string_start", PrimitiveKind::U64),
            column("open_string_count", PrimitiveKind::U64),
            column("trace_sequence_id", PrimitiveKind::U32),
            column("singlet_sequence_id", PrimitiveKind::U32),
            column("word_sequence_id", PrimitiveKind::U32),
            column("support_mask_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "public_lc_flows",
        columns: &[
            column("flow_id", PrimitiveKind::U32),
            column("public_id_string_id", PrimitiveKind::U32),
            column("construction_sector_id", PrimitiveKind::U32),
            column("word_sequence_id", PrimitiveKind::U32),
            column("source_slot_permutation_sequence_id", PrimitiveKind::U32),
            column("reduction_weight_factor_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "replay_partitions",
        columns: &[
            column("partition_id", PrimitiveKind::U32),
            column("representative_sector_id", PrimitiveKind::U32),
            column("materialized_sector_id", PrimitiveKind::U32),
            column("target_start", PrimitiveKind::U64),
            column("target_count", PrimitiveKind::U64),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            column("proof_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "replay_targets",
        columns: &[
            column("partition_id", PrimitiveKind::U32),
            column("sector_id", PrimitiveKind::U32),
            column("external_permutation_sequence_id", PrimitiveKind::U32),
            column("source_slot_permutation_sequence_id", PrimitiveKind::U32),
            column("amplitude_phase_factor_id", PrimitiveKind::U32),
            column("fermion_sign", PrimitiveKind::I32),
        ],
    },
    TableSpec {
        name: "selected_public_flow_coverage",
        columns: &[column("flow_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "selected_source_coverage",
        columns: &[
            column("source_slot", PrimitiveKind::U32),
            column("source_state_index", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "semantic_template_references",
        columns: &[
            column("kind_string_id", PrimitiveKind::U32),
            column("template_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
            column("prepared_kernel_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "source_states",
        columns: &[
            column("source_slot", PrimitiveKind::U32),
            column("state_index", PrimitiveKind::U32),
            column("public_helicity", PrimitiveKind::I32),
            column("chirality", PrimitiveKind::I32),
            column("spin_state", PrimitiveKind::I32),
            column("current_state_template_id", PrimitiveKind::U32),
            column("source_template_id", PrimitiveKind::U32),
            column("momentum_sign", PrimitiveKind::I32),
            column("crossing_phase_factor_id", PrimitiveKind::U32),
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
        name: "u32_sequence_ranges",
        columns: &[
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "u32_sequence_values",
        columns: &[column("value", PrimitiveKind::U32)],
    },
];

const FERMION_PAIRING_TABLE_SPECS: &[TableSpec] = &[
    TableSpec {
        name: "header",
        columns: &[
            column("schema_version", PrimitiveKind::U32),
            column("abi_string_id", PrimitiveKind::U32),
            column("process_key_string_id", PrimitiveKind::U32),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            column("source_count", PrimitiveKind::U32),
            column("endpoint_count", PrimitiveKind::U32),
            column("pairing_class_count", PrimitiveKind::U32),
            column("rule_count", PrimitiveKind::U32),
            column("endpoint_state_template_count", PrimitiveKind::U64),
            column("endpoint_anti_state_template_count", PrimitiveKind::U64),
            column("endpoint_basis_count", PrimitiveKind::U64),
            column("endpoint_color_representation_count", PrimitiveKind::U64),
            column("class_fundamental_slot_count", PrimitiveKind::U64),
            column("class_antifundamental_slot_count", PrimitiveKind::U64),
            column("class_reference_pairing_count", PrimitiveKind::U64),
            column("rule_class_pairing_index_count", PrimitiveKind::U64),
            column("rule_endpoint_pairing_count", PrimitiveKind::U64),
            column("rule_source_permutation_count", PrimitiveKind::U64),
            column("rule_lineage_count", PrimitiveKind::U64),
            column("exact_integer_count", PrimitiveKind::U32),
            column("exact_integer_limb_count", PrimitiveKind::U64),
            column("string_count", PrimitiveKind::U32),
            column("string_byte_count", PrimitiveKind::U64),
            column("no_fermion_line", PrimitiveKind::U32),
            shaped_column("topology_digest", PrimitiveKind::U8, &[32]),
            shaped_column("semantic_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "endpoints",
        columns: &[
            column("endpoint_id", PrimitiveKind::U32),
            column("source_slot", PrimitiveKind::U32),
            column("public_label", PrimitiveKind::U32),
            column("species_class_id", PrimitiveKind::U32),
            column("species_string_id", PrimitiveKind::U32),
            column("particle_orientation", PrimitiveKind::U8),
            column("color_orientation", PrimitiveKind::U8),
            column("state_template_start", PrimitiveKind::U64),
            column("state_template_count", PrimitiveKind::U64),
            column("anti_state_template_start", PrimitiveKind::U64),
            column("anti_state_template_count", PrimitiveKind::U64),
            column("basis_start", PrimitiveKind::U64),
            column("basis_count", PrimitiveKind::U64),
            column("color_representation_start", PrimitiveKind::U64),
            column("color_representation_count", PrimitiveKind::U64),
            shaped_column("contract_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "endpoint_state_template_ids",
        columns: &[column("string_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "endpoint_anti_state_template_ids",
        columns: &[column("string_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "endpoint_basis_ids",
        columns: &[column("string_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "endpoint_color_representations",
        columns: &[column("value", PrimitiveKind::I32)],
    },
    TableSpec {
        name: "pairing_classes",
        columns: &[
            column("class_id", PrimitiveKind::U32),
            column("species_class_id", PrimitiveKind::U32),
            column("species_string_id", PrimitiveKind::U32),
            column("fundamental_slot_start", PrimitiveKind::U64),
            column("fundamental_slot_count", PrimitiveKind::U64),
            column("antifundamental_slot_start", PrimitiveKind::U64),
            column("antifundamental_slot_count", PrimitiveKind::U64),
            column("reference_pairing_start", PrimitiveKind::U64),
            column("reference_pairing_count", PrimitiveKind::U64),
            column("pairing_count", PrimitiveKind::U64),
            shaped_column("proof_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "class_fundamental_slots",
        columns: &[column("source_slot", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "class_antifundamental_slots",
        columns: &[column("source_slot", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "class_reference_pairings",
        columns: &[
            column("fundamental_source_slot", PrimitiveKind::U32),
            column("antifundamental_source_slot", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "rules",
        columns: &[
            column("rule_id", PrimitiveKind::U32),
            column("class_pairing_index_start", PrimitiveKind::U64),
            column("class_pairing_index_count", PrimitiveKind::U64),
            column("endpoint_pairing_start", PrimitiveKind::U64),
            column("endpoint_pairing_count", PrimitiveKind::U64),
            column("source_permutation_start", PrimitiveKind::U64),
            column("source_permutation_count", PrimitiveKind::U64),
            column("lineage_start", PrimitiveKind::U64),
            column("lineage_count", PrimitiveKind::U64),
            column("fermion_parity", PrimitiveKind::I32),
            column("real_numerator_integer_id", PrimitiveKind::U32),
            column("real_denominator_integer_id", PrimitiveKind::U32),
            column("imag_numerator_integer_id", PrimitiveKind::U32),
            column("imag_denominator_integer_id", PrimitiveKind::U32),
            column("multiplicity", PrimitiveKind::U64),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            shaped_column("proof_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "rule_class_pairing_indices",
        columns: &[
            column("class_id", PrimitiveKind::U32),
            column("pairing_index", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "rule_endpoint_pairings",
        columns: &[
            column("fundamental_source_slot", PrimitiveKind::U32),
            column("antifundamental_source_slot", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "rule_source_slot_permutations",
        columns: &[column("source_slot", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "rule_lineages",
        columns: &[column("line_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "exact_integers",
        columns: &[
            column("integer_id", PrimitiveKind::U32),
            column("sign", PrimitiveKind::I32),
            column("limb_start", PrimitiveKind::U64),
            column("limb_count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "exact_integer_limbs",
        columns: &[column("value", PrimitiveKind::U64)],
    },
    TableSpec {
        name: "string_ranges",
        columns: &[
            column("string_id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "string_bytes",
        columns: &[column("value", PrimitiveKind::U8)],
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
            // Accepted multi-byte arrays are explicitly little-endian and this
            // module rejects big-endian hosts before extraction.
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
                "recurrence builder table {:?} has no column {name:?}",
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

#[derive(Clone)]
struct OwnedInput {
    abi: String,
    declared_digest: String,
    tables: Vec<OwnedTable>,
    table_by_name: BTreeMap<String, usize>,
    declared_fermion_pairing_digest: Option<String>,
    fermion_pairing_tables: Vec<OwnedTable>,
    fermion_pairing_table_by_name: BTreeMap<String, usize>,
}

impl OwnedInput {
    fn table(&self, name: &str) -> RusticolResult<&OwnedTable> {
        let index = self
            .table_by_name
            .get(name)
            .ok_or_else(|| invalid(format!("recurrence builder input has no table {name:?}")))?;
        Ok(&self.tables[*index])
    }

    fn column(&self, table: &str, column: &str) -> RusticolResult<&OwnedColumn> {
        self.table(table)?.column(column)
    }

    fn u8(&self, table: &str, column: &str) -> RusticolResult<&[u8]> {
        self.column(table, column)?
            .values
            .as_u8(&format!("{table}.{column}"))
    }

    fn u32(&self, table: &str, column: &str) -> RusticolResult<&[u32]> {
        self.column(table, column)?
            .values
            .as_u32(&format!("{table}.{column}"))
    }

    fn u64(&self, table: &str, column: &str) -> RusticolResult<&[u64]> {
        self.column(table, column)?
            .values
            .as_u64(&format!("{table}.{column}"))
    }

    fn i32(&self, table: &str, column: &str) -> RusticolResult<&[i32]> {
        self.column(table, column)?
            .values
            .as_i32(&format!("{table}.{column}"))
    }

    fn fermion_pairing_table(&self, name: &str) -> RusticolResult<&OwnedTable> {
        let index = self
            .fermion_pairing_table_by_name
            .get(name)
            .ok_or_else(|| invalid(format!("recurrence fermion pairing has no table {name:?}")))?;
        Ok(&self.fermion_pairing_tables[*index])
    }

    fn pairing_u8(&self, table: &str, column: &str) -> RusticolResult<&[u8]> {
        self.fermion_pairing_table(table)?.u8(column)
    }

    fn pairing_u32(&self, table: &str, column: &str) -> RusticolResult<&[u32]> {
        self.fermion_pairing_table(table)?.u32(column)
    }

    fn pairing_u64(&self, table: &str, column: &str) -> RusticolResult<&[u64]> {
        self.fermion_pairing_table(table)?.u64(column)
    }

    fn pairing_i32(&self, table: &str, column: &str) -> RusticolResult<&[i32]> {
        self.fermion_pairing_table(table)?.i32(column)
    }
}

const TEMPLATE_TABLE_INVENTORY: &[(&str, usize)] = &[
    ("catalog_header", 20),
    ("closures", 20),
    ("color_contractions", 14),
    ("color_nc_terms", 3),
    ("contact_orbit_certificates", 11),
    ("contact_orbit_steps", 10),
    ("coupling_order_ranges", 3),
    ("coupling_order_terms", 3),
    ("current_states", 17),
    ("digest_catalog", 2),
    ("evaluator_bindings", 14),
    ("exact_factors", 5),
    ("flavour_flow_ranges", 3),
    ("flavour_flow_values", 1),
    ("i32_sequence_ranges", 3),
    ("i32_sequence_values", 1),
    ("lc_color_transition_witnesses", 15),
    ("parameters", 11),
    ("propagators", 12),
    ("quantum_flows", 16),
    ("quantum_number_flow_ranges", 3),
    ("quantum_number_flow_terms", 3),
    ("runtime_helicity_contracts", 8),
    ("runtime_helicity_embeddings", 4),
    ("runtime_helicity_projections", 3),
    ("runtime_helicity_variants", 9),
    ("sources", 20),
    ("string_bytes", 1),
    ("string_ranges", 2),
    ("symmetry_proofs", 9),
    ("transitions", 20),
    ("u32_sequence_ranges", 3),
    ("u32_sequence_values", 1),
];

struct PreparedTemplateInput {
    input: OwnedInput,
    canonical_digest_property: String,
    catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
}

struct AuthenticatedBuilderInputs {
    builder_input_digest: String,
    template_input_digest: String,
    authenticated: AuthenticatedRecurrenceBuilderInput,
}

fn authenticate_builder_inputs(
    input: OwnedInput,
    prepared_template: PreparedTemplateInput,
    expected_pack_digest: SemanticDigest,
) -> RusticolResult<AuthenticatedBuilderInputs> {
    validate_inventory(&input)?;
    let builder_input_digest = canonical_digest(&input)?;
    if builder_input_digest != input.declared_digest {
        return Err(invalid(format!(
            "recurrence builder input digest mismatch: declared {}, found {builder_input_digest}",
            input.declared_digest
        )));
    }
    let template_input_digest = prepared_template.canonical_digest()?;
    if prepared_template.prepared_kernel_pack_digest != expected_pack_digest {
        return Err(RusticolError::integrity(
            "recurrence prepared-kernel pack digest does not match the authenticated template",
        ));
    }
    let process = decode_process_input(&input)?.validate()?;
    let template = prepared_template.into_core()?.validate()?;
    let authenticated = AuthenticatedRecurrenceBuilderInput::new(process, template)?;
    Ok(AuthenticatedBuilderInputs {
        builder_input_digest,
        template_input_digest,
        authenticated,
    })
}

impl PreparedTemplateInput {
    fn canonical_digest(&self) -> RusticolResult<String> {
        let mut digest = Sha256::new();
        hash_text(&mut digest, &self.input.abi)?;
        digest.update(self.catalog_digest.as_bytes());
        digest.update(self.compiled_model_digest.as_bytes());
        digest.update(self.prepared_kernel_pack_digest.as_bytes());
        hash_tables(&mut digest, &self.input.tables)?;
        Ok(hex_digest(digest.finalize()))
    }

    fn into_core(self) -> RusticolResult<template::OwnedRecurrenceTemplateInput> {
        let actual_digest = self.canonical_digest()?;
        if actual_digest != self.input.declared_digest
            || actual_digest != self.canonical_digest_property
        {
            return Err(invalid(format!(
                "recurrence template input digest mismatch: declared {}, found {actual_digest}",
                self.input.declared_digest
            )));
        }
        decode_template_input(
            &self.input,
            self.catalog_digest,
            self.compiled_model_digest,
            self.prepared_kernel_pack_digest,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DirectLoweringTimings {
    python_extraction_seconds: f64,
    catalog_authentication_seconds: f64,
    semantic_construction_seconds: f64,
    direct_lowering_seconds: f64,
    serialization_seconds: f64,
    native_total_seconds: f64,
}

#[derive(Debug)]
struct NativeDirectLoweringResult {
    builder_input_digest: String,
    template_input_digest: String,
    prepared_kernel_pack_digest: String,
    direct_template_catalog_digest: String,
    process_id: String,
    strategy: RecurrenceStrategy,
    semantic_digest: String,
    runtime_layout_digest: String,
    member_count: u64,
    unpacked_size_bytes: u64,
    index_sha256: String,
    container_size: u64,
    plan_payload_size: u64,
    plan_sha256: String,
    projection_certificate_payload_size: Option<u64>,
    projection_certificate_sha256: Option<String>,
    current_count: usize,
    source_row_count: usize,
    contribution_count: usize,
    finalization_count: usize,
    closure_count: usize,
    row_group_count: usize,
    momentum_form_count: usize,
    selector_domain_count: usize,
    replay_target_count: usize,
    amplitude_destination_count: usize,
    resolved_helicity_count: usize,
    retained_helicity_count: u64,
    exact_factor_count: usize,
    semantic_component_count: u64,
    current_arena_components: u32,
    parameter_value_count: u32,
    physical_sector_count: u32,
    direct_executor_count: u32,
    prepared_kernel_count: usize,
    resolved_helicities: Vec<Vec<i32>>,
    amplitude_destinations: Vec<(u32, Option<u32>)>,
    closure_target_domain_certificate: Option<NativeClosureTargetDomainCertificate>,
    selector_work: Vec<(DirectSelectorWorkSummary, u64)>,
    relation_discovery: Option<RecurrenceRelationDiscoveryReport>,
    exact_sections: NativeRecurrenceExactSections,
    construction: RecurrenceConstructionMetrics,
    generation_profile: RecurrenceGenerationTelemetry,
    timings: DirectLoweringTimings,
    helicity_dispatch: Option<NativeHelicityDispatchMetadata>,
}

#[derive(Debug)]
struct NativeClosureTargetDomainCertificate {
    physical_sector_count: u32,
    resolved_helicity_count: u32,
    certified_structural_zero_physical_sector_ids: Vec<u32>,
    semantic_digest: String,
}

#[derive(Debug)]
struct NativeHelicityDispatchMetadata {
    size_bytes: u64,
    sha256: String,
    base_runtime_layout_digest: String,
    resolved_helicity_count: usize,
}

#[derive(Clone, Debug, Default)]
struct RecurrenceConstructionMetrics {
    peak_current_count: usize,
    peak_contribution_count: usize,
    peak_dynamic_color_state_count: usize,
    color_target_prune_count: usize,
    candidate_parent_pair_count_by_stage: BTreeMap<usize, usize>,
}

impl RecurrenceConstructionMetrics {
    fn include(&mut self, progress: &RecurrenceBuildProgress) {
        self.peak_current_count = self.peak_current_count.max(progress.current_count);
        self.peak_contribution_count = self
            .peak_contribution_count
            .max(progress.contribution_count);
        self.peak_dynamic_color_state_count = self
            .peak_dynamic_color_state_count
            .max(progress.dynamic_color_state_count);
        self.color_target_prune_count = self
            .color_target_prune_count
            .max(progress.color_target_prune_count);
        if let Some(stage_index) = progress.stage_index {
            self.candidate_parent_pair_count_by_stage
                .entry(stage_index)
                .and_modify(|count| {
                    *count = (*count).max(progress.candidate_parent_pair_count);
                })
                .or_insert(progress.candidate_parent_pair_count);
        }
    }

    fn candidate_parent_pair_count(&self) -> usize {
        self.candidate_parent_pair_count_by_stage
            .values()
            .copied()
            .sum()
    }
}

struct AuthenticatedDirectTemplateCatalog {
    catalog: PreparedDirectExecutorCatalog,
    catalog_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
    prepared_kernel_count: usize,
    exact_executors: Vec<NativeRecurrenceExactExecutor>,
}

#[pyfunction(signature = (
    builder_input,
    prepared_template_input,
    direct_template_catalog_json,
    prepared_kernel_pack_digest,
    schedule_semantic_digest,
    destination,
    *,
    point_tile_size,
    workspace_mib,
    relation_discovery_mode,
    relation_discovery_precision_digits,
    relation_discovery_probe_count,
    relation_discovery_verification_probe_count,
    relation_discovery_relative_tolerance,
    relation_discovery_absolute_tolerance,
    relation_discovery_seed,
    color_accuracy,
    relation_discovery_evidence_json=None,
    helicity_dispatch_destination=None,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _lower_recurrence_direct_v2(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: String,
    schedule_semantic_digest: String,
    destination: PathBuf,
    point_tile_size: u32,
    workspace_mib: u32,
    relation_discovery_mode: String,
    relation_discovery_precision_digits: u32,
    relation_discovery_probe_count: u32,
    relation_discovery_verification_probe_count: u32,
    relation_discovery_relative_tolerance: f64,
    relation_discovery_absolute_tolerance: f64,
    relation_discovery_seed: u64,
    color_accuracy: String,
    relation_discovery_evidence_json: Option<&Bound<'_, PyBytes>>,
    helicity_dispatch_destination: Option<PathBuf>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let extraction_started = Instant::now();
    let owned = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    validate_sha256_text(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    validate_sha256_text(&schedule_semantic_digest, "recurrence schedule digest")?;
    let relation_discovery = RecurrenceRelationDiscoveryOptions::new(
        RecurrenceRelationDiscoveryMode::parse(&relation_discovery_mode).map_err(python_error)?,
        relation_discovery_precision_digits,
        relation_discovery_probe_count,
        relation_discovery_verification_probe_count,
        relation_discovery_relative_tolerance,
        relation_discovery_absolute_tolerance,
        relation_discovery_seed,
        color_accuracy,
    )
    .map_err(python_error)?;
    let relation_discovery_evidence_json =
        relation_discovery_evidence_json.map(|raw| raw.as_bytes().to_vec());
    let python_extraction_seconds = extraction_started.elapsed().as_secs_f64();

    let native = py
        .detach(move || {
            let mut report =
                |progress| report_recurrence_build_progress(progress_callback.as_ref(), progress);
            lower_recurrence_direct(
                owned,
                prepared_template,
                &direct_template_catalog_json,
                &prepared_kernel_pack_digest,
                &schedule_semantic_digest,
                &destination,
                point_tile_size,
                workspace_mib,
                relation_discovery,
                relation_discovery_evidence_json,
                helicity_dispatch_destination.as_deref(),
                python_extraction_seconds,
                &mut report,
            )
        })
        .map_err(python_error)?;
    direct_lowering_mapping(py, native)
}

#[cfg(feature = "on-the-fly-test-support")]
#[pyfunction(signature = (
    builder_input,
    prepared_template_input,
    direct_template_catalog_json,
    prepared_kernel_pack_digest,
    selected_public_flow_id,
    public_helicities
))]
pub(crate) fn _on_the_fly_test_support_probe_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: String,
    selected_public_flow_id: u32,
    public_helicities: Vec<i32>,
) -> PyResult<Py<PyAny>> {
    let input = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    validate_sha256_text(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let native = py
        .detach(move || {
            let expected_pack_digest = semantic_digest_from_hex(
                &prepared_kernel_pack_digest,
                "prepared kernel pack digest",
            )?;
            let authenticated =
                authenticate_builder_inputs(input, prepared_template, expected_pack_digest)?
                    .authenticated;
            let direct_catalog = parse_direct_template_catalog(
                &direct_template_catalog_json,
                expected_pack_digest,
                authenticated.template().summary().catalog_digest,
                authenticated.template().summary().compiled_model_digest,
            )?;
            on_the_fly_test_support_probe_v1(
                &authenticated,
                &direct_catalog.catalog,
                selected_public_flow_id,
                &public_helicities,
            )
        })
        .map_err(python_error)?;
    on_the_fly_test_support_mapping(py, native)
}

#[cfg(feature = "on-the-fly-test-support")]
#[pyfunction(signature = (
    builder_input,
    prepared_template_input,
    direct_template_catalog_json,
    prepared_kernel_pack_digest,
    selected_public_flow_ids,
    public_helicities,
    *,
    enable_color_projection=true
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _on_the_fly_query_family_census_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: String,
    selected_public_flow_ids: Vec<u32>,
    public_helicities: Vec<Vec<i32>>,
    enable_color_projection: bool,
) -> PyResult<Py<PyAny>> {
    if selected_public_flow_ids.len() != public_helicities.len() {
        return Err(PyValueError::new_err(
            "on-the-fly query-family flow and helicity columns have different lengths",
        ));
    }
    let input = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    validate_sha256_text(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let queries = selected_public_flow_ids
        .into_iter()
        .zip(public_helicities)
        .collect::<Vec<_>>();
    let native = py
        .detach(move || {
            let expected_pack_digest = semantic_digest_from_hex(
                &prepared_kernel_pack_digest,
                "prepared kernel pack digest",
            )?;
            let authenticated =
                authenticate_builder_inputs(input, prepared_template, expected_pack_digest)?
                    .authenticated;
            let direct_catalog = parse_direct_template_catalog(
                &direct_template_catalog_json,
                expected_pack_digest,
                authenticated.template().summary().catalog_digest,
                authenticated.template().summary().compiled_model_digest,
            )?;
            on_the_fly_query_family_census_v1(
                &authenticated,
                &direct_catalog.catalog,
                &queries,
                enable_color_projection,
            )
        })
        .map_err(python_error)?;
    on_the_fly_query_family_census_mapping(py, native)
}

#[cfg(feature = "on-the-fly-test-support")]
#[pyfunction(signature = (
    artifact_path,
    process_id,
    builder_input,
    prepared_template_input,
    direct_template_catalog_json,
    prepared_kernel_pack_digest,
    selected_public_flow_id,
    public_helicities,
    point_major_external_momenta,
    point_count,
    *,
    parameter_overrides=None,
    tamper_executor_key=false,
    benchmark=false,
    benchmark_warmup_repetitions=0,
    benchmark_repetitions=0,
    collect_current_diagnostics=true,
    enable_color_projection=true,
    query_family=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _on_the_fly_artifact_probe_v1(
    py: Python<'_>,
    artifact_path: PathBuf,
    process_id: String,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: String,
    selected_public_flow_id: u32,
    public_helicities: Vec<i32>,
    point_major_external_momenta: Vec<f64>,
    point_count: u32,
    parameter_overrides: Option<BTreeMap<String, Vec<f64>>>,
    tamper_executor_key: bool,
    benchmark: bool,
    benchmark_warmup_repetitions: u32,
    benchmark_repetitions: u32,
    collect_current_diagnostics: bool,
    enable_color_projection: bool,
    query_family: Option<Vec<(u32, Vec<i32>)>>,
) -> PyResult<Py<PyAny>> {
    let input = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    validate_sha256_text(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let parameter_overrides = parameter_overrides
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| {
            let value: [f64; 2] = value.try_into().map_err(|value: Vec<f64>| {
                PyValueError::new_err(format!(
                    "on-the-fly parameter override {name:?} has {} components, expected 2",
                    value.len()
                ))
            })?;
            Ok((name, value))
        })
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    let native = py
        .detach(move || {
            let expected_pack_digest = semantic_digest_from_hex(
                &prepared_kernel_pack_digest,
                "prepared kernel pack digest",
            )?;
            let authenticated =
                authenticate_builder_inputs(input, prepared_template, expected_pack_digest)?
                    .authenticated;
            let direct_catalog = parse_direct_template_catalog(
                &direct_template_catalog_json,
                expected_pack_digest,
                authenticated.template().summary().catalog_digest,
                authenticated.template().summary().compiled_model_digest,
            )?;
            NativeRuntime::on_the_fly_artifact_probe_v1(
                artifact_path,
                &process_id,
                &authenticated,
                &direct_catalog.catalog,
                selected_public_flow_id,
                &public_helicities,
                &point_major_external_momenta,
                point_count,
                &parameter_overrides,
                tamper_executor_key,
                benchmark,
                benchmark_warmup_repetitions,
                benchmark_repetitions,
                collect_current_diagnostics,
                enable_color_projection,
                query_family.as_deref(),
            )
        })
        .map_err(python_error)?;
    on_the_fly_artifact_probe_mapping(py, native)
}

#[cfg(feature = "on-the-fly-test-support")]
#[pyfunction(signature = (
    on_the_fly_artifact_path,
    on_the_fly_process_id,
    recurrence_artifact_path,
    recurrence_process_id,
    builder_input,
    prepared_template_input,
    direct_template_catalog_json,
    prepared_kernel_pack_digest,
    selected_public_flow_id,
    public_helicities,
    point_major_external_momenta,
    *,
    parameter_overrides=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _on_the_fly_execution_diagnostic_v1(
    py: Python<'_>,
    on_the_fly_artifact_path: PathBuf,
    on_the_fly_process_id: String,
    recurrence_artifact_path: PathBuf,
    recurrence_process_id: String,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: String,
    selected_public_flow_id: u32,
    public_helicities: Vec<i32>,
    point_major_external_momenta: Vec<f64>,
    parameter_overrides: Option<BTreeMap<String, Vec<f64>>>,
) -> PyResult<Py<PyAny>> {
    let input = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    validate_sha256_text(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let parameter_overrides = parameter_overrides
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| {
            let value: [f64; 2] = value.try_into().map_err(|value: Vec<f64>| {
                PyValueError::new_err(format!(
                    "execution diagnostic parameter override {name:?} has {} components, expected 2",
                    value.len()
                ))
            })?;
            Ok((name, value))
        })
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    let native = py
        .detach(move || {
            let expected_pack_digest = semantic_digest_from_hex(
                &prepared_kernel_pack_digest,
                "prepared kernel pack digest",
            )?;
            let authenticated =
                authenticate_builder_inputs(input, prepared_template, expected_pack_digest)?
                    .authenticated;
            let direct_catalog = parse_direct_template_catalog(
                &direct_template_catalog_json,
                expected_pack_digest,
                authenticated.template().summary().catalog_digest,
                authenticated.template().summary().compiled_model_digest,
            )?;
            NativeRuntime::on_the_fly_execution_diagnostic_v1(
                on_the_fly_artifact_path,
                &on_the_fly_process_id,
                recurrence_artifact_path,
                &recurrence_process_id,
                &authenticated,
                &direct_catalog.catalog,
                selected_public_flow_id,
                &public_helicities,
                &point_major_external_momenta,
                &parameter_overrides,
            )
        })
        .map_err(python_error)?;
    on_the_fly_execution_diagnostic_mapping(py, native)
}

fn build_on_the_fly_process_seeds_v1(
    ordered_source_projection_jsons: Vec<Vec<u8>>,
    recurrence_template_catalog_json: Vec<u8>,
    direct_template_catalog_json: Vec<u8>,
    prepared_kernel_pack_digest: String,
) -> RusticolResult<Vec<Vec<u8>>> {
    let expected_pack_digest =
        semantic_digest_from_hex(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let template_value: JsonValue = serde_json::from_slice(&recurrence_template_catalog_json)
        .map_err(|error| {
            invalid(format!(
                "recurrence-template catalog is not valid JSON: {error}"
            ))
        })?;
    let templates =
        rusticol_core::__private::project_recurrence_template_catalog_json_v1(&template_value)?
            .validate()?;
    let summary = templates.summary();
    if summary.prepared_kernel_pack_digest != expected_pack_digest {
        return Err(RusticolError::integrity(
            "recurrence-template catalog prepared-kernel pack differs from the requested pack",
        ));
    }
    let direct = parse_direct_template_catalog(
        &direct_template_catalog_json,
        expected_pack_digest,
        summary.catalog_digest,
        summary.compiled_model_digest,
    )?;
    rusticol_core::__private::build_on_the_fly_process_seed_bytes_batch_v1(
        &ordered_source_projection_jsons,
        &templates,
        &direct.catalog,
        expected_pack_digest,
    )
}

fn owned_on_the_fly_source_projection_jsons(values: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u8>>> {
    let iterator = values.try_iter().map_err(|_| {
        PyTypeError::new_err("ordered on-the-fly source projections must be iterable")
    })?;
    let mut result = Vec::new();
    for (index, item) in iterator.enumerate() {
        let item = item?;
        let value = item.cast::<PyBytes>().map_err(|_| {
            PyTypeError::new_err(format!(
                "on-the-fly source projection at index {index} must be bytes"
            ))
        })?;
        result.push(value.as_bytes().to_vec());
    }
    Ok(result)
}

/// Private ordered cold-path bridge used by the on-the-fly artifact writer.
///
/// The four byte/string inputs are the complete boundary: in particular this
/// function accepts no recurrence builder input, physical color plan, DAG, or
/// direct-plan payload.
#[pyfunction]
pub(crate) fn _build_on_the_fly_process_seeds_v1(
    py: Python<'_>,
    ordered_source_projection_jsons: &Bound<'_, PyAny>,
    recurrence_template_catalog_json: &Bound<'_, PyBytes>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: &str,
) -> PyResult<Py<PyList>> {
    let ordered_source_projection_jsons =
        owned_on_the_fly_source_projection_jsons(ordered_source_projection_jsons)?;
    let recurrence_template_catalog_json = recurrence_template_catalog_json.as_bytes().to_vec();
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    let prepared_kernel_pack_digest = prepared_kernel_pack_digest.to_owned();
    let encoded = py
        .detach(move || {
            build_on_the_fly_process_seeds_v1(
                ordered_source_projection_jsons,
                recurrence_template_catalog_json,
                direct_template_catalog_json,
                prepared_kernel_pack_digest,
            )
        })
        .map_err(python_error)?;
    let result = PyList::empty(py);
    for payload in encoded {
        result.append(PyBytes::new(py, &payload))?;
    }
    Ok(result.unbind())
}

/// Private singleton wrapper retained for callers of the original bridge.
#[pyfunction]
pub(crate) fn _build_on_the_fly_process_seed_v1(
    py: Python<'_>,
    source_projection_json: &Bound<'_, PyBytes>,
    recurrence_template_catalog_json: &Bound<'_, PyBytes>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: &str,
) -> PyResult<Py<PyBytes>> {
    let source_projection_json = source_projection_json.as_bytes().to_vec();
    let recurrence_template_catalog_json = recurrence_template_catalog_json.as_bytes().to_vec();
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    let prepared_kernel_pack_digest = prepared_kernel_pack_digest.to_owned();
    let mut encoded = py
        .detach(move || {
            build_on_the_fly_process_seeds_v1(
                vec![source_projection_json],
                recurrence_template_catalog_json,
                direct_template_catalog_json,
                prepared_kernel_pack_digest,
            )
        })
        .map_err(python_error)?;
    let payload = encoded.pop().ok_or_else(|| {
        PyValueError::new_err("singleton on-the-fly process-seed batch returned no payload")
    })?;
    Ok(PyBytes::new(py, &payload).unbind())
}

/// Private authoritative decoder used to bind a compact seed to the JSON
/// execution manifest and publication audit.
#[pyfunction]
pub(crate) fn _inspect_on_the_fly_process_seed_v1(
    py: Python<'_>,
    payload: &Bound<'_, PyBytes>,
) -> PyResult<String> {
    let payload = payload.as_bytes().to_vec();
    py.detach(move || {
        rusticol_core::__private::inspect_on_the_fly_process_seed_identity_json_v1(&payload)
    })
    .map_err(python_error)
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_artifact_probe_mapping(
    py: Python<'_>,
    native: NativeOnTheFlyArtifactProbeV1,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("artifact_id", native.artifact_id)?;
    result.set_item("process_id", native.process_id)?;
    result.set_item("seed_digest", native.seed_digest)?;
    result.set_item("query_digest", native.query_digest)?;
    result.set_item("trace_digest", native.trace_digest)?;
    result.set_item("point_count", native.point_count)?;
    result.set_item("raw_amplitudes", native.raw_amplitudes)?;
    result.set_item("normalized_values", native.normalized_values)?;
    result.set_item("normalization_factor", native.normalization_factor)?;
    if let Some(family) = native.query_family {
        let family_result = PyDict::new(py);
        let queries = PyList::empty(py);
        for query in family.queries {
            let row = PyDict::new(py);
            row.set_item("selected_public_flow_id", query.selected_public_flow_id)?;
            row.set_item("public_helicities", query.public_helicities)?;
            row.set_item("query_digest", query.query_digest)?;
            row.set_item("raw_amplitudes", query.raw_amplitudes)?;
            row.set_item("normalized_values", query.normalized_values)?;
            queries.append(row)?;
        }
        family_result.set_item("queries", queries)?;
        family_result.set_item(
            "census",
            on_the_fly_query_family_census_mapping(py, family.census)?,
        )?;
        family_result.set_item("execution_cache_hit", family.execution_cache_hit)?;
        family_result.set_item("execution_source_calls", family.execution_source_calls)?;
        family_result.set_item("execution_source_rows", family.execution_source_rows)?;
        family_result.set_item(
            "execution_contribution_calls",
            family.execution_contribution_calls,
        )?;
        family_result.set_item(
            "execution_contribution_rows",
            family.execution_contribution_rows,
        )?;
        family_result.set_item(
            "execution_finalization_calls",
            family.execution_finalization_calls,
        )?;
        family_result.set_item(
            "execution_finalization_rows",
            family.execution_finalization_rows,
        )?;
        family_result.set_item("execution_closure_calls", family.execution_closure_calls)?;
        family_result.set_item("execution_closure_rows", family.execution_closure_rows)?;
        family_result.set_item("cold_prepare_seconds", family.cold_prepare_seconds)?;
        family_result.set_item(
            "benchmark_warmup_repetitions",
            family.benchmark_warmup_repetitions,
        )?;
        family_result.set_item("benchmark_repetitions", family.benchmark_repetitions)?;
        family_result.set_item(
            "private_warmed_elapsed_seconds",
            family.benchmark_elapsed_seconds,
        )?;
        family_result.set_item(
            "private_warmed_seconds_per_point",
            family.benchmark_seconds_per_point,
        )?;
        family_result.set_item("private_timing_excludes_source_crossing", true)?;
        result.set_item("query_family", family_result)?;
    } else {
        result.set_item("query_family", py.None())?;
    }
    result.set_item("projection_enabled", native.projection_enabled)?;
    result.set_item("projection_applied", native.projection_applied)?;
    for (prefix, counts) in [
        ("pre", native.projection_counts[0]),
        ("post", native.projection_counts[1]),
    ] {
        for (name, value) in ["current", "contribution", "closure"]
            .into_iter()
            .zip(counts)
        {
            result.set_item(format!("{prefix}_projection_{name}_count"), value)?;
        }
    }
    result.set_item("work_census_basis", native.work_census_basis)?;
    result.set_item("logical_current_count", native.logical_current_count)?;
    result.set_item("resident_current_count", native.resident_current_count)?;
    result.set_item(
        "resident_current_component_count",
        native.resident_current_component_count,
    )?;
    result.set_item("source_operation_count", native.source_operation_count)?;
    result.set_item(
        "contribution_operation_count",
        native.contribution_operation_count,
    )?;
    result.set_item(
        "finalization_operation_count",
        native.finalization_operation_count,
    )?;
    result.set_item("closure_operation_count", native.closure_operation_count)?;
    result.set_item(
        "total_kernel_application_count",
        native.total_kernel_application_count,
    )?;
    result.set_item(
        "semantic_executor_binding_count",
        native.semantic_executor_binding_count,
    )?;
    result.set_item(
        "distinct_prepared_executor_count",
        native.distinct_prepared_executor_count,
    )?;
    result.set_item("trace_build_count", native.trace_build_count)?;
    result.set_item("trace_cache_hit_count", native.trace_cache_hit_count)?;
    result.set_item("momentum_fill_count", native.momentum_fill_count)?;
    result.set_item(
        "benchmark_warmup_repetitions",
        native.benchmark_warmup_repetitions,
    )?;
    result.set_item("benchmark_repetitions", native.benchmark_repetitions)?;
    result.set_item(
        "benchmark_elapsed_seconds",
        native.benchmark_elapsed_seconds,
    )?;
    result.set_item(
        "benchmark_seconds_per_point",
        native.benchmark_seconds_per_point,
    )?;
    result.set_item(
        "direct_plan_load_attempts",
        native.direct_plan_load_attempts,
    )?;
    result.set_item(
        "direct_plan_decode_attempts",
        native.direct_plan_decode_attempts,
    )?;
    result.set_item(
        "direct_plan_materialization_attempts",
        native.direct_plan_materialization_attempts,
    )?;
    result.set_item(
        "established_builder_attempts",
        native.established_builder_attempts,
    )?;
    let currents = PyList::empty(py);
    for current in native.currents {
        let row = PyDict::new(py);
        row.set_item("semantic_digest", current.semantic_digest)?;
        row.set_item("component_count", current.component_count)?;
        row.set_item("values", current.values)?;
        currents.append(row)?;
    }
    result.set_item("currents", currents)?;
    Ok(result.into_any().unbind())
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_execution_diagnostic_mapping(
    py: Python<'_>,
    native: NativeOnTheFlyExecutionDiagnosticV1,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("schema_version", 1)?;
    result.set_item("on_the_fly_artifact_id", native.on_the_fly_artifact_id)?;
    result.set_item("on_the_fly_process_id", native.on_the_fly_process_id)?;
    result.set_item("recurrence_artifact_id", native.recurrence_artifact_id)?;
    result.set_item("recurrence_process_id", native.recurrence_process_id)?;
    result.set_item("seed_digest", native.seed_digest)?;
    result.set_item("query_digest", native.query_digest)?;
    result.set_item("trace_digest", native.trace_digest)?;
    result.set_item(
        "direct_plan_semantic_digest",
        native.direct_plan_semantic_digest,
    )?;
    result.set_item(
        "direct_runtime_layout_digest",
        native.direct_runtime_layout_digest,
    )?;
    result.set_item("public_flow_id", native.public_flow_id)?;
    result.set_item("representative_flow_id", native.representative_flow_id)?;
    result.set_item("public_helicities", native.public_helicities)?;
    result.set_item(
        "public_direct_helicity_id",
        native.public_direct_helicity_id,
    )?;
    result.set_item(
        "representative_direct_helicity_id",
        native.representative_direct_helicity_id,
    )?;
    result.set_item("replay_phase", native.replay_phase.to_vec())?;
    result.set_item("replay_multiplicity", native.replay_multiplicity)?;
    result.set_item("replay_scale", native.replay_scale.to_vec())?;
    result.set_item(
        "on_the_fly_public_amplitude",
        native.on_the_fly_public_amplitude.to_vec(),
    )?;
    result.set_item(
        "recurrence_representative_amplitude",
        native.recurrence_representative_amplitude.to_vec(),
    )?;
    result.set_item(
        "recurrence_public_amplitude",
        native.recurrence_public_amplitude.to_vec(),
    )?;
    result.set_item(
        "amplitude_absolute_delta",
        native.amplitude_absolute_delta.to_vec(),
    )?;
    result.set_item("raw_bit_difference_count", native.raw_bit_difference_count)?;
    result.set_item("compared_current_count", native.compared_current_count)?;
    result.set_item(
        "excluded_direct_current_count",
        native.excluded_direct_current_count,
    )?;

    let first_raw_bit_difference = native.first_raw_bit_difference;
    let components = PyList::empty(py);
    for component in native.current_components {
        let row = PyDict::new(py);
        row.set_item("dependency_depth", component.dependency_depth)?;
        row.set_item("semantic_digest", component.semantic_digest)?;
        row.set_item("component", component.component)?;
        row.set_item("on_the_fly", component.on_the_fly.to_vec())?;
        row.set_item("recurrence", component.recurrence.to_vec())?;
        row.set_item("on_the_fly_bits", component.on_the_fly_bits.to_vec())?;
        row.set_item("recurrence_bits", component.recurrence_bits.to_vec())?;
        row.set_item("absolute_delta", component.absolute_delta.to_vec())?;
        components.append(row)?;
    }
    result.set_item("current_components", &components)?;
    if let Some(index) = first_raw_bit_difference {
        result.set_item("first_raw_bit_difference_index", index)?;
        result.set_item("first_raw_bit_difference", components.get_item(index)?)?;
    } else {
        result.set_item("first_raw_bit_difference_index", py.None())?;
        result.set_item("first_raw_bit_difference", py.None())?;
    }
    Ok(result.into_any().unbind())
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_query_family_census_mapping(
    py: Python<'_>,
    native: OnTheFlyQueryFamilyCensusV1,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("query_count", native.query_count)?;
    result.set_item(
        "source_frame_partition_count",
        native.source_frame_partition_count,
    )?;
    result.set_item(
        "projection_applied_query_count",
        native.projection_applied_query_count,
    )?;
    result.set_item(
        "projection_pre_current_count",
        native.projection_pre_current_count,
    )?;
    result.set_item(
        "projection_pre_contribution_count",
        native.projection_pre_contribution_count,
    )?;
    result.set_item(
        "projection_pre_closure_count",
        native.projection_pre_closure_count,
    )?;
    result.set_item(
        "projection_post_current_count",
        native.projection_post_current_count,
    )?;
    result.set_item(
        "projection_post_contribution_count",
        native.projection_post_contribution_count,
    )?;
    result.set_item(
        "projection_post_closure_count",
        native.projection_post_closure_count,
    )?;
    result.set_item(
        "dynamic_current_occurrence_count",
        native.dynamic_current_occurrence_count,
    )?;
    result.set_item(
        "dynamic_current_component_occurrence_count",
        native.dynamic_current_component_occurrence_count,
    )?;
    result.set_item("dynamic_source_rows", native.dynamic_source_rows)?;
    result.set_item(
        "dynamic_contribution_rows",
        native.dynamic_contribution_rows,
    )?;
    result.set_item(
        "dynamic_finalization_rows",
        native.dynamic_finalization_rows,
    )?;
    result.set_item("dynamic_closure_rows", native.dynamic_closure_rows)?;
    result.set_item("dynamic_source_calls", native.dynamic_source_calls)?;
    result.set_item(
        "dynamic_contribution_calls",
        native.dynamic_contribution_calls,
    )?;
    result.set_item(
        "dynamic_finalization_calls",
        native.dynamic_finalization_calls,
    )?;
    result.set_item("dynamic_closure_calls", native.dynamic_closure_calls)?;
    result.set_item(
        "union_unique_current_count",
        native.union_unique_current_count,
    )?;
    result.set_item(
        "union_unique_current_component_count",
        native.union_unique_current_component_count,
    )?;
    result.set_item("union_source_rows", native.union_source_rows)?;
    result.set_item("union_contribution_rows", native.union_contribution_rows)?;
    result.set_item("union_finalization_rows", native.union_finalization_rows)?;
    result.set_item("union_closure_rows", native.union_closure_rows)?;
    result.set_item(
        "union_amplitude_destination_count",
        native.union_amplitude_destination_count,
    )?;
    result.set_item(
        "union_source_executor_call_groups",
        native.union_source_executor_call_groups,
    )?;
    result.set_item(
        "union_contribution_executor_call_groups",
        native.union_contribution_executor_call_groups,
    )?;
    result.set_item(
        "union_finalization_executor_call_groups",
        native.union_finalization_executor_call_groups,
    )?;
    result.set_item(
        "union_closure_executor_call_groups",
        native.union_closure_executor_call_groups,
    )?;
    Ok(result.into_any().unbind())
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_exact_factor_mapping(
    py: Python<'_>,
    factor: ExactComplexRational,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    for (name, value) in [("real", factor.real()), ("imag", factor.imag())] {
        let part = PyDict::new(py);
        part.set_item("numerator", value.numerator().to_string())?;
        part.set_item("denominator", value.denominator().to_string())?;
        result.set_item(name, part)?;
    }
    Ok(result.into_any().unbind())
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_optional_exact_factor_mapping(
    py: Python<'_>,
    factor: Option<ExactComplexRational>,
) -> PyResult<Py<PyAny>> {
    match factor {
        Some(factor) => on_the_fly_exact_factor_mapping(py, factor),
        None => Ok(py.None()),
    }
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_transition_candidate_mapping(
    py: Python<'_>,
    row: ConstructionTransitionDiagnosticRowV1,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("materialized_sector_id", row.materialized_sector_id)?;
    result.set_item(
        "output_current_digest",
        row.output_current_digest.to_string(),
    )?;
    result.set_item(
        "ordered_parent_digests",
        row.ordered_parent_digests
            .into_iter()
            .map(|digest| digest.to_string())
            .collect::<Vec<_>>(),
    )?;
    result.set_item("transition_template_id", row.transition_template_id)?;
    result.set_item(
        "transition_semantic_digest",
        row.transition_semantic_digest.to_string(),
    )?;
    result.set_item(
        "evaluator_binding_semantic_digest",
        row.evaluator_binding_semantic_digest.to_string(),
    )?;
    result.set_item("result_state_template_id", row.result_state_template_id)?;
    result.set_item("quantum_flow_witness_id", row.quantum_flow_witness_id)?;
    result.set_item(
        "quantum_semantic_digest",
        row.quantum_semantic_digest.to_string(),
    )?;
    result.set_item(
        "color_contraction_template_id",
        row.color_contraction_template_id,
    )?;
    result.set_item("color_witness_ordinal", row.color_witness_ordinal)?;
    result.set_item(
        "color_witness_proof_digest",
        row.color_witness_proof_digest.to_string(),
    )?;
    result.set_item("output_projection_id", row.output_projection_id)?;
    let factors = PyDict::new(py);
    for (name, factor) in [
        ("transition", row.transition_factor),
        ("contraction", row.contraction_factor),
        ("output", row.output_factor),
        ("exchange", row.exchange_factor),
        ("witness", row.witness_factor),
        ("reversal", row.reversal_factor),
        ("candidate_product", row.candidate_factor),
        ("aggregate_after", row.aggregate_factor_after),
    ] {
        factors.set_item(name, on_the_fly_exact_factor_mapping(py, factor)?)?;
    }
    result.set_item("factor_components", factors)?;
    result.set_item("reversal_mask", row.reversal_mask)?;
    let reflections = PyDict::new(py);
    reflections.set_item(
        "parent_proof_digests",
        row.parent_reflection_proof_digests
            .into_iter()
            .map(|digest| digest.map(|value| value.to_string()))
            .collect::<Vec<_>>(),
    )?;
    let parent_phases = PyList::empty(py);
    for phase in row.parent_reflection_phases {
        parent_phases.append(on_the_fly_optional_exact_factor_mapping(py, phase)?)?;
    }
    reflections.set_item("parent_phases", parent_phases)?;
    reflections.set_item(
        "local_proof_digest",
        row.local_reflection_proof_digest
            .map(|digest| digest.to_string()),
    )?;
    reflections.set_item(
        "local_phase",
        on_the_fly_optional_exact_factor_mapping(py, row.local_reflection_phase)?,
    )?;
    reflections.set_item(
        "result_proof_digest",
        row.result_reflection_proof_digest
            .map(|digest| digest.to_string()),
    )?;
    reflections.set_item(
        "result_phase",
        on_the_fly_optional_exact_factor_mapping(py, row.result_reflection_phase)?,
    )?;
    result.set_item("reflection", reflections)?;
    result.set_item("output_color_orientation", row.output_color_orientation)?;
    result.set_item("post_row_complex_value", py.None())?;
    Ok(result.into_any().unbind())
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_transition_candidates_mapping(
    py: Python<'_>,
    mut rows: Vec<ConstructionTransitionDiagnosticRowV1>,
) -> PyResult<Py<PyList>> {
    rows.sort_by_cached_key(|row| format!("{row:?}"));
    let result = PyList::empty(py);
    for row in rows {
        result.append(on_the_fly_transition_candidate_mapping(py, row)?)?;
    }
    Ok(result.unbind())
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_test_support_mapping(
    py: Python<'_>,
    native: OnTheFlyTestSupportReportV1,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    let structural_parity = native.structural_parity();
    result.set_item("seed_digest", native.seed_digest.to_string())?;
    result.set_item("query_digest", native.query_digest.to_string())?;
    result.set_item("selector_digest", native.selector_digest.to_string())?;
    result.set_item("trace_digest", native.trace_digest.to_string())?;
    result.set_item("current_count", native.current_count)?;
    result.set_item("contribution_count", native.contribution_count)?;
    result.set_item("closure_count", native.closure_count)?;
    result.set_item(
        "established_current_count",
        native.established_current_count,
    )?;
    result.set_item(
        "established_contribution_count",
        native.established_contribution_count,
    )?;
    result.set_item(
        "established_closure_count",
        native.established_closure_count,
    )?;
    result.set_item(
        "current_multiset_digest",
        native.current_multiset_digest.to_string(),
    )?;
    result.set_item(
        "established_current_multiset_digest",
        native.established_current_multiset_digest.to_string(),
    )?;
    result.set_item(
        "contribution_multiset_digest",
        native.contribution_multiset_digest.to_string(),
    )?;
    result.set_item(
        "established_contribution_multiset_digest",
        native.established_contribution_multiset_digest.to_string(),
    )?;
    result.set_item(
        "closure_multiset_digest",
        native.closure_multiset_digest.to_string(),
    )?;
    result.set_item(
        "closure_parity_multiset_digest",
        native.closure_parity_multiset_digest.to_string(),
    )?;
    result.set_item(
        "established_closure_parity_multiset_digest",
        native
            .established_closure_parity_multiset_digest
            .to_string(),
    )?;
    result.set_item(
        "negative_contribution_factor_count",
        native.negative_contribution_factor_count,
    )?;
    result.set_item(
        "established_negative_contribution_factor_count",
        native.established_negative_contribution_factor_count,
    )?;
    result.set_item("source_domain_equal", native.source_domain_equal)?;
    result.set_item("pairing_oracle_equal", native.pairing_oracle_equal)?;
    result.set_item("pairing_fermion_parities", native.pairing_fermion_parities)?;
    result.set_item(
        "established_pairing_fermion_parities",
        native.established_pairing_fermion_parities,
    )?;
    result.set_item(
        "workspace_capacity_independent",
        native.workspace_capacity_independent,
    )?;
    result.set_item("structural_parity", structural_parity)?;
    result.set_item(
        "compact_transition_candidates",
        on_the_fly_transition_candidates_mapping(py, native.compact_transition_candidates)?,
    )?;
    result.set_item(
        "established_transition_candidates",
        on_the_fly_transition_candidates_mapping(py, native.established_transition_candidates)?,
    )?;
    Ok(result.into_any().unbind())
}

#[allow(clippy::too_many_arguments)]
fn lower_recurrence_direct(
    input: OwnedInput,
    prepared_template: PreparedTemplateInput,
    direct_template_catalog_json: &[u8],
    prepared_kernel_pack_digest: &str,
    schedule_semantic_digest: &str,
    destination: &std::path::Path,
    point_tile_size: u32,
    workspace_mib: u32,
    mut relation_discovery: RecurrenceRelationDiscoveryOptions,
    relation_discovery_evidence_json: Option<Vec<u8>>,
    helicity_dispatch_destination: Option<&std::path::Path>,
    python_extraction_seconds: f64,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<NativeDirectLoweringResult> {
    let native_started = Instant::now();
    if std::fs::symlink_metadata(destination).is_ok() {
        return Err(invalid(format!(
            "recurrence direct-plan destination already exists: {}",
            destination.display()
        )));
    }
    if let Some(dispatch_destination) = helicity_dispatch_destination {
        if std::fs::symlink_metadata(dispatch_destination).is_ok() {
            return Err(invalid(format!(
                "recurrence helicity-dispatch destination already exists: {}",
                dispatch_destination.display()
            )));
        }
        if relation_discovery.mode != RecurrenceRelationDiscoveryMode::Off
            || relation_discovery_evidence_json.is_some()
        {
            return Err(invalid(
                "recurrence helicity-dispatch lowering requires relation discovery off",
            ));
        }
    }

    let expected_pack_digest =
        semantic_digest_from_hex(prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let authenticated_inputs =
        authenticate_builder_inputs(input, prepared_template, expected_pack_digest)?;
    let builder_input_digest = authenticated_inputs.builder_input_digest;
    let template_input_digest = authenticated_inputs.template_input_digest;
    let authenticated = authenticated_inputs.authenticated;

    let catalog_started = Instant::now();
    let direct_catalog = parse_direct_template_catalog(
        direct_template_catalog_json,
        expected_pack_digest,
        authenticated.template().summary().catalog_digest,
        authenticated.template().summary().compiled_model_digest,
    )?;
    let catalog_authentication_seconds = catalog_started.elapsed().as_secs_f64();

    let process_id = authenticated.process().summary().process_id().to_owned();
    let strategy = authenticated.process().summary().strategy();
    let semantic_digest =
        semantic_digest_from_hex(schedule_semantic_digest, "recurrence schedule digest")?;
    if let Some(raw_evidence) = relation_discovery_evidence_json {
        let parameter_contract = authenticated_runtime_parameter_contract(
            authenticated.process(),
            authenticated.template(),
        )?;
        relation_discovery = relation_discovery.with_numerical_evidence(
            parse_numerical_relation_evidence(&raw_evidence, &parameter_contract)?,
        )?;
    }

    let semantic_started = Instant::now();
    let mut construction = RecurrenceConstructionMetrics::default();
    let mut tracked_progress = |snapshot: RecurrenceBuildProgress| {
        construction.include(&snapshot);
        progress(snapshot)
    };
    let (program, generation_profile) =
        authenticated.build_with_progress_and_telemetry(&mut tracked_progress)?;
    let closure_target_domain_certificate =
        program
            .closure_target_domain_certificate()
            .map(|certificate| NativeClosureTargetDomainCertificate {
                physical_sector_count: certificate.physical_sector_count(),
                resolved_helicity_count: certificate.resolved_helicity_count(),
                certified_structural_zero_physical_sector_ids: certificate
                    .certified_structural_zero_physical_sector_ids()
                    .to_vec(),
                semantic_digest: certificate.semantic_digest().to_string(),
            });
    let projection_certificate = program
        .color_projection_certificate_body()
        .map(|body| body.to_vec());
    let semantic_construction_seconds = semantic_started.elapsed().as_secs_f64();

    let direct_lowering_started = Instant::now();
    let runtime_options = DirectRecurrenceRuntimeOptions::new(point_tile_size, workspace_mib)?;
    if let Some(evidence) = relation_discovery.numerical_evidence.as_ref() {
        let baseline_plan = lower_recurrence_direct_plan_v2(
            &program,
            authenticated.template(),
            &direct_catalog.catalog,
            semantic_digest,
            expected_pack_digest,
            direct_catalog.catalog_digest,
            runtime_options,
        )?;
        authenticate_recurrence_numerical_relation_provenance(
            evidence,
            &baseline_plan,
            &process_id,
        )?;
    }
    let (plan, relation_discovery_report, helicity_dispatch) =
        if helicity_dispatch_destination.is_some() {
            let (plan, dispatch) = lower_recurrence_direct_plan_v2_with_helicity_dispatch(
                &program,
                authenticated.template(),
                &direct_catalog.catalog,
                semantic_digest,
                expected_pack_digest,
                direct_catalog.catalog_digest,
                runtime_options,
            )?;
            (plan, None, Some(dispatch))
        } else {
            let (plan, report) = lower_recurrence_direct_plan_v2_with_relation_discovery(
                &program,
                authenticated.template(),
                &direct_catalog.catalog,
                semantic_digest,
                expected_pack_digest,
                direct_catalog.catalog_digest,
                runtime_options,
                &relation_discovery,
            )?;
            (plan, report, None)
        };
    let direct_lowering_seconds = direct_lowering_started.elapsed().as_secs_f64();

    let resolved_helicities = resolved_helicities_from_direct_plan(&plan)?;
    let amplitude_destinations = plan
        .amplitude_destinations()
        .iter()
        .map(|destination| {
            (
                destination.target_sector_id,
                (destination.target_helicity_id_or_sentinel != DIRECT_NONE_U32)
                    .then_some(destination.target_helicity_id_or_sentinel),
            )
        })
        .collect();
    let semantic_component_count = plan.currents().iter().try_fold(0_u64, |total, current| {
        total
            .checked_add(u64::from(current.component_count))
            .ok_or_else(|| invalid("recurrence semantic component count exceeds u64"))
    })?;
    let selector_work =
        if strategy.uses_topology_replay_targets() && !plan.replay_targets().is_empty() {
            let mut public_flow_counts = BTreeMap::<u32, u64>::new();
            for target in plan.replay_targets() {
                let count = public_flow_counts
                    .entry(target.representative_id)
                    .or_default();
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| invalid("recurrence public-flow count exceeds u64"))?;
            }
            public_flow_counts
                .into_iter()
                .map(|(representative_sector_id, public_flow_count)| {
                    Ok((
                        plan.selector_work_summary(representative_sector_id)?,
                        public_flow_count,
                    ))
                })
                .collect::<RusticolResult<Vec<_>>>()?
        } else {
            vec![]
        };
    let exact_sections = exact_sections_from_direct_plan(
        &plan,
        process_id.clone(),
        direct_catalog.exact_executors.clone(),
    );

    let serialization_started = Instant::now();
    let metadata = write_recurrence_direct_plan_pacbin_with_projection_certificate(
        destination,
        &plan,
        projection_certificate.as_deref(),
    )?;
    let helicity_dispatch = match (helicity_dispatch_destination, helicity_dispatch.as_ref()) {
        (Some(dispatch_destination), Some(dispatch)) => {
            let dispatch_metadata = match write_recurrence_helicity_dispatch_v1_atomic(
                dispatch_destination,
                dispatch,
            ) {
                Ok(metadata) => metadata,
                Err(error) => {
                    let _ = std::fs::remove_file(destination);
                    return Err(error);
                }
            };
            Some(NativeHelicityDispatchMetadata {
                size_bytes: dispatch_metadata.size_bytes,
                sha256: hex_digest(dispatch_metadata.sha256),
                base_runtime_layout_digest: dispatch.runtime_layout_digest().to_string(),
                resolved_helicity_count: dispatch.resolved_helicity_count() as usize,
            })
        }
        (None, None) => None,
        _ => {
            let _ = std::fs::remove_file(destination);
            return Err(invalid(
                "recurrence helicity-dispatch destination and lowering result disagree",
            ));
        }
    };
    let serialization_seconds = serialization_started.elapsed().as_secs_f64();
    let native_total_seconds = native_started.elapsed().as_secs_f64();

    Ok(NativeDirectLoweringResult {
        builder_input_digest,
        template_input_digest,
        prepared_kernel_pack_digest: direct_catalog.prepared_kernel_pack_digest.to_string(),
        direct_template_catalog_digest: direct_catalog.catalog_digest.to_string(),
        process_id,
        strategy,
        semantic_digest: plan.semantic_digest().to_string(),
        runtime_layout_digest: plan.runtime_layout_digest().to_string(),
        member_count: metadata.member_count,
        unpacked_size_bytes: metadata.unpacked_size_bytes,
        index_sha256: hex_digest(metadata.index_sha256),
        container_size: metadata.container_size,
        plan_payload_size: metadata.plan_payload_size,
        plan_sha256: hex_digest(metadata.plan_sha256),
        projection_certificate_payload_size: metadata.projection_certificate_payload_size,
        projection_certificate_sha256: metadata.projection_certificate_sha256.map(hex_digest),
        current_count: plan.currents().len(),
        source_row_count: plan.sources().len(),
        contribution_count: plan.contributions().len(),
        finalization_count: plan.finalizations().len(),
        closure_count: plan.closures().len(),
        row_group_count: plan.row_groups().len(),
        momentum_form_count: plan.momentum_forms().len(),
        selector_domain_count: plan.selector_domains().len(),
        replay_target_count: plan.replay_targets().len(),
        amplitude_destination_count: plan.amplitude_destinations().len(),
        resolved_helicity_count: plan.resolved_helicities().len(),
        retained_helicity_count: plan.retained_helicity_count(),
        exact_factor_count: plan.exact_factors().len(),
        semantic_component_count,
        current_arena_components: plan.current_arena_components(),
        parameter_value_count: plan.parameter_value_count(),
        physical_sector_count: plan.physical_sector_count(),
        direct_executor_count: plan.direct_executor_count(),
        prepared_kernel_count: direct_catalog.prepared_kernel_count,
        resolved_helicities,
        amplitude_destinations,
        closure_target_domain_certificate,
        selector_work,
        relation_discovery: relation_discovery_report,
        exact_sections,
        construction,
        generation_profile,
        timings: DirectLoweringTimings {
            python_extraction_seconds,
            catalog_authentication_seconds,
            semantic_construction_seconds,
            direct_lowering_seconds,
            serialization_seconds,
            native_total_seconds,
        },
        helicity_dispatch,
    })
}

fn resolved_helicities_from_direct_plan(
    plan: &DirectRecurrencePlan,
) -> RusticolResult<Vec<Vec<i32>>> {
    let public = plan.public_helicities();
    plan.resolved_helicities()
        .iter()
        .enumerate()
        .map(|(index, row)| {
            if row.id != index as u32 {
                return Err(invalid(format!(
                    "direct resolved helicity {index} has non-canonical ID {}",
                    row.id
                )));
            }
            let start = usize::try_from(row.public_helicity_start)
                .map_err(|_| invalid("direct resolved-helicity offset exceeds usize"))?;
            let count = usize::try_from(row.public_helicity_count)
                .map_err(|_| invalid("direct resolved-helicity count exceeds usize"))?;
            let end = start
                .checked_add(count)
                .ok_or_else(|| invalid("direct resolved-helicity range exceeds usize"))?;
            public
                .get(start..end)
                .map(<[i32]>::to_vec)
                .ok_or_else(|| invalid("direct resolved-helicity range is out of bounds"))
        })
        .collect()
}

pub(super) fn canonical_json_sha256(value: &JsonValue, context: &str) -> RusticolResult<String> {
    Ok(hex_digest(Sha256::digest(canonical_json_bytes(
        value, context,
    )?)))
}

fn parse_direct_template_catalog(
    bytes: &[u8],
    expected_pack_digest: SemanticDigest,
    expected_template_catalog_digest: SemanticDigest,
    expected_compiled_model_digest: SemanticDigest,
) -> RusticolResult<AuthenticatedDirectTemplateCatalog> {
    if bytes.is_empty() {
        return Err(invalid("direct-template catalog JSON must not be empty"));
    }
    let value: JsonValue = serde_json::from_slice(bytes).map_err(|error| {
        invalid(format!(
            "direct-template catalog is not valid JSON: {error}"
        ))
    })?;
    let canonical = canonical_json_bytes(&value, "direct-template catalog")?;
    if canonical != bytes {
        return Err(invalid(
            "direct-template catalog JSON is not canonical ASCII JSON",
        ));
    }
    let object = json_object(&value, "direct-template catalog")?;
    require_json_fields(
        object,
        &[
            "abi",
            "backend",
            "backend_abi",
            "canonicalization_abi",
            "catalog_digest",
            "compiled_model_digest",
            "optimization_level",
            "optimization_settings_digest",
            "portable",
            "prepared_kernel_contract_digest",
            "prepared_kernel_pack_digest",
            "prepared_kernel_payload_digest",
            "recurrence_template_catalog_digest",
            "target_triple",
            "templates",
        ],
        "direct-template catalog",
    )?;
    require_json_string_value(
        object,
        "abi",
        RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct-template catalog",
    )?;
    require_json_string_value(
        object,
        "backend_abi",
        DIRECT_BACKEND_ABI,
        "direct-template catalog",
    )?;
    require_json_string_value(
        object,
        "canonicalization_abi",
        DIRECT_CANONICALIZATION_ABI,
        "direct-template catalog",
    )?;

    let catalog_digest = json_sha256(object, "catalog_digest", "direct-template catalog digest")?;
    let actual_catalog_digest = digest_json_without_field(
        &value,
        "catalog_digest",
        "direct-template catalog semantic payload",
    )?;
    if catalog_digest != actual_catalog_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog digest does not match its canonical payload",
        ));
    }
    let prepared_kernel_pack_digest = json_sha256(
        object,
        "prepared_kernel_pack_digest",
        "direct-template prepared-kernel pack digest",
    )?;
    if prepared_kernel_pack_digest != expected_pack_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog prepared-kernel pack digest does not match the requested pack",
        ));
    }
    let recurrence_template_catalog_digest = json_sha256(
        object,
        "recurrence_template_catalog_digest",
        "direct-template semantic catalog digest",
    )?;
    if recurrence_template_catalog_digest != expected_template_catalog_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog does not match the authenticated recurrence template",
        ));
    }
    let compiled_model_digest = json_sha256(
        object,
        "compiled_model_digest",
        "direct-template compiled-model digest",
    )?;
    if compiled_model_digest != expected_compiled_model_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog does not match the authenticated compiled model",
        ));
    }
    for (field, context) in [
        (
            "prepared_kernel_contract_digest",
            "direct-template prepared-kernel contract digest",
        ),
        (
            "prepared_kernel_payload_digest",
            "direct-template prepared-kernel payload digest",
        ),
        (
            "optimization_settings_digest",
            "direct-template optimization-settings digest",
        ),
    ] {
        json_sha256(object, field, context)?;
    }

    let backend = json_string(object, "backend", "direct-template backend")?;
    if !matches!(backend, "jit" | "cpp" | "asm") {
        return Err(invalid(format!(
            "direct-template catalog has unsupported backend {backend:?}"
        )));
    }
    let target_triple =
        json_nonempty_string(object, "target_triple", "direct-template target triple")?;
    let portable = json_bool(object, "portable", "direct-template portable flag")?;
    let optimization_level = json_u32(
        object,
        "optimization_level",
        "direct-template optimization level",
    )?;
    match backend {
        "jit" if !portable || optimization_level != 2 => {
            return Err(invalid(
                "prepared direct JIT catalogs must use portable SymJIT O2",
            ));
        }
        "cpp" | "asm" if portable => {
            return Err(invalid(
                "prepared direct C++/ASM catalogs must be target-native",
            ));
        }
        _ => {}
    }

    let templates = json_array(object, "templates", "direct-template templates")?;
    if templates.is_empty() {
        return Err(invalid(
            "direct-template executor catalog must not be empty",
        ));
    }
    let mut bindings = Vec::with_capacity(templates.len());
    let mut exact_executors = Vec::with_capacity(templates.len());
    let mut prepared_kernel_ids = BTreeSet::new();
    let mut identity_finalizer_seen = false;
    for (expected_executor_id, template_value) in templates.iter().enumerate() {
        let context = format!("direct template {expected_executor_id}");
        let template = json_object(template_value, &context)?;
        require_json_fields(
            template,
            &[
                "abi",
                "alignment_bytes",
                "backend",
                "coupling_slot_count",
                "destination_aliasing",
                "destination_component_count",
                "destination_operation",
                "direct_executor_id",
                "evaluator_binding_id",
                "evaluator_resolver_key",
                "exact_expression_digest",
                "momentum_operand_count",
                "optimization_level",
                "parameter_slot_count",
                "parent_arity",
                "parent_component_counts",
                "payload_binding",
                "portable",
                "role",
                "semantic_digest",
                "semantic_template_ids",
                "simd_axis",
                "target_triple",
                "template_id",
            ],
            &context,
        )?;
        require_json_string_value(template, "abi", RECURRENCE_DIRECT_TEMPLATE_ABI, &context)?;
        let actual_template_digest =
            digest_json_without_field(template_value, "semantic_digest", &context)?;
        let template_digest = json_sha256(
            template,
            "semantic_digest",
            &format!("{context} semantic digest"),
        )?;
        if template_digest != actual_template_digest {
            return Err(RusticolError::integrity(format!(
                "{context} semantic digest does not match its canonical payload"
            )));
        }

        let direct_executor_id = json_u32(
            template,
            "direct_executor_id",
            &format!("{context} executor ID"),
        )?;
        let expected_executor_id = u32::try_from(expected_executor_id)
            .map_err(|_| invalid("direct-template executor count exceeds u32"))?;
        if direct_executor_id != expected_executor_id {
            return Err(invalid(format!(
                "{context} has executor ID {direct_executor_id}, expected dense ID {expected_executor_id}"
            )));
        }
        let evaluator_binding_id = json_u32(
            template,
            "evaluator_binding_id",
            &format!("{context} evaluator-binding ID"),
        )?;
        let role_text = json_string(template, "role", &format!("{context} role"))?;
        let role = direct_executor_role(role_text, &context)?;
        let expected_operation = match role {
            DirectExecutorRole::Source => "initialize",
            DirectExecutorRole::Contribution => "add",
            DirectExecutorRole::Finalization => "finalize-in-place",
            DirectExecutorRole::Closure => "closure-add",
        };
        require_json_string_value(
            template,
            "destination_operation",
            expected_operation,
            &context,
        )?;
        require_json_string_value(template, "backend", backend, &context)?;
        require_json_string_value(template, "target_triple", target_triple, &context)?;
        if json_bool(template, "portable", &format!("{context} portable flag"))? != portable
            || json_u32(
                template,
                "optimization_level",
                &format!("{context} optimization level"),
            )? != optimization_level
        {
            return Err(invalid(format!(
                "{context} backend policy does not match its catalog"
            )));
        }
        require_json_string_value(template, "simd_axis", "points-contiguous", &context)?;
        let destination_aliasing = json_bool(
            template,
            "destination_aliasing",
            &format!("{context} destination aliasing"),
        )?;
        if destination_aliasing != (role == DirectExecutorRole::Finalization) {
            return Err(invalid(format!(
                "{context} has an invalid destination-aliasing contract"
            )));
        }
        json_nonempty_string(template, "template_id", &format!("{context} template ID"))?;
        json_nonempty_string(
            template,
            "evaluator_resolver_key",
            &format!("{context} evaluator resolver key"),
        )?;
        json_sha256(
            template,
            "exact_expression_digest",
            &format!("{context} exact-expression digest"),
        )?;
        validate_direct_template_shapes(template, &context)?;
        let parent_component_counts = json_array(
            template,
            "parent_component_counts",
            &format!("{context} parent component counts"),
        )?
        .iter()
        .enumerate()
        .map(|(index, value)| json_value_u32(value, &format!("{context} parent component {index}")))
        .collect::<RusticolResult<Vec<_>>>()?;
        let destination_component_count = json_u32(
            template,
            "destination_component_count",
            &format!("{context} destination component count"),
        )?;
        let momentum_operand_count = json_u32(
            template,
            "momentum_operand_count",
            &format!("{context} momentum operand count"),
        )?;

        let payload = json_object(
            json_field(template, "payload_binding", &context)?,
            &format!("{context} payload binding"),
        )?;
        require_json_fields_with_optional(
            payload,
            &[
                "abi",
                "destination_operation",
                "direct_application_abi",
                "exact_factor_scalar_slots",
                "input_plane_count",
                "input_plane_projections",
                "intrinsic_contract_digest",
                "kind",
                "output_alias_inputs",
                "contribution_parent_permutation",
                "parameter_bindings",
                "payload_digest",
                "payload_paths",
                "prepared_kernel_id",
                "prepared_template_semantic_digest",
                "role",
                "runtime_template",
                "scalar_input_count",
                "scalar_projections",
                "source_application_abi",
                "source_application_path",
                "source_application_sha256",
                "state_plane_indices",
            ],
            &["native_entry_point"],
            &format!("{context} payload binding"),
        )?;
        require_json_string_value(
            payload,
            "abi",
            DIRECT_PAYLOAD_BINDING_ABI,
            &format!("{context} payload binding"),
        )?;
        json_sha256(
            payload,
            "payload_digest",
            &format!("{context} payload digest"),
        )?;
        let payload_kind = json_string(payload, "kind", &format!("{context} payload kind"))?;
        if !matches!(
            payload_kind,
            "rusticol-intrinsic" | "prepared-direct-call" | "pending-direct-call-abi"
        ) {
            return Err(invalid(format!(
                "{context} has unsupported payload kind {payload_kind:?}"
            )));
        }
        if payload_kind == "pending-direct-call-abi" {
            return Err(RusticolError::compatibility(format!(
                "{context} has no executable Direct-Arena payload; rebuild the prepared model"
            )));
        }
        let parent_permutation_values = json_array(
            payload,
            "contribution_parent_permutation",
            &format!("{context} parent permutation"),
        )?;
        if parent_permutation_values.len() != 2 {
            return Err(invalid(format!(
                "{context} parent permutation must contain two entries"
            )));
        }
        let mut parent_permutation = [0_u8; 2];
        for (index, value) in parent_permutation_values.iter().enumerate() {
            parent_permutation[index] = u8::try_from(json_value_u32(
                value,
                &format!("{context} parent permutation entry {index}"),
            )?)
            .map_err(|_| invalid(format!("{context} parent permutation exceeds u8")))?;
        }
        if !matches!(parent_permutation, [0, 1] | [1, 0])
            || (parent_permutation != [0, 1]
                && !(payload_kind == "rusticol-intrinsic"
                    && role == DirectExecutorRole::Contribution))
        {
            return Err(invalid(format!(
                "{context} has an invalid parent permutation"
            )));
        }
        let runtime_template = json_optional_string(
            payload,
            "runtime_template",
            &format!("{context} runtime template"),
        )?;
        let prepared_kernel_id = json_optional_u32(
            payload,
            "prepared_kernel_id",
            &format!("{context} prepared-kernel ID"),
        )?;
        exact_executors.push(NativeRecurrenceExactExecutor {
            direct_executor_id,
            role: role_text.to_owned(),
            destination_operation: expected_operation.to_owned(),
            parent_component_counts,
            destination_component_count,
            momentum_operand_count,
            prepared_kernel_id,
            runtime_template: runtime_template.map(str::to_owned),
        });
        let native_entry_point = match payload.get("native_entry_point") {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::String(value)) if !value.is_empty() => Some(value.as_str()),
            Some(_) => {
                return Err(invalid(format!(
                    "{context} native entry point must be a nonempty string or null"
                )));
            }
        };
        match (payload_kind, backend) {
            ("prepared-direct-call", "cpp" | "asm") => {
                let kernel_id = prepared_kernel_id.ok_or_else(|| {
                    invalid(format!(
                        "{context} native prepared call has no prepared-kernel ID"
                    ))
                })?;
                let expected =
                    format!("pyamplicol_recurrence_direct_{role_text}_k{kernel_id:08x}_v1");
                if native_entry_point != Some(expected.as_str()) {
                    return Err(invalid(format!(
                        "{context} native prepared call has a noncanonical entry point"
                    )));
                }
            }
            _ if native_entry_point.is_some() => {
                return Err(invalid(format!(
                    "{context} non-native payload carries a native entry point"
                )));
            }
            _ => {}
        }
        let intrinsic_contract_digest = json_optional_string(
            payload,
            "intrinsic_contract_digest",
            &format!("{context} intrinsic contract digest"),
        )?;
        if let Some(digest) = intrinsic_contract_digest {
            semantic_digest_from_hex(digest, &format!("{context} intrinsic contract digest"))?;
        }
        if payload_kind == "rusticol-intrinsic"
            && role == DirectExecutorRole::Contribution
            && intrinsic_contract_digest.is_none()
        {
            return Err(invalid(format!(
                "{context} contribution intrinsic has no certified contract digest"
            )));
        }
        if payload_kind != "rusticol-intrinsic" && intrinsic_contract_digest.is_some() {
            return Err(invalid(format!(
                "{context} non-intrinsic payload carries an intrinsic contract digest"
            )));
        }
        validate_string_array(
            json_array(
                payload,
                "payload_paths",
                &format!("{context} payload paths"),
            )?,
            &format!("{context} payload paths"),
        )?;

        let identity_finalizer =
            runtime_template.is_some_and(|name| name == DIRECT_IDENTITY_FINALIZER);
        if identity_finalizer {
            if identity_finalizer_seen {
                return Err(invalid(
                    "direct-template catalog contains more than one generic identity finalizer",
                ));
            }
            if role != DirectExecutorRole::Finalization
                || payload_kind != "rusticol-intrinsic"
                || prepared_kernel_id.is_some()
            {
                return Err(invalid(format!(
                    "{context} has an invalid identity-finalizer contract"
                )));
            }
            identity_finalizer_seen = true;
            bindings.push(PreparedDirectExecutorBinding::identity_finalizer(
                direct_executor_id,
            ));
        } else {
            if let Some(kernel_id) = prepared_kernel_id {
                prepared_kernel_ids.insert(kernel_id);
            }
            bindings.push(
                PreparedDirectExecutorBinding::evaluator_with_parent_permutation(
                    role,
                    evaluator_binding_id,
                    direct_executor_id,
                    parent_permutation,
                ),
            );
        }
    }

    let catalog = PreparedDirectExecutorCatalog::new(catalog_digest, bindings)?;
    Ok(AuthenticatedDirectTemplateCatalog {
        catalog,
        catalog_digest,
        prepared_kernel_pack_digest,
        prepared_kernel_count: prepared_kernel_ids.len(),
        exact_executors,
    })
}

fn exact_sections_from_direct_plan(
    plan: &DirectRecurrencePlan,
    process_id: String,
    executors: Vec<NativeRecurrenceExactExecutor>,
) -> NativeRecurrenceExactSections {
    let public_flow_ids = match plan.strategy() {
        RecurrenceStrategy::TopologyReplay => plan
            .replay_targets()
            .iter()
            .map(|target| target.public_flow_id)
            .collect(),
        RecurrenceStrategy::AllFlowUnion => plan
            .amplitude_destinations()
            .iter()
            .map(|destination| destination.target_sector_id)
            .collect(),
        RecurrenceStrategy::ContractedColorUnion => Vec::new(),
    };
    let exact_factors = plan
        .exact_factors()
        .iter()
        .map(|factor| NativeRecurrenceExactFactor {
            real_numerator: factor.real().numerator().to_string(),
            real_denominator: factor.real().denominator().to_string(),
            imaginary_numerator: factor.imag().numerator().to_string(),
            imaginary_denominator: factor.imag().denominator().to_string(),
        })
        .collect();
    NativeRecurrenceExactSections {
        process_id,
        strategy: plan.strategy().as_str().to_owned(),
        semantic_digest: plan.semantic_digest().to_string(),
        runtime_layout_digest: plan.runtime_layout_digest().to_string(),
        current_arena_components: plan.current_arena_components(),
        amplitude_destination_count: plan.amplitude_destination_count(),
        parameter_value_count: plan.parameter_value_count(),
        external_source_count: plan.external_source_count(),
        currents: plan.currents().to_vec(),
        sources: plan.sources().to_vec(),
        contributions: plan.contributions().to_vec(),
        finalizations: plan.finalizations().to_vec(),
        closures: plan.closures().to_vec(),
        row_groups: plan.row_groups().to_vec(),
        momentum_forms: plan.momentum_forms().to_vec(),
        momentum_terms: plan.momentum_terms().to_vec(),
        replay_targets: plan.replay_targets().to_vec(),
        source_permutations: plan.source_permutations().to_vec(),
        replay_momentum_signs: plan.replay_momentum_signs().to_vec(),
        replay_helicity_map: plan.replay_helicity_map().to_vec(),
        amplitude_destinations: plan.amplitude_destinations().to_vec(),
        resolved_helicities: plan.resolved_helicities().to_vec(),
        public_helicities: plan.public_helicities().to_vec(),
        source_state_assignments: plan.source_state_assignments().to_vec(),
        source_dispatch_variants: plan.source_dispatch_variants().to_vec(),
        source_embeddings: plan.source_embeddings().to_vec(),
        source_projections: plan.source_projections().to_vec(),
        resolved_source_selections: plan.resolved_source_selections().to_vec(),
        exact_factors,
        public_flow_ids,
        executors,
    }
}

fn validate_direct_template_shapes(
    template: &JsonMap<String, JsonValue>,
    context: &str,
) -> RusticolResult<()> {
    let parent_arity = json_u32(template, "parent_arity", &format!("{context} parent arity"))?;
    let parent_counts = json_array(
        template,
        "parent_component_counts",
        &format!("{context} parent component counts"),
    )?;
    if parent_counts.len() != parent_arity as usize {
        return Err(invalid(format!(
            "{context} parent component counts do not match parent arity"
        )));
    }
    for (index, value) in parent_counts.iter().enumerate() {
        let count = json_value_u32(value, &format!("{context} parent component {index}"))?;
        if count == 0 {
            return Err(invalid(format!(
                "{context} parent component {index} is empty"
            )));
        }
    }
    for field in [
        "coupling_slot_count",
        "momentum_operand_count",
        "parameter_slot_count",
    ] {
        json_u32(template, field, &format!("{context} {field}"))?;
    }
    if json_u32(
        template,
        "destination_component_count",
        &format!("{context} destination component count"),
    )? == 0
    {
        return Err(invalid(format!(
            "{context} destination component count must be positive"
        )));
    }
    let alignment = json_u32(template, "alignment_bytes", &format!("{context} alignment"))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid(format!(
            "{context} alignment must be a positive power of two"
        )));
    }
    validate_string_array(
        json_array(
            template,
            "semantic_template_ids",
            &format!("{context} semantic template IDs"),
        )?,
        &format!("{context} semantic template IDs"),
    )
}

fn direct_executor_role(value: &str, context: &str) -> RusticolResult<DirectExecutorRole> {
    match value {
        "source" => Ok(DirectExecutorRole::Source),
        "contribution" => Ok(DirectExecutorRole::Contribution),
        "finalization" => Ok(DirectExecutorRole::Finalization),
        "closure" => Ok(DirectExecutorRole::Closure),
        _ => Err(invalid(format!(
            "{context} has unsupported direct role {value:?}"
        ))),
    }
}

pub(super) fn canonical_json_bytes(value: &JsonValue, context: &str) -> RusticolResult<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|error| invalid(format!("could not canonicalize {context}: {error}")))
}

fn digest_json_without_field(
    value: &JsonValue,
    field: &str,
    context: &str,
) -> RusticolResult<SemanticDigest> {
    let mut semantic = value.clone();
    let object = semantic
        .as_object_mut()
        .ok_or_else(|| invalid(format!("{context} must be a JSON object")))?;
    if object.remove(field).is_none() {
        return Err(invalid(format!("{context} has no {field:?} field")));
    }
    let digest: [u8; 32] = Sha256::digest(canonical_json_bytes(&semantic, context)?).into();
    SemanticDigest::new(digest)
}

pub(super) fn json_object<'a>(
    value: &'a JsonValue,
    context: &str,
) -> RusticolResult<&'a JsonMap<String, JsonValue>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{context} must be a JSON object")))
}

pub(super) fn require_json_fields(
    object: &JsonMap<String, JsonValue>,
    expected: &[&str],
    context: &str,
) -> RusticolResult<()> {
    require_json_fields_with_optional(object, expected, &[], context)
}

fn require_json_fields_with_optional(
    object: &JsonMap<String, JsonValue>,
    expected: &[&str],
    optional: &[&str],
    context: &str,
) -> RusticolResult<()> {
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    let optional = optional.iter().copied().collect::<BTreeSet<_>>();
    let allowed = expected.union(&optional).copied().collect::<BTreeSet<_>>();
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if expected.is_subset(&actual) && actual.is_subset(&allowed) {
        return Ok(());
    }
    let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let unexpected = actual.difference(&allowed).copied().collect::<Vec<_>>();
    Err(invalid(format!(
        "{context} fields do not match direct-template-v1; missing={missing:?}, unexpected={unexpected:?}"
    )))
}

pub(super) fn json_field<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a JsonValue> {
    object
        .get(field)
        .ok_or_else(|| invalid(format!("{context} has no {field:?} field")))
}

pub(super) fn json_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a str> {
    json_field(object, field, context)?
        .as_str()
        .ok_or_else(|| invalid(format!("{context} must be a string")))
}

pub(super) fn json_nonempty_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a str> {
    let value = json_string(object, field, context)?;
    if value.is_empty() {
        return Err(invalid(format!("{context} must not be empty")));
    }
    Ok(value)
}

pub(super) fn require_json_string_value(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    expected: &str,
    context: &str,
) -> RusticolResult<()> {
    let actual = json_string(object, field, context)?;
    if actual != expected {
        return Err(invalid(format!(
            "{context} {field:?} is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

pub(super) fn json_bool(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<bool> {
    json_field(object, field, context)?
        .as_bool()
        .ok_or_else(|| invalid(format!("{context} must be a boolean")))
}

pub(super) fn json_u32(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<u32> {
    json_value_u32(json_field(object, field, context)?, context)
}

fn json_value_u32(value: &JsonValue, context: &str) -> RusticolResult<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{context} must be a nonnegative u32")))
}

pub(super) fn json_optional_u32(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<Option<u32>> {
    let value = json_field(object, field, context)?;
    if value.is_null() {
        return Ok(None);
    }
    json_value_u32(value, context).map(Some)
}

fn json_optional_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<Option<&'a str>> {
    let value = json_field(object, field, context)?;
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(Some)
        .ok_or_else(|| invalid(format!("{context} must be null or a nonempty string")))
}

pub(super) fn json_array<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a [JsonValue]> {
    json_field(object, field, context)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("{context} must be a JSON array")))
}

fn validate_string_array(values: &[JsonValue], context: &str) -> RusticolResult<()> {
    if values
        .iter()
        .all(|value| value.as_str().is_some_and(|value| !value.is_empty()))
    {
        return Ok(());
    }
    Err(invalid(format!(
        "{context} must contain only nonempty strings"
    )))
}

fn json_sha256(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<SemanticDigest> {
    semantic_digest_from_hex(json_string(object, field, context)?, context)
}

fn report_recurrence_build_progress(
    callback: Option<&Py<PyAny>>,
    progress: RecurrenceBuildProgress,
) -> RusticolResult<()> {
    let Some(callback) = callback else {
        return Ok(());
    };
    Python::attach(|py| -> PyResult<()> {
        let payload = PyDict::new(py);
        payload.set_item("step", progress.phase)?;
        payload.set_item("phase_index", progress.phase_index)?;
        payload.set_item("phase_total", progress.phase_total)?;
        payload.set_item("stage_total", progress.stage_total)?;
        payload.set_item(
            "candidate_parent_pair_count",
            progress.candidate_parent_pair_count,
        )?;
        payload.set_item("current_count", progress.current_count)?;
        payload.set_item("contribution_count", progress.contribution_count)?;
        payload.set_item(
            "dynamic_color_state_count",
            progress.dynamic_color_state_count,
        )?;
        payload.set_item(
            "color_target_prune_count",
            progress.color_target_prune_count,
        )?;
        if let Some(stage_index) = progress.stage_index {
            payload.set_item("stage_index", stage_index)?;
        }
        if let Some(subset_size) = progress.subset_size {
            payload.set_item("subset_size", subset_size)?;
        }
        if let Some(total) = progress.candidate_parent_pair_total {
            payload.set_item("candidate_parent_pair_total", total)?;
        }
        callback.bind(py).call1((payload,))?;
        Ok(())
    })
    .map_err(|error| invalid(format!("recurrence progress callback failed: {error}")))
}

fn recurrence_generation_profile_mapping<'py>(
    py: Python<'py>,
    native: &NativeDirectLoweringResult,
) -> PyResult<Bound<'py, PyDict>> {
    let profile = PyDict::new(py);
    profile.set_item("schema_version", 1)?;
    profile.set_item("scope", "generation-only")?;

    let telemetry = &native.generation_profile;
    let timings = PyDict::new(py);
    for (name, nanoseconds) in [
        (
            "transition-catalog",
            telemetry.transition_catalog_nanoseconds,
        ),
        (
            "structural-feasibility",
            telemetry.structural_feasibility_nanoseconds,
        ),
        (
            "color-target-index",
            telemetry.color_target_index_nanoseconds,
        ),
        ("structural-demand", telemetry.structural_demand_nanoseconds),
        ("support-indexing", telemetry.support_indexing_nanoseconds),
        (
            "candidate-processing",
            telemetry.candidate_processing_nanoseconds,
        ),
        (
            "closure-processing",
            telemetry.closure_processing_nanoseconds,
        ),
        (
            "canonical-emission",
            telemetry.canonical_emission_nanoseconds,
        ),
    ] {
        timings.set_item(name, nanoseconds as f64 / 1_000_000_000.0)?;
    }
    for (name, seconds) in [
        (
            "python-extraction",
            native.timings.python_extraction_seconds,
        ),
        (
            "catalog-authentication",
            native.timings.catalog_authentication_seconds,
        ),
        (
            "semantic-construction-total",
            native.timings.semantic_construction_seconds,
        ),
        ("direct-lowering", native.timings.direct_lowering_seconds),
        ("serialization", native.timings.serialization_seconds),
        ("native-total", native.timings.native_total_seconds),
    ] {
        timings.set_item(name, seconds)?;
    }
    profile.set_item("timings_seconds", timings)?;

    let counters = PyDict::new(py);
    for (name, value) in [
        ("support_bucket_count", telemetry.support_bucket_count),
        (
            "support_bucket_probe_count",
            telemetry.support_bucket_probe_count,
        ),
        (
            "support_bucket_cache_hit_count",
            telemetry.support_bucket_cache_hit_count,
        ),
        (
            "support_bucket_cache_miss_count",
            telemetry.support_bucket_cache_miss_count,
        ),
        (
            "candidate_parent_pair_theoretical_count",
            telemetry.candidate_parent_pair_theoretical_count,
        ),
        (
            "candidate_parent_pair_visited_count",
            telemetry.candidate_parent_pair_visited_count,
        ),
        (
            "structural_feasible_support_count",
            telemetry.structural_feasible_support_count,
        ),
        (
            "structural_decomposition_count",
            telemetry.structural_decomposition_count,
        ),
        (
            "structural_forward_transition_probe_count",
            telemetry.structural_forward_transition_probe_count,
        ),
        (
            "structural_demand_support_count",
            telemetry.structural_demand_support_count,
        ),
        (
            "structural_demand_state_count",
            telemetry.structural_demand_state_count,
        ),
        ("structural_reject_count", telemetry.structural_reject_count),
        (
            "transition_index_hit_count",
            telemetry.transition_index_hit_count,
        ),
        (
            "transition_index_miss_count",
            telemetry.transition_index_miss_count,
        ),
        (
            "transition_candidate_count",
            telemetry.transition_candidate_count,
        ),
        ("quantum_match_count", telemetry.quantum_match_count),
        ("coupling_match_count", telemetry.coupling_match_count),
        ("transition_accept_count", telemetry.transition_accept_count),
        ("color_shape_match_count", telemetry.color_shape_match_count),
        ("color_result_count", telemetry.color_result_count),
        (
            "color_target_accept_count",
            telemetry.color_target_accept_count,
        ),
        (
            "color_target_reject_count",
            telemetry.color_target_reject_count,
        ),
        (
            "color_acceptance_cache_hit_count",
            telemetry.color_acceptance_cache_hit_count,
        ),
        (
            "color_acceptance_cache_miss_count",
            telemetry.color_acceptance_cache_miss_count,
        ),
        (
            "color_fragment_bucket_count",
            telemetry.color_fragment_bucket_count,
        ),
        (
            "color_fragment_hash_lookup_count",
            telemetry.color_fragment_hash_lookup_count,
        ),
        (
            "color_posting_incidence_count",
            telemetry.color_posting_incidence_count,
        ),
        (
            "color_sparse_posting_bucket_count",
            telemetry.color_sparse_posting_bucket_count,
        ),
        (
            "color_dense_posting_bucket_count",
            telemetry.color_dense_posting_bucket_count,
        ),
        (
            "color_sparse_posting_bytes",
            telemetry.color_sparse_posting_bytes,
        ),
        (
            "color_dense_posting_bytes",
            telemetry.color_dense_posting_bytes,
        ),
        (
            "accepted_parent_key_clone_count",
            telemetry.accepted_parent_key_clone_count,
        ),
        (
            "current_key_lookup_count",
            telemetry.current_key_lookup_count,
        ),
        ("current_key_hit_count", telemetry.current_key_hit_count),
        ("current_insert_count", telemetry.current_insert_count),
        ("current_key_clone_count", telemetry.current_key_clone_count),
        (
            "indexed_hash_lookup_count",
            telemetry.indexed_hash_lookup_count,
        ),
        (
            "contribution_attempt_count",
            telemetry.contribution_attempt_count,
        ),
        (
            "contribution_insert_count",
            telemetry.contribution_insert_count,
        ),
        (
            "contribution_merge_count",
            telemetry.contribution_merge_count,
        ),
        (
            "closure_candidate_theoretical_count",
            telemetry.closure_candidate_theoretical_count,
        ),
        ("closure_candidate_count", telemetry.closure_candidate_count),
        (
            "closure_support_lookup_count",
            telemetry.closure_support_lookup_count,
        ),
        (
            "closure_state_match_count",
            telemetry.closure_state_match_count,
        ),
        (
            "closure_color_attempt_count",
            telemetry.closure_color_attempt_count,
        ),
        ("closure_group_count", telemetry.closure_group_count),
        (
            "closure_proof_contribution_count",
            telemetry.closure_proof_contribution_count,
        ),
        (
            "constructed_current_count",
            telemetry.constructed_current_count,
        ),
        (
            "constructed_contribution_count",
            telemetry.constructed_contribution_count,
        ),
        (
            "constructed_interaction_count",
            telemetry.constructed_interaction_count,
        ),
        (
            "constructed_dynamic_color_state_count",
            telemetry.constructed_dynamic_color_state_count,
        ),
        ("emitted_current_count", telemetry.emitted_current_count),
        (
            "emitted_contribution_count",
            telemetry.emitted_contribution_count,
        ),
        (
            "emitted_interaction_count",
            telemetry.emitted_interaction_count,
        ),
        (
            "emitted_finalization_count",
            telemetry.emitted_finalization_count,
        ),
        ("emitted_closure_count", telemetry.emitted_closure_count),
    ] {
        counters.set_item(name, value)?;
    }
    profile.set_item("operation_counters", counters)?;

    let serialized_bytes = PyDict::new(py);
    serialized_bytes.set_item("plan_payload", native.plan_payload_size)?;
    serialized_bytes.set_item("container", native.container_size)?;
    serialized_bytes.set_item("unpacked_container", native.unpacked_size_bytes)?;
    profile.set_item("serialized_bytes", serialized_bytes)?;
    Ok(profile)
}

fn direct_lowering_mapping(
    py: Python<'_>,
    native: NativeDirectLoweringResult,
) -> PyResult<Py<PyAny>> {
    let generation_profile = recurrence_generation_profile_mapping(py, &native)?;
    let result = PyDict::new(py);
    result.set_item("kind", DIRECT_LOWERING_RESULT_KIND)?;
    result.set_item("schema_version", DIRECT_LOWERING_RESULT_SCHEMA_VERSION)?;
    result.set_item("builder_input_abi", DIRECT_BUILDER_INPUT_ABI)?;
    result.set_item("builder_input_sha256", native.builder_input_digest)?;
    result.set_item(
        "template_input_abi",
        template::RECURRENCE_TEMPLATE_INPUT_ABI,
    )?;
    result.set_item("template_input_sha256", native.template_input_digest)?;
    result.set_item(
        "prepared_kernel_pack_digest",
        native.prepared_kernel_pack_digest,
    )?;
    result.set_item("direct_template_abi", RECURRENCE_DIRECT_TEMPLATE_ABI)?;
    result.set_item(
        "direct_template_catalog_digest",
        native.direct_template_catalog_digest,
    )?;
    result.set_item("recurrence_plan_abi", RECURRENCE_DIRECT_PLAN_ABI)?;
    result.set_item("runtime_layout_abi", RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI)?;
    result.set_item(
        "required_runtime_capabilities",
        PyList::new(
            py,
            [
                match native.strategy {
                    RecurrenceStrategy::ContractedColorUnion => {
                        RECURRENCE_CONTRACTED_COLOR_CAPABILITY
                    }
                    RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::AllFlowUnion => {
                        RECURRENCE_LC_COLOR_CAPABILITY
                    }
                },
                RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
            ],
        )?,
    )?;

    let container = PyDict::new(py);
    container.set_item("kind", RUNTIME_CONTAINER_KIND)?;
    container.set_item("schema_version", RUNTIME_CONTAINER_SCHEMA_VERSION)?;
    container.set_item("storage_abi", STORAGE_ABI)?;
    container.set_item("member_count", native.member_count)?;
    container.set_item("unpacked_size_bytes", native.unpacked_size_bytes)?;
    container.set_item("index_sha256", native.index_sha256)?;
    result.set_item("runtime_container", container)?;

    let inspection = PyDict::new(py);
    inspection.set_item("execution_mode", "recurrence")?;
    inspection.set_item("recurrence_plan_abi", RECURRENCE_DIRECT_PLAN_ABI)?;
    inspection.set_item("runtime_layout_abi", RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI)?;
    inspection.set_item("direct_template_abi", RECURRENCE_DIRECT_TEMPLATE_ABI)?;
    inspection.set_item("process_id", native.process_id)?;
    inspection.set_item("lc_flow_layout", native.strategy.as_str())?;
    inspection.set_item("prepared_kernel_count", native.prepared_kernel_count)?;
    inspection.set_item("parameter_count", native.parameter_value_count)?;
    inspection.set_item("sector_count", native.physical_sector_count)?;
    inspection.set_item("direct_executor_count", native.direct_executor_count)?;
    inspection.set_item("semantic_digest", native.semantic_digest)?;
    inspection.set_item("runtime_layout_digest", native.runtime_layout_digest)?;

    let schedule = PyDict::new(py);
    schedule.set_item("current_count", native.current_count)?;
    schedule.set_item("source_row_count", native.source_row_count)?;
    schedule.set_item("contribution_count", native.contribution_count)?;
    schedule.set_item("finalization_count", native.finalization_count)?;
    schedule.set_item("closure_term_count", native.closure_count)?;
    schedule.set_item(
        "amplitude_destination_count",
        native.amplitude_destination_count,
    )?;
    schedule.set_item("replay_target_count", native.replay_target_count)?;
    schedule.set_item("resolved_helicity_count", native.resolved_helicity_count)?;
    schedule.set_item("retained_helicity_count", native.retained_helicity_count)?;
    schedule.set_item("exact_factor_count", native.exact_factor_count)?;
    inspection.set_item("schedule", schedule)?;

    let selector_work = PyDict::new(py);
    selector_work.set_item("schema_version", 1)?;
    selector_work.set_item("binding", "runtime_layout_digest")?;
    let persisted = PyDict::new(py);
    persisted.set_item("current_count", native.current_count)?;
    persisted.set_item("semantic_component_count", native.semantic_component_count)?;
    persisted.set_item("source_row_count", native.source_row_count)?;
    persisted.set_item("contribution_count", native.contribution_count)?;
    persisted.set_item("finalization_count", native.finalization_count)?;
    persisted.set_item("closure_count", native.closure_count)?;
    persisted.set_item(
        "row_count",
        native.source_row_count
            + native.contribution_count
            + native.finalization_count
            + native.closure_count,
    )?;
    selector_work.set_item("persisted_union", persisted)?;
    let representatives = PyList::empty(py);
    for (summary, public_flow_count) in native.selector_work {
        let entry = PyDict::new(py);
        entry.set_item("representative_sector_id", summary.physical_sector_id)?;
        entry.set_item("public_flow_count", public_flow_count)?;
        entry.set_item("current_count", summary.current_count)?;
        entry.set_item("semantic_component_count", summary.semantic_component_count)?;
        entry.set_item("source_row_count", summary.source_row_count)?;
        entry.set_item("contribution_count", summary.contribution_count)?;
        entry.set_item("finalization_count", summary.finalization_count)?;
        entry.set_item("closure_count", summary.closure_count)?;
        entry.set_item(
            "amplitude_destination_count",
            summary.amplitude_destination_count,
        )?;
        entry.set_item("row_count", summary.row_count())?;
        representatives.append(entry)?;
    }
    selector_work.set_item("representatives", representatives)?;
    inspection.set_item("selector_work_certificate", selector_work)?;

    let construction = PyDict::new(py);
    construction.set_item("peak_current_count", native.construction.peak_current_count)?;
    construction.set_item(
        "peak_contribution_count",
        native.construction.peak_contribution_count,
    )?;
    construction.set_item(
        "peak_contribution_count_semantics",
        "resident-pending-contributions-v1",
    )?;
    construction.set_item(
        "peak_dynamic_color_state_count",
        native.construction.peak_dynamic_color_state_count,
    )?;
    construction.set_item(
        "color_target_prune_count",
        native.construction.color_target_prune_count,
    )?;
    construction.set_item(
        "candidate_parent_pair_count",
        native.construction.candidate_parent_pair_count(),
    )?;
    construction.set_item(
        "peak_to_final_current_ratio",
        native.construction.peak_current_count as f64 / native.current_count.max(1) as f64,
    )?;
    construction.set_item(
        "peak_to_final_contribution_ratio",
        native.construction.peak_contribution_count as f64
            / native.contribution_count.max(1) as f64,
    )?;
    inspection.set_item("construction", construction)?;

    let direct_arena = PyDict::new(py);
    direct_arena.set_item("semantic_component_count", native.semantic_component_count)?;
    direct_arena.set_item("current_arena_components", native.current_arena_components)?;
    direct_arena.set_item(
        "arena_component_reuse_count",
        native
            .semantic_component_count
            .saturating_sub(u64::from(native.current_arena_components)),
    )?;
    direct_arena.set_item("momentum_form_count", native.momentum_form_count)?;
    direct_arena.set_item("selector_domain_count", native.selector_domain_count)?;
    direct_arena.set_item(
        "cache_footprint_policy",
        if native.strategy == RecurrenceStrategy::TopologyReplay {
            "selector-active-max-v1"
        } else {
            "persisted-arena-v1"
        },
    )?;
    direct_arena.set_item("row_group_count", native.row_group_count)?;
    direct_arena.set_item("packed_input_bytes", 0)?;
    direct_arena.set_item("packed_output_bytes", 0)?;
    direct_arena.set_item("scatter_bytes", 0)?;
    inspection.set_item("direct_arena", direct_arena)?;
    if let Some(report) = native.relation_discovery.as_ref() {
        inspection.set_item(
            "relation_discovery",
            relation_discovery_mapping(py, report)?,
        )?;
    }

    let member = PyDict::new(py);
    member.set_item("path", RECURRENCE_DIRECT_SCHEDULE_MEMBER)?;
    member.set_item("size_bytes", native.plan_payload_size)?;
    member.set_item("sha256", native.plan_sha256)?;
    member.set_item("container_size_bytes", native.container_size)?;
    inspection.set_item("runtime_container_member", member)?;
    if let (Some(size_bytes), Some(sha256)) = (
        native.projection_certificate_payload_size,
        native.projection_certificate_sha256,
    ) {
        let certificate = PyDict::new(py);
        certificate.set_item(
            "path",
            rusticol_core::recurrence::RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER,
        )?;
        certificate.set_item("schema_version", 1)?;
        certificate.set_item("proof_kind", "exact-rectangular-sum-projection")?;
        certificate.set_item("publishable", true)?;
        certificate.set_item("size_bytes", size_bytes)?;
        certificate.set_item("sha256", sha256)?;
        inspection.set_item("color_projection_certificate", certificate)?;
    }
    if let Some(certificate) = native.closure_target_domain_certificate.as_ref() {
        let provenance = PyDict::new(py);
        provenance.set_item("proof_kind", CLOSURE_TARGET_DOMAIN_PROOF_KIND)?;
        provenance.set_item("semantic_digest", &certificate.semantic_digest)?;
        provenance.set_item("sector_count", certificate.physical_sector_count)?;
        provenance.set_item(
            "resolved_helicity_count",
            certificate.resolved_helicity_count,
        )?;
        provenance.set_item(
            "certified_structural_zero_sector_count",
            certificate
                .certified_structural_zero_physical_sector_ids
                .len(),
        )?;
        inspection.set_item("structural_zero_physical_sector_certificate", provenance)?;
    }

    let timings = PyDict::new(py);
    timings.set_item(
        "python_extraction",
        native.timings.python_extraction_seconds,
    )?;
    timings.set_item(
        "catalog_authentication",
        native.timings.catalog_authentication_seconds,
    )?;
    timings.set_item(
        "semantic_construction",
        native.timings.semantic_construction_seconds,
    )?;
    timings.set_item("direct_lowering", native.timings.direct_lowering_seconds)?;
    timings.set_item("serialization", native.timings.serialization_seconds)?;
    timings.set_item("native_total", native.timings.native_total_seconds)?;
    inspection.set_item("generation_timings_seconds", timings)?;

    result.set_item("inspection_summary", inspection)?;
    result.set_item("resolved_helicities", native.resolved_helicities)?;
    result.set_item("amplitude_destinations", native.amplitude_destinations)?;
    if let Some(certificate) = native.closure_target_domain_certificate.as_ref() {
        let metadata = PyDict::new(py);
        metadata.set_item("proof_kind", CLOSURE_TARGET_DOMAIN_PROOF_KIND)?;
        metadata.set_item(
            "sector_ids",
            &certificate.certified_structural_zero_physical_sector_ids,
        )?;
        metadata.set_item("semantic_digest", &certificate.semantic_digest)?;
        metadata.set_item("sector_count", certificate.physical_sector_count)?;
        metadata.set_item(
            "resolved_helicity_count",
            certificate.resolved_helicity_count,
        )?;
        result.set_item("structural_zero_physical_sector_certificate", metadata)?;
    } else {
        result.set_item("structural_zero_physical_sector_certificate", py.None())?;
    }
    if let Some(dispatch) = native.helicity_dispatch.as_ref() {
        let metadata = PyDict::new(py);
        metadata.set_item("abi", RECURRENCE_HELICITY_DISPATCH_ABI)?;
        metadata.set_item("size_bytes", dispatch.size_bytes)?;
        metadata.set_item("sha256", &dispatch.sha256)?;
        metadata.set_item(
            "base_runtime_layout_digest",
            &dispatch.base_runtime_layout_digest,
        )?;
        metadata.set_item("resolved_helicity_count", dispatch.resolved_helicity_count)?;
        result.set_item("helicity_dispatch", metadata)?;
    } else {
        result.set_item("helicity_dispatch", py.None())?;
    }
    result.set_item(
        "exact_sections",
        crate::recurrence_exact_sections_to_python(py, native.exact_sections)?,
    )?;
    result.set_item("generation_profile", generation_profile)?;
    Ok(result.into_any().unbind())
}

fn relation_discovery_mapping<'py>(
    py: Python<'py>,
    report: &RecurrenceRelationDiscoveryReport,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("schema_version", 1)?;
    payload.set_item("requested_mode", report.requested_mode.as_str())?;
    payload.set_item("state", report.state)?;

    let scope = PyDict::new(py);
    scope.set_item("execution_mode", "recurrence")?;
    scope.set_item("color_accuracy", &report.color_accuracy)?;
    scope.set_item("representation", "recurrence-direct-plan-v2")?;
    scope.set_item("lc_flow_layout", report.strategy.as_str())?;
    payload.set_item("scope", scope)?;

    let probe = PyDict::new(py);
    probe.set_item("status", "completed")?;
    probe.set_item("precision_digits", report.precision_digits)?;
    probe.set_item("probe_count", report.probe_count)?;
    probe.set_item("verification_probe_count", report.verification_probe_count)?;
    probe.set_item(
        "effective_projection_count",
        report.effective_projection_count,
    )?;
    probe.set_item("seed", report.seed)?;
    probe.set_item("deterministic", true)?;
    probe.set_item(
        "candidate_only",
        report.certificate_algorithm
            != rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM,
    )?;
    probe.set_item("tested_hypothesis_count", report.tested_hypothesis_count)?;
    probe.set_item(
        "verification_rejected_count",
        report.verification_rejected_count,
    )?;
    probe.set_item(
        "runtime_parameter_schema_sha256",
        report
            .authenticated_runtime_parameter_schema_sha256
            .as_deref(),
    )?;
    probe.set_item(
        "candidate_observation_batch_sha256",
        report
            .authenticated_candidate_observation_batch_sha256
            .as_deref(),
    )?;
    probe.set_item(
        "verification_observation_batch_sha256",
        report
            .authenticated_verification_observation_batch_sha256
            .as_deref(),
    )?;
    probe.set_item(
        "decision_sha256",
        report.authenticated_decision_sha256.as_deref(),
    )?;
    probe.set_item(
        "rejection_decision_sha256",
        &report.rejected_decision_sha256,
    )?;
    payload.set_item("probe", probe)?;

    let replay = PyDict::new(py);
    replay.set_item("algorithm", &report.certificate_algorithm)?;
    replay.set_item(
        "status",
        if report.certificates.is_empty() {
            "no-certified-relations"
        } else {
            "verified"
        },
    )?;
    replay.set_item(
        "certificate_set_sha256",
        relation_certificate_set_sha256(report),
    )?;
    payload.set_item("certificate_replay", replay)?;
    payload.set_item(
        "numerical_candidate_count",
        report.numerical_candidate_count,
    )?;
    payload.set_item(
        "uncertified_candidate_count",
        report.uncertified_candidate_count,
    )?;
    payload.set_item(
        "rejected_hypothesis_count",
        report.rejected_hypothesis_count,
    )?;
    payload.set_item(
        "exact_certified_relation_count",
        report.exact_certified_relation_count,
    )?;
    payload.set_item("applied_relation_count", report.applied_relation_count)?;
    payload.set_item(
        "interaction_evaluation_count_before",
        report.interaction_evaluation_count_before,
    )?;
    payload.set_item(
        "interaction_evaluation_count_after",
        report.interaction_evaluation_count_after,
    )?;
    payload.set_item(
        "interaction_evaluation_savings",
        report
            .interaction_evaluation_count_before
            .saturating_sub(report.interaction_evaluation_count_after),
    )?;
    payload.set_item("current_count_before", report.current_count_before)?;
    payload.set_item("current_count_after", report.current_count_after)?;
    payload.set_item(
        "contribution_count_before",
        report.contribution_count_before,
    )?;
    payload.set_item("contribution_count_after", report.contribution_count_after)?;
    payload.set_item("scale_copy_row_count", report.scale_copy_row_count)?;

    let certificates = PyList::empty(py);
    // Recurrence schedules can contain many exact aliases. Keep generation
    // metadata bounded while the aggregate digest authenticates the complete
    // replayed certificate set and the plan binds every applied scale row.
    const CERTIFICATE_SAMPLE_LIMIT: usize = 16;
    for certificate in report.certificates.iter().take(CERTIFICATE_SAMPLE_LIMIT) {
        let entry = PyDict::new(py);
        entry.set_item("algorithm", &report.certificate_algorithm)?;
        entry.set_item("current_id", certificate.current_id)?;
        entry.set_item("representative_id", certificate.representative_id)?;
        let factor = PyDict::new(py);
        factor.set_item(
            "real_numerator",
            certificate.factor.real().numerator().to_string(),
        )?;
        factor.set_item(
            "real_denominator",
            certificate.factor.real().denominator().to_string(),
        )?;
        factor.set_item(
            "imag_numerator",
            certificate.factor.imag().numerator().to_string(),
        )?;
        factor.set_item(
            "imag_denominator",
            certificate.factor.imag().denominator().to_string(),
        )?;
        entry.set_item("factor_exact_rational", factor)?;
        entry.set_item(
            "current_term_vector_sha256",
            &certificate.current_expression_sha256,
        )?;
        entry.set_item(
            "representative_term_vector_sha256",
            &certificate.representative_expression_sha256,
        )?;
        entry.set_item("proof_sha256", &certificate.proof_sha256)?;
        certificates.append(entry)?;
    }
    payload.set_item("certificates", certificates)?;
    payload.set_item("certificate_count", report.certificates.len())?;
    payload.set_item(
        "certificates_truncated",
        report.certificates.len() > CERTIFICATE_SAMPLE_LIMIT,
    )?;

    let rejected = PyList::empty(py);
    for reason in &report.rejected_candidates {
        let entry = PyDict::new(py);
        entry.set_item("reason", reason)?;
        rejected.append(entry)?;
    }
    payload.set_item("rejected_candidates", rejected)?;
    let rejected_diagnostics = PyDict::new(py);
    rejected_diagnostics.set_item(
        "total_rejected_hypothesis_count",
        report.rejected_hypothesis_count,
    )?;
    rejected_diagnostics.set_item("retained_count", report.rejected_candidates.len())?;
    rejected_diagnostics.set_item(
        "truncated",
        report.rejected_hypothesis_count > report.rejected_candidates.len(),
    )?;
    let authenticated_raw_decisions = report.authenticated_decision_sha256.is_some();
    if authenticated_raw_decisions && !report.rejected_candidates.is_empty() {
        return Err(PyValueError::new_err(
            "authenticated raw relation diagnostics must use zero retained rows",
        ));
    }
    rejected_diagnostics.set_item(
        "truncation_policy",
        if authenticated_raw_decisions {
            "none-authenticated-full-rejection-digest-v1"
        } else {
            "first-16-in-canonical-discovery-order-v1"
        },
    )?;
    let retained_rows = report
        .rejected_candidates
        .iter()
        .map(|reason| serde_json::json!({"reason": reason}))
        .collect::<Vec<_>>();
    let retained_payload = serde_json::json!({
        "abi": "pyamplicol-rejected-relation-diagnostics-v1",
        "rows": retained_rows,
    });
    let retained_bytes =
        canonical_json_bytes(&retained_payload, "native recurrence rejected diagnostics")
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    rejected_diagnostics.set_item(
        "retained_sha256",
        hex_digest(Sha256::digest(retained_bytes)),
    )?;
    rejected_diagnostics.set_item(
        "full_census_sha256",
        hex_digest(Sha256::digest(
            canonical_json_bytes(
                &serde_json::json!({
                    "abi": "pyamplicol-relation-discovery-full-census-v1",
                    "tested_hypothesis_count": report.tested_hypothesis_count,
                    "numerical_candidate_count": report.numerical_candidate_count,
                    "verification_rejected_count": report.verification_rejected_count,
                    "uncertified_candidate_count":
                        report.uncertified_candidate_count,
                    "certified_relation_count": report.exact_certified_relation_count,
                    "rejected_hypothesis_count":
                        report.rejected_hypothesis_count,
                    "applied_relation_count": report.applied_relation_count,
                    "runtime_parameter_schema_sha256":
                        report.authenticated_runtime_parameter_schema_sha256,
                    "candidate_observation_batch_sha256":
                        report.authenticated_candidate_observation_batch_sha256,
                    "verification_observation_batch_sha256":
                        report.authenticated_verification_observation_batch_sha256,
                    "decision_sha256": report.authenticated_decision_sha256,
                    "rejection_decision_sha256":
                        report.rejected_decision_sha256,
                }),
                "native recurrence relation census",
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )),
    )?;
    rejected_diagnostics.set_item("full_rejection_sha256", &report.rejected_decision_sha256)?;
    payload.set_item("rejected_candidate_diagnostics", rejected_diagnostics)?;
    payload.set_item(
        "follow_up_boundary",
        "Direct-plan-v2 retains dense semantic current descriptors and exact closure bindings; exact certified reuse removes interaction rows and emits one scale-copy row per relation. Direct-Arena lifetime allocation continues to own physical component recycling.",
    )?;
    Ok(payload)
}

fn relation_certificate_set_sha256(report: &RecurrenceRelationDiscoveryReport) -> String {
    if let Some(digest) = report.authenticated_certificate_set_sha256.as_ref() {
        return digest.clone();
    }
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-relation-certificate-set-v1");
    hash.update((report.certificates.len() as u64).to_le_bytes());
    for certificate in &report.certificates {
        hash.update(certificate.current_id.to_le_bytes());
        hash.update(certificate.representative_id.to_le_bytes());
        for value in [
            certificate.factor.real().numerator(),
            certificate.factor.real().denominator(),
            certificate.factor.imag().numerator(),
            certificate.factor.imag().denominator(),
        ] {
            let text = value.to_string();
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        hash.update(certificate.current_expression_sha256.as_bytes());
        hash.update(certificate.representative_expression_sha256.as_bytes());
        hash.update(certificate.proof_sha256.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn parse_input(input: &Bound<'_, PyAny>) -> PyResult<OwnedInput> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "recurrence builder input v2 requires a little-endian target",
        ));
    }
    let abi = required_string(input, "abi")?;
    if abi != RECURRENCE_BUILDER_INPUT_ABI {
        return Err(PyValueError::new_err(format!(
            "unsupported recurrence builder input ABI {abi:?}"
        )));
    }
    let declared_digest = required_string(input, "digest")?;
    validate_sha256_text(&declared_digest, "recurrence builder input digest")?;

    let table_objects = iterable_attribute(input, "tables", "recurrence builder tables")?;
    if table_objects.len() != TABLE_SPECS.len() {
        return Err(PyValueError::new_err(format!(
            "recurrence builder table inventory has {} tables, expected {}",
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
                "recurrence builder table inventory mismatch: found {table_name:?}, expected {:?}",
                spec.name
            )));
        }
        let row_count = table_object
            .getattr("row_count")?
            .extract::<u64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "recurrence builder table {table_name:?} row_count must be u64"
                ))
            })?;
        let column_objects = iterable_attribute(
            &table_object,
            "columns",
            &format!("recurrence builder table {table_name:?} columns"),
        )?;
        if column_objects.len() != spec.columns.len() {
            return Err(PyValueError::new_err(format!(
                "recurrence builder table {table_name:?} has {} columns, expected {}",
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
                    "recurrence builder table {table_name:?} column mismatch: found {column_name:?}, expected {:?}",
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

    let pairing_table_objects = if input.hasattr("fermion_pairing_tables")? {
        iterable_attribute(
            input,
            "fermion_pairing_tables",
            "recurrence fermion-pairing tables",
        )?
    } else {
        Vec::new()
    };
    if !pairing_table_objects.is_empty()
        && pairing_table_objects.len() != FERMION_PAIRING_TABLE_SPECS.len()
    {
        return Err(PyValueError::new_err(format!(
            "recurrence fermion-pairing table inventory has {} tables, expected {}",
            pairing_table_objects.len(),
            FERMION_PAIRING_TABLE_SPECS.len()
        )));
    }
    let mut fermion_pairing_tables = Vec::with_capacity(pairing_table_objects.len());
    let mut fermion_pairing_table_by_name = BTreeMap::new();
    for (table_object, spec) in pairing_table_objects
        .into_iter()
        .zip(FERMION_PAIRING_TABLE_SPECS)
    {
        let table_name = required_nonempty_string(&table_object, "name", "table name")?;
        if table_name != spec.name {
            return Err(PyValueError::new_err(format!(
                "recurrence fermion-pairing table inventory mismatch: found {table_name:?}, expected {:?}",
                spec.name
            )));
        }
        let row_count = table_object
            .getattr("row_count")?
            .extract::<u64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "recurrence fermion-pairing table {table_name:?} row_count must be u64"
                ))
            })?;
        let column_objects = iterable_attribute(
            &table_object,
            "columns",
            &format!("recurrence fermion-pairing table {table_name:?} columns"),
        )?;
        if column_objects.len() != spec.columns.len() {
            return Err(PyValueError::new_err(format!(
                "recurrence fermion-pairing table {table_name:?} has {} columns, expected {}",
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
                    "recurrence fermion-pairing table {table_name:?} column mismatch: found {column_name:?}, expected {:?}",
                    column_spec.name
                )));
            }
            let context = format!("fermion_pairing.{table_name}.{column_name}");
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
        fermion_pairing_table_by_name.insert(table_name.clone(), fermion_pairing_tables.len());
        fermion_pairing_tables.push(OwnedTable {
            name: table_name,
            row_count,
            columns,
            column_by_name,
        });
    }
    let declared_fermion_pairing_digest = if fermion_pairing_tables.is_empty() {
        None
    } else {
        if !input.hasattr("fermion_pairing_digest")? {
            return Err(PyValueError::new_err(
                "recurrence fermion-pairing tables require a canonical digest",
            ));
        }
        let value = input.getattr("fermion_pairing_digest")?;
        if value.is_none() {
            return Err(PyValueError::new_err(
                "recurrence fermion-pairing tables require a canonical digest",
            ));
        }
        let digest = value.extract::<String>().map_err(|_| {
            PyTypeError::new_err("recurrence fermion-pairing digest must be a string")
        })?;
        validate_sha256_text(&digest, "recurrence fermion-pairing digest")?;
        Some(digest)
    };

    Ok(OwnedInput {
        abi,
        declared_digest,
        tables,
        table_by_name,
        declared_fermion_pairing_digest,
        fermion_pairing_tables,
        fermion_pairing_table_by_name,
    })
}

fn parse_prepared_template_input(input: &Bound<'_, PyAny>) -> PyResult<PreparedTemplateInput> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "recurrence template input v1 requires a little-endian target",
        ));
    }
    let abi = required_string(input, "abi")?;
    if abi != template::RECURRENCE_TEMPLATE_INPUT_ABI {
        return Err(PyValueError::new_err(format!(
            "unsupported recurrence template input ABI {abi:?}"
        )));
    }
    let declared_digest = required_string(input, "digest")?;
    validate_sha256_text(&declared_digest, "recurrence template input digest")?;
    let canonical_digest_property = required_string(input, "canonical_digest")?;
    validate_sha256_text(
        &canonical_digest_property,
        "recurrence template canonical digest",
    )?;
    let catalog_digest = semantic_digest_python_attribute(input, "catalog_digest")?;
    let compiled_model_digest = semantic_digest_python_attribute(input, "compiled_model_digest")?;
    let prepared_kernel_pack_digest =
        semantic_digest_python_attribute(input, "prepared_kernel_pack_digest")?;

    let table_objects = iterable_attribute(input, "tables", "recurrence template tables")?;
    if table_objects.len() != TEMPLATE_TABLE_INVENTORY.len() {
        return Err(PyValueError::new_err(format!(
            "recurrence template table inventory has {} tables, expected {}",
            table_objects.len(),
            TEMPLATE_TABLE_INVENTORY.len()
        )));
    }

    let mut tables = Vec::with_capacity(TEMPLATE_TABLE_INVENTORY.len());
    let mut table_by_name = BTreeMap::new();
    for (table_object, (expected_name, expected_column_count)) in
        table_objects.into_iter().zip(TEMPLATE_TABLE_INVENTORY)
    {
        let table_name = required_nonempty_string(&table_object, "name", "table name")?;
        if table_name != *expected_name {
            return Err(PyValueError::new_err(format!(
                "recurrence template table inventory mismatch: found {table_name:?}, expected {expected_name:?}"
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
        if column_objects.len() != *expected_column_count {
            return Err(PyValueError::new_err(format!(
                "recurrence template table {table_name:?} has {} columns, expected {expected_column_count}",
                column_objects.len()
            )));
        }

        let mut columns = Vec::with_capacity(*expected_column_count);
        let mut column_by_name = BTreeMap::new();
        for column_object in column_objects {
            let column_name = required_nonempty_string(&column_object, "name", "column name")?;
            if column_by_name.contains_key(&column_name) {
                return Err(PyValueError::new_err(format!(
                    "recurrence template table {table_name:?} repeats column {column_name:?}"
                )));
            }
            let context = format!("{table_name}.{column_name}");
            let values_object = column_object.getattr("values")?;
            let dtype = values_object
                .getattr("dtype")?
                .getattr("str")?
                .extract::<String>()?;
            let kind = PrimitiveKind::from_dtype(&dtype).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{context} has unsupported recurrence-template dtype {dtype:?}"
                ))
            })?;
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
            let (values, shape) = extract_owned_values(&values_object, kind, &context)?;
            if shape.first().copied() != Some(row_count) {
                return Err(PyValueError::new_err(format!(
                    "{context} first dimension does not match row_count {row_count}"
                )));
            }
            column_by_name.insert(column_name.clone(), columns.len());
            columns.push(OwnedColumn {
                name: column_name,
                dtype: kind.dtype(),
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

    Ok(PreparedTemplateInput {
        input: OwnedInput {
            abi,
            declared_digest,
            tables,
            table_by_name,
            declared_fermion_pairing_digest: None,
            fermion_pairing_tables: Vec::new(),
            fermion_pairing_table_by_name: BTreeMap::new(),
        },
        canonical_digest_property,
        catalog_digest,
        compiled_model_digest,
        prepared_kernel_pack_digest,
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

fn decode_process_input(
    input: &OwnedInput,
) -> RusticolResult<process::OwnedRecurrenceProcessInput> {
    let fermion_pairing = decode_fermion_pairing_input(input)?;
    let bitset_ranges = decode_process_rows(input, "bitset_ranges", |row| {
        Ok(process::ProcessBitsetRangeRow {
            id: input.u32("bitset_ranges", "id")?[row],
            range: CheckedTableRange {
                start: input.u64("bitset_ranges", "start")?[row],
                count: input.u64("bitset_ranges", "count")?[row],
            },
            bit_count: input.u64("bitset_ranges", "bit_count")?[row],
        })
    })?;
    let coupling_limits = decode_process_rows(input, "coupling_limits", |row| {
        Ok(process::ProcessCouplingLimitRow {
            name_string_id: input.u32("coupling_limits", "name_string_id")?[row],
            minimum: input.u32("coupling_limits", "minimum")?[row],
            maximum: input.u32("coupling_limits", "maximum")?[row],
        })
    })?;
    let digest_values = input.u8("digest_catalog", "value")?;
    let digest_catalog = decode_process_rows(input, "digest_catalog", |row| {
        let start = row
            .checked_mul(32)
            .ok_or_else(|| invalid("process digest catalog offset exceeds usize"))?;
        let value = digest_values
            .get(start..start + 32)
            .ok_or_else(|| invalid("process digest catalog row is truncated"))?
            .try_into()
            .map_err(|_| invalid("process digest catalog row must contain 32 bytes"))?;
        Ok(process::ProcessDigestCatalogRow {
            id: input.u32("digest_catalog", "id")?[row],
            value,
        })
    })?;
    let exact_factors = decode_process_rows(input, "exact_factors", |row| {
        Ok(process::ProcessExactFactorRow {
            id: input.u32("exact_factors", "id")?[row],
            real_numerator_string_id: input.u32("exact_factors", "real_numerator_string_id")?[row],
            real_denominator_string_id: input.u32("exact_factors", "real_denominator_string_id")?
                [row],
            imag_numerator_string_id: input.u32("exact_factors", "imag_numerator_string_id")?[row],
            imag_denominator_string_id: input.u32("exact_factors", "imag_denominator_string_id")?
                [row],
        })
    })?;
    let external_legs = decode_process_rows(input, "external_legs", |row| {
        Ok(process::ProcessExternalLegRow {
            source_slot: input.u32("external_legs", "source_slot")?[row],
            public_label: input.u32("external_legs", "public_label")?[row],
            physical_pdg: input.i32("external_legs", "physical_pdg")?[row],
            outgoing_pdg: input.i32("external_legs", "outgoing_pdg")?[row],
            is_initial: input.u8("external_legs", "is_initial")?[row],
            is_fermionic: input.u8("external_legs", "is_fermionic")?[row],
            source_state_range: CheckedTableRange {
                start: input.u64("external_legs", "source_state_start")?[row],
                count: input.u64("external_legs", "source_state_count")?[row],
            },
            momentum_mask_id: input.u32("external_legs", "momentum_mask_id")?[row],
            support_mask_id: input.u32("external_legs", "support_mask_id")?[row],
        })
    })?;
    let header = decode_process_rows(input, "header", |row| {
        Ok(process::ProcessHeaderRow {
            schema_version: input.u32("header", "schema_version")?[row],
            abi_string_id: input.u32("header", "abi_string_id")?[row],
            process_id_string_id: input.u32("header", "process_id_string_id")?[row],
            layout: input.u8("header", "layout")?[row],
            selected_flow_mode: input.u8("header", "selected_flow_mode")?[row],
            selected_source_mode: input.u8("header", "selected_source_mode")?[row],
            external_leg_count: input.u32("header", "external_leg_count")?[row],
            physical_sector_count: input.u32("header", "physical_sector_count")?[row],
            public_flow_count: input.u32("header", "public_flow_count")?[row],
            replay_partition_count: input.u32("header", "replay_partition_count")?[row],
            coupling_limit_count: input.u32("header", "coupling_limit_count")?[row],
            parameter_projection_count: input.u32("header", "parameter_projection_count")?[row],
            process_support_mask_id: input.u32("header", "process_support_mask_id")?[row],
        })
    })?;
    let header_digests = decode_process_rows(input, "header_digests", |row| {
        Ok(process::ProcessHeaderDigestRow {
            role_string_id: input.u32("header_digests", "role_string_id")?[row],
            digest_id: input.u32("header_digests", "digest_id")?[row],
        })
    })?;
    let lc_open_strings = decode_process_rows(input, "lc_open_strings", |row| {
        Ok(process::ProcessLCOpenStringRow {
            sector_id: input.u32("lc_open_strings", "sector_id")?[row],
            ordinal: input.u32("lc_open_strings", "ordinal")?[row],
            fundamental_source_slot: input.u32("lc_open_strings", "fundamental_source_slot")?[row],
            antifundamental_source_slot: input
                .u32("lc_open_strings", "antifundamental_source_slot")?[row],
            adjoint_sequence_id: input.u32("lc_open_strings", "adjoint_sequence_id")?[row],
            singlet_sequence_id: input.u32("lc_open_strings", "singlet_sequence_id")?[row],
        })
    })?;
    let normalization = decode_process_rows(input, "normalization", |row| {
        Ok(process::ProcessNormalizationRow {
            factor_id: input.u32("normalization", "factor_id")?[row],
            convention_string_id: input.u32("normalization", "convention_string_id")?[row],
            semantic_digest_id: input.u32("normalization", "semantic_digest_id")?[row],
        })
    })?;
    let parameter_projection = decode_process_rows(input, "parameter_projection", |row| {
        Ok(process::ProcessParameterProjectionRow {
            runtime_slot: input.u32("parameter_projection", "runtime_slot")?[row],
            runtime_name_string_id: input.u32("parameter_projection", "runtime_name_string_id")?
                [row],
            parameter_template_id: input.u32("parameter_projection", "parameter_template_id")?[row],
            prepared_parameter_id: input.u32("parameter_projection", "prepared_parameter_id")?[row],
            component: input.u32("parameter_projection", "component")?[row],
        })
    })?;
    let physical_lc_sectors = decode_process_rows(input, "physical_lc_sectors", |row| {
        Ok(process::ProcessPhysicalLCSectorRow {
            sector_id: input.u32("physical_lc_sectors", "sector_id")?[row],
            public_id_string_id: input.u32("physical_lc_sectors", "public_id_string_id")?[row],
            kind: input.u8("physical_lc_sectors", "kind")?[row],
            closure_source_slot: input.u32("physical_lc_sectors", "closure_source_slot")?[row],
            closure_proof_algorithm_string_id: input
                .u32("physical_lc_sectors", "closure_proof_algorithm_string_id")?[row],
            closure_proof_digest_id: input.u32("physical_lc_sectors", "closure_proof_digest_id")?
                [row],
            open_string_range: CheckedTableRange {
                start: input.u64("physical_lc_sectors", "open_string_start")?[row],
                count: input.u64("physical_lc_sectors", "open_string_count")?[row],
            },
            trace_sequence_id: input.u32("physical_lc_sectors", "trace_sequence_id")?[row],
            singlet_sequence_id: input.u32("physical_lc_sectors", "singlet_sequence_id")?[row],
            word_sequence_id: input.u32("physical_lc_sectors", "word_sequence_id")?[row],
            support_mask_id: input.u32("physical_lc_sectors", "support_mask_id")?[row],
        })
    })?;
    let public_lc_flows = decode_process_rows(input, "public_lc_flows", |row| {
        Ok(process::ProcessPublicLCFlowRow {
            flow_id: input.u32("public_lc_flows", "flow_id")?[row],
            public_id_string_id: input.u32("public_lc_flows", "public_id_string_id")?[row],
            construction_sector_id: input.u32("public_lc_flows", "construction_sector_id")?[row],
            word_sequence_id: input.u32("public_lc_flows", "word_sequence_id")?[row],
            source_slot_permutation_sequence_id: input
                .u32("public_lc_flows", "source_slot_permutation_sequence_id")?[row],
            reduction_weight_factor_id: input
                .u32("public_lc_flows", "reduction_weight_factor_id")?[row],
        })
    })?;
    let replay_partitions = decode_process_rows(input, "replay_partitions", |row| {
        Ok(process::ProcessReplayPartitionRow {
            partition_id: input.u32("replay_partitions", "partition_id")?[row],
            representative_sector_id: input.u32("replay_partitions", "representative_sector_id")?
                [row],
            materialized_sector_id: input.u32("replay_partitions", "materialized_sector_id")?[row],
            target_range: CheckedTableRange {
                start: input.u64("replay_partitions", "target_start")?[row],
                count: input.u64("replay_partitions", "target_count")?[row],
            },
            proof_algorithm_string_id: input
                .u32("replay_partitions", "proof_algorithm_string_id")?[row],
            proof_digest_id: input.u32("replay_partitions", "proof_digest_id")?[row],
        })
    })?;
    let replay_targets = decode_process_rows(input, "replay_targets", |row| {
        Ok(process::ProcessReplayTargetRow {
            partition_id: input.u32("replay_targets", "partition_id")?[row],
            sector_id: input.u32("replay_targets", "sector_id")?[row],
            external_permutation_sequence_id: input
                .u32("replay_targets", "external_permutation_sequence_id")?[row],
            source_slot_permutation_sequence_id: input
                .u32("replay_targets", "source_slot_permutation_sequence_id")?[row],
            amplitude_phase_factor_id: input.u32("replay_targets", "amplitude_phase_factor_id")?
                [row],
            fermion_sign: input.i32("replay_targets", "fermion_sign")?[row],
        })
    })?;
    let selected_public_flow_coverage =
        decode_process_rows(input, "selected_public_flow_coverage", |row| {
            Ok(process::ProcessSelectedPublicFlowRow {
                flow_id: input.u32("selected_public_flow_coverage", "flow_id")?[row],
            })
        })?;
    let selected_source_coverage = decode_process_rows(input, "selected_source_coverage", |row| {
        Ok(process::ProcessSelectedSourceStateRow {
            source_slot: input.u32("selected_source_coverage", "source_slot")?[row],
            source_state_index: input.u32("selected_source_coverage", "source_state_index")?[row],
        })
    })?;
    let semantic_template_references =
        decode_process_rows(input, "semantic_template_references", |row| {
            Ok(process::ProcessSemanticTemplateReferenceRow {
                kind_string_id: input.u32("semantic_template_references", "kind_string_id")?[row],
                template_id: input.u32("semantic_template_references", "template_id")?[row],
                semantic_digest_id: input
                    .u32("semantic_template_references", "semantic_digest_id")?[row],
                prepared_kernel_id: input
                    .u32("semantic_template_references", "prepared_kernel_id")?[row],
            })
        })?;
    let source_states = decode_process_rows(input, "source_states", |row| {
        Ok(process::ProcessSourceStateRow {
            source_slot: input.u32("source_states", "source_slot")?[row],
            state_index: input.u32("source_states", "state_index")?[row],
            public_helicity: input.i32("source_states", "public_helicity")?[row],
            chirality: input.i32("source_states", "chirality")?[row],
            spin_state: input.i32("source_states", "spin_state")?[row],
            current_state_template_id: input.u32("source_states", "current_state_template_id")?
                [row],
            source_template_id: input.u32("source_states", "source_template_id")?[row],
            momentum_sign: input.i32("source_states", "momentum_sign")?[row],
            crossing_phase_factor_id: input.u32("source_states", "crossing_phase_factor_id")?[row],
        })
    })?;

    Ok(process::OwnedRecurrenceProcessInput {
        input_abi: input.abi.clone(),
        declared_input_digest: semantic_digest_from_hex(
            &input.declared_digest,
            "recurrence process input digest",
        )?,
        fermion_pairing,
        bitset_ranges,
        bitset_words: input.u64("bitset_words", "value")?.to_vec(),
        coupling_limits,
        digest_catalog,
        exact_factors,
        external_legs,
        header,
        header_digests,
        lc_open_strings,
        normalization,
        parameter_projection,
        physical_lc_sectors,
        public_lc_flows,
        replay_partitions,
        replay_targets,
        selected_public_flow_coverage,
        selected_source_coverage,
        semantic_template_references,
        source_states,
        string_ranges: plain_ranges(input, "string_ranges")?,
        string_bytes: input.u8("string_bytes", "value")?.to_vec(),
        u32_sequence_ranges: plain_ranges(input, "u32_sequence_ranges")?,
        u32_sequence_values: input.u32("u32_sequence_values", "value")?.to_vec(),
    })
}

fn decode_fermion_pairing_input(
    input: &OwnedInput,
) -> RusticolResult<Option<process::OwnedFermionPairingInput>> {
    let Some(declared_digest) = input.declared_fermion_pairing_digest.as_deref() else {
        if !input.fermion_pairing_tables.is_empty() {
            return Err(invalid(
                "recurrence fermion-pairing tables have no declared digest",
            ));
        }
        return Ok(None);
    };
    if input.fermion_pairing_tables.is_empty() {
        return Err(invalid(
            "recurrence fermion-pairing digest has no fixed-width tables",
        ));
    }

    let header = decode_pairing_rows(input, "header", |row| {
        Ok(process::FermionPairingHeaderRow {
            schema_version: input.pairing_u32("header", "schema_version")?[row],
            abi_string_id: input.pairing_u32("header", "abi_string_id")?[row],
            process_key_string_id: input.pairing_u32("header", "process_key_string_id")?[row],
            proof_algorithm_string_id: input.pairing_u32("header", "proof_algorithm_string_id")?
                [row],
            source_count: input.pairing_u32("header", "source_count")?[row],
            endpoint_count: input.pairing_u32("header", "endpoint_count")?[row],
            pairing_class_count: input.pairing_u32("header", "pairing_class_count")?[row],
            rule_count: input.pairing_u32("header", "rule_count")?[row],
            endpoint_state_template_count: input
                .pairing_u64("header", "endpoint_state_template_count")?[row],
            endpoint_anti_state_template_count: input
                .pairing_u64("header", "endpoint_anti_state_template_count")?[row],
            endpoint_basis_count: input.pairing_u64("header", "endpoint_basis_count")?[row],
            endpoint_color_representation_count: input
                .pairing_u64("header", "endpoint_color_representation_count")?[row],
            class_fundamental_slot_count: input
                .pairing_u64("header", "class_fundamental_slot_count")?[row],
            class_antifundamental_slot_count: input
                .pairing_u64("header", "class_antifundamental_slot_count")?[row],
            class_reference_pairing_count: input
                .pairing_u64("header", "class_reference_pairing_count")?[row],
            rule_class_pairing_index_count: input
                .pairing_u64("header", "rule_class_pairing_index_count")?[row],
            rule_endpoint_pairing_count: input
                .pairing_u64("header", "rule_endpoint_pairing_count")?[row],
            rule_source_permutation_count: input
                .pairing_u64("header", "rule_source_permutation_count")?[row],
            rule_lineage_count: input.pairing_u64("header", "rule_lineage_count")?[row],
            exact_integer_count: input.pairing_u32("header", "exact_integer_count")?[row],
            exact_integer_limb_count: input.pairing_u64("header", "exact_integer_limb_count")?[row],
            string_count: input.pairing_u32("header", "string_count")?[row],
            string_byte_count: input.pairing_u64("header", "string_byte_count")?[row],
            no_fermion_line: input.pairing_u32("header", "no_fermion_line")?[row],
            topology_digest: pairing_digest_row(input, "header", "topology_digest", row)?,
            semantic_digest: pairing_digest_row(input, "header", "semantic_digest", row)?,
        })
    })?;
    let endpoints = decode_pairing_rows(input, "endpoints", |row| {
        Ok(process::FermionPairingEndpointRow {
            endpoint_id: input.pairing_u32("endpoints", "endpoint_id")?[row],
            source_slot: input.pairing_u32("endpoints", "source_slot")?[row],
            public_label: input.pairing_u32("endpoints", "public_label")?[row],
            species_class_id: input.pairing_u32("endpoints", "species_class_id")?[row],
            species_string_id: input.pairing_u32("endpoints", "species_string_id")?[row],
            particle_orientation: input.pairing_u8("endpoints", "particle_orientation")?[row],
            color_orientation: input.pairing_u8("endpoints", "color_orientation")?[row],
            state_template_range: pairing_range(input, "endpoints", "state_template", row)?,
            anti_state_template_range: pairing_range(
                input,
                "endpoints",
                "anti_state_template",
                row,
            )?,
            basis_range: pairing_range(input, "endpoints", "basis", row)?,
            color_representation_range: pairing_range(
                input,
                "endpoints",
                "color_representation",
                row,
            )?,
            contract_digest: pairing_digest_row(input, "endpoints", "contract_digest", row)?,
        })
    })?;
    let pairing_classes = decode_pairing_rows(input, "pairing_classes", |row| {
        Ok(process::FermionPairingClassRow {
            class_id: input.pairing_u32("pairing_classes", "class_id")?[row],
            species_class_id: input.pairing_u32("pairing_classes", "species_class_id")?[row],
            species_string_id: input.pairing_u32("pairing_classes", "species_string_id")?[row],
            fundamental_slot_range: pairing_range(
                input,
                "pairing_classes",
                "fundamental_slot",
                row,
            )?,
            antifundamental_slot_range: pairing_range(
                input,
                "pairing_classes",
                "antifundamental_slot",
                row,
            )?,
            reference_pairing_range: pairing_range(
                input,
                "pairing_classes",
                "reference_pairing",
                row,
            )?,
            pairing_count: input.pairing_u64("pairing_classes", "pairing_count")?[row],
            proof_digest: pairing_digest_row(input, "pairing_classes", "proof_digest", row)?,
        })
    })?;
    let rules = decode_pairing_rows(input, "rules", |row| {
        Ok(process::FermionPairingRuleRow {
            rule_id: input.pairing_u32("rules", "rule_id")?[row],
            class_pairing_index_range: pairing_range(input, "rules", "class_pairing_index", row)?,
            endpoint_pairing_range: pairing_range(input, "rules", "endpoint_pairing", row)?,
            source_permutation_range: pairing_range(input, "rules", "source_permutation", row)?,
            lineage_range: pairing_range(input, "rules", "lineage", row)?,
            fermion_parity: input.pairing_i32("rules", "fermion_parity")?[row],
            real_numerator_integer_id: input.pairing_u32("rules", "real_numerator_integer_id")?
                [row],
            real_denominator_integer_id: input
                .pairing_u32("rules", "real_denominator_integer_id")?[row],
            imag_numerator_integer_id: input.pairing_u32("rules", "imag_numerator_integer_id")?
                [row],
            imag_denominator_integer_id: input
                .pairing_u32("rules", "imag_denominator_integer_id")?[row],
            multiplicity: input.pairing_u64("rules", "multiplicity")?[row],
            proof_algorithm_string_id: input.pairing_u32("rules", "proof_algorithm_string_id")?
                [row],
            proof_digest: pairing_digest_row(input, "rules", "proof_digest", row)?,
        })
    })?;
    let rule_class_pairing_indices =
        decode_pairing_rows(input, "rule_class_pairing_indices", |row| {
            Ok(process::FermionPairingClassPairingIndexRow {
                class_id: input.pairing_u32("rule_class_pairing_indices", "class_id")?[row],
                pairing_index: input.pairing_u64("rule_class_pairing_indices", "pairing_index")?
                    [row],
            })
        })?;
    let decode_pairs = |table_name: &str| {
        decode_pairing_rows(input, table_name, |row| {
            Ok(process::FermionPairingEndpointPairRow {
                fundamental_source_slot: input
                    .pairing_u32(table_name, "fundamental_source_slot")?[row],
                antifundamental_source_slot: input
                    .pairing_u32(table_name, "antifundamental_source_slot")?[row],
            })
        })
    };
    let exact_integers = decode_pairing_rows(input, "exact_integers", |row| {
        Ok(process::FermionPairingExactIntegerRow {
            integer_id: input.pairing_u32("exact_integers", "integer_id")?[row],
            sign: input.pairing_i32("exact_integers", "sign")?[row],
            limb_range: pairing_range(input, "exact_integers", "limb", row)?,
        })
    })?;

    Ok(Some(process::OwnedFermionPairingInput {
        input_abi: process::RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI.to_owned(),
        declared_columnar_digest: semantic_digest_from_hex(
            declared_digest,
            "recurrence fermion-pairing digest",
        )?,
        header,
        endpoints,
        endpoint_state_template_ids: input
            .pairing_u32("endpoint_state_template_ids", "string_id")?
            .to_vec(),
        endpoint_anti_state_template_ids: input
            .pairing_u32("endpoint_anti_state_template_ids", "string_id")?
            .to_vec(),
        endpoint_basis_ids: input
            .pairing_u32("endpoint_basis_ids", "string_id")?
            .to_vec(),
        endpoint_color_representations: input
            .pairing_i32("endpoint_color_representations", "value")?
            .to_vec(),
        pairing_classes,
        class_fundamental_slots: input
            .pairing_u32("class_fundamental_slots", "source_slot")?
            .to_vec(),
        class_antifundamental_slots: input
            .pairing_u32("class_antifundamental_slots", "source_slot")?
            .to_vec(),
        class_reference_pairings: decode_pairs("class_reference_pairings")?,
        rules,
        rule_class_pairing_indices,
        rule_endpoint_pairings: decode_pairs("rule_endpoint_pairings")?,
        rule_source_slot_permutations: input
            .pairing_u32("rule_source_slot_permutations", "source_slot")?
            .to_vec(),
        rule_lineages: input.pairing_u32("rule_lineages", "line_id")?.to_vec(),
        exact_integers,
        exact_integer_limbs: input.pairing_u64("exact_integer_limbs", "value")?.to_vec(),
        string_ranges: pairing_plain_ranges(input, "string_ranges")?,
        string_bytes: input.pairing_u8("string_bytes", "value")?.to_vec(),
    }))
}

fn decode_pairing_rows<T>(
    input: &OwnedInput,
    table_name: &str,
    mut decode: impl FnMut(usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = checked_usize(
        input.fermion_pairing_table(table_name)?.row_count,
        &format!("fermion-pairing {table_name} row count"),
    )?;
    (0..row_count).map(&mut decode).collect()
}

fn pairing_range(
    input: &OwnedInput,
    table_name: &str,
    prefix: &str,
    row: usize,
) -> RusticolResult<CheckedTableRange> {
    Ok(CheckedTableRange {
        start: input.pairing_u64(table_name, &format!("{prefix}_start"))?[row],
        count: input.pairing_u64(table_name, &format!("{prefix}_count"))?[row],
    })
}

fn pairing_plain_ranges(
    input: &OwnedInput,
    table_name: &str,
) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_pairing_rows(input, table_name, |row| {
        Ok(CheckedTableRange {
            start: input.pairing_u64(table_name, "start")?[row],
            count: input.pairing_u64(table_name, "count")?[row],
        })
    })
}

fn pairing_digest_row(
    input: &OwnedInput,
    table_name: &str,
    column_name: &str,
    row: usize,
) -> RusticolResult<[u8; 32]> {
    let values = input.pairing_u8(table_name, column_name)?;
    let start = row
        .checked_mul(32)
        .ok_or_else(|| invalid("fermion-pairing digest offset exceeds usize"))?;
    values
        .get(start..start + 32)
        .ok_or_else(|| invalid("fermion-pairing digest row is truncated"))?
        .try_into()
        .map_err(|_| invalid("fermion-pairing digest row must contain 32 bytes"))
}

fn decode_process_rows<T>(
    input: &OwnedInput,
    table_name: &str,
    mut decode: impl FnMut(usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = checked_usize(
        input.table(table_name)?.row_count,
        &format!("{table_name} row count"),
    )?;
    (0..row_count).map(&mut decode).collect()
}

fn plain_ranges(input: &OwnedInput, table_name: &str) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_process_rows(input, table_name, |row| {
        Ok(CheckedTableRange {
            start: input.u64(table_name, "start")?[row],
            count: input.u64(table_name, "count")?[row],
        })
    })
}

fn decode_template_input(
    input: &OwnedInput,
    catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
) -> RusticolResult<template::OwnedRecurrenceTemplateInput> {
    use template::{
        CatalogHeaderRow, ClosureRow, ColorContractionRow, ColorNcTermRow,
        ContactOrbitCertificateRow, ContactOrbitStepRow, CouplingOrderTermRow, CurrentStateRow,
        EvaluatorBindingRow, ExactFactorRow, LCColorTransitionWitnessRow,
        OwnedRecurrenceTemplateInput, ParameterRow, PropagatorRow, QuantumFlowRow,
        QuantumNumberFlowTermRow, RuntimeHelicityContractRow, RuntimeHelicityEmbeddingRow,
        RuntimeHelicityProjectionRow, RuntimeHelicityVariantRow, SourceRow, SymmetryProofRow,
        TransitionRow,
    };
    let catalog_header = decode_template_rows(input.table("catalog_header")?, |table, row| {
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
    let coupling_order_ranges = template_indexed_ranges(input.table("coupling_order_ranges")?)?;
    let coupling_order_terms =
        decode_template_rows(input.table("coupling_order_terms")?, |table, row| {
            Ok(CouplingOrderTermRow {
                set_id: table.u32("set_id")?[row],
                name_string_id: table.u32("name_string_id")?[row],
                power: table.u32("power")?[row],
            })
        })?;
    let contact_orbit_certificates =
        decode_template_rows(input.table("contact_orbit_certificates")?, |table, row| {
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
    let contact_orbit_steps =
        decode_template_rows(input.table("contact_orbit_steps")?, |table, row| {
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
    let current_states = decode_template_rows(input.table("current_states")?, |table, row| {
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
    let digest_catalog = decode_template_digest_catalog(input.table("digest_catalog")?)?;
    let evaluator_bindings =
        decode_template_rows(input.table("evaluator_bindings")?, |table, row| {
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
    let exact_factors = decode_template_rows(input.table("exact_factors")?, |table, row| {
        Ok(ExactFactorRow {
            id: table.u32("id")?[row],
            real_numerator_string_id: table.u32("real_numerator_string_id")?[row],
            real_denominator_string_id: table.u32("real_denominator_string_id")?[row],
            imag_numerator_string_id: table.u32("imag_numerator_string_id")?[row],
            imag_denominator_string_id: table.u32("imag_denominator_string_id")?[row],
        })
    })?;
    let flavour_flow_ranges = template_indexed_ranges(input.table("flavour_flow_ranges")?)?;
    let flavour_flow_values = input.table("flavour_flow_values")?.i32("value")?.to_vec();
    let i32_sequence_ranges = template_indexed_ranges(input.table("i32_sequence_ranges")?)?;
    let i32_sequence_values = input.table("i32_sequence_values")?.i32("value")?.to_vec();
    let parameters = decode_template_rows(input.table("parameters")?, |table, row| {
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
    let propagators = decode_template_rows(input.table("propagators")?, |table, row| {
        Ok(PropagatorRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            state_template_id: table.u32("state_template_id")?[row],
            applies_propagator: table.u8("applies_propagator")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            numerator_expression_digest_id: table.u32("numerator_expression_digest_id")?[row],
            denominator_expression_digest_id: table.u32("denominator_expression_digest_id")?[row],
            mass_parameter_id: table.u32("mass_parameter_id")?[row],
            width_parameter_id: table.u32("width_parameter_id")?[row],
            gauge_string_id: table.u32("gauge_string_id")?[row],
            linearity_proof_template_id: table.u32("linearity_proof_template_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let quantum_flows = decode_template_rows(input.table("quantum_flows")?, |table, row| {
        Ok(QuantumFlowRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
            input_spin_sequence_id: table.u32("input_spin_sequence_id")?[row],
            input_flavour_sequence_id: table.u32("input_flavour_sequence_id")?[row],
            input_quantum_sequence_id: table.u32("input_quantum_sequence_id")?[row],
            flavour_flow_operation_string_id: table.u32("flavour_flow_operation_string_id")?[row],
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
    let quantum_number_flow_ranges =
        template_indexed_ranges(input.table("quantum_number_flow_ranges")?)?;
    let quantum_number_flow_terms =
        decode_template_rows(input.table("quantum_number_flow_terms")?, |table, row| {
            Ok(QuantumNumberFlowTermRow {
                flow_id: table.u32("flow_id")?[row],
                name_string_id: table.u32("name_string_id")?[row],
                expression_string_id: table.u32("expression_string_id")?[row],
            })
        })?;
    let runtime_helicity_contracts =
        decode_template_rows(input.table("runtime_helicity_contracts")?, |table, row| {
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
        decode_template_rows(input.table("runtime_helicity_embeddings")?, |table, row| {
            Ok(RuntimeHelicityEmbeddingRow {
                variant_id: table.u32("variant_id")?[row],
                full_component: table.u32("full_component")?[row],
                source_component: table.u32("source_component")?[row],
                factor_id: table.u32("factor_id")?[row],
            })
        })?;
    let runtime_helicity_projections = decode_template_rows(
        input.table("runtime_helicity_projections")?,
        |table, row| {
            Ok(RuntimeHelicityProjectionRow {
                variant_id: table.u32("variant_id")?[row],
                source_component: table.u32("source_component")?[row],
                full_component: table.u32("full_component")?[row],
            })
        },
    )?;
    let runtime_helicity_variants =
        decode_template_rows(input.table("runtime_helicity_variants")?, |table, row| {
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
    let sources = decode_template_rows(input.table("sources")?, |table, row| {
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
            wavefunction_expression_digest_id: table.u32("wavefunction_expression_digest_id")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            mass_parameter_id: table.u32("mass_parameter_id")?[row],
            width_parameter_id: table.u32("width_parameter_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let string_ranges = template_plain_ranges(input.table("string_ranges")?)?;
    let string_bytes = input.table("string_bytes")?.u8("value")?.to_vec();
    let symmetry_proofs = decode_template_rows(input.table("symmetry_proofs")?, |table, row| {
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
    let transitions = decode_template_rows(input.table("transitions")?, |table, row| {
        Ok(TransitionRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
            result_state_template_id: table.u32("result_state_template_id")?[row],
            quantum_flow_template_id: table.u32("quantum_flow_template_id")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            canonical_input_order_sequence_id: table.u32("canonical_input_order_sequence_id")?[row],
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
    let closures = decode_template_rows(input.table("closures")?, |table, row| {
        Ok(ClosureRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
            result_state_template_id: table.u32("result_state_template_id")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            canonical_input_order_sequence_id: table.u32("canonical_input_order_sequence_id")?[row],
            coupling_parameter_sequence_id: table.u32("coupling_parameter_sequence_id")?[row],
            coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
            eligible_quantum_flow_sequence_id: table.u32("eligible_quantum_flow_sequence_id")?[row],
            color_contraction_template_id: table.u32("color_contraction_template_id")?[row],
            binding_coupling_factor_id: table.u32("binding_coupling_factor_id")?[row],
            exact_factor_id: table.u32("exact_factor_id")?[row],
            output_factor_source: table.u8("output_factor_source")?[row],
            equivalence_class_string_id: table.u32("equivalence_class_string_id")?[row],
            input_exchange_factor_id: table.u32("input_exchange_factor_id")?[row],
            projection_string_id: table.u32("projection_string_id")?[row],
            component_coefficient_sequence_id: table.u32("component_coefficient_sequence_id")?[row],
            chirality_relation_string_id: table.u32("chirality_relation_string_id")?[row],
            metric_signature_string_id: table.u32("metric_signature_string_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let color_contractions =
        decode_template_rows(input.table("color_contractions")?, |table, row| {
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
    let lc_color_transition_witnesses = decode_template_rows(
        input.table("lc_color_transition_witnesses")?,
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
                input_port_pairing_sequence_id: table.u32("input_port_pairing_sequence_id")?[row],
                result_port_binding_sequence_id: table.u32("result_port_binding_sequence_id")?[row],
                provenance_sequence_id: table.u32("provenance_sequence_id")?[row],
            })
        },
    )?;
    let color_nc_terms = decode_template_rows(input.table("color_nc_terms")?, |table, row| {
        Ok(ColorNcTermRow {
            color_contraction_id: table.u32("color_contraction_id")?[row],
            exponent: table.i32("exponent")?[row],
            factor_id: table.u32("factor_id")?[row],
        })
    })?;
    let u32_sequence_ranges = template_indexed_ranges(input.table("u32_sequence_ranges")?)?;
    let u32_sequence_values = input.table("u32_sequence_values")?.u32("value")?.to_vec();

    Ok(OwnedRecurrenceTemplateInput {
        input_abi: input.abi.clone(),
        catalog_digest: catalog_digest,
        compiled_model_digest: compiled_model_digest,
        prepared_kernel_pack_digest: prepared_kernel_pack_digest,
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

fn decode_template_rows<T>(
    table: &OwnedTable,
    mut decode: impl FnMut(&OwnedTable, usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = table.row_count()?;
    (0..row_count).map(|row| decode(table, row)).collect()
}

fn template_indexed_ranges(table: &OwnedTable) -> RusticolResult<Vec<template::IndexedRangeRow>> {
    decode_template_rows(table, |table, row| {
        Ok(template::IndexedRangeRow {
            id: table.u32("id")?[row],
            range: CheckedTableRange::new(table.u64("start")?[row], table.u64("count")?[row]),
        })
    })
}

fn template_plain_ranges(table: &OwnedTable) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_template_rows(table, |table, row| {
        Ok(CheckedTableRange::new(
            table.u64("start")?[row],
            table.u64("count")?[row],
        ))
    })
}

fn decode_template_digest_catalog(
    table: &OwnedTable,
) -> RusticolResult<Vec<template::DigestCatalogRow>> {
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
            Ok(template::DigestCatalogRow {
                id: ids[row],
                value,
            })
        })
        .collect()
}

fn validate_inventory(input: &OwnedInput) -> RusticolResult<()> {
    if input.abi != RECURRENCE_BUILDER_INPUT_ABI {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence builder input ABI {:?}; expected {:?}",
            input.abi, RECURRENCE_BUILDER_INPUT_ABI
        )));
    }
    if input.tables.len() != TABLE_SPECS.len() {
        return Err(invalid(
            "recurrence builder table inventory changed after extraction",
        ));
    }
    for (table, spec) in input.tables.iter().zip(TABLE_SPECS) {
        if table.name != spec.name || table.columns.len() != spec.columns.len() {
            return Err(invalid(format!(
                "recurrence builder table {:?} schema changed after extraction",
                table.name
            )));
        }
        for (column, column_spec) in table.columns.iter().zip(spec.columns) {
            if column.name != column_spec.name || column.dtype != column_spec.kind.dtype() {
                return Err(invalid(format!(
                    "recurrence builder column {}.{} schema changed after extraction",
                    table.name, column.name
                )));
            }
            if column.shape.first().copied() != Some(table.row_count) {
                return Err(invalid(format!(
                    "recurrence builder column {}.{} row count changed after extraction",
                    table.name, column.name
                )));
            }
        }
    }
    Ok(())
}

fn canonical_digest(input: &OwnedInput) -> RusticolResult<String> {
    let mut digest = Sha256::new();
    hash_text(&mut digest, &input.abi)?;
    hash_tables(&mut digest, &input.tables)?;
    Ok(hex_digest(digest.finalize()))
}

fn hash_tables(digest: &mut Sha256, tables: &[OwnedTable]) -> RusticolResult<()> {
    digest.update(
        u64::try_from(tables.len())
            .map_err(|_| invalid("recurrence table count exceeds u64"))?
            .to_le_bytes(),
    );
    for table in tables {
        hash_text(digest, &table.name)?;
        digest.update(table.row_count.to_le_bytes());
        digest.update(
            u32::try_from(table.columns.len())
                .map_err(|_| invalid("recurrence table column count exceeds u32"))?
                .to_le_bytes(),
        );
        for column in &table.columns {
            hash_text(digest, &column.name)?;
            hash_text(digest, column.dtype)?;
            digest.update(
                u8::try_from(column.shape.len())
                    .map_err(|_| invalid("recurrence column rank exceeds u8"))?
                    .to_le_bytes(),
            );
            for dimension in &column.shape {
                digest.update(dimension.to_le_bytes());
            }
            digest.update(column.values.raw_bytes());
        }
    }
    Ok(())
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
            "recurrence builder input {attribute} must be a string"
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
            "recurrence builder {context} must not be empty"
        )));
    }
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

fn semantic_digest_python_attribute(
    value: &Bound<'_, PyAny>,
    attribute: &str,
) -> PyResult<SemanticDigest> {
    let digest = required_string(value, attribute)?;
    semantic_digest_from_hex(&digest, attribute).map_err(python_error)
}

pub(super) fn semantic_digest_from_hex(
    value: &str,
    context: &str,
) -> RusticolResult<SemanticDigest> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(invalid(format!(
            "{context} must be a lowercase SHA-256 digest"
        )));
    }
    let mut result = [0_u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| invalid(format!("{context} is not hexadecimal")))?;
    }
    SemanticDigest::new(result)
}

fn hash_text(digest: &mut Sha256, value: &str) -> RusticolResult<()> {
    digest.update(
        u64::try_from(value.len())
            .map_err(|_| invalid("recurrence digest text length exceeds u64"))?
            .to_le_bytes(),
    );
    digest.update(value.as_bytes());
    Ok(())
}

pub(super) fn hex_digest(value: impl AsRef<[u8]>) -> String {
    let mut result = String::with_capacity(value.as_ref().len() * 2);
    for byte in value.as_ref() {
        use std::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

pub(super) fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn wrong_type(context: &str, expected: &str) -> RusticolError {
    invalid(format!("{context} is not a {expected} recurrence column"))
}

#[cfg(test)]
mod direct_binding_tests {
    use super::*;
    use serde_json::json;

    fn construction_progress(
        stage_index: usize,
        candidate_parent_pair_count: usize,
    ) -> RecurrenceBuildProgress {
        RecurrenceBuildProgress {
            phase: "recurrence stage",
            phase_index: stage_index + 2,
            phase_total: 33,
            stage_index: Some(stage_index),
            stage_total: 30,
            subset_size: Some(stage_index % 5 + 2),
            candidate_parent_pair_count,
            candidate_parent_pair_total: Some(candidate_parent_pair_count),
            current_count: 0,
            contribution_count: 0,
            dynamic_color_state_count: 0,
            color_target_prune_count: 0,
        }
    }

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn refresh_digest(value: &mut JsonValue, field: &str, context: &str) {
        value.as_object_mut().unwrap().remove(field);
        let digest: [u8; 32] = Sha256::digest(canonical_json_bytes(value, context).unwrap()).into();
        value
            .as_object_mut()
            .unwrap()
            .insert(field.to_owned(), json!(hex_digest(digest)));
    }

    fn canonical_direct_catalog() -> (Vec<u8>, SemanticDigest, SemanticDigest, SemanticDigest) {
        let prepared_pack = semantic_digest_from_hex(&digest(1), "test pack").unwrap();
        let semantic_catalog =
            semantic_digest_from_hex(&digest(2), "test semantic catalog").unwrap();
        let compiled_model = semantic_digest_from_hex(&digest(3), "test compiled model").unwrap();
        let mut template = json!({
            "abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "alignment_bytes": 64,
            "backend": "jit",
            "coupling_slot_count": 0,
            "destination_aliasing": false,
            "destination_component_count": 4,
            "destination_operation": "initialize",
            "direct_executor_id": 0,
            "evaluator_binding_id": 7,
            "evaluator_resolver_key": "source:7",
            "exact_expression_digest": digest(4),
            "momentum_operand_count": 1,
            "optimization_level": 2,
            "parameter_slot_count": 0,
            "parent_arity": 0,
            "parent_component_counts": [],
            "payload_binding": {
                "abi": DIRECT_PAYLOAD_BINDING_ABI,
                "destination_operation": "initialize",
                "direct_application_abi": null,
                "exact_factor_scalar_slots": [],
                "input_plane_count": 0,
                "input_plane_projections": [],
                "intrinsic_contract_digest": null,
                "kind": "rusticol-intrinsic",
                "native_entry_point": null,
                "output_alias_inputs": [],
                "contribution_parent_permutation": [0, 1],
                "parameter_bindings": [],
                "payload_digest": digest(5),
                "payload_paths": [],
                "prepared_kernel_id": null,
                "prepared_template_semantic_digest": null,
                "role": "source",
                "runtime_template": "rusticol.source-fill.v1",
                "scalar_input_count": 0,
                "scalar_projections": [],
                "source_application_abi": null,
                "source_application_path": null,
                "source_application_sha256": null,
                "state_plane_indices": []
            },
            "portable": true,
            "role": "source",
            "semantic_template_ids": ["source:test"],
            "simd_axis": "points-contiguous",
            "target_triple": "symjit-storage-v3-portable",
            "template_id": "direct:source:test"
        });
        let template_digest: [u8; 32] =
            Sha256::digest(canonical_json_bytes(&template, "test template").unwrap()).into();
        template.as_object_mut().unwrap().insert(
            "semantic_digest".to_owned(),
            json!(hex_digest(template_digest)),
        );

        let mut catalog = json!({
            "abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "backend": "jit",
            "backend_abi": DIRECT_BACKEND_ABI,
            "canonicalization_abi": DIRECT_CANONICALIZATION_ABI,
            "compiled_model_digest": compiled_model.to_string(),
            "optimization_level": 2,
            "optimization_settings_digest": digest(6),
            "portable": true,
            "prepared_kernel_contract_digest": digest(7),
            "prepared_kernel_pack_digest": prepared_pack.to_string(),
            "prepared_kernel_payload_digest": digest(8),
            "recurrence_template_catalog_digest": semantic_catalog.to_string(),
            "target_triple": "symjit-storage-v3-portable",
            "templates": [template]
        });
        let catalog_digest: [u8; 32] =
            Sha256::digest(canonical_json_bytes(&catalog, "test catalog").unwrap()).into();
        catalog.as_object_mut().unwrap().insert(
            "catalog_digest".to_owned(),
            json!(hex_digest(catalog_digest)),
        );
        (
            canonical_json_bytes(&catalog, "test catalog").unwrap(),
            prepared_pack,
            semantic_catalog,
            compiled_model,
        )
    }

    #[test]
    fn canonical_catalog_authenticates_dense_role_binding() {
        let (bytes, pack, semantic, model) = canonical_direct_catalog();
        let parsed = parse_direct_template_catalog(&bytes, pack, semantic, model).unwrap();

        assert_eq!(parsed.catalog.direct_executor_count(), 1);
        assert_eq!(
            parsed
                .catalog
                .resolve_evaluator(DirectExecutorRole::Source, 7)
                .unwrap(),
            0
        );
        assert_eq!(parsed.prepared_kernel_count, 0);
    }

    #[test]
    fn construction_metrics_sum_lane_unique_stage_counts() {
        let mut metrics = RecurrenceConstructionMetrics::default();
        for lane_index in 0..6 {
            for stage_index_within_lane in 0..5 {
                let stage_index = lane_index * 5 + stage_index_within_lane;
                let final_count = (lane_index + 1) * (stage_index_within_lane + 2);
                metrics.include(&construction_progress(stage_index, final_count / 2));
                metrics.include(&construction_progress(stage_index, final_count));
            }
        }

        let expected = (0..6)
            .flat_map(|lane_index| {
                (0..5).map(move |stage_index_within_lane| {
                    (lane_index + 1) * (stage_index_within_lane + 2)
                })
            })
            .sum::<usize>();
        assert_eq!(metrics.candidate_parent_pair_count_by_stage.len(), 30);
        assert_eq!(metrics.candidate_parent_pair_count(), expected);
    }

    #[test]
    fn noncanonical_catalog_bytes_are_rejected() {
        let (mut bytes, pack, semantic, model) = canonical_direct_catalog();
        bytes.push(b'\n');

        let error = match parse_direct_template_catalog(&bytes, pack, semantic, model) {
            Ok(_) => panic!("noncanonical direct catalog was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("not canonical"));
    }

    #[test]
    fn catalog_digest_authenticates_executor_mapping() {
        let (bytes, pack, semantic, model) = canonical_direct_catalog();
        let mut value: JsonValue = serde_json::from_slice(&bytes).unwrap();
        value["templates"][0]["evaluator_binding_id"] = json!(8);
        let tampered = canonical_json_bytes(&value, "tampered catalog").unwrap();

        let error = match parse_direct_template_catalog(&tampered, pack, semantic, model) {
            Ok(_) => panic!("tampered direct catalog was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("catalog digest"));
    }

    #[test]
    fn pending_direct_payload_is_rejected_before_lowering() {
        let (bytes, pack, semantic, model) = canonical_direct_catalog();
        let mut value: JsonValue = serde_json::from_slice(&bytes).unwrap();
        let template = &mut value["templates"][0];
        template["payload_binding"]["kind"] = json!("pending-direct-call-abi");
        template["payload_binding"]["runtime_template"] = JsonValue::Null;
        template["payload_binding"]["prepared_kernel_id"] = json!(7);
        refresh_digest(template, "semantic_digest", "pending direct template");
        refresh_digest(&mut value, "catalog_digest", "pending direct catalog");
        let pending = canonical_json_bytes(&value, "pending direct catalog").unwrap();

        let error = match parse_direct_template_catalog(&pending, pack, semantic, model) {
            Ok(_) => panic!("pending direct payload was accepted"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("no executable Direct-Arena payload")
        );
    }
}
