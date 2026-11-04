// SPDX-License-Identifier: 0BSD

//! Validated columnar input and deterministic lowering for eager plan v3.
//!
//! This module deliberately stops at an owned, fixed-width in-memory plan. It
//! neither writes PACBIN nor replaces the plan-v2 execution runtime. The Python
//! binding can borrow contiguous NumPy columns through the views below, build a
//! validated owned input, release the GIL, and call [`lower_eager_plan_v3`].

use crate::eager_layout::{
    EAGER_LOWERING_INPUT_ABI, EAGER_PLAN_ABI, EagerSectionHeader, EagerSectionKind,
};
use crate::{MISSING_U32, RusticolError, RusticolResult};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const EXACT_SOURCE_BINARY64: u8 = 0;
const EXACT_SOURCE_CANONICAL_IR: u8 = 1;
const OUTPUT_FACTOR_NONE: u8 = 0;
const OUTPUT_FACTOR_COUPLING_REAL: u8 = 1;
const OUTPUT_FACTOR_COUPLING_IMAG: u8 = 2;

const CURRENT_FLAG_SOURCE: u32 = 1 << 0;

/// Complete, sorted table set defined by `pyamplicol-eager-lowering-input-v1`.
///
/// The binding supplies retained fragments for every table. Columns consumed
/// by typed views may be omitted from a fragment; all other columns are copied
/// bit-for-bit into the owned input and plan.
pub const EAGER_LOWERING_V1_TABLE_NAMES: &[&str] = &[
    "bitset_ranges",
    "bitset_words",
    "coherent_groups",
    "color_contraction_entries",
    "color_contraction_metadata",
    "color_open_lines",
    "color_plan_diagnostics",
    "color_replay_contraction_component_groups",
    "color_replay_contraction_entries",
    "color_replay_contraction_metadata",
    "color_replay_mappings",
    "color_replay_metadata",
    "color_replay_permutations",
    "color_replay_physical_groups",
    "color_replay_routes",
    "color_sectors",
    "color_selectors",
    "contraction_coefficients",
    "coupling_orders",
    "couplings",
    "currents",
    "exact_factors",
    "helicity_amplitude_classes",
    "helicity_amplitude_member_domains",
    "helicity_amplitude_members",
    "helicity_domain_states",
    "helicity_domains",
    "helicity_materialization_metadata",
    "helicity_materialized_amplitude_domains",
    "helicity_materialized_amplitude_routes",
    "helicity_materialized_current_map",
    "helicity_materialized_schedules",
    "helicity_materialized_source_routes",
    "helicity_proof_diagnostics",
    "helicity_proof_metadata",
    "helicity_recurrence_classes",
    "helicity_recurrence_members",
    "helicity_residual_currents",
    "helicity_residual_roots",
    "helicity_selectors",
    "helicity_source_mappings",
    "helicity_structural_zero_domains",
    "i32_sequence_ranges",
    "i32_sequence_values",
    "interaction_group_members",
    "interaction_groups",
    "interactions",
    "lc_replay_diagnostics",
    "lc_replay_members",
    "lc_replay_metadata",
    "lc_replay_partitions",
    "lc_replay_permutations",
    "lc_replay_residual_sectors",
    "metadata",
    "model_parameters",
    "momentum_masks",
    "quantum_flows",
    "reduction_members",
    "roots",
    "selected_color_sectors",
    "selected_source_helicities",
    "sources",
    "u32_sequence_ranges",
    "u32_sequence_values",
];

/// Borrowed primitive storage for a retained column fragment.
#[derive(Clone, Copy, Debug)]
pub enum EagerPrimitiveColumnView<'a> {
    U8(&'a [u8]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    I32(&'a [i32]),
    F64(&'a [f64]),
}

impl EagerPrimitiveColumnView<'_> {
    fn len(self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }
}

/// One borrowed fixed-shape column not consumed by a typed lowering view.
#[derive(Clone, Copy, Debug)]
pub struct EagerRetainedColumnView<'a> {
    pub name: &'a str,
    /// Number of primitive values per table row, including trailing shape.
    pub elements_per_row: u32,
    pub values: EagerPrimitiveColumnView<'a>,
}

/// Retained fragment for one source table.
#[derive(Clone, Copy, Debug)]
pub struct EagerRetainedTableView<'a> {
    pub name: &'a str,
    pub row_count: u64,
    pub columns: &'a [EagerRetainedColumnView<'a>],
}

/// Owned primitive storage for retained metadata/proof columns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EagerOwnedPrimitiveColumn {
    U8(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
    /// Raw IEEE-754 bits preserve signed zero exactly.
    F64Bits(Vec<u64>),
}

impl EagerOwnedPrimitiveColumn {
    pub fn element_count(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) | Self::F64Bits(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }
}

/// Owned retained column with a deterministic primitive shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerOwnedRetainedColumn {
    name: Box<str>,
    elements_per_row: u32,
    values: EagerOwnedPrimitiveColumn,
}

impl EagerOwnedRetainedColumn {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn elements_per_row(&self) -> u32 {
        self.elements_per_row
    }

    pub fn values(&self) -> &EagerOwnedPrimitiveColumn {
        &self.values
    }
}

/// Owned retained table fragment carried unchanged into plan-v3.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerOwnedRetainedTable {
    name: Box<str>,
    row_count: u64,
    columns: Vec<EagerOwnedRetainedColumn>,
}

impl EagerOwnedRetainedTable {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn columns(&self) -> &[EagerOwnedRetainedColumn] {
        &self.columns
    }
}

/// Borrowed columns from the `currents` input table.
#[derive(Clone, Copy, Debug)]
pub struct EagerCurrentColumns<'a> {
    pub id: &'a [u32],
    pub dimension: &'a [u32],
    pub is_source: &'a [u8],
    pub momentum_mask_bitset_id: &'a [u32],
    pub propagator_ir_id: &'a [u32],
    pub propagator_kernel_id: &'a [u32],
}

/// Borrowed columns from the `sources` input table.
#[derive(Clone, Copy, Debug)]
pub struct EagerSourceColumns<'a> {
    pub source_id: &'a [u32],
    pub current_id: &'a [u32],
    pub external_label: &'a [u32],
    pub input_momentum_slot: &'a [u32],
    pub source_ir_id: &'a [u32],
    pub crossing_ir_id: &'a [u32],
    pub crossing_factor_id: &'a [u32],
    pub declared_state_index: &'a [u32],
}

/// Borrowed columns from the `interactions` input table.
#[derive(Clone, Copy, Debug)]
pub struct EagerInteractionColumns<'a> {
    pub id: &'a [u32],
    pub stage_subset_size: &'a [u32],
    pub left_current_id: &'a [u32],
    pub right_current_id: &'a [u32],
    pub result_current_id: &'a [u32],
    pub coupling_id: &'a [u32],
    pub coupling_factor_id: &'a [u32],
    pub color_factor_id: &'a [u32],
    pub evaluation_factor_id: &'a [u32],
    pub evaluation_group_id: &'a [u32],
    pub full_tensor_network_ready: &'a [u8],
    pub kernel_id: &'a [u32],
    pub canonical_input_order: &'a [[u8; 2]],
    pub kernel_normalization_factor_id: &'a [u32],
    pub output_factor_source: &'a [u8],
}

/// Borrowed columns from the `roots` input table.
#[derive(Clone, Copy, Debug)]
pub struct EagerRootColumns<'a> {
    pub id: &'a [u32],
    pub left_current_id: &'a [u32],
    pub right_current_id: &'a [u32],
    pub color_factor_id: &'a [u32],
    pub contraction_ir_id: &'a [u32],
    pub coupling_id: &'a [u32],
    pub coupling_factor_id: &'a [u32],
    pub helicity_weight_factor_id: &'a [u32],
    pub coherent_group_id: &'a [u32],
    pub kernel_id: &'a [u32],
    pub canonical_input_order: &'a [[u8; 2]],
    pub kernel_normalization_factor_id: &'a [u32],
    pub output_factor_source: &'a [u8],
}

/// Borrowed columns from `interaction_groups` and
/// `interaction_group_members`.
#[derive(Clone, Copy, Debug)]
pub struct EagerInteractionGroupColumns<'a> {
    pub representative_interaction_id: &'a [u32],
    pub member_start: &'a [u64],
    pub member_count: &'a [u64],
    pub member_interaction_id: &'a [u32],
}

/// Borrowed columns from the `momentum_masks` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerMomentumMaskColumns<'a> {
    pub slot_id: &'a [u32],
    pub bitset_id: &'a [u32],
}

/// Borrowed arbitrary-width bitset catalog.
#[derive(Clone, Copy, Debug)]
pub struct EagerBitsetColumns<'a> {
    pub start: &'a [u64],
    pub count: &'a [u64],
    pub bit_count: &'a [u64],
    pub words: &'a [u64],
}

/// Borrowed unsigned sequence catalog.
#[derive(Clone, Copy, Debug)]
pub struct EagerU32SequenceColumns<'a> {
    pub start: &'a [u64],
    pub count: &'a [u64],
    pub values: &'a [u32],
}

/// Borrowed signed sequence catalog.
#[derive(Clone, Copy, Debug)]
pub struct EagerI32SequenceColumns<'a> {
    pub start: &'a [u64],
    pub count: &'a [u64],
    pub values: &'a [i32],
}

/// Borrowed columns from the `exact_factors` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerExactFactorColumns<'a> {
    pub real: &'a [f64],
    pub imaginary: &'a [f64],
    pub canonical_string_id: &'a [u32],
    pub exact_source: &'a [u8],
    pub exact_ir_id: &'a [u32],
    pub source_ir_id: &'a [u32],
}

/// Borrowed columns from the `couplings` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerCouplingColumns<'a> {
    pub constant_factor_id: &'a [u32],
    pub parameter_name_ids_sequence_id: &'a [u32],
}

/// Borrowed columns from the `model_parameters` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerModelParameterColumns<'a> {
    pub name_string_id: &'a [u32],
    pub kind_string_id: &'a [u32],
    pub default_value: &'a [f64],
    pub default_factor_id: &'a [u32],
    pub runtime_name_string_id: &'a [u32],
    pub complex_component: &'a [i32],
    pub derived: &'a [u8],
}

/// Borrowed columns from the `contraction_coefficients` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerContractionCoefficientColumns<'a> {
    pub contraction_ir_id: &'a [u32],
    pub component_index: &'a [u32],
    pub factor_id: &'a [u32],
}

/// Borrowed columns from the `coherent_groups` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerCoherentGroupColumns<'a> {
    pub id: &'a [u32],
    pub helicity_weight_factor_id: &'a [u32],
    pub all_sector_weight_factor_id: &'a [u32],
}

/// Borrowed columns from the `helicity_selectors` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerHelicitySelectorColumns<'a> {
    pub values_sequence_id: &'a [u32],
    pub representative_sequence_id: &'a [u32],
    pub coefficient_factor_id: &'a [u32],
    pub computed: &'a [u8],
    pub structural_zero: &'a [u8],
}

/// Borrowed columns from the `color_selectors` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerColorSelectorColumns<'a> {
    pub word_sequence_id: &'a [u32],
    pub representative_word_sequence_id: &'a [u32],
    pub coefficient_factor_id: &'a [u32],
    pub computed: &'a [u8],
}

/// Borrowed columns from the `reduction_members` table.
#[derive(Clone, Copy, Debug)]
pub struct EagerReductionMemberColumns<'a> {
    pub coherent_group_id: &'a [u32],
    pub helicity_selector_id: &'a [u32],
    pub color_selector_id: &'a [u32],
}

/// Scalar row from `color_contraction_metadata`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerColorContractionMetadata {
    pub present: u8,
    pub supported: u8,
    pub group_count: u64,
    pub includes_color_factor: u8,
}

/// Borrowed columns from `color_contraction_entries`.
#[derive(Clone, Copy, Debug)]
pub struct EagerColorContractionEntryColumns<'a> {
    pub left_group_id: &'a [u32],
    pub right_group_id: &'a [u32],
    pub weight_factor_id: &'a [u32],
    pub symmetry_factor_id: &'a [u32],
}

/// Complete borrowed core view of `pyamplicol-eager-lowering-input-v1`.
#[derive(Clone, Copy, Debug)]
pub struct EagerLoweringInputV1View<'a> {
    pub abi: &'a str,
    pub process_key: &'a str,
    pub model_name: &'a str,
    pub string_catalog: &'a [&'a str],
    pub canonical_ir_catalog: &'a [&'a str],
    pub semantic_limitations: &'a [&'a str],
    /// Exact v1 table inventory with unconsumed columns retained by the writer.
    pub retained_tables: &'a [EagerRetainedTableView<'a>],
    /// Tables the binding cannot represent. Any entry makes construction fail.
    pub unsupported_tables: &'a [&'a str],
    pub currents: EagerCurrentColumns<'a>,
    pub sources: EagerSourceColumns<'a>,
    pub interactions: EagerInteractionColumns<'a>,
    pub roots: EagerRootColumns<'a>,
    pub interaction_groups: EagerInteractionGroupColumns<'a>,
    pub momentum_masks: EagerMomentumMaskColumns<'a>,
    pub bitsets: EagerBitsetColumns<'a>,
    pub u32_sequences: EagerU32SequenceColumns<'a>,
    pub i32_sequences: EagerI32SequenceColumns<'a>,
    pub exact_factors: EagerExactFactorColumns<'a>,
    pub couplings: EagerCouplingColumns<'a>,
    pub model_parameters: EagerModelParameterColumns<'a>,
    pub contraction_coefficients: EagerContractionCoefficientColumns<'a>,
    pub coherent_groups: EagerCoherentGroupColumns<'a>,
    pub helicity_selectors: EagerHelicitySelectorColumns<'a>,
    pub color_selectors: EagerColorSelectorColumns<'a>,
    pub reduction_members: EagerReductionMemberColumns<'a>,
    pub color_contraction: EagerColorContractionMetadata,
    pub color_contraction_entries: EagerColorContractionEntryColumns<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputCurrent {
    id: u32,
    dimension: u32,
    is_source: bool,
    momentum_mask_bitset_id: u32,
    propagator_ir_id: u32,
    propagator_kernel_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputSource {
    source_id: u32,
    current_id: u32,
    external_label: u32,
    input_momentum_slot: u32,
    source_ir_id: u32,
    crossing_ir_id: u32,
    crossing_factor_id: u32,
    declared_state_index: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputInteraction {
    id: u32,
    stage_subset_size: u32,
    left_current_id: u32,
    right_current_id: u32,
    result_current_id: u32,
    coupling_id: u32,
    coupling_factor_id: u32,
    color_factor_id: u32,
    evaluation_factor_id: u32,
    evaluation_group_id: u32,
    kernel_id: u32,
    canonical_input_order: [u8; 2],
    kernel_normalization_factor_id: u32,
    output_factor_source: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputRoot {
    id: u32,
    left_current_id: u32,
    right_current_id: u32,
    color_factor_id: u32,
    contraction_ir_id: u32,
    coupling_id: u32,
    coupling_factor_id: u32,
    helicity_weight_factor_id: u32,
    coherent_group_id: u32,
    kernel_id: u32,
    canonical_input_order: [u8; 2],
    kernel_normalization_factor_id: u32,
    output_factor_source: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputRange {
    start: u64,
    count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputInteractionGroup {
    representative_interaction_id: u32,
    members: InputRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputBitsetRange {
    range: InputRange,
    bit_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputExactFactor {
    real_bits: u64,
    imaginary_bits: u64,
    canonical_string_id: u32,
    exact_source: u8,
    exact_ir_id: u32,
    source_ir_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputCoupling {
    constant_factor_id: u32,
    parameter_name_ids_sequence_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputModelParameter {
    name_string_id: u32,
    kind_string_id: u32,
    default_bits: u64,
    default_factor_id: u32,
    runtime_name_string_id: u32,
    complex_component: i32,
    derived: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputContractionCoefficient {
    contraction_ir_id: u32,
    component_index: u32,
    factor_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputCoherentGroup {
    id: u32,
    helicity_weight_factor_id: u32,
    all_sector_weight_factor_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputHelicitySelector {
    values_sequence_id: u32,
    representative_sequence_id: u32,
    coefficient_factor_id: u32,
    computed: bool,
    structural_zero: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputColorSelector {
    word_sequence_id: u32,
    representative_word_sequence_id: u32,
    coefficient_factor_id: u32,
    computed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputReductionMember {
    coherent_group_id: u32,
    helicity_selector_id: u32,
    color_selector_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputColorContractionEntry {
    left_group_id: u32,
    right_group_id: u32,
    weight_factor_id: u32,
    symmetry_factor_id: u32,
}

/// Owned, fully validated core representation of the v1 lowering input.
///
/// Fields are private so an instance cannot be put back into an invalid state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerLoweringInputV1 {
    process_key: String,
    model_name: String,
    string_catalog: Vec<Box<str>>,
    canonical_ir_catalog: Vec<Box<str>>,
    semantic_limitations: Vec<Box<str>>,
    retained_tables: Vec<EagerOwnedRetainedTable>,
    string_count: u32,
    canonical_ir_count: u32,
    currents: Vec<InputCurrent>,
    sources: Vec<InputSource>,
    interactions: Vec<InputInteraction>,
    roots: Vec<InputRoot>,
    interaction_groups: Vec<InputInteractionGroup>,
    interaction_group_members: Vec<u32>,
    momentum_masks: Vec<(u32, u32)>,
    bitset_ranges: Vec<InputBitsetRange>,
    bitset_words: Vec<u64>,
    u32_sequence_ranges: Vec<InputRange>,
    u32_sequence_values: Vec<u32>,
    i32_sequence_ranges: Vec<InputRange>,
    i32_sequence_values: Vec<i32>,
    exact_factors: Vec<InputExactFactor>,
    couplings: Vec<InputCoupling>,
    model_parameters: Vec<InputModelParameter>,
    contraction_coefficients: Vec<InputContractionCoefficient>,
    coherent_groups: Vec<InputCoherentGroup>,
    helicity_selectors: Vec<InputHelicitySelector>,
    color_selectors: Vec<InputColorSelector>,
    reduction_members: Vec<InputReductionMember>,
    color_contraction: EagerColorContractionMetadata,
    color_contraction_entries: Vec<InputColorContractionEntry>,
}

impl EagerLoweringInputV1 {
    /// Validate borrowed columns and copy them into a compact owned form.
    ///
    /// Bindings should construct the view directly over borrowed NumPy columns
    /// while holding the GIL, then call this method exactly once. The returned
    /// input owns every payload and may be lowered after releasing the GIL.
    pub fn try_from_view(view: EagerLoweringInputV1View<'_>) -> RusticolResult<Self> {
        if view.abi != EAGER_LOWERING_INPUT_ABI {
            return Err(invalid(format!(
                "unsupported eager lowering input ABI {:?}",
                view.abi
            )));
        }
        if view.process_key.is_empty() {
            return Err(invalid("eager lowering process key must not be empty"));
        }
        if view.model_name.is_empty() {
            return Err(invalid("eager lowering model name must not be empty"));
        }
        if view.semantic_limitations.is_empty() {
            return Err(invalid(
                "eager lowering exact-factor limitations must not be empty",
            ));
        }
        if !view.unsupported_tables.is_empty() {
            return Err(invalid(format!(
                "eager lowering input contains unsupported tables: {}",
                view.unsupported_tables.join(", ")
            )));
        }
        let string_catalog = copy_string_catalog(view.string_catalog, "string catalog")?;
        let canonical_ir_catalog =
            copy_string_catalog(view.canonical_ir_catalog, "canonical IR catalog")?;
        let semantic_limitations =
            copy_string_list(view.semantic_limitations, "semantic limitations")?;
        let retained_tables = copy_retained_tables(view.retained_tables)?;
        let string_count = usize_u32(string_catalog.len(), "string catalog count")?;
        let canonical_ir_count =
            usize_u32(canonical_ir_catalog.len(), "canonical IR catalog count")?;

        let current_count = equal_column_lengths(
            "currents",
            &[
                ("id", view.currents.id.len()),
                ("dimension", view.currents.dimension.len()),
                ("is_source", view.currents.is_source.len()),
                (
                    "momentum_mask_bitset_id",
                    view.currents.momentum_mask_bitset_id.len(),
                ),
                ("propagator_ir_id", view.currents.propagator_ir_id.len()),
                (
                    "propagator_kernel_id",
                    view.currents.propagator_kernel_id.len(),
                ),
            ],
        )?;
        let mut currents = reserved(current_count, "currents")?;
        for row in 0..current_count {
            currents.push(InputCurrent {
                id: view.currents.id[row],
                dimension: view.currents.dimension[row],
                is_source: bool_column(view.currents.is_source[row], "currents.is_source", row)?,
                momentum_mask_bitset_id: view.currents.momentum_mask_bitset_id[row],
                propagator_ir_id: view.currents.propagator_ir_id[row],
                propagator_kernel_id: view.currents.propagator_kernel_id[row],
            });
        }

        let source_count = equal_column_lengths(
            "sources",
            &[
                ("source_id", view.sources.source_id.len()),
                ("current_id", view.sources.current_id.len()),
                ("external_label", view.sources.external_label.len()),
                (
                    "input_momentum_slot",
                    view.sources.input_momentum_slot.len(),
                ),
                ("source_ir_id", view.sources.source_ir_id.len()),
                ("crossing_ir_id", view.sources.crossing_ir_id.len()),
                ("crossing_factor_id", view.sources.crossing_factor_id.len()),
                (
                    "declared_state_index",
                    view.sources.declared_state_index.len(),
                ),
            ],
        )?;
        let mut sources = reserved(source_count, "sources")?;
        for row in 0..source_count {
            sources.push(InputSource {
                source_id: view.sources.source_id[row],
                current_id: view.sources.current_id[row],
                external_label: view.sources.external_label[row],
                input_momentum_slot: view.sources.input_momentum_slot[row],
                source_ir_id: view.sources.source_ir_id[row],
                crossing_ir_id: view.sources.crossing_ir_id[row],
                crossing_factor_id: view.sources.crossing_factor_id[row],
                declared_state_index: view.sources.declared_state_index[row],
            });
        }

        let interaction_count = equal_column_lengths(
            "interactions",
            &[
                ("id", view.interactions.id.len()),
                (
                    "stage_subset_size",
                    view.interactions.stage_subset_size.len(),
                ),
                ("left_current_id", view.interactions.left_current_id.len()),
                ("right_current_id", view.interactions.right_current_id.len()),
                (
                    "result_current_id",
                    view.interactions.result_current_id.len(),
                ),
                ("coupling_id", view.interactions.coupling_id.len()),
                (
                    "coupling_factor_id",
                    view.interactions.coupling_factor_id.len(),
                ),
                ("color_factor_id", view.interactions.color_factor_id.len()),
                (
                    "evaluation_factor_id",
                    view.interactions.evaluation_factor_id.len(),
                ),
                (
                    "evaluation_group_id",
                    view.interactions.evaluation_group_id.len(),
                ),
                (
                    "full_tensor_network_ready",
                    view.interactions.full_tensor_network_ready.len(),
                ),
                ("kernel_id", view.interactions.kernel_id.len()),
                (
                    "canonical_input_order",
                    view.interactions.canonical_input_order.len(),
                ),
                (
                    "kernel_normalization_factor_id",
                    view.interactions.kernel_normalization_factor_id.len(),
                ),
                (
                    "output_factor_source",
                    view.interactions.output_factor_source.len(),
                ),
            ],
        )?;
        let mut interactions = reserved(interaction_count, "interactions")?;
        for row in 0..interaction_count {
            if !bool_column(
                view.interactions.full_tensor_network_ready[row],
                "interactions.full_tensor_network_ready",
                row,
            )? {
                return Err(invalid(format!(
                    "interaction {row} is not ready for tensor-network lowering"
                )));
            }
            interactions.push(InputInteraction {
                id: view.interactions.id[row],
                stage_subset_size: view.interactions.stage_subset_size[row],
                left_current_id: view.interactions.left_current_id[row],
                right_current_id: view.interactions.right_current_id[row],
                result_current_id: view.interactions.result_current_id[row],
                coupling_id: view.interactions.coupling_id[row],
                coupling_factor_id: view.interactions.coupling_factor_id[row],
                color_factor_id: view.interactions.color_factor_id[row],
                evaluation_factor_id: view.interactions.evaluation_factor_id[row],
                evaluation_group_id: view.interactions.evaluation_group_id[row],
                kernel_id: view.interactions.kernel_id[row],
                canonical_input_order: view.interactions.canonical_input_order[row],
                kernel_normalization_factor_id: view.interactions.kernel_normalization_factor_id
                    [row],
                output_factor_source: view.interactions.output_factor_source[row],
            });
        }

        let root_count = equal_column_lengths(
            "roots",
            &[
                ("id", view.roots.id.len()),
                ("left_current_id", view.roots.left_current_id.len()),
                ("right_current_id", view.roots.right_current_id.len()),
                ("color_factor_id", view.roots.color_factor_id.len()),
                ("contraction_ir_id", view.roots.contraction_ir_id.len()),
                ("coupling_id", view.roots.coupling_id.len()),
                ("coupling_factor_id", view.roots.coupling_factor_id.len()),
                (
                    "helicity_weight_factor_id",
                    view.roots.helicity_weight_factor_id.len(),
                ),
                ("coherent_group_id", view.roots.coherent_group_id.len()),
                ("kernel_id", view.roots.kernel_id.len()),
                (
                    "canonical_input_order",
                    view.roots.canonical_input_order.len(),
                ),
                (
                    "kernel_normalization_factor_id",
                    view.roots.kernel_normalization_factor_id.len(),
                ),
                (
                    "output_factor_source",
                    view.roots.output_factor_source.len(),
                ),
            ],
        )?;
        let mut roots = reserved(root_count, "roots")?;
        for row in 0..root_count {
            roots.push(InputRoot {
                id: view.roots.id[row],
                left_current_id: view.roots.left_current_id[row],
                right_current_id: view.roots.right_current_id[row],
                color_factor_id: view.roots.color_factor_id[row],
                contraction_ir_id: view.roots.contraction_ir_id[row],
                coupling_id: view.roots.coupling_id[row],
                coupling_factor_id: view.roots.coupling_factor_id[row],
                helicity_weight_factor_id: view.roots.helicity_weight_factor_id[row],
                coherent_group_id: view.roots.coherent_group_id[row],
                kernel_id: view.roots.kernel_id[row],
                canonical_input_order: view.roots.canonical_input_order[row],
                kernel_normalization_factor_id: view.roots.kernel_normalization_factor_id[row],
                output_factor_source: view.roots.output_factor_source[row],
            });
        }

        let group_count = equal_column_lengths(
            "interaction_groups",
            &[
                (
                    "representative_interaction_id",
                    view.interaction_groups.representative_interaction_id.len(),
                ),
                ("member_start", view.interaction_groups.member_start.len()),
                ("member_count", view.interaction_groups.member_count.len()),
            ],
        )?;
        let mut interaction_groups = reserved(group_count, "interaction groups")?;
        for row in 0..group_count {
            interaction_groups.push(InputInteractionGroup {
                representative_interaction_id: view
                    .interaction_groups
                    .representative_interaction_id[row],
                members: InputRange {
                    start: view.interaction_groups.member_start[row],
                    count: view.interaction_groups.member_count[row],
                },
            });
        }

        let momentum_count = equal_column_lengths(
            "momentum_masks",
            &[
                ("slot_id", view.momentum_masks.slot_id.len()),
                ("bitset_id", view.momentum_masks.bitset_id.len()),
            ],
        )?;
        let mut momentum_masks = reserved(momentum_count, "momentum masks")?;
        momentum_masks.extend(
            view.momentum_masks
                .slot_id
                .iter()
                .copied()
                .zip(view.momentum_masks.bitset_id.iter().copied()),
        );

        let bitset_count = equal_column_lengths(
            "bitset_ranges",
            &[
                ("start", view.bitsets.start.len()),
                ("count", view.bitsets.count.len()),
                ("bit_count", view.bitsets.bit_count.len()),
            ],
        )?;
        let mut bitset_ranges = reserved(bitset_count, "bitset ranges")?;
        for row in 0..bitset_count {
            bitset_ranges.push(InputBitsetRange {
                range: InputRange {
                    start: view.bitsets.start[row],
                    count: view.bitsets.count[row],
                },
                bit_count: view.bitsets.bit_count[row],
            });
        }

        let u32_sequence_count = equal_column_lengths(
            "u32_sequence_ranges",
            &[
                ("start", view.u32_sequences.start.len()),
                ("count", view.u32_sequences.count.len()),
            ],
        )?;
        let mut u32_sequence_ranges = reserved(u32_sequence_count, "u32 sequences")?;
        for row in 0..u32_sequence_count {
            u32_sequence_ranges.push(InputRange {
                start: view.u32_sequences.start[row],
                count: view.u32_sequences.count[row],
            });
        }

        let i32_sequence_count = equal_column_lengths(
            "i32_sequence_ranges",
            &[
                ("start", view.i32_sequences.start.len()),
                ("count", view.i32_sequences.count.len()),
            ],
        )?;
        let mut i32_sequence_ranges = reserved(i32_sequence_count, "i32 sequences")?;
        for row in 0..i32_sequence_count {
            i32_sequence_ranges.push(InputRange {
                start: view.i32_sequences.start[row],
                count: view.i32_sequences.count[row],
            });
        }

        let factor_count = equal_column_lengths(
            "exact_factors",
            &[
                ("real", view.exact_factors.real.len()),
                ("imaginary", view.exact_factors.imaginary.len()),
                (
                    "canonical_string_id",
                    view.exact_factors.canonical_string_id.len(),
                ),
                ("exact_source", view.exact_factors.exact_source.len()),
                ("exact_ir_id", view.exact_factors.exact_ir_id.len()),
                ("source_ir_id", view.exact_factors.source_ir_id.len()),
            ],
        )?;
        let mut exact_factors = reserved(factor_count, "exact factors")?;
        for row in 0..factor_count {
            let real = view.exact_factors.real[row];
            let imaginary = view.exact_factors.imaginary[row];
            if !real.is_finite() || !imaginary.is_finite() {
                return Err(invalid(format!("exact factor {row} is not finite")));
            }
            exact_factors.push(InputExactFactor {
                real_bits: real.to_bits(),
                imaginary_bits: imaginary.to_bits(),
                canonical_string_id: view.exact_factors.canonical_string_id[row],
                exact_source: view.exact_factors.exact_source[row],
                exact_ir_id: view.exact_factors.exact_ir_id[row],
                source_ir_id: view.exact_factors.source_ir_id[row],
            });
        }

        let coupling_count = equal_column_lengths(
            "couplings",
            &[
                (
                    "constant_factor_id",
                    view.couplings.constant_factor_id.len(),
                ),
                (
                    "parameter_name_ids_sequence_id",
                    view.couplings.parameter_name_ids_sequence_id.len(),
                ),
            ],
        )?;
        let mut couplings = reserved(coupling_count, "couplings")?;
        for row in 0..coupling_count {
            couplings.push(InputCoupling {
                constant_factor_id: view.couplings.constant_factor_id[row],
                parameter_name_ids_sequence_id: view.couplings.parameter_name_ids_sequence_id[row],
            });
        }

        let parameter_count = equal_column_lengths(
            "model_parameters",
            &[
                ("name_string_id", view.model_parameters.name_string_id.len()),
                ("kind_string_id", view.model_parameters.kind_string_id.len()),
                ("default_value", view.model_parameters.default_value.len()),
                (
                    "default_factor_id",
                    view.model_parameters.default_factor_id.len(),
                ),
                (
                    "runtime_name_string_id",
                    view.model_parameters.runtime_name_string_id.len(),
                ),
                (
                    "complex_component",
                    view.model_parameters.complex_component.len(),
                ),
                ("derived", view.model_parameters.derived.len()),
            ],
        )?;
        let mut model_parameters = reserved(parameter_count, "model parameters")?;
        for row in 0..parameter_count {
            let default = view.model_parameters.default_value[row];
            if !default.is_finite() {
                return Err(invalid(format!(
                    "model parameter {row} has a non-finite default"
                )));
            }
            model_parameters.push(InputModelParameter {
                name_string_id: view.model_parameters.name_string_id[row],
                kind_string_id: view.model_parameters.kind_string_id[row],
                default_bits: default.to_bits(),
                default_factor_id: view.model_parameters.default_factor_id[row],
                runtime_name_string_id: view.model_parameters.runtime_name_string_id[row],
                complex_component: view.model_parameters.complex_component[row],
                derived: bool_column(
                    view.model_parameters.derived[row],
                    "model_parameters.derived",
                    row,
                )?,
            });
        }

        let coefficient_count = equal_column_lengths(
            "contraction_coefficients",
            &[
                (
                    "contraction_ir_id",
                    view.contraction_coefficients.contraction_ir_id.len(),
                ),
                (
                    "component_index",
                    view.contraction_coefficients.component_index.len(),
                ),
                ("factor_id", view.contraction_coefficients.factor_id.len()),
            ],
        )?;
        let mut contraction_coefficients = reserved(coefficient_count, "contraction coefficients")?;
        for row in 0..coefficient_count {
            contraction_coefficients.push(InputContractionCoefficient {
                contraction_ir_id: view.contraction_coefficients.contraction_ir_id[row],
                component_index: view.contraction_coefficients.component_index[row],
                factor_id: view.contraction_coefficients.factor_id[row],
            });
        }

        let coherent_count = equal_column_lengths(
            "coherent_groups",
            &[
                ("id", view.coherent_groups.id.len()),
                (
                    "helicity_weight_factor_id",
                    view.coherent_groups.helicity_weight_factor_id.len(),
                ),
                (
                    "all_sector_weight_factor_id",
                    view.coherent_groups.all_sector_weight_factor_id.len(),
                ),
            ],
        )?;
        let mut coherent_groups = reserved(coherent_count, "coherent groups")?;
        for row in 0..coherent_count {
            coherent_groups.push(InputCoherentGroup {
                id: view.coherent_groups.id[row],
                helicity_weight_factor_id: view.coherent_groups.helicity_weight_factor_id[row],
                all_sector_weight_factor_id: view.coherent_groups.all_sector_weight_factor_id[row],
            });
        }

        let helicity_selector_count = equal_column_lengths(
            "helicity_selectors",
            &[
                (
                    "values_sequence_id",
                    view.helicity_selectors.values_sequence_id.len(),
                ),
                (
                    "representative_sequence_id",
                    view.helicity_selectors.representative_sequence_id.len(),
                ),
                (
                    "coefficient_factor_id",
                    view.helicity_selectors.coefficient_factor_id.len(),
                ),
                ("computed", view.helicity_selectors.computed.len()),
                (
                    "structural_zero",
                    view.helicity_selectors.structural_zero.len(),
                ),
            ],
        )?;
        let mut helicity_selectors = reserved(helicity_selector_count, "helicity selectors")?;
        for row in 0..helicity_selector_count {
            helicity_selectors.push(InputHelicitySelector {
                values_sequence_id: view.helicity_selectors.values_sequence_id[row],
                representative_sequence_id: view.helicity_selectors.representative_sequence_id[row],
                coefficient_factor_id: view.helicity_selectors.coefficient_factor_id[row],
                computed: bool_column(
                    view.helicity_selectors.computed[row],
                    "helicity_selectors.computed",
                    row,
                )?,
                structural_zero: bool_column(
                    view.helicity_selectors.structural_zero[row],
                    "helicity_selectors.structural_zero",
                    row,
                )?,
            });
        }

        let color_selector_count = equal_column_lengths(
            "color_selectors",
            &[
                (
                    "word_sequence_id",
                    view.color_selectors.word_sequence_id.len(),
                ),
                (
                    "representative_word_sequence_id",
                    view.color_selectors.representative_word_sequence_id.len(),
                ),
                (
                    "coefficient_factor_id",
                    view.color_selectors.coefficient_factor_id.len(),
                ),
                ("computed", view.color_selectors.computed.len()),
            ],
        )?;
        let mut color_selectors = reserved(color_selector_count, "color selectors")?;
        for row in 0..color_selector_count {
            color_selectors.push(InputColorSelector {
                word_sequence_id: view.color_selectors.word_sequence_id[row],
                representative_word_sequence_id: view
                    .color_selectors
                    .representative_word_sequence_id[row],
                coefficient_factor_id: view.color_selectors.coefficient_factor_id[row],
                computed: bool_column(
                    view.color_selectors.computed[row],
                    "color_selectors.computed",
                    row,
                )?,
            });
        }

        let reduction_count = equal_column_lengths(
            "reduction_members",
            &[
                (
                    "coherent_group_id",
                    view.reduction_members.coherent_group_id.len(),
                ),
                (
                    "helicity_selector_id",
                    view.reduction_members.helicity_selector_id.len(),
                ),
                (
                    "color_selector_id",
                    view.reduction_members.color_selector_id.len(),
                ),
            ],
        )?;
        let mut reduction_members = reserved(reduction_count, "reduction members")?;
        for row in 0..reduction_count {
            reduction_members.push(InputReductionMember {
                coherent_group_id: view.reduction_members.coherent_group_id[row],
                helicity_selector_id: view.reduction_members.helicity_selector_id[row],
                color_selector_id: view.reduction_members.color_selector_id[row],
            });
        }

        let contraction_entry_count = equal_column_lengths(
            "color_contraction_entries",
            &[
                (
                    "left_group_id",
                    view.color_contraction_entries.left_group_id.len(),
                ),
                (
                    "right_group_id",
                    view.color_contraction_entries.right_group_id.len(),
                ),
                (
                    "weight_factor_id",
                    view.color_contraction_entries.weight_factor_id.len(),
                ),
                (
                    "symmetry_factor_id",
                    view.color_contraction_entries.symmetry_factor_id.len(),
                ),
            ],
        )?;
        let mut color_contraction_entries =
            reserved(contraction_entry_count, "color contraction entries")?;
        for row in 0..contraction_entry_count {
            color_contraction_entries.push(InputColorContractionEntry {
                left_group_id: view.color_contraction_entries.left_group_id[row],
                right_group_id: view.color_contraction_entries.right_group_id[row],
                weight_factor_id: view.color_contraction_entries.weight_factor_id[row],
                symmetry_factor_id: view.color_contraction_entries.symmetry_factor_id[row],
            });
        }

        let input = Self {
            process_key: view.process_key.to_owned(),
            model_name: view.model_name.to_owned(),
            string_catalog,
            canonical_ir_catalog,
            semantic_limitations,
            retained_tables,
            string_count,
            canonical_ir_count,
            currents,
            sources,
            interactions,
            roots,
            interaction_groups,
            interaction_group_members: copy_slice(
                view.interaction_groups.member_interaction_id,
                "interaction group members",
            )?,
            momentum_masks,
            bitset_ranges,
            bitset_words: copy_slice(view.bitsets.words, "bitset words")?,
            u32_sequence_ranges,
            u32_sequence_values: copy_slice(view.u32_sequences.values, "u32 sequence values")?,
            i32_sequence_ranges,
            i32_sequence_values: copy_slice(view.i32_sequences.values, "i32 sequence values")?,
            exact_factors,
            couplings,
            model_parameters,
            contraction_coefficients,
            coherent_groups,
            helicity_selectors,
            color_selectors,
            reduction_members,
            color_contraction: view.color_contraction,
            color_contraction_entries,
        };
        input.validate()?;
        Ok(input)
    }

    pub fn process_key(&self) -> &str {
        &self.process_key
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn string_catalog(&self) -> &[Box<str>] {
        &self.string_catalog
    }

    pub fn canonical_ir_catalog(&self) -> &[Box<str>] {
        &self.canonical_ir_catalog
    }

    pub fn semantic_limitations(&self) -> &[Box<str>] {
        &self.semantic_limitations
    }

    pub fn retained_tables(&self) -> &[EagerOwnedRetainedTable] {
        &self.retained_tables
    }

    pub fn current_count(&self) -> usize {
        self.currents.len()
    }

    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }

    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    pub fn exact_factor_count(&self) -> usize {
        self.exact_factors.len()
    }

    fn validate(&self) -> RusticolResult<()> {
        validate_retained_table_counts(self)?;
        if self.currents.is_empty() {
            return Err(invalid("eager lowering input has no currents"));
        }
        if self.roots.is_empty() {
            return Err(invalid("eager lowering input has no amplitude roots"));
        }
        validate_dense_ids(
            self.currents.iter().map(|row| row.id),
            self.currents.len(),
            "current",
        )?;
        validate_dense_ids(
            self.interactions.iter().map(|row| row.id),
            self.interactions.len(),
            "interaction",
        )?;
        validate_dense_ids(
            self.roots.iter().map(|row| row.id),
            self.roots.len(),
            "root",
        )?;
        validate_dense_ids(
            self.sources.iter().map(|row| row.source_id),
            self.sources.len(),
            "source",
        )?;
        validate_dense_ids(
            self.coherent_groups.iter().map(|row| row.id),
            self.coherent_groups.len(),
            "coherent group",
        )?;

        validate_flat_ranges(
            self.bitset_ranges.iter().map(|row| row.range),
            self.bitset_words.len(),
            "bitset catalog",
        )?;
        for (bitset_id, row) in self.bitset_ranges.iter().enumerate() {
            let words = range_slice(&self.bitset_words, row.range, "bitset words")?;
            let actual_count = words.iter().try_fold(0_u64, |total, word| {
                total
                    .checked_add(u64::from(word.count_ones()))
                    .ok_or_else(|| {
                        invalid(format!("bitset {bitset_id} population count exceeds u64"))
                    })
            })?;
            if actual_count != row.bit_count {
                return Err(invalid(format!(
                    "bitset {bitset_id} declares {} set bits, found {actual_count}",
                    row.bit_count
                )));
            }
            if words.last().is_some_and(|word| *word == 0) {
                return Err(invalid(format!(
                    "bitset {bitset_id} has a noncanonical trailing zero word"
                )));
            }
        }
        validate_flat_ranges(
            self.u32_sequence_ranges.iter().copied(),
            self.u32_sequence_values.len(),
            "u32 sequence catalog",
        )?;
        validate_flat_ranges(
            self.i32_sequence_ranges.iter().copied(),
            self.i32_sequence_values.len(),
            "i32 sequence catalog",
        )?;

        let mut previous_momentum_bitset = None;
        let mut momentum_by_bitset = vec![MISSING_U32; self.bitset_ranges.len()];
        for (row, (slot_id, bitset_id)) in self.momentum_masks.iter().copied().enumerate() {
            expect_dense_id(slot_id, row, "momentum slot")?;
            let bitset_index = index(bitset_id, self.bitset_ranges.len(), "momentum bitset")?;
            if momentum_by_bitset[bitset_index] != MISSING_U32 {
                return Err(invalid(format!(
                    "momentum bitset {bitset_id} is assigned more than once"
                )));
            }
            if let Some(previous) = previous_momentum_bitset
                && compare_bitsets(self, previous, bitset_id)? != Ordering::Less
            {
                return Err(invalid(
                    "momentum masks must be unique and ordered by population then value",
                ));
            }
            previous_momentum_bitset = Some(bitset_id);
            momentum_by_bitset[bitset_index] = slot_id;
        }

        let mut source_seen = vec![false; self.currents.len()];
        for source in &self.sources {
            let current_index = index(source.current_id, self.currents.len(), "source current")?;
            if !self.currents[current_index].is_source {
                return Err(invalid(format!(
                    "source {} references non-source current {}",
                    source.source_id, source.current_id
                )));
            }
            if std::mem::replace(&mut source_seen[current_index], true) {
                return Err(invalid(format!(
                    "source current {} is listed more than once",
                    source.current_id
                )));
            }
            required_index(source.source_ir_id, self.canonical_ir_count, "source IR")?;
            required_index(
                source.crossing_ir_id,
                self.canonical_ir_count,
                "source crossing IR",
            )?;
            required_index(
                source.crossing_factor_id,
                self.exact_factors.len(),
                "source crossing factor",
            )?;
        }
        for (current_id, current) in self.currents.iter().enumerate() {
            if current.dimension == 0 {
                return Err(invalid(format!("current {current_id} has zero dimension")));
            }
            required_index(
                current.momentum_mask_bitset_id,
                self.bitset_ranges.len(),
                "current momentum bitset",
            )?;
            if momentum_by_bitset[current.momentum_mask_bitset_id as usize] == MISSING_U32 {
                return Err(invalid(format!(
                    "current {current_id} momentum bitset {} has no momentum slot",
                    current.momentum_mask_bitset_id
                )));
            }
            required_index(
                current.propagator_ir_id,
                self.canonical_ir_count,
                "current propagator IR",
            )?;
            if current.is_source != source_seen[current_id] {
                return Err(invalid(format!(
                    "current {current_id} source flag disagrees with the source table"
                )));
            }
        }

        for (factor_id, factor) in self.exact_factors.iter().enumerate() {
            required_index(
                factor.canonical_string_id,
                self.string_count,
                "exact factor canonical string",
            )?;
            required_index(
                factor.exact_ir_id,
                self.canonical_ir_count,
                "exact factor IR",
            )?;
            optional_index(
                factor.source_ir_id,
                self.canonical_ir_count,
                "exact factor source IR",
            )?;
            if !matches!(
                factor.exact_source,
                EXACT_SOURCE_BINARY64 | EXACT_SOURCE_CANONICAL_IR
            ) {
                return Err(invalid(format!(
                    "exact factor {factor_id} has unsupported provenance {}",
                    factor.exact_source
                )));
            }
        }

        for (parameter_id, parameter) in self.model_parameters.iter().enumerate() {
            required_index(
                parameter.name_string_id,
                self.string_count,
                "parameter name",
            )?;
            required_index(
                parameter.kind_string_id,
                self.string_count,
                "parameter kind",
            )?;
            optional_index(
                parameter.runtime_name_string_id,
                self.string_count,
                "parameter runtime name",
            )?;
            let factor = required_row(
                &self.exact_factors,
                parameter.default_factor_id,
                "parameter default factor",
            )?;
            if factor.real_bits != parameter.default_bits || !factor_is_real_zero(factor) {
                return Err(invalid(format!(
                    "model parameter {parameter_id} default disagrees with its exact factor"
                )));
            }
            if !matches!(parameter.complex_component, -1..=1) {
                return Err(invalid(format!(
                    "model parameter {parameter_id} has invalid complex component {}",
                    parameter.complex_component
                )));
            }
            let _ = parameter.derived;
        }

        for (coupling_id, coupling) in self.couplings.iter().enumerate() {
            required_index(
                coupling.constant_factor_id,
                self.exact_factors.len(),
                "coupling constant factor",
            )?;
            let names = self.u32_sequence(coupling.parameter_name_ids_sequence_id)?;
            if names.len() > 2 {
                return Err(invalid(format!(
                    "coupling {coupling_id} has more than two parameter names"
                )));
            }
            for name in names.iter().copied().filter(|name| *name != MISSING_U32) {
                required_index(name, self.string_count, "coupling parameter name")?;
            }
        }

        let mut result_stage = vec![None; self.currents.len()];
        for (interaction_id, interaction) in self.interactions.iter().enumerate() {
            if interaction.stage_subset_size == 0 {
                return Err(invalid(format!(
                    "interaction {interaction_id} has zero stage subset size"
                )));
            }
            let left = index(
                interaction.left_current_id,
                self.currents.len(),
                "interaction left current",
            )?;
            let right = index(
                interaction.right_current_id,
                self.currents.len(),
                "interaction right current",
            )?;
            let result = index(
                interaction.result_current_id,
                self.currents.len(),
                "interaction result current",
            )?;
            if self.currents[result].is_source {
                return Err(invalid(format!(
                    "interaction {interaction_id} writes source current {result}"
                )));
            }
            for input in [left, right] {
                if !self.currents[input].is_source
                    && result_stage[input]
                        .is_some_and(|stage| stage >= interaction.stage_subset_size)
                {
                    return Err(invalid(format!(
                        "interaction {interaction_id} does not follow DAG stage order"
                    )));
                }
            }
            match result_stage[result] {
                Some(stage) if stage != interaction.stage_subset_size => {
                    return Err(invalid(format!(
                        "current {result} is produced in multiple stages"
                    )));
                }
                None => result_stage[result] = Some(interaction.stage_subset_size),
                _ => {}
            }
            required_index(
                interaction.coupling_id,
                self.couplings.len(),
                "interaction coupling",
            )?;
            for (name, factor_id) in [
                ("coupling", interaction.coupling_factor_id),
                ("color", interaction.color_factor_id),
                ("evaluation", interaction.evaluation_factor_id),
                (
                    "kernel normalization",
                    interaction.kernel_normalization_factor_id,
                ),
            ] {
                required_index(factor_id, self.exact_factors.len(), name)?;
            }
            if interaction.coupling_factor_id
                != self.couplings[interaction.coupling_id as usize].constant_factor_id
            {
                return Err(invalid(format!(
                    "interaction {interaction_id} coupling factor disagrees with coupling {}",
                    interaction.coupling_id
                )));
            }
            validate_kernel_transform(
                interaction.kernel_id,
                interaction.canonical_input_order,
                interaction.output_factor_source,
                false,
                &format!("interaction {interaction_id}"),
            )?;
        }

        for (current_id, current) in self.currents.iter().enumerate() {
            if !current.is_source && result_stage[current_id].is_none() {
                return Err(invalid(format!(
                    "non-source current {current_id} has no producing interaction"
                )));
            }
        }

        validate_flat_ranges(
            self.interaction_groups.iter().map(|row| row.members),
            self.interaction_group_members.len(),
            "interaction groups",
        )?;
        if self.interaction_group_members.len() != self.interactions.len() {
            return Err(invalid(
                "interaction groups must contain every interaction exactly once",
            ));
        }
        let mut grouped = vec![false; self.interactions.len()];
        for (group_id, group) in self.interaction_groups.iter().enumerate() {
            if group.members.count == 0 {
                return Err(invalid(format!("interaction group {group_id} is empty")));
            }
            let members = range_slice(
                &self.interaction_group_members,
                group.members,
                "interaction group members",
            )?;
            if members[0] != group.representative_interaction_id {
                return Err(invalid(format!(
                    "interaction group {group_id} representative is not its first member"
                )));
            }
            let representative = required_row(
                &self.interactions,
                group.representative_interaction_id,
                "interaction group representative",
            )?;
            if representative.evaluation_group_id != group_id as u32 {
                return Err(invalid(format!(
                    "interaction group {group_id} representative has a different group ID"
                )));
            }
            let representative_is_zero = factor_is_complex_zero(
                &self.exact_factors[representative.evaluation_factor_id as usize],
            );
            let mut group_has_nonzero_member = false;
            for member_id in members {
                let member_index = index(*member_id, self.interactions.len(), "group member")?;
                if std::mem::replace(&mut grouped[member_index], true) {
                    return Err(invalid(format!(
                        "interaction {member_id} appears in more than one group"
                    )));
                }
                let member = &self.interactions[member_index];
                group_has_nonzero_member |= !factor_is_complex_zero(
                    &self.exact_factors[member.evaluation_factor_id as usize],
                );
                if member.evaluation_group_id != group_id as u32 {
                    return Err(invalid(format!(
                        "interaction {member_id} has group {}, expected {group_id}",
                        member.evaluation_group_id
                    )));
                }
                validate_group_signature(representative, member, group_id)?;
            }
            if representative_is_zero && group_has_nonzero_member {
                return Err(invalid(format!(
                    "interaction group {group_id} places a zero evaluation factor \
                     before a live member"
                )));
            }
        }
        if grouped.iter().any(|seen| !seen) {
            return Err(invalid("interaction group partition is incomplete"));
        }

        for (root_id, root) in self.roots.iter().enumerate() {
            required_index(
                root.left_current_id,
                self.currents.len(),
                "root left current",
            )?;
            required_index(
                root.right_current_id,
                self.currents.len(),
                "root right current",
            )?;
            required_index(
                root.color_factor_id,
                self.exact_factors.len(),
                "root color factor",
            )?;
            required_index(
                root.contraction_ir_id,
                self.canonical_ir_count,
                "root contraction IR",
            )?;
            required_index(
                root.coupling_factor_id,
                self.exact_factors.len(),
                "root coupling factor",
            )?;
            required_index(
                root.helicity_weight_factor_id,
                self.exact_factors.len(),
                "root helicity weight",
            )?;
            required_index(
                root.kernel_normalization_factor_id,
                self.exact_factors.len(),
                "root kernel normalization",
            )?;
            required_index(
                root.coherent_group_id,
                self.coherent_groups.len(),
                "root coherent group",
            )?;
            if root.coupling_id != MISSING_U32 {
                let coupling = required_row(&self.couplings, root.coupling_id, "root coupling")?;
                if coupling.constant_factor_id != root.coupling_factor_id {
                    return Err(invalid(format!(
                        "root {root_id} coupling factor disagrees with coupling {}",
                        root.coupling_id
                    )));
                }
            }
            validate_kernel_transform(
                root.kernel_id,
                root.canonical_input_order,
                root.output_factor_source,
                true,
                &format!("root {root_id}"),
            )?;
        }

        let mut previous_coefficient: Option<(u32, u32)> = None;
        for (row, coefficient) in self.contraction_coefficients.iter().enumerate() {
            required_index(
                coefficient.contraction_ir_id,
                self.canonical_ir_count,
                "contraction coefficient IR",
            )?;
            required_index(
                coefficient.factor_id,
                self.exact_factors.len(),
                "contraction coefficient factor",
            )?;
            match previous_coefficient {
                Some((previous_ir, _)) if coefficient.contraction_ir_id < previous_ir => {
                    return Err(invalid(format!(
                        "contraction coefficient row {row} is not canonically ordered"
                    )));
                }
                Some((previous_ir, previous_component))
                    if coefficient.contraction_ir_id == previous_ir =>
                {
                    let expected_component =
                        previous_component.checked_add(1).ok_or_else(|| {
                            invalid(format!(
                                "contraction IR {} component index exceeds u32",
                                coefficient.contraction_ir_id
                            ))
                        })?;
                    if coefficient.component_index != expected_component {
                        return Err(invalid(format!(
                            "contraction coefficient row {row} is not canonically ordered"
                        )));
                    }
                }
                _ if coefficient.component_index != 0 => {
                    return Err(invalid(format!(
                        "contraction IR {} starts at component {}",
                        coefficient.contraction_ir_id, coefficient.component_index
                    )));
                }
                _ => {}
            }
            previous_coefficient =
                Some((coefficient.contraction_ir_id, coefficient.component_index));
        }

        for group in &self.coherent_groups {
            required_index(
                group.helicity_weight_factor_id,
                self.exact_factors.len(),
                "coherent-group helicity weight",
            )?;
            required_index(
                group.all_sector_weight_factor_id,
                self.exact_factors.len(),
                "coherent-group all-sector weight",
            )?;
        }
        for (selector_id, selector) in self.helicity_selectors.iter().enumerate() {
            let values = self.i32_sequence(selector.values_sequence_id)?;
            let representative = self.i32_sequence(selector.representative_sequence_id)?;
            if values.len() != representative.len() {
                return Err(invalid(format!(
                    "helicity selector {selector_id} alias has a different width"
                )));
            }
            required_index(
                selector.coefficient_factor_id,
                self.exact_factors.len(),
                "helicity selector coefficient",
            )?;
            if selector.computed && selector.structural_zero {
                return Err(invalid(format!(
                    "helicity selector {selector_id} is both computed and structural zero"
                )));
            }
            if selector.structural_zero
                && !factor_is_complex_zero(
                    &self.exact_factors[selector.coefficient_factor_id as usize],
                )
            {
                return Err(invalid(format!(
                    "structural-zero helicity selector {selector_id} has a nonzero coefficient"
                )));
            }
        }
        for (selector_id, selector) in self.color_selectors.iter().enumerate() {
            let values = self.u32_sequence(selector.word_sequence_id)?;
            let representative = self.u32_sequence(selector.representative_word_sequence_id)?;
            if values.len() != representative.len() {
                return Err(invalid(format!(
                    "color selector {selector_id} alias has a different width"
                )));
            }
            required_index(
                selector.coefficient_factor_id,
                self.exact_factors.len(),
                "color selector coefficient",
            )?;
        }
        let mut reduction_keys = BTreeSet::new();
        let mut group_has_reduction = vec![false; self.coherent_groups.len()];
        for member in &self.reduction_members {
            let group = index(
                member.coherent_group_id,
                self.coherent_groups.len(),
                "reduction coherent group",
            )?;
            required_index(
                member.helicity_selector_id,
                self.helicity_selectors.len(),
                "reduction helicity selector",
            )?;
            required_index(
                member.color_selector_id,
                self.color_selectors.len(),
                "reduction color selector",
            )?;
            if !reduction_keys.insert((
                member.coherent_group_id,
                member.helicity_selector_id,
                member.color_selector_id,
            )) {
                return Err(invalid("duplicate selector reduction member"));
            }
            group_has_reduction[group] = true;
        }
        if group_has_reduction.iter().any(|present| !present) {
            return Err(invalid("every coherent group must have reduction metadata"));
        }

        let contraction = self.color_contraction;
        validate_bool_scalar(contraction.present, "color contraction present")?;
        validate_bool_scalar(contraction.supported, "color contraction supported")?;
        validate_bool_scalar(
            contraction.includes_color_factor,
            "color contraction includes-color-factor",
        )?;
        if contraction.present == 0 {
            if !self.color_contraction_entries.is_empty() || contraction.group_count != 0 {
                return Err(invalid(
                    "absent color contraction has entries or a nonzero group count",
                ));
            }
        } else {
            if contraction.supported == 0 {
                return Err(invalid("unsupported color contraction cannot be lowered"));
            }
            if contraction.group_count != self.coherent_groups.len() as u64 {
                return Err(invalid(
                    "color contraction group count disagrees with coherent groups",
                ));
            }
            if self.color_contraction_entries.is_empty() {
                return Err(invalid("present color contraction has no entries"));
            }
        }
        for entry in &self.color_contraction_entries {
            required_index(
                entry.left_group_id,
                self.coherent_groups.len(),
                "color contraction left group",
            )?;
            required_index(
                entry.right_group_id,
                self.coherent_groups.len(),
                "color contraction right group",
            )?;
            required_index(
                entry.weight_factor_id,
                self.exact_factors.len(),
                "color contraction weight",
            )?;
            required_index(
                entry.symmetry_factor_id,
                self.exact_factors.len(),
                "color contraction symmetry",
            )?;
        }
        Ok(())
    }

    fn u32_sequence(&self, id: u32) -> RusticolResult<&[u32]> {
        let range = required_row(&self.u32_sequence_ranges, id, "u32 sequence")?;
        range_slice(&self.u32_sequence_values, *range, "u32 sequence values")
    }

    fn i32_sequence(&self, id: u32) -> RusticolResult<&[i32]> {
        let range = required_row(&self.i32_sequence_ranges, id, "i32 sequence")?;
        range_slice(&self.i32_sequence_values, *range, "i32 sequence values")
    }
}

/// Value-slot role in the compact plan.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum EagerValueSlotKind {
    Source = 1,
    Unpropagated = 2,
    Propagated = 3,
}

/// One logical current and its global component range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanCurrentRow {
    pub current_id: u32,
    pub component_start: u64,
    pub component_count: u32,
    pub momentum_slot_id: u32,
    pub flags: u32,
}

impl EagerPlanCurrentRow {
    pub const ENCODED_LEN: u32 = 24;
}

/// One value slot and its global component range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanValueRow {
    pub value_slot_id: u32,
    pub current_id: u32,
    pub component_start: u64,
    pub component_count: u32,
    pub kind: EagerValueSlotKind,
}

impl EagerPlanValueRow {
    pub const ENCODED_LEN: u32 = 24;
}

/// One canonical arbitrary-width momentum mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanMomentumRow {
    pub momentum_slot_id: u32,
    pub bitset_id: u32,
    pub component_start: u64,
    pub component_count: u32,
}

impl EagerPlanMomentumRow {
    pub const ENCODED_LEN: u32 = 24;
}

/// Source initialization metadata retained for exact execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanSourceFillRow {
    pub source_id: u32,
    pub current_id: u32,
    pub value_slot_id: u32,
    pub external_label: u32,
    pub input_momentum_slot: u32,
    pub source_ir_id: u32,
    pub crossing_ir_id: u32,
    pub crossing_factor_id: u32,
    pub declared_state_index: u32,
}

impl EagerPlanSourceFillRow {
    pub const ENCODED_LEN: u32 = 40;
}

/// Model-parameter metadata for the compact parameter layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanParameterRow {
    pub parameter_id: u32,
    pub name_string_id: u32,
    pub kind_string_id: u32,
    pub default_factor_id: u32,
    pub runtime_name_string_id: u32,
    pub complex_component: i32,
    pub flags: u32,
}

impl EagerPlanParameterRow {
    pub const ENCODED_LEN: u32 = 32;
}

/// One stage and its contiguous ranges in the execution tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanStageRow {
    pub stage_index: u32,
    pub subset_size: u32,
    pub invocation_start: u64,
    pub invocation_count: u64,
    pub attachment_start: u64,
    pub attachment_count: u64,
    pub finalization_start: u64,
    pub finalization_count: u64,
}

impl EagerPlanStageRow {
    pub const ENCODED_LEN: u32 = 56;
}

/// Runtime coupling with exact constant provenance and optional parameter slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanCouplingRow {
    pub coupling_id: u32,
    pub real_parameter_id: u32,
    pub imaginary_parameter_id: u32,
    pub constant_factor_id: u32,
}

impl EagerPlanCouplingRow {
    pub const ENCODED_LEN: u32 = 16;
}

/// One prepared vertex invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanInvocationRow {
    pub evaluation_group_id: u32,
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub left_momentum_slot_id: u32,
    pub right_momentum_slot_id: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u8,
    pub attachment_start: u64,
    pub attachment_count: u64,
    pub selector_domain_id: u32,
}

impl EagerPlanInvocationRow {
    pub const ENCODED_LEN: u32 = 56;
}

/// One invocation fan-out attachment.
///
/// Its exact factor is `color * evaluation * normalization /
/// representative_evaluation`. Keeping all four IDs avoids an irreversible
/// f64 collapse during lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanAttachmentRow {
    pub interaction_id: u32,
    pub result_current_id: u32,
    pub color_factor_id: u32,
    pub evaluation_factor_id: u32,
    pub normalization_factor_id: u32,
    pub representative_evaluation_factor_id: u32,
    pub selector_domain_id: u32,
}

impl EagerPlanAttachmentRow {
    pub const ENCODED_LEN: u32 = 28;
}

/// One current finalization and optional propagator application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanFinalizationRow {
    pub kernel_id: u32,
    pub current_id: u32,
    pub unpropagated_value_slot_id: u32,
    pub propagated_value_slot_id: u32,
    pub momentum_slot_id: u32,
    pub unpropagated_selector_domain_id: u32,
    pub propagated_selector_domain_id: u32,
}

impl EagerPlanFinalizationRow {
    pub const ENCODED_LEN: u32 = 28;
}

/// One amplitude closure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanClosureRow {
    pub root_id: u32,
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub amplitude_index: u32,
    pub coherent_group_id: u32,
    pub coupling_slot_id: u32,
    pub coupling_factor_id: u32,
    pub output_factor_source: u8,
    pub color_factor_id: u32,
    pub normalization_factor_id: u32,
    pub direct_coefficient_start: u64,
    pub direct_coefficient_count: u64,
    pub selector_domain_id: u32,
}

impl EagerPlanClosureRow {
    pub const ENCODED_LEN: u32 = 64;
}

/// Exact coefficient for one direct closure component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanDirectCoefficientRow {
    pub contraction_ir_id: u32,
    pub component_index: u32,
    pub factor_id: u32,
}

/// One interned coherent-group selector domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanSelectorDomainRow {
    pub member_start: u64,
    pub member_count: u64,
}

impl EagerPlanSelectorDomainRow {
    pub const ENCODED_LEN: u32 = 16;
}

/// Logical currents, kernel evaluations, and routes executed for one selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanActiveWorkCounts {
    pub current_count: u64,
    pub evaluation_count: u64,
    pub attachment_count: u64,
}

/// Worst exact runtime work for each report selector axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanSelectorWorkSummary {
    pub selected_flow_selector_count: u64,
    pub selected_flow_worst: EagerPlanActiveWorkCounts,
    pub all_flow_selector_count: u64,
    pub all_flow_worst: EagerPlanActiveWorkCounts,
    pub contracted: EagerPlanActiveWorkCounts,
}

/// Public helicity selector, including aliases and structural zeros.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanHelicitySelectorRow {
    pub selector_id: u32,
    pub values_sequence_id: u32,
    pub representative_sequence_id: u32,
    pub coefficient_factor_id: u32,
    pub computed: u8,
    pub structural_zero: u8,
}

/// Public color selector and representative alias.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanColorSelectorRow {
    pub selector_id: u32,
    pub word_sequence_id: u32,
    pub representative_word_sequence_id: u32,
    pub coefficient_factor_id: u32,
    pub computed: u8,
}

/// One coherent reduction group and its flat entry ranges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanReductionGroupRow {
    pub coherent_group_id: u32,
    pub amplitude_entry_start: u64,
    pub amplitude_entry_count: u64,
    pub selector_entry_start: u64,
    pub selector_entry_count: u64,
    pub helicity_weight_factor_id: u32,
    pub all_sector_weight_factor_id: u32,
}

impl EagerPlanReductionGroupRow {
    pub const ENCODED_LEN: u32 = 48;
}

/// Tagged fixed-width reduction entry kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EagerPlanReductionEntryKind {
    AmplitudeMember = 1,
    SelectorMember = 2,
    ColorContraction = 3,
}

/// Tagged fixed-width reduction entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanReductionEntryRow {
    pub kind: EagerPlanReductionEntryKind,
    pub owner_id: u32,
    pub left_id: u32,
    pub right_id: u32,
    pub factor_id: u32,
    pub auxiliary_factor_id: u32,
}

impl EagerPlanReductionEntryRow {
    pub const ENCODED_LEN: u32 = 24;
}

/// Exact-factor catalog row with bit-exact binary64 fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanExactFactorRow {
    pub factor_id: u32,
    pub real_bits: u64,
    pub imaginary_bits: u64,
    pub canonical_string_id: u32,
    pub exact_source: u8,
    pub exact_ir_id: u32,
    pub source_ir_id: u32,
}

/// Canonical range into a flat fixed-width catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanCatalogRangeRow {
    pub start: u64,
    pub count: u64,
}

impl EagerPlanCatalogRangeRow {
    pub const ENCODED_LEN: u32 = 16;
}

impl EagerPlanExactFactorRow {
    pub const ENCODED_LEN: u32 = 40;
}

/// A section shape ready to be wrapped by the common eager section header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerPlanSectionShape {
    pub kind: EagerSectionKind,
    pub record_size: u32,
    pub record_count: u64,
}

impl EagerPlanSectionShape {
    pub fn header(self) -> RusticolResult<EagerSectionHeader> {
        EagerSectionHeader::new(self.kind, self.record_size, self.record_count)
    }
}

/// Flat, deterministic plan-v3 representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerPlanV3 {
    process_key: String,
    model_name: String,
    string_catalog: Vec<Box<str>>,
    canonical_ir_catalog: Vec<Box<str>>,
    semantic_limitations: Vec<Box<str>>,
    retained_tables: Vec<EagerOwnedRetainedTable>,
    currents: Vec<EagerPlanCurrentRow>,
    values: Vec<EagerPlanValueRow>,
    momenta: Vec<EagerPlanMomentumRow>,
    sources: Vec<EagerPlanSourceFillRow>,
    parameters: Vec<EagerPlanParameterRow>,
    stages: Vec<EagerPlanStageRow>,
    couplings: Vec<EagerPlanCouplingRow>,
    invocations: Vec<EagerPlanInvocationRow>,
    attachments: Vec<EagerPlanAttachmentRow>,
    finalizations: Vec<EagerPlanFinalizationRow>,
    closures: Vec<EagerPlanClosureRow>,
    direct_coefficients: Vec<EagerPlanDirectCoefficientRow>,
    selector_domains: Vec<EagerPlanSelectorDomainRow>,
    selector_memberships: Vec<u32>,
    helicity_selectors: Vec<EagerPlanHelicitySelectorRow>,
    color_selectors: Vec<EagerPlanColorSelectorRow>,
    reduction_groups: Vec<EagerPlanReductionGroupRow>,
    reduction_entries: Vec<EagerPlanReductionEntryRow>,
    exact_factors: Vec<EagerPlanExactFactorRow>,
    bitset_ranges: Vec<EagerPlanCatalogRangeRow>,
    bitset_populations: Vec<u64>,
    bitset_words: Vec<u64>,
    u32_sequence_ranges: Vec<EagerPlanCatalogRangeRow>,
    u32_sequence_values: Vec<u32>,
    i32_sequence_ranges: Vec<EagerPlanCatalogRangeRow>,
    i32_sequence_values: Vec<i32>,
    color_contraction_entry_start: u64,
    color_contraction_entry_count: u64,
    current_component_count: u64,
    value_component_count: u64,
    momentum_component_count: u64,
}

macro_rules! plan_slice_getter {
    ($name:ident, $field:ident, $row:ty) => {
        pub fn $name(&self) -> &[$row] {
            &self.$field
        }
    };
}

fn selector_domain_active(
    plan: &EagerPlanV3,
    domain_id: u32,
    active_groups: &BTreeSet<u32>,
) -> RusticolResult<bool> {
    if domain_id == MISSING_U32 {
        return Ok(true);
    }
    let domain = plan
        .selector_domains
        .get(domain_id as usize)
        .ok_or_else(|| invalid("eager selector-work census references an unknown domain"))?;
    let members = checked_usize_range(
        domain.member_start,
        domain.member_count,
        plan.selector_memberships.len(),
        "selector-work domain memberships",
    )?;
    Ok(plan.selector_memberships[members]
        .iter()
        .any(|member| active_groups.contains(member)))
}

fn active_selector_work(
    plan: &EagerPlanV3,
    active_groups: &BTreeSet<u32>,
) -> RusticolResult<EagerPlanActiveWorkCounts> {
    let mut currents = plan
        .sources
        .iter()
        .map(|source| source.current_id)
        .collect::<BTreeSet<_>>();
    let mut evaluations = 0_u64;
    let mut attachments_count = 0_u64;
    for invocation in &plan.invocations {
        if !selector_domain_active(plan, invocation.selector_domain_id, active_groups)? {
            continue;
        }
        let attachments = checked_usize_range(
            invocation.attachment_start,
            invocation.attachment_count,
            plan.attachments.len(),
            "selector-work invocation attachments",
        )?;
        let mut invocation_active = false;
        for attachment in &plan.attachments[attachments] {
            if !selector_domain_active(plan, attachment.selector_domain_id, active_groups)? {
                continue;
            }
            invocation_active = true;
            currents.insert(attachment.result_current_id);
            attachments_count = attachments_count
                .checked_add(1)
                .ok_or_else(|| invalid("selector-work attachment count exceeds u64"))?;
        }
        if invocation_active {
            evaluations = evaluations
                .checked_add(1)
                .ok_or_else(|| invalid("selector-work evaluation count exceeds u64"))?;
        }
    }
    Ok(EagerPlanActiveWorkCounts {
        current_count: usize_u64(currents.len(), "selector-work current count")?,
        evaluation_count: evaluations,
        attachment_count: attachments_count,
    })
}

fn worst_selector_work<'a>(
    plan: &EagerPlanV3,
    selectors: impl Iterator<Item = &'a BTreeSet<u32>>,
    context: &str,
) -> RusticolResult<EagerPlanActiveWorkCounts> {
    let mut worst = None::<EagerPlanActiveWorkCounts>;
    for groups in selectors {
        let counts = active_selector_work(plan, groups)?;
        worst = Some(match worst {
            None => counts,
            Some(previous) => EagerPlanActiveWorkCounts {
                current_count: previous.current_count.max(counts.current_count),
                evaluation_count: previous.evaluation_count.max(counts.evaluation_count),
                attachment_count: previous.attachment_count.max(counts.attachment_count),
            },
        });
    }
    worst.ok_or_else(|| invalid(format!("eager selector-work census has no {context}")))
}

impl EagerPlanV3 {
    pub fn abi(&self) -> &'static str {
        EAGER_PLAN_ABI
    }

    pub fn process_key(&self) -> &str {
        &self.process_key
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn string_catalog(&self) -> &[Box<str>] {
        &self.string_catalog
    }

    pub fn canonical_ir_catalog(&self) -> &[Box<str>] {
        &self.canonical_ir_catalog
    }

    pub fn semantic_limitations(&self) -> &[Box<str>] {
        &self.semantic_limitations
    }

    pub fn retained_tables(&self) -> &[EagerOwnedRetainedTable] {
        &self.retained_tables
    }

    plan_slice_getter!(currents, currents, EagerPlanCurrentRow);
    plan_slice_getter!(values, values, EagerPlanValueRow);
    plan_slice_getter!(momenta, momenta, EagerPlanMomentumRow);
    plan_slice_getter!(sources, sources, EagerPlanSourceFillRow);
    plan_slice_getter!(parameters, parameters, EagerPlanParameterRow);
    plan_slice_getter!(stages, stages, EagerPlanStageRow);
    plan_slice_getter!(couplings, couplings, EagerPlanCouplingRow);
    plan_slice_getter!(invocations, invocations, EagerPlanInvocationRow);
    plan_slice_getter!(attachments, attachments, EagerPlanAttachmentRow);
    plan_slice_getter!(finalizations, finalizations, EagerPlanFinalizationRow);
    plan_slice_getter!(closures, closures, EagerPlanClosureRow);
    plan_slice_getter!(
        direct_coefficients,
        direct_coefficients,
        EagerPlanDirectCoefficientRow
    );
    plan_slice_getter!(
        selector_domains,
        selector_domains,
        EagerPlanSelectorDomainRow
    );
    plan_slice_getter!(
        helicity_selectors,
        helicity_selectors,
        EagerPlanHelicitySelectorRow
    );
    plan_slice_getter!(color_selectors, color_selectors, EagerPlanColorSelectorRow);
    plan_slice_getter!(
        reduction_groups,
        reduction_groups,
        EagerPlanReductionGroupRow
    );
    plan_slice_getter!(
        reduction_entries,
        reduction_entries,
        EagerPlanReductionEntryRow
    );
    plan_slice_getter!(exact_factors, exact_factors, EagerPlanExactFactorRow);

    pub fn selector_memberships(&self) -> &[u32] {
        &self.selector_memberships
    }

    /// Count the exact rows admitted by the same selector domains as runtime.
    pub fn selector_work_summary(&self) -> RusticolResult<EagerPlanSelectorWorkSummary> {
        let mut groups_by_helicity = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut groups_by_color = BTreeMap::<u32, BTreeSet<u32>>::new();
        for entry in &self.reduction_entries {
            if entry.kind != EagerPlanReductionEntryKind::SelectorMember {
                continue;
            }
            groups_by_helicity
                .entry(entry.left_id)
                .or_default()
                .insert(entry.owner_id);
            groups_by_color
                .entry(entry.right_id)
                .or_default()
                .insert(entry.owner_id);
        }
        if groups_by_helicity.is_empty() || groups_by_color.is_empty() {
            return Err(invalid(
                "eager selector-work census has no physical selector memberships",
            ));
        }
        let selected_flow_worst =
            worst_selector_work(self, groups_by_color.values(), "color selector")?;
        let all_flow_worst =
            worst_selector_work(self, groups_by_helicity.values(), "helicity selector")?;
        Ok(EagerPlanSelectorWorkSummary {
            selected_flow_selector_count: usize_u64(
                groups_by_color.len(),
                "selected-flow selector count",
            )?,
            selected_flow_worst,
            all_flow_selector_count: usize_u64(
                groups_by_helicity.len(),
                "all-flow selector count",
            )?,
            all_flow_worst,
            contracted: EagerPlanActiveWorkCounts {
                current_count: usize_u64(self.currents.len(), "contracted current count")?,
                evaluation_count: usize_u64(self.invocations.len(), "contracted evaluation count")?,
                attachment_count: usize_u64(self.attachments.len(), "contracted attachment count")?,
            },
        })
    }

    pub fn bitset_ranges(&self) -> &[EagerPlanCatalogRangeRow] {
        &self.bitset_ranges
    }

    pub fn bitset_populations(&self) -> &[u64] {
        &self.bitset_populations
    }

    pub fn bitset_words(&self) -> &[u64] {
        &self.bitset_words
    }

    pub fn u32_sequence_ranges(&self) -> &[EagerPlanCatalogRangeRow] {
        &self.u32_sequence_ranges
    }

    pub fn u32_sequence_values(&self) -> &[u32] {
        &self.u32_sequence_values
    }

    pub fn i32_sequence_ranges(&self) -> &[EagerPlanCatalogRangeRow] {
        &self.i32_sequence_ranges
    }

    pub fn i32_sequence_values(&self) -> &[i32] {
        &self.i32_sequence_values
    }

    pub fn current_component_count(&self) -> u64 {
        self.current_component_count
    }

    pub fn value_component_count(&self) -> u64 {
        self.value_component_count
    }

    pub fn momentum_component_count(&self) -> u64 {
        self.momentum_component_count
    }

    pub fn color_contraction_entry_range(&self) -> (u64, u64) {
        (
            self.color_contraction_entry_start,
            self.color_contraction_entry_count,
        )
    }

    pub fn section_shapes(&self) -> RusticolResult<Vec<EagerPlanSectionShape>> {
        let mut shapes = reserved(17, "eager section shapes")?;
        let mut add = |kind, record_size, count: usize| -> RusticolResult<()> {
            shapes.push(EagerPlanSectionShape {
                kind,
                record_size,
                record_count: usize_u64(count, "eager section record count")?,
            });
            Ok(())
        };
        add(EagerSectionKind::Metadata, 64, 1)?;
        add(
            EagerSectionKind::CurrentLayout,
            EagerPlanCurrentRow::ENCODED_LEN,
            self.currents.len(),
        )?;
        add(
            EagerSectionKind::ValueLayout,
            EagerPlanValueRow::ENCODED_LEN,
            self.values.len(),
        )?;
        add(
            EagerSectionKind::MomentumLayout,
            EagerPlanMomentumRow::ENCODED_LEN,
            self.momenta.len(),
        )?;
        add(
            EagerSectionKind::SourceFill,
            EagerPlanSourceFillRow::ENCODED_LEN,
            self.sources.len(),
        )?;
        add(
            EagerSectionKind::ParameterLayout,
            EagerPlanParameterRow::ENCODED_LEN,
            self.parameters.len(),
        )?;
        add(
            EagerSectionKind::Stages,
            EagerPlanStageRow::ENCODED_LEN,
            self.stages.len(),
        )?;
        add(
            EagerSectionKind::Couplings,
            EagerPlanCouplingRow::ENCODED_LEN,
            self.couplings.len(),
        )?;
        add(
            EagerSectionKind::Invocations,
            EagerPlanInvocationRow::ENCODED_LEN,
            self.invocations.len(),
        )?;
        add(
            EagerSectionKind::Attachments,
            EagerPlanAttachmentRow::ENCODED_LEN,
            self.attachments.len(),
        )?;
        add(
            EagerSectionKind::Finalizations,
            EagerPlanFinalizationRow::ENCODED_LEN,
            self.finalizations.len(),
        )?;
        add(
            EagerSectionKind::Closures,
            EagerPlanClosureRow::ENCODED_LEN,
            self.closures.len(),
        )?;
        add(
            EagerSectionKind::SelectorDomains,
            EagerPlanSelectorDomainRow::ENCODED_LEN,
            self.selector_domains.len(),
        )?;
        add(
            EagerSectionKind::SelectorMemberships,
            4,
            self.selector_memberships.len(),
        )?;
        add(
            EagerSectionKind::ReductionGroups,
            EagerPlanReductionGroupRow::ENCODED_LEN,
            self.reduction_groups.len(),
        )?;
        add(
            EagerSectionKind::ReductionEntries,
            EagerPlanReductionEntryRow::ENCODED_LEN,
            self.reduction_entries.len(),
        )?;
        add(
            EagerSectionKind::ExactFactors,
            EagerPlanExactFactorRow::ENCODED_LEN,
            self.exact_factors.len(),
        )?;
        Ok(shapes)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ValueSlots {
    source: Option<u32>,
    unpropagated: Option<u32>,
    propagated: Option<u32>,
    input: Option<u32>,
}

/// Deterministically lower one validated v1 input into a flat plan-v3.
///
/// The input is consumed so catalogs and retained column payloads move into the
/// plan without a second allocation or copy.
pub fn lower_eager_plan_v3(input: EagerLoweringInputV1) -> RusticolResult<EagerPlanV3> {
    let (currents, current_component_count, momentum_by_bitset) = lower_currents(&input)?;
    let (values, value_slots, value_component_count) = lower_values(&input)?;
    let (momenta, momentum_component_count) = lower_momenta(&input)?;
    let sources = lower_sources(&input, &value_slots)?;
    let parameters = lower_parameters(&input)?;
    let couplings = lower_couplings(&input)?;
    let (stages, mut invocations, mut attachments, mut finalizations, producer_stage_by_current) =
        lower_stages(&input, &value_slots, &momentum_by_bitset)?;
    let (mut closures, direct_coefficients) = lower_closures(&input, &value_slots)?;
    let (selector_domains, selector_memberships) = lower_selector_domains(
        &stages,
        &mut invocations,
        &mut attachments,
        &mut finalizations,
        &mut closures,
        values.len(),
        input.currents.len(),
    )?;
    validate_stage_dependencies(&input, &producer_stage_by_current)?;
    let helicity_selectors = lower_helicity_selectors(&input)?;
    let color_selectors = lower_color_selectors(&input)?;
    let (
        reduction_groups,
        reduction_entries,
        color_contraction_entry_start,
        color_contraction_entry_count,
    ) = lower_reductions(&input)?;
    let exact_factors = input
        .exact_factors
        .iter()
        .enumerate()
        .map(|(factor_id, factor)| {
            Ok(EagerPlanExactFactorRow {
                factor_id: usize_u32(factor_id, "exact factor ID")?,
                real_bits: factor.real_bits,
                imaginary_bits: factor.imaginary_bits,
                canonical_string_id: factor.canonical_string_id,
                exact_source: factor.exact_source,
                exact_ir_id: factor.exact_ir_id,
                source_ir_id: factor.source_ir_id,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let bitset_ranges = input
        .bitset_ranges
        .iter()
        .map(|row| EagerPlanCatalogRangeRow {
            start: row.range.start,
            count: row.range.count,
        })
        .collect();
    let bitset_populations = input
        .bitset_ranges
        .iter()
        .map(|row| row.bit_count)
        .collect();
    let u32_sequence_ranges = input
        .u32_sequence_ranges
        .iter()
        .map(|row| EagerPlanCatalogRangeRow {
            start: row.start,
            count: row.count,
        })
        .collect();
    let i32_sequence_ranges = input
        .i32_sequence_ranges
        .iter()
        .map(|row| EagerPlanCatalogRangeRow {
            start: row.start,
            count: row.count,
        })
        .collect();

    Ok(EagerPlanV3 {
        process_key: input.process_key,
        model_name: input.model_name,
        string_catalog: input.string_catalog,
        canonical_ir_catalog: input.canonical_ir_catalog,
        semantic_limitations: input.semantic_limitations,
        retained_tables: input.retained_tables,
        currents,
        values,
        momenta,
        sources,
        parameters,
        stages,
        couplings,
        invocations,
        attachments,
        finalizations,
        closures,
        direct_coefficients,
        selector_domains,
        selector_memberships,
        helicity_selectors,
        color_selectors,
        reduction_groups,
        reduction_entries,
        exact_factors,
        bitset_ranges,
        bitset_populations,
        bitset_words: input.bitset_words,
        u32_sequence_ranges,
        u32_sequence_values: input.u32_sequence_values,
        i32_sequence_ranges,
        i32_sequence_values: input.i32_sequence_values,
        color_contraction_entry_start,
        color_contraction_entry_count,
        current_component_count,
        value_component_count,
        momentum_component_count,
    })
}

fn lower_currents(
    input: &EagerLoweringInputV1,
) -> RusticolResult<(Vec<EagerPlanCurrentRow>, u64, Vec<u32>)> {
    let mut momentum_by_bitset = vec![MISSING_U32; input.bitset_ranges.len()];
    for (slot_id, bitset_id) in &input.momentum_masks {
        momentum_by_bitset[*bitset_id as usize] = *slot_id;
    }
    let mut rows = reserved(input.currents.len(), "current layout")?;
    let mut component_start = 0_u64;
    for current in &input.currents {
        rows.push(EagerPlanCurrentRow {
            current_id: current.id,
            component_start,
            component_count: current.dimension,
            momentum_slot_id: momentum_by_bitset[current.momentum_mask_bitset_id as usize],
            flags: if current.is_source {
                CURRENT_FLAG_SOURCE
            } else {
                0
            },
        });
        component_start = component_start
            .checked_add(u64::from(current.dimension))
            .ok_or_else(|| invalid("current component layout exceeds u64"))?;
    }
    Ok((rows, component_start, momentum_by_bitset))
}

fn lower_values(
    input: &EagerLoweringInputV1,
) -> RusticolResult<(Vec<EagerPlanValueRow>, Vec<ValueSlots>, u64)> {
    let mut used_as_input = vec![false; input.currents.len()];
    let mut used_as_root = vec![false; input.currents.len()];
    for interaction in &input.interactions {
        used_as_input[interaction.left_current_id as usize] = true;
        used_as_input[interaction.right_current_id as usize] = true;
    }
    for root in &input.roots {
        used_as_root[root.left_current_id as usize] = true;
        used_as_root[root.right_current_id as usize] = true;
    }

    let capacity = input
        .currents
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("value slot capacity exceeds usize"))?;
    let mut rows = reserved(capacity, "value layout")?;
    let mut slots = vec![ValueSlots::default(); input.currents.len()];
    let mut component_start = 0_u64;
    for current in &input.currents {
        let current_index = current.id as usize;
        if current.is_source {
            let slot = push_value_row(
                &mut rows,
                current.id,
                current.dimension,
                EagerValueSlotKind::Source,
                &mut component_start,
            )?;
            slots[current_index].source = Some(slot);
            if used_as_input[current_index] {
                slots[current_index].input = Some(slot);
            }
            continue;
        }
        let needs_propagated =
            used_as_input[current_index] && current.propagator_kernel_id != MISSING_U32;
        let needs_unpropagated = used_as_root[current_index] || !needs_propagated;
        if needs_unpropagated {
            slots[current_index].unpropagated = Some(push_value_row(
                &mut rows,
                current.id,
                current.dimension,
                EagerValueSlotKind::Unpropagated,
                &mut component_start,
            )?);
        }
        if needs_propagated {
            slots[current_index].propagated = Some(push_value_row(
                &mut rows,
                current.id,
                current.dimension,
                EagerValueSlotKind::Propagated,
                &mut component_start,
            )?);
        }
        if used_as_input[current_index] {
            slots[current_index].input = if needs_propagated {
                slots[current_index].propagated
            } else {
                slots[current_index].unpropagated
            };
        }
    }
    Ok((rows, slots, component_start))
}

fn push_value_row(
    rows: &mut Vec<EagerPlanValueRow>,
    current_id: u32,
    component_count: u32,
    kind: EagerValueSlotKind,
    component_start: &mut u64,
) -> RusticolResult<u32> {
    let value_slot_id = usize_u32(rows.len(), "value slot ID")?;
    rows.push(EagerPlanValueRow {
        value_slot_id,
        current_id,
        component_start: *component_start,
        component_count,
        kind,
    });
    *component_start = component_start
        .checked_add(u64::from(component_count))
        .ok_or_else(|| invalid("value component layout exceeds u64"))?;
    Ok(value_slot_id)
}

fn lower_momenta(input: &EagerLoweringInputV1) -> RusticolResult<(Vec<EagerPlanMomentumRow>, u64)> {
    let mut rows = reserved(input.momentum_masks.len(), "momentum layout")?;
    let mut component_start = 0_u64;
    for (slot_id, bitset_id) in &input.momentum_masks {
        rows.push(EagerPlanMomentumRow {
            momentum_slot_id: *slot_id,
            bitset_id: *bitset_id,
            component_start,
            component_count: 4,
        });
        component_start = component_start
            .checked_add(4)
            .ok_or_else(|| invalid("momentum component layout exceeds u64"))?;
    }
    Ok((rows, component_start))
}

fn lower_sources(
    input: &EagerLoweringInputV1,
    value_slots: &[ValueSlots],
) -> RusticolResult<Vec<EagerPlanSourceFillRow>> {
    let mut rows = reserved(input.sources.len(), "source fill")?;
    for source in &input.sources {
        let value_slot_id = value_slots[source.current_id as usize]
            .source
            .ok_or_else(|| {
                invalid(format!(
                    "source {} has no source value slot",
                    source.source_id
                ))
            })?;
        rows.push(EagerPlanSourceFillRow {
            source_id: source.source_id,
            current_id: source.current_id,
            value_slot_id,
            external_label: source.external_label,
            input_momentum_slot: source.input_momentum_slot,
            source_ir_id: source.source_ir_id,
            crossing_ir_id: source.crossing_ir_id,
            crossing_factor_id: source.crossing_factor_id,
            declared_state_index: source.declared_state_index,
        });
    }
    Ok(rows)
}

fn lower_parameters(input: &EagerLoweringInputV1) -> RusticolResult<Vec<EagerPlanParameterRow>> {
    let mut rows = reserved(input.model_parameters.len(), "parameter layout")?;
    for (parameter_id, parameter) in input.model_parameters.iter().enumerate() {
        rows.push(EagerPlanParameterRow {
            parameter_id: usize_u32(parameter_id, "parameter ID")?,
            name_string_id: parameter.name_string_id,
            kind_string_id: parameter.kind_string_id,
            default_factor_id: parameter.default_factor_id,
            runtime_name_string_id: parameter.runtime_name_string_id,
            complex_component: parameter.complex_component,
            flags: u32::from(parameter.derived),
        });
    }
    Ok(rows)
}

fn lower_couplings(input: &EagerLoweringInputV1) -> RusticolResult<Vec<EagerPlanCouplingRow>> {
    let mut rows = reserved(input.couplings.len(), "coupling layout")?;
    for (coupling_id, coupling) in input.couplings.iter().enumerate() {
        let names = input.u32_sequence(coupling.parameter_name_ids_sequence_id)?;
        let mut real = MISSING_U32;
        let mut imaginary = MISSING_U32;
        let first = names.first().copied().filter(|id| *id != MISSING_U32);
        if let Some(logical_name) = first {
            let logical: Vec<_> = input
                .model_parameters
                .iter()
                .enumerate()
                .filter(|(_, parameter)| parameter.runtime_name_string_id == logical_name)
                .collect();
            if !logical.is_empty() {
                for (parameter_id, parameter) in logical {
                    let parameter_id = usize_u32(parameter_id, "coupling parameter ID")?;
                    match parameter.complex_component {
                        0 if real == MISSING_U32 => real = parameter_id,
                        1 if imaginary == MISSING_U32 => imaginary = parameter_id,
                        _ => {
                            return Err(invalid(format!(
                                "coupling {coupling_id} has ambiguous logical parameter components"
                            )));
                        }
                    }
                }
                if real == MISSING_U32 || imaginary == MISSING_U32 {
                    return Err(invalid(format!(
                        "coupling {coupling_id} logical parameter lacks real/imaginary slots"
                    )));
                }
            }
        }
        if real == MISSING_U32 && imaginary == MISSING_U32 {
            for (component, name) in names.iter().copied().enumerate() {
                if name == MISSING_U32 {
                    continue;
                }
                let matches = input
                    .model_parameters
                    .iter()
                    .enumerate()
                    .filter(|(_, parameter)| parameter.name_string_id == name)
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                if matches.len() != 1 {
                    return Err(invalid(format!(
                        "coupling {coupling_id} parameter name {name} is not unique"
                    )));
                }
                let parameter_id = usize_u32(matches[0], "coupling parameter ID")?;
                if component == 0 {
                    real = parameter_id;
                } else {
                    imaginary = parameter_id;
                }
            }
        }
        rows.push(EagerPlanCouplingRow {
            coupling_id: usize_u32(coupling_id, "coupling ID")?,
            real_parameter_id: real,
            imaginary_parameter_id: imaginary,
            constant_factor_id: coupling.constant_factor_id,
        });
    }
    Ok(rows)
}

type LoweredStages = (
    Vec<EagerPlanStageRow>,
    Vec<EagerPlanInvocationRow>,
    Vec<EagerPlanAttachmentRow>,
    Vec<EagerPlanFinalizationRow>,
    Vec<u32>,
);

fn lower_stages(
    input: &EagerLoweringInputV1,
    value_slots: &[ValueSlots],
    momentum_by_bitset: &[u32],
) -> RusticolResult<LoweredStages> {
    let stage_sizes = input
        .interactions
        .iter()
        .map(|row| row.stage_subset_size)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut stage_position_by_size = BTreeMap::new();
    for (position, subset_size) in stage_sizes.iter().copied().enumerate() {
        stage_position_by_size.insert(subset_size, position);
    }
    let mut groups_by_stage = vec![Vec::new(); stage_sizes.len()];
    for (group_id, group) in input.interaction_groups.iter().enumerate() {
        let representative = &input.interactions[group.representative_interaction_id as usize];
        let position = stage_position_by_size[&representative.stage_subset_size];
        groups_by_stage[position].push(group_id);
    }
    let mut outputs_by_stage = vec![BTreeSet::new(); stage_sizes.len()];
    let mut producer_stage_by_current = vec![MISSING_U32; input.currents.len()];
    for interaction in &input.interactions {
        let position = stage_position_by_size[&interaction.stage_subset_size];
        outputs_by_stage[position].insert(interaction.result_current_id);
        producer_stage_by_current[interaction.result_current_id as usize] =
            usize_u32(position, "producer stage")?;
    }

    let mut stages = reserved(stage_sizes.len(), "stages")?;
    let mut invocations = reserved(input.interaction_groups.len(), "invocations")?;
    let mut attachments = reserved(input.interactions.len(), "attachments")?;
    let finalization_capacity = outputs_by_stage.iter().map(BTreeSet::len).sum();
    let mut finalizations = reserved(finalization_capacity, "finalizations")?;
    for (stage_position, subset_size) in stage_sizes.iter().copied().enumerate() {
        let invocation_start = usize_u64(invocations.len(), "invocation start")?;
        let attachment_start = usize_u64(attachments.len(), "attachment start")?;
        for group_id in &groups_by_stage[stage_position] {
            let group = input.interaction_groups[*group_id];
            let representative = input.interactions[group.representative_interaction_id as usize];
            if factor_is_complex_zero(
                &input.exact_factors[representative.evaluation_factor_id as usize],
            ) {
                continue;
            }
            let members = range_slice(
                &input.interaction_group_members,
                group.members,
                "interaction group members",
            )?;
            let ordered_currents = match representative.canonical_input_order {
                [0, 1] => [
                    representative.left_current_id,
                    representative.right_current_id,
                ],
                [1, 0] => [
                    representative.right_current_id,
                    representative.left_current_id,
                ],
                _ => unreachable!("validated canonical input order"),
            };
            let invocation_attachment_start_index = attachments.len();
            let invocation_attachment_start = usize_u64(
                invocation_attachment_start_index,
                "invocation attachment start",
            )?;
            for member_id in members {
                let member = input.interactions[*member_id as usize];
                if factor_is_complex_zero(
                    &input.exact_factors[member.evaluation_factor_id as usize],
                ) {
                    continue;
                }
                attachments.push(EagerPlanAttachmentRow {
                    interaction_id: member.id,
                    result_current_id: member.result_current_id,
                    color_factor_id: member.color_factor_id,
                    evaluation_factor_id: member.evaluation_factor_id,
                    normalization_factor_id: representative.kernel_normalization_factor_id,
                    representative_evaluation_factor_id: representative.evaluation_factor_id,
                    selector_domain_id: MISSING_U32,
                });
            }
            let invocation_attachment_count = usize_u64(
                attachments.len() - invocation_attachment_start_index,
                "invocation attachment count",
            )?;
            invocations.push(EagerPlanInvocationRow {
                evaluation_group_id: *group_id as u32,
                kernel_id: representative.kernel_id,
                left_value_slot_id: value_slots[ordered_currents[0] as usize]
                    .input
                    .ok_or_else(|| invalid("invocation left current has no input value slot"))?,
                right_value_slot_id: value_slots[ordered_currents[1] as usize]
                    .input
                    .ok_or_else(|| invalid("invocation right current has no input value slot"))?,
                left_momentum_slot_id: momentum_by_bitset
                    [input.currents[ordered_currents[0] as usize].momentum_mask_bitset_id as usize],
                right_momentum_slot_id: momentum_by_bitset
                    [input.currents[ordered_currents[1] as usize].momentum_mask_bitset_id as usize],
                coupling_slot_id: representative.coupling_id,
                output_factor_source: representative.output_factor_source,
                attachment_start: invocation_attachment_start,
                attachment_count: invocation_attachment_count,
                selector_domain_id: MISSING_U32,
            });
        }
        let finalization_start = usize_u64(finalizations.len(), "finalization start")?;
        for current_id in &outputs_by_stage[stage_position] {
            let current = input.currents[*current_id as usize];
            let slots = value_slots[*current_id as usize];
            if slots.unpropagated.is_none() && slots.propagated.is_none() {
                return Err(invalid(format!(
                    "stage output current {current_id} has no output value slot"
                )));
            }
            if slots.propagated.is_some() && current.propagator_kernel_id == MISSING_U32 {
                return Err(invalid(format!(
                    "propagated current {current_id} has no prepared propagator kernel"
                )));
            }
            finalizations.push(EagerPlanFinalizationRow {
                kernel_id: if slots.propagated.is_some() {
                    current.propagator_kernel_id
                } else {
                    MISSING_U32
                },
                current_id: *current_id,
                unpropagated_value_slot_id: slots.unpropagated.unwrap_or(MISSING_U32),
                propagated_value_slot_id: slots.propagated.unwrap_or(MISSING_U32),
                momentum_slot_id: momentum_by_bitset[current.momentum_mask_bitset_id as usize],
                unpropagated_selector_domain_id: MISSING_U32,
                propagated_selector_domain_id: MISSING_U32,
            });
        }
        stages.push(EagerPlanStageRow {
            stage_index: usize_u32(stage_position + 1, "stage index")?,
            subset_size,
            invocation_start,
            invocation_count: usize_u64(invocations.len(), "invocation stop")?
                .checked_sub(invocation_start)
                .ok_or_else(|| invalid("invocation range underflow"))?,
            attachment_start,
            attachment_count: usize_u64(attachments.len(), "attachment stop")?
                .checked_sub(attachment_start)
                .ok_or_else(|| invalid("attachment range underflow"))?,
            finalization_start,
            finalization_count: usize_u64(finalizations.len(), "finalization stop")?
                .checked_sub(finalization_start)
                .ok_or_else(|| invalid("finalization range underflow"))?,
        });
    }
    Ok((
        stages,
        invocations,
        attachments,
        finalizations,
        producer_stage_by_current,
    ))
}

fn lower_closures(
    input: &EagerLoweringInputV1,
    value_slots: &[ValueSlots],
) -> RusticolResult<(Vec<EagerPlanClosureRow>, Vec<EagerPlanDirectCoefficientRow>)> {
    let mut coefficient_ranges = BTreeMap::<u32, InputRange>::new();
    let mut coefficients = reserved(
        input.contraction_coefficients.len(),
        "direct closure coefficients",
    )?;
    let mut cursor = 0;
    while cursor < input.contraction_coefficients.len() {
        let ir_id = input.contraction_coefficients[cursor].contraction_ir_id;
        let start = coefficients.len();
        while cursor < input.contraction_coefficients.len()
            && input.contraction_coefficients[cursor].contraction_ir_id == ir_id
        {
            let row = input.contraction_coefficients[cursor];
            coefficients.push(EagerPlanDirectCoefficientRow {
                contraction_ir_id: row.contraction_ir_id,
                component_index: row.component_index,
                factor_id: row.factor_id,
            });
            cursor += 1;
        }
        coefficient_ranges.insert(
            ir_id,
            InputRange {
                start: usize_u64(start, "direct coefficient start")?,
                count: usize_u64(coefficients.len() - start, "direct coefficient count")?,
            },
        );
    }
    let mut closures = reserved(input.roots.len(), "closures")?;
    for root in &input.roots {
        let unordered = [root.left_current_id, root.right_current_id];
        let ordered = [
            unordered[root.canonical_input_order[0] as usize],
            unordered[root.canonical_input_order[1] as usize],
        ];
        let value_slot = |current_id: u32| -> RusticolResult<u32> {
            let current = input.currents[current_id as usize];
            let slots = value_slots[current_id as usize];
            if current.is_source {
                slots.source
            } else {
                slots.unpropagated
            }
            .ok_or_else(|| {
                invalid(format!(
                    "root current {current_id} has no amplitude value slot"
                ))
            })
        };
        let direct_range = if root.kernel_id == MISSING_U32 {
            coefficient_ranges
                .get(&root.contraction_ir_id)
                .copied()
                .ok_or_else(|| {
                    invalid(format!(
                        "direct root {} has no contraction coefficients",
                        root.id
                    ))
                })?
        } else {
            InputRange { start: 0, count: 0 }
        };
        closures.push(EagerPlanClosureRow {
            root_id: root.id,
            kernel_id: root.kernel_id,
            left_value_slot_id: value_slot(ordered[0])?,
            right_value_slot_id: value_slot(ordered[1])?,
            amplitude_index: root.id,
            coherent_group_id: root.coherent_group_id,
            coupling_slot_id: root.coupling_id,
            coupling_factor_id: root.coupling_factor_id,
            output_factor_source: root.output_factor_source,
            color_factor_id: root.color_factor_id,
            normalization_factor_id: root.kernel_normalization_factor_id,
            direct_coefficient_start: direct_range.start,
            direct_coefficient_count: direct_range.count,
            selector_domain_id: MISSING_U32,
        });
    }
    Ok((closures, coefficients))
}

#[derive(Default)]
struct DomainInterner {
    by_members: BTreeMap<Vec<u32>, u32>,
    members: Vec<Vec<u32>>,
}

impl DomainInterner {
    fn intern(&mut self, members: &[u32]) -> RusticolResult<u32> {
        if let Some(existing) = self.by_members.get(members) {
            return Ok(*existing);
        }
        let id = usize_present_u32(self.members.len(), "selector domain ID")?;
        let owned = copy_slice(members, "selector domain members")?;
        self.by_members.insert(owned.clone(), id);
        self.members.push(owned);
        Ok(id)
    }
}

fn lower_selector_domains(
    stages: &[EagerPlanStageRow],
    invocations: &mut [EagerPlanInvocationRow],
    attachments: &mut [EagerPlanAttachmentRow],
    finalizations: &mut [EagerPlanFinalizationRow],
    closures: &mut [EagerPlanClosureRow],
    value_count: usize,
    current_count: usize,
) -> RusticolResult<(Vec<EagerPlanSelectorDomainRow>, Vec<u32>)> {
    let mut interner = DomainInterner::default();
    let empty_domain = interner.intern(&[])?;
    let mut value_domains = vec![Vec::<u32>::new(); value_count];
    for closure in closures.iter_mut() {
        let members = [closure.coherent_group_id];
        closure.selector_domain_id = interner.intern(&members)?;
        union_sorted_into(
            &mut value_domains[closure.left_value_slot_id as usize],
            &members,
        )?;
        union_sorted_into(
            &mut value_domains[closure.right_value_slot_id as usize],
            &members,
        )?;
    }

    let mut current_domains = vec![Vec::<u32>::new(); current_count];
    for stage in stages.iter().rev() {
        let finalization_range = checked_usize_range(
            stage.finalization_start,
            stage.finalization_count,
            finalizations.len(),
            "stage finalizations",
        )?;
        for finalization in &mut finalizations[finalization_range.clone()] {
            let unpropagated = if finalization.unpropagated_value_slot_id == MISSING_U32 {
                Vec::new()
            } else {
                value_domains[finalization.unpropagated_value_slot_id as usize].clone()
            };
            let propagated = if finalization.propagated_value_slot_id == MISSING_U32 {
                Vec::new()
            } else {
                value_domains[finalization.propagated_value_slot_id as usize].clone()
            };
            finalization.unpropagated_selector_domain_id = interner.intern(&unpropagated)?;
            finalization.propagated_selector_domain_id = interner.intern(&propagated)?;
            let mut current = unpropagated;
            union_sorted_into(&mut current, &propagated)?;
            current_domains[finalization.current_id as usize] = current;
            if finalization.unpropagated_value_slot_id != MISSING_U32 {
                value_domains[finalization.unpropagated_value_slot_id as usize].clear();
            }
            if finalization.propagated_value_slot_id != MISSING_U32 {
                value_domains[finalization.propagated_value_slot_id as usize].clear();
            }
        }

        let attachment_range = checked_usize_range(
            stage.attachment_start,
            stage.attachment_count,
            attachments.len(),
            "stage attachments",
        )?;
        for attachment in &mut attachments[attachment_range] {
            attachment.selector_domain_id =
                interner.intern(&current_domains[attachment.result_current_id as usize])?;
        }

        let invocation_range = checked_usize_range(
            stage.invocation_start,
            stage.invocation_count,
            invocations.len(),
            "stage invocations",
        )?;
        for invocation in &mut invocations[invocation_range] {
            let range = checked_usize_range(
                invocation.attachment_start,
                invocation.attachment_count,
                attachments.len(),
                "invocation attachments",
            )?;
            let mut domain = Vec::new();
            for attachment in &attachments[range] {
                union_sorted_into(
                    &mut domain,
                    &current_domains[attachment.result_current_id as usize],
                )?;
            }
            invocation.selector_domain_id = interner.intern(&domain)?;
            union_sorted_into(
                &mut value_domains[invocation.left_value_slot_id as usize],
                &domain,
            )?;
            union_sorted_into(
                &mut value_domains[invocation.right_value_slot_id as usize],
                &domain,
            )?;
        }
        for finalization in &finalizations[finalization_range] {
            current_domains[finalization.current_id as usize].clear();
        }
    }

    let mut order = (0..interner.members.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        compare_domain_members(&interner.members[*left], &interner.members[*right])
    });
    let mut remap = vec![0_u32; order.len()];
    let mut domain_rows = reserved(order.len(), "selector domains")?;
    let member_capacity = interner
        .members
        .iter()
        .try_fold(0_usize, |total, members| total.checked_add(members.len()))
        .ok_or_else(|| invalid("selector membership count exceeds usize"))?;
    let mut memberships = reserved(member_capacity, "selector memberships")?;
    for (final_id, provisional_id) in order.into_iter().enumerate() {
        remap[provisional_id] = usize_present_u32(final_id, "final selector domain ID")?;
        let members = &interner.members[provisional_id];
        domain_rows.push(EagerPlanSelectorDomainRow {
            member_start: usize_u64(memberships.len(), "selector member start")?,
            member_count: usize_u64(members.len(), "selector member count")?,
        });
        memberships.extend_from_slice(members);
    }
    let remap_id = |id: &mut u32| {
        debug_assert_ne!(*id, MISSING_U32);
        *id = remap[*id as usize];
    };
    for row in invocations {
        remap_id(&mut row.selector_domain_id);
    }
    for row in attachments {
        remap_id(&mut row.selector_domain_id);
    }
    for row in finalizations {
        remap_id(&mut row.unpropagated_selector_domain_id);
        remap_id(&mut row.propagated_selector_domain_id);
    }
    for row in closures {
        remap_id(&mut row.selector_domain_id);
    }
    let _ = empty_domain;
    Ok((domain_rows, memberships))
}

fn lower_helicity_selectors(
    input: &EagerLoweringInputV1,
) -> RusticolResult<Vec<EagerPlanHelicitySelectorRow>> {
    input
        .helicity_selectors
        .iter()
        .enumerate()
        .map(|(selector_id, row)| {
            Ok(EagerPlanHelicitySelectorRow {
                selector_id: usize_u32(selector_id, "helicity selector ID")?,
                values_sequence_id: row.values_sequence_id,
                representative_sequence_id: row.representative_sequence_id,
                coefficient_factor_id: row.coefficient_factor_id,
                computed: u8::from(row.computed),
                structural_zero: u8::from(row.structural_zero),
            })
        })
        .collect()
}

fn lower_color_selectors(
    input: &EagerLoweringInputV1,
) -> RusticolResult<Vec<EagerPlanColorSelectorRow>> {
    input
        .color_selectors
        .iter()
        .enumerate()
        .map(|(selector_id, row)| {
            Ok(EagerPlanColorSelectorRow {
                selector_id: usize_u32(selector_id, "color selector ID")?,
                word_sequence_id: row.word_sequence_id,
                representative_word_sequence_id: row.representative_word_sequence_id,
                coefficient_factor_id: row.coefficient_factor_id,
                computed: u8::from(row.computed),
            })
        })
        .collect()
}

type LoweredReductions = (
    Vec<EagerPlanReductionGroupRow>,
    Vec<EagerPlanReductionEntryRow>,
    u64,
    u64,
);

fn lower_reductions(input: &EagerLoweringInputV1) -> RusticolResult<LoweredReductions> {
    let mut amplitudes_by_group = vec![Vec::new(); input.coherent_groups.len()];
    for root in &input.roots {
        amplitudes_by_group[root.coherent_group_id as usize].push(root.id);
    }
    if amplitudes_by_group.iter().any(Vec::is_empty) {
        return Err(invalid("every coherent group must own an amplitude root"));
    }
    let mut selectors_by_group = vec![Vec::new(); input.coherent_groups.len()];
    for member in &input.reduction_members {
        selectors_by_group[member.coherent_group_id as usize].push(*member);
    }
    let base_entry_count = amplitudes_by_group
        .iter()
        .map(Vec::len)
        .chain(selectors_by_group.iter().map(Vec::len))
        .try_fold(0_usize, |total, count| total.checked_add(count))
        .ok_or_else(|| invalid("reduction entry count exceeds usize"))?;
    let contraction_count = if input.color_contraction.present != 0 {
        input.color_contraction_entries.len()
    } else {
        input.coherent_groups.len()
    };
    let capacity = base_entry_count
        .checked_add(contraction_count)
        .ok_or_else(|| invalid("reduction entry count exceeds usize"))?;
    let mut groups = reserved(input.coherent_groups.len(), "reduction groups")?;
    let mut entries = reserved(capacity, "reduction entries")?;
    for group in &input.coherent_groups {
        let group_index = group.id as usize;
        let amplitude_start = usize_u64(entries.len(), "amplitude reduction start")?;
        for amplitude_id in &amplitudes_by_group[group_index] {
            entries.push(EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::AmplitudeMember,
                owner_id: group.id,
                left_id: *amplitude_id,
                right_id: MISSING_U32,
                factor_id: MISSING_U32,
                auxiliary_factor_id: MISSING_U32,
            });
        }
        let amplitude_count = usize_u64(
            amplitudes_by_group[group_index].len(),
            "amplitude reduction count",
        )?;
        let selector_start = usize_u64(entries.len(), "selector reduction start")?;
        for selector in &selectors_by_group[group_index] {
            entries.push(EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::SelectorMember,
                owner_id: group.id,
                left_id: selector.helicity_selector_id,
                right_id: selector.color_selector_id,
                factor_id: MISSING_U32,
                auxiliary_factor_id: MISSING_U32,
            });
        }
        groups.push(EagerPlanReductionGroupRow {
            coherent_group_id: group.id,
            amplitude_entry_start: amplitude_start,
            amplitude_entry_count: amplitude_count,
            selector_entry_start: selector_start,
            selector_entry_count: usize_u64(
                selectors_by_group[group_index].len(),
                "selector reduction count",
            )?,
            helicity_weight_factor_id: group.helicity_weight_factor_id,
            all_sector_weight_factor_id: group.all_sector_weight_factor_id,
        });
    }
    let contraction_start = usize_u64(entries.len(), "color contraction start")?;
    if input.color_contraction.present != 0 {
        for entry in &input.color_contraction_entries {
            entries.push(EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::ColorContraction,
                owner_id: MISSING_U32,
                left_id: entry.left_group_id,
                right_id: entry.right_group_id,
                factor_id: entry.weight_factor_id,
                auxiliary_factor_id: entry.symmetry_factor_id,
            });
        }
    } else {
        for group in &input.coherent_groups {
            entries.push(EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::ColorContraction,
                owner_id: MISSING_U32,
                left_id: group.id,
                right_id: group.id,
                factor_id: group.all_sector_weight_factor_id,
                auxiliary_factor_id: MISSING_U32,
            });
        }
    }
    let contraction_entry_count = usize_u64(entries.len(), "color contraction stop")?
        .checked_sub(contraction_start)
        .ok_or_else(|| invalid("color contraction range underflow"))?;
    Ok((groups, entries, contraction_start, contraction_entry_count))
}

fn validate_stage_dependencies(
    input: &EagerLoweringInputV1,
    producer_stage_by_current: &[u32],
) -> RusticolResult<()> {
    let mut stage_by_size = BTreeMap::new();
    for size in input
        .interactions
        .iter()
        .map(|row| row.stage_subset_size)
        .collect::<BTreeSet<_>>()
    {
        let index = usize_u32(stage_by_size.len(), "stage position")?;
        stage_by_size.insert(size, index);
    }
    for interaction in &input.interactions {
        let stage = stage_by_size[&interaction.stage_subset_size];
        for current_id in [interaction.left_current_id, interaction.right_current_id] {
            if !input.currents[current_id as usize].is_source
                && producer_stage_by_current[current_id as usize] >= stage
            {
                return Err(invalid(format!(
                    "interaction {} consumes current {current_id} before it is finalized",
                    interaction.id
                )));
            }
        }
    }
    Ok(())
}

fn validate_group_signature(
    representative: &InputInteraction,
    member: &InputInteraction,
    group_id: usize,
) -> RusticolResult<()> {
    // Evaluation-group identity is the compiler's exact proof that these
    // oriented interactions share one representative kernel evaluation. The
    // attachment rows retain each member's factors and output contract. Only
    // crossing a stage boundary would make the shared invocation unschedulable.
    if representative.stage_subset_size != member.stage_subset_size {
        return Err(invalid(format!(
            "interaction group {group_id} crosses recursion stages"
        )));
    }
    Ok(())
}

fn validate_kernel_transform(
    kernel_id: u32,
    order: [u8; 2],
    factor_source: u8,
    kernel_optional: bool,
    context: &str,
) -> RusticolResult<()> {
    if kernel_id == MISSING_U32 && !kernel_optional {
        return Err(invalid(format!("{context} has no prepared kernel")));
    }
    if !matches!(order, [0, 1] | [1, 0]) {
        return Err(invalid(format!(
            "{context} canonical input order is not a permutation"
        )));
    }
    if !matches!(
        factor_source,
        OUTPUT_FACTOR_NONE | OUTPUT_FACTOR_COUPLING_REAL | OUTPUT_FACTOR_COUPLING_IMAG
    ) {
        return Err(invalid(format!(
            "{context} has unsupported output factor source {factor_source}"
        )));
    }
    if kernel_id == MISSING_U32 && factor_source != OUTPUT_FACTOR_NONE {
        return Err(invalid(format!(
            "direct {context} cannot use a prepared-kernel output factor"
        )));
    }
    Ok(())
}

fn compare_domain_members(left: &[u32], right: &[u32]) -> Ordering {
    left.len()
        .cmp(&right.len())
        .then_with(|| left.iter().rev().cmp(right.iter().rev()))
}

fn union_sorted_into(target: &mut Vec<u32>, source: &[u32]) -> RusticolResult<()> {
    if source.is_empty() {
        return Ok(());
    }
    if target.is_empty() {
        target
            .try_reserve_exact(source.len())
            .map_err(|error| invalid(format!("could not reserve selector domain: {error}")))?;
        target.extend_from_slice(source);
        return Ok(());
    }
    let capacity = target
        .len()
        .checked_add(source.len())
        .ok_or_else(|| invalid("selector domain union exceeds usize"))?;
    let mut merged = reserved(capacity, "selector domain union")?;
    let (mut left, mut right) = (0, 0);
    while left < target.len() || right < source.len() {
        let value = match (target.get(left), source.get(right)) {
            (Some(a), Some(b)) if a < b => {
                left += 1;
                *a
            }
            (Some(a), Some(b)) if b < a => {
                right += 1;
                *b
            }
            (Some(a), Some(_)) => {
                left += 1;
                right += 1;
                *a
            }
            (Some(a), None) => {
                left += 1;
                *a
            }
            (None, Some(b)) => {
                right += 1;
                *b
            }
            (None, None) => break,
        };
        merged.push(value);
    }
    *target = merged;
    Ok(())
}

fn compare_bitsets(
    input: &EagerLoweringInputV1,
    left_id: u32,
    right_id: u32,
) -> RusticolResult<Ordering> {
    let left = required_row(&input.bitset_ranges, left_id, "left bitset")?;
    let right = required_row(&input.bitset_ranges, right_id, "right bitset")?;
    let by_population = left.bit_count.cmp(&right.bit_count);
    if by_population != Ordering::Equal {
        return Ok(by_population);
    }
    let left_words = range_slice(&input.bitset_words, left.range, "left bitset words")?;
    let right_words = range_slice(&input.bitset_words, right.range, "right bitset words")?;
    Ok(left_words
        .len()
        .cmp(&right_words.len())
        .then_with(|| left_words.iter().rev().cmp(right_words.iter().rev())))
}

fn factor_is_complex_zero(factor: &InputExactFactor) -> bool {
    f64::from_bits(factor.real_bits) == 0.0 && f64::from_bits(factor.imaginary_bits) == 0.0
}

fn factor_is_real_zero(factor: &InputExactFactor) -> bool {
    f64::from_bits(factor.imaginary_bits) == 0.0
}

fn equal_column_lengths(context: &str, columns: &[(&str, usize)]) -> RusticolResult<usize> {
    let expected = columns.first().map(|(_, length)| *length).unwrap_or(0);
    for (name, length) in columns {
        if *length != expected {
            return Err(invalid(format!(
                "{context}.{name} has {length} rows, expected {expected}"
            )));
        }
    }
    usize_u32(expected, &format!("{context} row count"))?;
    Ok(expected)
}

fn bool_column(value: u8, context: &str, row: usize) -> RusticolResult<bool> {
    validate_bool_scalar(value, &format!("{context}[{row}]"))?;
    Ok(value != 0)
}

fn validate_bool_scalar(value: u8, context: &str) -> RusticolResult<()> {
    if value > 1 {
        return Err(invalid(format!("{context} must be zero or one")));
    }
    Ok(())
}

fn validate_dense_ids(
    ids: impl Iterator<Item = u32>,
    count: usize,
    context: &str,
) -> RusticolResult<()> {
    for (expected, actual) in ids.enumerate() {
        expect_dense_id(actual, expected, context)?;
    }
    usize_u32(count, &format!("{context} count"))?;
    Ok(())
}

fn expect_dense_id(actual: u32, expected: usize, context: &str) -> RusticolResult<()> {
    let expected = usize_u32(expected, &format!("{context} ID"))?;
    if actual != expected {
        return Err(invalid(format!(
            "{context} IDs must be dense: found {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn validate_flat_ranges(
    ranges: impl Iterator<Item = InputRange>,
    flat_len: usize,
    context: &str,
) -> RusticolResult<()> {
    let mut cursor = 0_u64;
    for (row, range) in ranges.enumerate() {
        if range.start != cursor {
            return Err(invalid(format!(
                "{context} range {row} starts at {}, expected {cursor}",
                range.start
            )));
        }
        cursor = range
            .start
            .checked_add(range.count)
            .ok_or_else(|| invalid(format!("{context} range {row} exceeds u64")))?;
    }
    if cursor != usize_u64(flat_len, &format!("{context} flat length"))? {
        return Err(invalid(format!(
            "{context} ranges cover {cursor} rows, expected {flat_len}"
        )));
    }
    Ok(())
}

fn range_slice<'a, T>(
    values: &'a [T],
    range: InputRange,
    context: &str,
) -> RusticolResult<&'a [T]> {
    let bounds = checked_usize_range(range.start, range.count, values.len(), context)?;
    Ok(&values[bounds])
}

fn checked_usize_range(
    start: u64,
    count: u64,
    length: usize,
    context: &str,
) -> RusticolResult<std::ops::Range<usize>> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| invalid(format!("{context} range exceeds u64")))?;
    let start = usize::try_from(start)
        .map_err(|_| invalid(format!("{context} start exceeds platform bounds")))?;
    let end = usize::try_from(end)
        .map_err(|_| invalid(format!("{context} end exceeds platform bounds")))?;
    if start > length || end > length {
        return Err(invalid(format!("{context} range exceeds table bounds")));
    }
    Ok(start..end)
}

fn required_row<'a, T>(values: &'a [T], id: u32, context: &str) -> RusticolResult<&'a T> {
    let index = index(id, values.len(), context)?;
    Ok(&values[index])
}

fn required_index(id: u32, count: impl TryInto<usize>, context: &str) -> RusticolResult<usize> {
    let count = count
        .try_into()
        .map_err(|_| invalid(format!("{context} count exceeds platform bounds")))?;
    index(id, count, context)
}

fn optional_index(id: u32, count: impl TryInto<usize>, context: &str) -> RusticolResult<()> {
    if id != MISSING_U32 {
        required_index(id, count, context)?;
    }
    Ok(())
}

fn index(id: u32, count: usize, context: &str) -> RusticolResult<usize> {
    let index = usize::try_from(id)
        .map_err(|_| invalid(format!("{context} ID {id} exceeds platform bounds")))?;
    if index >= count {
        return Err(invalid(format!(
            "{context} ID {id} references an absent row (count {count})"
        )));
    }
    Ok(index)
}

#[cfg(test)]
fn count_u32(count: u64, context: &str) -> RusticolResult<u32> {
    u32::try_from(count).map_err(|_| invalid(format!("{context} count exceeds u32")))
}

fn usize_u32(value: usize, context: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{context} exceeds u32")))
}

fn usize_present_u32(value: usize, context: &str) -> RusticolResult<u32> {
    let value = usize_u32(value, context)?;
    if value == MISSING_U32 {
        return Err(invalid(format!(
            "{context} collides with the missing-ID sentinel"
        )));
    }
    Ok(value)
}

fn usize_u64(value: usize, context: &str) -> RusticolResult<u64> {
    u64::try_from(value).map_err(|_| invalid(format!("{context} exceeds u64")))
}

fn reserved<T>(capacity: usize, context: &str) -> RusticolResult<Vec<T>> {
    let mut result = Vec::new();
    result.try_reserve_exact(capacity).map_err(|error| {
        invalid(format!(
            "could not reserve {context} ({capacity} rows): {error}"
        ))
    })?;
    Ok(result)
}

fn copy_slice<T: Copy>(values: &[T], context: &str) -> RusticolResult<Vec<T>> {
    let mut result = reserved(values.len(), context)?;
    result.extend_from_slice(values);
    Ok(result)
}

fn copy_retained_tables(
    tables: &[EagerRetainedTableView<'_>],
) -> RusticolResult<Vec<EagerOwnedRetainedTable>> {
    if tables.len() != EAGER_LOWERING_V1_TABLE_NAMES.len() {
        return Err(invalid(format!(
            "eager lowering retained table inventory has {} entries, expected {}",
            tables.len(),
            EAGER_LOWERING_V1_TABLE_NAMES.len()
        )));
    }
    let mut owned_tables = reserved(tables.len(), "retained eager tables")?;
    for (table_index, (table, expected_name)) in
        tables.iter().zip(EAGER_LOWERING_V1_TABLE_NAMES).enumerate()
    {
        if table.name != *expected_name {
            return Err(invalid(format!(
                "eager lowering retained table {table_index} is {:?}, expected {:?}",
                table.name, expected_name
            )));
        }
        let row_count = usize::try_from(table.row_count).map_err(|_| {
            invalid(format!(
                "retained table {:?} row count exceeds platform bounds",
                table.name
            ))
        })?;
        usize_u32(
            row_count,
            &format!("retained table {:?} row count", table.name),
        )?;
        let mut seen_columns = BTreeSet::new();
        let mut columns = reserved(table.columns.len(), "retained eager columns")?;
        for (column_index, column) in table.columns.iter().enumerate() {
            if column.name.is_empty() {
                return Err(invalid(format!(
                    "retained table {:?} column {column_index} has an empty name",
                    table.name
                )));
            }
            if !seen_columns.insert(column.name) {
                return Err(invalid(format!(
                    "retained table {:?} repeats column {:?}",
                    table.name, column.name
                )));
            }
            if column.elements_per_row == 0 {
                return Err(invalid(format!(
                    "retained column {}.{} has zero elements per row",
                    table.name, column.name
                )));
            }
            let expected_elements = table
                .row_count
                .checked_mul(u64::from(column.elements_per_row))
                .ok_or_else(|| {
                    invalid(format!(
                        "retained column {}.{} shape exceeds u64",
                        table.name, column.name
                    ))
                })?;
            if usize_u64(column.values.len(), "retained column element count")? != expected_elements
            {
                return Err(invalid(format!(
                    "retained column {}.{} has {} elements, expected {expected_elements}",
                    table.name,
                    column.name,
                    column.values.len()
                )));
            }
            let values = match column.values {
                EagerPrimitiveColumnView::U8(values) => {
                    EagerOwnedPrimitiveColumn::U8(copy_slice(values, "retained u8 column")?)
                }
                EagerPrimitiveColumnView::U32(values) => {
                    EagerOwnedPrimitiveColumn::U32(copy_slice(values, "retained u32 column")?)
                }
                EagerPrimitiveColumnView::U64(values) => {
                    EagerOwnedPrimitiveColumn::U64(copy_slice(values, "retained u64 column")?)
                }
                EagerPrimitiveColumnView::I32(values) => {
                    EagerOwnedPrimitiveColumn::I32(copy_slice(values, "retained i32 column")?)
                }
                EagerPrimitiveColumnView::F64(values) => {
                    let mut bits = reserved(values.len(), "retained f64 column")?;
                    for (row, value) in values.iter().copied().enumerate() {
                        if !value.is_finite() {
                            return Err(invalid(format!(
                                "retained column {}.{} element {row} is not finite",
                                table.name, column.name
                            )));
                        }
                        bits.push(value.to_bits());
                    }
                    EagerOwnedPrimitiveColumn::F64Bits(bits)
                }
            };
            columns.push(EagerOwnedRetainedColumn {
                name: copy_boxed_str(column.name, "retained column name", column_index)?,
                elements_per_row: column.elements_per_row,
                values,
            });
        }
        owned_tables.push(EagerOwnedRetainedTable {
            name: copy_boxed_str(table.name, "retained table name", table_index)?,
            row_count: table.row_count,
            columns,
        });
    }
    Ok(owned_tables)
}

fn validate_retained_table_counts(input: &EagerLoweringInputV1) -> RusticolResult<()> {
    for table in &input.retained_tables {
        let expected = match table.name() {
            "bitset_ranges" => Some(input.bitset_ranges.len()),
            "bitset_words" => Some(input.bitset_words.len()),
            "coherent_groups" => Some(input.coherent_groups.len()),
            "color_contraction_entries" => Some(input.color_contraction_entries.len()),
            "color_contraction_metadata"
            | "metadata"
            | "lc_replay_metadata"
            | "helicity_proof_metadata"
            | "helicity_materialization_metadata" => Some(1),
            "color_selectors" => Some(input.color_selectors.len()),
            "contraction_coefficients" => Some(input.contraction_coefficients.len()),
            "couplings" => Some(input.couplings.len()),
            "currents" => Some(input.currents.len()),
            "exact_factors" => Some(input.exact_factors.len()),
            "helicity_selectors" => Some(input.helicity_selectors.len()),
            "i32_sequence_ranges" => Some(input.i32_sequence_ranges.len()),
            "i32_sequence_values" => Some(input.i32_sequence_values.len()),
            "interaction_group_members" => Some(input.interaction_group_members.len()),
            "interaction_groups" => Some(input.interaction_groups.len()),
            "interactions" => Some(input.interactions.len()),
            "model_parameters" => Some(input.model_parameters.len()),
            "momentum_masks" => Some(input.momentum_masks.len()),
            "reduction_members" => Some(input.reduction_members.len()),
            "roots" => Some(input.roots.len()),
            "sources" => Some(input.sources.len()),
            "u32_sequence_ranges" => Some(input.u32_sequence_ranges.len()),
            "u32_sequence_values" => Some(input.u32_sequence_values.len()),
            _ => None,
        };
        if let Some(expected) = expected
            && table.row_count != usize_u64(expected, "retained table expected row count")?
        {
            return Err(invalid(format!(
                "retained table {:?} has {} rows, expected {expected}",
                table.name(),
                table.row_count
            )));
        }
    }
    Ok(())
}

fn copy_string_catalog(values: &[&str], context: &str) -> RusticolResult<Vec<Box<str>>> {
    let mut seen = BTreeSet::new();
    for (index, value) in values.iter().copied().enumerate() {
        if !seen.insert(value) {
            return Err(invalid(format!(
                "{context} contains duplicate value at row {index}"
            )));
        }
    }
    let mut result = reserved(values.len(), context)?;
    for (index, value) in values.iter().copied().enumerate() {
        result.push(copy_boxed_str(value, context, index)?);
    }
    Ok(result)
}

fn copy_string_list(values: &[&str], context: &str) -> RusticolResult<Vec<Box<str>>> {
    let mut result = reserved(values.len(), context)?;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_empty() {
            return Err(invalid(format!("{context} row {index} must not be empty")));
        }
        result.push(copy_boxed_str(value, context, index)?);
    }
    Ok(result)
}

fn copy_boxed_str(value: &str, context: &str, index: usize) -> RusticolResult<Box<str>> {
    let mut owned = String::new();
    owned.try_reserve_exact(value.len()).map_err(|error| {
        invalid(format!(
            "could not reserve {context} row {index} ({} bytes): {error}",
            value.len()
        ))
    })?;
    owned.push_str(value);
    Ok(owned.into_boxed_str())
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[cfg(test)]
#[path = "eager_lowering_v3_tests.rs"]
mod tests;
