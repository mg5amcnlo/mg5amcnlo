// SPDX-License-Identifier: 0BSD

//! Compact decoder for authenticated eager plan-v3 PACBIN containers.
//!
//! The decoder borrows each PACBIN member while checking its fixed-width wire
//! contract, then allocates only the final typed runtime/exact structures.  It
//! deliberately does not recreate the former expanded JSON `runtime_schema`.

use super::eager_manifest::PreparedKernelPackManifest;
use super::eager_v3_manifest::EagerV3ExecutionManifest;
use crate::eager_layout::{
    EAGER_LOWERING_INPUT_ABI, EAGER_PLAN_ABI, EAGER_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_CONTAINER_KIND, EAGER_RUNTIME_CONTAINER_SCHEMA, EAGER_RUNTIME_LAYOUT_ABI,
    EagerSectionHeader, EagerSectionKind,
};
use crate::eager_lowering_v3::{
    EagerPlanAttachmentRow, EagerPlanCatalogRangeRow, EagerPlanClosureRow,
    EagerPlanColorSelectorRow, EagerPlanCouplingRow, EagerPlanCurrentRow,
    EagerPlanDirectCoefficientRow, EagerPlanExactFactorRow, EagerPlanFinalizationRow,
    EagerPlanHelicitySelectorRow, EagerPlanInvocationRow, EagerPlanMomentumRow,
    EagerPlanParameterRow, EagerPlanReductionEntryKind, EagerPlanReductionEntryRow,
    EagerPlanReductionGroupRow, EagerPlanSelectorDomainRow, EagerPlanSourceFillRow,
    EagerPlanStageRow, EagerPlanValueRow, EagerValueSlotKind,
};
use crate::eager_runtime::{EagerKernelRole, EagerKernelSpec, EagerPlanDimensions};
use crate::pacbin::PacbinReader;
use crate::{MISSING_U32, RusticolError, RusticolResult};
use std::collections::{BTreeMap, BTreeSet, HashSet};

const SERIALIZATION_SCHEMA: u32 = 1;
const METADATA_RECORD_SIZE: u32 = 224;
const INSPECTION_RECORD_SIZE: u32 = 160;
const RETAINED_TABLE_RECORD_SIZE: u32 = 40;
const RETAINED_COLUMN_RECORD_SIZE: u32 = 40;
const DIRECT_COEFFICIENT_RECORD_SIZE: u32 = 16;
const HELICITY_SELECTOR_RECORD_SIZE: u32 = 24;
const COLOR_SELECTOR_RECORD_SIZE: u32 = 24;

const RETAINED_U8: u8 = 1;
const RETAINED_U32: u8 = 2;
const RETAINED_U64: u8 = 3;
const RETAINED_I32: u8 = 4;
const RETAINED_F64_BITS: u8 = 5;

const IDENTITY_PROCESS_KEY: usize = 5;
const IDENTITY_MODEL_NAME: usize = 6;
const IDENTITY_COUNT: usize = 7;

/// One retained primitive column used by exact execution and inspection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum DecodedEagerPrimitiveColumn {
    U8(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
    /// Raw binary64 bits preserve signed zero and NaN payloads.
    F64Bits(Vec<u64>),
    /// Authenticated and shape-validated retained data not needed by f64 load.
    ValidatedOnly {
        len: usize,
    },
}

impl DecodedEagerPrimitiveColumn {
    fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) | Self::F64Bits(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::ValidatedOnly { len } => *len,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecodedEagerRetainedColumn {
    pub(super) name: Box<str>,
    pub(super) elements_per_row: u32,
    pub(super) values: DecodedEagerPrimitiveColumn,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecodedEagerRetainedTable {
    pub(super) name: Box<str>,
    pub(super) row_count: u64,
    pub(super) columns: Vec<DecodedEagerRetainedColumn>,
}

/// Final compact owned form consumed by the plan-v3 runtime adapter.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DecodedEagerRuntimeV3 {
    pub(super) process_key: Box<str>,
    pub(super) model_name: Box<str>,
    pub(super) runtime_options: DecodedEagerRuntimeOptions,
    pub(super) dimensions: EagerPlanDimensions,
    pub(super) kernel_specs: Vec<EagerKernelSpec>,
    pub(super) strings: Vec<Box<str>>,
    pub(super) exact_ir: Vec<Box<str>>,
    pub(super) semantic_limitations: Vec<Box<str>>,
    pub(super) retained_tables: Vec<DecodedEagerRetainedTable>,
    pub(super) currents: Vec<EagerPlanCurrentRow>,
    pub(super) values: Vec<EagerPlanValueRow>,
    pub(super) momenta: Vec<EagerPlanMomentumRow>,
    pub(super) sources: Vec<EagerPlanSourceFillRow>,
    pub(super) parameters: Vec<EagerPlanParameterRow>,
    pub(super) stages: Vec<EagerPlanStageRow>,
    pub(super) couplings: Vec<EagerPlanCouplingRow>,
    pub(super) invocations: Vec<EagerPlanInvocationRow>,
    pub(super) attachments: Vec<EagerPlanAttachmentRow>,
    pub(super) finalizations: Vec<EagerPlanFinalizationRow>,
    pub(super) closures: Vec<EagerPlanClosureRow>,
    pub(super) direct_coefficients: Vec<EagerPlanDirectCoefficientRow>,
    pub(super) selector_domains: Vec<EagerPlanSelectorDomainRow>,
    pub(super) selector_memberships: Vec<u32>,
    pub(super) helicity_selectors: Vec<EagerPlanHelicitySelectorRow>,
    pub(super) color_selectors: Vec<EagerPlanColorSelectorRow>,
    pub(super) reduction_groups: Vec<EagerPlanReductionGroupRow>,
    pub(super) reduction_entries: Vec<EagerPlanReductionEntryRow>,
    pub(super) exact_factors: Vec<EagerPlanExactFactorRow>,
    pub(super) bitset_ranges: Vec<EagerPlanCatalogRangeRow>,
    pub(super) bitset_populations: Vec<u64>,
    pub(super) bitset_words: Vec<u64>,
    pub(super) u32_sequence_ranges: Vec<EagerPlanCatalogRangeRow>,
    pub(super) u32_sequence_values: Vec<u32>,
    pub(super) i32_sequence_ranges: Vec<EagerPlanCatalogRangeRow>,
    pub(super) i32_sequence_values: Vec<i32>,
    pub(super) color_contraction_entry_start: u64,
    pub(super) color_contraction_entry_count: u64,
}

/// Reduction-only owned form used to hydrate compact public physics metadata.
///
/// The enclosing PACBIN has already been authenticated in full. This view
/// materializes only reduction rows, selector rows, and selector-referenced
/// sequence values; all unrelated runtime tables and global sequence-catalog
/// entries are limited to borrowed fixed-width validation.
#[derive(Debug)]
pub(super) struct DecodedEagerReductionGroupsV3 {
    pub(super) helicity_selectors: Vec<EagerPlanHelicitySelectorRow>,
    pub(super) color_selectors: Vec<EagerPlanColorSelectorRow>,
    pub(super) reduction_groups: Vec<EagerPlanReductionGroupRow>,
    pub(super) reduction_entries: Vec<EagerPlanReductionEntryRow>,
    pub(super) helicity_sequences: BTreeMap<u32, Vec<i32>>,
    pub(super) color_sequences: BTreeMap<u32, Vec<u32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DecodedEagerRuntimeOptions {
    pub(super) point_tile_size: usize,
    pub(super) workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct Metadata {
    retained_column_count: u64,
    current_component_count: u64,
    value_component_count: u64,
    momentum_component_count: u64,
    color_contraction_entry_start: u64,
    color_contraction_entry_count: u64,
    string_count: u64,
    exact_ir_count: u64,
    limitation_count: u64,
    retained_table_count: u64,
    current_count: u64,
    value_count: u64,
    momentum_count: u64,
    source_count: u64,
    stage_count: u64,
    invocation_count: u64,
    closure_count: u64,
    direct_coefficient_count: u64,
    helicity_selector_count: u64,
    color_selector_count: u64,
    exact_factor_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct RetainedTableRecord {
    table_id: u32,
    name_id: u32,
    row_count: u64,
    column_start: u64,
    column_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct RetainedColumnRecord {
    table_id: u32,
    column_id: u32,
    name_id: u32,
    primitive_kind: u8,
    elements_per_row: u32,
    value_start: u64,
    value_count: u64,
}

struct RetainedPrimitiveSections<'a> {
    u8_values: DecodedSection<'a>,
    u32_values: DecodedSection<'a>,
    u64_values: DecodedSection<'a>,
    i32_values: DecodedSection<'a>,
    f64_values: DecodedSection<'a>,
}

#[derive(Clone, Copy)]
struct CountedPlanSection {
    path: &'static str,
    kind: EagerSectionKind,
    record_size: u32,
}

const COUNTED_PLAN_SECTIONS: [CountedPlanSection; 20] = [
    CountedPlanSection {
        path: "tables/currents.bin",
        kind: EagerSectionKind::CurrentLayout,
        record_size: EagerPlanCurrentRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/values.bin",
        kind: EagerSectionKind::ValueLayout,
        record_size: EagerPlanValueRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/momenta.bin",
        kind: EagerSectionKind::MomentumLayout,
        record_size: EagerPlanMomentumRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/sources.bin",
        kind: EagerSectionKind::SourceFill,
        record_size: EagerPlanSourceFillRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/parameters.bin",
        kind: EagerSectionKind::ParameterLayout,
        record_size: EagerPlanParameterRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/stages.bin",
        kind: EagerSectionKind::Stages,
        record_size: EagerPlanStageRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/couplings.bin",
        kind: EagerSectionKind::Couplings,
        record_size: EagerPlanCouplingRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/invocations.bin",
        kind: EagerSectionKind::Invocations,
        record_size: EagerPlanInvocationRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/attachments.bin",
        kind: EagerSectionKind::Attachments,
        record_size: EagerPlanAttachmentRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/finalizations.bin",
        kind: EagerSectionKind::Finalizations,
        record_size: EagerPlanFinalizationRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/closures.bin",
        kind: EagerSectionKind::Closures,
        record_size: EagerPlanClosureRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "tables/direct-coefficients.bin",
        kind: EagerSectionKind::Closures,
        record_size: DIRECT_COEFFICIENT_RECORD_SIZE,
    },
    CountedPlanSection {
        path: "selectors/domains.bin",
        kind: EagerSectionKind::SelectorDomains,
        record_size: EagerPlanSelectorDomainRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "selectors/memberships.bin",
        kind: EagerSectionKind::SelectorMemberships,
        record_size: 4,
    },
    CountedPlanSection {
        path: "selectors/helicities.bin",
        kind: EagerSectionKind::SelectorDomains,
        record_size: HELICITY_SELECTOR_RECORD_SIZE,
    },
    CountedPlanSection {
        path: "selectors/colors.bin",
        kind: EagerSectionKind::SelectorDomains,
        record_size: COLOR_SELECTOR_RECORD_SIZE,
    },
    CountedPlanSection {
        path: "reductions/groups.bin",
        kind: EagerSectionKind::ReductionGroups,
        record_size: EagerPlanReductionGroupRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "reductions/entries.bin",
        kind: EagerSectionKind::ReductionEntries,
        record_size: EagerPlanReductionEntryRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "catalogs/exact-factors.bin",
        kind: EagerSectionKind::ExactFactors,
        record_size: EagerPlanExactFactorRow::ENCODED_LEN,
    },
    CountedPlanSection {
        path: "retained/tables.bin",
        kind: EagerSectionKind::Metadata,
        record_size: RETAINED_TABLE_RECORD_SIZE,
    },
];

/// Decode only the authenticated rows required to hydrate native reduction
/// groups. Runtime-sized DAG tables are never converted into Rust rows.
pub(super) fn decode_eager_v3_reduction_groups(
    reader: &PacbinReader,
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<DecodedEagerReductionGroupsV3> {
    let metadata = decode_metadata(reader)?;
    let inspection = decode_inspection(reader)?;
    let identity = decode_text_catalog(reader, "metadata/identity", EagerSectionKind::Metadata)?;
    validate_identity(&identity, manifest)?;
    validate_reduction_plan_metadata(reader, manifest, metadata, &inspection)?;

    let helicity_selectors = decode_helicity_selectors(reader)?;
    let color_selectors = decode_color_selectors(reader)?;
    let reduction_groups = decode_reduction_groups(reader)?;
    let reduction_entries = decode_reduction_entries(reader)?;
    check_count(
        "reduction-only helicity selector",
        helicity_selectors.len(),
        inspection[14],
    )?;
    check_count(
        "reduction-only color selector",
        color_selectors.len(),
        inspection[15],
    )?;
    check_count(
        "reduction-only reduction group",
        reduction_groups.len(),
        inspection[16],
    )?;
    check_count(
        "reduction-only reduction entry",
        reduction_entries.len(),
        inspection[17],
    )?;

    let helicity_sequence_ids = helicity_selectors
        .iter()
        .flat_map(|row| [row.values_sequence_id, row.representative_sequence_id])
        .collect::<BTreeSet<_>>();
    let color_sequence_ids = color_selectors
        .iter()
        .flat_map(|row| [row.word_sequence_id, row.representative_word_sequence_id])
        .collect::<BTreeSet<_>>();
    let helicity_sequences = decode_referenced_sequences(
        reader,
        "catalogs/i32-sequences/ranges.bin",
        "catalogs/i32-sequences/values.bin",
        &helicity_sequence_ids,
        "helicity i32 sequence",
        |row| read_i32(row, 0),
    )?;
    let color_sequences = decode_referenced_sequences(
        reader,
        "catalogs/u32-sequences/ranges.bin",
        "catalogs/u32-sequences/values.bin",
        &color_sequence_ids,
        "color u32 sequence",
        |row| read_u32(row, 0),
    )?;
    validate_reduction_subset_semantics(
        metadata,
        inspection[18],
        &helicity_selectors,
        &color_selectors,
        &reduction_groups,
        &reduction_entries,
        &helicity_sequences,
        &color_sequences,
    )?;

    Ok(DecodedEagerReductionGroupsV3 {
        helicity_selectors,
        color_selectors,
        reduction_groups,
        reduction_entries,
        helicity_sequences,
        color_sequences,
    })
}

/// Decode one already authenticated/preflighted eager plan-v3 container.
///
/// `pack` must be the validated prepared pack referenced by the execution
/// manifest.  The decoder resolves its final kernel specs and rejects every
/// plan reference whose prepared-kernel role is incompatible.
pub(super) fn decode_eager_v3_runtime(
    reader: &PacbinReader,
    manifest: &EagerV3ExecutionManifest,
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<DecodedEagerRuntimeV3> {
    let metadata = decode_metadata(reader)?;
    let inspection = decode_inspection(reader)?;
    let identity = decode_text_catalog(reader, "metadata/identity", EagerSectionKind::Metadata)?;
    validate_identity(&identity, manifest)?;

    let strings = decode_text_catalog(reader, "catalogs/strings", EagerSectionKind::Metadata)?;
    let exact_ir = decode_text_catalog(reader, "catalogs/exact-ir", EagerSectionKind::Metadata)?;
    let semantic_limitations = decode_text_catalog(
        reader,
        "catalogs/semantic-limitations",
        EagerSectionKind::Metadata,
    )?;
    check_count("string catalog", strings.len(), metadata.string_count)?;
    check_count("exact IR catalog", exact_ir.len(), metadata.exact_ir_count)?;
    check_count(
        "semantic limitation catalog",
        semantic_limitations.len(),
        metadata.limitation_count,
    )?;

    let retained_tables = decode_retained_tables(reader)?;
    check_count(
        "retained table catalog",
        retained_tables.len(),
        metadata.retained_table_count,
    )?;
    let retained_column_count = retained_tables.iter().try_fold(0_usize, |total, table| {
        total
            .checked_add(table.columns.len())
            .ok_or_else(|| RusticolError::artifact("decoded retained column count exceeds usize"))
    })?;
    check_count(
        "retained column catalog",
        retained_column_count,
        metadata.retained_column_count,
    )?;

    let currents = decode_currents(reader)?;
    let values = decode_values(reader)?;
    let momenta = decode_momenta(reader)?;
    let sources = decode_sources(reader)?;
    let parameters = decode_parameters(reader)?;
    let stages = decode_stages(reader)?;
    let couplings = decode_couplings(reader)?;
    let invocations = decode_invocations(reader)?;
    let attachments = decode_attachments(reader)?;
    let finalizations = decode_finalizations(reader)?;
    let closures = decode_closures(reader)?;
    let direct_coefficients = decode_direct_coefficients(reader)?;
    let selector_domains = decode_selector_domains(reader)?;
    let selector_memberships = decode_u32_section(
        reader,
        "selectors/memberships.bin",
        EagerSectionKind::SelectorMemberships,
    )?;
    let helicity_selectors = decode_helicity_selectors(reader)?;
    let color_selectors = decode_color_selectors(reader)?;
    let reduction_groups = decode_reduction_groups(reader)?;
    let reduction_entries = decode_reduction_entries(reader)?;
    let exact_factors = decode_exact_factors(reader)?;
    let bitset_ranges = decode_catalog_ranges(reader, "catalogs/bitsets/ranges.bin")?;
    let bitset_populations = decode_u64_section(
        reader,
        "catalogs/bitsets/populations.bin",
        EagerSectionKind::Metadata,
    )?;
    let bitset_words = decode_u64_section(
        reader,
        "catalogs/bitsets/words.bin",
        EagerSectionKind::Metadata,
    )?;
    let u32_sequence_ranges = decode_catalog_ranges(reader, "catalogs/u32-sequences/ranges.bin")?;
    let u32_sequence_values = decode_u32_section(
        reader,
        "catalogs/u32-sequences/values.bin",
        EagerSectionKind::Metadata,
    )?;
    let i32_sequence_ranges = decode_catalog_ranges(reader, "catalogs/i32-sequences/ranges.bin")?;
    let i32_sequence_values = decode_i32_section(
        reader,
        "catalogs/i32-sequences/values.bin",
        EagerSectionKind::Metadata,
    )?;

    validate_declared_counts(
        manifest,
        metadata,
        &inspection,
        &currents,
        &values,
        &momenta,
        &sources,
        &parameters,
        &stages,
        &couplings,
        &invocations,
        &attachments,
        &finalizations,
        &closures,
        &direct_coefficients,
        &selector_domains,
        &selector_memberships,
        &helicity_selectors,
        &color_selectors,
        &reduction_groups,
        &reduction_entries,
        &exact_factors,
        retained_tables.len(),
    )?;

    validate_catalog_ranges("bitset", &bitset_ranges, bitset_words.len())?;
    if bitset_ranges.len() != bitset_populations.len() {
        return Err(integrity(
            "bitset range and population catalogs have different lengths",
        ));
    }
    for (index, (range, expected)) in bitset_ranges.iter().zip(&bitset_populations).enumerate() {
        let words = checked_range(&bitset_words, range.start, range.count, "bitset words")?;
        let actual = words
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum::<u64>();
        if actual != *expected {
            return Err(integrity(format!(
                "bitset {index} population {expected} does not match its {actual} set bits"
            )));
        }
    }
    validate_catalog_ranges(
        "u32 sequence",
        &u32_sequence_ranges,
        u32_sequence_values.len(),
    )?;
    validate_catalog_ranges(
        "i32 sequence",
        &i32_sequence_ranges,
        i32_sequence_values.len(),
    )?;

    let mut kernel_specs = pack.kernel_specs()?;
    validate_semantics(
        metadata,
        manifest.dag_summary.interaction_count,
        &currents,
        &values,
        &momenta,
        &sources,
        &parameters,
        &stages,
        &couplings,
        &invocations,
        &attachments,
        &finalizations,
        &closures,
        &direct_coefficients,
        &selector_domains,
        &selector_memberships,
        &helicity_selectors,
        &color_selectors,
        &reduction_groups,
        &reduction_entries,
        &exact_factors,
        &bitset_ranges,
        &u32_sequence_ranges,
        &i32_sequence_ranges,
        &strings,
        &exact_ir,
        &kernel_specs,
    )?;
    let active_kernel_ids = active_kernel_ids(
        invocations.iter().map(|row| row.kernel_id),
        finalizations.iter().map(|row| row.kernel_id),
        closures.iter().map(|row| row.kernel_id),
    );
    kernel_specs.retain(|kernel| active_kernel_ids.contains(&kernel.kernel_id));

    let amplitude_count = closures
        .iter()
        .map(|row| row.amplitude_index)
        .max()
        .map_or(Ok(0_u32), |maximum| {
            maximum
                .checked_add(1)
                .ok_or_else(|| RusticolError::artifact("eager amplitude count exceeds u32"))
        })?;
    if amplitude_count == 0 {
        return Err(integrity("eager plan-v3 has no amplitude closures"));
    }
    let dimensions = EagerPlanDimensions {
        value_slot_component_counts: values.iter().map(|row| row.component_count).collect(),
        momentum_slot_component_counts: momenta.iter().map(|row| row.component_count).collect(),
        current_component_counts: currents.iter().map(|row| row.component_count).collect(),
        parameter_count: usize_u32(parameters.len(), "parameter count")?,
        amplitude_count,
    };
    let workspace_mib = usize::try_from(manifest.runtime_options.workspace_mib)
        .map_err(|_| RusticolError::artifact("eager workspace MiB does not fit usize"))?;
    let workspace_bytes = workspace_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| RusticolError::artifact("eager workspace byte count exceeds usize"))?;

    Ok(DecodedEagerRuntimeV3 {
        process_key: identity[IDENTITY_PROCESS_KEY].clone(),
        model_name: identity[IDENTITY_MODEL_NAME].clone(),
        runtime_options: DecodedEagerRuntimeOptions {
            point_tile_size: usize::try_from(manifest.runtime_options.point_tile_size)
                .map_err(|_| RusticolError::artifact("eager point tile size does not fit usize"))?,
            workspace_bytes,
        },
        dimensions,
        kernel_specs,
        strings,
        exact_ir,
        semantic_limitations,
        retained_tables,
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
        bitset_words,
        u32_sequence_ranges,
        u32_sequence_values,
        i32_sequence_ranges,
        i32_sequence_values,
        color_contraction_entry_start: metadata.color_contraction_entry_start,
        color_contraction_entry_count: metadata.color_contraction_entry_count,
    })
}

fn active_kernel_ids(
    invocation_ids: impl IntoIterator<Item = u32>,
    finalization_ids: impl IntoIterator<Item = u32>,
    closure_ids: impl IntoIterator<Item = u32>,
) -> BTreeSet<u32> {
    invocation_ids
        .into_iter()
        .chain(finalization_ids)
        .chain(closure_ids)
        .filter(|kernel_id| *kernel_id != MISSING_U32)
        .collect()
}

fn decode_metadata(reader: &PacbinReader) -> RusticolResult<Metadata> {
    let section = section(
        reader,
        "metadata/core.bin",
        EagerSectionKind::Metadata,
        METADATA_RECORD_SIZE,
    )?;
    require_count(&section, 1, "eager metadata")?;
    let row = section.payload;
    if read_u32(row, 0)? != SERIALIZATION_SCHEMA
        || read_u16(row, 4)? != EAGER_RUNTIME_CONTAINER_SCHEMA
    {
        return Err(RusticolError::compatibility(
            "unsupported eager plan-v3 serialization schema",
        ));
    }
    require_zero(row, 6, 8, "metadata reserved bytes")?;
    let identity_ids = [
        read_u32(row, 8)?,
        read_u32(row, 12)?,
        read_u32(row, 16)?,
        read_u32(row, 20)?,
        read_u32(row, 24)?,
        read_u32(row, 28)?,
        read_u32(row, 32)?,
    ];
    if identity_ids != [0, 1, 2, 3, 4, 5, 6] || read_u32(row, 36)? != 7 {
        return Err(integrity(
            "eager metadata identity catalog mapping is not canonical",
        ));
    }
    let mut values = [0_u64; 21];
    for (index, value) in values.iter_mut().enumerate() {
        *value = read_u64(row, 40 + index * 8)?;
    }
    require_zero(row, 208, 224, "metadata trailing reserved bytes")?;
    Ok(Metadata {
        retained_column_count: values[0],
        current_component_count: values[1],
        value_component_count: values[2],
        momentum_component_count: values[3],
        color_contraction_entry_start: values[4],
        color_contraction_entry_count: values[5],
        string_count: values[6],
        exact_ir_count: values[7],
        limitation_count: values[8],
        retained_table_count: values[9],
        current_count: values[10],
        value_count: values[11],
        momentum_count: values[12],
        source_count: values[13],
        stage_count: values[14],
        invocation_count: values[15],
        closure_count: values[16],
        direct_coefficient_count: values[17],
        helicity_selector_count: values[18],
        color_selector_count: values[19],
        exact_factor_count: values[20],
    })
}

fn decode_inspection(reader: &PacbinReader) -> RusticolResult<[u64; 20]> {
    let section = section(
        reader,
        "inspection/summary.bin",
        EagerSectionKind::InspectionSummary,
        INSPECTION_RECORD_SIZE,
    )?;
    require_count(&section, 1, "eager inspection summary")?;
    let mut values = [0_u64; 20];
    for (index, value) in values.iter_mut().enumerate() {
        *value = read_u64(section.payload, index * 8)?;
    }
    Ok(values)
}

fn validate_identity(
    identity: &[Box<str>],
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<()> {
    if identity.len() != IDENTITY_COUNT {
        return Err(integrity(
            "eager identity catalog must contain exactly seven entries",
        ));
    }
    let expected = [
        EAGER_LOWERING_INPUT_ABI,
        EAGER_PLAN_ABI,
        EAGER_RUNTIME_LAYOUT_ABI,
        EAGER_RUNTIME_CAPABILITY,
        EAGER_RUNTIME_CONTAINER_KIND,
        manifest.key.as_str(),
        manifest.plan.inspection_summary.model_name.as_str(),
    ];
    for (index, expected) in expected.into_iter().enumerate() {
        if identity[index].as_ref() != expected {
            return Err(integrity(format!(
                "eager identity entry {index} does not match its manifest contract"
            )));
        }
    }
    Ok(())
}

fn decode_text_catalog(
    reader: &PacbinReader,
    base: &str,
    kind: EagerSectionKind,
) -> RusticolResult<Vec<Box<str>>> {
    let (ranges_path, bytes_path) = match base {
        "metadata/identity" | "retained/name" => {
            (format!("{base}-ranges.bin"), format!("{base}-bytes.bin"))
        }
        _ => (format!("{base}/ranges.bin"), format!("{base}/bytes.bin")),
    };
    let ranges = decode_catalog_ranges(reader, &ranges_path)?;
    let bytes = section(reader, &bytes_path, kind, 1)?;
    validate_catalog_ranges(base, &ranges, bytes.payload.len())?;
    let mut result = Vec::new();
    reserve(&mut result, ranges.len(), base)?;
    for range in ranges {
        let value = checked_range(bytes.payload, range.start, range.count, base)?;
        let text = std::str::from_utf8(value).map_err(|error| {
            RusticolError::integrity(format!("{base} contains invalid UTF-8: {error}"))
        })?;
        result.push(Box::<str>::from(text));
    }
    Ok(result)
}

fn decode_retained_tables(reader: &PacbinReader) -> RusticolResult<Vec<DecodedEagerRetainedTable>> {
    let names = decode_text_catalog(reader, "retained/name", EagerSectionKind::Metadata)?;
    let tables = decode_rows(
        reader,
        "retained/tables.bin",
        EagerSectionKind::Metadata,
        RETAINED_TABLE_RECORD_SIZE,
        |row| {
            require_zero(row, 32, 40, "retained table padding")?;
            Ok(RetainedTableRecord {
                table_id: read_u32(row, 0)?,
                name_id: read_u32(row, 4)?,
                row_count: read_u64(row, 8)?,
                column_start: read_u64(row, 16)?,
                column_count: read_u64(row, 24)?,
            })
        },
    )?;
    let columns = decode_rows(
        reader,
        "retained/columns.bin",
        EagerSectionKind::Metadata,
        RETAINED_COLUMN_RECORD_SIZE,
        |row| {
            require_zero(row, 13, 16, "retained column padding")?;
            require_zero(row, 20, 24, "retained column padding")?;
            Ok(RetainedColumnRecord {
                table_id: read_u32(row, 0)?,
                column_id: read_u32(row, 4)?,
                name_id: read_u32(row, 8)?,
                primitive_kind: read_u8(row, 12)?,
                elements_per_row: read_u32(row, 16)?,
                value_start: read_u64(row, 24)?,
                value_count: read_u64(row, 32)?,
            })
        },
    )?;
    let primitive_sections = RetainedPrimitiveSections {
        u8_values: section(
            reader,
            "retained/values-u8.bin",
            EagerSectionKind::Metadata,
            1,
        )?,
        u32_values: section(
            reader,
            "retained/values-u32.bin",
            EagerSectionKind::Metadata,
            4,
        )?,
        u64_values: section(
            reader,
            "retained/values-u64.bin",
            EagerSectionKind::Metadata,
            8,
        )?,
        i32_values: section(
            reader,
            "retained/values-i32.bin",
            EagerSectionKind::Metadata,
            4,
        )?,
        f64_values: section(
            reader,
            "retained/values-f64-bits.bin",
            EagerSectionKind::Metadata,
            8,
        )?,
    };

    let mut primitive_cursors = BTreeMap::from([
        (RETAINED_U8, 0_u64),
        (RETAINED_U32, 0),
        (RETAINED_U64, 0),
        (RETAINED_I32, 0),
        (RETAINED_F64_BITS, 0),
    ]);
    let mut result = Vec::new();
    reserve(&mut result, tables.len(), "retained tables")?;
    let mut column_cursor = 0_u64;
    let mut name_cursor = 0_u32;
    for (table_index, table) in tables.iter().enumerate() {
        if table.table_id != usize_u32(table_index, "retained table ID")?
            || table.column_start != column_cursor
            || table.name_id != name_cursor
        {
            return Err(integrity(
                "retained table descriptors are not canonical and contiguous",
            ));
        }
        let table_columns = checked_range(
            &columns,
            table.column_start,
            table.column_count,
            "retained table columns",
        )?;
        let mut decoded_columns = Vec::new();
        reserve(
            &mut decoded_columns,
            table_columns.len(),
            "retained table columns",
        )?;
        name_cursor = name_cursor
            .checked_add(1)
            .ok_or_else(|| RusticolError::artifact("retained name count exceeds u32"))?;
        let table_name: Box<str> =
            required_catalog(&names, table.name_id, "retained table name")?.into();
        for (column_index, column) in table_columns.iter().enumerate() {
            if column.table_id != table.table_id
                || column.column_id != usize_u32(column_index, "retained column ID")?
                || column.name_id != name_cursor
                || column.elements_per_row == 0
            {
                return Err(integrity("retained column descriptor is not canonical"));
            }
            let expected_count = table
                .row_count
                .checked_mul(u64::from(column.elements_per_row))
                .ok_or_else(|| RusticolError::artifact("retained column shape exceeds u64"))?;
            if expected_count != column.value_count {
                return Err(integrity(
                    "retained column shape does not match its value count",
                ));
            }
            let cursor = primitive_cursors
                .get_mut(&column.primitive_kind)
                .ok_or_else(|| {
                    RusticolError::compatibility(format!(
                        "unknown retained primitive kind {}",
                        column.primitive_kind
                    ))
                })?;
            if column.value_start != *cursor {
                return Err(integrity("retained primitive ranges are not contiguous"));
            }
            let column_name: Box<str> =
                required_catalog(&names, column.name_id, "retained column name")?.into();
            let values = decode_retained_column_values(
                &primitive_sections,
                column,
                retained_column_is_runtime_required(&table_name, &column_name),
            )?;
            if values.len() != usize_count(column.value_count, "retained value count")? {
                return Err(integrity(
                    "retained decoded value count changed unexpectedly",
                ));
            }
            decoded_columns.push(DecodedEagerRetainedColumn {
                name: column_name,
                elements_per_row: column.elements_per_row,
                values,
            });
            *cursor = cursor
                .checked_add(column.value_count)
                .ok_or_else(|| RusticolError::artifact("retained primitive cursor exceeds u64"))?;
            name_cursor = name_cursor
                .checked_add(1)
                .ok_or_else(|| RusticolError::artifact("retained name count exceeds u32"))?;
        }
        column_cursor = column_cursor
            .checked_add(table.column_count)
            .ok_or_else(|| RusticolError::artifact("retained column cursor exceeds u64"))?;
        result.push(DecodedEagerRetainedTable {
            name: table_name,
            row_count: table.row_count,
            columns: decoded_columns,
        });
    }
    if usize_count(column_cursor, "retained column cursor")? != columns.len()
        || usize::try_from(name_cursor).ok() != Some(names.len())
        || primitive_cursors[&RETAINED_U8] != primitive_sections.u8_values.count
        || primitive_cursors[&RETAINED_U32] != primitive_sections.u32_values.count
        || primitive_cursors[&RETAINED_U64] != primitive_sections.u64_values.count
        || primitive_cursors[&RETAINED_I32] != primitive_sections.i32_values.count
        || primitive_cursors[&RETAINED_F64_BITS] != primitive_sections.f64_values.count
    {
        return Err(integrity(
            "retained catalogs contain unreachable trailing data",
        ));
    }
    Ok(result)
}

fn retained_column_is_runtime_required(table: &str, column: &str) -> bool {
    if table.starts_with("lc_replay_") || table.starts_with("color_replay_") {
        return true;
    }
    matches!(
        (table, column),
        ("metadata", "normalization_ir_id")
            | ("coherent_groups", "color_sector_id")
            | ("model_parameters", "pdg")
            | ("currents", "particle_id")
            | ("currents", "source_leg_label")
            | ("currents", "source_helicity")
            | ("currents", "chirality")
            | ("currents", "spin_state_sequence_id")
            | ("currents", "helicity_ancestry_bitset_id")
    )
}

fn decode_retained_column_values(
    sections: &RetainedPrimitiveSections<'_>,
    column: &RetainedColumnRecord,
    materialize: bool,
) -> RusticolResult<DecodedEagerPrimitiveColumn> {
    let (section, width, context) = match column.primitive_kind {
        RETAINED_U8 => (&sections.u8_values, 1_usize, "retained u8"),
        RETAINED_U32 => (&sections.u32_values, 4, "retained u32"),
        RETAINED_U64 => (&sections.u64_values, 8, "retained u64"),
        RETAINED_I32 => (&sections.i32_values, 4, "retained i32"),
        RETAINED_F64_BITS => (&sections.f64_values, 8, "retained f64 bits"),
        _ => unreachable!("primitive kind checked before retained-value decoding"),
    };
    let end = column
        .value_start
        .checked_add(column.value_count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range exceeds u64")))?;
    if end > section.count {
        return Err(integrity(format!(
            "{context} range exceeds its primitive catalog"
        )));
    }
    let count = usize_count(column.value_count, context)?;
    if !materialize {
        return Ok(DecodedEagerPrimitiveColumn::ValidatedOnly { len: count });
    }
    let start = usize_count(column.value_start, context)?;
    let byte_start = start
        .checked_mul(width)
        .ok_or_else(|| RusticolError::artifact(format!("{context} byte offset exceeds usize")))?;
    let byte_count = count
        .checked_mul(width)
        .ok_or_else(|| RusticolError::artifact(format!("{context} byte count exceeds usize")))?;
    let byte_end = byte_start
        .checked_add(byte_count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} byte range exceeds usize")))?;
    let payload = section.payload.get(byte_start..byte_end).ok_or_else(|| {
        integrity(format!(
            "{context} byte range exceeds its primitive payload"
        ))
    })?;
    match column.primitive_kind {
        RETAINED_U8 => Ok(DecodedEagerPrimitiveColumn::U8(payload.to_vec())),
        RETAINED_U32 => Ok(DecodedEagerPrimitiveColumn::U32(decode_primitive_rows(
            payload,
            width,
            count,
            context,
            |row| read_u32(row, 0),
        )?)),
        RETAINED_U64 => Ok(DecodedEagerPrimitiveColumn::U64(decode_primitive_rows(
            payload,
            width,
            count,
            context,
            |row| read_u64(row, 0),
        )?)),
        RETAINED_I32 => Ok(DecodedEagerPrimitiveColumn::I32(decode_primitive_rows(
            payload,
            width,
            count,
            context,
            |row| read_i32(row, 0),
        )?)),
        RETAINED_F64_BITS => Ok(DecodedEagerPrimitiveColumn::F64Bits(decode_primitive_rows(
            payload,
            width,
            count,
            context,
            |row| read_u64(row, 0),
        )?)),
        _ => unreachable!("primitive kind checked before retained-value materialization"),
    }
}

fn decode_primitive_rows<T>(
    payload: &[u8],
    width: usize,
    count: usize,
    context: &str,
    mut decode: impl FnMut(&[u8]) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let mut values = Vec::new();
    reserve(&mut values, count, context)?;
    for row in payload.chunks_exact(width) {
        values.push(decode(row)?);
    }
    if values.len() != count {
        return Err(integrity(format!(
            "{context} decoded row count changed unexpectedly"
        )));
    }
    Ok(values)
}

fn decode_currents(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanCurrentRow>> {
    decode_rows(
        reader,
        "tables/currents.bin",
        EagerSectionKind::CurrentLayout,
        EagerPlanCurrentRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanCurrentRow {
                current_id: read_u32(row, 0)?,
                component_start: read_u64(row, 4)?,
                component_count: read_u32(row, 12)?,
                momentum_slot_id: read_u32(row, 16)?,
                flags: read_u32(row, 20)?,
            })
        },
    )
}

fn decode_values(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanValueRow>> {
    decode_rows(
        reader,
        "tables/values.bin",
        EagerSectionKind::ValueLayout,
        EagerPlanValueRow::ENCODED_LEN,
        |row| {
            require_zero(row, 21, 24, "value row padding")?;
            let kind = match read_u8(row, 20)? {
                1 => EagerValueSlotKind::Source,
                2 => EagerValueSlotKind::Unpropagated,
                3 => EagerValueSlotKind::Propagated,
                value => {
                    return Err(RusticolError::compatibility(format!(
                        "unknown eager value-slot kind {value}"
                    )));
                }
            };
            Ok(EagerPlanValueRow {
                value_slot_id: read_u32(row, 0)?,
                current_id: read_u32(row, 4)?,
                component_start: read_u64(row, 8)?,
                component_count: read_u32(row, 16)?,
                kind,
            })
        },
    )
}

fn decode_momenta(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanMomentumRow>> {
    decode_rows(
        reader,
        "tables/momenta.bin",
        EagerSectionKind::MomentumLayout,
        EagerPlanMomentumRow::ENCODED_LEN,
        |row| {
            require_zero(row, 20, 24, "momentum row padding")?;
            Ok(EagerPlanMomentumRow {
                momentum_slot_id: read_u32(row, 0)?,
                bitset_id: read_u32(row, 4)?,
                component_start: read_u64(row, 8)?,
                component_count: read_u32(row, 16)?,
            })
        },
    )
}

fn decode_sources(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanSourceFillRow>> {
    decode_rows(
        reader,
        "tables/sources.bin",
        EagerSectionKind::SourceFill,
        EagerPlanSourceFillRow::ENCODED_LEN,
        |row| {
            require_zero(row, 36, 40, "source row padding")?;
            Ok(EagerPlanSourceFillRow {
                source_id: read_u32(row, 0)?,
                current_id: read_u32(row, 4)?,
                value_slot_id: read_u32(row, 8)?,
                external_label: read_u32(row, 12)?,
                input_momentum_slot: read_u32(row, 16)?,
                source_ir_id: read_u32(row, 20)?,
                crossing_ir_id: read_u32(row, 24)?,
                crossing_factor_id: read_u32(row, 28)?,
                declared_state_index: read_u32(row, 32)?,
            })
        },
    )
}

fn decode_parameters(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanParameterRow>> {
    decode_rows(
        reader,
        "tables/parameters.bin",
        EagerSectionKind::ParameterLayout,
        EagerPlanParameterRow::ENCODED_LEN,
        |row| {
            require_zero(row, 28, 32, "parameter row padding")?;
            Ok(EagerPlanParameterRow {
                parameter_id: read_u32(row, 0)?,
                name_string_id: read_u32(row, 4)?,
                kind_string_id: read_u32(row, 8)?,
                default_factor_id: read_u32(row, 12)?,
                runtime_name_string_id: read_u32(row, 16)?,
                complex_component: read_i32(row, 20)?,
                flags: read_u32(row, 24)?,
            })
        },
    )
}

fn decode_stages(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanStageRow>> {
    decode_rows(
        reader,
        "tables/stages.bin",
        EagerSectionKind::Stages,
        EagerPlanStageRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanStageRow {
                stage_index: read_u32(row, 0)?,
                subset_size: read_u32(row, 4)?,
                invocation_start: read_u64(row, 8)?,
                invocation_count: read_u64(row, 16)?,
                attachment_start: read_u64(row, 24)?,
                attachment_count: read_u64(row, 32)?,
                finalization_start: read_u64(row, 40)?,
                finalization_count: read_u64(row, 48)?,
            })
        },
    )
}

fn decode_couplings(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanCouplingRow>> {
    decode_rows(
        reader,
        "tables/couplings.bin",
        EagerSectionKind::Couplings,
        EagerPlanCouplingRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanCouplingRow {
                coupling_id: read_u32(row, 0)?,
                real_parameter_id: read_u32(row, 4)?,
                imaginary_parameter_id: read_u32(row, 8)?,
                constant_factor_id: read_u32(row, 12)?,
            })
        },
    )
}

fn decode_invocations(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanInvocationRow>> {
    decode_rows(
        reader,
        "tables/invocations.bin",
        EagerSectionKind::Invocations,
        EagerPlanInvocationRow::ENCODED_LEN,
        |row| {
            require_zero(row, 29, 32, "invocation row padding")?;
            require_zero(row, 52, 56, "invocation row padding")?;
            Ok(EagerPlanInvocationRow {
                evaluation_group_id: read_u32(row, 0)?,
                kernel_id: read_u32(row, 4)?,
                left_value_slot_id: read_u32(row, 8)?,
                right_value_slot_id: read_u32(row, 12)?,
                left_momentum_slot_id: read_u32(row, 16)?,
                right_momentum_slot_id: read_u32(row, 20)?,
                coupling_slot_id: read_u32(row, 24)?,
                output_factor_source: read_u8(row, 28)?,
                attachment_start: read_u64(row, 32)?,
                attachment_count: read_u64(row, 40)?,
                selector_domain_id: read_u32(row, 48)?,
            })
        },
    )
}

fn decode_attachments(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanAttachmentRow>> {
    decode_rows(
        reader,
        "tables/attachments.bin",
        EagerSectionKind::Attachments,
        EagerPlanAttachmentRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanAttachmentRow {
                interaction_id: read_u32(row, 0)?,
                result_current_id: read_u32(row, 4)?,
                color_factor_id: read_u32(row, 8)?,
                evaluation_factor_id: read_u32(row, 12)?,
                normalization_factor_id: read_u32(row, 16)?,
                representative_evaluation_factor_id: read_u32(row, 20)?,
                selector_domain_id: read_u32(row, 24)?,
            })
        },
    )
}

fn decode_finalizations(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanFinalizationRow>> {
    decode_rows(
        reader,
        "tables/finalizations.bin",
        EagerSectionKind::Finalizations,
        EagerPlanFinalizationRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanFinalizationRow {
                kernel_id: read_u32(row, 0)?,
                current_id: read_u32(row, 4)?,
                unpropagated_value_slot_id: read_u32(row, 8)?,
                propagated_value_slot_id: read_u32(row, 12)?,
                momentum_slot_id: read_u32(row, 16)?,
                unpropagated_selector_domain_id: read_u32(row, 20)?,
                propagated_selector_domain_id: read_u32(row, 24)?,
            })
        },
    )
}

fn decode_closures(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanClosureRow>> {
    decode_rows(
        reader,
        "tables/closures.bin",
        EagerSectionKind::Closures,
        EagerPlanClosureRow::ENCODED_LEN,
        |row| {
            require_zero(row, 33, 36, "closure row padding")?;
            Ok(EagerPlanClosureRow {
                root_id: read_u32(row, 0)?,
                kernel_id: read_u32(row, 4)?,
                left_value_slot_id: read_u32(row, 8)?,
                right_value_slot_id: read_u32(row, 12)?,
                amplitude_index: read_u32(row, 16)?,
                coherent_group_id: read_u32(row, 20)?,
                coupling_slot_id: read_u32(row, 24)?,
                coupling_factor_id: read_u32(row, 28)?,
                output_factor_source: read_u8(row, 32)?,
                color_factor_id: read_u32(row, 36)?,
                normalization_factor_id: read_u32(row, 40)?,
                direct_coefficient_start: read_u64(row, 44)?,
                direct_coefficient_count: read_u64(row, 52)?,
                selector_domain_id: read_u32(row, 60)?,
            })
        },
    )
}

fn decode_direct_coefficients(
    reader: &PacbinReader,
) -> RusticolResult<Vec<EagerPlanDirectCoefficientRow>> {
    decode_rows(
        reader,
        "tables/direct-coefficients.bin",
        EagerSectionKind::Closures,
        DIRECT_COEFFICIENT_RECORD_SIZE,
        |row| {
            require_zero(row, 12, 16, "direct coefficient row padding")?;
            Ok(EagerPlanDirectCoefficientRow {
                contraction_ir_id: read_u32(row, 0)?,
                component_index: read_u32(row, 4)?,
                factor_id: read_u32(row, 8)?,
            })
        },
    )
}

fn decode_selector_domains(
    reader: &PacbinReader,
) -> RusticolResult<Vec<EagerPlanSelectorDomainRow>> {
    decode_rows(
        reader,
        "selectors/domains.bin",
        EagerSectionKind::SelectorDomains,
        EagerPlanSelectorDomainRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanSelectorDomainRow {
                member_start: read_u64(row, 0)?,
                member_count: read_u64(row, 8)?,
            })
        },
    )
}

fn decode_helicity_selectors(
    reader: &PacbinReader,
) -> RusticolResult<Vec<EagerPlanHelicitySelectorRow>> {
    decode_rows(
        reader,
        "selectors/helicities.bin",
        EagerSectionKind::SelectorDomains,
        HELICITY_SELECTOR_RECORD_SIZE,
        |row| {
            require_zero(row, 18, 24, "helicity selector padding")?;
            Ok(EagerPlanHelicitySelectorRow {
                selector_id: read_u32(row, 0)?,
                values_sequence_id: read_u32(row, 4)?,
                representative_sequence_id: read_u32(row, 8)?,
                coefficient_factor_id: read_u32(row, 12)?,
                computed: read_bool(row, 16, "helicity computed flag")?,
                structural_zero: read_bool(row, 17, "helicity structural-zero flag")?,
            })
        },
    )
}

fn decode_color_selectors(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanColorSelectorRow>> {
    decode_rows(
        reader,
        "selectors/colors.bin",
        EagerSectionKind::SelectorDomains,
        COLOR_SELECTOR_RECORD_SIZE,
        |row| {
            require_zero(row, 17, 24, "color selector padding")?;
            Ok(EagerPlanColorSelectorRow {
                selector_id: read_u32(row, 0)?,
                word_sequence_id: read_u32(row, 4)?,
                representative_word_sequence_id: read_u32(row, 8)?,
                coefficient_factor_id: read_u32(row, 12)?,
                computed: read_bool(row, 16, "color computed flag")?,
            })
        },
    )
}

fn decode_reduction_groups(
    reader: &PacbinReader,
) -> RusticolResult<Vec<EagerPlanReductionGroupRow>> {
    decode_rows(
        reader,
        "reductions/groups.bin",
        EagerSectionKind::ReductionGroups,
        EagerPlanReductionGroupRow::ENCODED_LEN,
        |row| {
            require_zero(row, 4, 8, "reduction group padding")?;
            Ok(EagerPlanReductionGroupRow {
                coherent_group_id: read_u32(row, 0)?,
                amplitude_entry_start: read_u64(row, 8)?,
                amplitude_entry_count: read_u64(row, 16)?,
                selector_entry_start: read_u64(row, 24)?,
                selector_entry_count: read_u64(row, 32)?,
                helicity_weight_factor_id: read_u32(row, 40)?,
                all_sector_weight_factor_id: read_u32(row, 44)?,
            })
        },
    )
}

fn decode_reduction_entries(
    reader: &PacbinReader,
) -> RusticolResult<Vec<EagerPlanReductionEntryRow>> {
    decode_rows(
        reader,
        "reductions/entries.bin",
        EagerSectionKind::ReductionEntries,
        EagerPlanReductionEntryRow::ENCODED_LEN,
        |row| {
            require_zero(row, 1, 4, "reduction entry padding")?;
            let kind = match read_u8(row, 0)? {
                1 => EagerPlanReductionEntryKind::AmplitudeMember,
                2 => EagerPlanReductionEntryKind::SelectorMember,
                3 => EagerPlanReductionEntryKind::ColorContraction,
                value => {
                    return Err(RusticolError::compatibility(format!(
                        "unknown eager reduction entry kind {value}"
                    )));
                }
            };
            Ok(EagerPlanReductionEntryRow {
                kind,
                owner_id: read_u32(row, 4)?,
                left_id: read_u32(row, 8)?,
                right_id: read_u32(row, 12)?,
                factor_id: read_u32(row, 16)?,
                auxiliary_factor_id: read_u32(row, 20)?,
            })
        },
    )
}

fn decode_exact_factors(reader: &PacbinReader) -> RusticolResult<Vec<EagerPlanExactFactorRow>> {
    decode_rows(
        reader,
        "catalogs/exact-factors.bin",
        EagerSectionKind::ExactFactors,
        EagerPlanExactFactorRow::ENCODED_LEN,
        |row| {
            require_zero(row, 4, 8, "exact factor padding")?;
            require_zero(row, 29, 32, "exact factor padding")?;
            let exact_source = read_u8(row, 28)?;
            if exact_source > 1 {
                return Err(RusticolError::compatibility(format!(
                    "unknown exact-factor source {exact_source}"
                )));
            }
            Ok(EagerPlanExactFactorRow {
                factor_id: read_u32(row, 0)?,
                real_bits: read_u64(row, 8)?,
                imaginary_bits: read_u64(row, 16)?,
                canonical_string_id: read_u32(row, 24)?,
                exact_source,
                exact_ir_id: read_u32(row, 32)?,
                source_ir_id: read_u32(row, 36)?,
            })
        },
    )
}

fn decode_catalog_ranges(
    reader: &PacbinReader,
    path: &str,
) -> RusticolResult<Vec<EagerPlanCatalogRangeRow>> {
    decode_rows(
        reader,
        path,
        EagerSectionKind::Metadata,
        EagerPlanCatalogRangeRow::ENCODED_LEN,
        |row| {
            Ok(EagerPlanCatalogRangeRow {
                start: read_u64(row, 0)?,
                count: read_u64(row, 8)?,
            })
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_declared_counts(
    manifest: &EagerV3ExecutionManifest,
    metadata: Metadata,
    inspection: &[u64; 20],
    currents: &[EagerPlanCurrentRow],
    values: &[EagerPlanValueRow],
    momenta: &[EagerPlanMomentumRow],
    sources: &[EagerPlanSourceFillRow],
    parameters: &[EagerPlanParameterRow],
    stages: &[EagerPlanStageRow],
    couplings: &[EagerPlanCouplingRow],
    invocations: &[EagerPlanInvocationRow],
    attachments: &[EagerPlanAttachmentRow],
    finalizations: &[EagerPlanFinalizationRow],
    closures: &[EagerPlanClosureRow],
    direct_coefficients: &[EagerPlanDirectCoefficientRow],
    selector_domains: &[EagerPlanSelectorDomainRow],
    selector_memberships: &[u32],
    helicity_selectors: &[EagerPlanHelicitySelectorRow],
    color_selectors: &[EagerPlanColorSelectorRow],
    reduction_groups: &[EagerPlanReductionGroupRow],
    reduction_entries: &[EagerPlanReductionEntryRow],
    exact_factors: &[EagerPlanExactFactorRow],
    retained_table_count: usize,
) -> RusticolResult<()> {
    let actual = [
        currents.len(),
        values.len(),
        momenta.len(),
        sources.len(),
        parameters.len(),
        stages.len(),
        couplings.len(),
        invocations.len(),
        attachments.len(),
        finalizations.len(),
        closures.len(),
        direct_coefficients.len(),
        selector_domains.len(),
        selector_memberships.len(),
        helicity_selectors.len(),
        color_selectors.len(),
        reduction_groups.len(),
        reduction_entries.len(),
        exact_factors.len(),
        retained_table_count,
    ];
    for (index, (actual, declared)) in actual.into_iter().zip(inspection).enumerate() {
        check_count(&format!("inspection field {index}"), actual, *declared)?;
    }
    let metadata_counts = [
        ("metadata currents", currents.len(), metadata.current_count),
        ("metadata values", values.len(), metadata.value_count),
        ("metadata momenta", momenta.len(), metadata.momentum_count),
        ("metadata sources", sources.len(), metadata.source_count),
        ("metadata stages", stages.len(), metadata.stage_count),
        (
            "metadata invocations",
            invocations.len(),
            metadata.invocation_count,
        ),
        ("metadata closures", closures.len(), metadata.closure_count),
        (
            "metadata direct coefficients",
            direct_coefficients.len(),
            metadata.direct_coefficient_count,
        ),
        (
            "metadata helicity selectors",
            helicity_selectors.len(),
            metadata.helicity_selector_count,
        ),
        (
            "metadata color selectors",
            color_selectors.len(),
            metadata.color_selector_count,
        ),
        (
            "metadata exact factors",
            exact_factors.len(),
            metadata.exact_factor_count,
        ),
    ];
    for (name, actual, declared) in metadata_counts {
        check_count(name, actual, declared)?;
    }
    let summary = &manifest.plan.inspection_summary;
    let manifest_counts = [
        summary.current_count,
        summary.value_count,
        summary.momentum_count,
        summary.source_count,
        summary.parameter_count,
        summary.stage_count,
        summary.coupling_count,
        summary.invocation_count,
        summary.attachment_count,
        summary.finalization_count,
        summary.closure_count,
        summary.selector_domain_count,
        summary.reduction_group_count,
        summary.reduction_entry_count,
        summary.retained_table_count,
    ];
    let inspection_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 19];
    for (manifest_count, index) in manifest_counts.into_iter().zip(inspection_indices) {
        if manifest_count != inspection[index] {
            return Err(integrity(format!(
                "execution-manifest inspection count disagrees with PACBIN field {index}"
            )));
        }
    }
    if summary.current_component_count != metadata.current_component_count
        || summary.value_component_count != metadata.value_component_count
        || summary.momentum_component_count != metadata.momentum_component_count
    {
        return Err(integrity(
            "execution-manifest component counts disagree with PACBIN metadata",
        ));
    }
    Ok(())
}

fn validate_reduction_plan_metadata(
    reader: &PacbinReader,
    manifest: &EagerV3ExecutionManifest,
    metadata: Metadata,
    inspection: &[u64; 20],
) -> RusticolResult<()> {
    // The complete container was authenticated before this decoder was
    // entered. Inspect the canonical shape/count header of every plan table
    // without touching or allocating its payload rows.
    for (index, expected) in COUNTED_PLAN_SECTIONS.iter().enumerate() {
        let actual = section(reader, expected.path, expected.kind, expected.record_size)?.count;
        if actual != inspection[index] {
            return Err(integrity(format!(
                "reduction-only plan section {:?} has {actual} rows, inspection field {index} declares {}",
                expected.path, inspection[index]
            )));
        }
    }

    for (path, declared, context) in [
        (
            "catalogs/strings/ranges.bin",
            metadata.string_count,
            "string catalog",
        ),
        (
            "catalogs/exact-ir/ranges.bin",
            metadata.exact_ir_count,
            "exact IR catalog",
        ),
        (
            "catalogs/semantic-limitations/ranges.bin",
            metadata.limitation_count,
            "semantic limitation catalog",
        ),
        (
            "retained/columns.bin",
            metadata.retained_column_count,
            "retained column catalog",
        ),
    ] {
        let record_size = if path == "retained/columns.bin" {
            RETAINED_COLUMN_RECORD_SIZE
        } else {
            EagerPlanCatalogRangeRow::ENCODED_LEN
        };
        let actual = section(reader, path, EagerSectionKind::Metadata, record_size)?.count;
        if actual != declared {
            return Err(integrity(format!(
                "reduction-only {context} has {actual} rows, metadata declares {declared}"
            )));
        }
    }

    let metadata_counts = [
        ("currents", metadata.current_count, inspection[0]),
        ("values", metadata.value_count, inspection[1]),
        ("momenta", metadata.momentum_count, inspection[2]),
        ("sources", metadata.source_count, inspection[3]),
        ("stages", metadata.stage_count, inspection[5]),
        ("invocations", metadata.invocation_count, inspection[7]),
        ("closures", metadata.closure_count, inspection[10]),
        (
            "direct coefficients",
            metadata.direct_coefficient_count,
            inspection[11],
        ),
        (
            "helicity selectors",
            metadata.helicity_selector_count,
            inspection[14],
        ),
        (
            "color selectors",
            metadata.color_selector_count,
            inspection[15],
        ),
        ("exact factors", metadata.exact_factor_count, inspection[18]),
        (
            "retained tables",
            metadata.retained_table_count,
            inspection[19],
        ),
    ];
    for (context, declared, actual) in metadata_counts {
        if declared != actual {
            return Err(integrity(format!(
                "reduction-only metadata {context} count {declared} disagrees with inspection count {actual}"
            )));
        }
    }

    let summary = &manifest.plan.inspection_summary;
    let manifest_counts = [
        summary.current_count,
        summary.value_count,
        summary.momentum_count,
        summary.source_count,
        summary.parameter_count,
        summary.stage_count,
        summary.coupling_count,
        summary.invocation_count,
        summary.attachment_count,
        summary.finalization_count,
        summary.closure_count,
        summary.selector_domain_count,
        summary.reduction_group_count,
        summary.reduction_entry_count,
        summary.retained_table_count,
    ];
    let inspection_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 19];
    for (manifest_count, index) in manifest_counts.into_iter().zip(inspection_indices) {
        if manifest_count != inspection[index] {
            return Err(integrity(format!(
                "execution-manifest inspection count disagrees with PACBIN field {index}"
            )));
        }
    }
    if summary.current_component_count != metadata.current_component_count
        || summary.value_component_count != metadata.value_component_count
        || summary.momentum_component_count != metadata.momentum_component_count
    {
        return Err(integrity(
            "execution-manifest component counts disagree with PACBIN metadata",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_reduction_subset_semantics(
    metadata: Metadata,
    exact_factor_count: u64,
    helicity_selectors: &[EagerPlanHelicitySelectorRow],
    color_selectors: &[EagerPlanColorSelectorRow],
    reduction_groups: &[EagerPlanReductionGroupRow],
    reduction_entries: &[EagerPlanReductionEntryRow],
    helicity_sequences: &BTreeMap<u32, Vec<i32>>,
    color_sequences: &BTreeMap<u32, Vec<u32>>,
) -> RusticolResult<()> {
    validate_dense_ids(
        helicity_selectors.iter().map(|row| row.selector_id),
        "helicity selector",
    )?;
    for row in helicity_selectors {
        required_projected_sequence(
            helicity_sequences,
            row.values_sequence_id,
            "helicity values sequence",
        )?;
        required_projected_sequence(
            helicity_sequences,
            row.representative_sequence_id,
            "helicity representative sequence",
        )?;
        required_declared_index(
            row.coefficient_factor_id,
            exact_factor_count,
            "helicity coefficient factor",
        )?;
    }

    validate_dense_ids(
        color_selectors.iter().map(|row| row.selector_id),
        "color selector",
    )?;
    for row in color_selectors {
        required_projected_sequence(color_sequences, row.word_sequence_id, "color word sequence")?;
        required_projected_sequence(
            color_sequences,
            row.representative_word_sequence_id,
            "color representative sequence",
        )?;
        required_declared_index(
            row.coefficient_factor_id,
            exact_factor_count,
            "color coefficient factor",
        )?;
    }

    let known_groups = reduction_groups
        .iter()
        .map(|group| group.coherent_group_id)
        .collect::<HashSet<_>>();
    if known_groups.len() != reduction_groups.len() {
        return Err(integrity("reduction groups repeat coherent-group IDs"));
    }
    let mut covered = Vec::new();
    reserve(
        &mut covered,
        reduction_entries.len(),
        "reduction entry coverage",
    )?;
    covered.resize(reduction_entries.len(), false);
    for group in reduction_groups {
        for (start, count, expected_kind, context) in [
            (
                group.amplitude_entry_start,
                group.amplitude_entry_count,
                EagerPlanReductionEntryKind::AmplitudeMember,
                "amplitude reduction entries",
            ),
            (
                group.selector_entry_start,
                group.selector_entry_count,
                EagerPlanReductionEntryKind::SelectorMember,
                "selector reduction entries",
            ),
        ] {
            let entries = checked_range(reduction_entries, start, count, context)?;
            let start = usize_count(start, context)?;
            for (offset, entry) in entries.iter().enumerate() {
                if entry.kind != expected_kind || entry.owner_id != group.coherent_group_id {
                    return Err(integrity(format!(
                        "{context} have inconsistent tags/owners"
                    )));
                }
                if entry.kind == EagerPlanReductionEntryKind::SelectorMember {
                    required_index(
                        entry.left_id,
                        helicity_selectors.len(),
                        "reduction helicity selector",
                    )?;
                    required_index(
                        entry.right_id,
                        color_selectors.len(),
                        "reduction color selector",
                    )?;
                }
                covered[start + offset] = true;
            }
        }
        required_declared_index(
            group.helicity_weight_factor_id,
            exact_factor_count,
            "reduction helicity weight",
        )?;
        required_declared_index(
            group.all_sector_weight_factor_id,
            exact_factor_count,
            "reduction all-sector weight",
        )?;
    }

    let contraction = checked_range(
        reduction_entries,
        metadata.color_contraction_entry_start,
        metadata.color_contraction_entry_count,
        "color contraction entries",
    )?;
    let contraction_start = usize_count(
        metadata.color_contraction_entry_start,
        "color contraction entry start",
    )?;
    for (offset, entry) in contraction.iter().enumerate() {
        if entry.kind != EagerPlanReductionEntryKind::ColorContraction {
            return Err(integrity(
                "color contraction range contains another entry kind",
            ));
        }
        covered[contraction_start + offset] = true;
        if !known_groups.contains(&entry.left_id) || !known_groups.contains(&entry.right_id) {
            return Err(integrity(
                "color contraction references an unknown coherent group",
            ));
        }
        required_declared_index(
            entry.factor_id,
            exact_factor_count,
            "color contraction factor",
        )?;
        optional_declared_index(
            entry.auxiliary_factor_id,
            exact_factor_count,
            "color contraction symmetry factor",
        )?;
    }
    if covered.iter().any(|covered| !covered) {
        return Err(integrity("reduction entry catalog contains unowned rows"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_semantics(
    metadata: Metadata,
    invocation_group_upper_bound: u64,
    currents: &[EagerPlanCurrentRow],
    values: &[EagerPlanValueRow],
    momenta: &[EagerPlanMomentumRow],
    sources: &[EagerPlanSourceFillRow],
    parameters: &[EagerPlanParameterRow],
    stages: &[EagerPlanStageRow],
    couplings: &[EagerPlanCouplingRow],
    invocations: &[EagerPlanInvocationRow],
    attachments: &[EagerPlanAttachmentRow],
    finalizations: &[EagerPlanFinalizationRow],
    closures: &[EagerPlanClosureRow],
    direct_coefficients: &[EagerPlanDirectCoefficientRow],
    selector_domains: &[EagerPlanSelectorDomainRow],
    selector_memberships: &[u32],
    helicity_selectors: &[EagerPlanHelicitySelectorRow],
    color_selectors: &[EagerPlanColorSelectorRow],
    reduction_groups: &[EagerPlanReductionGroupRow],
    reduction_entries: &[EagerPlanReductionEntryRow],
    exact_factors: &[EagerPlanExactFactorRow],
    bitset_ranges: &[EagerPlanCatalogRangeRow],
    u32_sequence_ranges: &[EagerPlanCatalogRangeRow],
    i32_sequence_ranges: &[EagerPlanCatalogRangeRow],
    strings: &[Box<str>],
    exact_ir: &[Box<str>],
    kernel_specs: &[EagerKernelSpec],
) -> RusticolResult<()> {
    validate_component_layout(
        "current",
        currents.iter().enumerate().map(|(index, row)| {
            (
                index,
                row.current_id,
                row.component_start,
                row.component_count,
            )
        }),
        metadata.current_component_count,
    )?;
    validate_component_layout(
        "value",
        values.iter().enumerate().map(|(index, row)| {
            (
                index,
                row.value_slot_id,
                row.component_start,
                row.component_count,
            )
        }),
        metadata.value_component_count,
    )?;
    validate_component_layout(
        "momentum",
        momenta.iter().enumerate().map(|(index, row)| {
            (
                index,
                row.momentum_slot_id,
                row.component_start,
                row.component_count,
            )
        }),
        metadata.momentum_component_count,
    )?;
    for row in currents {
        required_index(row.momentum_slot_id, momenta.len(), "current momentum slot")?;
        if row.flags & !1 != 0 {
            return Err(RusticolError::compatibility("unknown eager current flags"));
        }
    }
    for row in values {
        let current = required_index(row.current_id, currents.len(), "value current")?;
        if currents[current].component_count != row.component_count {
            return Err(integrity("value/current component counts disagree"));
        }
    }
    for row in momenta {
        required_index(row.bitset_id, bitset_ranges.len(), "momentum bitset")?;
    }
    validate_dense_ids(sources.iter().map(|row| row.source_id), "source")?;
    for row in sources {
        let current = required_index(row.current_id, currents.len(), "source current")?;
        let value = required_index(row.value_slot_id, values.len(), "source value slot")?;
        required_index(
            row.input_momentum_slot,
            momenta.len(),
            "source input momentum",
        )?;
        required_index(row.source_ir_id, exact_ir.len(), "source exact IR")?;
        required_index(row.crossing_ir_id, exact_ir.len(), "source crossing IR")?;
        required_index(
            row.crossing_factor_id,
            exact_factors.len(),
            "source crossing factor",
        )?;
        if currents[current].flags & 1 == 0 || values[value].kind != EagerValueSlotKind::Source {
            return Err(integrity(
                "source row does not reference source current/value slots",
            ));
        }
    }
    validate_dense_ids(parameters.iter().map(|row| row.parameter_id), "parameter")?;
    for row in parameters {
        required_index(row.name_string_id, strings.len(), "parameter name")?;
        required_index(row.kind_string_id, strings.len(), "parameter kind")?;
        optional_index(
            row.runtime_name_string_id,
            strings.len(),
            "parameter runtime name",
        )?;
        required_index(
            row.default_factor_id,
            exact_factors.len(),
            "parameter default factor",
        )?;
        if !(-1..=1).contains(&row.complex_component) || row.flags & !1 != 0 {
            return Err(RusticolError::compatibility(
                "unsupported eager parameter component or flags",
            ));
        }
    }
    validate_stages(
        stages,
        invocations.len(),
        attachments.len(),
        finalizations.len(),
    )?;
    validate_dense_ids(couplings.iter().map(|row| row.coupling_id), "coupling")?;
    for row in couplings {
        optional_index(
            row.real_parameter_id,
            parameters.len(),
            "coupling real parameter",
        )?;
        optional_index(
            row.imaginary_parameter_id,
            parameters.len(),
            "coupling imaginary parameter",
        )?;
        required_index(
            row.constant_factor_id,
            exact_factors.len(),
            "coupling constant factor",
        )?;
    }

    let kernels = kernel_specs
        .iter()
        .map(|kernel| (kernel.kernel_id, kernel.role))
        .collect::<BTreeMap<_, _>>();
    if kernels.len() != kernel_specs.len() {
        return Err(integrity("prepared kernel specs repeat a kernel ID"));
    }
    validate_ordered_sparse_ids(
        invocations.iter().map(|row| row.evaluation_group_id),
        invocation_group_upper_bound,
        "invocation evaluation group",
    )?;
    for row in invocations {
        require_kernel(&kernels, row.kernel_id, EagerKernelRole::Vertex)?;
        required_index(
            row.left_value_slot_id,
            values.len(),
            "invocation left value",
        )?;
        required_index(
            row.right_value_slot_id,
            values.len(),
            "invocation right value",
        )?;
        required_index(
            row.left_momentum_slot_id,
            momenta.len(),
            "invocation left momentum",
        )?;
        required_index(
            row.right_momentum_slot_id,
            momenta.len(),
            "invocation right momentum",
        )?;
        optional_index(row.coupling_slot_id, couplings.len(), "invocation coupling")?;
        validate_output_factor(row.output_factor_source, row.coupling_slot_id)?;
        checked_range(
            attachments,
            row.attachment_start,
            row.attachment_count,
            "invocation attachments",
        )?;
        required_index(
            row.selector_domain_id,
            selector_domains.len(),
            "invocation selector",
        )?;
    }
    for row in attachments {
        required_index(
            row.result_current_id,
            currents.len(),
            "attachment result current",
        )?;
        for (factor, context) in [
            (row.color_factor_id, "attachment color factor"),
            (row.evaluation_factor_id, "attachment evaluation factor"),
            (
                row.normalization_factor_id,
                "attachment normalization factor",
            ),
            (
                row.representative_evaluation_factor_id,
                "attachment representative factor",
            ),
        ] {
            required_index(factor, exact_factors.len(), context)?;
        }
        required_index(
            row.selector_domain_id,
            selector_domains.len(),
            "attachment selector",
        )?;
    }
    for row in finalizations {
        required_index(row.current_id, currents.len(), "finalization current")?;
        optional_index(
            row.unpropagated_value_slot_id,
            values.len(),
            "finalization unpropagated value",
        )?;
        optional_index(
            row.propagated_value_slot_id,
            values.len(),
            "finalization propagated value",
        )?;
        required_index(row.momentum_slot_id, momenta.len(), "finalization momentum")?;
        required_index(
            row.unpropagated_selector_domain_id,
            selector_domains.len(),
            "finalization unpropagated selector",
        )?;
        required_index(
            row.propagated_selector_domain_id,
            selector_domains.len(),
            "finalization propagated selector",
        )?;
        if row.kernel_id == MISSING_U32 {
            if row.propagated_value_slot_id != MISSING_U32 {
                return Err(integrity("propagated finalization lacks a prepared kernel"));
            }
        } else {
            require_kernel(&kernels, row.kernel_id, EagerKernelRole::Finalization)?;
        }
    }
    validate_dense_ids(closures.iter().map(|row| row.root_id), "closure root")?;
    let mut amplitudes = BTreeSet::new();
    for row in closures {
        required_index(row.left_value_slot_id, values.len(), "closure left value")?;
        required_index(row.right_value_slot_id, values.len(), "closure right value")?;
        amplitudes.insert(row.amplitude_index);
        optional_index(row.coupling_slot_id, couplings.len(), "closure coupling")?;
        for (factor, context) in [
            (row.coupling_factor_id, "closure coupling factor"),
            (row.color_factor_id, "closure color factor"),
            (row.normalization_factor_id, "closure normalization factor"),
        ] {
            required_index(factor, exact_factors.len(), context)?;
        }
        validate_output_factor(row.output_factor_source, row.coupling_slot_id)?;
        let direct = checked_range(
            direct_coefficients,
            row.direct_coefficient_start,
            row.direct_coefficient_count,
            "closure direct coefficients",
        )?;
        if row.kernel_id == MISSING_U32 {
            if direct.is_empty() {
                return Err(integrity("direct eager closure has no coefficients"));
            }
        } else {
            require_kernel(&kernels, row.kernel_id, EagerKernelRole::Closure)?;
            if !direct.is_empty() {
                return Err(integrity(
                    "prepared eager closure also has direct coefficients",
                ));
            }
        }
        required_index(
            row.selector_domain_id,
            selector_domains.len(),
            "closure selector",
        )?;
    }
    if amplitudes
        .iter()
        .copied()
        .ne(0..usize_u32(amplitudes.len(), "amplitude count")?)
    {
        return Err(integrity("closure amplitude indices are not dense"));
    }
    for row in direct_coefficients {
        required_index(
            row.contraction_ir_id,
            exact_ir.len(),
            "direct coefficient IR",
        )?;
        required_index(
            row.factor_id,
            exact_factors.len(),
            "direct coefficient factor",
        )?;
    }

    let known_groups = reduction_groups
        .iter()
        .map(|group| group.coherent_group_id)
        .collect::<HashSet<_>>();
    if known_groups.len() != reduction_groups.len() {
        return Err(integrity("reduction groups repeat coherent-group IDs"));
    }
    let mut selector_cursor = 0_u64;
    for domain in selector_domains {
        if domain.member_start != selector_cursor {
            return Err(integrity("selector-domain ranges are not contiguous"));
        }
        selector_cursor = checked_add(
            domain.member_start,
            domain.member_count,
            "selector-domain memberships",
        )?;
    }
    if selector_cursor != usize_u64(selector_memberships.len(), "selector memberships")? {
        return Err(integrity(
            "selector-domain ranges do not cover the membership catalog",
        ));
    }
    for member in selector_memberships {
        if !known_groups.contains(member) {
            return Err(integrity(
                "selector domain references an unknown coherent group",
            ));
        }
    }
    validate_dense_ids(
        helicity_selectors.iter().map(|row| row.selector_id),
        "helicity selector",
    )?;
    for row in helicity_selectors {
        required_index(
            row.values_sequence_id,
            i32_sequence_ranges.len(),
            "helicity values sequence",
        )?;
        required_index(
            row.representative_sequence_id,
            i32_sequence_ranges.len(),
            "helicity representative sequence",
        )?;
        required_index(
            row.coefficient_factor_id,
            exact_factors.len(),
            "helicity coefficient factor",
        )?;
    }
    validate_dense_ids(
        color_selectors.iter().map(|row| row.selector_id),
        "color selector",
    )?;
    for row in color_selectors {
        required_index(
            row.word_sequence_id,
            u32_sequence_ranges.len(),
            "color word sequence",
        )?;
        required_index(
            row.representative_word_sequence_id,
            u32_sequence_ranges.len(),
            "color representative sequence",
        )?;
        required_index(
            row.coefficient_factor_id,
            exact_factors.len(),
            "color coefficient factor",
        )?;
    }

    let mut covered = vec![false; reduction_entries.len()];
    for group in reduction_groups {
        for (start, count, expected_kind, context) in [
            (
                group.amplitude_entry_start,
                group.amplitude_entry_count,
                EagerPlanReductionEntryKind::AmplitudeMember,
                "amplitude reduction entries",
            ),
            (
                group.selector_entry_start,
                group.selector_entry_count,
                EagerPlanReductionEntryKind::SelectorMember,
                "selector reduction entries",
            ),
        ] {
            let entries = checked_range(reduction_entries, start, count, context)?;
            let start = usize_count(start, context)?;
            for (offset, entry) in entries.iter().enumerate() {
                if entry.kind != expected_kind || entry.owner_id != group.coherent_group_id {
                    return Err(integrity(format!(
                        "{context} have inconsistent tags/owners"
                    )));
                }
                covered[start + offset] = true;
            }
        }
        required_index(
            group.helicity_weight_factor_id,
            exact_factors.len(),
            "reduction helicity weight",
        )?;
        required_index(
            group.all_sector_weight_factor_id,
            exact_factors.len(),
            "reduction all-sector weight",
        )?;
    }
    let contraction = checked_range(
        reduction_entries,
        metadata.color_contraction_entry_start,
        metadata.color_contraction_entry_count,
        "color contraction entries",
    )?;
    let contraction_start = usize_count(
        metadata.color_contraction_entry_start,
        "color contraction entry start",
    )?;
    for (offset, entry) in contraction.iter().enumerate() {
        if entry.kind != EagerPlanReductionEntryKind::ColorContraction {
            return Err(integrity(
                "color contraction range contains another entry kind",
            ));
        }
        covered[contraction_start + offset] = true;
        if !known_groups.contains(&entry.left_id) || !known_groups.contains(&entry.right_id) {
            return Err(integrity(
                "color contraction references an unknown coherent group",
            ));
        }
        required_index(
            entry.factor_id,
            exact_factors.len(),
            "color contraction factor",
        )?;
        optional_index(
            entry.auxiliary_factor_id,
            exact_factors.len(),
            "color contraction symmetry factor",
        )?;
    }
    if covered.iter().any(|covered| !covered) {
        return Err(integrity("reduction entry catalog contains unowned rows"));
    }

    validate_dense_ids(
        exact_factors.iter().map(|row| row.factor_id),
        "exact factor",
    )?;
    for row in exact_factors {
        required_index(
            row.canonical_string_id,
            strings.len(),
            "exact factor canonical string",
        )?;
        optional_index(row.exact_ir_id, exact_ir.len(), "exact factor IR")?;
        optional_index(row.source_ir_id, exact_ir.len(), "exact factor source IR")?;
    }
    Ok(())
}

fn validate_stages(
    stages: &[EagerPlanStageRow],
    invocation_count: usize,
    attachment_count: usize,
    finalization_count: usize,
) -> RusticolResult<()> {
    let mut invocations = 0_u64;
    let mut attachments = 0_u64;
    let mut finalizations = 0_u64;
    let mut previous_subset = 0_u32;
    for (index, stage) in stages.iter().enumerate() {
        if stage.stage_index != usize_u32(index + 1, "stage index")?
            || stage.subset_size <= previous_subset
            || stage.invocation_start != invocations
            || stage.attachment_start != attachments
            || stage.finalization_start != finalizations
        {
            return Err(integrity(
                "eager stage descriptors are not canonical and contiguous",
            ));
        }
        invocations = checked_add(
            stage.invocation_start,
            stage.invocation_count,
            "stage invocations",
        )?;
        attachments = checked_add(
            stage.attachment_start,
            stage.attachment_count,
            "stage attachments",
        )?;
        finalizations = checked_add(
            stage.finalization_start,
            stage.finalization_count,
            "stage finalizations",
        )?;
        previous_subset = stage.subset_size;
    }
    if invocations != usize_u64(invocation_count, "invocations")?
        || attachments != usize_u64(attachment_count, "attachments")?
        || finalizations != usize_u64(finalization_count, "finalizations")?
    {
        return Err(integrity(
            "eager stage ranges do not cover their complete tables",
        ));
    }
    Ok(())
}

fn validate_output_factor(source: u8, coupling_id: u32) -> RusticolResult<()> {
    if source > 2 || (coupling_id == MISSING_U32 && source != 0) {
        return Err(RusticolError::compatibility(
            "unsupported eager output-factor source",
        ));
    }
    Ok(())
}

fn require_kernel(
    kernels: &BTreeMap<u32, EagerKernelRole>,
    id: u32,
    expected: EagerKernelRole,
) -> RusticolResult<()> {
    match kernels.get(&id) {
        Some(actual) if *actual == expected => Ok(()),
        Some(actual) => Err(integrity(format!(
            "prepared kernel {id} has role {actual:?}, expected {expected:?}"
        ))),
        None => Err(integrity(format!(
            "eager plan references missing prepared kernel {id}"
        ))),
    }
}

fn validate_component_layout(
    name: &str,
    rows: impl Iterator<Item = (usize, u32, u64, u32)>,
    declared_components: u64,
) -> RusticolResult<()> {
    let mut cursor = 0_u64;
    for (index, id, start, count) in rows {
        if id != usize_u32(index, &format!("{name} ID"))? || start != cursor || count == 0 {
            return Err(integrity(format!(
                "{name} component layout is not dense and contiguous"
            )));
        }
        cursor = cursor
            .checked_add(u64::from(count))
            .ok_or_else(|| RusticolError::artifact(format!("{name} components exceed u64")))?;
    }
    if cursor != declared_components {
        return Err(integrity(format!(
            "{name} component layout total disagrees with metadata"
        )));
    }
    Ok(())
}

fn validate_dense_ids(ids: impl Iterator<Item = u32>, context: &str) -> RusticolResult<()> {
    for (index, id) in ids.enumerate() {
        if id != usize_u32(index, &format!("{context} ID"))? {
            return Err(integrity(format!("{context} IDs are not dense")));
        }
    }
    Ok(())
}

fn validate_ordered_sparse_ids(
    ids: impl Iterator<Item = u32>,
    exclusive_upper_bound: u64,
    context: &str,
) -> RusticolResult<()> {
    let mut previous = None;
    for id in ids {
        if id == MISSING_U32 || u64::from(id) >= exclusive_upper_bound {
            return Err(integrity(format!(
                "{context} ID {id} is outside its authenticated domain"
            )));
        }
        if previous.is_some_and(|previous| id <= previous) {
            return Err(integrity(format!(
                "{context} IDs are not strictly increasing and unique"
            )));
        }
        previous = Some(id);
    }
    Ok(())
}

fn validate_catalog_ranges(
    context: &str,
    ranges: &[EagerPlanCatalogRangeRow],
    value_count: usize,
) -> RusticolResult<()> {
    let mut cursor = 0_u64;
    for range in ranges {
        if range.start != cursor {
            return Err(integrity(format!("{context} ranges are not contiguous")));
        }
        cursor = checked_add(range.start, range.count, context)?;
    }
    if cursor != usize_u64(value_count, context)? {
        return Err(integrity(format!(
            "{context} ranges do not cover their value catalog"
        )));
    }
    Ok(())
}

struct DecodedSection<'a> {
    count: u64,
    payload: &'a [u8],
}

fn decode_referenced_sequences<T>(
    reader: &PacbinReader,
    ranges_path: &str,
    values_path: &str,
    referenced_ids: &BTreeSet<u32>,
    context: &str,
    decode: impl FnMut(&[u8]) -> RusticolResult<T>,
) -> RusticolResult<BTreeMap<u32, Vec<T>>> {
    let ranges = section(
        reader,
        ranges_path,
        EagerSectionKind::Metadata,
        EagerPlanCatalogRangeRow::ENCODED_LEN,
    )?;
    let values = section(reader, values_path, EagerSectionKind::Metadata, 4)?;
    project_referenced_sequences(&ranges, &values, referenced_ids, context, decode)
}

/// Validate the complete borrowed range catalog, then own only the values
/// reachable from `referenced_ids`.
fn project_referenced_sequences<T>(
    ranges: &DecodedSection<'_>,
    values: &DecodedSection<'_>,
    referenced_ids: &BTreeSet<u32>,
    context: &str,
    mut decode: impl FnMut(&[u8]) -> RusticolResult<T>,
) -> RusticolResult<BTreeMap<u32, Vec<T>>> {
    let range_width = usize::try_from(EagerPlanCatalogRangeRow::ENCODED_LEN)
        .map_err(|_| RusticolError::artifact(format!("{context} range width exceeds usize")))?;
    let value_width = 4_usize;
    let range_count = usize_count(ranges.count, context)?;
    if ranges.payload.chunks_exact(range_width).len() != range_count {
        return Err(integrity(format!(
            "{context} range row count changed unexpectedly"
        )));
    }

    // Authentication covers every byte before this point. This streaming pass
    // additionally checks that even unreferenced ranges have canonical,
    // gap-free coverage without allocating their rows.
    let mut cursor = 0_u64;
    for row in ranges.payload.chunks_exact(range_width) {
        let start = read_u64(row, 0)?;
        let count = read_u64(row, 8)?;
        if start != cursor {
            return Err(integrity(format!("{context} ranges are not contiguous")));
        }
        cursor = checked_add(start, count, context)?;
    }
    if cursor != values.count {
        return Err(integrity(format!(
            "{context} ranges do not cover their value catalog"
        )));
    }

    let mut projected = BTreeMap::new();
    for &sequence_id in referenced_ids {
        required_declared_index(sequence_id, ranges.count, context)?;
        let sequence_index = usize::try_from(sequence_id)
            .map_err(|_| RusticolError::artifact(format!("{context} ID exceeds usize")))?;
        let range_start = sequence_index
            .checked_mul(range_width)
            .ok_or_else(|| RusticolError::artifact(format!("{context} range offset overflows")))?;
        let range_stop = range_start
            .checked_add(range_width)
            .ok_or_else(|| RusticolError::artifact(format!("{context} range offset overflows")))?;
        let range = ranges.payload.get(range_start..range_stop).ok_or_else(|| {
            integrity(format!(
                "{context} {sequence_id} range row is out of bounds"
            ))
        })?;
        let start = usize_count(read_u64(range, 0)?, context)?;
        let count = usize_count(read_u64(range, 8)?, context)?;
        let stop = start
            .checked_add(count)
            .ok_or_else(|| RusticolError::artifact(format!("{context} range exceeds usize")))?;
        let byte_start = start
            .checked_mul(value_width)
            .ok_or_else(|| RusticolError::artifact(format!("{context} byte range overflows")))?;
        let byte_stop = stop
            .checked_mul(value_width)
            .ok_or_else(|| RusticolError::artifact(format!("{context} byte range overflows")))?;
        let encoded = values.payload.get(byte_start..byte_stop).ok_or_else(|| {
            integrity(format!(
                "{context} {sequence_id} value range is out of bounds"
            ))
        })?;

        let mut sequence = Vec::new();
        reserve(&mut sequence, count, context)?;
        for row in encoded.chunks_exact(value_width) {
            sequence.push(decode(row)?);
        }
        if sequence.len() != count {
            return Err(integrity(format!(
                "{context} {sequence_id} decoded length changed unexpectedly"
            )));
        }
        projected.insert(sequence_id, sequence);
    }
    Ok(projected)
}

fn section<'a>(
    reader: &'a PacbinReader,
    path: &str,
    kind: EagerSectionKind,
    record_size: u32,
) -> RusticolResult<DecodedSection<'a>> {
    let bytes = reader.member_bytes(path)?;
    let (header, payload) = EagerSectionHeader::decode(bytes)?;
    if header.kind() != kind || header.record_size() != record_size {
        return Err(RusticolError::compatibility(format!(
            "eager PACBIN member {path:?} has shape ({:?}, {}), expected ({kind:?}, {record_size})",
            header.kind(),
            header.record_size(),
        )));
    }
    Ok(DecodedSection {
        count: header.record_count(),
        payload,
    })
}

fn decode_rows<T>(
    reader: &PacbinReader,
    path: &str,
    kind: EagerSectionKind,
    record_size: u32,
    mut decode: impl FnMut(&[u8]) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let section = section(reader, path, kind, record_size)?;
    let count = usize_count(section.count, path)?;
    let width = usize::try_from(record_size)
        .map_err(|_| RusticolError::artifact(format!("{path} record size does not fit usize")))?;
    let mut result = Vec::new();
    reserve(&mut result, count, path)?;
    for row in section.payload.chunks_exact(width) {
        result.push(decode(row)?);
    }
    if result.len() != count {
        return Err(integrity(format!(
            "{path} decoded row count changed unexpectedly"
        )));
    }
    Ok(result)
}

fn decode_u32_section(
    reader: &PacbinReader,
    path: &str,
    kind: EagerSectionKind,
) -> RusticolResult<Vec<u32>> {
    decode_rows(reader, path, kind, 4, |row| read_u32(row, 0))
}

fn decode_i32_section(
    reader: &PacbinReader,
    path: &str,
    kind: EagerSectionKind,
) -> RusticolResult<Vec<i32>> {
    decode_rows(reader, path, kind, 4, |row| read_i32(row, 0))
}

fn decode_u64_section(
    reader: &PacbinReader,
    path: &str,
    kind: EagerSectionKind,
) -> RusticolResult<Vec<u64>> {
    decode_rows(reader, path, kind, 8, |row| read_u64(row, 0))
}

fn read_bool(row: &[u8], offset: usize, context: &str) -> RusticolResult<u8> {
    let value = read_u8(row, offset)?;
    if value > 1 {
        return Err(integrity(format!(
            "{context} must be encoded as zero or one"
        )));
    }
    Ok(value)
}

fn read_u8(bytes: &[u8], offset: usize) -> RusticolResult<u8> {
    bytes
        .get(offset)
        .copied()
        .ok_or_else(|| integrity("truncated eager fixed-width record"))
}

fn read_u16(bytes: &[u8], offset: usize) -> RusticolResult<u16> {
    Ok(u16::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> RusticolResult<u32> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_i32(bytes: &[u8], offset: usize) -> RusticolResult<i32> {
    Ok(i32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> RusticolResult<u64> {
    Ok(u64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> RusticolResult<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| integrity("eager fixed-width record offset overflow"))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| integrity("truncated eager fixed-width record"))?
        .try_into()
        .map_err(|_| integrity("truncated eager fixed-width record"))
}

fn require_zero(bytes: &[u8], start: usize, end: usize, context: &str) -> RusticolResult<()> {
    if bytes
        .get(start..end)
        .ok_or_else(|| integrity(format!("truncated {context}")))?
        .iter()
        .any(|value| *value != 0)
    {
        return Err(integrity(format!("{context} must be zero")));
    }
    Ok(())
}

fn require_count(section: &DecodedSection<'_>, expected: u64, context: &str) -> RusticolResult<()> {
    if section.count != expected {
        return Err(integrity(format!(
            "{context} has {} records, expected {expected}",
            section.count
        )));
    }
    Ok(())
}

fn required_catalog<'a>(values: &'a [Box<str>], id: u32, context: &str) -> RusticolResult<&'a str> {
    values
        .get(required_index(id, values.len(), context)?)
        .map(Box::as_ref)
        .ok_or_else(|| integrity(format!("{context} is out of range")))
}

fn required_index(id: u32, length: usize, context: &str) -> RusticolResult<usize> {
    let index = usize::try_from(id)
        .map_err(|_| RusticolError::artifact(format!("{context} does not fit usize")))?;
    if index >= length {
        return Err(integrity(format!(
            "{context} {id} is outside a catalog of length {length}"
        )));
    }
    Ok(index)
}

fn required_declared_index(id: u32, length: u64, context: &str) -> RusticolResult<()> {
    if u64::from(id) >= length {
        return Err(integrity(format!(
            "{context} {id} is outside a catalog of length {length}"
        )));
    }
    Ok(())
}

fn required_projected_sequence<'a, T>(
    sequences: &'a BTreeMap<u32, Vec<T>>,
    id: u32,
    context: &str,
) -> RusticolResult<&'a [T]> {
    sequences.get(&id).map(Vec::as_slice).ok_or_else(|| {
        integrity(format!(
            "{context} {id} is absent from the referenced catalog projection"
        ))
    })
}

fn optional_index(id: u32, length: usize, context: &str) -> RusticolResult<()> {
    if id != MISSING_U32 {
        required_index(id, length, context)?;
    }
    Ok(())
}

fn optional_declared_index(id: u32, length: u64, context: &str) -> RusticolResult<()> {
    if id != MISSING_U32 {
        required_declared_index(id, length, context)?;
    }
    Ok(())
}

fn checked_range<'a, T>(
    values: &'a [T],
    start: u64,
    count: u64,
    context: &str,
) -> RusticolResult<&'a [T]> {
    let start = usize_count(start, context)?;
    let count = usize_count(count, context)?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range exceeds usize")))?;
    values.get(start..end).ok_or_else(|| {
        integrity(format!(
            "{context} range {start}..{end} exceeds length {}",
            values.len()
        ))
    })
}

fn checked_add(start: u64, count: u64, context: &str) -> RusticolResult<u64> {
    start
        .checked_add(count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range exceeds u64")))
}

fn check_count(context: &str, actual: usize, declared: u64) -> RusticolResult<()> {
    if usize_u64(actual, context)? != declared {
        return Err(integrity(format!(
            "{context} count {actual} does not match declared count {declared}"
        )));
    }
    Ok(())
}

fn usize_count(value: u64, context: &str) -> RusticolResult<usize> {
    usize::try_from(value)
        .map_err(|_| RusticolError::artifact(format!("{context} count does not fit usize")))
}

fn usize_u32(value: usize, context: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| RusticolError::artifact(format!("{context} exceeds u32")))
}

fn usize_u64(value: usize, context: &str) -> RusticolResult<u64> {
    u64::try_from(value).map_err(|_| RusticolError::artifact(format!("{context} exceeds u64")))
}

fn reserve<T>(values: &mut Vec<T>, count: usize, context: &str) -> RusticolResult<()> {
    values.try_reserve_exact(count).map_err(|error| {
        RusticolError::artifact(format!(
            "could not reserve {context} ({count} rows): {error}"
        ))
    })
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message.into())
}

#[cfg(test)]
mod reduction_subset_tests {
    use super::*;
    use crate::pacbin::{
        PacbinMemberKind, PacbinReader, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic,
    };
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_INVOCATION_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn invocation(evaluation_group_id: u32) -> EagerPlanInvocationRow {
        EagerPlanInvocationRow {
            evaluation_group_id,
            kernel_id: 0,
            left_value_slot_id: 0,
            right_value_slot_id: 0,
            left_momentum_slot_id: 0,
            right_momentum_slot_id: 0,
            coupling_slot_id: 0,
            output_factor_source: 0,
            attachment_start: 0,
            attachment_count: 1,
            selector_domain_id: MISSING_U32,
        }
    }

    fn put_u32(row: &mut [u8], offset: usize, value: u32) {
        row[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(row: &mut [u8], offset: usize, value: u64) {
        row[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn decode_invocation_fixture(
        rows: &[EagerPlanInvocationRow],
        corrupt_padding: bool,
    ) -> RusticolResult<Vec<EagerPlanInvocationRow>> {
        let mut bytes = EagerSectionHeader::new(
            EagerSectionKind::Invocations,
            EagerPlanInvocationRow::ENCODED_LEN,
            rows.len() as u64,
        )?
        .encode()
        .to_vec();
        for (index, source) in rows.iter().enumerate() {
            let mut row = [0_u8; EagerPlanInvocationRow::ENCODED_LEN as usize];
            for (offset, value) in [
                source.evaluation_group_id,
                source.kernel_id,
                source.left_value_slot_id,
                source.right_value_slot_id,
                source.left_momentum_slot_id,
                source.right_momentum_slot_id,
                source.coupling_slot_id,
            ]
            .into_iter()
            .enumerate()
            {
                put_u32(&mut row, offset * 4, value);
            }
            row[28] = source.output_factor_source;
            put_u64(&mut row, 32, source.attachment_start);
            put_u64(&mut row, 40, source.attachment_count);
            put_u32(&mut row, 48, source.selector_domain_id);
            if corrupt_padding && index == 0 {
                row[29] = 1;
            }
            bytes.extend_from_slice(&row);
        }

        let root = std::env::temp_dir().join(format!(
            "rusticol-eager-sparse-invocations-{}-{}",
            std::process::id(),
            NEXT_INVOCATION_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&root).map_err(|error| {
            RusticolError::artifact(format!("could not create test directory: {error}"))
        })?;
        let path = root.join("invocations.pacbin");
        let member = PacbinWriteMember::from_bytes(
            "tables/invocations.bin",
            PacbinMemberKind::EagerRuntimeTable,
            &bytes,
        )?;
        write_pacbin_atomic(&path, [member], PacbinWriteOptions::default())?;
        let reader = PacbinReader::open(&path)?;
        let decoded = decode_invocations(&reader);
        drop(reader);
        let _ = fs::remove_dir_all(root);
        decoded
    }

    fn metadata() -> Metadata {
        Metadata {
            retained_column_count: 0,
            current_component_count: 0,
            value_component_count: 0,
            momentum_component_count: 0,
            color_contraction_entry_start: 2,
            color_contraction_entry_count: 1,
            string_count: 0,
            exact_ir_count: 0,
            limitation_count: 0,
            retained_table_count: 0,
            current_count: 0,
            value_count: 0,
            momentum_count: 0,
            source_count: 0,
            stage_count: 0,
            invocation_count: 0,
            closure_count: 0,
            direct_coefficient_count: 0,
            helicity_selector_count: 1,
            color_selector_count: 1,
            exact_factor_count: 2,
        }
    }

    fn color_selectors() -> Vec<EagerPlanColorSelectorRow> {
        vec![EagerPlanColorSelectorRow {
            selector_id: 0,
            word_sequence_id: 0,
            representative_word_sequence_id: 0,
            coefficient_factor_id: 0,
            computed: 1,
        }]
    }

    fn helicity_selectors() -> Vec<EagerPlanHelicitySelectorRow> {
        vec![EagerPlanHelicitySelectorRow {
            selector_id: 0,
            values_sequence_id: 0,
            representative_sequence_id: 0,
            coefficient_factor_id: 0,
            computed: 1,
            structural_zero: 0,
        }]
    }

    fn groups() -> Vec<EagerPlanReductionGroupRow> {
        vec![EagerPlanReductionGroupRow {
            coherent_group_id: 7,
            amplitude_entry_start: 0,
            amplitude_entry_count: 1,
            selector_entry_start: 1,
            selector_entry_count: 1,
            helicity_weight_factor_id: 0,
            all_sector_weight_factor_id: 1,
        }]
    }

    fn entries() -> Vec<EagerPlanReductionEntryRow> {
        vec![
            EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::AmplitudeMember,
                owner_id: 7,
                left_id: 0,
                right_id: MISSING_U32,
                factor_id: MISSING_U32,
                auxiliary_factor_id: MISSING_U32,
            },
            EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::SelectorMember,
                owner_id: 7,
                left_id: 0,
                right_id: 0,
                factor_id: MISSING_U32,
                auxiliary_factor_id: MISSING_U32,
            },
            EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::ColorContraction,
                owner_id: MISSING_U32,
                left_id: 7,
                right_id: 7,
                factor_id: 1,
                auxiliary_factor_id: MISSING_U32,
            },
        ]
    }

    fn validate(entries: &[EagerPlanReductionEntryRow]) -> RusticolResult<()> {
        validate_reduction_subset_semantics(
            metadata(),
            2,
            &helicity_selectors(),
            &color_selectors(),
            &groups(),
            entries,
            &BTreeMap::from([(0, vec![1, -1, 1])]),
            &BTreeMap::from([(0, vec![1, 2])]),
        )
    }

    #[test]
    fn bounded_reduction_semantics_accept_canonical_partition() {
        validate(&entries()).expect("canonical reduction subset");
    }

    #[test]
    fn active_kernel_census_uses_invocations_and_non_sentinel_terminal_rows() {
        assert_eq!(
            active_kernel_ids([7, 7], [MISSING_U32, 9], [11, MISSING_U32]),
            BTreeSet::from([7, 9, 11])
        );
    }

    #[test]
    fn pacbin_decode_accepts_sparse_invocation_groups_after_zero_first_group() {
        let decoded = decode_invocation_fixture(&[invocation(1), invocation(3)], false)
            .expect("decode authenticated sparse invocation rows");
        assert_eq!(
            decoded
                .iter()
                .map(|row| row.evaluation_group_id)
                .collect::<Vec<_>>(),
            [1, 3]
        );
        validate_ordered_sparse_ids(
            decoded.iter().map(|row| row.evaluation_group_id),
            4,
            "invocation evaluation group",
        )
        .expect("zero-first-group pruning retains authenticated sparse IDs");
    }

    #[test]
    fn sparse_invocation_group_ids_reject_duplicates_order_and_domain_escape() {
        for (ids, message) in [
            (&[1_u32, 1][..], "strictly increasing and unique"),
            (&[2_u32, 1][..], "strictly increasing and unique"),
            (&[1_u32, 4][..], "outside its authenticated domain"),
            (&[MISSING_U32][..], "outside its authenticated domain"),
        ] {
            let rows = ids.iter().copied().map(invocation).collect::<Vec<_>>();
            let decoded = decode_invocation_fixture(&rows, false)
                .expect("fixed-width invocation rows remain decodable");
            let error = validate_ordered_sparse_ids(
                decoded.iter().map(|row| row.evaluation_group_id),
                4,
                "invocation evaluation group",
            )
            .expect_err("malformed sparse group IDs must fail closed");
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn pacbin_invocation_padding_mutation_fails_closed() {
        let error = decode_invocation_fixture(&[invocation(1)], true)
            .expect_err("mutated invocation padding must fail closed");
        assert!(
            error
                .to_string()
                .contains("invocation row padding must be zero"),
            "{error}"
        );
    }

    #[test]
    fn bounded_reduction_semantics_reject_corrupt_selector_and_coverage() {
        let mut wrong_owner = entries();
        wrong_owner[1].owner_id = 8;
        assert!(
            validate(&wrong_owner)
                .unwrap_err()
                .to_string()
                .contains("inconsistent tags/owners")
        );

        let mut wrong_color = entries();
        wrong_color[1].right_id = 1;
        assert!(
            validate(&wrong_color)
                .unwrap_err()
                .to_string()
                .contains("reduction color selector")
        );

        let mut unowned = entries();
        unowned.push(EagerPlanReductionEntryRow {
            kind: EagerPlanReductionEntryKind::AmplitudeMember,
            owner_id: 7,
            left_id: 1,
            right_id: MISSING_U32,
            factor_id: MISSING_U32,
            auxiliary_factor_id: MISSING_U32,
        });
        assert!(
            validate(&unowned)
                .unwrap_err()
                .to_string()
                .contains("unowned rows")
        );
    }

    #[test]
    fn large_irrelevant_catalog_projects_only_referenced_values() {
        use std::cell::Cell;

        let catalog_size = 250_000_u32;
        let mut range_bytes = Vec::with_capacity(catalog_size as usize * 16);
        let mut value_bytes = Vec::with_capacity(catalog_size as usize * 4);
        for id in 0..catalog_size {
            range_bytes.extend_from_slice(&u64::from(id).to_le_bytes());
            range_bytes.extend_from_slice(&1_u64.to_le_bytes());
            value_bytes.extend_from_slice(&id.to_le_bytes());
        }
        let ranges = DecodedSection {
            count: u64::from(catalog_size),
            payload: &range_bytes,
        };
        let values = DecodedSection {
            count: u64::from(catalog_size),
            payload: &value_bytes,
        };
        let last = catalog_size - 3;
        let referenced = BTreeSet::from([7, last]);
        let decoded_value_count = Cell::new(0_usize);

        let projected = project_referenced_sequences(
            &ranges,
            &values,
            &referenced,
            "bounded test sequence",
            |row| {
                decoded_value_count.set(decoded_value_count.get() + 1);
                read_u32(row, 0)
            },
        )
        .expect("project referenced catalog");

        assert_eq!(projected.len(), 2);
        assert_eq!(projected.values().map(Vec::len).sum::<usize>(), 2);
        assert_eq!(decoded_value_count.get(), 2);
        assert_eq!(projected[&7], [7]);
        assert_eq!(projected[&last], [last]);
    }

    #[test]
    fn referenced_projection_validates_unreferenced_range_rows() {
        let mut range_bytes = Vec::new();
        for (start, count) in [(0_u64, 1_u64), (9, 1), (2, 1)] {
            range_bytes.extend_from_slice(&start.to_le_bytes());
            range_bytes.extend_from_slice(&count.to_le_bytes());
        }
        let value_bytes = [0_u32, 1, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let ranges = DecodedSection {
            count: 3,
            payload: &range_bytes,
        };
        let values = DecodedSection {
            count: 3,
            payload: &value_bytes,
        };

        let error = project_referenced_sequences(
            &ranges,
            &values,
            &BTreeSet::from([0]),
            "authenticated sequence",
            |row| read_u32(row, 0),
        )
        .unwrap_err();

        assert!(error.to_string().contains("ranges are not contiguous"));
    }
}
