// SPDX-License-Identifier: 0BSD

//! Deterministic, bounded-memory serialization for compact eager plan-v3.
//!
//! Every PACBIN member starts with an [`EagerSectionHeader`]. Fixed-width
//! tables are encoded one row at a time, while string and retained-column
//! catalogs are streamed from borrowed segments. The serializer therefore
//! never materializes an aggregate eager runtime payload in memory.

use crate::eager_layout::{
    EAGER_LOWERING_INPUT_ABI, EAGER_PLAN_ABI, EAGER_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_CONTAINER_KIND, EAGER_RUNTIME_CONTAINER_SCHEMA, EAGER_RUNTIME_LAYOUT_ABI,
    EAGER_SECTION_HEADER_SIZE, EagerSectionHeader, EagerSectionKind,
};
use crate::eager_lowering_v3::{
    EagerOwnedPrimitiveColumn, EagerOwnedRetainedTable, EagerPlanAttachmentRow,
    EagerPlanCatalogRangeRow, EagerPlanClosureRow, EagerPlanColorSelectorRow, EagerPlanCouplingRow,
    EagerPlanCurrentRow, EagerPlanDirectCoefficientRow, EagerPlanExactFactorRow,
    EagerPlanFinalizationRow, EagerPlanHelicitySelectorRow, EagerPlanInvocationRow,
    EagerPlanMomentumRow, EagerPlanParameterRow, EagerPlanReductionEntryRow,
    EagerPlanReductionGroupRow, EagerPlanSelectorDomainRow, EagerPlanSourceFillRow,
    EagerPlanStageRow, EagerPlanV3, EagerPlanValueRow,
};
use crate::pacbin::{PacbinMemberKind, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic};
use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const SERIALIZATION_SCHEMA: u32 = 1;
const METADATA_RECORD_SIZE: u32 = 224;
const INSPECTION_RECORD_SIZE: u32 = 160;
const RETAINED_TABLE_RECORD_SIZE: u32 = 40;
const RETAINED_COLUMN_RECORD_SIZE: u32 = 40;
const DIRECT_COEFFICIENT_RECORD_SIZE: u32 = 16;
const HELICITY_SELECTOR_RECORD_SIZE: u32 = 24;
const COLOR_SELECTOR_RECORD_SIZE: u32 = 24;
const MAX_ENCODED_RECORD_SIZE: usize = METADATA_RECORD_SIZE as usize;
const FILE_HASH_BUFFER_SIZE: usize = 1024 * 1024;

const IDENTITY_LOWERING_ABI: u32 = 0;
const IDENTITY_PLAN_ABI: u32 = 1;
const IDENTITY_RUNTIME_LAYOUT_ABI: u32 = 2;
const IDENTITY_RUNTIME_CAPABILITY: u32 = 3;
const IDENTITY_CONTAINER_KIND: u32 = 4;
const IDENTITY_PROCESS_KEY: u32 = 5;
const IDENTITY_MODEL_NAME: u32 = 6;

const RETAINED_U8: u8 = 1;
const RETAINED_U32: u8 = 2;
const RETAINED_U64: u8 = 3;
const RETAINED_I32: u8 = 4;
const RETAINED_F64_BITS: u8 = 5;

/// Bounded publication metadata returned after the complete container is
/// atomically installed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerPlanV3PacbinMetadata {
    pub member_count: u64,
    pub unpacked_bytes: u64,
    pub index_sha256: [u8; 32],
    pub file_size: u64,
    pub file_sha256: [u8; 32],
}

/// Atomically write one complete `eager-runtime.pacbin` from an eager plan-v3.
pub fn write_eager_plan_v3_pacbin(
    plan: &EagerPlanV3,
    destination: impl AsRef<Path>,
) -> RusticolResult<EagerPlanV3PacbinMetadata> {
    let view = PlanView::from(plan);
    write_plan_view(&view, destination.as_ref())
}

#[derive(Clone, Copy)]
struct PlanView<'a> {
    abi: &'a str,
    process_key: &'a str,
    model_name: &'a str,
    string_catalog: &'a [Box<str>],
    canonical_ir_catalog: &'a [Box<str>],
    semantic_limitations: &'a [Box<str>],
    retained_tables: &'a [EagerOwnedRetainedTable],
    currents: &'a [EagerPlanCurrentRow],
    values: &'a [EagerPlanValueRow],
    momenta: &'a [EagerPlanMomentumRow],
    sources: &'a [EagerPlanSourceFillRow],
    parameters: &'a [EagerPlanParameterRow],
    stages: &'a [EagerPlanStageRow],
    couplings: &'a [EagerPlanCouplingRow],
    invocations: &'a [EagerPlanInvocationRow],
    attachments: &'a [EagerPlanAttachmentRow],
    finalizations: &'a [EagerPlanFinalizationRow],
    closures: &'a [EagerPlanClosureRow],
    direct_coefficients: &'a [EagerPlanDirectCoefficientRow],
    selector_domains: &'a [EagerPlanSelectorDomainRow],
    selector_memberships: &'a [u32],
    helicity_selectors: &'a [EagerPlanHelicitySelectorRow],
    color_selectors: &'a [EagerPlanColorSelectorRow],
    reduction_groups: &'a [EagerPlanReductionGroupRow],
    reduction_entries: &'a [EagerPlanReductionEntryRow],
    exact_factors: &'a [EagerPlanExactFactorRow],
    bitset_ranges: &'a [EagerPlanCatalogRangeRow],
    bitset_populations: &'a [u64],
    bitset_words: &'a [u64],
    u32_sequence_ranges: &'a [EagerPlanCatalogRangeRow],
    u32_sequence_values: &'a [u32],
    i32_sequence_ranges: &'a [EagerPlanCatalogRangeRow],
    i32_sequence_values: &'a [i32],
    current_component_count: u64,
    value_component_count: u64,
    momentum_component_count: u64,
    color_contraction_entry_start: u64,
    color_contraction_entry_count: u64,
}

impl<'a> From<&'a EagerPlanV3> for PlanView<'a> {
    fn from(plan: &'a EagerPlanV3) -> Self {
        let (color_contraction_entry_start, color_contraction_entry_count) =
            plan.color_contraction_entry_range();
        Self {
            abi: plan.abi(),
            process_key: plan.process_key(),
            model_name: plan.model_name(),
            string_catalog: plan.string_catalog(),
            canonical_ir_catalog: plan.canonical_ir_catalog(),
            semantic_limitations: plan.semantic_limitations(),
            retained_tables: plan.retained_tables(),
            currents: plan.currents(),
            values: plan.values(),
            momenta: plan.momenta(),
            sources: plan.sources(),
            parameters: plan.parameters(),
            stages: plan.stages(),
            couplings: plan.couplings(),
            invocations: plan.invocations(),
            attachments: plan.attachments(),
            finalizations: plan.finalizations(),
            closures: plan.closures(),
            direct_coefficients: plan.direct_coefficients(),
            selector_domains: plan.selector_domains(),
            selector_memberships: plan.selector_memberships(),
            helicity_selectors: plan.helicity_selectors(),
            color_selectors: plan.color_selectors(),
            reduction_groups: plan.reduction_groups(),
            reduction_entries: plan.reduction_entries(),
            exact_factors: plan.exact_factors(),
            bitset_ranges: plan.bitset_ranges(),
            bitset_populations: plan.bitset_populations(),
            bitset_words: plan.bitset_words(),
            u32_sequence_ranges: plan.u32_sequence_ranges(),
            u32_sequence_values: plan.u32_sequence_values(),
            i32_sequence_ranges: plan.i32_sequence_ranges(),
            i32_sequence_values: plan.i32_sequence_values(),
            current_component_count: plan.current_component_count(),
            value_component_count: plan.value_component_count(),
            momentum_component_count: plan.momentum_component_count(),
            color_contraction_entry_start,
            color_contraction_entry_count,
        }
    }
}

fn write_plan_view(
    plan: &PlanView<'_>,
    destination: &Path,
) -> RusticolResult<EagerPlanV3PacbinMetadata> {
    let context = SerializationContext::new(plan)?;
    let mut sections = context.sections(plan)?;
    let mut members = Vec::with_capacity(sections.len());
    for section in &mut sections {
        members.push(PacbinWriteMember::from_reader(
            section.logical_path,
            section.member_kind,
            &mut section.reader,
        )?);
    }
    let index = write_pacbin_atomic(destination, members, PacbinWriteOptions::default())?;
    let unpacked_bytes = index.members().iter().try_fold(0_u64, |total, member| {
        total.checked_add(member.length()).ok_or_else(|| {
            RusticolError::serialization("eager PACBIN unpacked byte count exceeds u64")
        })
    })?;
    let file_sha256 = hash_file(destination)?;
    Ok(EagerPlanV3PacbinMetadata {
        member_count: u64::try_from(index.members().len())
            .map_err(|_| RusticolError::serialization("eager PACBIN member count exceeds u64"))?,
        unpacked_bytes,
        index_sha256: *index.index_sha256(),
        file_size: index.file_size(),
        file_sha256,
    })
}

fn hash_file(path: &Path) -> RusticolResult<[u8; 32]> {
    let mut file = File::open(path).map_err(|error| {
        RusticolError::artifact(format!(
            "could not open published eager PACBIN {} for hashing: {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; FILE_HASH_BUFFER_SIZE];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            RusticolError::artifact(format!(
                "could not hash published eager PACBIN {}: {error}",
                path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().into())
}

struct SerializationContext<'a> {
    identity: TextCatalog<'a>,
    strings: TextCatalog<'a>,
    exact_ir: TextCatalog<'a>,
    limitations: TextCatalog<'a>,
    retained: RetainedCatalog<'a>,
    metadata: MetadataRecord,
    inspection: InspectionRecord,
}

impl<'a> SerializationContext<'a> {
    fn new(plan: &PlanView<'a>) -> RusticolResult<Self> {
        if plan.abi != EAGER_PLAN_ABI {
            return Err(RusticolError::compatibility(format!(
                "cannot serialize unsupported eager plan ABI {:?}",
                plan.abi
            )));
        }
        let identity = TextCatalog::new([
            EAGER_LOWERING_INPUT_ABI.as_bytes(),
            plan.abi.as_bytes(),
            EAGER_RUNTIME_LAYOUT_ABI.as_bytes(),
            EAGER_RUNTIME_CAPABILITY.as_bytes(),
            EAGER_RUNTIME_CONTAINER_KIND.as_bytes(),
            plan.process_key.as_bytes(),
            plan.model_name.as_bytes(),
        ])?;
        let retained = RetainedCatalog::new(plan.retained_tables)?;
        let metadata = MetadataRecord::new(plan, &retained)?;
        let inspection = InspectionRecord::new(plan)?;
        Ok(Self {
            identity,
            strings: TextCatalog::new(plan.string_catalog.iter().map(|value| value.as_bytes()))?,
            exact_ir: TextCatalog::new(
                plan.canonical_ir_catalog
                    .iter()
                    .map(|value| value.as_bytes()),
            )?,
            limitations: TextCatalog::new(
                plan.semantic_limitations
                    .iter()
                    .map(|value| value.as_bytes()),
            )?,
            retained,
            metadata,
            inspection,
        })
    }

    fn sections<'b>(&'b self, plan: &'b PlanView<'a>) -> RusticolResult<Vec<NamedSection<'b>>>
    where
        'a: 'b,
    {
        let mut sections = Vec::with_capacity(48);
        push_fixed(
            &mut sections,
            "metadata/core.bin",
            PacbinMemberKind::EagerRuntimeMetadata,
            EagerSectionKind::Metadata,
            METADATA_RECORD_SIZE,
            std::slice::from_ref(&self.metadata),
            encode_metadata,
        )?;
        push_text_catalog(
            &mut sections,
            "metadata/identity",
            EagerSectionKind::Metadata,
            &self.identity,
        )?;
        push_text_catalog(
            &mut sections,
            "catalogs/strings",
            EagerSectionKind::Metadata,
            &self.strings,
        )?;
        push_text_catalog(
            &mut sections,
            "catalogs/exact-ir",
            EagerSectionKind::Metadata,
            &self.exact_ir,
        )?;
        push_text_catalog(
            &mut sections,
            "catalogs/semantic-limitations",
            EagerSectionKind::Metadata,
            &self.limitations,
        )?;
        self.retained.push_sections(&mut sections)?;

        push_fixed_table(
            &mut sections,
            "tables/currents.bin",
            EagerSectionKind::CurrentLayout,
            EagerPlanCurrentRow::ENCODED_LEN,
            plan.currents,
            encode_current,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/values.bin",
            EagerSectionKind::ValueLayout,
            EagerPlanValueRow::ENCODED_LEN,
            plan.values,
            encode_value,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/momenta.bin",
            EagerSectionKind::MomentumLayout,
            EagerPlanMomentumRow::ENCODED_LEN,
            plan.momenta,
            encode_momentum,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/sources.bin",
            EagerSectionKind::SourceFill,
            EagerPlanSourceFillRow::ENCODED_LEN,
            plan.sources,
            encode_source,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/parameters.bin",
            EagerSectionKind::ParameterLayout,
            EagerPlanParameterRow::ENCODED_LEN,
            plan.parameters,
            encode_parameter,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/stages.bin",
            EagerSectionKind::Stages,
            EagerPlanStageRow::ENCODED_LEN,
            plan.stages,
            encode_stage,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/couplings.bin",
            EagerSectionKind::Couplings,
            EagerPlanCouplingRow::ENCODED_LEN,
            plan.couplings,
            encode_coupling,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/invocations.bin",
            EagerSectionKind::Invocations,
            EagerPlanInvocationRow::ENCODED_LEN,
            plan.invocations,
            encode_invocation,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/attachments.bin",
            EagerSectionKind::Attachments,
            EagerPlanAttachmentRow::ENCODED_LEN,
            plan.attachments,
            encode_attachment,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/finalizations.bin",
            EagerSectionKind::Finalizations,
            EagerPlanFinalizationRow::ENCODED_LEN,
            plan.finalizations,
            encode_finalization,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/closures.bin",
            EagerSectionKind::Closures,
            EagerPlanClosureRow::ENCODED_LEN,
            plan.closures,
            encode_closure,
        )?;
        push_fixed_table(
            &mut sections,
            "tables/direct-coefficients.bin",
            EagerSectionKind::Closures,
            DIRECT_COEFFICIENT_RECORD_SIZE,
            plan.direct_coefficients,
            encode_direct_coefficient,
        )?;
        push_fixed_table(
            &mut sections,
            "selectors/domains.bin",
            EagerSectionKind::SelectorDomains,
            EagerPlanSelectorDomainRow::ENCODED_LEN,
            plan.selector_domains,
            encode_selector_domain,
        )?;
        push_primitive(
            &mut sections,
            "selectors/memberships.bin",
            EagerSectionKind::SelectorMemberships,
            plan.selector_memberships,
        )?;
        push_fixed_table(
            &mut sections,
            "selectors/helicities.bin",
            EagerSectionKind::SelectorDomains,
            HELICITY_SELECTOR_RECORD_SIZE,
            plan.helicity_selectors,
            encode_helicity_selector,
        )?;
        push_fixed_table(
            &mut sections,
            "selectors/colors.bin",
            EagerSectionKind::SelectorDomains,
            COLOR_SELECTOR_RECORD_SIZE,
            plan.color_selectors,
            encode_color_selector,
        )?;
        push_fixed_table(
            &mut sections,
            "reductions/groups.bin",
            EagerSectionKind::ReductionGroups,
            EagerPlanReductionGroupRow::ENCODED_LEN,
            plan.reduction_groups,
            encode_reduction_group,
        )?;
        push_fixed_table(
            &mut sections,
            "reductions/entries.bin",
            EagerSectionKind::ReductionEntries,
            EagerPlanReductionEntryRow::ENCODED_LEN,
            plan.reduction_entries,
            encode_reduction_entry,
        )?;
        push_fixed_table(
            &mut sections,
            "catalogs/exact-factors.bin",
            EagerSectionKind::ExactFactors,
            EagerPlanExactFactorRow::ENCODED_LEN,
            plan.exact_factors,
            encode_exact_factor,
        )?;
        push_fixed_table(
            &mut sections,
            "catalogs/bitsets/ranges.bin",
            EagerSectionKind::Metadata,
            EagerPlanCatalogRangeRow::ENCODED_LEN,
            plan.bitset_ranges,
            encode_catalog_range,
        )?;
        push_primitive(
            &mut sections,
            "catalogs/bitsets/populations.bin",
            EagerSectionKind::Metadata,
            plan.bitset_populations,
        )?;
        push_primitive(
            &mut sections,
            "catalogs/bitsets/words.bin",
            EagerSectionKind::Metadata,
            plan.bitset_words,
        )?;
        push_fixed_table(
            &mut sections,
            "catalogs/u32-sequences/ranges.bin",
            EagerSectionKind::Metadata,
            EagerPlanCatalogRangeRow::ENCODED_LEN,
            plan.u32_sequence_ranges,
            encode_catalog_range,
        )?;
        push_primitive(
            &mut sections,
            "catalogs/u32-sequences/values.bin",
            EagerSectionKind::Metadata,
            plan.u32_sequence_values,
        )?;
        push_fixed_table(
            &mut sections,
            "catalogs/i32-sequences/ranges.bin",
            EagerSectionKind::Metadata,
            EagerPlanCatalogRangeRow::ENCODED_LEN,
            plan.i32_sequence_ranges,
            encode_catalog_range,
        )?;
        push_primitive(
            &mut sections,
            "catalogs/i32-sequences/values.bin",
            EagerSectionKind::Metadata,
            plan.i32_sequence_values,
        )?;
        push_fixed(
            &mut sections,
            "inspection/summary.bin",
            PacbinMemberKind::EagerRuntimeMetadata,
            EagerSectionKind::InspectionSummary,
            INSPECTION_RECORD_SIZE,
            std::slice::from_ref(&self.inspection),
            encode_inspection,
        )?;
        Ok(sections)
    }
}

#[derive(Clone, Copy)]
struct MetadataRecord {
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

impl MetadataRecord {
    fn new(plan: &PlanView<'_>, retained: &RetainedCatalog<'_>) -> RusticolResult<Self> {
        Ok(Self {
            retained_column_count: usize_u64(retained.columns.len(), "retained column count")?,
            current_component_count: plan.current_component_count,
            value_component_count: plan.value_component_count,
            momentum_component_count: plan.momentum_component_count,
            color_contraction_entry_start: plan.color_contraction_entry_start,
            color_contraction_entry_count: plan.color_contraction_entry_count,
            string_count: usize_u64(plan.string_catalog.len(), "string catalog count")?,
            exact_ir_count: usize_u64(plan.canonical_ir_catalog.len(), "exact IR count")?,
            limitation_count: usize_u64(
                plan.semantic_limitations.len(),
                "semantic limitation count",
            )?,
            retained_table_count: usize_u64(plan.retained_tables.len(), "retained table count")?,
            current_count: usize_u64(plan.currents.len(), "current count")?,
            value_count: usize_u64(plan.values.len(), "value count")?,
            momentum_count: usize_u64(plan.momenta.len(), "momentum count")?,
            source_count: usize_u64(plan.sources.len(), "source count")?,
            stage_count: usize_u64(plan.stages.len(), "stage count")?,
            invocation_count: usize_u64(plan.invocations.len(), "invocation count")?,
            closure_count: usize_u64(plan.closures.len(), "closure count")?,
            direct_coefficient_count: usize_u64(
                plan.direct_coefficients.len(),
                "direct coefficient count",
            )?,
            helicity_selector_count: usize_u64(
                plan.helicity_selectors.len(),
                "helicity selector count",
            )?,
            color_selector_count: usize_u64(plan.color_selectors.len(), "color selector count")?,
            exact_factor_count: usize_u64(plan.exact_factors.len(), "exact factor count")?,
        })
    }
}

#[derive(Clone, Copy)]
struct InspectionRecord {
    counts: [u64; 20],
}

impl InspectionRecord {
    fn new(plan: &PlanView<'_>) -> RusticolResult<Self> {
        Ok(Self {
            counts: [
                usize_u64(plan.currents.len(), "inspection current count")?,
                usize_u64(plan.values.len(), "inspection value count")?,
                usize_u64(plan.momenta.len(), "inspection momentum count")?,
                usize_u64(plan.sources.len(), "inspection source count")?,
                usize_u64(plan.parameters.len(), "inspection parameter count")?,
                usize_u64(plan.stages.len(), "inspection stage count")?,
                usize_u64(plan.couplings.len(), "inspection coupling count")?,
                usize_u64(plan.invocations.len(), "inspection invocation count")?,
                usize_u64(plan.attachments.len(), "inspection attachment count")?,
                usize_u64(plan.finalizations.len(), "inspection finalization count")?,
                usize_u64(plan.closures.len(), "inspection closure count")?,
                usize_u64(
                    plan.direct_coefficients.len(),
                    "inspection direct coefficient count",
                )?,
                usize_u64(
                    plan.selector_domains.len(),
                    "inspection selector domain count",
                )?,
                usize_u64(
                    plan.selector_memberships.len(),
                    "inspection selector membership count",
                )?,
                usize_u64(
                    plan.helicity_selectors.len(),
                    "inspection helicity selector count",
                )?,
                usize_u64(
                    plan.color_selectors.len(),
                    "inspection color selector count",
                )?,
                usize_u64(
                    plan.reduction_groups.len(),
                    "inspection reduction group count",
                )?,
                usize_u64(
                    plan.reduction_entries.len(),
                    "inspection reduction entry count",
                )?,
                usize_u64(plan.exact_factors.len(), "inspection exact factor count")?,
                usize_u64(
                    plan.retained_tables.len(),
                    "inspection retained table count",
                )?,
            ],
        })
    }
}

struct TextCatalog<'a> {
    ranges: Vec<EagerPlanCatalogRangeRow>,
    segments: Vec<&'a [u8]>,
    byte_count: u64,
}

impl<'a> TextCatalog<'a> {
    fn new(values: impl IntoIterator<Item = &'a [u8]>) -> RusticolResult<Self> {
        let values = values.into_iter();
        let (minimum, _) = values.size_hint();
        let mut ranges = Vec::with_capacity(minimum);
        let mut segments = Vec::with_capacity(minimum);
        let mut byte_count = 0_u64;
        for value in values {
            let count = usize_u64(value.len(), "text catalog entry length")?;
            ranges.push(EagerPlanCatalogRangeRow {
                start: byte_count,
                count,
            });
            byte_count = byte_count.checked_add(count).ok_or_else(|| {
                RusticolError::serialization("text catalog byte count exceeds u64")
            })?;
            segments.push(value);
        }
        Ok(Self {
            ranges,
            segments,
            byte_count,
        })
    }
}

#[derive(Clone, Copy)]
struct RetainedTableRecord {
    table_id: u32,
    name_id: u32,
    row_count: u64,
    column_start: u64,
    column_count: u64,
}

#[derive(Clone, Copy)]
struct RetainedColumnRecord {
    table_id: u32,
    column_id: u32,
    name_id: u32,
    primitive_kind: u8,
    elements_per_row: u32,
    value_start: u64,
    value_count: u64,
}

struct RetainedCatalog<'a> {
    names: TextCatalog<'a>,
    tables: Vec<RetainedTableRecord>,
    columns: Vec<RetainedColumnRecord>,
    u8_segments: Vec<&'a [u8]>,
    u32_segments: Vec<&'a [u32]>,
    u64_segments: Vec<&'a [u64]>,
    i32_segments: Vec<&'a [i32]>,
    f64_bits_segments: Vec<&'a [u64]>,
    u8_count: u64,
    u32_count: u64,
    u64_count: u64,
    i32_count: u64,
    f64_bits_count: u64,
}

impl<'a> RetainedCatalog<'a> {
    fn new(tables: &'a [EagerOwnedRetainedTable]) -> RusticolResult<Self> {
        let column_capacity = tables.iter().try_fold(0_usize, |total, table| {
            total
                .checked_add(table.columns().len())
                .ok_or_else(|| RusticolError::serialization("retained column count exceeds usize"))
        })?;
        let names = TextCatalog::new(tables.iter().flat_map(|table| {
            std::iter::once(table.name().as_bytes()).chain(
                table
                    .columns()
                    .iter()
                    .map(|column| column.name().as_bytes()),
            )
        }))?;
        let mut records = Vec::with_capacity(tables.len());
        let mut columns = Vec::with_capacity(column_capacity);
        let mut u8_segments = Vec::new();
        let mut u32_segments = Vec::new();
        let mut u64_segments = Vec::new();
        let mut i32_segments = Vec::new();
        let mut f64_bits_segments = Vec::new();
        let mut name_id = 0_u32;
        let mut u8_count = 0_u64;
        let mut u32_count = 0_u64;
        let mut u64_count = 0_u64;
        let mut i32_count = 0_u64;
        let mut f64_bits_count = 0_u64;
        for (table_index, table) in tables.iter().enumerate() {
            let table_id = usize_u32(table_index, "retained table ID")?;
            let column_start = usize_u64(columns.len(), "retained column start")?;
            let table_name_id = name_id;
            name_id = name_id
                .checked_add(1)
                .ok_or_else(|| RusticolError::serialization("retained name count exceeds u32"))?;
            for (column_index, column) in table.columns().iter().enumerate() {
                let column_id = usize_u32(column_index, "retained column ID")?;
                let column_name_id = name_id;
                name_id = name_id.checked_add(1).ok_or_else(|| {
                    RusticolError::serialization("retained name count exceeds u32")
                })?;
                let (primitive_kind, value_start, value_count) = match column.values() {
                    EagerOwnedPrimitiveColumn::U8(values) => {
                        append_segment(&mut u8_segments, &mut u8_count, values, RETAINED_U8)?
                    }
                    EagerOwnedPrimitiveColumn::U32(values) => {
                        append_segment(&mut u32_segments, &mut u32_count, values, RETAINED_U32)?
                    }
                    EagerOwnedPrimitiveColumn::U64(values) => {
                        append_segment(&mut u64_segments, &mut u64_count, values, RETAINED_U64)?
                    }
                    EagerOwnedPrimitiveColumn::I32(values) => {
                        append_segment(&mut i32_segments, &mut i32_count, values, RETAINED_I32)?
                    }
                    EagerOwnedPrimitiveColumn::F64Bits(values) => append_segment(
                        &mut f64_bits_segments,
                        &mut f64_bits_count,
                        values,
                        RETAINED_F64_BITS,
                    )?,
                };
                columns.push(RetainedColumnRecord {
                    table_id,
                    column_id,
                    name_id: column_name_id,
                    primitive_kind,
                    elements_per_row: column.elements_per_row(),
                    value_start,
                    value_count,
                });
            }
            records.push(RetainedTableRecord {
                table_id,
                name_id: table_name_id,
                row_count: table.row_count(),
                column_start,
                column_count: usize_u64(table.columns().len(), "retained table column count")?,
            });
        }
        Ok(Self {
            names,
            tables: records,
            columns,
            u8_segments,
            u32_segments,
            u64_segments,
            i32_segments,
            f64_bits_segments,
            u8_count,
            u32_count,
            u64_count,
            i32_count,
            f64_bits_count,
        })
    }

    fn push_sections<'b>(&'b self, sections: &mut Vec<NamedSection<'b>>) -> RusticolResult<()> {
        push_text_catalog(
            sections,
            "retained/names",
            EagerSectionKind::Metadata,
            &self.names,
        )?;
        push_fixed_table(
            sections,
            "retained/tables.bin",
            EagerSectionKind::Metadata,
            RETAINED_TABLE_RECORD_SIZE,
            &self.tables,
            encode_retained_table,
        )?;
        push_fixed_table(
            sections,
            "retained/columns.bin",
            EagerSectionKind::Metadata,
            RETAINED_COLUMN_RECORD_SIZE,
            &self.columns,
            encode_retained_column,
        )?;
        push_segmented_primitive(
            sections,
            "retained/values-u8.bin",
            EagerSectionKind::Metadata,
            &self.u8_segments,
            self.u8_count,
        )?;
        push_segmented_primitive(
            sections,
            "retained/values-u32.bin",
            EagerSectionKind::Metadata,
            &self.u32_segments,
            self.u32_count,
        )?;
        push_segmented_primitive(
            sections,
            "retained/values-u64.bin",
            EagerSectionKind::Metadata,
            &self.u64_segments,
            self.u64_count,
        )?;
        push_segmented_primitive(
            sections,
            "retained/values-i32.bin",
            EagerSectionKind::Metadata,
            &self.i32_segments,
            self.i32_count,
        )?;
        push_segmented_primitive(
            sections,
            "retained/values-f64-bits.bin",
            EagerSectionKind::Metadata,
            &self.f64_bits_segments,
            self.f64_bits_count,
        )?;
        Ok(())
    }
}

fn append_segment<'a, T>(
    segments: &mut Vec<&'a [T]>,
    total: &mut u64,
    values: &'a [T],
    primitive_kind: u8,
) -> RusticolResult<(u8, u64, u64)> {
    let start = *total;
    let count = usize_u64(values.len(), "retained value count")?;
    *total = total
        .checked_add(count)
        .ok_or_else(|| RusticolError::serialization("retained value count exceeds u64"))?;
    segments.push(values);
    Ok((primitive_kind, start, count))
}

fn usize_u64(value: usize, description: &str) -> RusticolResult<u64> {
    u64::try_from(value)
        .map_err(|_| RusticolError::serialization(format!("{description} exceeds u64")))
}

fn usize_u32(value: usize, description: &str) -> RusticolResult<u32> {
    u32::try_from(value)
        .map_err(|_| RusticolError::serialization(format!("{description} exceeds u32")))
}

struct NamedSection<'a> {
    logical_path: &'static str,
    member_kind: PacbinMemberKind,
    reader: EagerSectionStream<'a>,
}

struct EagerSectionStream<'a> {
    header: [u8; EAGER_SECTION_HEADER_SIZE],
    header_position: usize,
    payload: Box<dyn Read + 'a>,
    payload_remaining: u64,
}

impl<'a> EagerSectionStream<'a> {
    fn new(
        kind: EagerSectionKind,
        record_size: u32,
        record_count: u64,
        payload: impl Read + 'a,
    ) -> RusticolResult<Self> {
        let header = EagerSectionHeader::new(kind, record_size, record_count)?;
        Ok(Self {
            header: header.encode(),
            header_position: 0,
            payload: Box::new(payload),
            payload_remaining: header.payload_length(),
        })
    }
}

impl Read for EagerSectionStream<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        if self.header_position < self.header.len() {
            let count = (self.header.len() - self.header_position).min(output.len());
            output[..count]
                .copy_from_slice(&self.header[self.header_position..self.header_position + count]);
            self.header_position += count;
            written += count;
        }
        if written == output.len() || self.payload_remaining == 0 {
            return Ok(written);
        }
        let maximum = usize::try_from(self.payload_remaining)
            .unwrap_or(usize::MAX)
            .min(output.len() - written);
        let count = self.payload.read(&mut output[written..written + maximum])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "eager section payload ended before its declared length",
            ));
        }
        let count_u64 = u64::try_from(count)
            .map_err(|_| io::Error::other("eager section read count exceeds u64"))?;
        self.payload_remaining -= count_u64;
        written += count;
        Ok(written)
    }
}

struct FixedRowsReader<'a, T> {
    rows: &'a [T],
    record_size: usize,
    total_bytes: u64,
    byte_position: u64,
    encoder: fn(&T, &mut [u8]),
    scratch: [u8; MAX_ENCODED_RECORD_SIZE],
}

impl<'a, T> FixedRowsReader<'a, T> {
    fn new(rows: &'a [T], record_size: u32, encoder: fn(&T, &mut [u8])) -> RusticolResult<Self> {
        let record_size_u64 = u64::from(record_size);
        let record_size = usize::try_from(record_size).map_err(|_| {
            RusticolError::serialization("eager record size exceeds platform usize")
        })?;
        if record_size == 0 || record_size > MAX_ENCODED_RECORD_SIZE {
            return Err(RusticolError::serialization(format!(
                "eager record size {record_size} exceeds streaming scratch bound {MAX_ENCODED_RECORD_SIZE}"
            )));
        }
        let row_count = usize_u64(rows.len(), "fixed eager table row count")?;
        let total_bytes = row_count
            .checked_mul(record_size_u64)
            .ok_or_else(|| RusticolError::serialization("fixed eager table size exceeds u64"))?;
        Ok(Self {
            rows,
            record_size,
            total_bytes,
            byte_position: 0,
            encoder,
            scratch: [0; MAX_ENCODED_RECORD_SIZE],
        })
    }
}

impl<T> Read for FixedRowsReader<'_, T> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.byte_position >= self.total_bytes || output.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        while written < output.len() && self.byte_position < self.total_bytes {
            let record_size_u64 = u64::try_from(self.record_size)
                .map_err(|_| io::Error::other("eager record size exceeds u64"))?;
            let row_index = usize::try_from(self.byte_position / record_size_u64)
                .map_err(|_| io::Error::other("eager row index exceeds usize"))?;
            let row_offset = usize::try_from(self.byte_position % record_size_u64)
                .map_err(|_| io::Error::other("eager row offset exceeds usize"))?;
            self.scratch[..self.record_size].fill(0);
            (self.encoder)(&self.rows[row_index], &mut self.scratch[..self.record_size]);
            let count = (self.record_size - row_offset).min(output.len() - written);
            output[written..written + count]
                .copy_from_slice(&self.scratch[row_offset..row_offset + count]);
            written += count;
            self.byte_position = self
                .byte_position
                .checked_add(
                    u64::try_from(count)
                        .map_err(|_| io::Error::other("eager read count exceeds u64"))?,
                )
                .ok_or_else(|| io::Error::other("eager byte position exceeds u64"))?;
        }
        Ok(written)
    }
}

trait LittleEndianPrimitive: Copy {
    const ENCODED_LEN: u32;
    fn encode(self, output: &mut [u8]);
}

impl LittleEndianPrimitive for u8 {
    const ENCODED_LEN: u32 = 1;

    fn encode(self, output: &mut [u8]) {
        output[0] = self;
    }
}

impl LittleEndianPrimitive for u32 {
    const ENCODED_LEN: u32 = 4;

    fn encode(self, output: &mut [u8]) {
        output.copy_from_slice(&self.to_le_bytes());
    }
}

impl LittleEndianPrimitive for i32 {
    const ENCODED_LEN: u32 = 4;

    fn encode(self, output: &mut [u8]) {
        output.copy_from_slice(&self.to_le_bytes());
    }
}

impl LittleEndianPrimitive for u64 {
    const ENCODED_LEN: u32 = 8;

    fn encode(self, output: &mut [u8]) {
        output.copy_from_slice(&self.to_le_bytes());
    }
}

struct SegmentedPrimitiveReader<'a, T> {
    segments: &'a [&'a [T]],
    segment_index: usize,
    value_index: usize,
    pending: [u8; 8],
    pending_position: usize,
    pending_length: usize,
}

impl<'a, T> SegmentedPrimitiveReader<'a, T> {
    fn new(segments: &'a [&'a [T]]) -> Self {
        Self {
            segments,
            segment_index: 0,
            value_index: 0,
            pending: [0; 8],
            pending_position: 0,
            pending_length: 0,
        }
    }
}

impl<T: LittleEndianPrimitive> Read for SegmentedPrimitiveReader<'_, T> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        while written < output.len() {
            if self.pending_position < self.pending_length {
                let count =
                    (self.pending_length - self.pending_position).min(output.len() - written);
                output[written..written + count].copy_from_slice(
                    &self.pending[self.pending_position..self.pending_position + count],
                );
                self.pending_position += count;
                written += count;
                continue;
            }
            while self.segment_index < self.segments.len()
                && self.value_index == self.segments[self.segment_index].len()
            {
                self.segment_index += 1;
                self.value_index = 0;
            }
            if self.segment_index == self.segments.len() {
                break;
            }
            self.pending.fill(0);
            let value = self.segments[self.segment_index][self.value_index];
            value.encode(&mut self.pending[..T::ENCODED_LEN as usize]);
            self.pending_position = 0;
            self.pending_length = T::ENCODED_LEN as usize;
            self.value_index += 1;
        }
        Ok(written)
    }
}

fn push_fixed_table<'a, T>(
    sections: &mut Vec<NamedSection<'a>>,
    logical_path: &'static str,
    section_kind: EagerSectionKind,
    record_size: u32,
    rows: &'a [T],
    encoder: fn(&T, &mut [u8]),
) -> RusticolResult<()> {
    push_fixed(
        sections,
        logical_path,
        PacbinMemberKind::EagerRuntimeTable,
        section_kind,
        record_size,
        rows,
        encoder,
    )
}

fn push_fixed<'a, T>(
    sections: &mut Vec<NamedSection<'a>>,
    logical_path: &'static str,
    member_kind: PacbinMemberKind,
    section_kind: EagerSectionKind,
    record_size: u32,
    rows: &'a [T],
    encoder: fn(&T, &mut [u8]),
) -> RusticolResult<()> {
    let count = usize_u64(rows.len(), "eager section row count")?;
    sections.push(NamedSection {
        logical_path,
        member_kind,
        reader: EagerSectionStream::new(
            section_kind,
            record_size,
            count,
            FixedRowsReader::new(rows, record_size, encoder)?,
        )?,
    });
    Ok(())
}

fn push_primitive<'a, T: LittleEndianPrimitive + 'a>(
    sections: &mut Vec<NamedSection<'a>>,
    logical_path: &'static str,
    section_kind: EagerSectionKind,
    values: &'a [T],
) -> RusticolResult<()> {
    push_fixed(
        sections,
        logical_path,
        PacbinMemberKind::EagerRuntimeTable,
        section_kind,
        T::ENCODED_LEN,
        values,
        encode_primitive::<T>,
    )
}

fn encode_primitive<T: LittleEndianPrimitive>(value: &T, output: &mut [u8]) {
    value.encode(output);
}

fn push_segmented_primitive<'a, T: LittleEndianPrimitive + 'a>(
    sections: &mut Vec<NamedSection<'a>>,
    logical_path: &'static str,
    section_kind: EagerSectionKind,
    segments: &'a [&'a [T]],
    count: u64,
) -> RusticolResult<()> {
    let actual_count = segments.iter().try_fold(0_u64, |total, values| {
        total
            .checked_add(usize_u64(values.len(), "segmented primitive count")?)
            .ok_or_else(|| RusticolError::serialization("segmented primitive count exceeds u64"))
    })?;
    if actual_count != count {
        return Err(RusticolError::serialization(format!(
            "segmented primitive count mismatch: declared {count}, encoded {actual_count}"
        )));
    }
    sections.push(NamedSection {
        logical_path,
        member_kind: PacbinMemberKind::EagerRuntimeTable,
        reader: EagerSectionStream::new(
            section_kind,
            T::ENCODED_LEN,
            count,
            SegmentedPrimitiveReader::new(segments),
        )?,
    });
    Ok(())
}

fn push_text_catalog<'a>(
    sections: &mut Vec<NamedSection<'a>>,
    prefix: &'static str,
    section_kind: EagerSectionKind,
    catalog: &'a TextCatalog<'a>,
) -> RusticolResult<()> {
    let (ranges_path, bytes_path) = match prefix {
        "metadata/identity" => (
            "metadata/identity-ranges.bin",
            "metadata/identity-bytes.bin",
        ),
        "catalogs/strings" => ("catalogs/strings/ranges.bin", "catalogs/strings/bytes.bin"),
        "catalogs/exact-ir" => (
            "catalogs/exact-ir/ranges.bin",
            "catalogs/exact-ir/bytes.bin",
        ),
        "catalogs/semantic-limitations" => (
            "catalogs/semantic-limitations/ranges.bin",
            "catalogs/semantic-limitations/bytes.bin",
        ),
        "retained/names" => ("retained/name-ranges.bin", "retained/name-bytes.bin"),
        _ => {
            return Err(RusticolError::internal(format!(
                "unknown eager text catalog prefix: {prefix}"
            )));
        }
    };
    push_fixed(
        sections,
        ranges_path,
        PacbinMemberKind::EagerRuntimeMetadata,
        section_kind,
        EagerPlanCatalogRangeRow::ENCODED_LEN,
        &catalog.ranges,
        encode_catalog_range,
    )?;
    sections.push(NamedSection {
        logical_path: bytes_path,
        member_kind: PacbinMemberKind::EagerRuntimeMetadata,
        reader: EagerSectionStream::new(
            section_kind,
            1,
            catalog.byte_count,
            SegmentedPrimitiveReader::new(&catalog.segments),
        )?,
    });
    Ok(())
}

fn encode_metadata(row: &MetadataRecord, output: &mut [u8]) {
    put_u32(output, 0, SERIALIZATION_SCHEMA);
    put_u16(output, 4, EAGER_RUNTIME_CONTAINER_SCHEMA);
    put_u32(output, 8, IDENTITY_LOWERING_ABI);
    put_u32(output, 12, IDENTITY_PLAN_ABI);
    put_u32(output, 16, IDENTITY_RUNTIME_LAYOUT_ABI);
    put_u32(output, 20, IDENTITY_RUNTIME_CAPABILITY);
    put_u32(output, 24, IDENTITY_CONTAINER_KIND);
    put_u32(output, 28, IDENTITY_PROCESS_KEY);
    put_u32(output, 32, IDENTITY_MODEL_NAME);
    put_u32(output, 36, 7);
    let values = [
        row.retained_column_count,
        row.current_component_count,
        row.value_component_count,
        row.momentum_component_count,
        row.color_contraction_entry_start,
        row.color_contraction_entry_count,
        row.string_count,
        row.exact_ir_count,
        row.limitation_count,
        row.retained_table_count,
        row.current_count,
        row.value_count,
        row.momentum_count,
        row.source_count,
        row.stage_count,
        row.invocation_count,
        row.closure_count,
        row.direct_coefficient_count,
        row.helicity_selector_count,
        row.color_selector_count,
        row.exact_factor_count,
    ];
    for (index, value) in values.into_iter().enumerate() {
        put_u64(output, 40 + index * 8, value);
    }
}

fn encode_inspection(row: &InspectionRecord, output: &mut [u8]) {
    for (index, value) in row.counts.into_iter().enumerate() {
        put_u64(output, index * 8, value);
    }
}

fn encode_retained_table(row: &RetainedTableRecord, output: &mut [u8]) {
    put_u32(output, 0, row.table_id);
    put_u32(output, 4, row.name_id);
    put_u64(output, 8, row.row_count);
    put_u64(output, 16, row.column_start);
    put_u64(output, 24, row.column_count);
}

fn encode_retained_column(row: &RetainedColumnRecord, output: &mut [u8]) {
    put_u32(output, 0, row.table_id);
    put_u32(output, 4, row.column_id);
    put_u32(output, 8, row.name_id);
    put_u8(output, 12, row.primitive_kind);
    put_u32(output, 16, row.elements_per_row);
    put_u64(output, 24, row.value_start);
    put_u64(output, 32, row.value_count);
}

fn encode_current(row: &EagerPlanCurrentRow, output: &mut [u8]) {
    put_u32(output, 0, row.current_id);
    put_u64(output, 4, row.component_start);
    put_u32(output, 12, row.component_count);
    put_u32(output, 16, row.momentum_slot_id);
    put_u32(output, 20, row.flags);
}

fn encode_value(row: &EagerPlanValueRow, output: &mut [u8]) {
    put_u32(output, 0, row.value_slot_id);
    put_u32(output, 4, row.current_id);
    put_u64(output, 8, row.component_start);
    put_u32(output, 16, row.component_count);
    put_u8(output, 20, row.kind as u8);
}

fn encode_momentum(row: &EagerPlanMomentumRow, output: &mut [u8]) {
    put_u32(output, 0, row.momentum_slot_id);
    put_u32(output, 4, row.bitset_id);
    put_u64(output, 8, row.component_start);
    put_u32(output, 16, row.component_count);
}

fn encode_source(row: &EagerPlanSourceFillRow, output: &mut [u8]) {
    let values = [
        row.source_id,
        row.current_id,
        row.value_slot_id,
        row.external_label,
        row.input_momentum_slot,
        row.source_ir_id,
        row.crossing_ir_id,
        row.crossing_factor_id,
        row.declared_state_index,
    ];
    put_u32_values(output, &values);
}

fn encode_parameter(row: &EagerPlanParameterRow, output: &mut [u8]) {
    put_u32(output, 0, row.parameter_id);
    put_u32(output, 4, row.name_string_id);
    put_u32(output, 8, row.kind_string_id);
    put_u32(output, 12, row.default_factor_id);
    put_u32(output, 16, row.runtime_name_string_id);
    put_i32(output, 20, row.complex_component);
    put_u32(output, 24, row.flags);
}

fn encode_stage(row: &EagerPlanStageRow, output: &mut [u8]) {
    put_u32(output, 0, row.stage_index);
    put_u32(output, 4, row.subset_size);
    let values = [
        row.invocation_start,
        row.invocation_count,
        row.attachment_start,
        row.attachment_count,
        row.finalization_start,
        row.finalization_count,
    ];
    for (index, value) in values.into_iter().enumerate() {
        put_u64(output, 8 + index * 8, value);
    }
}

fn encode_coupling(row: &EagerPlanCouplingRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.coupling_id,
            row.real_parameter_id,
            row.imaginary_parameter_id,
            row.constant_factor_id,
        ],
    );
}

fn encode_invocation(row: &EagerPlanInvocationRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.evaluation_group_id,
            row.kernel_id,
            row.left_value_slot_id,
            row.right_value_slot_id,
            row.left_momentum_slot_id,
            row.right_momentum_slot_id,
            row.coupling_slot_id,
        ],
    );
    put_u8(output, 28, row.output_factor_source);
    put_u64(output, 32, row.attachment_start);
    put_u64(output, 40, row.attachment_count);
    put_u32(output, 48, row.selector_domain_id);
}

fn encode_attachment(row: &EagerPlanAttachmentRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.interaction_id,
            row.result_current_id,
            row.color_factor_id,
            row.evaluation_factor_id,
            row.normalization_factor_id,
            row.representative_evaluation_factor_id,
            row.selector_domain_id,
        ],
    );
}

fn encode_finalization(row: &EagerPlanFinalizationRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.kernel_id,
            row.current_id,
            row.unpropagated_value_slot_id,
            row.propagated_value_slot_id,
            row.momentum_slot_id,
            row.unpropagated_selector_domain_id,
            row.propagated_selector_domain_id,
        ],
    );
}

fn encode_closure(row: &EagerPlanClosureRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.root_id,
            row.kernel_id,
            row.left_value_slot_id,
            row.right_value_slot_id,
            row.amplitude_index,
            row.coherent_group_id,
            row.coupling_slot_id,
            row.coupling_factor_id,
        ],
    );
    put_u8(output, 32, row.output_factor_source);
    put_u32(output, 36, row.color_factor_id);
    put_u32(output, 40, row.normalization_factor_id);
    put_u64(output, 44, row.direct_coefficient_start);
    put_u64(output, 52, row.direct_coefficient_count);
    put_u32(output, 60, row.selector_domain_id);
}

fn encode_direct_coefficient(row: &EagerPlanDirectCoefficientRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[row.contraction_ir_id, row.component_index, row.factor_id],
    );
}

fn encode_selector_domain(row: &EagerPlanSelectorDomainRow, output: &mut [u8]) {
    put_u64(output, 0, row.member_start);
    put_u64(output, 8, row.member_count);
}

fn encode_helicity_selector(row: &EagerPlanHelicitySelectorRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.selector_id,
            row.values_sequence_id,
            row.representative_sequence_id,
            row.coefficient_factor_id,
        ],
    );
    put_u8(output, 16, row.computed);
    put_u8(output, 17, row.structural_zero);
}

fn encode_color_selector(row: &EagerPlanColorSelectorRow, output: &mut [u8]) {
    put_u32_values(
        output,
        &[
            row.selector_id,
            row.word_sequence_id,
            row.representative_word_sequence_id,
            row.coefficient_factor_id,
        ],
    );
    put_u8(output, 16, row.computed);
}

fn encode_reduction_group(row: &EagerPlanReductionGroupRow, output: &mut [u8]) {
    put_u32(output, 0, row.coherent_group_id);
    put_u64(output, 8, row.amplitude_entry_start);
    put_u64(output, 16, row.amplitude_entry_count);
    put_u64(output, 24, row.selector_entry_start);
    put_u64(output, 32, row.selector_entry_count);
    put_u32(output, 40, row.helicity_weight_factor_id);
    put_u32(output, 44, row.all_sector_weight_factor_id);
}

fn encode_reduction_entry(row: &EagerPlanReductionEntryRow, output: &mut [u8]) {
    put_u8(output, 0, row.kind as u8);
    put_u32(output, 4, row.owner_id);
    put_u32(output, 8, row.left_id);
    put_u32(output, 12, row.right_id);
    put_u32(output, 16, row.factor_id);
    put_u32(output, 20, row.auxiliary_factor_id);
}

fn encode_exact_factor(row: &EagerPlanExactFactorRow, output: &mut [u8]) {
    put_u32(output, 0, row.factor_id);
    put_u64(output, 8, row.real_bits);
    put_u64(output, 16, row.imaginary_bits);
    put_u32(output, 24, row.canonical_string_id);
    put_u8(output, 28, row.exact_source);
    put_u32(output, 32, row.exact_ir_id);
    put_u32(output, 36, row.source_ir_id);
}

fn encode_catalog_range(row: &EagerPlanCatalogRangeRow, output: &mut [u8]) {
    put_u64(output, 0, row.start);
    put_u64(output, 8, row.count);
}

fn put_u32_values(output: &mut [u8], values: &[u32]) {
    for (index, value) in values.iter().copied().enumerate() {
        put_u32(output, index * 4, value);
    }
}

fn put_u8(output: &mut [u8], offset: usize, value: u8) {
    output[offset] = value;
}

fn put_u16(output: &mut [u8], offset: usize, value: u16) {
    output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(output: &mut [u8], offset: usize, value: i32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
#[path = "eager_plan_v3_pacbin_tests.rs"]
mod tests;
