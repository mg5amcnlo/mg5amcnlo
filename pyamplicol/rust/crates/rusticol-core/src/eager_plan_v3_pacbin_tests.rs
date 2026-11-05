// SPDX-License-Identifier: 0BSD

use super::*;
use crate::RusticolErrorKind;
use crate::eager_lowering_v3::{EagerPlanReductionEntryKind, EagerValueSlotKind};
use crate::pacbin::PacbinReader;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

const EXPECTED_MEMBERS: &[&str] = &[
    "catalogs/bitsets/populations.bin",
    "catalogs/bitsets/ranges.bin",
    "catalogs/bitsets/words.bin",
    "catalogs/exact-factors.bin",
    "catalogs/exact-ir/bytes.bin",
    "catalogs/exact-ir/ranges.bin",
    "catalogs/i32-sequences/ranges.bin",
    "catalogs/i32-sequences/values.bin",
    "catalogs/semantic-limitations/bytes.bin",
    "catalogs/semantic-limitations/ranges.bin",
    "catalogs/strings/bytes.bin",
    "catalogs/strings/ranges.bin",
    "catalogs/u32-sequences/ranges.bin",
    "catalogs/u32-sequences/values.bin",
    "inspection/summary.bin",
    "metadata/core.bin",
    "metadata/identity-bytes.bin",
    "metadata/identity-ranges.bin",
    "reductions/entries.bin",
    "reductions/groups.bin",
    "retained/columns.bin",
    "retained/name-bytes.bin",
    "retained/name-ranges.bin",
    "retained/tables.bin",
    "retained/values-f64-bits.bin",
    "retained/values-i32.bin",
    "retained/values-u32.bin",
    "retained/values-u64.bin",
    "retained/values-u8.bin",
    "selectors/colors.bin",
    "selectors/domains.bin",
    "selectors/helicities.bin",
    "selectors/memberships.bin",
    "tables/attachments.bin",
    "tables/closures.bin",
    "tables/couplings.bin",
    "tables/currents.bin",
    "tables/direct-coefficients.bin",
    "tables/finalizations.bin",
    "tables/invocations.bin",
    "tables/momenta.bin",
    "tables/parameters.bin",
    "tables/sources.bin",
    "tables/stages.bin",
    "tables/values.bin",
];

struct Fixture {
    strings: Vec<Box<str>>,
    exact_ir: Vec<Box<str>>,
    limitations: Vec<Box<str>>,
    retained: Vec<EagerOwnedRetainedTable>,
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
    u32_ranges: Vec<EagerPlanCatalogRangeRow>,
    u32_values: Vec<u32>,
    i32_ranges: Vec<EagerPlanCatalogRangeRow>,
    i32_values: Vec<i32>,
}

impl Fixture {
    fn populated() -> Self {
        Self {
            strings: vec!["one".into(), "two".into()],
            exact_ir: vec!["{\"ir\":1}".into()],
            limitations: vec!["exact limitation".into()],
            retained: vec![],
            currents: vec![EagerPlanCurrentRow {
                current_id: 11,
                component_start: 12,
                component_count: 13,
                momentum_slot_id: 14,
                flags: 15,
            }],
            values: vec![EagerPlanValueRow {
                value_slot_id: 21,
                current_id: 22,
                component_start: 23,
                component_count: 24,
                kind: EagerValueSlotKind::Propagated,
            }],
            momenta: vec![EagerPlanMomentumRow {
                momentum_slot_id: 31,
                bitset_id: 32,
                component_start: 33,
                component_count: 34,
            }],
            sources: vec![EagerPlanSourceFillRow {
                source_id: 41,
                current_id: 42,
                value_slot_id: 43,
                external_label: 44,
                input_momentum_slot: 45,
                source_ir_id: 46,
                crossing_ir_id: 47,
                crossing_factor_id: 48,
                declared_state_index: 49,
            }],
            parameters: vec![EagerPlanParameterRow {
                parameter_id: 51,
                name_string_id: 52,
                kind_string_id: 53,
                default_factor_id: 54,
                runtime_name_string_id: 55,
                complex_component: -56,
                flags: 57,
            }],
            stages: vec![EagerPlanStageRow {
                stage_index: 61,
                subset_size: 62,
                invocation_start: 63,
                invocation_count: 64,
                attachment_start: 65,
                attachment_count: 66,
                finalization_start: 67,
                finalization_count: 68,
            }],
            couplings: vec![EagerPlanCouplingRow {
                coupling_id: 71,
                real_parameter_id: 72,
                imaginary_parameter_id: 73,
                constant_factor_id: 74,
            }],
            invocations: vec![EagerPlanInvocationRow {
                evaluation_group_id: 81,
                kernel_id: 82,
                left_value_slot_id: 83,
                right_value_slot_id: 84,
                left_momentum_slot_id: 85,
                right_momentum_slot_id: 86,
                coupling_slot_id: 87,
                output_factor_source: 88,
                attachment_start: 89,
                attachment_count: 90,
                selector_domain_id: 91,
            }],
            attachments: vec![EagerPlanAttachmentRow {
                interaction_id: 101,
                result_current_id: 102,
                color_factor_id: 103,
                evaluation_factor_id: 104,
                normalization_factor_id: 105,
                representative_evaluation_factor_id: 106,
                selector_domain_id: 107,
            }],
            finalizations: vec![EagerPlanFinalizationRow {
                kernel_id: 111,
                current_id: 112,
                unpropagated_value_slot_id: 113,
                propagated_value_slot_id: 114,
                momentum_slot_id: 115,
                unpropagated_selector_domain_id: 116,
                propagated_selector_domain_id: 117,
            }],
            closures: vec![EagerPlanClosureRow {
                root_id: 121,
                kernel_id: 122,
                left_value_slot_id: 123,
                right_value_slot_id: 124,
                amplitude_index: 125,
                coherent_group_id: 126,
                coupling_slot_id: 127,
                coupling_factor_id: 128,
                output_factor_source: 129,
                color_factor_id: 130,
                normalization_factor_id: 131,
                direct_coefficient_start: 132,
                direct_coefficient_count: 133,
                selector_domain_id: 134,
            }],
            direct_coefficients: vec![EagerPlanDirectCoefficientRow {
                contraction_ir_id: 141,
                component_index: 142,
                factor_id: 143,
            }],
            selector_domains: vec![EagerPlanSelectorDomainRow {
                member_start: 151,
                member_count: 152,
            }],
            selector_memberships: vec![153],
            helicity_selectors: vec![EagerPlanHelicitySelectorRow {
                selector_id: 161,
                values_sequence_id: 162,
                representative_sequence_id: 163,
                coefficient_factor_id: 164,
                computed: 1,
                structural_zero: 0,
            }],
            color_selectors: vec![EagerPlanColorSelectorRow {
                selector_id: 171,
                word_sequence_id: 172,
                representative_word_sequence_id: 173,
                coefficient_factor_id: 174,
                computed: 1,
            }],
            reduction_groups: vec![EagerPlanReductionGroupRow {
                coherent_group_id: 181,
                amplitude_entry_start: 182,
                amplitude_entry_count: 183,
                selector_entry_start: 184,
                selector_entry_count: 185,
                helicity_weight_factor_id: 186,
                all_sector_weight_factor_id: 187,
            }],
            reduction_entries: vec![EagerPlanReductionEntryRow {
                kind: EagerPlanReductionEntryKind::ColorContraction,
                owner_id: 191,
                left_id: 192,
                right_id: 193,
                factor_id: 194,
                auxiliary_factor_id: 195,
            }],
            exact_factors: vec![EagerPlanExactFactorRow {
                factor_id: 201,
                real_bits: (-0.0_f64).to_bits(),
                imaginary_bits: 2.5_f64.to_bits(),
                canonical_string_id: 202,
                exact_source: 203,
                exact_ir_id: 204,
                source_ir_id: 205,
            }],
            bitset_ranges: vec![EagerPlanCatalogRangeRow {
                start: 211,
                count: 212,
            }],
            bitset_populations: vec![213],
            bitset_words: vec![214],
            u32_ranges: vec![EagerPlanCatalogRangeRow {
                start: 221,
                count: 222,
            }],
            u32_values: vec![223],
            i32_ranges: vec![EagerPlanCatalogRangeRow {
                start: 231,
                count: 232,
            }],
            i32_values: vec![-233],
        }
    }

    fn view(&self) -> PlanView<'_> {
        PlanView {
            abi: EAGER_PLAN_ABI,
            process_key: "fixture-process",
            model_name: "fixture-model",
            string_catalog: &self.strings,
            canonical_ir_catalog: &self.exact_ir,
            semantic_limitations: &self.limitations,
            retained_tables: &self.retained,
            currents: &self.currents,
            values: &self.values,
            momenta: &self.momenta,
            sources: &self.sources,
            parameters: &self.parameters,
            stages: &self.stages,
            couplings: &self.couplings,
            invocations: &self.invocations,
            attachments: &self.attachments,
            finalizations: &self.finalizations,
            closures: &self.closures,
            direct_coefficients: &self.direct_coefficients,
            selector_domains: &self.selector_domains,
            selector_memberships: &self.selector_memberships,
            helicity_selectors: &self.helicity_selectors,
            color_selectors: &self.color_selectors,
            reduction_groups: &self.reduction_groups,
            reduction_entries: &self.reduction_entries,
            exact_factors: &self.exact_factors,
            bitset_ranges: &self.bitset_ranges,
            bitset_populations: &self.bitset_populations,
            bitset_words: &self.bitset_words,
            u32_sequence_ranges: &self.u32_ranges,
            u32_sequence_values: &self.u32_values,
            i32_sequence_ranges: &self.i32_ranges,
            i32_sequence_values: &self.i32_values,
            current_component_count: 301,
            value_component_count: 302,
            momentum_component_count: 303,
            color_contraction_entry_start: 304,
            color_contraction_entry_count: 305,
        }
    }
}

fn temporary_directory(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "rusticol-eager-v3-pacbin-{label}-{}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn section_payload<'a>(reader: &'a PacbinReader, path: &str) -> &'a [u8] {
    let bytes = reader.member_bytes(path).unwrap();
    let (_, payload) = EagerSectionHeader::decode(bytes).unwrap();
    payload
}

#[test]
fn eager_plan_v3_pacbin_is_byte_identical_and_metadata_is_bounded() {
    let directory = temporary_directory("deterministic");
    let first = directory.join("first.pacbin");
    let second = directory.join("second.pacbin");
    let fixture = Fixture::populated();
    let first_metadata = write_plan_view(&fixture.view(), &first).unwrap();
    let second_metadata = write_plan_view(&fixture.view(), &second).unwrap();
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    assert_eq!(first_metadata, second_metadata);
    assert_eq!(first_metadata.member_count, EXPECTED_MEMBERS.len() as u64);
    assert_eq!(
        first_metadata.file_size,
        fs::metadata(&first).unwrap().len()
    );
    let expected_file_digest: [u8; 32] = Sha256::digest(fs::read(&first).unwrap()).into();
    assert_eq!(first_metadata.file_sha256, expected_file_digest);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn reader_authenticates_allowlisted_members_and_every_plan_field() {
    let directory = temporary_directory("coverage");
    let destination = directory.join("eager-runtime.pacbin");
    let fixture = Fixture::populated();
    write_plan_view(&fixture.view(), &destination).unwrap();
    let reader = PacbinReader::open(&destination).unwrap();
    reader.verify_payloads().unwrap();
    let actual: Vec<_> = reader
        .members()
        .iter()
        .map(|member| member.logical_path())
        .collect();
    assert_eq!(actual, EXPECTED_MEMBERS);
    for member in reader.members() {
        let expected_kind = if member.logical_path().starts_with("metadata/")
            || member.logical_path().starts_with("inspection/")
            || member.logical_path().ends_with("/bytes.bin")
            || member.logical_path().ends_with("/ranges.bin")
                && (member.logical_path().starts_with("catalogs/strings")
                    || member.logical_path().starts_with("catalogs/exact-ir")
                    || member
                        .logical_path()
                        .starts_with("catalogs/semantic-limitations"))
            || member.logical_path().starts_with("retained/name-")
        {
            PacbinMemberKind::EagerRuntimeMetadata
        } else {
            PacbinMemberKind::EagerRuntimeTable
        };
        assert_eq!(member.kind(), expected_kind, "{}", member.logical_path());
        let (header, payload) =
            EagerSectionHeader::decode(reader.member_bytes(member.logical_path()).unwrap())
                .unwrap();
        assert_eq!(payload.len() as u64, header.payload_length());
    }

    let metadata = section_payload(&reader, "metadata/core.bin");
    assert_eq!(u32_at(metadata, 0), SERIALIZATION_SCHEMA);
    assert_eq!(u32_at(metadata, 8), IDENTITY_LOWERING_ABI);
    assert_eq!(u32_at(metadata, 32), IDENTITY_MODEL_NAME);
    assert_eq!(u64_at(metadata, 48), 301);
    assert_eq!(u64_at(metadata, 56), 302);
    assert_eq!(u64_at(metadata, 64), 303);
    assert_eq!(u64_at(metadata, 72), 304);
    assert_eq!(u64_at(metadata, 80), 305);

    let current = section_payload(&reader, "tables/currents.bin");
    assert_eq!(u32_at(current, 0), 11);
    assert_eq!(u64_at(current, 4), 12);
    assert_eq!(u32_at(current, 20), 15);
    let closure = section_payload(&reader, "tables/closures.bin");
    assert_eq!(u32_at(closure, 0), 121);
    assert_eq!(u64_at(closure, 44), 132);
    assert_eq!(u32_at(closure, 60), 134);
    let direct = section_payload(&reader, "tables/direct-coefficients.bin");
    assert_eq!(
        (u32_at(direct, 0), u32_at(direct, 4), u32_at(direct, 8)),
        (141, 142, 143)
    );
    let helicity = section_payload(&reader, "selectors/helicities.bin");
    assert_eq!(
        &helicity[..18],
        &[161, 0, 0, 0, 162, 0, 0, 0, 163, 0, 0, 0, 164, 0, 0, 0, 1, 0]
    );
    let exact = section_payload(&reader, "catalogs/exact-factors.bin");
    assert_eq!(u64_at(exact, 8), (-0.0_f64).to_bits());
    assert_eq!(u64_at(exact, 16), 2.5_f64.to_bits());
    assert_eq!(u32_at(exact, 36), 205);
    let identity = section_payload(&reader, "metadata/identity-bytes.bin");
    let identity_text = std::str::from_utf8(identity).unwrap();
    assert!(identity_text.contains(EAGER_PLAN_ABI));
    assert!(identity_text.contains("fixture-process"));
    assert!(identity_text.contains("fixture-model"));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn malformed_destination_preserves_existing_target_and_cleans_staging() {
    let directory = temporary_directory("rollback");
    let destination = directory.join("eager-runtime.pacbin");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("sentinel"), b"keep").unwrap();
    let fixture = Fixture::populated();
    let error = write_plan_view(&fixture.view(), &destination).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert_eq!(fs::read(destination.join("sentinel")).unwrap(), b"keep");
    let names: Vec<_> = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        names,
        vec![std::ffi::OsString::from("eager-runtime.pacbin")]
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn large_catalog_serialization_keeps_payload_independent_scratch() {
    let directory = temporary_directory("bounded");
    let destination = directory.join("eager-runtime.pacbin");
    let mut fixture = Fixture::populated();
    fixture.u32_values = vec![0x1234_5678; 1_000_000];
    fixture.u32_ranges = vec![EagerPlanCatalogRangeRow {
        start: 0,
        count: fixture.u32_values.len() as u64,
    }];
    let metadata = write_plan_view(&fixture.view(), &destination).unwrap();
    assert!(metadata.unpacked_bytes > 4_000_000);
    assert_eq!(MAX_ENCODED_RECORD_SIZE, 224);
    assert!(
        std::mem::size_of::<FixedRowsReader<'_, EagerPlanCurrentRow>>()
            < MAX_ENCODED_RECORD_SIZE + 128
    );
    assert!(
        std::mem::size_of::<SegmentedPrimitiveReader<'_, u32>>() < 128,
        "segmented reader must retain only cursors and one primitive"
    );
    let reader = PacbinReader::open(&destination).unwrap();
    let values = section_payload(&reader, "catalogs/u32-sequences/values.bin");
    assert_eq!(values.len(), 4_000_000);
    assert_eq!(u32_at(values, values.len() - 4), 0x1234_5678);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn retained_table_and_column_records_preserve_every_descriptor_field() {
    let table = RetainedTableRecord {
        table_id: 1,
        name_id: 2,
        row_count: 3,
        column_start: 4,
        column_count: 5,
    };
    let mut table_bytes = [0_u8; RETAINED_TABLE_RECORD_SIZE as usize];
    encode_retained_table(&table, &mut table_bytes);
    assert_eq!(u32_at(&table_bytes, 0), 1);
    assert_eq!(u32_at(&table_bytes, 4), 2);
    assert_eq!(u64_at(&table_bytes, 8), 3);
    assert_eq!(u64_at(&table_bytes, 16), 4);
    assert_eq!(u64_at(&table_bytes, 24), 5);

    let column = RetainedColumnRecord {
        table_id: 6,
        column_id: 7,
        name_id: 8,
        primitive_kind: RETAINED_F64_BITS,
        elements_per_row: 9,
        value_start: 10,
        value_count: 11,
    };
    let mut column_bytes = [0_u8; RETAINED_COLUMN_RECORD_SIZE as usize];
    encode_retained_column(&column, &mut column_bytes);
    assert_eq!(u32_at(&column_bytes, 0), 6);
    assert_eq!(u32_at(&column_bytes, 4), 7);
    assert_eq!(u32_at(&column_bytes, 8), 8);
    assert_eq!(column_bytes[12], RETAINED_F64_BITS);
    assert_eq!(u32_at(&column_bytes, 16), 9);
    assert_eq!(u64_at(&column_bytes, 24), 10);
    assert_eq!(u64_at(&column_bytes, 32), 11);
}
