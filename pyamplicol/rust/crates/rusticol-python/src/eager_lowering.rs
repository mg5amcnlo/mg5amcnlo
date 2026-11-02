// SPDX-License-Identifier: 0BSD

//! Private Python boundary for compact eager plan-v3 lowering.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rusticol_core::eager_layout::{
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY, EAGER_LOWERING_INPUT_ABI, EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY, EAGER_RUNTIME_CONTAINER_KIND, EAGER_RUNTIME_CONTAINER_SCHEMA,
    EAGER_RUNTIME_LAYOUT_ABI,
};
use rusticol_core::{
    EAGER_LOWERING_V1_TABLE_NAMES, EagerBitsetColumns, EagerCoherentGroupColumns,
    EagerColorContractionEntryColumns, EagerColorContractionMetadata, EagerColorSelectorColumns,
    EagerContractionCoefficientColumns, EagerCouplingColumns, EagerCurrentColumns,
    EagerExactFactorColumns, EagerHelicitySelectorColumns, EagerI32SequenceColumns,
    EagerInteractionColumns, EagerInteractionGroupColumns, EagerLoweringInputV1,
    EagerLoweringInputV1View, EagerModelParameterColumns, EagerMomentumMaskColumns,
    EagerPrimitiveColumnView, EagerReductionMemberColumns, EagerRetainedColumnView,
    EagerRetainedTableView, EagerRootColumns, EagerSourceColumns, EagerU32SequenceColumns,
    RusticolError, RusticolResult, lower_eager_plan_v3, write_eager_plan_v3_pacbin,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::python_error;

const INPUT_SCHEMA_SHA256: &str =
    "59d5a3193ca25af6224c296297425782c394a34e352be622f47045e90d2bce29";
const RESULT_KIND: &str = "pyamplicol-eager-runtime-lowering-result";
const RESULT_SCHEMA_VERSION: u32 = 1;
const STORAGE_ABI: &str = "pacbin-v1";
const SELECTOR_WORK_ABI: &str = "pyamplicol-eager-selector-work-v1";

struct NativeLoweringResult {
    process_key: String,
    model_name: String,
    member_count: u64,
    unpacked_size_bytes: u64,
    index_sha256: String,
    current_count: usize,
    value_count: usize,
    momentum_count: usize,
    source_count: usize,
    parameter_count: usize,
    stage_count: usize,
    coupling_count: usize,
    invocation_count: usize,
    attachment_count: usize,
    finalization_count: usize,
    closure_count: usize,
    selector_domain_count: usize,
    reduction_group_count: usize,
    reduction_entry_count: usize,
    retained_table_count: usize,
    current_component_count: u64,
    value_component_count: u64,
    momentum_component_count: u64,
    selected_flow_selector_count: u64,
    selected_flow_current_count: u64,
    selected_flow_evaluation_count: u64,
    selected_flow_attachment_count: u64,
    all_flow_selector_count: u64,
    all_flow_current_count: u64,
    all_flow_evaluation_count: u64,
    all_flow_attachment_count: u64,
    contracted_current_count: u64,
    contracted_evaluation_count: u64,
    contracted_attachment_count: u64,
}

enum BorrowedValues<'py> {
    U8(PyReadonlyArrayDyn<'py, u8>),
    U32(PyReadonlyArrayDyn<'py, u32>),
    U64(PyReadonlyArrayDyn<'py, u64>),
    I32(PyReadonlyArrayDyn<'py, i32>),
    F64(PyReadonlyArrayDyn<'py, f64>),
}

impl BorrowedValues<'_> {
    fn raw_bytes(&self) -> PyResult<&[u8]> {
        fn bytes<T>(values: &[T]) -> &[u8] {
            // All accepted dtypes are explicit little-endian primitives.
            unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            }
        }
        match self {
            Self::U8(values) => values.as_slice().map_err(not_contiguous),
            Self::U32(values) => values.as_slice().map(bytes).map_err(not_contiguous),
            Self::U64(values) => values.as_slice().map(bytes).map_err(not_contiguous),
            Self::I32(values) => values.as_slice().map(bytes).map_err(not_contiguous),
            Self::F64(values) => values.as_slice().map(bytes).map_err(not_contiguous),
        }
    }

    fn primitive_view(&self) -> RusticolResult<EagerPrimitiveColumnView<'_>> {
        match self {
            Self::U8(values) => Ok(EagerPrimitiveColumnView::U8(
                values.as_slice().map_err(core_not_contiguous)?,
            )),
            Self::U32(values) => Ok(EagerPrimitiveColumnView::U32(
                values.as_slice().map_err(core_not_contiguous)?,
            )),
            Self::U64(values) => Ok(EagerPrimitiveColumnView::U64(
                values.as_slice().map_err(core_not_contiguous)?,
            )),
            Self::I32(values) => Ok(EagerPrimitiveColumnView::I32(
                values.as_slice().map_err(core_not_contiguous)?,
            )),
            Self::F64(values) => Ok(EagerPrimitiveColumnView::F64(
                values.as_slice().map_err(core_not_contiguous)?,
            )),
        }
    }

    fn as_u8(&self, context: &str) -> RusticolResult<&[u8]> {
        match self.primitive_view()? {
            EagerPrimitiveColumnView::U8(values) => Ok(values),
            _ => Err(wrong_type(context, "u8")),
        }
    }

    fn as_u32(&self, context: &str) -> RusticolResult<&[u32]> {
        match self.primitive_view()? {
            EagerPrimitiveColumnView::U32(values) => Ok(values),
            _ => Err(wrong_type(context, "u32")),
        }
    }

    fn as_u64(&self, context: &str) -> RusticolResult<&[u64]> {
        match self.primitive_view()? {
            EagerPrimitiveColumnView::U64(values) => Ok(values),
            _ => Err(wrong_type(context, "u64")),
        }
    }

    fn as_i32(&self, context: &str) -> RusticolResult<&[i32]> {
        match self.primitive_view()? {
            EagerPrimitiveColumnView::I32(values) => Ok(values),
            _ => Err(wrong_type(context, "i32")),
        }
    }

    fn as_f64(&self, context: &str) -> RusticolResult<&[f64]> {
        match self.primitive_view()? {
            EagerPrimitiveColumnView::F64(values) => Ok(values),
            _ => Err(wrong_type(context, "f64")),
        }
    }
}

struct BorrowedColumn<'py> {
    name: String,
    shape: Vec<usize>,
    values: BorrowedValues<'py>,
}

struct BorrowedTable<'py> {
    name: String,
    row_count: u64,
    columns: Vec<BorrowedColumn<'py>>,
    column_by_name: BTreeMap<String, usize>,
}

impl BorrowedTable<'_> {
    fn column(&self, name: &str) -> RusticolResult<&BorrowedValues<'_>> {
        let index = self.column_by_name.get(name).ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "eager lowering table {:?} has no column {name:?}",
                self.name
            ))
        })?;
        Ok(&self.columns[*index].values)
    }
}

struct BorrowedInput<'py> {
    abi: String,
    process_key: String,
    model_name: String,
    string_catalog: Vec<String>,
    canonical_ir_catalog: Vec<String>,
    semantic_limitations: Vec<String>,
    tables: Vec<BorrowedTable<'py>>,
    table_by_name: BTreeMap<String, usize>,
    digest: String,
}

impl BorrowedInput<'_> {
    fn table(&self, name: &str) -> RusticolResult<&BorrowedTable<'_>> {
        let index = self.table_by_name.get(name).ok_or_else(|| {
            RusticolError::invalid_argument(format!("eager lowering input has no table {name:?}"))
        })?;
        Ok(&self.tables[*index])
    }

    fn column(&self, table: &str, column: &str) -> RusticolResult<&BorrowedValues<'_>> {
        self.table(table)?.column(column)
    }

    fn into_core_input(&self) -> RusticolResult<EagerLoweringInputV1> {
        let strings = self
            .string_catalog
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let canonical_ir = self
            .canonical_ir_catalog
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let limitations = self
            .semantic_limitations
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let retained_columns =
            self.tables
                .iter()
                .map(|table| {
                    table
                        .columns
                        .iter()
                        .filter(|column| !is_consumed_column(&table.name, &column.name))
                        .map(|column| {
                            let elements_per_row = column.shape.iter().skip(1).try_fold(
                                1_u32,
                                |product, dimension| {
                                    let dimension = u32::try_from(*dimension).map_err(|_| {
                                        RusticolError::invalid_argument(format!(
                                            "{}.{} trailing shape exceeds u32",
                                            table.name, column.name
                                        ))
                                    })?;
                                    product.checked_mul(dimension).ok_or_else(|| {
                                        RusticolError::invalid_argument(format!(
                                            "{}.{} elements per row exceed u32",
                                            table.name, column.name
                                        ))
                                    })
                                },
                            )?;
                            Ok(EagerRetainedColumnView {
                                name: &column.name,
                                elements_per_row,
                                values: column.values.primitive_view()?,
                            })
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .collect::<RusticolResult<Vec<_>>>()?;
        let retained_tables = self
            .tables
            .iter()
            .zip(&retained_columns)
            .map(|(table, columns)| EagerRetainedTableView {
                name: &table.name,
                row_count: table.row_count,
                columns,
            })
            .collect::<Vec<_>>();
        let u8c = |table, column| {
            self.column(table, column)?
                .as_u8(&format!("{table}.{column}"))
        };
        let u8x2 = |table, column| -> RusticolResult<&[[u8; 2]]> {
            let values = u8c(table, column)?;
            if values.len() % 2 != 0 {
                return Err(RusticolError::invalid_argument(format!(
                    "{table}.{column} has an invalid flattened shape"
                )));
            }
            Ok(unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<[u8; 2]>(), values.len() / 2)
            })
        };
        let u32c = |table, column| {
            self.column(table, column)?
                .as_u32(&format!("{table}.{column}"))
        };
        let u64c = |table, column| {
            self.column(table, column)?
                .as_u64(&format!("{table}.{column}"))
        };
        let i32c = |table, column| {
            self.column(table, column)?
                .as_i32(&format!("{table}.{column}"))
        };
        let f64c = |table, column| {
            self.column(table, column)?
                .as_f64(&format!("{table}.{column}"))
        };
        let contraction = self.table("color_contraction_metadata")?;
        if contraction.row_count != 1 {
            return Err(RusticolError::invalid_argument(
                "color_contraction_metadata must contain exactly one row",
            ));
        }
        EagerLoweringInputV1::try_from_view(EagerLoweringInputV1View {
            abi: &self.abi,
            process_key: &self.process_key,
            model_name: &self.model_name,
            string_catalog: &strings,
            canonical_ir_catalog: &canonical_ir,
            semantic_limitations: &limitations,
            retained_tables: &retained_tables,
            unsupported_tables: &[],
            currents: EagerCurrentColumns {
                id: u32c("currents", "id")?,
                dimension: u32c("currents", "dimension")?,
                is_source: u8c("currents", "is_source")?,
                momentum_mask_bitset_id: u32c("currents", "momentum_mask_bitset_id")?,
                propagator_ir_id: u32c("currents", "propagator_ir_id")?,
                propagator_kernel_id: u32c("currents", "propagator_kernel_id")?,
            },
            sources: EagerSourceColumns {
                source_id: u32c("sources", "source_id")?,
                current_id: u32c("sources", "current_id")?,
                external_label: u32c("sources", "external_label")?,
                input_momentum_slot: u32c("sources", "input_momentum_slot")?,
                source_ir_id: u32c("sources", "source_ir_id")?,
                crossing_ir_id: u32c("sources", "crossing_ir_id")?,
                crossing_factor_id: u32c("sources", "crossing_factor_id")?,
                declared_state_index: u32c("sources", "declared_state_index")?,
            },
            interactions: EagerInteractionColumns {
                id: u32c("interactions", "id")?,
                stage_subset_size: u32c("interactions", "stage_subset_size")?,
                left_current_id: u32c("interactions", "left_current_id")?,
                right_current_id: u32c("interactions", "right_current_id")?,
                result_current_id: u32c("interactions", "result_current_id")?,
                coupling_id: u32c("interactions", "coupling_id")?,
                coupling_factor_id: u32c("interactions", "coupling_factor_id")?,
                color_factor_id: u32c("interactions", "color_factor_id")?,
                evaluation_factor_id: u32c("interactions", "evaluation_factor_id")?,
                evaluation_group_id: u32c("interactions", "evaluation_group_id")?,
                full_tensor_network_ready: u8c("interactions", "full_tensor_network_ready")?,
                kernel_id: u32c("interactions", "kernel_id")?,
                canonical_input_order: u8x2("interactions", "canonical_input_order")?,
                kernel_normalization_factor_id: u32c(
                    "interactions",
                    "kernel_normalization_factor_id",
                )?,
                output_factor_source: u8c("interactions", "output_factor_source")?,
            },
            roots: EagerRootColumns {
                id: u32c("roots", "id")?,
                left_current_id: u32c("roots", "left_current_id")?,
                right_current_id: u32c("roots", "right_current_id")?,
                color_factor_id: u32c("roots", "color_factor_id")?,
                contraction_ir_id: u32c("roots", "contraction_ir_id")?,
                coupling_id: u32c("roots", "coupling_id")?,
                coupling_factor_id: u32c("roots", "coupling_factor_id")?,
                helicity_weight_factor_id: u32c("roots", "helicity_weight_factor_id")?,
                coherent_group_id: u32c("roots", "coherent_group_id")?,
                kernel_id: u32c("roots", "kernel_id")?,
                canonical_input_order: u8x2("roots", "canonical_input_order")?,
                kernel_normalization_factor_id: u32c("roots", "kernel_normalization_factor_id")?,
                output_factor_source: u8c("roots", "output_factor_source")?,
            },
            interaction_groups: EagerInteractionGroupColumns {
                representative_interaction_id: u32c(
                    "interaction_groups",
                    "representative_interaction_id",
                )?,
                member_start: u64c("interaction_groups", "member_start")?,
                member_count: u64c("interaction_groups", "member_count")?,
                member_interaction_id: u32c("interaction_group_members", "interaction_id")?,
            },
            momentum_masks: EagerMomentumMaskColumns {
                slot_id: u32c("momentum_masks", "slot_id")?,
                bitset_id: u32c("momentum_masks", "bitset_id")?,
            },
            bitsets: EagerBitsetColumns {
                start: u64c("bitset_ranges", "start")?,
                count: u64c("bitset_ranges", "count")?,
                bit_count: u64c("bitset_ranges", "bit_count")?,
                words: u64c("bitset_words", "value")?,
            },
            u32_sequences: EagerU32SequenceColumns {
                start: u64c("u32_sequence_ranges", "start")?,
                count: u64c("u32_sequence_ranges", "count")?,
                values: u32c("u32_sequence_values", "value")?,
            },
            i32_sequences: EagerI32SequenceColumns {
                start: u64c("i32_sequence_ranges", "start")?,
                count: u64c("i32_sequence_ranges", "count")?,
                values: i32c("i32_sequence_values", "value")?,
            },
            exact_factors: EagerExactFactorColumns {
                real: f64c("exact_factors", "real")?,
                imaginary: f64c("exact_factors", "imaginary")?,
                canonical_string_id: u32c("exact_factors", "canonical_string_id")?,
                exact_source: u8c("exact_factors", "exact_source")?,
                exact_ir_id: u32c("exact_factors", "exact_ir_id")?,
                source_ir_id: u32c("exact_factors", "source_ir_id")?,
            },
            couplings: EagerCouplingColumns {
                constant_factor_id: u32c("couplings", "constant_factor_id")?,
                parameter_name_ids_sequence_id: u32c(
                    "couplings",
                    "parameter_name_ids_sequence_id",
                )?,
            },
            model_parameters: EagerModelParameterColumns {
                name_string_id: u32c("model_parameters", "name_string_id")?,
                kind_string_id: u32c("model_parameters", "kind_string_id")?,
                default_value: f64c("model_parameters", "default_value")?,
                default_factor_id: u32c("model_parameters", "default_factor_id")?,
                runtime_name_string_id: u32c("model_parameters", "runtime_name_string_id")?,
                complex_component: i32c("model_parameters", "complex_component")?,
                derived: u8c("model_parameters", "derived")?,
            },
            contraction_coefficients: EagerContractionCoefficientColumns {
                contraction_ir_id: u32c("contraction_coefficients", "contraction_ir_id")?,
                component_index: u32c("contraction_coefficients", "component_index")?,
                factor_id: u32c("contraction_coefficients", "factor_id")?,
            },
            coherent_groups: EagerCoherentGroupColumns {
                id: u32c("coherent_groups", "id")?,
                helicity_weight_factor_id: u32c("coherent_groups", "helicity_weight_factor_id")?,
                all_sector_weight_factor_id: u32c(
                    "coherent_groups",
                    "all_sector_weight_factor_id",
                )?,
            },
            helicity_selectors: EagerHelicitySelectorColumns {
                values_sequence_id: u32c("helicity_selectors", "values_sequence_id")?,
                representative_sequence_id: u32c(
                    "helicity_selectors",
                    "representative_sequence_id",
                )?,
                coefficient_factor_id: u32c("helicity_selectors", "coefficient_factor_id")?,
                computed: u8c("helicity_selectors", "computed")?,
                structural_zero: u8c("helicity_selectors", "structural_zero")?,
            },
            color_selectors: EagerColorSelectorColumns {
                word_sequence_id: u32c("color_selectors", "word_sequence_id")?,
                representative_word_sequence_id: u32c(
                    "color_selectors",
                    "representative_word_sequence_id",
                )?,
                coefficient_factor_id: u32c("color_selectors", "coefficient_factor_id")?,
                computed: u8c("color_selectors", "computed")?,
            },
            reduction_members: EagerReductionMemberColumns {
                coherent_group_id: u32c("reduction_members", "coherent_group_id")?,
                helicity_selector_id: u32c("reduction_members", "helicity_selector_id")?,
                color_selector_id: u32c("reduction_members", "color_selector_id")?,
            },
            color_contraction: EagerColorContractionMetadata {
                present: u8c("color_contraction_metadata", "present")?[0],
                supported: u8c("color_contraction_metadata", "supported")?[0],
                group_count: u64c("color_contraction_metadata", "group_count")?[0],
                includes_color_factor: u8c("color_contraction_metadata", "includes_color_factor")?
                    [0],
            },
            color_contraction_entries: EagerColorContractionEntryColumns {
                left_group_id: u32c("color_contraction_entries", "left_group_id")?,
                right_group_id: u32c("color_contraction_entries", "right_group_id")?,
                weight_factor_id: u32c("color_contraction_entries", "weight_factor_id")?,
                symmetry_factor_id: u32c("color_contraction_entries", "symmetry_factor_id")?,
            },
        })
    }
}

#[pyfunction]
pub(crate) fn _lower_eager_runtime_v1(
    py: Python<'_>,
    lowering_input: &Bound<'_, PyAny>,
    destination: PathBuf,
) -> PyResult<Py<PyAny>> {
    let borrowed = parse_input(lowering_input)?;
    let owned = borrowed.into_core_input().map_err(python_error)?;
    let input_digest = borrowed.digest.clone();
    let native = py
        .detach(move || {
            let plan = lower_eager_plan_v3(owned)?;
            let selector_work = plan.selector_work_summary()?;
            let result = NativeLoweringResult {
                process_key: plan.process_key().to_owned(),
                model_name: plan.model_name().to_owned(),
                current_count: plan.currents().len(),
                value_count: plan.values().len(),
                momentum_count: plan.momenta().len(),
                source_count: plan.sources().len(),
                parameter_count: plan.parameters().len(),
                stage_count: plan.stages().len(),
                coupling_count: plan.couplings().len(),
                invocation_count: plan.invocations().len(),
                attachment_count: plan.attachments().len(),
                finalization_count: plan.finalizations().len(),
                closure_count: plan.closures().len(),
                selector_domain_count: plan.selector_domains().len(),
                reduction_group_count: plan.reduction_groups().len(),
                reduction_entry_count: plan.reduction_entries().len(),
                retained_table_count: plan.retained_tables().len(),
                current_component_count: plan.current_component_count(),
                value_component_count: plan.value_component_count(),
                momentum_component_count: plan.momentum_component_count(),
                selected_flow_selector_count: selector_work.selected_flow_selector_count,
                selected_flow_current_count: selector_work.selected_flow_worst.current_count,
                selected_flow_evaluation_count: selector_work.selected_flow_worst.evaluation_count,
                selected_flow_attachment_count: selector_work.selected_flow_worst.attachment_count,
                all_flow_selector_count: selector_work.all_flow_selector_count,
                all_flow_current_count: selector_work.all_flow_worst.current_count,
                all_flow_evaluation_count: selector_work.all_flow_worst.evaluation_count,
                all_flow_attachment_count: selector_work.all_flow_worst.attachment_count,
                contracted_current_count: selector_work.contracted.current_count,
                contracted_evaluation_count: selector_work.contracted.evaluation_count,
                contracted_attachment_count: selector_work.contracted.attachment_count,
                member_count: 0,
                unpacked_size_bytes: 0,
                index_sha256: String::new(),
            };
            let metadata = write_eager_plan_v3_pacbin(&plan, &destination)?;
            Ok::<_, RusticolError>(NativeLoweringResult {
                member_count: metadata.member_count,
                unpacked_size_bytes: metadata.unpacked_bytes,
                index_sha256: hex_digest(metadata.index_sha256),
                ..result
            })
        })
        .map_err(python_error)?;
    result_mapping(py, &input_digest, native)
}

fn result_mapping(
    py: Python<'_>,
    input_digest: &str,
    native: NativeLoweringResult,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("kind", RESULT_KIND)?;
    result.set_item("schema_version", RESULT_SCHEMA_VERSION)?;
    result.set_item("lowering_input_abi", EAGER_LOWERING_INPUT_ABI)?;
    result.set_item("lowering_input_sha256", input_digest)?;
    result.set_item("eager_plan_abi", EAGER_PLAN_ABI)?;
    result.set_item("runtime_layout_abi", EAGER_RUNTIME_LAYOUT_ABI)?;
    result.set_item(
        "required_runtime_capabilities",
        PyList::new(
            py,
            [
                EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
                EAGER_RUNTIME_CAPABILITY,
            ],
        )?,
    )?;
    let container = PyDict::new(py);
    container.set_item("kind", EAGER_RUNTIME_CONTAINER_KIND)?;
    container.set_item("schema_version", EAGER_RUNTIME_CONTAINER_SCHEMA)?;
    container.set_item("storage_abi", STORAGE_ABI)?;
    container.set_item("member_count", native.member_count)?;
    container.set_item("unpacked_size_bytes", native.unpacked_size_bytes)?;
    container.set_item("index_sha256", native.index_sha256)?;
    result.set_item("runtime_container", container)?;

    let inspection = PyDict::new(py);
    inspection.set_item("execution_mode", "eager")?;
    inspection.set_item("eager_plan_abi", EAGER_PLAN_ABI)?;
    inspection.set_item("runtime_layout_abi", EAGER_RUNTIME_LAYOUT_ABI)?;
    inspection.set_item("process_id", native.process_key)?;
    inspection.set_item("model_name", native.model_name)?;
    inspection.set_item("current_count", native.current_count)?;
    inspection.set_item("value_count", native.value_count)?;
    inspection.set_item("momentum_count", native.momentum_count)?;
    inspection.set_item("source_count", native.source_count)?;
    inspection.set_item("parameter_count", native.parameter_count)?;
    inspection.set_item("stage_count", native.stage_count)?;
    inspection.set_item("coupling_count", native.coupling_count)?;
    inspection.set_item("invocation_count", native.invocation_count)?;
    inspection.set_item("attachment_count", native.attachment_count)?;
    inspection.set_item("finalization_count", native.finalization_count)?;
    inspection.set_item("closure_count", native.closure_count)?;
    inspection.set_item("selector_domain_count", native.selector_domain_count)?;
    inspection.set_item("reduction_group_count", native.reduction_group_count)?;
    inspection.set_item("reduction_entry_count", native.reduction_entry_count)?;
    inspection.set_item("retained_table_count", native.retained_table_count)?;
    inspection.set_item("current_component_count", native.current_component_count)?;
    inspection.set_item("value_component_count", native.value_component_count)?;
    inspection.set_item("momentum_component_count", native.momentum_component_count)?;
    let selector_work = PyDict::new(py);
    selector_work.set_item("abi", SELECTOR_WORK_ABI)?;
    selector_work.set_item(
        "selected_flow_selector_count",
        native.selected_flow_selector_count,
    )?;
    selector_work.set_item(
        "selected_flow_current_count",
        native.selected_flow_current_count,
    )?;
    selector_work.set_item(
        "selected_flow_evaluation_count",
        native.selected_flow_evaluation_count,
    )?;
    selector_work.set_item(
        "selected_flow_attachment_count",
        native.selected_flow_attachment_count,
    )?;
    selector_work.set_item("all_flow_selector_count", native.all_flow_selector_count)?;
    selector_work.set_item("all_flow_current_count", native.all_flow_current_count)?;
    selector_work.set_item(
        "all_flow_evaluation_count",
        native.all_flow_evaluation_count,
    )?;
    selector_work.set_item(
        "all_flow_attachment_count",
        native.all_flow_attachment_count,
    )?;
    selector_work.set_item("contracted_current_count", native.contracted_current_count)?;
    selector_work.set_item(
        "contracted_evaluation_count",
        native.contracted_evaluation_count,
    )?;
    selector_work.set_item(
        "contracted_attachment_count",
        native.contracted_attachment_count,
    )?;
    inspection.set_item("selector_work", selector_work)?;
    result.set_item("inspection_summary", inspection)?;
    Ok(result.into_any().unbind())
}

fn parse_input<'py>(input: &Bound<'py, PyAny>) -> PyResult<BorrowedInput<'py>> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "eager lowering input v1 requires a little-endian target",
        ));
    }
    let abi = required_string(input, "abi")?;
    if abi != EAGER_LOWERING_INPUT_ABI {
        return Err(PyValueError::new_err(format!(
            "unsupported eager lowering input ABI {abi:?}"
        )));
    }
    let process_key = nonempty_string(input, "process_key")?;
    let model_name = nonempty_string(input, "model_name")?;
    let string_catalog = unique_strings(input, "string_catalog", false)?;
    let canonical_ir_catalog = unique_strings(input, "canonical_ir_catalog", false)?;
    let semantic_limitations = unique_strings(input, "semantic_limitations", true)?;

    let mut content_digest = Sha256::new();
    for value in [&abi, &process_key, &model_name] {
        hash_text(&mut content_digest, value);
    }
    for catalog in [&string_catalog, &canonical_ir_catalog] {
        hash_u64(&mut content_digest, catalog.len())?;
        for value in catalog {
            hash_text(&mut content_digest, value);
        }
    }

    let tables_object = input.getattr("tables")?;
    let table_objects = tables_object
        .try_iter()
        .map_err(|_| PyTypeError::new_err("eager lowering tables must be iterable"))?
        .collect::<PyResult<Vec<_>>>()?;
    hash_u64(&mut content_digest, table_objects.len())?;
    let mut schema_digest = Sha256::new();
    hash_u64(&mut schema_digest, table_objects.len())?;
    let mut tables = Vec::with_capacity(table_objects.len());
    let mut table_by_name = BTreeMap::new();
    let mut previous_table_name: Option<String> = None;
    for table_object in table_objects {
        let table_name = nonempty_string(&table_object, "name")?;
        if previous_table_name
            .as_deref()
            .is_some_and(|name| name >= table_name.as_str())
        {
            return Err(PyValueError::new_err(
                "eager lowering table names must be unique and sorted",
            ));
        }
        previous_table_name = Some(table_name.clone());
        let row_count = table_object.getattr("row_count")?.extract::<u64>()?;
        hash_text(&mut content_digest, &table_name);
        content_digest.update(row_count.to_le_bytes());
        hash_text(&mut schema_digest, &table_name);

        let columns_object = table_object.getattr("columns")?;
        let column_objects = columns_object
            .try_iter()
            .map_err(|_| PyTypeError::new_err("eager lowering columns must be iterable"))?
            .collect::<PyResult<Vec<_>>>()?;
        hash_u32(&mut content_digest, column_objects.len())?;
        hash_u32(&mut schema_digest, column_objects.len())?;
        let mut columns = Vec::with_capacity(column_objects.len());
        let mut column_by_name = BTreeMap::new();
        for column_object in column_objects {
            let column_name = nonempty_string(&column_object, "name")?;
            if column_by_name.contains_key(&column_name) {
                return Err(PyValueError::new_err(format!(
                    "table {table_name:?} contains duplicate column {column_name:?}"
                )));
            }
            let values_object = column_object.getattr("values")?;
            let dtype = values_object
                .getattr("dtype")?
                .getattr("str")?
                .extract::<String>()?;
            let writeable = values_object
                .getattr("flags")?
                .getattr("writeable")?
                .extract::<bool>()?;
            if writeable {
                return Err(PyValueError::new_err(format!(
                    "{table_name}.{column_name} must be read-only"
                )));
            }
            let values = extract_values(&values_object, &dtype, &table_name, &column_name)?;
            let shape = shape(&values);
            if shape.is_empty() || u64::try_from(shape[0]).ok() != Some(row_count) {
                return Err(PyValueError::new_err(format!(
                    "{table_name}.{column_name} shape does not match row_count"
                )));
            }
            hash_text(&mut content_digest, &column_name);
            hash_text(&mut content_digest, &dtype);
            hash_u8(&mut content_digest, shape.len())?;
            for dimension in &shape {
                hash_u64(&mut content_digest, *dimension)?;
            }
            content_digest.update(values.raw_bytes()?);
            hash_text(&mut schema_digest, &column_name);
            hash_text(&mut schema_digest, &dtype);
            hash_u8(&mut schema_digest, shape.len())?;
            for dimension in shape.iter().skip(1) {
                hash_u64(&mut schema_digest, *dimension)?;
            }
            column_by_name.insert(column_name.clone(), columns.len());
            columns.push(BorrowedColumn {
                name: column_name,
                shape,
                values,
            });
        }
        table_by_name.insert(table_name.clone(), tables.len());
        tables.push(BorrowedTable {
            name: table_name,
            row_count,
            columns,
            column_by_name,
        });
    }
    let names = tables
        .iter()
        .map(|table| table.name.as_str())
        .collect::<Vec<_>>();
    if names != EAGER_LOWERING_V1_TABLE_NAMES {
        return Err(PyValueError::new_err(
            "eager lowering table inventory does not match input ABI v1",
        ));
    }
    let schema_digest = hex_digest(schema_digest.finalize());
    if schema_digest != INPUT_SCHEMA_SHA256 {
        return Err(PyValueError::new_err(format!(
            "eager lowering table/column schema does not match input ABI v1: {schema_digest}"
        )));
    }
    for limitation in &semantic_limitations {
        hash_text(&mut content_digest, limitation);
    }
    let digest = hex_digest(content_digest.finalize());
    let declared_digest = required_string(input, "digest")?;
    if declared_digest != digest {
        return Err(PyValueError::new_err(
            "eager lowering input digest does not match its primitive columns",
        ));
    }
    Ok(BorrowedInput {
        abi,
        process_key,
        model_name,
        string_catalog,
        canonical_ir_catalog,
        semantic_limitations,
        tables,
        table_by_name,
        digest,
    })
}

fn extract_values<'py>(
    value: &Bound<'py, PyAny>,
    dtype: &str,
    table: &str,
    column: &str,
) -> PyResult<BorrowedValues<'py>> {
    let context = format!("{table}.{column}");
    macro_rules! extract {
        ($ty:ty, $variant:ident) => {{
            let array = value
                .extract::<PyReadonlyArrayDyn<'py, $ty>>()
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{context} is not a NumPy array with dtype {dtype}"
                    ))
                })?;
            if !array.is_c_contiguous() {
                return Err(PyValueError::new_err(format!(
                    "{context} must be C-contiguous"
                )));
            }
            BorrowedValues::$variant(array)
        }};
    }
    match dtype {
        "|u1" => Ok(extract!(u8, U8)),
        "<u4" => Ok(extract!(u32, U32)),
        "<u8" => Ok(extract!(u64, U64)),
        "<i4" => Ok(extract!(i32, I32)),
        "<f8" => Ok(extract!(f64, F64)),
        _ => Err(PyValueError::new_err(format!(
            "{context} has unsupported dtype {dtype:?}"
        ))),
    }
}

fn shape(values: &BorrowedValues<'_>) -> Vec<usize> {
    match values {
        BorrowedValues::U8(array) => array.shape().to_vec(),
        BorrowedValues::U32(array) => array.shape().to_vec(),
        BorrowedValues::U64(array) => array.shape().to_vec(),
        BorrowedValues::I32(array) => array.shape().to_vec(),
        BorrowedValues::F64(array) => array.shape().to_vec(),
    }
}

fn required_string(value: &Bound<'_, PyAny>, attribute: &str) -> PyResult<String> {
    value
        .getattr(attribute)?
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err(format!("eager lowering {attribute} must be a string")))
}

fn nonempty_string(value: &Bound<'_, PyAny>, attribute: &str) -> PyResult<String> {
    let result = required_string(value, attribute)?;
    if result.is_empty() {
        return Err(PyValueError::new_err(format!(
            "eager lowering {attribute} must not be empty"
        )));
    }
    Ok(result)
}

fn unique_strings(
    value: &Bound<'_, PyAny>,
    attribute: &str,
    require_nonempty: bool,
) -> PyResult<Vec<String>> {
    let object = value.getattr(attribute)?;
    let result = object
        .try_iter()
        .map_err(|_| PyTypeError::new_err(format!("{attribute} must be iterable")))?
        .map(|item| item?.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    if require_nonempty && result.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{attribute} must not be empty"
        )));
    }
    if result.iter().any(String::is_empty) {
        return Err(PyValueError::new_err(format!(
            "{attribute} entries must not be empty"
        )));
    }
    let unique = result.iter().collect::<BTreeSet<_>>();
    if unique.len() != result.len() {
        return Err(PyValueError::new_err(format!(
            "{attribute} contains duplicate strings"
        )));
    }
    Ok(result)
}

fn is_consumed_column(table: &str, column: &str) -> bool {
    matches!(
        (table, column),
        (
            "currents",
            "id" | "dimension"
                | "is_source"
                | "momentum_mask_bitset_id"
                | "propagator_ir_id"
                | "propagator_kernel_id"
        ) | (
            "sources",
            "source_id"
                | "current_id"
                | "external_label"
                | "input_momentum_slot"
                | "source_ir_id"
                | "crossing_ir_id"
                | "crossing_factor_id"
                | "declared_state_index"
        ) | (
            "interactions",
            "id" | "stage_subset_size"
                | "left_current_id"
                | "right_current_id"
                | "result_current_id"
                | "coupling_id"
                | "coupling_factor_id"
                | "color_factor_id"
                | "evaluation_factor_id"
                | "evaluation_group_id"
                | "full_tensor_network_ready"
                | "kernel_id"
                | "canonical_input_order"
                | "kernel_normalization_factor_id"
                | "output_factor_source"
        ) | (
            "roots",
            "id" | "left_current_id"
                | "right_current_id"
                | "color_factor_id"
                | "contraction_ir_id"
                | "coupling_id"
                | "coupling_factor_id"
                | "helicity_weight_factor_id"
                | "coherent_group_id"
                | "kernel_id"
                | "canonical_input_order"
                | "kernel_normalization_factor_id"
                | "output_factor_source"
        ) | (
            "interaction_groups",
            "representative_interaction_id" | "member_start" | "member_count"
        ) | ("interaction_group_members", "interaction_id")
            | ("momentum_masks", "slot_id" | "bitset_id")
            | ("bitset_ranges", "start" | "count" | "bit_count")
            | ("bitset_words", "value")
            | ("u32_sequence_ranges", "start" | "count")
            | ("u32_sequence_values", "value")
            | ("i32_sequence_ranges", "start" | "count")
            | ("i32_sequence_values", "value")
            | (
                "exact_factors",
                "real"
                    | "imaginary"
                    | "canonical_string_id"
                    | "exact_source"
                    | "exact_ir_id"
                    | "source_ir_id"
            )
            | (
                "couplings",
                "constant_factor_id" | "parameter_name_ids_sequence_id"
            )
            | (
                "model_parameters",
                "name_string_id"
                    | "kind_string_id"
                    | "default_value"
                    | "default_factor_id"
                    | "runtime_name_string_id"
                    | "complex_component"
                    | "derived"
            )
            | (
                "contraction_coefficients",
                "contraction_ir_id" | "component_index" | "factor_id"
            )
            | (
                "coherent_groups",
                "id" | "helicity_weight_factor_id" | "all_sector_weight_factor_id"
            )
            | (
                "helicity_selectors",
                "values_sequence_id"
                    | "representative_sequence_id"
                    | "coefficient_factor_id"
                    | "computed"
                    | "structural_zero"
            )
            | (
                "color_selectors",
                "word_sequence_id"
                    | "representative_word_sequence_id"
                    | "coefficient_factor_id"
                    | "computed"
            )
            | (
                "reduction_members",
                "coherent_group_id" | "helicity_selector_id" | "color_selector_id"
            )
            | (
                "color_contraction_metadata",
                "present" | "supported" | "group_count" | "includes_color_factor"
            )
            | (
                "color_contraction_entries",
                "left_group_id" | "right_group_id" | "weight_factor_id" | "symmetry_factor_id"
            )
    )
}

fn hash_text(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn hash_u8(digest: &mut Sha256, value: usize) -> PyResult<()> {
    digest
        .update([u8::try_from(value)
            .map_err(|_| PyValueError::new_err("eager lowering rank exceeds u8"))?]);
    Ok(())
}

fn hash_u32(digest: &mut Sha256, value: usize) -> PyResult<()> {
    digest.update(
        u32::try_from(value)
            .map_err(|_| PyValueError::new_err("eager lowering count exceeds u32"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_u64(digest: &mut Sha256, value: usize) -> PyResult<()> {
    digest.update(
        u64::try_from(value)
            .map_err(|_| PyValueError::new_err("eager lowering count exceeds u64"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn hex_digest(value: impl AsRef<[u8]>) -> String {
    value
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn not_contiguous(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("NumPy column is not contiguous: {error}"))
}

fn core_not_contiguous(error: impl std::fmt::Display) -> RusticolError {
    RusticolError::invalid_argument(format!("NumPy column is not contiguous: {error}"))
}

fn wrong_type(context: &str, dtype: &str) -> RusticolError {
    RusticolError::invalid_argument(format!("{context} is not a {dtype} column"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_v1_table_is_classified_for_retention() {
        assert_eq!(EAGER_LOWERING_V1_TABLE_NAMES.len(), 64);
        assert!(
            EAGER_LOWERING_V1_TABLE_NAMES
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        for table in [
            "color_replay_contraction_component_groups",
            "color_replay_contraction_entries",
            "color_replay_contraction_metadata",
            "color_replay_mappings",
            "color_replay_metadata",
            "color_replay_permutations",
            "color_replay_physical_groups",
            "color_replay_routes",
        ] {
            assert!(EAGER_LOWERING_V1_TABLE_NAMES.contains(&table));
        }
    }

    #[test]
    fn consumed_columns_are_narrow_and_explicit() {
        assert!(is_consumed_column("currents", "id"));
        assert!(is_consumed_column("interactions", "canonical_input_order"));
        assert!(!is_consumed_column("currents", "particle_id"));
        assert!(!is_consumed_column("metadata", "process_ir_id"));
        assert!(!is_consumed_column(
            "helicity_source_mappings",
            "source_contract_digest_string_id"
        ));
        assert!(!is_consumed_column(
            "helicity_materialized_schedules",
            "active_current_sequence_id"
        ));
        assert!(!is_consumed_column(
            "lc_replay_partitions",
            "proof_digest_string_id"
        ));
        assert!(!is_consumed_column(
            "color_replay_metadata",
            "physical_group_count"
        ));
        assert!(!is_consumed_column(
            "color_replay_routes",
            "source_group_id"
        ));
    }
}
