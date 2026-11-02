// SPDX-License-Identifier: 0BSD

use super::evaluator::recurrence_intrinsic_direct::{
    FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE, RecurrenceContributionIntrinsicKind,
    WEYL_PROPAGATOR_NEGATIVE_TEMPLATE, WEYL_PROPAGATOR_POSITIVE_TEMPLATE,
};
use super::*;
use crate::{
    EAGER_HOMOGENEOUS_LINEAR_CURRENT_PROOF, EAGER_INDEPENDENT_BLOCK_SIZE, EAGER_KERNEL_ABI,
    EAGER_PLAN_ABI, EAGER_SELECTOR_DOMAINS_ABI, EagerDirectClosureSpec, EagerKernelInput,
    EagerKernelRole, EagerKernelSpec, EagerPlanDefinition, EagerPlanDimensions,
    EagerReductionEntry, EagerReductionGroup,
};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

pub(super) const EAGER_EXECUTION_KIND: &str = "pyamplicol-runtime-eager-execution";
const PREPARED_KERNEL_VARIANT_ABI: &str = "pyamplicol-prepared-kernel-variant-v2";
const PREPARED_INDEPENDENT_BLOCK_VARIANT_ID: &str = "independent-block-4";
const PREPARED_INDEPENDENT_BLOCK_PROOF: &str = "prepared-kernel-independent-current-block-v1";
const SYMJIT_APPLICATION_STORAGE_V3_ABI: &str = "symjit-application-storage-v3";
const PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL: u64 = 2;
const PREPARED_JIT_PORTABLE_TARGET: &str = "symjit-storage-v3-portable";
const RECURRENCE_DIRECT_TEMPLATE_ABI_V1: &str = "pyamplicol-recurrence-direct-template-v1";
const RECURRENCE_DIRECT_BACKEND_ABI_V1: &str = "rusticol.recurrence-direct-backend.v1";
const RECURRENCE_DIRECT_CANONICALIZATION_ABI_V1: &str = "pyamplicol-canonical-json-v1";
const RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI_V2: &str = "pyamplicol-recurrence-plane-binding-v2";
const SYMJIT_PLANE_APPLICATION_V2_ABI: &str = "pyamplicol-symjit-plane-application-v2";
const NATIVE_DIRECT_APPLICATION_V1_ABI: &str = "pyamplicol-recurrence-native-direct-library-v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    #[serde(default)]
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) eager_plan_abi: String,
    pub(super) kernel_pack: EagerKernelPackReferenceManifest,
    pub(super) runtime_options: EagerRuntimeOptionsManifest,
    #[serde(default)]
    pub(super) lc_topology_replay: Option<LcTopologyReplayManifest>,
    pub(super) plan: EagerPlanManifest,
    pub(super) dag_summary: ExecutionSummary,
    pub(super) runtime_schema: ExecutionPlan,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerKernelPackReferenceManifest {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerRuntimeOptionsManifest {
    pub(super) point_tile_size: usize,
    pub(super) workspace_mib: usize,
}

impl EagerRuntimeOptionsManifest {
    pub(super) fn validate(self) -> RusticolResult<crate::EagerRuntimeOptions> {
        if self.point_tile_size == 0 {
            return Err(RusticolError::artifact(
                "eager point_tile_size must be positive",
            ));
        }
        if self.workspace_mib == 0 {
            return Err(RusticolError::artifact(
                "eager workspace_mib must be positive",
            ));
        }
        crate::EagerRuntimeOptions::from_mib(self.point_tile_size, self.workspace_mib)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerPlanManifest {
    pub(super) kind: String,
    pub(super) eager_plan_abi: String,
    #[serde(default)]
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process_key: String,
    pub(super) couplings: EagerTableManifest,
    pub(super) stages: Vec<EagerStageTablesManifest>,
    pub(super) closures: EagerTableManifest,
    #[serde(default)]
    pub(super) selector_closures: Option<EagerSelectorDomainsManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerStageTablesManifest {
    pub(super) stage_index: u32,
    pub(super) subset_size: usize,
    pub(super) invocations: EagerTableManifest,
    pub(super) attachments: EagerTableManifest,
    pub(super) finalizations: EagerTableManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerSelectorDomainsManifest {
    pub(super) abi: String,
    pub(super) domains: EagerTableManifest,
    pub(super) domain_group_ids: EagerTableManifest,
    pub(super) stages: Vec<EagerSelectorStageManifest>,
    pub(super) closure_domains: EagerTableManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerSelectorStageManifest {
    pub(super) stage_index: u32,
    pub(super) invocation_domains: EagerTableManifest,
    pub(super) attachment_domains: EagerTableManifest,
    pub(super) unpropagated_finalization_domains: EagerTableManifest,
    pub(super) propagated_finalization_domains: EagerTableManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerTableManifest {
    pub(super) path: String,
    pub(super) count: usize,
    pub(super) row_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelPackManifest {
    pub(super) eager_kernel_abi: String,
    pub(super) backend: String,
    pub(super) optimization_settings: Value,
    pub(super) producer: Value,
    pub(super) dependency_abis: Value,
    pub(super) provenance: Value,
    pub(super) target: PreparedKernelTargetManifest,
    pub(super) resolver_manifest: Value,
    pub(super) kernels: Vec<PreparedKernelManifest>,
    #[serde(default)]
    pub(super) kernel_variants: Vec<PreparedKernelVariantManifest>,
    #[serde(default)]
    pub(super) recurrence_template: Option<Box<RawValue>>,
    #[serde(default)]
    pub(super) recurrence_direct_template: Option<Box<RawValue>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelTargetManifest {
    pub(super) portable: bool,
    pub(super) word_bits: u8,
    pub(super) endianness: String,
    pub(super) target_triple: String,
    #[serde(default)]
    pub(super) cpu_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelManifest {
    pub(super) kernel_id: u32,
    pub(super) contract_kind: String,
    pub(super) canonical_signature: String,
    pub(super) input_arity: usize,
    pub(super) output_arity: u32,
    pub(super) input_layout: Vec<String>,
    pub(super) input_contracts: Vec<PreparedKernelInputManifest>,
    pub(super) output_layout: Vec<String>,
    pub(super) exact_expressions: Vec<String>,
    #[serde(default)]
    pub(super) proof_classes: Vec<String>,
    pub(super) exact_evaluator_state_path: String,
    pub(super) f64_evaluator_manifest: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerDirectTableManifest {
    pub(super) capability: String,
    pub(super) source_application_abi: String,
    pub(super) descriptor_abi: String,
    pub(super) binding_abi: String,
    #[serde(default)]
    pub(super) descriptor_path: Option<String>,
    #[serde(default)]
    pub(super) descriptor_size_bytes: Option<u64>,
    #[serde(default)]
    pub(super) descriptor_sha256: Option<String>,
    #[serde(default)]
    pub(super) library_path: Option<String>,
    #[serde(default)]
    pub(super) function_name: Option<String>,
    #[serde(default)]
    pub(super) evaluator_state_sha256: Option<String>,
    #[serde(default)]
    pub(super) invocation_stride: Option<u32>,
    #[serde(default)]
    pub(super) attachment_stride: Option<u32>,
    #[serde(default)]
    pub(super) simd_lane_width: Option<u32>,
    #[serde(default)]
    pub(super) instruction_count: Option<u32>,
    #[serde(default)]
    pub(super) temporary_count: Option<u32>,
    pub(super) input_complex_count: u32,
    pub(super) output_complex_count: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelVariantManifest {
    pub(super) variant_id: String,
    pub(super) variant_abi: String,
    pub(super) kind: String,
    pub(super) block_size: u32,
    pub(super) lane_layout: String,
    pub(super) base_kernel_id: u32,
    pub(super) base_canonical_signature: String,
    pub(super) base_expression_digest: String,
    pub(super) base_input_contract_digest: String,
    pub(super) base_output_contract_digest: String,
    pub(super) backend: String,
    pub(super) optimization_settings_digest: String,
    pub(super) input_arity: usize,
    pub(super) output_arity: usize,
    pub(super) input_lane_stride: usize,
    pub(super) output_lane_stride: usize,
    pub(super) input_layout: Vec<String>,
    pub(super) output_layout: Vec<String>,
    pub(super) f64_evaluator_manifest: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelInputManifest {
    pub(super) role: String,
    pub(super) component: u32,
    pub(super) symbol: String,
    pub(super) model_parameter_name: Option<String>,
    pub(super) model_parameter_index: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceDirectTemplateCatalogManifest {
    pub(super) abi: String,
    pub(super) backend_abi: String,
    pub(super) canonicalization_abi: String,
    pub(super) backend: String,
    pub(super) target_triple: String,
    pub(super) portable: bool,
    pub(super) optimization_level: u32,
    pub(super) compiled_model_digest: String,
    pub(super) recurrence_template_catalog_digest: String,
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) prepared_kernel_contract_digest: String,
    pub(super) prepared_kernel_payload_digest: String,
    pub(super) optimization_settings_digest: String,
    pub(super) templates: Vec<RecurrenceDirectTemplateManifest>,
    pub(super) catalog_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceDirectTemplateManifest {
    pub(super) abi: String,
    pub(super) template_id: String,
    pub(super) direct_executor_id: u32,
    pub(super) evaluator_binding_id: u32,
    pub(super) evaluator_resolver_key: String,
    pub(super) role: String,
    pub(super) parent_arity: u32,
    pub(super) parent_component_counts: Vec<u32>,
    pub(super) destination_component_count: u32,
    pub(super) momentum_operand_count: u32,
    pub(super) destination_operation: String,
    pub(super) coupling_slot_count: u32,
    pub(super) parameter_slot_count: u32,
    pub(super) semantic_template_ids: Vec<String>,
    pub(super) exact_expression_digest: String,
    pub(super) payload_binding: RecurrenceDirectPayloadBindingManifest,
    pub(super) backend: String,
    pub(super) target_triple: String,
    pub(super) portable: bool,
    pub(super) optimization_level: u32,
    pub(super) alignment_bytes: u32,
    pub(super) simd_axis: String,
    pub(super) destination_aliasing: bool,
    pub(super) semantic_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceDirectPayloadBindingManifest {
    pub(super) abi: String,
    pub(super) kind: String,
    pub(super) payload_digest: String,
    pub(super) prepared_kernel_id: Option<u32>,
    pub(super) runtime_template: Option<String>,
    pub(super) payload_paths: Vec<String>,
    pub(super) source_application_path: Option<String>,
    pub(super) source_application_sha256: Option<String>,
    pub(super) source_application_abi: Option<String>,
    pub(super) direct_application_abi: Option<String>,
    pub(super) native_entry_point: Option<String>,
    pub(super) role: Option<String>,
    pub(super) destination_operation: Option<String>,
    pub(super) exact_factor_scalar_slots: Vec<u32>,
    pub(super) state_plane_indices: Vec<u32>,
    pub(super) parameter_bindings: Vec<RecurrenceDirectParameterBindingManifest>,
    pub(super) input_plane_count: u32,
    pub(super) scalar_input_count: u32,
    pub(super) output_alias_inputs: Vec<u32>,
    pub(super) contribution_parent_permutation: Vec<u8>,
    pub(super) input_plane_projections: Vec<RecurrenceDirectPlaneProjectionManifest>,
    pub(super) scalar_projections: Vec<RecurrenceDirectScalarProjectionManifest>,
    pub(super) intrinsic_contract_digest: Option<String>,
    pub(super) prepared_template_semantic_digest: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum RecurrenceDirectParameterBindingManifest {
    Plane { index: u32 },
    Scalar { index: u32 },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum RecurrenceDirectPlaneProjectionManifest {
    ParentCurrent {
        parent: u8,
        component: u16,
        imaginary: bool,
    },
    Momentum {
        operand: u8,
        lorentz_component: u16,
    },
    DestinationCurrent {
        component: u16,
        imaginary: bool,
    },
    DestinationAmplitude {
        component: u16,
        imaginary: bool,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum RecurrenceDirectScalarProjectionManifest {
    ExactFactor {
        imaginary: bool,
    },
    Parameter {
        index: u32,
        imaginary: bool,
    },
    Literal {
        value: f64,
    },
    #[serde(rename = "intrinsic-scale-v1")]
    IntrinsicScale {
        constant_real_bits: u64,
        constant_imag_bits: u64,
        parameter_index: Option<u32>,
    },
}

impl EagerExecutionManifest {
    pub(super) fn validate_header(&self) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != EAGER_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.eager_plan_abi != EAGER_PLAN_ABI
            || self.plan.eager_plan_abi != EAGER_PLAN_ABI
            || self.plan.kind != EAGER_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager plan ABI {:?}",
                self.eager_plan_abi
            )));
        }
        if self.plan.process_key != self.key {
            return Err(RusticolError::integrity(
                "eager plan process key does not match its execution manifest",
            ));
        }
        if let Some(selector) = &self.plan.selector_closures {
            if selector.abi != EAGER_SELECTOR_DOMAINS_ABI {
                return Err(RusticolError::compatibility(format!(
                    "unsupported eager selector-domain ABI {:?}",
                    selector.abi
                )));
            }
            if selector.stages.len() != self.plan.stages.len() {
                return Err(RusticolError::integrity(
                    "eager selector domains do not cover every execution stage",
                ));
            }
            for (selector_stage, execution_stage) in selector.stages.iter().zip(&self.plan.stages) {
                if selector_stage.stage_index != execution_stage.stage_index {
                    return Err(RusticolError::integrity(
                        "eager selector-domain stage index mismatch",
                    ));
                }
            }
        }
        validate_capability_list_match(
            &self.required_runtime_capabilities,
            &self.plan.required_runtime_capabilities,
            "eager execution and plan",
        )?;
        let expected_capabilities = if self.lc_topology_replay.is_some() {
            vec![
                EAGER_DAG_RUNTIME_CAPABILITY.to_string(),
                EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY.to_string(),
            ]
        } else {
            vec![EAGER_DAG_RUNTIME_CAPABILITY.to_string()]
        };
        if self.required_runtime_capabilities != expected_capabilities {
            return Err(RusticolError::compatibility(
                "eager execution runtime capabilities do not match its topology-replay contract",
            ));
        }
        self.runtime_options.validate()?;
        Ok(())
    }

    pub(super) fn compiled_metadata_manifest(&self) -> ExecutionManifest {
        let materialization_census =
            ExecutionMaterializationCensus::from_summary(&self.dag_summary);
        ExecutionManifest {
            schema_version: self.schema_version,
            kind: "pyamplicol-runtime-execution".to_string(),
            required_runtime_capabilities: Vec::new(),
            process: self.process.clone(),
            key: self.key.clone(),
            color_accuracy: self.color_accuracy.clone(),
            external_pdg_order: self.external_pdg_order.clone(),
            compiled: EvaluatorSetManifest {
                kind: "eager-runtime-metadata".to_string(),
                runtime_available: true,
                runtime_unavailable_message: None,
                lc_topology_replay: self.lc_topology_replay.clone(),
                color_topology_replay: None,
                model_parameter_evaluator: None,
                stage_evaluators: None,
                momentum_slot_ids: None,
            },
            dag_summary: self.dag_summary.clone(),
            materialization_census,
            runtime_schema: self.runtime_schema.clone(),
            color_contraction_payload: None,
            physics_reduction: None,
            helicity_sum_execution: None,
            helicity_selector_executions: Vec::new(),
            color_selector_executions: Vec::new(),
        }
    }

    pub(super) fn plan_definition(
        &self,
        pack: &PreparedKernelPackManifest,
        prepared_parameter_count: u32,
    ) -> RusticolResult<EagerPlanDefinition> {
        let dimensions = EagerPlanDimensions {
            value_slot_component_counts: contiguous_value_slot_widths(&self.runtime_schema)?,
            momentum_slot_component_counts: contiguous_momentum_slot_widths(&self.runtime_schema)?,
            current_component_counts: contiguous_current_slot_widths(&self.runtime_schema)?,
            parameter_count: prepared_parameter_count,
            amplitude_count: u32::try_from(self.runtime_schema.amplitude_stage.output_count)
                .map_err(|_| RusticolError::artifact("eager amplitude count exceeds u32"))?,
        };
        let kernels = pack.kernel_specs()?;
        let direct_closures = self.direct_closure_specs()?;
        let (reduction_groups, reduction_entries) = self.reduction_plan()?;
        Ok(EagerPlanDefinition {
            dimensions,
            kernels,
            direct_closures,
            reduction_groups,
            reduction_entries,
        })
    }

    fn direct_closure_specs(&self) -> RusticolResult<Vec<EagerDirectClosureSpec>> {
        self.runtime_schema
            .amplitude_stage
            .roots
            .iter()
            .enumerate()
            .filter(|(_, root)| root.kind == "direct-contraction")
            .map(|(index, root)| {
                let coefficients = root
                    .contraction_ir
                    .coefficients
                    .iter()
                    .map(|value| crate::EagerComplex64::new(value[0], value[1]))
                    .collect();
                Ok(EagerDirectClosureSpec {
                    closure_index: u32::try_from(index).map_err(|_| {
                        RusticolError::artifact("eager direct closure index exceeds u32")
                    })?,
                    coefficients,
                })
            })
            .collect()
    }

    fn reduction_plan(
        &self,
    ) -> RusticolResult<(Vec<EagerReductionGroup>, Vec<EagerReductionEntry>)> {
        let (groups, contraction) = self.raw_reduction_runtime()?;
        let reduction_groups = groups
            .iter()
            .map(|group| {
                Ok(EagerReductionGroup {
                    coherent_group_id: u32::try_from(group.id).map_err(|_| {
                        RusticolError::artifact("eager coherent reduction-group ID exceeds u32")
                    })?,
                    amplitude_indices: group
                        .indices
                        .iter()
                        .map(|index| {
                            u32::try_from(*index).map_err(|_| {
                                RusticolError::artifact(
                                    "eager reduction amplitude index exceeds u32",
                                )
                            })
                        })
                        .collect::<RusticolResult<Vec<_>>>()?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut entries = Vec::new();
        if let Some(contraction) = contraction {
            entries
                .try_reserve_exact(contraction.logical_entry_count()?)
                .map_err(|error| {
                    RusticolError::artifact(format!(
                        "cannot reserve eager color-contraction entries: {error}"
                    ))
                })?;
            for entry in contraction.logical_entries() {
                let coefficient = crate::EagerComplex64::new(entry.weight_re, entry.weight_im)
                    * entry.symmetry_factor;
                entries.push(EagerReductionEntry {
                    left_group_index: u32::try_from(entry.left_group_index).map_err(|_| {
                        RusticolError::artifact(
                            "eager color-contraction left group index exceeds u32",
                        )
                    })?,
                    right_group_index: u32::try_from(entry.right_group_index).map_err(|_| {
                        RusticolError::artifact(
                            "eager color-contraction right group index exceeds u32",
                        )
                    })?,
                    coefficient,
                });
            }
        } else {
            entries.try_reserve_exact(groups.len()).map_err(|error| {
                RusticolError::artifact(format!(
                    "cannot reserve eager diagonal reduction entries: {error}"
                ))
            })?;
            for (group_index, group) in groups.iter().enumerate() {
                let group_index = u32::try_from(group_index)
                    .map_err(|_| RusticolError::artifact("eager reduction group exceeds u32"))?;
                entries.push(EagerReductionEntry {
                    left_group_index: group_index,
                    right_group_index: group_index,
                    coefficient: crate::EagerComplex64::new(group.all_sector_weight, 0.0),
                });
            }
        }
        Ok((reduction_groups, entries))
    }

    pub(super) fn raw_reduction_runtime(
        &self,
    ) -> RusticolResult<(Vec<RawSumGroup>, Option<ColorContractionRuntime>)> {
        let roots = &self.runtime_schema.amplitude_stage.roots;
        let output_count = self.runtime_schema.amplitude_stage.output_count;
        let weights = roots
            .iter()
            .map(|root| root.helicity_weight)
            .collect::<Vec<_>>();
        let all_sector_weights = roots
            .iter()
            .map(|root| root.all_sector_weight.unwrap_or(root.helicity_weight))
            .collect::<Vec<_>>();
        let color_sector_ids = roots
            .iter()
            .map(|root| root.color_sector_id)
            .collect::<Vec<_>>();
        let group_ids = roots
            .iter()
            .map(generic_root_group_id)
            .collect::<RusticolResult<Vec<_>>>()?;
        let groups = build_raw_sum_groups(
            output_count,
            &weights,
            &all_sector_weights,
            &group_ids,
            &color_sector_ids,
        )?;
        let contraction = build_color_contraction_runtime(
            self.runtime_schema
                .amplitude_stage
                .color_contraction
                .as_ref(),
            &groups,
        )?;
        Ok((groups, contraction))
    }
}

impl PreparedKernelPackManifest {
    pub(super) fn validate(&self) -> RusticolResult<()> {
        if self.eager_kernel_abi != EAGER_KERNEL_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager kernel ABI {:?}",
                self.eager_kernel_abi
            )));
        }
        if !matches!(self.backend.as_str(), "jit" | "asm" | "cpp") {
            return Err(RusticolError::artifact(format!(
                "unsupported prepared backend {:?}",
                self.backend
            )));
        }
        for (name, value) in [
            ("optimization_settings", &self.optimization_settings),
            ("producer", &self.producer),
            ("dependency_abis", &self.dependency_abis),
            ("provenance", &self.provenance),
            ("resolver_manifest", &self.resolver_manifest),
        ] {
            if value.as_object().is_none_or(|mapping| mapping.is_empty()) {
                return Err(RusticolError::artifact(format!(
                    "prepared kernel pack {name} must be a nonempty object"
                )));
            }
        }
        if self.resolver_manifest.get("abi").and_then(Value::as_str)
            != Some("pyamplicol-prepared-kernel-catalog-v1")
        {
            return Err(RusticolError::compatibility(
                "prepared kernel resolver manifest has an unsupported ABI",
            ));
        }
        if self.kernels.is_empty() {
            return Err(RusticolError::artifact("prepared kernel pack is empty"));
        }
        if self.target.word_bits != 64 || self.target.endianness != "little" {
            return Err(RusticolError::compatibility(
                "prepared eager kernels require a 64-bit little-endian target",
            ));
        }
        if self
            .target
            .cpu_features
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(RusticolError::integrity(
                "prepared target CPU features must be sorted and unique",
            ));
        }
        if self.backend == "jit" {
            match std::env::consts::ARCH {
                "aarch64" | "x86_64" => {}
                other => {
                    return Err(RusticolError::compatibility(format!(
                        "prepared JIT kernels do not support host architecture {other:?}"
                    )));
                }
            }
            if self
                .dependency_abis
                .get("symjit_application")
                .and_then(Value::as_str)
                != Some(SYMJIT_APPLICATION_STORAGE_V3_ABI)
            {
                return Err(RusticolError::compatibility(
                    "prepared JIT kernels declare an unsupported SymJIT application ABI",
                ));
            }
            if self
                .dependency_abis
                .get("symjit_plane_application")
                .and_then(Value::as_str)
                != Some(SYMJIT_PLANE_APPLICATION_V2_ABI)
            {
                return Err(RusticolError::compatibility(
                    "prepared JIT kernels declare an unsupported SymJIT plane-application ABI; regenerate the prepared model",
                ));
            }
            if !self.target.portable
                || self.target.target_triple != PREPARED_JIT_PORTABLE_TARGET
                || !self.target.cpu_features.is_empty()
            {
                return Err(RusticolError::compatibility(format!(
                    "prepared JIT kernels target {:?}, expected portable target {:?}",
                    self.target.target_triple, PREPARED_JIT_PORTABLE_TARGET,
                )));
            }
            if self
                .optimization_settings
                .get("jit_optimization_level")
                .and_then(Value::as_u64)
                != Some(PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL)
            {
                return Err(RusticolError::compatibility(format!(
                    "prepared JIT kernels must use portable SymJIT optimization level {}",
                    PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
                )));
            }
        } else {
            let host = crate::runtime_target_info();
            let host_features = host.cpu_features.into_iter().collect::<BTreeSet<_>>();
            if self.target.portable
                || self.target.target_triple != host.triple
                || self
                    .target
                    .cpu_features
                    .iter()
                    .any(|feature| !host_features.contains(feature))
            {
                return Err(RusticolError::compatibility(format!(
                    "prepared {} kernels target {:?} with features {:?}, incompatible with host {:?}",
                    self.backend, self.target.target_triple, self.target.cpu_features, host.triple,
                )));
            }
        }
        let mut ids = BTreeSet::new();
        let mut signatures = BTreeSet::new();
        for kernel in &self.kernels {
            if !ids.insert(kernel.kernel_id) {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel pack repeats kernel id {}",
                    kernel.kernel_id
                )));
            }
            if kernel.canonical_signature.is_empty()
                || !signatures.insert(kernel.canonical_signature.as_str())
            {
                return Err(RusticolError::integrity(
                    "prepared kernel signatures must be nonempty and unique",
                ));
            }
            if kernel.input_arity != kernel.input_contracts.len()
                || kernel.input_arity != kernel.input_layout.len()
                || usize::try_from(kernel.output_arity).ok() != Some(kernel.output_layout.len())
                || usize::try_from(kernel.output_arity).ok() != Some(kernel.exact_expressions.len())
            {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} has inconsistent input/output arities",
                    kernel.kernel_id
                )));
            }
            if kernel.exact_evaluator_state_path.is_empty() {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} lacks exact evaluator state",
                    kernel.kernel_id
                )));
            }
            if kernel
                .proof_classes
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} proof classes must be sorted and unique",
                    kernel.kernel_id
                )));
            }
            if kernel
                .proof_classes
                .iter()
                .any(|proof| proof == EAGER_HOMOGENEOUS_LINEAR_CURRENT_PROOF)
                && kernel.contract_kind != "propagator"
            {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} applies a current-linearity proof to {:?}",
                    kernel.kernel_id, kernel.contract_kind
                )));
            }
            if kernel
                .proof_classes
                .iter()
                .any(|proof| proof == PREPARED_INDEPENDENT_BLOCK_PROOF)
                && (kernel.contract_kind != "vertex"
                    || kernel.input_contracts.iter().any(|input| {
                        !matches!(input.role.as_str(), "left-current" | "right-current")
                    }))
            {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} applies an independent-block proof to a non-current vertex",
                    kernel.kernel_id
                )));
            }
            kernel.validate_evaluator_metadata(self)?;
        }
        if self.backend != "jit" && !self.kernel_variants.is_empty() {
            return Err(RusticolError::integrity(
                "prepared C++/ASM packs cannot contain JIT block variants",
            ));
        }
        let kernels_by_id = self
            .kernels
            .iter()
            .map(|kernel| (kernel.kernel_id, kernel))
            .collect::<BTreeMap<_, _>>();
        let mut variant_keys = BTreeSet::new();
        let mut variant_bases = BTreeSet::new();
        for variant in &self.kernel_variants {
            if !variant_keys.insert((variant.base_kernel_id, variant.variant_id.as_str())) {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel pack repeats variant {:?} for kernel {}",
                    variant.variant_id, variant.base_kernel_id
                )));
            }
            if !variant_bases.insert(variant.base_kernel_id) {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} has more than one block variant",
                    variant.base_kernel_id
                )));
            }
            let base = kernels_by_id.get(&variant.base_kernel_id).ok_or_else(|| {
                RusticolError::integrity(format!(
                    "prepared variant {:?} references missing kernel {}",
                    variant.variant_id, variant.base_kernel_id
                ))
            })?;
            variant.validate(self, base)?;
        }
        Ok(())
    }

    /// Parse and authenticate the optional Direct-Arena recurrence companion.
    ///
    /// Eager execution deliberately does not require this companion. Recurrence
    /// loading calls this method with the digests authenticated by its process
    /// manifest and plan before resolving any executable payload.
    pub(super) fn recurrence_direct_template_catalog(
        &self,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<RecurrenceDirectTemplateCatalogManifest> {
        let raw = self.recurrence_direct_template.as_ref().ok_or_else(|| {
            RusticolError::compatibility(
                "prepared model has no Direct-Arena recurrence template catalog; recompile the model",
            )
        })?;
        self.parse_recurrence_direct_template_catalog(
            raw.get(),
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )
    }

    pub(super) fn take_recurrence_direct_template_catalog(
        &mut self,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<RecurrenceDirectTemplateCatalogManifest> {
        let raw = self.recurrence_direct_template.take().ok_or_else(|| {
            RusticolError::compatibility(
                "prepared model has no Direct-Arena recurrence template catalog; recompile the model",
            )
        })?;
        self.parse_recurrence_direct_template_catalog(
            raw.get(),
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )
    }

    fn parse_recurrence_direct_template_catalog(
        &self,
        raw: &str,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<RecurrenceDirectTemplateCatalogManifest> {
        let raw: Value = serde_json::from_str(raw).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse prepared Direct-Arena recurrence template catalog: {error}"
            ))
        })?;
        let catalog =
            RecurrenceDirectTemplateCatalogManifest::deserialize(&raw).map_err(|error| {
                RusticolError::serialization(format!(
                    "could not parse prepared Direct-Arena recurrence template catalog: {error}"
                ))
            })?;
        catalog.validate(
            &raw,
            self,
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )?;
        Ok(catalog)
    }

    pub(super) fn kernel_specs(&self) -> RusticolResult<Vec<EagerKernelSpec>> {
        let block_sizes = self
            .kernel_variants
            .iter()
            .map(|variant| (variant.base_kernel_id, variant.block_size))
            .collect::<BTreeMap<_, _>>();
        self.kernels
            .iter()
            .filter(|kernel| kernel.contract_kind != "model-parameter")
            .map(|kernel| {
                let role = match kernel.contract_kind.as_str() {
                    "vertex" => EagerKernelRole::Vertex,
                    "propagator" => EagerKernelRole::Finalization,
                    "closure" => EagerKernelRole::Closure,
                    other => {
                        return Err(RusticolError::artifact(format!(
                            "unsupported prepared kernel contract kind {other:?}"
                        )));
                    }
                };
                let inputs = kernel
                    .input_contracts
                    .iter()
                    .map(PreparedKernelInputManifest::to_eager_input)
                    .collect::<RusticolResult<Vec<_>>>()?;
                Ok(EagerKernelSpec {
                    kernel_id: kernel.kernel_id,
                    role,
                    inputs,
                    output_component_count: kernel.output_arity,
                    homogeneous_linear_first_current: kernel
                        .proof_classes
                        .iter()
                        .any(|proof| proof == EAGER_HOMOGENEOUS_LINEAR_CURRENT_PROOF),
                    independent_block_size: block_sizes
                        .get(&kernel.kernel_id)
                        .copied()
                        .unwrap_or(1),
                })
            })
            .collect()
    }

    pub(super) fn validate_portable_process_artifact_target(
        &self,
        outer_target: &crate::Target,
        context: &str,
    ) -> RusticolResult<()> {
        if outer_target.triple != crate::artifact::PORTABLE_64LE_ARTIFACT_TARGET {
            return Ok(());
        }
        let optimization_level = self
            .optimization_settings
            .get("jit_optimization_level")
            .and_then(Value::as_u64);
        if !outer_target.cpu_features.is_empty()
            || self.backend != "jit"
            || !self.target.portable
            || self.target.target_triple != PREPARED_JIT_PORTABLE_TARGET
            || !self.target.cpu_features.is_empty()
            || optimization_level != Some(PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL)
        {
            return Err(RusticolError::compatibility(format!(
                "portable-64le {context} artifacts require an authenticated portable O2 SymJIT prepared kernel pack; C++ and ASM packs remain target-specific"
            )));
        }
        Ok(())
    }
}

impl RecurrenceDirectTemplateCatalogManifest {
    fn validate(
        &self,
        raw: &Value,
        pack: &PreparedKernelPackManifest,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<()> {
        if self.abi != RECURRENCE_DIRECT_TEMPLATE_ABI_V1
            || self.backend_abi != RECURRENCE_DIRECT_BACKEND_ABI_V1
            || self.canonicalization_abi != RECURRENCE_DIRECT_CANONICALIZATION_ABI_V1
        {
            return Err(RusticolError::compatibility(
                "prepared Direct-Arena recurrence catalog has an unsupported ABI",
            ));
        }
        validate_sha256_text(&self.catalog_digest, "Direct-Arena catalog digest")?;
        validate_sha256_text(
            &self.prepared_kernel_pack_digest,
            "Direct-Arena prepared-pack digest",
        )?;
        for (name, digest) in [
            ("compiled-model", self.compiled_model_digest.as_str()),
            (
                "recurrence-template catalog",
                self.recurrence_template_catalog_digest.as_str(),
            ),
            (
                "prepared-kernel contract",
                self.prepared_kernel_contract_digest.as_str(),
            ),
            (
                "prepared-kernel payload",
                self.prepared_kernel_payload_digest.as_str(),
            ),
            (
                "optimization-settings",
                self.optimization_settings_digest.as_str(),
            ),
        ] {
            validate_sha256_text(digest, &format!("Direct-Arena {name} digest"))?;
        }
        if self.catalog_digest != expected_catalog_digest
            || self.prepared_kernel_pack_digest != expected_prepared_pack_digest
        {
            return Err(RusticolError::integrity(
                "prepared Direct-Arena catalog identity does not match the recurrence artifact",
            ));
        }
        if pack
            .provenance
            .get("prepared_kernel_pack_digest")
            .and_then(Value::as_str)
            != Some(expected_prepared_pack_digest)
            || pack
                .provenance
                .get("direct_template_catalog_digest")
                .and_then(Value::as_str)
                != Some(expected_catalog_digest)
        {
            return Err(RusticolError::integrity(
                "prepared kernel-pack provenance does not authenticate the Direct-Arena catalog",
            ));
        }
        if self.backend != pack.backend
            || self.target_triple != pack.target.target_triple
            || self.portable != pack.target.portable
        {
            return Err(RusticolError::integrity(
                "Direct-Arena catalog backend or target does not match its prepared kernel pack",
            ));
        }
        if self.backend == "jit" {
            if !self.portable
                || self.optimization_level != PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL as u32
                || self.target_triple != PREPARED_JIT_PORTABLE_TARGET
            {
                return Err(RusticolError::compatibility(
                    "Direct-Arena JIT templates require portable SymJIT O2 applications",
                ));
            }
        } else if self.portable {
            return Err(RusticolError::compatibility(
                "Direct-Arena C++/ASM templates must be target-native",
            ));
        }
        let mut semantic = raw.clone();
        semantic
            .as_object_mut()
            .ok_or_else(|| RusticolError::artifact("Direct-Arena catalog must be an object"))?
            .remove("catalog_digest");
        if canonical_json_digest_exact(&semantic)? != self.catalog_digest {
            return Err(RusticolError::integrity(
                "Direct-Arena catalog digest does not match its canonical metadata",
            ));
        }
        if self.templates.is_empty() {
            return Err(RusticolError::artifact(
                "Direct-Arena template catalog is empty",
            ));
        }
        let raw_templates = raw
            .get("templates")
            .and_then(Value::as_array)
            .ok_or_else(|| RusticolError::artifact("Direct-Arena templates must be an array"))?;
        if raw_templates.len() != self.templates.len() {
            return Err(RusticolError::integrity(
                "Direct-Arena template count changed while parsing",
            ));
        }
        let kernels = pack
            .kernels
            .iter()
            .map(|kernel| (kernel.kernel_id, kernel))
            .collect::<BTreeMap<_, _>>();
        let mut template_ids = BTreeSet::new();
        let mut binding_keys = BTreeSet::new();
        for (expected_id, (template, raw_template)) in
            self.templates.iter().zip(raw_templates).enumerate()
        {
            if template.direct_executor_id as usize != expected_id {
                return Err(RusticolError::integrity(
                    "Direct-Arena executor IDs must be dense and zero-based",
                ));
            }
            if !template_ids.insert(template.template_id.as_str())
                || !binding_keys.insert((template.role.as_str(), template.evaluator_binding_id))
            {
                return Err(RusticolError::integrity(
                    "Direct-Arena template IDs and semantic binding keys must be unique",
                ));
            }
            template.validate(raw_template, self, &kernels)?;
        }
        Ok(())
    }
}

impl RecurrenceDirectTemplateManifest {
    fn validate(
        &self,
        raw: &Value,
        catalog: &RecurrenceDirectTemplateCatalogManifest,
        kernels: &BTreeMap<u32, &PreparedKernelManifest>,
    ) -> RusticolResult<()> {
        if self.abi != RECURRENCE_DIRECT_TEMPLATE_ABI_V1
            || self.backend != catalog.backend
            || self.target_triple != catalog.target_triple
            || self.portable != catalog.portable
            || self.optimization_level != catalog.optimization_level
        {
            return Err(RusticolError::integrity(
                "Direct-Arena template policy does not match its catalog",
            ));
        }
        if self.template_id.is_empty()
            || self.evaluator_resolver_key.is_empty()
            || self.semantic_template_ids.is_empty()
            || self.parent_arity as usize != self.parent_component_counts.len()
            || self.parent_component_counts.contains(&0)
            || self.destination_component_count == 0
            || self.alignment_bytes == 0
            || !self.alignment_bytes.is_power_of_two()
            || self.simd_axis != "points-contiguous"
        {
            return Err(RusticolError::artifact(
                "Direct-Arena template has an invalid shape or SIMD contract",
            ));
        }
        if self
            .semantic_template_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(RusticolError::integrity(
                "Direct-Arena semantic template IDs must be sorted and unique",
            ));
        }
        validate_sha256_text(
            &self.exact_expression_digest,
            "Direct-Arena exact-expression digest",
        )?;
        validate_sha256_text(&self.semantic_digest, "Direct-Arena template digest")?;
        let expected_operation = expected_direct_operation(&self.role)?;
        if self.destination_operation != expected_operation
            || self.destination_aliasing != (self.role == "finalization")
        {
            return Err(RusticolError::integrity(
                "Direct-Arena destination operation or aliasing does not match its role",
            ));
        }
        self.payload_binding
            .validate(self, raw.get("payload_binding"), kernels)
    }
}

impl RecurrenceDirectPayloadBindingManifest {
    fn validate(
        &self,
        template: &RecurrenceDirectTemplateManifest,
        raw: Option<&Value>,
        kernels: &BTreeMap<u32, &PreparedKernelManifest>,
    ) -> RusticolResult<()> {
        if self.abi != RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI_V2 {
            return Err(RusticolError::compatibility(format!(
                "Direct-Arena payload binding ABI {:?} is unsupported; regenerate the prepared model with this pyAmpliCol version",
                self.abi
            )));
        }
        validate_sha256_text(&self.payload_digest, "Direct-Arena payload digest")?;
        if !matches!(
            self.contribution_parent_permutation.as_slice(),
            [0, 1] | [1, 0]
        ) || (self.contribution_parent_permutation.as_slice() != [0, 1]
            && !(self.kind == "rusticol-intrinsic" && template.role == "contribution"))
        {
            return Err(RusticolError::integrity(
                "Direct-Arena payload binding has an invalid parent permutation",
            ));
        }
        match self.kind.as_str() {
            "rusticol-intrinsic" => {
                let runtime_template = self.runtime_template.as_deref().ok_or_else(|| {
                    RusticolError::artifact("Direct-Arena intrinsic has no runtime template")
                })?;
                let prepared_metadata_present = self.prepared_kernel_id.is_some()
                    || !self.payload_paths.is_empty()
                    || self.source_application_path.is_some()
                    || self.source_application_sha256.is_some()
                    || self.source_application_abi.is_some()
                    || self.direct_application_abi.is_some()
                    || self.native_entry_point.is_some()
                    || !self.exact_factor_scalar_slots.is_empty()
                    || !self.state_plane_indices.is_empty()
                    || !self.parameter_bindings.is_empty()
                    || self.input_plane_count != 0
                    || !self.output_alias_inputs.is_empty()
                    || !self.input_plane_projections.is_empty()
                    || self.prepared_template_semantic_digest.is_some();
                if prepared_metadata_present {
                    return Err(RusticolError::integrity(
                        "Rusticol Direct-Arena intrinsics carry prepared-call metadata",
                    ));
                }
                if template.role == "contribution" {
                    if self.role.as_deref() != Some("contribution")
                        || self.destination_operation.as_deref() != Some("add")
                        || self.scalar_input_count != 1
                        || self.scalar_projections.len() != 1
                        || !matches!(
                            self.scalar_projections.first(),
                            Some(RecurrenceDirectScalarProjectionManifest::IntrinsicScale { .. })
                        )
                    {
                        return Err(RusticolError::integrity(
                            "Direct-Arena contribution intrinsic metadata is inconsistent",
                        ));
                    }
                    validate_sha256_text(
                        self.intrinsic_contract_digest
                            .as_deref()
                            .unwrap_or_default(),
                        "Direct-Arena intrinsic contract digest",
                    )?;
                    RecurrenceContributionIntrinsicKind::from_runtime_template(runtime_template)?;
                } else if self.role.is_some()
                    || self.destination_operation.is_some()
                    || self.scalar_input_count != 0
                    || !self.scalar_projections.is_empty()
                    || self.intrinsic_contract_digest.is_some()
                {
                    return Err(RusticolError::integrity(
                        "non-contribution Direct-Arena intrinsic carries contribution metadata",
                    ));
                } else if !match template.role.as_str() {
                    "source" => runtime_template.starts_with("rusticol.source-fill."),
                    "finalization" => matches!(
                        runtime_template,
                        "rusticol.identity-finalize-in-place.v1"
                            | WEYL_PROPAGATOR_POSITIVE_TEMPLATE
                            | WEYL_PROPAGATOR_NEGATIVE_TEMPLATE
                            | FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE
                    ),
                    "closure" => runtime_template.starts_with("rusticol.closure-reduce.v1:"),
                    _ => false,
                } {
                    return Err(RusticolError::compatibility(format!(
                        "unsupported Direct-Arena intrinsic {runtime_template:?} for role {:?}",
                        template.role
                    )));
                }
            }
            "prepared-direct-call" => {
                if self.runtime_template.is_some()
                    || self.intrinsic_contract_digest.is_some()
                    || self.role.as_deref() != Some(template.role.as_str())
                    || self.destination_operation.as_deref()
                        != Some(template.destination_operation.as_str())
                    || self.exact_factor_scalar_slots != [0, 1]
                    || self.input_plane_count as usize != self.input_plane_projections.len()
                    || self.scalar_input_count as usize != self.scalar_projections.len()
                {
                    return Err(RusticolError::integrity(
                        "prepared Direct-Arena callable metadata is inconsistent",
                    ));
                }
                let kernel_id = self.prepared_kernel_id.ok_or_else(|| {
                    RusticolError::artifact("prepared Direct-Arena callable has no kernel ID")
                })?;
                let kernel = kernels.get(&kernel_id).ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "Direct-Arena callable references absent prepared kernel {kernel_id}"
                    ))
                })?;
                let source_path = self.source_application_path.as_deref().ok_or_else(|| {
                    RusticolError::artifact("Direct-Arena callable has no source application")
                })?;
                if self.payload_paths.len() != 1 || self.payload_paths[0] != source_path {
                    return Err(RusticolError::integrity(
                        "Direct-Arena callable must reference exactly its source application",
                    ));
                }
                validate_sha256_text(
                    self.source_application_sha256
                        .as_deref()
                        .unwrap_or_default(),
                    "Direct-Arena source application digest",
                )?;
                validate_sha256_text(
                    self.prepared_template_semantic_digest
                        .as_deref()
                        .unwrap_or_default(),
                    "Direct-Arena prepared-template digest",
                )?;
                let evaluator = kernel.f64_evaluator_manifest.as_object().ok_or_else(|| {
                    RusticolError::artifact("prepared kernel evaluator metadata is not an object")
                })?;
                match template.backend.as_str() {
                    "jit" => {
                        let plane = evaluator
                            .get("plane_application")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                RusticolError::artifact(
                                    "prepared JIT kernel has no SymJIT plane application",
                                )
                            })?;
                        if self.direct_application_abi.as_deref()
                            != Some(SYMJIT_PLANE_APPLICATION_V2_ABI)
                            || self.source_application_abi.as_deref()
                                != Some(SYMJIT_PLANE_APPLICATION_V2_ABI)
                            || self.native_entry_point.is_some()
                            || plane.get("application_path").and_then(Value::as_str)
                                != Some(source_path)
                            || plane.get("application_abi").and_then(Value::as_str)
                                != self.source_application_abi.as_deref()
                            || plane.get("storage_abi").and_then(Value::as_str)
                                != Some(SYMJIT_APPLICATION_STORAGE_V3_ABI)
                            || plane.get("optimization_level").and_then(Value::as_u64)
                                != Some(PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL)
                            || !self.state_plane_indices.is_empty()
                            || !self.parameter_bindings.len().is_multiple_of(2)
                            || !self.output_alias_inputs.len().is_multiple_of(2)
                        {
                            return Err(RusticolError::integrity(
                                "Direct-Arena callable source does not match its portable O2 SymJIT plane kernel",
                            ));
                        }
                    }
                    backend @ ("cpp" | "asm") => {
                        let expected_source_abi = match backend {
                            "cpp" => SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
                            "asm" => SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
                            _ => unreachable!("native Direct-Arena backend was matched above"),
                        };
                        let expected_entry_point = format!(
                            "pyamplicol_recurrence_direct_{}_k{kernel_id:08x}_v1",
                            template.role
                        );
                        if self.direct_application_abi.as_deref()
                            != Some(NATIVE_DIRECT_APPLICATION_V1_ABI)
                            || self.source_application_abi.as_deref() != Some(expected_source_abi)
                            || self.native_entry_point.as_deref()
                                != Some(expected_entry_point.as_str())
                            || evaluator.get("kind").and_then(Value::as_str)
                                != Some("compiled-complex-evaluator")
                            || evaluator.get("library_path").and_then(Value::as_str)
                                != Some(source_path)
                        {
                            return Err(RusticolError::integrity(
                                "Direct-Arena callable source does not match its target-native prepared kernel",
                            ));
                        }
                    }
                    other => {
                        return Err(RusticolError::compatibility(format!(
                            "unsupported Direct-Arena prepared-call backend {other:?}"
                        )));
                    }
                }
                if self
                    .state_plane_indices
                    .iter()
                    .chain(&self.output_alias_inputs)
                    .any(|index| *index >= self.input_plane_count)
                {
                    return Err(RusticolError::integrity(
                        "Direct-Arena callable plane metadata is out of bounds",
                    ));
                }
                for binding in &self.parameter_bindings {
                    let (index, limit) = match *binding {
                        RecurrenceDirectParameterBindingManifest::Plane { index } => {
                            (index, self.input_plane_count)
                        }
                        RecurrenceDirectParameterBindingManifest::Scalar { index } => {
                            (index, self.scalar_input_count)
                        }
                    };
                    if index >= limit {
                        return Err(RusticolError::integrity(
                            "Direct-Arena parameter binding is out of bounds",
                        ));
                    }
                }
                for projection in &self.input_plane_projections {
                    let valid = match *projection {
                        RecurrenceDirectPlaneProjectionManifest::ParentCurrent {
                            parent,
                            component,
                            ..
                        } => template
                            .parent_component_counts
                            .get(parent as usize)
                            .is_some_and(|count| u32::from(component) < *count),
                        RecurrenceDirectPlaneProjectionManifest::Momentum {
                            operand,
                            lorentz_component,
                        } => {
                            u32::from(operand) < template.momentum_operand_count
                                && lorentz_component < 4
                        }
                        RecurrenceDirectPlaneProjectionManifest::DestinationCurrent {
                            component,
                            ..
                        }
                        | RecurrenceDirectPlaneProjectionManifest::DestinationAmplitude {
                            component,
                            ..
                        } => u32::from(component) < template.destination_component_count,
                    };
                    if !valid {
                        return Err(RusticolError::integrity(
                            "Direct-Arena input-plane projection is outside its template shape",
                        ));
                    }
                }
                if !matches!(
                    self.scalar_projections.first(),
                    Some(RecurrenceDirectScalarProjectionManifest::ExactFactor {
                        imaginary: false
                    })
                ) || !matches!(
                    self.scalar_projections.get(1),
                    Some(RecurrenceDirectScalarProjectionManifest::ExactFactor { imaginary: true })
                ) {
                    return Err(RusticolError::integrity(
                        "Direct-Arena scalar slots 0 and 1 must project the exact complex factor",
                    ));
                }
                let _raw = raw.ok_or_else(|| {
                    RusticolError::artifact("Direct-Arena payload binding is absent")
                })?;
            }
            "pending-direct-call-abi" => {
                return Err(RusticolError::compatibility(
                    "prepared Direct-Arena callable is not executable; recompile with a supported backend",
                ));
            }
            other => {
                return Err(RusticolError::compatibility(format!(
                    "unsupported Direct-Arena payload binding kind {other:?}"
                )));
            }
        }
        Ok(())
    }
}

fn expected_direct_operation(role: &str) -> RusticolResult<&'static str> {
    match role {
        "source" => Ok("initialize"),
        "contribution" => Ok("add"),
        "finalization" => Ok("finalize-in-place"),
        "closure" => Ok("closure-add"),
        other => Err(RusticolError::compatibility(format!(
            "unsupported Direct-Arena executor role {other:?}"
        ))),
    }
}

fn validate_sha256_text(value: &str, label: &str) -> RusticolResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RusticolError::artifact(format!(
            "{label} must be a lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

impl PreparedKernelManifest {
    pub(super) fn runtime_evaluator_manifest(&self) -> RusticolResult<EvaluatorManifest> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator manifest must be an object",
                self.kernel_id
            ))
        })?;
        let kind = object.get("kind").and_then(Value::as_str).ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator kind must be a nonempty string",
                self.kernel_id
            ))
        })?;
        let metadata_fields: &[&str] = match kind {
            "symjit-application-evaluator" => &[
                "backend",
                "label",
                "settings",
                "build_timing",
                "direct_table",
            ],
            "compiled-complex-evaluator" => &[
                "backend",
                "settings",
                "source_path",
                "build_timing",
                "direct_table",
            ],
            other => {
                return Err(RusticolError::compatibility(format!(
                    "prepared kernel {} has unsupported f64 evaluator kind {other:?}",
                    self.kernel_id
                )));
            }
        };
        validate_prepared_evaluator_keys(self.kernel_id, kind, object)?;
        let mut runtime = object.clone();
        for field in metadata_fields {
            runtime.remove(*field);
        }
        serde_json::from_value(Value::Object(runtime)).map_err(|error| {
            RusticolError::serialization(format!(
                "prepared kernel {} has invalid runtime evaluator metadata: {error}",
                self.kernel_id
            ))
        })
    }

    pub(super) fn extra_evaluator_payload_paths(&self) -> RusticolResult<Vec<&str>> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact("prepared f64 evaluator manifest must be an object")
        })?;
        match object.get("kind").and_then(Value::as_str) {
            Some("symjit-application-evaluator") => {
                let Some(direct) = object.get("direct_table") else {
                    return Ok(Vec::new());
                };
                let direct = direct.as_object().ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "prepared kernel {} DirectTable metadata must be an object",
                        self.kernel_id
                    ))
                })?;
                if let Some(path) = direct.get("descriptor_path") {
                    Ok(vec![path.as_str().ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "prepared kernel {} DirectTable descriptor path must be text",
                            self.kernel_id
                        ))
                    })?])
                } else if let Some(path) = direct.get("library_path") {
                    Ok(vec![path.as_str().ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "prepared kernel {} DirectTable library path must be text",
                            self.kernel_id
                        ))
                    })?])
                } else {
                    Err(RusticolError::artifact(format!(
                        "prepared kernel {} DirectTable has no payload path",
                        self.kernel_id
                    )))
                }
            }
            Some("compiled-complex-evaluator") => {
                let direct = object.get("direct_table").and_then(Value::as_object);
                let mut paths = vec![required_nonempty_string(
                    object,
                    "source_path",
                    self.kernel_id,
                )?];
                if let Some(direct) = direct {
                    paths.push(required_nonempty_string(
                        direct,
                        "library_path",
                        self.kernel_id,
                    )?);
                }
                Ok(paths)
            }
            _ => Err(RusticolError::compatibility(format!(
                "prepared kernel {} has an unsupported f64 evaluator kind",
                self.kernel_id
            ))),
        }
    }

    pub(super) fn eager_direct_table_manifest(&self) -> RusticolResult<EagerDirectTableManifest> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator manifest must be an object",
                self.kernel_id
            ))
        })?;
        let raw = object.get("direct_table").ok_or_else(|| {
            RusticolError::compatibility(format!(
                "prepared eager kernel {} predates {:?}; regenerate the artifact with the \
                 current `pyamplicol generate`",
                self.kernel_id,
                crate::eager_layout::EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            ))
        })?;
        let direct: EagerDirectTableManifest =
            serde_json::from_value(raw.clone()).map_err(|error| {
                RusticolError::serialization(format!(
                    "prepared eager kernel {} has invalid DirectTable metadata: {error}",
                    self.kernel_id
                ))
            })?;
        if direct.capability != crate::eager_layout::EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY
            || !matches!(
                direct.source_application_abi.as_str(),
                crate::eager_layout::EAGER_DIRECT_SOURCE_APPLICATION_ABI
                    | crate::eager_layout::EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI
            )
            || direct.descriptor_abi != crate::eager_layout::EAGER_DIRECT_TABLE_DESCRIPTOR_ABI
            || direct.binding_abi != crate::eager_layout::EAGER_DIRECT_TABLE_BINDING_ABI
        {
            return Err(RusticolError::compatibility(format!(
                "prepared eager kernel {} has an unsupported DirectTable ABI contract; \
                 regenerate the artifact with the current `pyamplicol generate`",
                self.kernel_id
            )));
        }
        match direct.source_application_abi.as_str() {
            crate::eager_layout::EAGER_DIRECT_SOURCE_APPLICATION_ABI => {
                if direct.descriptor_path.as_deref().is_none_or(str::is_empty) {
                    return Err(RusticolError::artifact(format!(
                        "prepared eager kernel {} DirectTable descriptor path is empty",
                        self.kernel_id
                    )));
                }
                let size = direct.descriptor_size_bytes.unwrap_or(0);
                if size == 0 {
                    return Err(RusticolError::artifact(format!(
                        "prepared eager kernel {} DirectTable descriptor must not be empty",
                        self.kernel_id
                    )));
                }
                validate_sha256_text(
                    direct.descriptor_sha256.as_deref().unwrap_or_default(),
                    "eager DirectTable descriptor digest",
                )?;
            }
            crate::eager_layout::EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI => {
                if direct.library_path.as_deref().is_none_or(str::is_empty)
                    || direct.function_name.as_deref().is_none_or(str::is_empty)
                    || direct.invocation_stride.unwrap_or(0) == 0
                    || direct.attachment_stride.unwrap_or(0) == 0
                    || !matches!(direct.simd_lane_width, Some(2 | 4))
                {
                    return Err(RusticolError::artifact(format!(
                        "prepared eager kernel {} native DirectTable metadata is incomplete",
                        self.kernel_id
                    )));
                }
                validate_sha256_text(
                    direct.evaluator_state_sha256.as_deref().unwrap_or_default(),
                    "eager native DirectTable evaluator-state digest",
                )?;
            }
            _ => unreachable!("source ABI admitted above"),
        }
        if usize::try_from(direct.input_complex_count).ok() != Some(self.input_arity)
            || direct.output_complex_count != self.output_arity
        {
            return Err(RusticolError::integrity(format!(
                "prepared eager kernel {} DirectTable I/O ({}, {}) does not match ({}, {})",
                self.kernel_id,
                direct.input_complex_count,
                direct.output_complex_count,
                self.input_arity,
                self.output_arity,
            )));
        }
        Ok(direct)
    }

    fn validate_evaluator_metadata(&self, pack: &PreparedKernelPackManifest) -> RusticolResult<()> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator manifest must be an object",
                self.kernel_id
            ))
        })?;
        let kind = required_nonempty_string(object, "kind", self.kernel_id)?;
        let backend = required_nonempty_string(object, "backend", self.kernel_id)?;
        let (expected_kind, expected_backend, expected_capability) = match pack.backend.as_str() {
            "jit" => (
                "symjit-application-evaluator",
                "jit",
                SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            ),
            "asm" => (
                "compiled-complex-evaluator",
                "compiled-complex",
                SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
            ),
            "cpp" => (
                "compiled-complex-evaluator",
                "compiled-complex",
                SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            ),
            _ => unreachable!("prepared pack backend validated before its kernels"),
        };
        if kind != expected_kind || backend != expected_backend {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} evaluator ({kind:?}, {backend:?}) does not match pack backend {:?}, expected ({expected_kind:?}, {expected_backend:?})",
                self.kernel_id, pack.backend,
            )));
        }
        let settings = object.get("settings").ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} evaluator lacks settings metadata",
                self.kernel_id
            ))
        })?;
        if settings != &pack.optimization_settings {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} evaluator settings do not match its pack",
                self.kernel_id
            )));
        }
        let build_timing = object
            .get("build_timing")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "prepared kernel {} evaluator build_timing must be an object",
                    self.kernel_id
                ))
            })?;
        if build_timing.values().any(|value| {
            value
                .as_f64()
                .is_none_or(|seconds| !seconds.is_finite() || seconds < 0.0)
        }) {
            return Err(RusticolError::artifact(format!(
                "prepared kernel {} evaluator build timings must be finite nonnegative numbers",
                self.kernel_id
            )));
        }
        if kind == "symjit-application-evaluator" {
            validate_prepared_symjit_plane_application(
                self.kernel_id,
                object,
                self.input_arity,
                usize::try_from(self.output_arity).unwrap_or(usize::MAX),
                PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL as u8,
            )?;
        }
        let runtime = self.runtime_evaluator_manifest()?;
        let capabilities = evaluator_runtime_capabilities(&runtime)?;
        if capabilities != BTreeSet::from([expected_capability.to_string()]) {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} pack backend {:?} declares evaluator capabilities {capabilities:?}, expected {expected_capability:?}",
                self.kernel_id, pack.backend,
            )));
        }
        let (input_len, output_len) = runtime.io_len()?;
        if input_len != self.input_arity
            || output_len != usize::try_from(self.output_arity).unwrap_or(usize::MAX)
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} evaluator I/O ({input_len}, {output_len}) does not match ({}, {})",
                self.kernel_id, self.input_arity, self.output_arity
            )));
        }
        match kind {
            "symjit-application-evaluator" => {
                if pack.backend != "jit" {
                    return Err(RusticolError::integrity(format!(
                        "prepared kernel {} uses SymJIT under backend {:?}",
                        self.kernel_id, pack.backend
                    )));
                }
                if required_nonempty_string(object, "label", self.kernel_id)?.is_empty() {
                    return Err(RusticolError::artifact("prepared evaluator label is empty"));
                }
                let application_abi =
                    required_nonempty_string(object, "application_abi", self.kernel_id)?;
                if pack
                    .dependency_abis
                    .get("symjit_application")
                    .and_then(Value::as_str)
                    != Some(application_abi)
                {
                    return Err(RusticolError::compatibility(format!(
                        "prepared kernel {} SymJIT application ABI does not match its pack",
                        self.kernel_id
                    )));
                }
                if object.get("optimization_level").and_then(Value::as_u64)
                    != Some(PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL)
                {
                    return Err(RusticolError::compatibility(format!(
                        "prepared kernel {} must use portable SymJIT optimization level {}",
                        self.kernel_id, PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
                    )));
                }
                if object.contains_key("direct_table") {
                    let direct = self.eager_direct_table_manifest()?;
                    let plane = object
                        .get("plane_application")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            RusticolError::compatibility(format!(
                                "prepared eager JIT kernel {} predates the SymJIT plane-application ABI; regenerate the prepared model",
                                self.kernel_id
                            ))
                        })?;
                    if direct.source_application_abi
                        != crate::eager_layout::EAGER_DIRECT_SOURCE_APPLICATION_ABI
                        || plane.get("application_abi").and_then(Value::as_str)
                            != Some(crate::eager_layout::EAGER_DIRECT_SOURCE_APPLICATION_ABI)
                        || plane.get("storage_abi").and_then(Value::as_str)
                            != Some(SYMJIT_APPLICATION_STORAGE_V3_ABI)
                        || plane
                            .get("application_path")
                            .and_then(Value::as_str)
                            .is_none_or(str::is_empty)
                        || plane.get("input_complex_count").and_then(Value::as_u64)
                            != u64::try_from(self.input_arity).ok()
                        || plane.get("output_complex_count").and_then(Value::as_u64)
                            != Some(u64::from(self.output_arity))
                        || plane.get("optimization_level").and_then(Value::as_u64)
                            != Some(PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL)
                    {
                        return Err(RusticolError::compatibility(format!(
                            "prepared eager JIT kernel {} has an incompatible SymJIT plane application",
                            self.kernel_id
                        )));
                    }
                }
            }
            "compiled-complex-evaluator" => {
                if !matches!(pack.backend.as_str(), "asm" | "cpp") {
                    return Err(RusticolError::integrity(format!(
                        "prepared kernel {} uses a compiled evaluator under backend {:?}",
                        self.kernel_id, pack.backend
                    )));
                }
                required_nonempty_string(object, "source_path", self.kernel_id)?;
                let direct = self.eager_direct_table_manifest()?;
                if direct.source_application_abi
                    != crate::eager_layout::EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI
                {
                    return Err(RusticolError::compatibility(format!(
                        "prepared native kernel {} does not provide a native eager DirectTable",
                        self.kernel_id
                    )));
                }
            }
            _ => unreachable!("runtime evaluator projection validated the kind"),
        }
        if object.get("evaluator_state_path").and_then(Value::as_str)
            != Some(self.exact_evaluator_state_path.as_str())
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} exact evaluator state does not match its f64 evaluator fallback",
                self.kernel_id
            )));
        }
        Ok(())
    }
}

impl PreparedKernelVariantManifest {
    pub(super) fn runtime_evaluator_manifest(&self) -> RusticolResult<EvaluatorManifest> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} variant {:?} evaluator manifest must be an object",
                self.base_kernel_id, self.variant_id
            ))
        })?;
        let kind = object.get("kind").and_then(Value::as_str).ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} variant {:?} evaluator kind must be a nonempty string",
                self.base_kernel_id, self.variant_id
            ))
        })?;
        if kind != "symjit-application-evaluator" {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {} variant {:?} has unsupported evaluator kind {kind:?}",
                self.base_kernel_id, self.variant_id
            )));
        }
        validate_prepared_evaluator_keys(self.base_kernel_id, kind, object)?;
        let mut runtime = object.clone();
        for field in ["backend", "label", "settings", "build_timing"] {
            runtime.remove(field);
        }
        serde_json::from_value(Value::Object(runtime)).map_err(|error| {
            RusticolError::serialization(format!(
                "prepared kernel {} variant {:?} has invalid runtime evaluator metadata: {error}",
                self.base_kernel_id, self.variant_id
            ))
        })
    }

    pub(super) fn extra_evaluator_payload_paths(&self) -> RusticolResult<Vec<&str>> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact("prepared block evaluator manifest must be an object")
        })?;
        if object.get("kind").and_then(Value::as_str) != Some("symjit-application-evaluator") {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {} variant {:?} has an unsupported evaluator kind",
                self.base_kernel_id, self.variant_id
            )));
        }
        Ok(Vec::new())
    }

    fn validate(
        &self,
        pack: &PreparedKernelPackManifest,
        base: &PreparedKernelManifest,
    ) -> RusticolResult<()> {
        if self.variant_abi != PREPARED_KERNEL_VARIANT_ABI
            || self.variant_id != PREPARED_INDEPENDENT_BLOCK_VARIANT_ID
            || self.kind != "independent-block"
            || self.block_size != EAGER_INDEPENDENT_BLOCK_SIZE
            || self.lane_layout != "lane-major"
        {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {} has unsupported block variant metadata; \
                 regenerate the artifact with the current `pyamplicol generate`",
                self.base_kernel_id
            )));
        }
        if pack.backend != "jit" || self.backend != pack.backend {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block variant backend does not match its JIT pack",
                self.base_kernel_id
            )));
        }
        if base.contract_kind != "vertex"
            || !base
                .proof_classes
                .iter()
                .any(|proof| proof == PREPARED_INDEPENDENT_BLOCK_PROOF)
            || base
                .input_contracts
                .iter()
                .any(|input| !matches!(input.role.as_str(), "left-current" | "right-current"))
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block variant lacks its current-only vertex proof",
                self.base_kernel_id
            )));
        }
        if self.base_canonical_signature != base.canonical_signature
            || self.input_lane_stride != base.input_arity
            || self.output_lane_stride != usize::try_from(base.output_arity).unwrap_or(usize::MAX)
            || self.input_arity != self.input_lane_stride * self.block_size as usize
            || self.output_arity != self.output_lane_stride * self.block_size as usize
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block variant does not match its scalar arities",
                self.base_kernel_id
            )));
        }
        let expected_input_layout = (0..self.block_size)
            .flat_map(|lane| {
                base.input_layout
                    .iter()
                    .map(move |item| format!("lane:{lane}:{item}"))
            })
            .collect::<Vec<_>>();
        let expected_output_layout = (0..self.block_size)
            .flat_map(|lane| {
                base.output_layout
                    .iter()
                    .map(move |item| format!("lane:{lane}:{item}"))
            })
            .collect::<Vec<_>>();
        if self.input_layout != expected_input_layout
            || self.output_layout != expected_output_layout
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block variant has an incompatible lane layout",
                self.base_kernel_id
            )));
        }
        let input_contracts = base
            .input_contracts
            .iter()
            .map(|input| {
                serde_json::json!({
                    "role": input.role,
                    "component": input.component,
                    "symbol": input.symbol,
                    "model_parameter_name": input.model_parameter_name,
                    "model_parameter_index": input.model_parameter_index,
                })
            })
            .collect::<Vec<_>>();
        let expected_expression_digest = canonical_json_digest(&serde_json::json!({
            "exact_expressions": base.exact_expressions,
        }))?;
        let expected_input_digest = canonical_json_digest(&serde_json::json!({
            "input_arity": base.input_arity,
            "input_layout": base.input_layout,
            "input_contracts": input_contracts,
        }))?;
        let expected_output_digest = canonical_json_digest(&serde_json::json!({
            "output_arity": base.output_arity,
            "output_layout": base.output_layout,
        }))?;
        let expected_optimization_digest = canonical_json_digest(&pack.optimization_settings)?;
        if self.base_expression_digest != expected_expression_digest
            || self.base_input_contract_digest != expected_input_digest
            || self.base_output_contract_digest != expected_output_digest
            || self.optimization_settings_digest != expected_optimization_digest
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block variant digest does not match its scalar contract",
                self.base_kernel_id
            )));
        }

        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact("prepared block evaluator manifest must be an object")
        })?;
        if required_nonempty_string(object, "kind", self.base_kernel_id)?
            != "symjit-application-evaluator"
            || required_nonempty_string(object, "backend", self.base_kernel_id)? != "jit"
            || object.get("settings") != Some(&pack.optimization_settings)
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block evaluator does not match its JIT pack",
                self.base_kernel_id
            )));
        }
        if object.get("optimization_level").and_then(Value::as_u64)
            != Some(PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL)
        {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {} block evaluator must use portable SymJIT optimization level {}",
                self.base_kernel_id, PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
            )));
        }
        let build_timing = object
            .get("build_timing")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                RusticolError::artifact("prepared block build_timing must be an object")
            })?;
        if build_timing.values().any(|value| {
            value
                .as_f64()
                .is_none_or(|seconds| !seconds.is_finite() || seconds < 0.0)
        }) {
            return Err(RusticolError::artifact(
                "prepared block build timings must be finite nonnegative numbers",
            ));
        }
        required_nonempty_string(object, "label", self.base_kernel_id)?;
        required_nonempty_string(object, "evaluator_state_path", self.base_kernel_id)?;
        validate_prepared_symjit_plane_application(
            self.base_kernel_id,
            object,
            self.input_arity,
            self.output_arity,
            PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL as u8,
        )?;
        let application_abi =
            required_nonempty_string(object, "application_abi", self.base_kernel_id)?;
        if pack
            .dependency_abis
            .get("symjit_application")
            .and_then(Value::as_str)
            != Some(application_abi)
        {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {} block evaluator SymJIT ABI does not match its pack",
                self.base_kernel_id
            )));
        }
        let runtime = self.runtime_evaluator_manifest()?;
        if evaluator_runtime_capabilities(&runtime)?
            != BTreeSet::from([SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()])
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block evaluator has incompatible capabilities",
                self.base_kernel_id
            )));
        }
        let (input_len, output_len) = runtime.io_len()?;
        if input_len != self.input_arity || output_len != self.output_arity {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} block evaluator I/O ({input_len}, {output_len}) does not match ({}, {})",
                self.base_kernel_id, self.input_arity, self.output_arity
            )));
        }
        Ok(())
    }
}

fn canonical_json_digest(value: &Value) -> RusticolResult<String> {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical)?;
    canonical.push('\n');
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn canonical_json_digest_exact(value: &Value) -> RusticolResult<String> {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn write_canonical_json(value: &Value, output: &mut String) -> RusticolResult<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&python_json_number(value)),
        Value::String(value) => write_ascii_json_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_ascii_json_string(key, output);
                output.push(':');
                write_canonical_json(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn python_json_number(number: &serde_json::Number) -> String {
    let rendered = number.to_string();
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return rendered;
    };
    let (sign, digits) = if let Some(digits) = exponent.strip_prefix('-') {
        ('-', digits)
    } else if let Some(digits) = exponent.strip_prefix('+') {
        ('+', digits)
    } else {
        ('+', exponent)
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

fn write_ascii_json_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0020}'..='\u{007e}' => output.push(character),
            character if u32::from(character) <= 0xffff => {
                let _ = write!(output, "\\u{:04x}", u32::from(character));
            }
            character => {
                let scalar = u32::from(character) - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                let _ = write!(output, "\\u{high:04x}\\u{low:04x}");
            }
        }
    }
    output.push('"');
}

fn validate_prepared_evaluator_keys(
    kernel_id: u32,
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> RusticolResult<()> {
    if kind == "symjit-application-evaluator" && !object.contains_key("plane_application") {
        return Err(RusticolError::compatibility(format!(
            "prepared JIT kernel {kernel_id} predates the SymJIT plane-application ABI; \
             regenerate the prepared model"
        )));
    }
    let expected = match kind {
        "symjit-application-evaluator" => [
            "application_abi",
            "application_path",
            "backend",
            "batch_layout",
            "build_timing",
            "compiler_type",
            "direct_table",
            "element_layout",
            "endianness",
            "evaluator_state_path",
            "evaluator_state_runtime_capability",
            "input_len",
            "kind",
            "label",
            "optimization_level",
            "output_len",
            "plane_application",
            "required_defuns",
            "runtime_capability",
            "settings",
            "translation_mode",
            "word_bits",
        ]
        .as_slice(),
        "compiled-complex-evaluator" => [
            "backend",
            "build_timing",
            "direct_table",
            "evaluator_state_path",
            "function_name",
            "input_len",
            "kind",
            "library_path",
            "number_type",
            "output_len",
            "runtime_capability",
            "settings",
            "source_path",
        ]
        .as_slice(),
        _ => {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {kernel_id} has unsupported evaluator kind {kind:?}"
            )));
        }
    };
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    let expected_without_direct = expected
        .iter()
        .copied()
        .filter(|field| *field != "direct_table")
        .collect::<BTreeSet<_>>();
    if actual != expected && actual != expected_without_direct {
        return Err(RusticolError::artifact(format!(
            "prepared kernel {kernel_id} evaluator fields {actual:?} do not match {expected:?}"
        )));
    }
    Ok(())
}

fn validate_prepared_symjit_plane_application(
    kernel_id: u32,
    object: &serde_json::Map<String, Value>,
    input_len: usize,
    output_len: usize,
    optimization_level: u8,
) -> RusticolResult<()> {
    let raw = object.get("plane_application").ok_or_else(|| {
        RusticolError::compatibility(format!(
            "prepared JIT kernel {kernel_id} predates the SymJIT plane-application ABI; \
             regenerate the prepared model"
        ))
    })?;
    let plane: SymjitPlaneApplicationManifest =
        serde_json::from_value(raw.clone()).map_err(|error| {
            RusticolError::compatibility(format!(
                "prepared JIT kernel {kernel_id} has invalid SymJIT plane-application metadata; \
                 regenerate the prepared model: {error}"
            ))
        })?;
    plane.validate(input_len, output_len, optimization_level)
}

fn required_nonempty_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    kernel_id: u32,
) -> RusticolResult<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {kernel_id} evaluator field {field:?} must be a nonempty string"
            ))
        })
}

impl PreparedKernelInputManifest {
    fn to_eager_input(&self) -> RusticolResult<EagerKernelInput> {
        if self.symbol.is_empty() {
            return Err(RusticolError::integrity(
                "prepared kernel input symbol must be nonempty",
            ));
        }
        let input = match self.role.as_str() {
            "left-current" => EagerKernelInput::FirstCurrentComponent(self.component),
            "right-current" => EagerKernelInput::SecondCurrentComponent(self.component),
            "left-momentum" => EagerKernelInput::FirstMomentumComponent(self.component),
            "right-momentum" => EagerKernelInput::SecondMomentumComponent(self.component),
            "current" => EagerKernelInput::FirstCurrentComponent(self.component),
            "momentum" => EagerKernelInput::FirstMomentumComponent(self.component),
            "coupling-real" => EagerKernelInput::CouplingReal,
            "coupling-imag" => EagerKernelInput::CouplingImag,
            "model-parameter" => {
                EagerKernelInput::ModelParameter(self.model_parameter_index.ok_or_else(|| {
                    RusticolError::integrity(
                        "prepared model-parameter input lacks its stable index",
                    )
                })?)
            }
            other => {
                return Err(RusticolError::artifact(format!(
                    "unsupported prepared kernel input role {other:?}"
                )));
            }
        };
        if self.role == "model-parameter" {
            if self
                .model_parameter_name
                .as_deref()
                .unwrap_or("")
                .is_empty()
            {
                return Err(RusticolError::integrity(
                    "prepared model-parameter input lacks its name",
                ));
            }
        } else if self.model_parameter_name.is_some() || self.model_parameter_index.is_some() {
            return Err(RusticolError::integrity(
                "only prepared model-parameter inputs may carry parameter metadata",
            ));
        }
        Ok(input)
    }
}

fn contiguous_value_slot_widths(plan: &ExecutionPlan) -> RusticolResult<Vec<u32>> {
    let pairs = plan
        .value_storage
        .value_slots
        .iter()
        .map(|slot| (slot.value_slot_id, slot.dimension))
        .collect::<Vec<_>>();
    contiguous_widths("value", pairs)
}

fn contiguous_momentum_slot_widths(plan: &ExecutionPlan) -> RusticolResult<Vec<u32>> {
    let pairs = plan
        .momentum_slots
        .iter()
        .map(|slot| {
            let width = slot
                .component_stop
                .checked_sub(slot.component_start)
                .ok_or_else(|| {
                    RusticolError::artifact("eager momentum slot has an inverted component range")
                })?;
            Ok((slot.momentum_slot_id, width))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    contiguous_widths("momentum", pairs)
}

fn contiguous_current_slot_widths(plan: &ExecutionPlan) -> RusticolResult<Vec<u32>> {
    let pairs = plan
        .current_storage
        .current_slots
        .iter()
        .map(|slot| (slot.current_id, slot.dimension))
        .collect::<Vec<_>>();
    contiguous_widths("current", pairs)
}

fn contiguous_widths(name: &str, mut pairs: Vec<(usize, usize)>) -> RusticolResult<Vec<u32>> {
    pairs.sort_unstable_by_key(|(id, _)| *id);
    if pairs
        .iter()
        .enumerate()
        .any(|(expected, (id, _))| expected != *id)
    {
        return Err(RusticolError::artifact(format!(
            "eager {name} slot ids must be contiguous from zero"
        )));
    }
    pairs
        .into_iter()
        .map(|(id, width)| {
            if width == 0 {
                return Err(RusticolError::artifact(format!(
                    "eager {name} slot {id} has zero width"
                )));
            }
            u32::try_from(width)
                .map_err(|_| RusticolError::artifact(format!("eager {name} width exceeds u32")))
        })
        .collect()
}
