// SPDX-License-Identifier: 0BSD

//! Process-owned operational Direct-executor records embedded in PACRDBN4.

use super::eager_manifest::{
    RecurrenceDirectParameterBindingManifest, RecurrenceDirectPlaneProjectionManifest,
    RecurrenceDirectScalarProjectionManifest,
};
use crate::recurrence::{
    DIRECT_NONE_U32, DirectExecutorRole, DirectRecurrencePlan, SemanticDigest,
};
use crate::{RusticolError, RusticolResult, Target};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const RECURRENCE_PROCESS_BINDING_MAGIC_V4: &[u8; 8] = b"PACRDBN4";
pub(super) const RECURRENCE_PROCESS_BINDING_VERSION_V4: u32 = 4;
pub(super) const RECURRENCE_PROCESS_BINDING_HEADER_SIZE_V4: usize = 344;

const PROCESS_EXECUTOR_FLAG_USES_EXACT_FACTOR: u16 = 1 << 0;
const PROCESS_EXECUTOR_FLAGS_KNOWN: u16 = PROCESS_EXECUTOR_FLAG_USES_EXACT_FACTOR;
const PREPARED_JIT_PORTABLE_TARGET: &str = "symjit-storage-v3-portable";
const PREPARED_JIT_OPTIMIZATION_LEVEL: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProcessExecutorBackend {
    Jit,
    Cpp,
    Asm,
}

impl ProcessExecutorBackend {
    pub(super) fn decode(value: u8) -> RusticolResult<Self> {
        match value {
            0 => Ok(Self::Jit),
            1 => Ok(Self::Cpp),
            2 => Ok(Self::Asm),
            _ => Err(RusticolError::compatibility(format!(
                "unsupported recurrence process-executor backend tag {value}"
            ))),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Jit => "jit",
            Self::Cpp => "cpp",
            Self::Asm => "asm",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ProcessExecutorTarget {
    pub(super) backend: ProcessExecutorBackend,
    pub(super) target_triple: String,
    pub(super) portable: bool,
    pub(super) cpu_features: Vec<String>,
}

impl ProcessExecutorTarget {
    pub(super) fn validate(&self, outer: &Target) -> RusticolResult<()> {
        if self.cpu_features.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RusticolError::integrity(
                "recurrence process-executor CPU features are not sorted and unique",
            ));
        }
        match self.backend {
            ProcessExecutorBackend::Jit => {
                if !matches!(std::env::consts::ARCH, "aarch64" | "x86_64")
                    || !self.portable
                    || self.target_triple != PREPARED_JIT_PORTABLE_TARGET
                    || !self.cpu_features.is_empty()
                    || outer.triple != crate::artifact::PORTABLE_64LE_ARTIFACT_TARGET
                    || !outer.cpu_features.is_empty()
                {
                    return Err(RusticolError::compatibility(
                        "portable recurrence JIT executors have an incompatible target",
                    ));
                }
            }
            ProcessExecutorBackend::Cpp | ProcessExecutorBackend::Asm => {
                let host = crate::runtime_target_info();
                let host_features = host.cpu_features.into_iter().collect::<BTreeSet<_>>();
                if self.portable
                    || self.target_triple != host.triple
                    || outer.triple != host.triple
                    || self
                        .cpu_features
                        .iter()
                        .any(|feature| !host_features.contains(feature))
                    || self.cpu_features != outer.cpu_features
                {
                    return Err(RusticolError::compatibility(
                        "target-native recurrence executors are incompatible with this host",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct ProcessExecutorIdentities {
    pub(super) compiled_model_digest: SemanticDigest,
    pub(super) recurrence_template_catalog_digest: SemanticDigest,
    pub(super) prepared_kernel_pack_digest: SemanticDigest,
    pub(super) direct_template_catalog_digest: SemanticDigest,
    pub(super) runtime_layout_digest: SemanticDigest,
}

#[derive(Clone, Debug)]
pub(super) struct ProcessDirectExecutorPack {
    pub(super) identities: ProcessExecutorIdentities,
    pub(super) target: ProcessExecutorTarget,
    pub(super) catalog_executor_count: u32,
    pub(super) descriptors: Vec<ProcessDirectExecutorDescriptor>,
}

impl ProcessDirectExecutorPack {
    pub(super) fn validate_for_plan(
        &self,
        plan: &DirectRecurrencePlan,
        expected_prepared_pack_digest: SemanticDigest,
        expected_catalog_digest: SemanticDigest,
        expected_root_runtime_layout_digest: SemanticDigest,
    ) -> RusticolResult<()> {
        if self.catalog_executor_count != plan.direct_executor_count()
            || self.identities.prepared_kernel_pack_digest != expected_prepared_pack_digest
            || self.identities.prepared_kernel_pack_digest != plan.prepared_pack_digest()
            || self.identities.direct_template_catalog_digest != expected_catalog_digest
            || self.identities.direct_template_catalog_digest
                != plan.direct_template_catalog_digest()
            || self.identities.runtime_layout_digest != expected_root_runtime_layout_digest
        {
            return Err(RusticolError::integrity(
                "recurrence process-executor identities disagree with the remapped plan",
            ));
        }
        let descriptors = self
            .descriptors
            .iter()
            .map(|descriptor| (descriptor.direct_executor_id, descriptor))
            .collect::<BTreeMap<_, _>>();
        if descriptors.len() != self.descriptors.len() {
            return Err(RusticolError::integrity(
                "recurrence process-executor pack repeats an executor ID",
            ));
        }
        if descriptors
            .keys()
            .any(|executor_id| *executor_id >= self.catalog_executor_count)
        {
            return Err(RusticolError::integrity(
                "recurrence process-executor pack contains an out-of-range executor ID",
            ));
        }
        let mut required = BTreeMap::<u32, DirectExecutorRole>::new();
        for group in plan.row_groups() {
            if group.direct_executor_id == DIRECT_NONE_U32 {
                continue;
            }
            if group.direct_executor_id >= self.catalog_executor_count {
                return Err(RusticolError::integrity(
                    "recurrence plan references an executor outside its complete catalog",
                ));
            }
            if required
                .insert(group.direct_executor_id, group.role)
                .is_some_and(|previous| previous != group.role)
            {
                return Err(RusticolError::integrity(format!(
                    "recurrence executor {} is used with multiple roles",
                    group.direct_executor_id
                )));
            }
        }
        if required.len() != descriptors.len()
            || required.keys().copied().ne(descriptors.keys().copied())
        {
            return Err(RusticolError::integrity(
                "recurrence process-executor pack has a missing or extra required executor",
            ));
        }
        for (executor_id, role) in required {
            let descriptor = descriptors.get(&executor_id).ok_or_else(|| {
                RusticolError::integrity("required recurrence executor descriptor is absent")
            })?;
            if descriptor.role != role {
                return Err(RusticolError::integrity(format!(
                    "recurrence executor {executor_id} has the wrong role"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct ProcessDirectExecutorDescriptor {
    pub(super) direct_executor_id: u32,
    pub(super) role: DirectExecutorRole,
    pub(super) destination_component_count: u32,
    pub(super) uses_exact_factor: bool,
    pub(super) binding: ProcessDirectExecutorBinding,
}

#[derive(Clone, Debug)]
pub(super) enum ProcessDirectExecutorBinding {
    Source,
    Intrinsic(ProcessIntrinsicExecutor),
    Jit(ProcessJitExecutor),
    Native(ProcessNativeExecutor),
}

#[derive(Clone, Debug)]
pub(super) struct ProcessIntrinsicExecutor {
    pub(super) runtime_template: String,
    pub(super) scale: Option<ProcessIntrinsicScale>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ProcessIntrinsicScale {
    pub(super) constant_real_bits: u64,
    pub(super) constant_imag_bits: u64,
    pub(super) parameter_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct ProcessJitExecutor {
    pub(super) optimization_level: u32,
    pub(super) plane_compression: bool,
    pub(super) source_application_sha256: [u8; 32],
    pub(super) source_application_path: String,
    pub(super) source_application_abi: String,
    pub(super) parameter_bindings: Vec<RecurrenceDirectParameterBindingManifest>,
    pub(super) input_plane_projections: Vec<RecurrenceDirectPlaneProjectionManifest>,
    pub(super) scalar_projections: Vec<RecurrenceDirectScalarProjectionManifest>,
    pub(super) output_alias_inputs: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct ProcessNativeExecutor {
    pub(super) prepared_kernel_id: u32,
    pub(super) library_path: String,
    pub(super) native_entry_point: String,
    pub(super) coupling: Option<(f64, f64)>,
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
#[cfg_attr(target_vendor = "apple", cold)]
pub(super) fn decode_process_executor_descriptors(
    bytes: &[u8],
    cursor: &mut usize,
    descriptor_count: u32,
    catalog_executor_count: u32,
    target: ProcessExecutorTarget,
    identities: ProcessExecutorIdentities,
) -> RusticolResult<ProcessDirectExecutorPack> {
    if descriptor_count > catalog_executor_count {
        return Err(RusticolError::integrity(
            "recurrence process-executor descriptor count is outside its catalog domain",
        ));
    }
    let capacity = usize::try_from(descriptor_count)
        .map_err(|_| RusticolError::artifact("recurrence process-executor count exceeds usize"))?;
    let mut descriptors = Vec::with_capacity(capacity);
    let mut previous_id = None;
    for _ in 0..descriptor_count {
        let record_start = *cursor;
        let record_size = usize::try_from(read_u32(bytes, cursor, "executor record size")?)
            .map_err(|_| RusticolError::artifact("executor record size exceeds usize"))?;
        if record_size < 16 {
            return Err(RusticolError::artifact(
                "recurrence process-executor record is shorter than its fixed header",
            ));
        }
        let record_end = record_start
            .checked_add(record_size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| RusticolError::artifact("truncated process-executor record"))?;
        let executor_id = read_u32(bytes, cursor, "executor ID")?;
        if executor_id >= catalog_executor_count
            || previous_id.is_some_and(|previous| executor_id <= previous)
        {
            return Err(RusticolError::integrity(
                "recurrence process-executor IDs are not sorted, unique, and in range",
            ));
        }
        previous_id = Some(executor_id);
        let role =
            DirectExecutorRole::try_from(u16::from(read_u8(bytes, cursor, "executor role")?))
                .map_err(|_| RusticolError::compatibility("unsupported process-executor role"))?;
        let binding_tag = read_u8(bytes, cursor, "executor binding tag")?;
        let flags = read_u16(bytes, cursor, "executor flags")?;
        if flags & !PROCESS_EXECUTOR_FLAGS_KNOWN != 0 {
            return Err(RusticolError::compatibility(
                "recurrence process executor has unknown flags",
            ));
        }
        let destination_component_count =
            read_u32(bytes, cursor, "executor destination component count")?;
        if destination_component_count == 0 {
            return Err(RusticolError::artifact(
                "recurrence process executor has an empty destination shape",
            ));
        }
        let binding = match binding_tag {
            0 => ProcessDirectExecutorBinding::Source,
            1 => ProcessDirectExecutorBinding::Intrinsic(decode_intrinsic(bytes, cursor)?),
            2 => ProcessDirectExecutorBinding::Jit(decode_jit(bytes, cursor)?),
            3 => ProcessDirectExecutorBinding::Native(decode_native(bytes, cursor)?),
            _ => {
                return Err(RusticolError::compatibility(format!(
                    "unsupported process-executor binding tag {binding_tag}"
                )));
            }
        };
        if *cursor != record_end {
            return Err(RusticolError::integrity(
                "recurrence process-executor record has trailing bytes",
            ));
        }
        validate_binding_policy(role, &binding, target.backend)?;
        descriptors.push(ProcessDirectExecutorDescriptor {
            direct_executor_id: executor_id,
            role,
            destination_component_count,
            uses_exact_factor: flags & PROCESS_EXECUTOR_FLAG_USES_EXACT_FACTOR != 0,
            binding,
        });
    }
    Ok(ProcessDirectExecutorPack {
        identities,
        target,
        catalog_executor_count,
        descriptors,
    })
}

fn validate_binding_policy(
    role: DirectExecutorRole,
    binding: &ProcessDirectExecutorBinding,
    backend: ProcessExecutorBackend,
) -> RusticolResult<()> {
    let valid = match binding {
        ProcessDirectExecutorBinding::Source => role == DirectExecutorRole::Source,
        ProcessDirectExecutorBinding::Intrinsic(intrinsic) => {
            role != DirectExecutorRole::Source
                && (role == DirectExecutorRole::Contribution) == intrinsic.scale.is_some()
        }
        ProcessDirectExecutorBinding::Jit(jit) => {
            role != DirectExecutorRole::Source
                && backend == ProcessExecutorBackend::Jit
                && jit.optimization_level == PREPARED_JIT_OPTIMIZATION_LEVEL
        }
        ProcessDirectExecutorBinding::Native(_) => {
            role != DirectExecutorRole::Source
                && matches!(
                    backend,
                    ProcessExecutorBackend::Cpp | ProcessExecutorBackend::Asm
                )
        }
    };
    if !valid {
        return Err(RusticolError::integrity(
            "recurrence process-executor binding does not match its role/backend",
        ));
    }
    Ok(())
}

fn decode_intrinsic(bytes: &[u8], cursor: &mut usize) -> RusticolResult<ProcessIntrinsicExecutor> {
    let has_scale = read_bool(bytes, cursor, "intrinsic scale flag")?;
    require_zero(bytes, cursor, 3, "intrinsic reserved bytes")?;
    let constant_real_bits = read_u64(bytes, cursor, "intrinsic real bits")?;
    let constant_imag_bits = read_u64(bytes, cursor, "intrinsic imaginary bits")?;
    let parameter = read_u32(bytes, cursor, "intrinsic parameter index")?;
    let runtime_template = read_string(bytes, cursor, "intrinsic runtime template")?;
    let scale = if has_scale {
        Some(ProcessIntrinsicScale {
            constant_real_bits,
            constant_imag_bits,
            parameter_index: (parameter != DIRECT_NONE_U32).then_some(parameter),
        })
    } else {
        if constant_real_bits != 0 || constant_imag_bits != 0 || parameter != DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "non-scale intrinsic has noncanonical scale fields",
            ));
        }
        None
    };
    Ok(ProcessIntrinsicExecutor {
        runtime_template,
        scale,
    })
}

fn decode_jit(bytes: &[u8], cursor: &mut usize) -> RusticolResult<ProcessJitExecutor> {
    let prepared_kernel_id = read_u32(bytes, cursor, "JIT prepared-kernel ID")?;
    if prepared_kernel_id == DIRECT_NONE_U32 {
        return Err(RusticolError::artifact(
            "JIT process executor uses the missing kernel sentinel",
        ));
    }
    let optimization_level = read_u32(bytes, cursor, "JIT optimization level")?;
    let plane_compression = read_bool(bytes, cursor, "JIT compression flag")?;
    require_zero(bytes, cursor, 3, "JIT reserved bytes")?;
    let source_application_sha256 = read_array::<32>(bytes, cursor, "JIT source digest")?;
    let source_application_path = read_string(bytes, cursor, "JIT source path")?;
    let source_application_abi = read_string(bytes, cursor, "JIT source ABI")?;
    let parameter_count = read_u32(bytes, cursor, "JIT parameter-binding count")?;
    let plane_count = read_u32(bytes, cursor, "JIT plane-projection count")?;
    let scalar_count = read_u32(bytes, cursor, "JIT scalar-projection count")?;
    let alias_count = read_u32(bytes, cursor, "JIT output-alias count")?;
    let parameter_bindings = (0..parameter_count)
        .map(|_| decode_parameter_binding(bytes, cursor))
        .collect::<RusticolResult<Vec<_>>>()?;
    let input_plane_projections = (0..plane_count)
        .map(|_| decode_plane_projection(bytes, cursor))
        .collect::<RusticolResult<Vec<_>>>()?;
    let scalar_projections = (0..scalar_count)
        .map(|_| decode_scalar_projection(bytes, cursor))
        .collect::<RusticolResult<Vec<_>>>()?;
    let output_alias_inputs = (0..alias_count)
        .map(|_| read_u32(bytes, cursor, "JIT output-alias input"))
        .collect::<RusticolResult<Vec<_>>>()?;
    for binding in &parameter_bindings {
        let (index, limit) = match *binding {
            RecurrenceDirectParameterBindingManifest::Plane { index } => (index, plane_count),
            RecurrenceDirectParameterBindingManifest::Scalar { index } => (index, scalar_count),
        };
        if index >= limit {
            return Err(RusticolError::integrity(
                "JIT process executor parameter binding is out of bounds",
            ));
        }
    }
    if output_alias_inputs
        .iter()
        .any(|index| *index >= plane_count)
    {
        return Err(RusticolError::integrity(
            "JIT process executor output alias is out of bounds",
        ));
    }
    Ok(ProcessJitExecutor {
        optimization_level,
        plane_compression,
        source_application_sha256,
        source_application_path,
        source_application_abi,
        parameter_bindings,
        input_plane_projections,
        scalar_projections,
        output_alias_inputs,
    })
}

fn decode_native(bytes: &[u8], cursor: &mut usize) -> RusticolResult<ProcessNativeExecutor> {
    let prepared_kernel_id = read_u32(bytes, cursor, "native prepared-kernel ID")?;
    if prepared_kernel_id == DIRECT_NONE_U32 {
        return Err(RusticolError::artifact(
            "native process executor uses the missing kernel sentinel",
        ));
    }
    let has_coupling = read_bool(bytes, cursor, "native coupling flag")?;
    require_zero(bytes, cursor, 3, "native reserved bytes")?;
    let coupling_real = f64::from_bits(read_u64(bytes, cursor, "native coupling real bits")?);
    let coupling_imag = f64::from_bits(read_u64(bytes, cursor, "native coupling imaginary bits")?);
    let coupling = if has_coupling {
        if !coupling_real.is_finite() || !coupling_imag.is_finite() {
            return Err(RusticolError::artifact(
                "native process executor coupling is not finite",
            ));
        }
        Some((coupling_real, coupling_imag))
    } else {
        if coupling_real.to_bits() != 0 || coupling_imag.to_bits() != 0 {
            return Err(RusticolError::integrity(
                "native process executor has noncanonical absent coupling bits",
            ));
        }
        None
    };
    Ok(ProcessNativeExecutor {
        prepared_kernel_id,
        library_path: read_string(bytes, cursor, "native library path")?,
        native_entry_point: read_string(bytes, cursor, "native entry point")?,
        coupling,
    })
}

fn decode_parameter_binding(
    bytes: &[u8],
    cursor: &mut usize,
) -> RusticolResult<RecurrenceDirectParameterBindingManifest> {
    let tag = read_u8(bytes, cursor, "JIT parameter-binding tag")?;
    require_zero(bytes, cursor, 3, "JIT parameter-binding reserved bytes")?;
    let index = read_u32(bytes, cursor, "JIT parameter-binding index")?;
    match tag {
        0 => Ok(RecurrenceDirectParameterBindingManifest::Plane { index }),
        1 => Ok(RecurrenceDirectParameterBindingManifest::Scalar { index }),
        _ => Err(RusticolError::compatibility(
            "unsupported JIT parameter-binding tag",
        )),
    }
}

fn decode_plane_projection(
    bytes: &[u8],
    cursor: &mut usize,
) -> RusticolResult<RecurrenceDirectPlaneProjectionManifest> {
    let tag = read_u8(bytes, cursor, "JIT plane-projection tag")?;
    let operand = read_u8(bytes, cursor, "JIT plane-projection operand")?;
    let component = read_u16(bytes, cursor, "JIT plane-projection component")?;
    let imaginary = read_bool(bytes, cursor, "JIT plane-projection imaginary flag")?;
    require_zero(bytes, cursor, 3, "JIT plane-projection reserved bytes")?;
    match tag {
        0 => Ok(RecurrenceDirectPlaneProjectionManifest::ParentCurrent {
            parent: operand,
            component,
            imaginary,
        }),
        1 if !imaginary => Ok(RecurrenceDirectPlaneProjectionManifest::Momentum {
            operand,
            lorentz_component: component,
        }),
        2 if operand == 0 => Ok(
            RecurrenceDirectPlaneProjectionManifest::DestinationCurrent {
                component,
                imaginary,
            },
        ),
        3 if operand == 0 => Ok(
            RecurrenceDirectPlaneProjectionManifest::DestinationAmplitude {
                component,
                imaginary,
            },
        ),
        _ => Err(RusticolError::integrity(
            "JIT plane-projection record is noncanonical",
        )),
    }
}

fn decode_scalar_projection(
    bytes: &[u8],
    cursor: &mut usize,
) -> RusticolResult<RecurrenceDirectScalarProjectionManifest> {
    let tag = read_u8(bytes, cursor, "JIT scalar-projection tag")?;
    let imaginary = read_bool(bytes, cursor, "JIT scalar-projection imaginary flag")?;
    if read_u16(bytes, cursor, "JIT scalar-projection reserved bytes")? != 0 {
        return Err(RusticolError::integrity(
            "JIT scalar-projection reserved bytes are nonzero",
        ));
    }
    let index = read_u32(bytes, cursor, "JIT scalar-projection index")?;
    let bits = read_u64(bytes, cursor, "JIT scalar-projection bits")?;
    match tag {
        0 if index == 0 && bits == 0 => {
            Ok(RecurrenceDirectScalarProjectionManifest::ExactFactor { imaginary })
        }
        1 if bits == 0 => {
            Ok(RecurrenceDirectScalarProjectionManifest::Parameter { index, imaginary })
        }
        2 if !imaginary && index == 0 => {
            let value = f64::from_bits(bits);
            if !value.is_finite() {
                return Err(RusticolError::artifact(
                    "JIT literal scalar projection is not finite",
                ));
            }
            Ok(RecurrenceDirectScalarProjectionManifest::Literal { value })
        }
        _ => Err(RusticolError::integrity(
            "JIT scalar-projection record is noncanonical",
        )),
    }
}

pub(super) fn semantic_digest_from_bytes(
    bytes: [u8; 32],
    context: &str,
) -> RusticolResult<SemanticDigest> {
    SemanticDigest::new(bytes)
        .map_err(|_| RusticolError::artifact(format!("{context} digest must not be all zero")))
}

pub(super) fn read_u8(bytes: &[u8], cursor: &mut usize, context: &str) -> RusticolResult<u8> {
    let value = *bytes
        .get(*cursor)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context}")))?;
    *cursor += 1;
    Ok(value)
}

pub(super) fn read_u16(bytes: &[u8], cursor: &mut usize, context: &str) -> RusticolResult<u16> {
    Ok(u16::from_le_bytes(read_array(bytes, cursor, context)?))
}

pub(super) fn read_u32(bytes: &[u8], cursor: &mut usize, context: &str) -> RusticolResult<u32> {
    Ok(u32::from_le_bytes(read_array(bytes, cursor, context)?))
}

pub(super) fn read_u64(bytes: &[u8], cursor: &mut usize, context: &str) -> RusticolResult<u64> {
    Ok(u64::from_le_bytes(read_array(bytes, cursor, context)?))
}

pub(super) fn read_array<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
    context: &str,
) -> RusticolResult<[u8; N]> {
    let end = cursor
        .checked_add(N)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range overflows")))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context}")))?
        .try_into()
        .map_err(|_| RusticolError::internal("fixed-width process-pack read drifted"))?;
    *cursor = end;
    Ok(value)
}

pub(super) fn read_string(
    bytes: &[u8],
    cursor: &mut usize,
    context: &str,
) -> RusticolResult<String> {
    let length = usize::try_from(read_u32(bytes, cursor, context)?)
        .map_err(|_| RusticolError::artifact(format!("{context} length exceeds usize")))?;
    if length == 0 {
        return Err(RusticolError::artifact(format!("{context} is empty")));
    }
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range overflows")))?;
    let raw = bytes
        .get(*cursor..end)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context}")))?;
    let value = std::str::from_utf8(raw)
        .map_err(|_| RusticolError::artifact(format!("{context} is not UTF-8")))?
        .to_owned();
    *cursor = end;
    Ok(value)
}

fn read_bool(bytes: &[u8], cursor: &mut usize, context: &str) -> RusticolResult<bool> {
    match read_u8(bytes, cursor, context)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(RusticolError::artifact(format!(
            "{context} is not a canonical boolean"
        ))),
    }
}

fn require_zero(
    bytes: &[u8],
    cursor: &mut usize,
    count: usize,
    context: &str,
) -> RusticolResult<()> {
    let end = cursor
        .checked_add(count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range overflows")))?;
    if bytes
        .get(*cursor..end)
        .is_none_or(|reserved| reserved.iter().any(|byte| *byte != 0))
    {
        return Err(RusticolError::integrity(format!(
            "{context} are nonzero or truncated"
        )));
    }
    *cursor = end;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{valid_direct_plan_fixture, valid_direct_plan_parts_fixture};

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn descriptor(
        direct_executor_id: u32,
        role: DirectExecutorRole,
    ) -> ProcessDirectExecutorDescriptor {
        let binding = match role {
            DirectExecutorRole::Source => ProcessDirectExecutorBinding::Source,
            DirectExecutorRole::Contribution => {
                ProcessDirectExecutorBinding::Intrinsic(ProcessIntrinsicExecutor {
                    runtime_template: "rusticol.test-contribution.v1".to_owned(),
                    scale: Some(ProcessIntrinsicScale {
                        constant_real_bits: 1.0_f64.to_bits(),
                        constant_imag_bits: 0,
                        parameter_index: None,
                    }),
                })
            }
            DirectExecutorRole::Finalization | DirectExecutorRole::Closure => {
                ProcessDirectExecutorBinding::Intrinsic(ProcessIntrinsicExecutor {
                    runtime_template: "rusticol.test-noncontribution.v1".to_owned(),
                    scale: None,
                })
            }
        };
        ProcessDirectExecutorDescriptor {
            direct_executor_id,
            role,
            destination_component_count: 1,
            uses_exact_factor: role == DirectExecutorRole::Contribution,
            binding,
        }
    }

    fn pack_for(plan: &DirectRecurrencePlan) -> ProcessDirectExecutorPack {
        let descriptors = plan
            .row_groups()
            .iter()
            .filter(|group| group.direct_executor_id != DIRECT_NONE_U32)
            .map(|group| {
                (
                    group.direct_executor_id,
                    descriptor(group.direct_executor_id, group.role),
                )
            })
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect();
        ProcessDirectExecutorPack {
            identities: ProcessExecutorIdentities {
                compiled_model_digest: digest(0x44),
                recurrence_template_catalog_digest: digest(0x55),
                prepared_kernel_pack_digest: plan.prepared_pack_digest(),
                direct_template_catalog_digest: plan.direct_template_catalog_digest(),
                runtime_layout_digest: plan.runtime_layout_digest(),
            },
            target: ProcessExecutorTarget {
                backend: ProcessExecutorBackend::Jit,
                target_triple: PREPARED_JIT_PORTABLE_TARGET.to_owned(),
                portable: true,
                cpu_features: Vec::new(),
            },
            catalog_executor_count: plan.direct_executor_count(),
            descriptors,
        }
    }

    fn validate(
        pack: &ProcessDirectExecutorPack,
        plan: &DirectRecurrencePlan,
    ) -> RusticolResult<()> {
        pack.validate_for_plan(
            plan,
            plan.prepared_pack_digest(),
            plan.direct_template_catalog_digest(),
            plan.runtime_layout_digest(),
        )
    }

    #[test]
    fn required_executor_set_is_exact_and_role_typed() {
        let plan = valid_direct_plan_fixture();
        let pack = pack_for(&plan);
        validate(&pack, &plan).unwrap();

        let mut missing = pack.clone();
        missing.descriptors.pop();
        assert!(
            validate(&missing, &plan)
                .unwrap_err()
                .to_string()
                .contains("missing or extra")
        );

        let mut duplicate = pack.clone();
        duplicate.descriptors.push(duplicate.descriptors[0].clone());
        assert!(
            validate(&duplicate, &plan)
                .unwrap_err()
                .to_string()
                .contains("repeats an executor ID")
        );

        let mut wrong_role = pack.clone();
        wrong_role.descriptors[0].role = DirectExecutorRole::Closure;
        assert!(
            validate(&wrong_role, &plan)
                .unwrap_err()
                .to_string()
                .contains("wrong role")
        );

        let mut out_of_range = pack;
        out_of_range.descriptors[0].direct_executor_id = plan.direct_executor_count();
        assert!(
            validate(&out_of_range, &plan)
                .unwrap_err()
                .to_string()
                .contains("out-of-range")
        );
    }

    #[test]
    fn required_executor_set_rejects_an_extra_in_range_record() {
        let mut parts = valid_direct_plan_parts_fixture();
        parts.direct_executor_count = 5;
        let plan = DirectRecurrencePlan::new(parts).unwrap();
        let mut pack = pack_for(&plan);
        pack.descriptors
            .push(descriptor(4, DirectExecutorRole::Closure));

        assert!(
            validate(&pack, &plan)
                .unwrap_err()
                .to_string()
                .contains("missing or extra")
        );
    }

    #[test]
    fn decodes_python_source_descriptor_golden() {
        // Python struct.pack("<IIBBHI", 16, 2, 0, 0, 0, 4).
        let bytes = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0];
        let mut cursor = 0;
        let decoded = decode_process_executor_descriptors(
            &bytes,
            &mut cursor,
            1,
            4,
            ProcessExecutorTarget {
                backend: ProcessExecutorBackend::Jit,
                target_triple: PREPARED_JIT_PORTABLE_TARGET.to_owned(),
                portable: true,
                cpu_features: Vec::new(),
            },
            ProcessExecutorIdentities {
                compiled_model_digest: digest(1),
                recurrence_template_catalog_digest: digest(2),
                prepared_kernel_pack_digest: digest(3),
                direct_template_catalog_digest: digest(4),
                runtime_layout_digest: digest(5),
            },
        )
        .unwrap();

        assert_eq!(cursor, bytes.len());
        assert_eq!(decoded.catalog_executor_count, 4);
        assert_eq!(decoded.descriptors.len(), 1);
        assert_eq!(decoded.descriptors[0].direct_executor_id, 2);
        assert_eq!(decoded.descriptors[0].role, DirectExecutorRole::Source);
        assert_eq!(decoded.descriptors[0].destination_component_count, 4);
        assert!(matches!(
            decoded.descriptors[0].binding,
            ProcessDirectExecutorBinding::Source
        ));
    }
}
