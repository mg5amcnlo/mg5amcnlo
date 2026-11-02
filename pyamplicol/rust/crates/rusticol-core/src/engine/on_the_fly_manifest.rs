// SPDX-License-Identifier: 0BSD

//! Compact execution-manifest validation for the private on-the-fly lane.
//!
//! The verified outer artifact owns file sizes and digests, while PACBIN owns
//! its member index and payload digests. This manifest therefore records only
//! the identities and canonical paths needed to select the lane.

use super::recurrence_manifest::{
    RecurrenceColorContractionReference, RecurrenceExternalLeg, RecurrenceNormalization,
    RecurrenceParameterProjection, RecurrenceParticleMass, RecurrenceRuntimeParameter,
};
use super::{
    ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY, ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY, SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
    confined_internal_path,
};
use crate::recurrence::RECURRENCE_COLOR_CONTRACTION_CODEC_ABI;
use crate::recurrence::on_the_fly::{
    ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI, OnTheFlyProcessSeedIdentityV1,
};
use crate::{ArtifactProcess, PROCESS_ARTIFACT_SCHEMA_VERSION, RusticolError, RusticolResult};
use serde::Deserialize;
use std::collections::BTreeSet;

pub(super) const ON_THE_FLY_EXECUTION_KIND: &str = "pyamplicol-runtime-on-the-fly-execution";
pub(super) const ON_THE_FLY_RUNTIME_CONTAINER_KIND: &str =
    "pyamplicol-on-the-fly-runtime-container";
pub(super) const ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA: u16 = 1;
pub(super) const ON_THE_FLY_RUNTIME_STORAGE_ABI: &str = "pacbin-v1";
pub(super) const ON_THE_FLY_RUNTIME_CONTAINER_PATH: &str = "on-the-fly-runtime.pacbin";
pub(super) const ON_THE_FLY_PROCESS_SEED_MEMBER: &str = "on-the-fly/process-seed-v1.bin";
pub(super) const ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH: &str = "model/eager-kernel-pack.json";
pub(super) const ON_THE_FLY_KERNEL_PAYLOAD_ROOT: &str = "model/eager-kernels";
pub(super) const ON_THE_FLY_COLOR_CONTRACTION_PATH: &str = "on-the-fly-color.bin";
#[expect(
    dead_code,
    reason = "retained to reject legacy v1 companion manifests explicitly"
)]
pub(super) const RECURRENCE_HELICITY_SELECTOR_COMPANION_KIND: &str =
    "pyamplicol-recurrence-helicity-selector-companion-v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) kernel_pack: OnTheFlyKernelPackReference,
    pub(super) runtime_options: OnTheFlyRuntimeOptions,
    pub(super) selector_policy: OnTheFlySelectorPolicy,
    pub(super) runtime_metadata: OnTheFlyRuntimeMetadata,
    pub(super) runtime_container: OnTheFlyRuntimeContainer,
}

/// Irreducible process/runtime state shared with the established recurrence
/// lane. Dense public color-flow catalogs and source-template replicas do not
/// belong to an on-the-fly artifact: selectors and source dispatch are derived
/// from the compact seed. NLC/full add only a bounded reference to their loose
/// authenticated color-contraction payload.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyRuntimeMetadata {
    pub(super) runtime_parameters: Vec<RecurrenceRuntimeParameter>,
    pub(super) prepared_parameter_defaults: Vec<[f64; 2]>,
    pub(super) parameter_projection: Vec<RecurrenceParameterProjection>,
    pub(super) external_legs: Vec<RecurrenceExternalLeg>,
    pub(super) particle_masses: Vec<RecurrenceParticleMass>,
    pub(super) normalization: RecurrenceNormalization,
    pub(super) process_seed_identity: OnTheFlyProcessSeedIdentityV1,
    #[serde(default)]
    pub(super) color_contraction: Option<RecurrenceColorContractionReference>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyKernelPackReference {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyRuntimeOptions {
    pub(super) point_tile_size: u32,
    pub(super) query_construction_threads: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum OnTheFlyColorCoverage {
    Complete,
    Contracted,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum OnTheFlyColorBasis {
    #[default]
    Trace,
    Adjoint,
}

/// Compact facts that affect the internal structural color basis but cannot be
/// reconstructed from external color roles alone.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlySelectorPolicy {
    pub(super) color_coverage: OnTheFlyColorCoverage,
    #[serde(default)]
    pub(super) color_basis: OnTheFlyColorBasis,
    #[serde(default)]
    pub(super) reference_color_word: Option<Vec<u32>>,
    pub(super) trace_reflections_folded: bool,
    pub(super) selector_census: OnTheFlySelectorCensus,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlySelectorCensus {
    pub(super) physical_helicity_count: u64,
    pub(super) physical_color_flow_count: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyRuntimeContainer {
    pub(super) kind: String,
    pub(super) schema_version: u16,
    pub(super) storage_abi: String,
    pub(super) path: String,
    pub(super) seed_member_path: String,
}

/// Lane-only OTF state embedded by a complete contracted recurrence artifact.
///
/// Runtime parameters, external legs, normalization, and point tiling remain
/// owned by the enclosing recurrence manifest.  The companion retains only
/// the compact seed identity and policy needed to construct a selected query
/// family plus its independent one-component color reducer.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    dead_code,
    reason = "retained to reject legacy v1 companion manifests explicitly"
)]
pub(super) struct OnTheFlyHelicitySelectorCompanionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) process_digest: String,
    pub(super) process_seed_identity: OnTheFlyProcessSeedIdentityV1,
    pub(super) query_construction_threads: u32,
    pub(super) selector_policy: OnTheFlySelectorPolicy,
    pub(super) runtime_container: OnTheFlyRuntimeContainer,
    pub(super) color_contraction: RecurrenceColorContractionReference,
}

impl OnTheFlyHelicitySelectorCompanionManifest {
    #[expect(
        dead_code,
        reason = "retained to reject legacy v1 companion manifests explicitly"
    )]
    pub(super) fn validate(
        &self,
        process_digest: Option<&str>,
        color_accuracy: &str,
    ) -> RusticolResult<()> {
        if self.schema_version != 1 || self.kind != RECURRENCE_HELICITY_SELECTOR_COMPANION_KIND {
            return Err(RusticolError::compatibility(
                "unsupported recurrence helicity-selector companion; regenerate the artifact",
            ));
        }
        if process_digest != Some(self.process_digest.as_str())
            || self.process_seed_identity.process_digest != self.process_digest
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector companion disagrees with the process digest",
            ));
        }
        if self.query_construction_threads == 0 {
            return Err(RusticolError::artifact(
                "recurrence helicity-selector companion query thread count must be positive",
            ));
        }
        if !matches!(color_accuracy, "nlc" | "full") {
            return Err(RusticolError::integrity(
                "only contracted recurrence may carry a helicity-selector companion",
            ));
        }
        self.selector_policy.validate(color_accuracy)?;
        self.runtime_container.validate()?;
        validate_contracted_color_reference(&self.color_contraction, color_accuracy)?;
        if self.color_contraction.component_count != 1 {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector companion color payload must have one component",
            ));
        }
        Ok(())
    }
}

impl OnTheFlyExecutionManifest {
    fn validate(&self, outer: &ArtifactProcess) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != ON_THE_FLY_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported on-the-fly execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process != outer.expression
            || self.key != outer.id
            || self.color_accuracy != outer.color_accuracy
            || self.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "on-the-fly execution manifest does not match outer process {:?}",
                outer.id
            )));
        }
        let actual = self
            .required_runtime_capabilities
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let color_capability = match self.color_accuracy.as_str() {
            "lc" => ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            "nlc" | "full" => ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            _ => {
                return Err(RusticolError::compatibility(format!(
                    "unsupported on-the-fly color accuracy {:?}",
                    self.color_accuracy
                )));
            }
        };
        let uses_symmetric_group_fft = self
            .runtime_metadata
            .color_contraction
            .as_ref()
            .and_then(|reference| reference.factorization.as_ref())
            .is_some_and(|factorization| factorization.kind == "symmetric-group-fourier");
        let mut expected = BTreeSet::from([ON_THE_FLY_RUNTIME_CAPABILITY, color_capability]);
        if uses_symmetric_group_fft {
            expected.insert(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY);
        }
        if actual.len() != self.required_runtime_capabilities.len() || actual != expected {
            return Err(RusticolError::integrity(format!(
                "on-the-fly execution must require exactly {expected:?}"
            )));
        }
        if self.kernel_pack.manifest_path != ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH
            || self.kernel_pack.payload_root != ON_THE_FLY_KERNEL_PAYLOAD_ROOT
        {
            return Err(RusticolError::security(
                "on-the-fly prepared-kernel paths are not canonical",
            ));
        }
        if self.runtime_options.point_tile_size == 0 {
            return Err(RusticolError::artifact(
                "on-the-fly point_tile_size must be positive",
            ));
        }
        if self.runtime_options.query_construction_threads == 0 {
            return Err(RusticolError::artifact(
                "on-the-fly query_construction_threads must be positive",
            ));
        }
        self.selector_policy.validate(&self.color_accuracy)?;
        self.runtime_metadata
            .validate(&self.external_pdg_order, &self.color_accuracy)?;
        self.runtime_container.validate()
    }

    pub(super) fn uses_contracted_color(&self) -> bool {
        matches!(self.color_accuracy.as_str(), "nlc" | "full")
    }
}

impl OnTheFlySelectorPolicy {
    pub(super) fn validate(&self, color_accuracy: &str) -> RusticolResult<()> {
        if let Some(word) = &self.reference_color_word
            && (word.is_empty()
                || word.contains(&0)
                || word.iter().copied().collect::<BTreeSet<_>>().len() != word.len())
        {
            return Err(RusticolError::artifact(
                "on-the-fly reference color word must contain unique positive external labels",
            ));
        }
        self.selector_census.validate()?;
        let coverage_matches = matches!(
            (color_accuracy, self.color_coverage),
            ("lc", OnTheFlyColorCoverage::Complete)
                | ("nlc" | "full", OnTheFlyColorCoverage::Contracted)
        );
        if !coverage_matches
            || self.color_basis == OnTheFlyColorBasis::Adjoint
                && self.color_coverage != OnTheFlyColorCoverage::Contracted
            || self.color_coverage == OnTheFlyColorCoverage::Contracted
                && (self.trace_reflections_folded
                    || self.selector_census.physical_color_flow_count != 1)
        {
            return Err(RusticolError::integrity(
                "on-the-fly selector policy disagrees with its color accuracy",
            ));
        }
        Ok(())
    }
}

impl OnTheFlySelectorCensus {
    fn validate(&self) -> RusticolResult<()> {
        if self.physical_helicity_count == 0 || self.physical_color_flow_count == 0 {
            return Err(RusticolError::artifact(
                "on-the-fly selector census counts must be positive",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_against(
        &self,
        physical_helicity_count: usize,
        physical_color_flow_count: usize,
    ) -> RusticolResult<()> {
        let helicities = u64::try_from(physical_helicity_count).map_err(|_| {
            RusticolError::artifact("on-the-fly physical helicity count exceeds u64")
        })?;
        let color_flows = u64::try_from(physical_color_flow_count).map_err(|_| {
            RusticolError::artifact("on-the-fly physical color-flow count exceeds u64")
        })?;
        if self.physical_helicity_count != helicities
            || self.physical_color_flow_count != color_flows
        {
            return Err(RusticolError::integrity(
                "on-the-fly selector census disagrees with the authenticated compact seed",
            ));
        }
        Ok(())
    }
}

impl OnTheFlyRuntimeMetadata {
    fn validate(&self, external_pdgs: &[i32], color_accuracy: &str) -> RusticolResult<()> {
        for (context, len) in [
            ("runtime parameters", self.runtime_parameters.len()),
            (
                "prepared parameter defaults",
                self.prepared_parameter_defaults.len(),
            ),
            ("parameter projection", self.parameter_projection.len()),
            ("external legs", self.external_legs.len()),
            ("particle masses", self.particle_masses.len()),
        ] {
            u32::try_from(len).map_err(|_| {
                RusticolError::artifact(format!(
                    "on-the-fly {context} count exceeds the u32 identifier domain"
                ))
            })?;
        }
        if self
            .prepared_parameter_defaults
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        {
            return Err(RusticolError::artifact(
                "on-the-fly prepared-parameter defaults must be finite complex-f64 values",
            ));
        }
        if self.runtime_parameters.len() != self.parameter_projection.len() {
            return Err(RusticolError::integrity(
                "on-the-fly runtime parameters do not cover the parameter projection",
            ));
        }
        let mut previous_projection_key: Option<(&str, u32)> = None;
        let mut current_name: Option<&str> = None;
        let mut current_template = None;
        let mut current_prepared = None;
        for (runtime_slot, row) in self.parameter_projection.iter().enumerate() {
            if row.runtime_slot as usize != runtime_slot
                || row.runtime_name.is_empty()
                || row.component > 1
                || row.parameter_template_id as usize >= self.prepared_parameter_defaults.len()
                || row
                    .prepared_parameter_id
                    .is_some_and(|id| id as usize >= self.prepared_parameter_defaults.len())
            {
                return Err(RusticolError::integrity(
                    "on-the-fly parameter projection is outside its prepared-parameter domain",
                ));
            }
            let key = (row.runtime_name.as_str(), row.component);
            if previous_projection_key.is_some_and(|previous| previous >= key) {
                return Err(RusticolError::artifact(
                    "on-the-fly parameter projection is not in strict name/component order",
                ));
            }
            previous_projection_key = Some(key);
            if current_name == Some(row.runtime_name.as_str()) {
                if row.component != 1
                    || current_template != Some(row.parameter_template_id)
                    || current_prepared != Some(row.prepared_parameter_id)
                {
                    return Err(RusticolError::integrity(
                        "on-the-fly complex parameter projection rows are inconsistent",
                    ));
                }
            } else {
                if row.component != 0 {
                    return Err(RusticolError::artifact(
                        "on-the-fly parameter projection must begin at component zero",
                    ));
                }
                current_name = Some(&row.runtime_name);
                current_template = Some(row.parameter_template_id);
                current_prepared = Some(row.prepared_parameter_id);
            }
        }
        let mut previous_runtime_name: Option<&str> = None;
        let mut previous_kind: Option<&str> = None;
        for (parameter_index, (parameter, projection)) in self
            .runtime_parameters
            .iter()
            .zip(&self.parameter_projection)
            .enumerate()
        {
            if parameter.parameter_index as usize != parameter_index
                || parameter.parameter_index != projection.runtime_slot
                || parameter.name.is_empty()
                || parameter.kind.is_empty()
                || !parameter.default.is_finite()
            {
                return Err(RusticolError::integrity(
                    "on-the-fly runtime parameter identity/default is invalid",
                ));
            }
            match (&parameter.runtime_name, &parameter.complex_component) {
                (Some(runtime_name), Some(component)) => {
                    let expected_component = match projection.component {
                        0 => "real",
                        1 => "imag",
                        _ => unreachable!("projection component was validated"),
                    };
                    if runtime_name.is_empty()
                        || component != expected_component
                        || projection.runtime_name != *runtime_name
                        || parameter.name != format!("{runtime_name}.{component}")
                        || (previous_runtime_name == Some(runtime_name)
                            && previous_kind != Some(parameter.kind.as_str()))
                    {
                        return Err(RusticolError::integrity(
                            "on-the-fly complex runtime parameter disagrees with its projection",
                        ));
                    }
                    previous_runtime_name = Some(runtime_name);
                    previous_kind = Some(&parameter.kind);
                }
                (None, None) => {
                    if projection.component != 0 || parameter.name != projection.runtime_name {
                        return Err(RusticolError::integrity(
                            "on-the-fly real runtime parameter disagrees with its projection",
                        ));
                    }
                    previous_runtime_name = None;
                    previous_kind = None;
                }
                _ => {
                    return Err(RusticolError::artifact(
                        "on-the-fly runtime_name and complex_component must appear together",
                    ));
                }
            }
            if let Some(prepared_id) = projection.prepared_parameter_id
                && parameter.default
                    != self.prepared_parameter_defaults[prepared_id as usize]
                        [projection.component as usize]
            {
                return Err(RusticolError::integrity(
                    "on-the-fly runtime parameter default disagrees with its prepared default",
                ));
            }
        }

        if self.external_legs.len() != external_pdgs.len() {
            return Err(RusticolError::integrity(
                "on-the-fly external legs do not match the execution PDG order",
            ));
        }
        let mut public_labels = BTreeSet::new();
        let mut saw_final = false;
        for (source_slot, (leg, expected_pdg)) in
            self.external_legs.iter().zip(external_pdgs).enumerate()
        {
            if leg.source_slot as usize != source_slot
                || leg.public_label == 0
                || !public_labels.insert(leg.public_label)
                || leg.physical_pdg != *expected_pdg
                || (saw_final && leg.is_initial)
            {
                return Err(RusticolError::integrity(
                    "on-the-fly external-leg order or identity is not canonical",
                ));
            }
            saw_final |= !leg.is_initial;
        }
        let mut previous_mass_pdg = None;
        for row in &self.particle_masses {
            if previous_mass_pdg.is_some_and(|previous| previous >= row.outgoing_pdg)
                || !row.mass.is_finite()
                || row.mass < 0.0
            {
                return Err(RusticolError::artifact(
                    "on-the-fly particle masses are not strictly ordered finite nonnegative values",
                ));
            }
            previous_mass_pdg = Some(row.outgoing_pdg);
        }
        let normalization = &self.normalization;
        if normalization.color_accuracy != color_accuracy
            || [
                normalization.color_factor,
                normalization.average_factor,
                normalization.identical_factor,
                normalization.global_coupling_factor,
            ]
            .iter()
            .any(|value| !value.is_finite())
            || normalization.average_factor <= 0.0
            || normalization.identical_factor <= 0.0
            || normalization.qcd_coupling_power.is_some()
                != normalization.electroweak_coupling_power.is_some()
            || normalization.coupling_policy.is_empty()
        {
            return Err(RusticolError::integrity(
                "on-the-fly normalization metadata is invalid",
            ));
        }
        match (color_accuracy, self.color_contraction.as_ref()) {
            ("lc", None) => {}
            ("nlc" | "full", Some(reference)) => {
                validate_contracted_color_reference(reference, color_accuracy)?;
            }
            ("lc", Some(_)) => {
                return Err(RusticolError::integrity(
                    "LC on-the-fly execution must not carry a color-contraction companion",
                ));
            }
            ("nlc" | "full", None) => {
                return Err(RusticolError::integrity(
                    "NLC/full on-the-fly execution requires a color-contraction companion",
                ));
            }
            _ => {
                return Err(RusticolError::compatibility(
                    "unsupported on-the-fly color accuracy",
                ));
            }
        }
        self.validate_process_seed_identity()?;
        Ok(())
    }

    fn validate_process_seed_identity(&self) -> RusticolResult<()> {
        let identity = &self.process_seed_identity;
        if identity.abi != ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI {
            return Err(RusticolError::compatibility(
                "unsupported on-the-fly process-seed identity ABI; regenerate the artifact",
            ));
        }
        for digest in [
            &identity.process_digest,
            &identity.compiled_model_digest,
            &identity.recurrence_template_catalog_digest,
            &identity.prepared_kernel_pack_digest,
            &identity.recurrence_direct_template_catalog_digest,
            &identity.semantic_digest,
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(RusticolError::integrity(
                    "on-the-fly process-seed identity contains an invalid SHA-256",
                ));
            }
        }
        if identity.external_permutation.len() != self.external_legs.len()
            || identity.external_sources.len() != self.external_legs.len()
        {
            return Err(RusticolError::integrity(
                "on-the-fly process-seed identity has the wrong external domain",
            ));
        }
        let mut permutation = identity.external_permutation.clone();
        permutation.sort_unstable();
        if permutation
            .iter()
            .enumerate()
            .any(|(index, slot)| *slot as usize != index)
        {
            return Err(RusticolError::integrity(
                "on-the-fly process-seed identity permutation is not dense",
            ));
        }
        for (source_slot, (source, leg)) in identity
            .external_sources
            .iter()
            .zip(&self.external_legs)
            .enumerate()
        {
            if source.source_slot as usize != source_slot
                || source.public_label != leg.public_label
                || source.is_initial != leg.is_initial
                || source.states.is_empty()
                || source
                    .states
                    .windows(2)
                    .any(|pair| pair[0].state_index >= pair[1].state_index)
            {
                return Err(RusticolError::integrity(
                    "on-the-fly process-seed source identity disagrees with external legs",
                ));
            }
        }
        Ok(())
    }
}

fn validate_contracted_color_reference(
    reference: &RecurrenceColorContractionReference,
    color_accuracy: &str,
) -> RusticolResult<()> {
    let digest_is_canonical = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    if reference.abi != RECURRENCE_COLOR_CONTRACTION_CODEC_ABI
        || reference.path != ON_THE_FLY_COLOR_CONTRACTION_PATH
    {
        return Err(RusticolError::compatibility(
            "unsupported on-the-fly color-contraction payload contract",
        ));
    }
    if reference.color_accuracy != color_accuracy
        || !matches!(color_accuracy, "nlc" | "full")
        || !matches!(
            reference.storage.as_str(),
            "expanded" | "convolution-kernels"
        )
        || !reference.includes_color_factor
        || reference.component_count != 1
        || reference.group_count == 0
        || reference.group_count != reference.active_sector_count
        || reference.group_count != reference.destination_count
        || reference.sector_count < reference.active_sector_count
        || reference
            .materialized_destination_count
            .is_some_and(|count| count != reference.destination_count)
        || reference.entry_count == 0
        || reference.logical_entry_count != reference.entry_count
        || reference.size_bytes == 0
        || reference.sha256 != reference.semantic_digest
        || !digest_is_canonical(&reference.sha256)
        || !digest_is_canonical(&reference.semantic_digest)
        || [
            reference.group_count,
            reference.sector_count,
            reference.active_sector_count,
            reference.component_count,
            reference.destination_count,
        ]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return Err(RusticolError::integrity(
            "on-the-fly color-contraction summary is inconsistent",
        ));
    }
    let symmetric_group_fft = reference
        .factorization
        .as_ref()
        .is_some_and(|factorization| {
            factorization.kind == "symmetric-group-fourier"
                && (2..=10).contains(&factorization.rank)
                && factorization.coset_count > 0
        });
    if reference.factorization.is_some() != symmetric_group_fft
        || (reference.storage == "convolution-kernels") != symmetric_group_fft
    {
        return Err(RusticolError::integrity(
            "on-the-fly convolution-kernel storage requires symmetric-group Fourier factorization",
        ));
    }
    if symmetric_group_fft {
        let factorization = reference
            .factorization
            .as_ref()
            .expect("validated symmetric-group factorization");
        let provenance = reference.fft_provenance.as_ref().ok_or_else(|| {
            RusticolError::integrity(
                "on-the-fly symmetric-group Fourier summary is missing FFT provenance",
            )
        })?;
        reference.validate_fft_provenance(factorization, provenance)?;
    } else if reference.fft_provenance.is_some() {
        return Err(RusticolError::integrity(
            "on-the-fly non-FFT color summary carries FFT provenance",
        ));
    }
    confined_internal_path(&reference.path, "on-the-fly color-contraction payload")?;
    Ok(())
}

impl OnTheFlyRuntimeContainer {
    pub(super) fn validate(&self) -> RusticolResult<()> {
        if self.kind != ON_THE_FLY_RUNTIME_CONTAINER_KIND
            || self.schema_version != ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA
            || self.storage_abi != ON_THE_FLY_RUNTIME_STORAGE_ABI
        {
            return Err(RusticolError::compatibility(
                "unsupported on-the-fly runtime-container contract",
            ));
        }
        if self.path != ON_THE_FLY_RUNTIME_CONTAINER_PATH
            || self.seed_member_path != ON_THE_FLY_PROCESS_SEED_MEMBER
        {
            return Err(RusticolError::security(
                "on-the-fly runtime container or seed path is not canonical",
            ));
        }
        confined_internal_path(&self.path, "on-the-fly runtime container")?;
        confined_internal_path(&self.seed_member_path, "on-the-fly process seed")?;
        Ok(())
    }
}

pub(super) fn parse_on_the_fly_execution_manifest(
    bytes: &[u8],
    outer: &ArtifactProcess,
) -> RusticolResult<OnTheFlyExecutionManifest> {
    let manifest: OnTheFlyExecutionManifest = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse on-the-fly execution manifest: {error}"
        ))
    })?;
    manifest.validate(outer)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn outer_with_accuracy(color_accuracy: &str) -> ArtifactProcess {
        let color_capability = if color_accuracy == "lc" {
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY
        } else {
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY
        };
        ArtifactProcess {
            id: "d_dbar_to_t_tbar_g_g".into(),
            expression: "d d~ > t t~ g g".into(),
            color_accuracy: color_accuracy.into(),
            external_pdgs: vec![1, -1, 6, -6, 21, 21],
            physics_path: "processes/d_dbar_to_t_tbar_g_g/physics.json".into(),
            required_runtime_capabilities: vec![
                color_capability.into(),
                ON_THE_FLY_RUNTIME_CAPABILITY.into(),
            ],
            aliases: Vec::new(),
        }
    }

    fn outer() -> ArtifactProcess {
        outer_with_accuracy("lc")
    }

    fn manifest() -> Value {
        let outer = outer();
        json!({
            "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "kind": ON_THE_FLY_EXECUTION_KIND,
            "required_runtime_capabilities": [
                ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
                ON_THE_FLY_RUNTIME_CAPABILITY,
            ],
            "process": outer.expression,
            "key": outer.id,
            "color_accuracy": "lc",
            "external_pdg_order": outer.external_pdgs,
            "kernel_pack": {
                "manifest_path": ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH,
                "payload_root": ON_THE_FLY_KERNEL_PAYLOAD_ROOT,
            },
            "runtime_options": {
                "point_tile_size": 64,
                "query_construction_threads": 4
            },
            "selector_policy": {
                "color_coverage": "complete",
                "reference_color_word": null,
                "trace_reflections_folded": false,
                "selector_census": {
                    "physical_helicity_count": 64,
                    "physical_color_flow_count": 6
                }
            },
            "runtime_metadata": {
                "runtime_parameters": [],
                "prepared_parameter_defaults": [],
                "parameter_projection": [],
                "external_legs": [
                    {"source_slot": 0, "public_label": 1, "physical_pdg": 1, "outgoing_pdg": -1, "is_initial": true},
                    {"source_slot": 1, "public_label": 2, "physical_pdg": -1, "outgoing_pdg": 1, "is_initial": true},
                    {"source_slot": 2, "public_label": 3, "physical_pdg": 6, "outgoing_pdg": 6, "is_initial": false},
                    {"source_slot": 3, "public_label": 4, "physical_pdg": -6, "outgoing_pdg": -6, "is_initial": false},
                    {"source_slot": 4, "public_label": 5, "physical_pdg": 21, "outgoing_pdg": 21, "is_initial": false},
                    {"source_slot": 5, "public_label": 6, "physical_pdg": 21, "outgoing_pdg": 21, "is_initial": false}
                ],
                "particle_masses": [],
                "normalization": {
                    "color_accuracy": "lc",
                    "color_factor": 1.0,
                    "average_factor": 1.0,
                    "identical_factor": 1.0,
                    "global_coupling_factor": 1.0,
                    "qcd_coupling_power": 0,
                    "electroweak_coupling_power": 0,
                    "couplings_in_stage_evaluators": true,
                    "coupling_policy": "local"
                },
                "process_seed_identity": {
                    "abi": ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI,
                    "process_digest": "11".repeat(32),
                    "compiled_model_digest": "22".repeat(32),
                    "recurrence_template_catalog_digest": "33".repeat(32),
                    "prepared_kernel_pack_digest": "44".repeat(32),
                    "recurrence_direct_template_catalog_digest": "55".repeat(32),
                    "semantic_digest": "66".repeat(32),
                    "external_permutation": [0, 1, 2, 3, 4, 5],
                    "external_sources": [
                        {"source_slot": 0, "public_label": 1, "is_initial": true, "states": [{"state_index": 0, "public_helicity": -1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 1, "public_label": 2, "is_initial": true, "states": [{"state_index": 0, "public_helicity": 1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 2, "public_label": 3, "is_initial": false, "states": [{"state_index": 0, "public_helicity": -1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 3, "public_label": 4, "is_initial": false, "states": [{"state_index": 0, "public_helicity": 1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 4, "public_label": 5, "is_initial": false, "states": [{"state_index": 0, "public_helicity": -1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 5, "public_label": 6, "is_initial": false, "states": [{"state_index": 0, "public_helicity": 1, "prepared_mass_parameter_slot": null}]}
                    ]
                }
            },
            "runtime_container": {
                "kind": ON_THE_FLY_RUNTIME_CONTAINER_KIND,
                "schema_version": ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA,
                "storage_abi": ON_THE_FLY_RUNTIME_STORAGE_ABI,
                "path": ON_THE_FLY_RUNTIME_CONTAINER_PATH,
                "seed_member_path": ON_THE_FLY_PROCESS_SEED_MEMBER,
            },
        })
    }

    fn parse(value: &Value) -> RusticolResult<OnTheFlyExecutionManifest> {
        parse_on_the_fly_execution_manifest(&serde_json::to_vec(value).unwrap(), &outer())
    }

    fn contracted_manifest(color_accuracy: &str) -> Value {
        let mut value = manifest();
        value["color_accuracy"] = json!(color_accuracy);
        value["required_runtime_capabilities"] = json!([
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        ]);
        value["selector_policy"]["color_coverage"] = json!("contracted");
        value["selector_policy"]["selector_census"]["physical_color_flow_count"] = json!(1);
        value["runtime_metadata"]["normalization"]["color_accuracy"] = json!(color_accuracy);
        value["runtime_metadata"]["color_contraction"] = json!({
            "abi": RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
            "path": ON_THE_FLY_COLOR_CONTRACTION_PATH,
            "size_bytes": 256,
            "sha256": "77".repeat(32),
            "color_accuracy": color_accuracy,
            "storage": "expanded",
            "includes_color_factor": true,
            "group_count": 6,
            "sector_count": 36,
            "active_sector_count": 6,
            "component_count": 1,
            "destination_count": 6,
            "entry_count": 12,
            "logical_entry_count": 12,
            "semantic_digest": "77".repeat(32),
            "factorization": null,
        });
        value
    }

    fn parse_contracted(
        value: &Value,
        color_accuracy: &str,
    ) -> RusticolResult<OnTheFlyExecutionManifest> {
        parse_on_the_fly_execution_manifest(
            &serde_json::to_vec(value).unwrap(),
            &outer_with_accuracy(color_accuracy),
        )
    }

    fn symmetric_group_contracted_manifest(color_accuracy: &str) -> Value {
        let mut value = contracted_manifest(color_accuracy);
        value["required_runtime_capabilities"] = json!([
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
            SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        ]);
        let reference = &mut value["runtime_metadata"]["color_contraction"];
        reference["storage"] = json!("convolution-kernels");
        reference["factorization"] = json!({
            "kind": "symmetric-group-fourier",
            "rank": 2,
            "coset_count": 3,
        });
        reference["fft_provenance"] = json!({
            "method": "symmetric-group-fourier",
            "degree": 2,
            "channel_count": 3,
            "covered_local_group_count": 6,
            "residual_group_count": 0,
            "residual_entry_count": 0,
            "raw_kernel_bytes": 192,
            "transformed_kernel_bytes": 96,
            "capability": SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        });
        value
    }

    #[test]
    fn accepts_only_the_compact_lc_contract() {
        let parsed = parse(&manifest()).unwrap();
        assert_eq!(parsed.runtime_options.point_tile_size, 64);
        assert_eq!(parsed.runtime_options.query_construction_threads, 4);
        assert_eq!(
            parsed.selector_policy.selector_census,
            OnTheFlySelectorCensus {
                physical_helicity_count: 64,
                physical_color_flow_count: 6,
            }
        );
        assert_eq!(
            parsed.runtime_container.seed_member_path,
            ON_THE_FLY_PROCESS_SEED_MEMBER
        );

        let mut nlc = manifest();
        nlc["color_accuracy"] = json!("nlc");
        assert!(parse(&nlc).is_err());

        let mut extra_capability = manifest();
        extra_capability["required_runtime_capabilities"]
            .as_array_mut()
            .unwrap()
            .push(json!("unrelated.capability.v1"));
        assert!(parse(&extra_capability).is_err());

        let mut zero_census = manifest();
        zero_census["selector_policy"]["selector_census"]["physical_helicity_count"] = json!(0);
        assert!(parse(&zero_census).is_err());

        let mut incomplete_census = manifest();
        incomplete_census["selector_policy"]["selector_census"]
            .as_object_mut()
            .unwrap()
            .remove("physical_color_flow_count");
        assert!(parse(&incomplete_census).is_err());

        let mut zero_query_construction_threads = manifest();
        zero_query_construction_threads["runtime_options"]["query_construction_threads"] = json!(0);
        assert!(parse(&zero_query_construction_threads).is_err());
    }

    #[test]
    fn contracted_color_requires_the_canonical_companion_and_public_census() {
        for accuracy in ["nlc", "full"] {
            let value = contracted_manifest(accuracy);
            let parsed = parse_contracted(&value, accuracy).unwrap();
            assert!(parsed.uses_contracted_color());
            assert_eq!(
                parsed
                    .runtime_metadata
                    .color_contraction
                    .as_ref()
                    .unwrap()
                    .path,
                ON_THE_FLY_COLOR_CONTRACTION_PATH
            );

            let mut materialized = value.clone();
            materialized["runtime_metadata"]["color_contraction"]["materialized_destination_count"] =
                json!(6);
            parse_contracted(&materialized, accuracy).unwrap();

            materialized["runtime_metadata"]["color_contraction"]["materialized_destination_count"] =
                json!(5);
            assert!(parse_contracted(&materialized, accuracy).is_err());

            let mut missing = value.clone();
            missing["runtime_metadata"]
                .as_object_mut()
                .unwrap()
                .remove("color_contraction");
            assert!(parse_contracted(&missing, accuracy).is_err());

            let mut lc_capability = value.clone();
            lc_capability["required_runtime_capabilities"][0] =
                json!(ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY);
            assert!(parse_contracted(&lc_capability, accuracy).is_err());

            let mut public_flows = value.clone();
            public_flows["selector_policy"]["selector_census"]["physical_color_flow_count"] =
                json!(6);
            assert!(parse_contracted(&public_flows, accuracy).is_err());

            let mut sparse_destinations = value;
            sparse_destinations["runtime_metadata"]["color_contraction"]["destination_count"] =
                json!(7);
            assert!(parse_contracted(&sparse_destinations, accuracy).is_err());
        }

        let mut lc_with_companion = manifest();
        lc_with_companion["runtime_metadata"]["color_contraction"] =
            contracted_manifest("full")["runtime_metadata"]["color_contraction"].clone();
        assert!(parse(&lc_with_companion).is_err());
    }

    #[test]
    fn symmetric_group_contracted_color_keeps_the_exact_otf_owner_domain() {
        let value = symmetric_group_contracted_manifest("full");
        parse_contracted(&value, "full").unwrap();

        let mut retained_alias_destination = value.clone();
        retained_alias_destination["runtime_metadata"]["color_contraction"]["destination_count"] =
            json!(7);
        assert!(parse_contracted(&retained_alias_destination, "full").is_err());

        let mut missing_capability = value.clone();
        missing_capability["required_runtime_capabilities"] = json!([
            ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        ]);
        assert!(parse_contracted(&missing_capability, "full").is_err());

        let mut stale_transformed_size = value;
        stale_transformed_size["runtime_metadata"]["color_contraction"]["fft_provenance"]["transformed_kernel_bytes"] =
            json!(88);
        assert!(parse_contracted(&stale_transformed_size, "full").is_err());
    }

    #[test]
    fn adjoint_basis_is_explicit_and_limited_to_contracted_color() {
        assert_eq!(
            parse(&manifest()).unwrap().selector_policy.color_basis,
            OnTheFlyColorBasis::Trace
        );
        for accuracy in ["nlc", "full"] {
            let mut value = symmetric_group_contracted_manifest(accuracy);
            value["selector_policy"]["color_basis"] = json!("adjoint");
            assert_eq!(
                parse_contracted(&value, accuracy)
                    .unwrap()
                    .selector_policy
                    .color_basis,
                OnTheFlyColorBasis::Adjoint
            );
            value["selector_policy"]["color_basis"] = json!("unknown");
            assert!(parse_contracted(&value, accuracy).is_err());
        }
        let mut lc = manifest();
        lc["selector_policy"]["color_basis"] = json!("adjoint");
        assert!(parse(&lc).is_err());
    }

    #[test]
    fn selector_census_fails_closed_against_authenticated_adapter_counts() {
        let census = parse(&manifest()).unwrap().selector_policy.selector_census;
        census.validate_against(64, 6).unwrap();
        assert!(census.validate_against(63, 6).is_err());
        assert!(census.validate_against(64, 5).is_err());
    }

    #[test]
    fn rejects_noncanonical_runtime_and_kernel_paths() {
        let mut runtime_escape = manifest();
        runtime_escape["runtime_container"]["seed_member_path"] = json!("../seed.bin");
        assert!(parse(&runtime_escape).is_err());

        let mut kernel_alias = manifest();
        kernel_alias["kernel_pack"]["manifest_path"] = json!("model/other.json");
        assert!(parse(&kernel_alias).is_err());
    }

    #[test]
    fn rejects_unbound_process_seed_identity_metadata() {
        let mut stale_abi = manifest();
        stale_abi["runtime_metadata"]["process_seed_identity"]["abi"] = json!("stale-v0");
        assert!(parse(&stale_abi).is_err());

        let mut changed_anchor = manifest();
        changed_anchor["runtime_metadata"]["process_seed_identity"]["external_sources"][0]["public_label"] =
            json!(9);
        assert!(parse(&changed_anchor).is_err());

        let mut unknown_field = manifest();
        unknown_field["runtime_metadata"]["process_seed_identity"]["opaque"] = json!(true);
        assert!(parse(&unknown_field).is_err());
    }
}
