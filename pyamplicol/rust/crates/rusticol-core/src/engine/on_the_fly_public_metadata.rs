// SPDX-License-Identifier: 0BSD

//! Compact public metadata for the on-the-fly lane.
//!
//! Ordinary evaluation needs only this O(externals + parameters) payload.
//! The complete public helicity axis and LC color axis remain in the compact
//! selector adapter and are materialized only when the established physics
//! metadata API is explicitly requested. NLC/full expose one contracted color.

use super::on_the_fly_manifest::OnTheFlyExecutionManifest;
use super::on_the_fly_selectors::{
    OnTheFlyCompactSelectorAdapterV1, OnTheFlySelectorIntrospectionCacheV1,
};
use crate::{
    ArtifactProcess, ColorAccuracy, ColorComponent, ContractedColor, Coverage, ExternalParticle,
    Helicity, LcColorFlow, ModelParameter, ParameterKind, ProcessPhysics as ProcessPhysicsV1,
    Reduction, ReductionKind, RusticolError, RusticolResult, SelectorCapabilities,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const ON_THE_FLY_PUBLIC_METADATA_KIND: &str = "pyamplicol-on-the-fly-public-metadata";
const ON_THE_FLY_PUBLIC_METADATA_SCHEMA: u32 = 1;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyPublicMetadataV1 {
    schema_version: u32,
    kind: String,
    process_id: String,
    process: String,
    color_accuracy: ColorAccuracy,
    external_particles: Vec<ExternalParticle>,
    model_parameters: Vec<ModelParameter>,
}

impl OnTheFlyPublicMetadataV1 {
    #[cfg(test)]
    pub(super) fn for_test(
        process_id: &str,
        process: &str,
        external_particles: Vec<ExternalParticle>,
    ) -> Self {
        Self {
            schema_version: ON_THE_FLY_PUBLIC_METADATA_SCHEMA,
            kind: ON_THE_FLY_PUBLIC_METADATA_KIND.to_string(),
            process_id: process_id.to_string(),
            process: process.to_string(),
            color_accuracy: ColorAccuracy::Lc,
            external_particles,
            model_parameters: Vec::new(),
        }
    }

    fn validate(
        &self,
        path: &str,
        outer: &ArtifactProcess,
        manifest: &OnTheFlyExecutionManifest,
    ) -> RusticolResult<()> {
        if self.schema_version != ON_THE_FLY_PUBLIC_METADATA_SCHEMA
            || self.kind != ON_THE_FLY_PUBLIC_METADATA_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported on-the-fly public metadata kind {:?} schema {} at {path:?}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process_id != outer.id
            || self.process != outer.expression
            || self.process_id != manifest.key
            || self.process != manifest.process
            || self.color_accuracy.as_str() != outer.color_accuracy
            || self.color_accuracy.as_str() != manifest.color_accuracy
        {
            return Err(RusticolError::integrity(format!(
                "on-the-fly public metadata {path:?} does not match representative process {:?}",
                outer.id
            )));
        }
        if self.external_particles.len() != outer.external_pdgs.len()
            || self.external_particles.len() != manifest.runtime_metadata.external_legs.len()
            || self.external_particles.len() < 3
        {
            return Err(RusticolError::integrity(format!(
                "on-the-fly public metadata {path:?} has inconsistent external-particle coverage"
            )));
        }
        let expected_components = ["E", "px", "py", "pz"];
        for (index, ((particle, expected_pdg), runtime_leg)) in self
            .external_particles
            .iter()
            .zip(&outer.external_pdgs)
            .zip(&manifest.runtime_metadata.external_legs)
            .enumerate()
        {
            let expected_role = if runtime_leg.is_initial {
                crate::ParticleRole::Initial
            } else {
                crate::ParticleRole::Final
            };
            if particle.index != index
                || particle.label != index + 1
                || particle.momentum_slot != index
                || particle.particle.is_empty()
                || particle.pdg != *expected_pdg
                || particle.pdg != runtime_leg.physical_pdg
                || particle.role != expected_role
                || particle
                    .momentum_components
                    .iter()
                    .map(String::as_str)
                    .ne(expected_components)
            {
                return Err(RusticolError::integrity(format!(
                    "on-the-fly public metadata {path:?} has invalid external particle {index}"
                )));
            }
        }
        let mut parameter_names = BTreeSet::new();
        for parameter in &self.model_parameters {
            if parameter.name.is_empty()
                || !parameter_names.insert(parameter.name.as_str())
                || !parameter.default_real.is_finite()
                || !parameter.default_imaginary.is_finite()
                || (parameter.kind == ParameterKind::Derived && parameter.mutable)
            {
                return Err(RusticolError::artifact(format!(
                    "on-the-fly public metadata {path:?} has invalid model parameter {:?}",
                    parameter.name
                )));
            }
        }
        Ok(())
    }

    /// Materialize the existing public physics schema only for explicit
    /// metadata access. Every selector is executed directly by this lane, so
    /// each public member is its own exact representative with unit weight.
    pub(super) fn synthesize(
        &self,
        selectors: &OnTheFlyCompactSelectorAdapterV1,
    ) -> RusticolResult<ProcessPhysicsV1> {
        let mut introspection = OnTheFlySelectorIntrospectionCacheV1::default();
        let helicities = introspection
            .helicities(selectors)?
            .iter()
            .map(|record| Helicity {
                id: record.id.clone(),
                index: record.index,
                values: record.values.to_vec(),
                computed: true,
                structural_zero: false,
                representative_id: record.id.clone(),
                coefficient: 1.0,
            })
            .collect();
        let (color_components, coverage_color, coverage_color_kind, reduction, selectors_metadata) =
            match self.color_accuracy {
                ColorAccuracy::Lc => {
                    let color_components = introspection
                        .colors(selectors)?
                        .iter()
                        .map(|record| {
                            Ok(ColorComponent::LcFlow(LcColorFlow {
                                id: record.id.clone(),
                                index: record.index,
                                word: record
                                    .word
                                    .iter()
                                    .copied()
                                    .map(|label| {
                                        usize::try_from(label).map_err(|_| {
                                            RusticolError::artifact(
                                                "on-the-fly public color label exceeds usize",
                                            )
                                        })
                                    })
                                    .collect::<RusticolResult<Vec<_>>>()?,
                                computed: true,
                                representative_id: record.id.clone(),
                                coefficient: 1.0,
                            }))
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    (
                        color_components,
                        "complete".to_string(),
                        "physical-lc-flows".to_string(),
                        Reduction {
                            kind: ReductionKind::LcDiagonal,
                            groups: Vec::new(),
                        },
                        SelectorCapabilities {
                            helicity: true,
                            color_flow: true,
                            contracted_color: false,
                        },
                    )
                }
                ColorAccuracy::Nlc | ColorAccuracy::Full => (
                    vec![ColorComponent::ContractedColor(ContractedColor {
                        id: "color:contracted".to_string(),
                        index: 0,
                        description:
                            "coherent sparse contraction of the complete ordered color basis"
                                .to_string(),
                    })],
                    "contracted".to_string(),
                    "contracted-color".to_string(),
                    Reduction {
                        kind: ReductionKind::ContractedColor,
                        groups: Vec::new(),
                    },
                    SelectorCapabilities {
                        helicity: true,
                        color_flow: false,
                        contracted_color: false,
                    },
                ),
            };
        let physics = ProcessPhysicsV1 {
            schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
            kind: "pyamplicol-resolved-physics".to_string(),
            process_id: self.process_id.clone(),
            process: self.process.clone(),
            color_accuracy: self.color_accuracy,
            coverage: Coverage {
                helicities: "complete".to_string(),
                color: coverage_color,
                color_kind: coverage_color_kind,
                structural_zero_helicity_count: 0,
            },
            external_particles: self.external_particles.clone(),
            helicities,
            color_components,
            // Unlike eager/recurrence, the on-the-fly lane has no
            // pre-materialized public quotient groups to hydrate here.
            reduction,
            model_parameters: self.model_parameters.clone(),
            selectors: selectors_metadata,
            extensions: BTreeMap::new(),
        };
        physics.validate()?;
        Ok(physics)
    }
}

pub(super) fn parse_on_the_fly_public_metadata(
    bytes: &[u8],
    path: &str,
    outer: &ArtifactProcess,
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<OnTheFlyPublicMetadataV1> {
    let metadata: OnTheFlyPublicMetadataV1 = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse on-the-fly public metadata {path:?}: {error}"
        ))
    })?;
    metadata.validate(path, outer, manifest)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::on_the_fly_manifest::{
        ON_THE_FLY_EXECUTION_KIND, ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH,
        ON_THE_FLY_KERNEL_PAYLOAD_ROOT, ON_THE_FLY_PROCESS_SEED_MEMBER,
        ON_THE_FLY_RUNTIME_CONTAINER_KIND, ON_THE_FLY_RUNTIME_CONTAINER_PATH,
        ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA, ON_THE_FLY_RUNTIME_STORAGE_ABI,
        parse_on_the_fly_execution_manifest,
    };
    use crate::engine::{ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY, ON_THE_FLY_RUNTIME_CAPABILITY};
    use serde_json::{Value, json};

    fn outer_with_accuracy(color_accuracy: &str) -> ArtifactProcess {
        ArtifactProcess {
            id: "d_dbar_to_z".into(),
            expression: "d d~ > z".into(),
            color_accuracy: color_accuracy.into(),
            external_pdgs: vec![1, -1, 23],
            physics_path: "processes/d_dbar_to_z/physics.json".into(),
            required_runtime_capabilities: vec![
                ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY.into(),
                ON_THE_FLY_RUNTIME_CAPABILITY.into(),
            ],
            aliases: Vec::new(),
        }
    }

    fn outer() -> ArtifactProcess {
        outer_with_accuracy("lc")
    }

    fn execution() -> OnTheFlyExecutionManifest {
        let outer = outer();
        let value = json!({
            "schema_version": crate::PROCESS_ARTIFACT_SCHEMA_VERSION,
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
                "query_construction_threads": 1
            },
            "selector_policy": {
                "color_coverage": "complete",
                "reference_color_word": null,
                "trace_reflections_folded": false,
                "selector_census": {
                    "physical_helicity_count": 1,
                    "physical_color_flow_count": 1,
                },
            },
            "runtime_metadata": {
                "runtime_parameters": [],
                "prepared_parameter_defaults": [],
                "parameter_projection": [],
                "external_legs": [
                    {"source_slot": 0, "public_label": 1, "physical_pdg": 1, "outgoing_pdg": -1, "is_initial": true},
                    {"source_slot": 1, "public_label": 2, "physical_pdg": -1, "outgoing_pdg": 1, "is_initial": true},
                    {"source_slot": 2, "public_label": 3, "physical_pdg": 23, "outgoing_pdg": 23, "is_initial": false},
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
                    "coupling_policy": "local",
                },
                "process_seed_identity": {
                    "abi": crate::recurrence::on_the_fly::ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI,
                    "process_digest": "11".repeat(32),
                    "compiled_model_digest": "22".repeat(32),
                    "recurrence_template_catalog_digest": "33".repeat(32),
                    "prepared_kernel_pack_digest": "44".repeat(32),
                    "recurrence_direct_template_catalog_digest": "55".repeat(32),
                    "semantic_digest": "66".repeat(32),
                    "external_permutation": [0, 1, 2],
                    "external_sources": [
                        {"source_slot": 0, "public_label": 1, "is_initial": true, "states": [{"state_index": 0, "public_helicity": -1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 1, "public_label": 2, "is_initial": true, "states": [{"state_index": 0, "public_helicity": 1, "prepared_mass_parameter_slot": null}]},
                        {"source_slot": 2, "public_label": 3, "is_initial": false, "states": [{"state_index": 0, "public_helicity": 0, "prepared_mass_parameter_slot": null}]}
                    ]
                },
            },
            "runtime_container": {
                "kind": ON_THE_FLY_RUNTIME_CONTAINER_KIND,
                "schema_version": ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA,
                "storage_abi": ON_THE_FLY_RUNTIME_STORAGE_ABI,
                "path": ON_THE_FLY_RUNTIME_CONTAINER_PATH,
                "seed_member_path": ON_THE_FLY_PROCESS_SEED_MEMBER,
            },
        });
        parse_on_the_fly_execution_manifest(&serde_json::to_vec(&value).unwrap(), &outer).unwrap()
    }

    fn compact() -> Value {
        json!({
            "schema_version": ON_THE_FLY_PUBLIC_METADATA_SCHEMA,
            "kind": ON_THE_FLY_PUBLIC_METADATA_KIND,
            "process_id": "d_dbar_to_z",
            "process": "d d~ > z",
            "color_accuracy": "lc",
            "external_particles": [
                {"index": 0, "label": 1, "particle": "d", "pdg": 1, "role": "initial", "momentum_slot": 0, "momentum_components": ["E", "px", "py", "pz"]},
                {"index": 1, "label": 2, "particle": "d~", "pdg": -1, "role": "initial", "momentum_slot": 1, "momentum_components": ["E", "px", "py", "pz"]},
                {"index": 2, "label": 3, "particle": "z", "pdg": 23, "role": "final", "momentum_slot": 2, "momentum_components": ["E", "px", "py", "pz"]},
            ],
            "model_parameters": [
                {"name": "alpha_s", "kind": "external", "default_real": 0.118, "default_imaginary": 0.0, "mutable": true},
            ],
        })
    }

    fn parse(value: &Value) -> RusticolResult<OnTheFlyPublicMetadataV1> {
        parse_on_the_fly_public_metadata(
            &serde_json::to_vec(value).unwrap(),
            &outer().physics_path,
            &outer(),
            &execution(),
        )
    }

    fn parse_contracted(
        value: &Value,
        color_accuracy: &str,
    ) -> RusticolResult<OnTheFlyPublicMetadataV1> {
        let outer = outer_with_accuracy(color_accuracy);
        let mut manifest = execution();
        manifest.color_accuracy = color_accuracy.to_string();
        parse_on_the_fly_public_metadata(
            &serde_json::to_vec(value).unwrap(),
            &outer.physics_path,
            &outer,
            &manifest,
        )
    }

    #[test]
    fn compact_metadata_authenticates_identity_externals_and_parameters() {
        parse(&compact()).unwrap();

        let mut cases = Vec::new();
        let mut wrong_kind = compact();
        wrong_kind["kind"] = json!("pyamplicol-resolved-physics");
        cases.push(wrong_kind);
        let mut wrong_schema = compact();
        wrong_schema["schema_version"] = json!(2);
        cases.push(wrong_schema);
        let mut wrong_id = compact();
        wrong_id["process_id"] = json!("other");
        cases.push(wrong_id);
        let mut wrong_process = compact();
        wrong_process["process"] = json!("d d~ > a");
        cases.push(wrong_process);
        let mut wrong_accuracy = compact();
        wrong_accuracy["color_accuracy"] = json!("nlc");
        cases.push(wrong_accuracy);

        for invalid in cases {
            assert!(parse(&invalid).is_err());
        }
    }

    #[test]
    fn compact_metadata_accepts_contracted_nlc_and_full_accuracy() {
        for accuracy in ["nlc", "full"] {
            let mut value = compact();
            value["color_accuracy"] = json!(accuracy);
            let parsed = parse_contracted(&value, accuracy).unwrap();
            assert_eq!(parsed.color_accuracy.as_str(), accuracy);

            value["color_accuracy"] = json!("lc");
            assert!(parse_contracted(&value, accuracy).is_err());
        }
    }

    #[test]
    fn compact_metadata_tampering_fails_closed() {
        let mut cases = Vec::new();
        let mut wrong_pdg = compact();
        wrong_pdg["external_particles"][2]["pdg"] = json!(22);
        cases.push(wrong_pdg);
        let mut wrong_role = compact();
        wrong_role["external_particles"][1]["role"] = json!("final");
        cases.push(wrong_role);
        let mut wrong_slot = compact();
        wrong_slot["external_particles"][2]["momentum_slot"] = json!(1);
        cases.push(wrong_slot);
        let mut wrong_components = compact();
        wrong_components["external_particles"][0]["momentum_components"][0] = json!("energy");
        cases.push(wrong_components);
        let mut invalid_parameter = compact();
        invalid_parameter["model_parameters"][0]["kind"] = json!("derived");
        cases.push(invalid_parameter);
        let mut unknown_field = compact();
        unknown_field["dense_helicities"] = json!([]);
        cases.push(unknown_field);

        for invalid in cases {
            assert!(parse(&invalid).is_err());
        }
    }
}
