// SPDX-License-Identifier: 0BSD

use crate::{RUNTIME_PHYSICS_SCHEMA_VERSION, RusticolError, RusticolResult};
use bincode::Decode;
#[cfg(any(feature = "python-generation-bridge", test))]
use bincode::Encode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "lowercase")]
pub enum ColorAccuracy {
    Lc,
    Nlc,
    Full,
}

impl ColorAccuracy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lc => "lc",
            Self::Nlc => "nlc",
            Self::Full => "full",
        }
    }
}

impl std::fmt::Display for ColorAccuracy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "lowercase")]
pub enum ParticleRole {
    Initial,
    Final,
}

#[derive(Clone, Copy, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub enum ParameterKind {
    Normalization,
    Mass,
    Width,
    Coupling,
    External,
    Derived,
}

#[derive(Clone, Copy, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub enum ReductionKind {
    #[serde(rename = "lc-diagonal")]
    LcDiagonal,
    #[serde(rename = "contracted-color")]
    ContractedColor,
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub helicities: String,
    pub color: String,
    pub color_kind: String,
    pub structural_zero_helicity_count: usize,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct ExternalParticle {
    pub index: usize,
    pub label: usize,
    pub particle: String,
    pub pdg: i32,
    pub role: ParticleRole,
    pub momentum_slot: usize,
    pub momentum_components: [String; 4],
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct Helicity {
    pub id: String,
    pub index: usize,
    pub values: Vec<i32>,
    pub computed: bool,
    pub structural_zero: bool,
    pub representative_id: String,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct LcColorFlow {
    pub id: String,
    pub index: usize,
    pub word: Vec<usize>,
    pub computed: bool,
    pub representative_id: String,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct ContractedColor {
    pub id: String,
    pub index: usize,
    pub description: String,
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(tag = "kind")]
pub enum ColorComponent {
    #[serde(rename = "lc-flow")]
    LcFlow(LcColorFlow),
    #[serde(rename = "contracted-color")]
    ContractedColor(ContractedColor),
}

impl ColorComponent {
    pub fn id(&self) -> &str {
        match self {
            Self::LcFlow(value) => &value.id,
            Self::ContractedColor(value) => &value.id,
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::LcFlow(value) => value.index,
            Self::ContractedColor(value) => value.index,
        }
    }

    pub fn coefficient(&self) -> f64 {
        match self {
            Self::LcFlow(value) => value.coefficient,
            Self::ContractedColor(_) => 1.0,
        }
    }
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct ReductionGroup {
    pub id: String,
    pub representative_helicity_id: String,
    pub representative_color_id: String,
    pub physical_helicity_ids: Vec<String>,
    pub physical_color_ids: Vec<String>,
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct Reduction {
    pub kind: ReductionKind,
    pub groups: Vec<ReductionGroup>,
}

#[derive(Clone, Debug, Decode, Deserialize, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct ModelParameter {
    pub name: String,
    pub kind: ParameterKind,
    pub default_real: f64,
    pub default_imaginary: f64,
    pub mutable: bool,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub struct SelectorCapabilities {
    pub helicity: bool,
    pub color_flow: bool,
    pub contracted_color: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessPhysics {
    pub schema_version: u32,
    pub kind: String,
    pub process_id: String,
    pub process: String,
    pub color_accuracy: ColorAccuracy,
    pub coverage: Coverage,
    pub external_particles: Vec<ExternalParticle>,
    pub helicities: Vec<Helicity>,
    pub color_components: Vec<ColorComponent>,
    pub reduction: Reduction,
    pub model_parameters: Vec<ModelParameter>,
    pub selectors: SelectorCapabilities,
    #[serde(default)]
    pub extensions: BTreeMap<String, Value>,
}

impl ProcessPhysics {
    pub fn from_json(bytes: &[u8], path: &str) -> RusticolResult<Self> {
        let value: Self = serde_json::from_slice(bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse runtime physics payload {path:?}: {error}"
            ))
        })?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> RusticolResult<()> {
        if self.schema_version != RUNTIME_PHYSICS_SCHEMA_VERSION
            || self.kind != "pyamplicol-resolved-physics"
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported runtime physics kind {:?} schema {}; regenerate the artifact with pyAmpliCol 0.1 or newer",
                self.kind, self.schema_version
            )));
        }
        if self.external_particles.len() < 3 || self.helicities.is_empty() {
            return Err(RusticolError::artifact(
                "runtime physics requires at least three particles and one helicity",
            ));
        }
        if self.process.is_empty()
            || !matches!(self.coverage.helicities.as_str(), "complete" | "selected")
            || !matches!(
                self.coverage.color.as_str(),
                "complete" | "selected" | "contracted"
            )
            || !matches!(
                self.coverage.color_kind.as_str(),
                "physical-lc-flows" | "contracted-color"
            )
        {
            return Err(RusticolError::artifact(
                "runtime physics has an invalid process or coverage value",
            ));
        }
        validate_public_id(&self.process_id, "process id")?;
        let expected_components = ["E", "px", "py", "pz"];
        let mut momentum_slots = BTreeSet::new();
        for (index, particle) in self.external_particles.iter().enumerate() {
            if particle.index != index || particle.label != index + 1 {
                return Err(RusticolError::artifact(format!(
                    "external particle {index} has inconsistent index/label metadata"
                )));
            }
            if particle.particle.is_empty() {
                return Err(RusticolError::artifact(format!(
                    "external particle {index} has an empty particle name"
                )));
            }
            if !momentum_slots.insert(particle.momentum_slot) {
                return Err(RusticolError::artifact(format!(
                    "external particle {index} reuses momentum slot {}",
                    particle.momentum_slot
                )));
            }
            if particle
                .momentum_components
                .iter()
                .map(String::as_str)
                .ne(expected_components)
            {
                return Err(RusticolError::artifact(format!(
                    "external particle {index} must use momentum components [E, px, py, pz]"
                )));
            }
        }
        if momentum_slots != (0..self.external_particles.len()).collect() {
            return Err(RusticolError::artifact(
                "external particle momentum slots must form a complete zero-based permutation",
            ));
        }

        let mut helicity_ids = BTreeMap::new();
        for (index, helicity) in self.helicities.iter().enumerate() {
            validate_public_id(&helicity.id, "helicity id")?;
            validate_public_id(&helicity.representative_id, "representative helicity id")?;
            if helicity.index != index || helicity.values.len() != self.external_particles.len() {
                return Err(RusticolError::artifact(format!(
                    "helicity {} has inconsistent index or particle count",
                    helicity.id
                )));
            }
            if !helicity.coefficient.is_finite() || helicity.coefficient < 0.0 {
                return Err(RusticolError::artifact(format!(
                    "helicity {} has an invalid reduction coefficient",
                    helicity.id
                )));
            }
            if helicity_ids
                .insert(helicity.id.as_str(), helicity)
                .is_some()
            {
                return Err(RusticolError::artifact(format!(
                    "duplicate helicity id {:?}",
                    helicity.id
                )));
            }
        }
        for helicity in &self.helicities {
            if !helicity_ids.contains_key(helicity.representative_id.as_str()) {
                return Err(RusticolError::artifact(format!(
                    "helicity {} references unknown representative {:?}",
                    helicity.id, helicity.representative_id
                )));
            }
            if helicity.structural_zero && helicity.computed {
                return Err(RusticolError::artifact(format!(
                    "structural-zero helicity {} may not be marked computed",
                    helicity.id
                )));
            }
            if helicity.structural_zero && helicity.coefficient != 0.0 {
                return Err(RusticolError::artifact(format!(
                    "structural-zero helicity {} must have a zero reduction coefficient",
                    helicity.id
                )));
            }
            if !helicity.structural_zero {
                if helicity.coefficient <= 0.0 {
                    return Err(RusticolError::artifact(format!(
                        "nonzero helicity {} must have a positive reduction coefficient",
                        helicity.id
                    )));
                }
                let representative = helicity_ids
                    .get(helicity.representative_id.as_str())
                    .expect("representative existence checked");
                if !representative.computed || representative.structural_zero {
                    return Err(RusticolError::artifact(format!(
                        "helicity {} references a non-computed representative {:?}",
                        helicity.id, helicity.representative_id
                    )));
                }
                if helicity.computed && helicity.representative_id != helicity.id {
                    return Err(RusticolError::artifact(format!(
                        "computed helicity {} must represent itself",
                        helicity.id
                    )));
                }
            }
        }
        let structural_zeros = self
            .helicities
            .iter()
            .filter(|helicity| helicity.structural_zero)
            .count();
        if structural_zeros != self.coverage.structural_zero_helicity_count {
            return Err(RusticolError::artifact(format!(
                "coverage declares {} structural-zero helicities, metadata contains {structural_zeros}",
                self.coverage.structural_zero_helicity_count
            )));
        }

        let mut color_ids = BTreeMap::new();
        for (index, color) in self.color_components.iter().enumerate() {
            validate_public_id(color.id(), "color id")?;
            if color.index() != index
                || !color.coefficient().is_finite()
                || color.coefficient() < 0.0
            {
                return Err(RusticolError::artifact(format!(
                    "color component {} has an invalid index or coefficient",
                    color.id()
                )));
            }
            if color_ids.insert(color.id(), color).is_some() {
                return Err(RusticolError::artifact(format!(
                    "duplicate color component id {:?}",
                    color.id()
                )));
            }
            if let ColorComponent::LcFlow(flow) = color {
                if flow
                    .word
                    .iter()
                    .any(|label| *label == 0 || *label > self.external_particles.len())
                {
                    return Err(RusticolError::artifact(format!(
                        "LC color component {} contains an out-of-range external label",
                        flow.id
                    )));
                }
            } else if let ColorComponent::ContractedColor(color) = color
                && color.description.is_empty()
            {
                return Err(RusticolError::artifact(format!(
                    "contracted color component {} has an empty description",
                    color.id
                )));
            }
        }
        for color in &self.color_components {
            if let ColorComponent::LcFlow(flow) = color {
                let Some(representative) = color_ids.get(flow.representative_id.as_str()) else {
                    return Err(RusticolError::artifact(format!(
                        "color component {} references unknown representative {:?}",
                        flow.id, flow.representative_id
                    )));
                };
                let ColorComponent::LcFlow(representative) = representative else {
                    return Err(RusticolError::artifact(format!(
                        "LC color component {} references a contracted representative",
                        flow.id
                    )));
                };
                if !representative.computed {
                    return Err(RusticolError::artifact(format!(
                        "LC color component {} references a non-computed representative {:?}",
                        flow.id, flow.representative_id
                    )));
                }
                if flow.computed && flow.representative_id != flow.id {
                    return Err(RusticolError::artifact(format!(
                        "computed LC color component {} must represent itself",
                        flow.id
                    )));
                }
            }
        }
        match self.color_accuracy {
            ColorAccuracy::Lc => {
                if self.color_components.is_empty()
                    || self
                        .color_components
                        .iter()
                        .any(|item| !matches!(item, ColorComponent::LcFlow(_)))
                {
                    return Err(RusticolError::artifact(
                        "LC physics requires one or more physical LC color flows",
                    ));
                }
                if self.reduction.kind != ReductionKind::LcDiagonal
                    || self.coverage.color_kind != "physical-lc-flows"
                    || !self.selectors.color_flow
                    || self.selectors.contracted_color
                {
                    return Err(RusticolError::artifact(
                        "LC physics has inconsistent reduction, coverage, or selector metadata",
                    ));
                }
            }
            ColorAccuracy::Nlc | ColorAccuracy::Full => {
                if self.color_components.len() != 1
                    || !matches!(
                        self.color_components.first(),
                        Some(ColorComponent::ContractedColor(ContractedColor {
                            index: 0,
                            ..
                        }))
                    )
                {
                    return Err(RusticolError::artifact(
                        "NLC/full physics requires exactly one contracted color component",
                    ));
                }
                if self.reduction.kind != ReductionKind::ContractedColor
                    || self.coverage.color != "contracted"
                    || self.coverage.color_kind != "contracted-color"
                    || self.selectors.color_flow
                    || self.selectors.contracted_color
                {
                    return Err(RusticolError::artifact(
                        "NLC/full physics has inconsistent reduction, coverage, or selector metadata",
                    ));
                }
            }
        }

        let mut parameter_names = BTreeSet::new();
        for parameter in &self.model_parameters {
            if parameter.name.is_empty()
                || !parameter_names.insert(&parameter.name)
                || !parameter.default_real.is_finite()
                || !parameter.default_imaginary.is_finite()
                || (parameter.kind == ParameterKind::Derived && parameter.mutable)
            {
                return Err(RusticolError::artifact(format!(
                    "invalid or duplicate model parameter {:?}",
                    parameter.name
                )));
            }
        }

        let mut seen_groups = BTreeSet::new();
        for group in &self.reduction.groups {
            validate_public_id(&group.id, "reduction group id")?;
            if !seen_groups.insert(&group.id) {
                return Err(RusticolError::artifact(format!(
                    "duplicate reduction group id {:?}",
                    group.id
                )));
            }
            if group.physical_helicity_ids.is_empty() || group.physical_color_ids.is_empty() {
                return Err(RusticolError::artifact(format!(
                    "reduction group {} has an empty physical expansion",
                    group.id
                )));
            }
            let has_representative_helicity =
                helicity_ids.contains_key(group.representative_helicity_id.as_str());
            let has_representative_color =
                color_ids.contains_key(group.representative_color_id.as_str());
            if !has_representative_helicity || !has_representative_color {
                return Err(RusticolError::artifact(format!(
                    "reduction group {} references an unknown representative: helicity {:?} present={has_representative_helicity}, color {:?} present={has_representative_color}; available colors={:?}",
                    group.id,
                    group.representative_helicity_id,
                    group.representative_color_id,
                    color_ids.keys().collect::<Vec<_>>(),
                )));
            }
            if !group
                .physical_helicity_ids
                .contains(&group.representative_helicity_id)
                || !group
                    .physical_color_ids
                    .contains(&group.representative_color_id)
            {
                return Err(RusticolError::artifact(format!(
                    "reduction group {} does not include its representative in its physical expansion",
                    group.id
                )));
            }
            let mut physical_helicities = BTreeSet::new();
            for id in &group.physical_helicity_ids {
                if !helicity_ids.contains_key(id.as_str()) || !physical_helicities.insert(id) {
                    return Err(RusticolError::artifact(format!(
                        "reduction group {} has an unknown or duplicate helicity {:?}",
                        group.id, id
                    )));
                }
            }
            let mut physical_colors = BTreeSet::new();
            for id in &group.physical_color_ids {
                if !color_ids.contains_key(id.as_str()) || !physical_colors.insert(id) {
                    return Err(RusticolError::artifact(format!(
                        "reduction group {} has an unknown or duplicate color {:?}",
                        group.id, id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn validate_public_id(value: &str, description: &str) -> RusticolResult<()> {
    let valid = !value.is_empty()
        && value.len() <= 255
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric()
                || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'+' | b',' | b'~' | b'-'))
        });
    if !valid {
        return Err(RusticolError::artifact(format!(
            "invalid {description} {value:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
