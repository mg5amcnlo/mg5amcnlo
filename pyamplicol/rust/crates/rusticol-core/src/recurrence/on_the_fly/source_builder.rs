// SPDX-License-Identifier: 0BSD

#![allow(dead_code)] // Generation-bridge helpers are absent from default runtime builds.

//! Cold-path construction of the compact on-the-fly process seed.
//!
//! The source projection is process-owned but contains no materialized
//! recurrence graph, color plan, public-flow table, or direct plan. Model
//! semantics are recovered from the already validated recurrence-template and
//! prepared direct-executor catalogs before the existing seed codec is used.

#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;

use serde::Deserialize;

use super::source_seed::{
    OnTheFlyCouplingOrderPolicyV1, OnTheFlyExternalColorRoleV1, OnTheFlyPairingClassV1,
    OnTheFlyPairingEndpointV1, OnTheFlyProcessSeedV1, OnTheFlySourceAnchorV1,
    OnTheFlySourceOrientationV1, OnTheFlySourceStateV1, OnTheFlySourceWavefunctionFamilyV1,
    validate_permutation,
};
use super::{TemplateCatalog, integrity, invalid};
use crate::recurrence::process::ProcessSourceStateRow;
use crate::recurrence::template::{
    CurrentOrientation, CurrentStateRow, MISSING_U32, ParameterValueType, ParticleStatistics,
    SourceRow, ValidatedRecurrenceTemplateInput,
};
use crate::recurrence::{
    DirectExecutorRole, ExactComplexRational, PreparedDirectExecutorCatalog, SemanticDigest,
};
use crate::{RusticolError, RusticolResult};

const SOURCE_PROJECTION_SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
thread_local! {
    static PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OnTheFlyProcessSeedProjectionV1 {
    schema_version: u32,
    process_digest: String,
    external_permutation: Vec<u32>,
    external_sources: Vec<SourceAnchorProjection>,
    parameter_projection: Vec<ParameterProjection>,
    coupling_order_policy: CouplingOrderPolicyProjection,
    coupling_hierarchies: Vec<CouplingHierarchyProjection>,
    coupling_limits: Vec<CouplingLimitProjection>,
    fermion_pairing: Option<FermionPairingProjection>,
    normalization: NormalizationProjection,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceAnchorProjection {
    source_slot: u32,
    public_label: u32,
    is_initial: bool,
    states: Vec<SourceStateProjection>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceStateProjection {
    state_index: u32,
    public_helicity: i32,
    source_helicity: i32,
    source_template_id: u32,
    current_state_template_id: u32,
    momentum_sign: i32,
    crossing_phase: ExactFactorProjection,
    spin_state: i32,
    chirality: i32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterProjection {
    parameter_template_id: u32,
    prepared_parameter_id: Option<u32>,
    component: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CouplingLimitProjection {
    name: String,
    maximum: u32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CouplingOrderPolicyProjection {
    Minimal,
    Explicit,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CouplingHierarchyProjection {
    name: String,
    hierarchy: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FermionPairingProjection {
    endpoints: Vec<PairingEndpointProjection>,
    classes: Vec<PairingClassProjection>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PairingEndpointOrientation {
    Fundamental,
    Antifundamental,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairingEndpointProjection {
    source_slot: u32,
    color_orientation: PairingEndpointOrientation,
    contract_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairingClassProjection {
    species: String,
    proof_digest: String,
    fundamental_source_slots: Vec<u32>,
    antifundamental_source_slots: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NormalizationProjection {
    factor: ExactFactorProjection,
    convention: String,
    semantic_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactFactorProjection {
    real_numerator: String,
    real_denominator: String,
    imag_numerator: String,
    imag_denominator: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCrossingProjection {
    momentum_transform: SourceMomentumTransformProjection,
    helicity_factor: i32,
    chirality_factor: i32,
    spin_state_factor: i32,
    phase: ExactFactorProjection,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum SourceMomentumTransformProjection {
    #[serde(rename = "identity")]
    Identity,
    #[serde(rename = "negate-four-momentum")]
    NegateFourMomentum,
}

#[derive(Clone, Copy, Debug)]
struct AppliedSourceCrossing {
    momentum_sign: i32,
    helicity_factor: i32,
    chirality_factor: i32,
    spin_state_factor: i32,
    phase: ExactComplexRational,
}

impl AppliedSourceCrossing {
    const IDENTITY: Self = Self {
        momentum_sign: 1,
        helicity_factor: 1,
        chirality_factor: 1,
        spin_state_factor: 1,
        phase: ExactComplexRational::ONE,
    };
}

impl ExactFactorProjection {
    fn exact(&self, label: &str) -> RusticolResult<ExactComplexRational> {
        ExactComplexRational::parse_parts(
            &self.real_numerator,
            &self.real_denominator,
            &self.imag_numerator,
            &self.imag_denominator,
        )
        .map_err(|error| invalid(format!("{label} is invalid: {error}")))
    }
}

pub(crate) fn parse_on_the_fly_process_seed_projection_v1(
    bytes: &[u8],
) -> RusticolResult<OnTheFlyProcessSeedProjectionV1> {
    if bytes.is_empty() {
        return Err(invalid(
            "on-the-fly source projection JSON must not be empty",
        ));
    }
    let projection: OnTheFlyProcessSeedProjectionV1 =
        serde_json::from_slice(bytes).map_err(|error| {
            invalid(format!(
                "invalid on-the-fly source projection JSON: {error}"
            ))
        })?;
    if projection.schema_version != SOURCE_PROJECTION_SCHEMA_VERSION {
        return Err(invalid(format!(
            "unsupported on-the-fly source projection schema version {}",
            projection.schema_version
        )));
    }
    Ok(projection)
}

/// Build one compact process seed from source-only process projection and
/// already validated model-wide catalogs.
///
/// This deliberately has no argument capable of carrying the established
/// recurrence builder, a color-flow plan, a DAG, or a `DirectRecurrencePlan`.
pub(crate) fn build_on_the_fly_process_seed_v1(
    projection: OnTheFlyProcessSeedProjectionV1,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    prepared_pack_digest: SemanticDigest,
) -> RusticolResult<OnTheFlyProcessSeedV1> {
    let mut seeds = build_on_the_fly_process_seeds_v1(
        vec![projection],
        templates,
        direct,
        prepared_pack_digest,
    )?;
    seeds
        .pop()
        .ok_or_else(|| RusticolError::internal("singleton process-seed batch returned no seed"))
}

/// Build ordered compact process seeds while authenticating and indexing the
/// shared model catalogs exactly once.
pub(crate) fn build_on_the_fly_process_seeds_v1(
    projections: Vec<OnTheFlyProcessSeedProjectionV1>,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    prepared_pack_digest: SemanticDigest,
) -> RusticolResult<Vec<OnTheFlyProcessSeedV1>> {
    let summary = templates.summary();
    if prepared_pack_digest != summary.prepared_kernel_pack_digest {
        return Err(integrity(
            "on-the-fly source projection names a different prepared-kernel pack",
        ));
    }

    #[cfg(test)]
    PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| count.set(count.get() + 1));
    let template_catalog = TemplateCatalog::new(templates.input())?;
    let mut seeds = Vec::new();
    seeds
        .try_reserve_exact(projections.len())
        .map_err(|error| invalid(format!("process-seed allocation failed: {error}")))?;
    for (index, projection) in projections.into_iter().enumerate() {
        let process_digest = projection.process_digest.clone();
        let seed = build_on_the_fly_process_seed_from_catalog_v1(
            projection,
            templates,
            &template_catalog,
            direct,
            prepared_pack_digest,
        )
        .map_err(|error| {
            RusticolError::with_kind(
                error.kind(),
                format!(
                    "on-the-fly process seed at index {index} (process digest {process_digest:?}) failed: {error}"
                ),
            )
        })?;
        seeds.push(seed);
    }
    Ok(seeds)
}

fn build_on_the_fly_process_seed_from_catalog_v1(
    projection: OnTheFlyProcessSeedProjectionV1,
    templates: &ValidatedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
    direct: &PreparedDirectExecutorCatalog,
    prepared_pack_digest: SemanticDigest,
) -> RusticolResult<OnTheFlyProcessSeedV1> {
    let summary = templates.summary();
    validate_permutation(
        &projection.external_permutation,
        projection.external_sources.len(),
        "source-projection external permutation",
    )?;

    let endpoint_contracts = pairing_endpoint_contracts(projection.fermion_pairing.as_ref())?;
    validate_parameter_projection(&projection.parameter_projection)?;
    let source_anchors = source_anchors(
        &projection.external_sources,
        templates,
        template_catalog,
        direct,
        &endpoint_contracts,
        &projection.parameter_projection,
    )?;
    let pairing_classes =
        pairing_classes(projection.fermion_pairing.as_ref(), &endpoint_contracts)?;
    let (coupling_order_policy, coupling_hierarchies, coupling_limits) = coupling_policy(
        projection.coupling_order_policy,
        &projection.coupling_hierarchies,
        &projection.coupling_limits,
        template_catalog,
    )?;
    let normalization_factor = projection
        .normalization
        .factor
        .exact("on-the-fly normalization factor")?;

    OnTheFlyProcessSeedV1::new(
        parse_digest(&projection.process_digest, "process digest")?,
        summary.compiled_model_digest,
        summary.catalog_digest,
        prepared_pack_digest,
        direct.direct_template_catalog_digest(),
        parse_digest(
            &projection.normalization.semantic_digest,
            "normalization semantic digest",
        )?,
        projection.normalization.convention,
        normalization_factor,
        source_anchors,
        projection.external_permutation,
        coupling_order_policy,
        coupling_hierarchies,
        coupling_limits,
        pairing_classes,
    )
}

fn source_anchors(
    projections: &[SourceAnchorProjection],
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    direct: &PreparedDirectExecutorCatalog,
    endpoint_contracts: &BTreeMap<u32, (OnTheFlyExternalColorRoleV1, SemanticDigest)>,
    parameter_projection: &[ParameterProjection],
) -> RusticolResult<Vec<OnTheFlySourceAnchorV1>> {
    let input = templates.input();
    let mut anchors = Vec::new();
    anchors
        .try_reserve_exact(projections.len())
        .map_err(|error| invalid(format!("source-anchor allocation failed: {error}")))?;
    for projection in projections {
        let mut states = Vec::new();
        states
            .try_reserve_exact(projection.states.len())
            .map_err(|error| invalid(format!("source-state allocation failed: {error}")))?;
        let mut role = None;
        let mut anchor_fermionic = None;
        let mut pairing_contract = None;
        for state in &projection.states {
            let source = canonical_source(input, state.source_template_id)?;
            let current_state = canonical_current_state(input, state.current_state_template_id)?;
            validate_source_projection(
                projection.source_slot,
                projection.is_initial,
                state,
                source,
                current_state,
                input,
                catalog,
            )?;
            direct.resolve_evaluator(DirectExecutorRole::Source, source.evaluator_binding_id)?;
            let family = source_family(catalog, source, current_state)?;
            let (state_role, state_pairing_contract) = color_role(
                current_state.color_representation,
                endpoint_contracts.get(&projection.source_slot).copied(),
            )?;
            merge_equal(&mut role, state_role, "one source anchor mixes color roles")?;
            merge_equal(
                &mut anchor_fermionic,
                family.is_fermionic(),
                "one source anchor mixes statistics",
            )?;
            merge_equal_option(
                &mut pairing_contract,
                state_pairing_contract,
                "one source anchor mixes pairing contracts",
            )?;
            let color_seed = catalog.source_seed(source)?;
            states.push(OnTheFlySourceStateV1::new(
                state.state_index,
                state.public_helicity,
                state.source_helicity,
                source.id,
                current_state.id,
                catalog.digest(source.semantic_digest_id, "source semantic")?,
                catalog.digest(current_state.semantic_digest_id, "current-state semantic")?,
                state.momentum_sign,
                state.crossing_phase.exact("source crossing phase")?,
                state.spin_state,
                state.chirality,
                catalog
                    .flavour_flow(source.flavour_flow_id, "source flavour flow")?
                    .to_vec(),
                source.quantum_number_flow_id,
                color_seed.proof_digest(),
                family,
                source_orientation(current_state)?,
                resolve_prepared_mass_slot(
                    source,
                    current_state,
                    input,
                    catalog,
                    parameter_projection.iter().map(|row| {
                        (
                            row.parameter_template_id,
                            row.prepared_parameter_id,
                            row.component,
                        )
                    }),
                )?,
            )?);
        }
        anchors.push(OnTheFlySourceAnchorV1::new(
            projection.source_slot,
            projection.public_label,
            projection.is_initial,
            role.ok_or_else(|| integrity("source anchor is empty"))?,
            anchor_fermionic.ok_or_else(|| integrity("source statistics are absent"))?,
            pairing_contract,
            states,
        )?);
    }
    Ok(anchors)
}

fn canonical_source(
    input: &crate::recurrence::template::OwnedRecurrenceTemplateInput,
    source_template_id: u32,
) -> RusticolResult<SourceRow> {
    let source = *input
        .sources
        .get(source_template_id as usize)
        .ok_or_else(|| integrity("source template is absent"))?;
    if source.id != source_template_id {
        return Err(integrity("source-template catalog is not canonical"));
    }
    Ok(source)
}

fn canonical_current_state(
    input: &crate::recurrence::template::OwnedRecurrenceTemplateInput,
    current_state_template_id: u32,
) -> RusticolResult<CurrentStateRow> {
    let state = *input
        .current_states
        .get(current_state_template_id as usize)
        .ok_or_else(|| integrity("current-state template is absent"))?;
    if state.id != current_state_template_id {
        return Err(integrity("current-state catalog is not canonical"));
    }
    Ok(state)
}

fn validate_source_projection(
    source_slot: u32,
    is_initial: bool,
    projected: &SourceStateProjection,
    source: SourceRow,
    effective: CurrentStateRow,
    input: &crate::recurrence::template::OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<()> {
    let canonical = canonical_current_state(input, source.state_template_id)?;
    super::validate_crossed_source_state(
        is_initial,
        &ProcessSourceStateRow {
            source_slot,
            state_index: projected.state_index,
            public_helicity: projected.public_helicity,
            chirality: projected.chirality,
            spin_state: projected.spin_state,
            current_state_template_id: projected.current_state_template_id,
            source_template_id: projected.source_template_id,
            momentum_sign: projected.momentum_sign,
            crossing_phase_factor_id: 0,
        },
        source,
        input,
    )?;

    let crossing = applied_source_crossing(is_initial, source, catalog)?;
    let expected_helicity = source
        .helicity
        .checked_mul(crossing.helicity_factor)
        .ok_or_else(|| integrity("source crossing overflows the helicity state"))?;
    let expected_chirality = canonical
        .chirality
        .checked_mul(crossing.chirality_factor)
        .ok_or_else(|| integrity("source crossing overflows the chirality state"))?;
    let expected_spin_state = source
        .spin_state
        .checked_mul(crossing.spin_state_factor)
        .ok_or_else(|| integrity("source crossing overflows the spin state"))?;
    let projected_phase = projected.crossing_phase.exact("source crossing phase")?;
    if projected.public_helicity != expected_helicity
        || projected.source_helicity != expected_helicity
        || projected.chirality != expected_chirality
        || effective.chirality != expected_chirality
        || projected.spin_state != expected_spin_state
        || projected.momentum_sign != crossing.momentum_sign
        || projected_phase != crossing.phase
    {
        return Err(integrity(
            "crossed source execution values disagree with their prepared source/current templates",
        ));
    }
    Ok(())
}

fn applied_source_crossing(
    is_initial: bool,
    source: SourceRow,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<AppliedSourceCrossing> {
    if !is_initial {
        return Ok(AppliedSourceCrossing::IDENTITY);
    }
    let encoded = catalog.string(source.crossing_string_id, "source crossing")?;
    let crossing: SourceCrossingProjection = serde_json::from_str(encoded)
        .map_err(|error| integrity(format!("source crossing contract is invalid: {error}")))?;
    if !matches!(crossing.helicity_factor, -1 | 1)
        || !matches!(crossing.chirality_factor, -1 | 1)
        || !matches!(crossing.spin_state_factor, -1 | 1)
    {
        return Err(integrity(
            "source crossing contract has a non-sign state factor",
        ));
    }
    let phase = crossing.phase.exact("source crossing phase")?;
    if phase.is_zero() {
        return Err(integrity("source crossing phase is zero"));
    }
    Ok(AppliedSourceCrossing {
        momentum_sign: match crossing.momentum_transform {
            SourceMomentumTransformProjection::Identity => 1,
            SourceMomentumTransformProjection::NegateFourMomentum => -1,
        },
        helicity_factor: crossing.helicity_factor,
        chirality_factor: crossing.chirality_factor,
        spin_state_factor: crossing.spin_state_factor,
        phase,
    })
}

fn source_family(
    catalog: &TemplateCatalog<'_>,
    source: SourceRow,
    current_state: CurrentStateRow,
) -> RusticolResult<OnTheFlySourceWavefunctionFamilyV1> {
    let family = catalog.string(
        source.wavefunction_family_string_id,
        "source wavefunction family",
    )?;
    let statistics = ParticleStatistics::try_from(current_state.statistics)?;
    match (family, statistics, current_state.dimension) {
        ("scalar", ParticleStatistics::Boson, 1) => Ok(OnTheFlySourceWavefunctionFamilyV1::Scalar),
        ("fermion", ParticleStatistics::Fermion, 2) => {
            Ok(OnTheFlySourceWavefunctionFamilyV1::WeylFermion)
        }
        ("fermion", ParticleStatistics::Fermion, 4) => {
            Ok(OnTheFlySourceWavefunctionFamilyV1::DiracFermion)
        }
        ("vector", ParticleStatistics::Boson, 4) => Ok(OnTheFlySourceWavefunctionFamilyV1::Vector),
        ("spin2", ParticleStatistics::Boson, 16) => Ok(OnTheFlySourceWavefunctionFamilyV1::Spin2),
        ("ghost" | "auxiliary", _, _) => Err(invalid(
            "on-the-fly source execution does not support ghost or auxiliary external states",
        )),
        _ => Err(integrity(
            "source family, statistics, and component dimension disagree",
        )),
    }
}

fn source_orientation(
    current_state: CurrentStateRow,
) -> RusticolResult<OnTheFlySourceOrientationV1> {
    Ok(
        match CurrentOrientation::try_from(current_state.orientation)? {
            CurrentOrientation::Particle => OnTheFlySourceOrientationV1::Particle,
            CurrentOrientation::Antiparticle => OnTheFlySourceOrientationV1::Antiparticle,
            CurrentOrientation::SelfConjugate => OnTheFlySourceOrientationV1::SelfConjugate,
        },
    )
}

fn color_role(
    representation: i32,
    endpoint: Option<(OnTheFlyExternalColorRoleV1, SemanticDigest)>,
) -> RusticolResult<(OnTheFlyExternalColorRoleV1, Option<SemanticDigest>)> {
    match (representation, endpoint) {
        (1, None) => Ok((OnTheFlyExternalColorRoleV1::Singlet, None)),
        (8, None) => Ok((OnTheFlyExternalColorRoleV1::Adjoint, None)),
        (3, Some((OnTheFlyExternalColorRoleV1::Fundamental, digest))) => {
            Ok((OnTheFlyExternalColorRoleV1::Fundamental, Some(digest)))
        }
        (-3, Some((OnTheFlyExternalColorRoleV1::Antifundamental, digest))) => {
            Ok((OnTheFlyExternalColorRoleV1::Antifundamental, Some(digest)))
        }
        _ => Err(integrity(
            "source color representation disagrees with the compact pairing endpoints",
        )),
    }
}

fn validate_parameter_projection(rows: &[ParameterProjection]) -> RusticolResult<()> {
    let mut result = BTreeMap::new();
    for row in rows.iter().filter(|row| row.component == 0) {
        let Some(prepared) = row.prepared_parameter_id else {
            continue;
        };
        if result.insert(row.parameter_template_id, prepared).is_some() {
            return Err(integrity(
                "parameter projection repeats a real prepared parameter",
            ));
        }
    }
    Ok(())
}

/// Resolve the prepared real-parameter slot used by one authenticated source.
///
/// Explicit template IDs remain authoritative. Legacy catalogs in which both
/// source rows omit that ID may recover it only from the exact canonical
/// particle-mass parameter name authenticated by the current-state
/// orientation. Keeping this resolver shared with the feature-only oracle
/// prevents diagnostic reconstruction from silently using different mass
/// semantics than the compact production seed.
pub(super) fn resolve_prepared_mass_slot(
    source: SourceRow,
    current_state: CurrentStateRow,
    input: &crate::recurrence::template::OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    projection: impl IntoIterator<Item = (u32, Option<u32>, u32)>,
) -> RusticolResult<Option<u32>> {
    if source.mass_parameter_id != current_state.mass_parameter_id {
        return Err(integrity(
            "source and current state disagree on their mass parameter",
        ));
    }

    let parameter_id = if source.mass_parameter_id != MISSING_U32 {
        source.mass_parameter_id
    } else {
        let canonical_particle = match CurrentOrientation::try_from(current_state.orientation)? {
            CurrentOrientation::Particle => current_state.particle_id,
            CurrentOrientation::Antiparticle => current_state.anti_particle_id,
            CurrentOrientation::SelfConjugate => {
                if current_state.particle_id != current_state.anti_particle_id {
                    return Err(integrity(
                        "self-conjugate source mass orientation has distinct particle labels",
                    ));
                }
                current_state.particle_id
            }
        };
        let expected_name = format!("particle.{canonical_particle}.mass");
        let mut match_id = None;
        for (index, parameter) in input.parameters.iter().enumerate() {
            if usize::try_from(parameter.id).ok() != Some(index) {
                return Err(integrity("mass parameter catalog is not canonical"));
            }
            if catalog.string(parameter.name_string_id, "mass parameter name")? != expected_name {
                continue;
            }
            if match_id.replace(parameter.id).is_some() {
                return Err(integrity(format!(
                    "canonical source mass parameter {expected_name:?} is ambiguous"
                )));
            }
        }
        let Some(parameter_id) = match_id else {
            return Ok(None);
        };
        parameter_id
    };

    let parameter = input
        .parameters
        .get(parameter_id as usize)
        .ok_or_else(|| integrity("source mass parameter is absent"))?;
    if parameter.id != parameter_id {
        return Err(integrity("source mass parameter catalog is not canonical"));
    }
    if ParameterValueType::try_from(parameter.value_type)? != ParameterValueType::Real {
        return Err(integrity("source mass parameter is not real"));
    }
    let catalog_prepared = parameter.prepared_parameter_id;
    if catalog_prepared == MISSING_U32 {
        return Err(invalid(
            "source mass parameter has no prepared execution slot",
        ));
    }
    let prepared_slot = usize::try_from(catalog_prepared)
        .map_err(|_| integrity("source mass prepared slot exceeds usize"))?;
    if prepared_slot >= input.parameters.len() {
        return Err(integrity(
            "source mass prepared slot is outside the dense parameter arena",
        ));
    }
    if input
        .parameters
        .iter()
        .filter(|candidate| candidate.prepared_parameter_id == catalog_prepared)
        .count()
        != 1
    {
        return Err(integrity(
            "source mass prepared slot has conflicting catalog owners",
        ));
    }

    let matches = projection
        .into_iter()
        .filter(|(projected_id, _, component)| *projected_id == parameter_id && *component == 0)
        .collect::<Vec<_>>();
    let [(_, projected_prepared, _)] = matches.as_slice() else {
        return Err(integrity(format!(
            "source mass parameter has {} component-0 projections, expected one",
            matches.len()
        )));
    };
    let projected_prepared = (*projected_prepared)
        .ok_or_else(|| invalid("source mass parameter has no prepared real projection"))?;
    if projected_prepared != catalog_prepared {
        return Err(integrity(
            "source mass projection disagrees with the prepared template catalog",
        ));
    }
    Ok(Some(catalog_prepared))
}

/// Rebind every decoded seed source to the authenticated model-wide parameter
/// catalog before the runtime turns its prepared slots into executor inputs.
/// This is a cold load-time check; selected-query execution never reaches it.
pub(crate) fn validate_on_the_fly_source_mass_bindings_v1(
    seed: &OnTheFlyProcessSeedV1,
    templates: &ValidatedRecurrenceTemplateInput,
    parameter_projection: &[(u32, Option<u32>, u32)],
) -> RusticolResult<()> {
    let input = templates.input();
    let catalog = TemplateCatalog::new(input)?;
    for anchor in seed.source_anchors() {
        for state in anchor.states() {
            let source = canonical_source(input, state.source_template_id)?;
            let current_state = canonical_current_state(input, state.current_state_template_id)?;
            let expected = resolve_prepared_mass_slot(
                source,
                current_state,
                input,
                &catalog,
                parameter_projection.iter().copied(),
            )?;
            if state.prepared_mass_parameter_slot != expected {
                return Err(integrity(
                    "on-the-fly source prepared mass slot disagrees with the authenticated recurrence template",
                ));
            }
        }
    }
    Ok(())
}

fn coupling_policy(
    policy: CouplingOrderPolicyProjection,
    hierarchy_rows: &[CouplingHierarchyProjection],
    limit_rows: &[CouplingLimitProjection],
    templates: &TemplateCatalog<'_>,
) -> RusticolResult<(OnTheFlyCouplingOrderPolicyV1, Vec<u32>, Vec<Option<u32>>)> {
    let mut hierarchies_by_name = BTreeMap::new();
    for row in hierarchy_rows {
        if row.name.is_empty()
            || row.hierarchy == 0
            || hierarchies_by_name
                .insert(row.name.as_str(), row.hierarchy)
                .is_some()
        {
            return Err(invalid(
                "coupling hierarchies require nonempty unique names and positive weights",
            ));
        }
    }
    let mut limits_by_name = BTreeMap::new();
    for row in limit_rows {
        if row.name.is_empty()
            || limits_by_name
                .insert(row.name.as_str(), row.maximum)
                .is_some()
        {
            return Err(invalid(
                "coupling limits require nonempty unique model-order names",
            ));
        }
    }
    let expected = templates.coupling_order_names();
    if limit_rows
        .iter()
        .any(|row| expected.binary_search(&row.name.as_str()).is_err())
    {
        return Err(integrity(
            "source projection names a coupling limit absent from the template catalog",
        ));
    }
    let hierarchies = expected
        .iter()
        .map(|name| {
            hierarchies_by_name.get(name).copied().ok_or_else(|| {
                integrity(format!(
                    "source projection has no coupling hierarchy for model order {name:?}"
                ))
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let limits = expected
        .iter()
        .map(|name| limits_by_name.get(name).copied())
        .collect::<Vec<_>>();
    if hierarchies_by_name.len() != expected.len() || limits_by_name.len() > expected.len() {
        return Err(integrity(
            "source projection names a coupling hierarchy or limit absent from the template catalog",
        ));
    }
    Ok((
        match policy {
            CouplingOrderPolicyProjection::Minimal => OnTheFlyCouplingOrderPolicyV1::Minimal,
            CouplingOrderPolicyProjection::Explicit => OnTheFlyCouplingOrderPolicyV1::Explicit,
        },
        hierarchies,
        limits,
    ))
}

fn pairing_endpoint_contracts(
    pairing: Option<&FermionPairingProjection>,
) -> RusticolResult<BTreeMap<u32, (OnTheFlyExternalColorRoleV1, SemanticDigest)>> {
    let Some(pairing) = pairing else {
        return Ok(BTreeMap::new());
    };
    let mut result = BTreeMap::new();
    for endpoint in &pairing.endpoints {
        let role = match endpoint.color_orientation {
            PairingEndpointOrientation::Fundamental => OnTheFlyExternalColorRoleV1::Fundamental,
            PairingEndpointOrientation::Antifundamental => {
                OnTheFlyExternalColorRoleV1::Antifundamental
            }
        };
        let digest = parse_digest(
            &endpoint.contract_digest,
            "pairing endpoint contract digest",
        )?;
        if result
            .insert(endpoint.source_slot, (role, digest))
            .is_some()
        {
            return Err(integrity("pairing endpoint repeats a source slot"));
        }
    }
    Ok(result)
}

fn pairing_classes(
    pairing: Option<&FermionPairingProjection>,
    contracts: &BTreeMap<u32, (OnTheFlyExternalColorRoleV1, SemanticDigest)>,
) -> RusticolResult<Vec<OnTheFlyPairingClassV1>> {
    let Some(pairing) = pairing else {
        if contracts.is_empty() {
            return Ok(Vec::new());
        }
        return Err(integrity("pairing endpoints have no pairing classes"));
    };
    pairing
        .classes
        .iter()
        .map(|class| {
            let endpoints = |slots: &[u32],
                             role: OnTheFlyExternalColorRoleV1|
             -> RusticolResult<Vec<OnTheFlyPairingEndpointV1>> {
                slots
                    .iter()
                    .copied()
                    .map(|source_slot| {
                        let (actual_role, digest) = contracts
                            .get(&source_slot)
                            .copied()
                            .ok_or_else(|| integrity("pairing class endpoint is absent"))?;
                        if actual_role != role {
                            return Err(integrity("pairing class endpoint role changed"));
                        }
                        Ok(OnTheFlyPairingEndpointV1 {
                            source_slot,
                            source_contract_digest: digest,
                        })
                    })
                    .collect()
            };
            OnTheFlyPairingClassV1::new(
                class.species.as_str(),
                parse_digest(&class.proof_digest, "pairing class proof digest")?,
                endpoints(
                    &class.fundamental_source_slots,
                    OnTheFlyExternalColorRoleV1::Fundamental,
                )?,
                endpoints(
                    &class.antifundamental_source_slots,
                    OnTheFlyExternalColorRoleV1::Antifundamental,
                )?,
            )
        })
        .collect()
}

fn parse_digest(value: &str, label: &str) -> RusticolResult<SemanticDigest> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(invalid(format!(
            "{label} must be a lowercase SHA-256 digest"
        )));
    }
    let mut bytes = [0_u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| invalid(format!("{label} is not hexadecimal")))?;
    }
    SemanticDigest::new(bytes).map_err(|error| RusticolError::integrity(error.message()))
}

fn merge_equal<T: Copy + Eq>(slot: &mut Option<T>, value: T, message: &str) -> RusticolResult<()> {
    match slot {
        None => *slot = Some(value),
        Some(previous) if *previous == value => {}
        Some(_) => return Err(integrity(message)),
    }
    Ok(())
}

fn merge_equal_option<T: Copy + Eq>(
    slot: &mut Option<T>,
    value: Option<T>,
    message: &str,
) -> RusticolResult<()> {
    match (*slot, value) {
        (None, Some(value)) => *slot = Some(value),
        (None, None) | (Some(_), None) => {}
        (Some(previous), Some(value)) if previous == value => {}
        (Some(_), Some(_)) => return Err(integrity(message)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::on_the_fly::seed_codec::{
        decode_on_the_fly_process_seed_v1, encode_on_the_fly_process_seed_v1,
    };
    use crate::recurrence::on_the_fly::{OnTheFlyForbiddenWorkGuardV1, scalar_adapter_test_seed};
    use crate::recurrence::template::{
        CouplingOrderTermRow, IndexedRangeRow, ParameterKind, ParameterRow, ParameterValueType,
    };
    use crate::recurrence::{
        CheckedTableRange, PreparedDirectExecutorBinding, validated_template_fixture,
    };
    use serde_json::{Value, json};

    fn digest(seed: u8) -> SemanticDigest {
        SemanticDigest::new([seed; 32]).unwrap()
    }

    fn scalar_inputs() -> (
        Value,
        ValidatedRecurrenceTemplateInput,
        PreparedDirectExecutorCatalog,
    ) {
        let mut input = validated_template_fixture().into_input();
        input.coupling_order_ranges.push(IndexedRangeRow {
            id: 1,
            range: CheckedTableRange::new(0, 1),
        });
        input.coupling_order_terms.push(CouplingOrderTermRow {
            set_id: 1,
            name_string_id: 0,
            power: 1,
        });
        let templates = input.validate().unwrap();
        let direct = PreparedDirectExecutorCatalog::new(
            digest(40),
            vec![
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 3, 1),
                PreparedDirectExecutorBinding::identity_finalizer(2),
            ],
        )
        .unwrap();
        let source = json!({
            "schema_version": 1,
            "process_digest": digest(91).to_string(),
            "external_permutation": [0, 1],
            "external_sources": [
                {
                    "source_slot": 0,
                    "public_label": 0,
                    "is_initial": false,
                    "states": [{
                        "state_index": 0,
                        "public_helicity": 0,
                        "source_helicity": 0,
                        "source_template_id": 0,
                        "current_state_template_id": 0,
                        "momentum_sign": 1,
                        "crossing_phase": {
                            "real_numerator": "1", "real_denominator": "1",
                            "imag_numerator": "0", "imag_denominator": "1"
                        },
                        "spin_state": 50000,
                        "chirality": 0
                    }]
                },
                {
                    "source_slot": 1,
                    "public_label": 1,
                    "is_initial": false,
                    "states": [{
                        "state_index": 0,
                        "public_helicity": 0,
                        "source_helicity": 0,
                        "source_template_id": 0,
                        "current_state_template_id": 0,
                        "momentum_sign": 1,
                        "crossing_phase": {
                            "real_numerator": "1", "real_denominator": "1",
                            "imag_numerator": "0", "imag_denominator": "1"
                        },
                        "spin_state": 50000,
                        "chirality": 0
                    }]
                }
            ],
            "parameter_projection": [],
            "coupling_order_policy": "explicit",
            "coupling_hierarchies": [{"name": "0", "hierarchy": 1}],
            "coupling_limits": [{"name": "0", "maximum": 0}],
            "fermion_pairing": null,
            "normalization": {
                "factor": {
                    "real_numerator": "1", "real_denominator": "1",
                    "imag_numerator": "0", "imag_denominator": "1"
                },
                "convention": "raw-amplitude-test",
                "semantic_digest": digest(92).to_string()
            }
        });
        (source, templates, direct)
    }

    fn crossed_scalar_inputs() -> (
        Value,
        ValidatedRecurrenceTemplateInput,
        PreparedDirectExecutorCatalog,
    ) {
        let (mut projection, templates, direct) = scalar_inputs();
        let mut input = templates.into_input();
        let crossing = serde_json::to_vec(&json!({
            "chirality_factor": -1,
            "helicity_factor": -1,
            "momentum_transform": "negate-four-momentum",
            "phase": {
                "imag_denominator": "1",
                "imag_numerator": "0",
                "real_denominator": "1",
                "real_numerator": "-1"
            },
            "spin_state_factor": -1
        }))
        .unwrap();
        let previous = input.string_ranges.last().unwrap();
        let previous = previous
            .as_usize_range(input.string_bytes.len(), "template string")
            .unwrap();
        assert!(&input.string_bytes[previous] < crossing.as_slice());
        let crossing_string_id = u32::try_from(input.string_ranges.len()).unwrap();
        input.string_ranges.push(CheckedTableRange::new(
            u64::try_from(input.string_bytes.len()).unwrap(),
            u64::try_from(crossing.len()).unwrap(),
        ));
        input.string_bytes.extend_from_slice(&crossing);
        input.sources[0].crossing_string_id = crossing_string_id;
        input.sources[0].helicity = 1;
        let templates = input.validate().unwrap();

        for source_slot in 0..2 {
            projection["external_sources"][source_slot]["is_initial"] = json!(true);
            let state = &mut projection["external_sources"][source_slot]["states"][0];
            state["public_helicity"] = json!(-1);
            state["source_helicity"] = json!(-1);
            state["momentum_sign"] = json!(-1);
            state["crossing_phase"]["real_numerator"] = json!("-1");
            state["spin_state"] = json!(-50_000);
        }
        (projection, templates, direct)
    }

    fn mass_bound_scalar_inputs() -> (
        Value,
        ValidatedRecurrenceTemplateInput,
        PreparedDirectExecutorCatalog,
    ) {
        let (mut projection, templates, direct) = scalar_inputs();
        let mut input = templates.into_input();
        let source_evaluator = input.evaluator_bindings[0];
        input.catalog_header[0].parameter_count = 1;
        input.parameters.push(ParameterRow {
            id: 0,
            template_string_id: source_evaluator.resolver_key_string_id,
            name_string_id: source_evaluator.resolver_key_string_id,
            kind: ParameterKind::External as u8,
            value_type: ParameterValueType::Real as u8,
            mutable: 1,
            default_factor_id: 0,
            exact_expression_digest_id: MISSING_U32,
            dependency_sequence_id: source_evaluator.input_state_sequence_id,
            prepared_parameter_id: 0,
            semantic_digest_id: source_evaluator.callable_signature_digest_id,
        });
        input.current_states[0].mass_parameter_id = 0;
        input.sources[0].mass_parameter_id = 0;
        projection["parameter_projection"] = json!([{
            "parameter_template_id": 0,
            "prepared_parameter_id": 0,
            "component": 0
        }]);
        (projection, input.validate().unwrap(), direct)
    }

    fn replace_template_string(
        input: &mut crate::recurrence::template::OwnedRecurrenceTemplateInput,
        old: &str,
        new: &str,
    ) -> u32 {
        let mut strings = input
            .string_ranges
            .iter()
            .map(|range| {
                let range = range
                    .as_usize_range(input.string_bytes.len(), "test template string")
                    .unwrap();
                std::str::from_utf8(&input.string_bytes[range])
                    .unwrap()
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let id = strings.iter().position(|value| value == old).unwrap();
        strings[id] = new.to_owned();
        assert!(strings.windows(2).all(|pair| pair[0] < pair[1]));

        input.string_ranges.clear();
        input.string_bytes.clear();
        for value in strings {
            input.string_ranges.push(CheckedTableRange::new(
                u64::try_from(input.string_bytes.len()).unwrap(),
                u64::try_from(value.len()).unwrap(),
            ));
            input.string_bytes.extend_from_slice(value.as_bytes());
        }
        u32::try_from(id).unwrap()
    }

    fn legacy_null_mass_inputs(
        particle_id: i32,
        anti_particle_id: i32,
        orientation: CurrentOrientation,
    ) -> (
        Value,
        ValidatedRecurrenceTemplateInput,
        PreparedDirectExecutorCatalog,
    ) {
        let (mut projection, templates, direct) = scalar_inputs();
        let mut input = templates.into_input();
        let source_evaluator = input.evaluator_bindings[0];
        let mass_name_string_id =
            replace_template_string(&mut input, "projection", "particle.6.mass");
        input.catalog_header[0].parameter_count = 1;
        input.parameters.push(ParameterRow {
            id: 0,
            template_string_id: source_evaluator.resolver_key_string_id,
            name_string_id: mass_name_string_id,
            kind: ParameterKind::External as u8,
            value_type: ParameterValueType::Real as u8,
            mutable: 1,
            default_factor_id: 0,
            exact_expression_digest_id: MISSING_U32,
            dependency_sequence_id: source_evaluator.input_state_sequence_id,
            prepared_parameter_id: 0,
            semantic_digest_id: source_evaluator.callable_signature_digest_id,
        });
        input.current_states[0].particle_id = particle_id;
        input.current_states[0].anti_particle_id = anti_particle_id;
        input.current_states[0].orientation = orientation as u8;
        input.flavour_flow_values[0] = particle_id;
        assert_eq!(input.current_states[0].mass_parameter_id, MISSING_U32);
        assert_eq!(input.sources[0].mass_parameter_id, MISSING_U32);
        projection["parameter_projection"] = json!([{
            "parameter_template_id": 0,
            "prepared_parameter_id": 0,
            "component": 0
        }]);
        (projection, input.validate().unwrap(), direct)
    }

    fn build_test_seed(
        source: &Value,
        templates: &ValidatedRecurrenceTemplateInput,
        direct: &PreparedDirectExecutorCatalog,
    ) -> RusticolResult<OnTheFlyProcessSeedV1> {
        build_on_the_fly_process_seed_v1(
            parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(source).unwrap())?,
            templates,
            direct,
            templates.summary().prepared_kernel_pack_digest,
        )
    }

    #[test]
    fn compact_projection_matches_the_established_seed_codec_exactly() {
        let (source, templates, direct) = scalar_inputs();
        let bytes = serde_json::to_vec(&source).unwrap();
        let projection = parse_on_the_fly_process_seed_projection_v1(&bytes).unwrap();
        let guard = OnTheFlyForbiddenWorkGuardV1::begin().unwrap();
        let seed = build_on_the_fly_process_seed_v1(
            projection,
            &templates,
            &direct,
            templates.summary().prepared_kernel_pack_digest,
        )
        .unwrap();
        let forbidden = guard.finish().unwrap();
        assert_eq!(forbidden.direct_plan_load_attempts, 0);
        assert_eq!(forbidden.direct_plan_decode_attempts, 0);
        assert_eq!(forbidden.direct_plan_materialization_attempts, 0);
        assert_eq!(forbidden.established_builder_attempts, 0);

        let expected = scalar_adapter_test_seed(
            templates.summary().compiled_model_digest,
            templates.summary().catalog_digest,
            templates.summary().prepared_kernel_pack_digest,
            direct.direct_template_catalog_digest(),
        )
        .unwrap();
        assert_eq!(seed, expected);
        assert_eq!(
            encode_on_the_fly_process_seed_v1(&seed).unwrap(),
            encode_on_the_fly_process_seed_v1(&expected).unwrap()
        );
    }

    #[test]
    fn crossed_execution_state_is_authenticated_and_round_trips_exactly() {
        let (source, templates, direct) = crossed_scalar_inputs();
        let build = |value: &Value| {
            build_on_the_fly_process_seed_v1(
                parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(value).unwrap())?,
                &templates,
                &direct,
                templates.summary().prepared_kernel_pack_digest,
            )
        };
        let seed = build(&source).unwrap();
        for anchor in seed.source_anchors() {
            let state = &anchor.states()[0];
            assert!(anchor.is_initial());
            assert_eq!(state.public_helicity, -1);
            assert_eq!(state.source_helicity, -1);
            assert_eq!(state.spin_state, -50_000);
            assert_eq!(state.momentum_sign, -1);
            assert_eq!(
                state.crossing_phase,
                ExactComplexRational::ONE.checked_neg().unwrap()
            );
        }
        let encoded = encode_on_the_fly_process_seed_v1(&seed).unwrap();
        let decoded = decode_on_the_fly_process_seed_v1(&encoded).unwrap();
        assert_eq!(decoded, seed);
        assert_eq!(
            encode_on_the_fly_process_seed_v1(&decoded).unwrap(),
            encoded
        );

        for (field, stale) in [
            ("public_helicity", json!(1)),
            ("source_helicity", json!(1)),
            ("spin_state", json!(50_000)),
            ("momentum_sign", json!(1)),
        ] {
            let mut tampered = source.clone();
            tampered["external_sources"][0]["states"][0][field] = stale;
            assert!(
                build(&tampered)
                    .unwrap_err()
                    .message()
                    .contains("crossed source execution values disagree")
            );
        }
        let mut stale_phase = source.clone();
        stale_phase["external_sources"][0]["states"][0]["crossing_phase"]["real_numerator"] =
            json!("1");
        assert!(
            build(&stale_phase)
                .unwrap_err()
                .message()
                .contains("crossed source execution values disagree")
        );
        let mut stale_chirality = source.clone();
        stale_chirality["external_sources"][0]["states"][0]["chirality"] = json!(1);
        assert!(
            build(&stale_chirality)
                .unwrap_err()
                .message()
                .contains("process source chirality")
        );
    }

    #[test]
    fn compact_mass_binding_projects_one_prepared_slot_and_fails_closed() {
        let (source, templates, direct) = mass_bound_scalar_inputs();
        let build = |value: &Value| {
            build_on_the_fly_process_seed_v1(
                parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(value).unwrap())?,
                &templates,
                &direct,
                templates.summary().prepared_kernel_pack_digest,
            )
        };

        let seed = build(&source).unwrap();
        assert!(seed.source_anchors().iter().all(|anchor| {
            anchor
                .states()
                .iter()
                .all(|state| state.prepared_mass_parameter_slot == Some(0))
        }));
        let encoded = encode_on_the_fly_process_seed_v1(&seed).unwrap();
        assert_eq!(decode_on_the_fly_process_seed_v1(&encoded).unwrap(), seed);

        let mut missing = source.clone();
        missing["parameter_projection"] = json!([]);
        assert!(
            build(&missing)
                .unwrap_err()
                .message()
                .contains("component-0 projections")
        );

        let mut mismatched = source;
        mismatched["parameter_projection"][0]["prepared_parameter_id"] = json!(2);
        assert!(
            build(&mismatched)
                .unwrap_err()
                .message()
                .contains("mass projection disagrees with the prepared template catalog")
        );
    }

    #[test]
    fn cold_mass_binding_rejects_wrong_unique_and_missing_prepared_slots() {
        let (source, templates, direct) = mass_bound_scalar_inputs();
        let mut seed = build_test_seed(&source, &templates, &direct).unwrap();
        let mut input = templates.into_input();
        let second_evaluator = input.evaluator_bindings[1];
        let mut second_parameter = input.parameters[0];
        second_parameter.id = 1;
        second_parameter.template_string_id = second_evaluator.resolver_key_string_id;
        second_parameter.prepared_parameter_id = 1;
        second_parameter.semantic_digest_id = second_evaluator.callable_signature_digest_id;
        input.parameters.push(second_parameter);
        input.catalog_header[0].parameter_count = 2;
        let templates = input.validate().unwrap();
        let projection = [(0, Some(0), 0), (1, Some(1), 0)];

        validate_on_the_fly_source_mass_bindings_v1(&seed, &templates, &projection)
            .expect("the source mass parameter owns prepared slot zero");

        seed.source_anchors[0].states[0].prepared_mass_parameter_slot = Some(1);
        let wrong = validate_on_the_fly_source_mass_bindings_v1(&seed, &templates, &projection)
            .expect_err("another uniquely owned prepared slot is not this source's mass");
        assert!(
            wrong
                .to_string()
                .contains("authenticated recurrence template")
        );

        seed.source_anchors[0].states[0].prepared_mass_parameter_slot = None;
        let missing = validate_on_the_fly_source_mass_bindings_v1(&seed, &templates, &projection)
            .expect_err("an explicit source mass parameter cannot lose its slot");
        assert!(
            missing
                .to_string()
                .contains("authenticated recurrence template")
        );
    }

    #[test]
    fn legacy_null_mass_binding_uses_authenticated_orientation_without_abs_shortcuts() {
        for (particle, anti_particle, orientation, expected) in [
            (6, -6, CurrentOrientation::Particle, Some(0)),
            (-6, 6, CurrentOrientation::Antiparticle, Some(0)),
            (-6, 6, CurrentOrientation::Particle, None),
            (1, -1, CurrentOrientation::Particle, None),
            (21, 21, CurrentOrientation::SelfConjugate, None),
        ] {
            let (source, templates, direct) =
                legacy_null_mass_inputs(particle, anti_particle, orientation);
            let seed = build_test_seed(&source, &templates, &direct).unwrap();
            assert!(seed.source_anchors().iter().all(|anchor| {
                anchor
                    .states()
                    .iter()
                    .all(|state| state.prepared_mass_parameter_slot == expected)
            }));
            let encoded = encode_on_the_fly_process_seed_v1(&seed).unwrap();
            assert_eq!(decode_on_the_fly_process_seed_v1(&encoded).unwrap(), seed);
        }
    }

    #[test]
    fn legacy_null_mass_binding_fails_closed_on_ambiguity_type_and_orientation_conflicts() {
        let (source, templates, direct) =
            legacy_null_mass_inputs(6, -6, CurrentOrientation::Particle);

        let mut ambiguous_input = templates.clone().into_input();
        let second_evaluator = ambiguous_input.evaluator_bindings[1];
        let mut second_parameter = ambiguous_input.parameters[0];
        second_parameter.id = 1;
        second_parameter.template_string_id = second_evaluator.resolver_key_string_id;
        second_parameter.prepared_parameter_id = 1;
        second_parameter.semantic_digest_id = second_evaluator.callable_signature_digest_id;
        ambiguous_input.parameters.push(second_parameter);
        ambiguous_input.catalog_header[0].parameter_count = 2;
        let ambiguous_templates = ambiguous_input.validate().unwrap();
        let error = build_test_seed(&source, &ambiguous_templates, &direct).unwrap_err();
        assert!(
            error
                .message()
                .contains("mass parameter \"particle.6.mass\" is ambiguous")
        );

        let mut complex_input = templates.clone().into_input();
        complex_input.parameters[0].value_type = ParameterValueType::Complex as u8;
        let complex_templates = complex_input.validate().unwrap();
        assert!(
            build_test_seed(&source, &complex_templates, &direct)
                .unwrap_err()
                .message()
                .contains("mass parameter is not real")
        );

        let mut out_of_range_source = source.clone();
        out_of_range_source["parameter_projection"][0]["prepared_parameter_id"] = json!(1);
        let mut out_of_range_input = templates.clone().into_input();
        out_of_range_input.parameters[0].prepared_parameter_id = 1;
        let out_of_range_templates = out_of_range_input.validate().unwrap();
        assert!(
            build_test_seed(&out_of_range_source, &out_of_range_templates, &direct)
                .unwrap_err()
                .message()
                .contains("outside the dense parameter arena")
        );

        let (self_conjugate_source, self_conjugate_templates, self_conjugate_direct) =
            legacy_null_mass_inputs(21, 22, CurrentOrientation::SelfConjugate);
        assert!(
            build_test_seed(
                &self_conjugate_source,
                &self_conjugate_templates,
                &self_conjugate_direct,
            )
            .unwrap_err()
            .message()
            .contains("self-conjugate source mass orientation")
        );

        let mut mismatched_input = templates.into_input();
        mismatched_input.sources[0].mass_parameter_id = 0;
        let mismatched_templates = mismatched_input.validate().unwrap();
        assert!(
            build_test_seed(&source, &mismatched_templates, &direct)
                .unwrap_err()
                .message()
                .contains("source and current state disagree")
        );
    }

    #[test]
    fn minimal_projection_keeps_unspecified_model_orders_unbounded_for_cold_resolution() {
        let (mut source, templates, direct) = scalar_inputs();
        source["coupling_order_policy"] = json!("minimal");
        source["coupling_limits"] = json!([]);
        let projection =
            parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(&source).unwrap())
                .unwrap();
        let seed = build_on_the_fly_process_seed_v1(
            projection,
            &templates,
            &direct,
            templates.summary().prepared_kernel_pack_digest,
        )
        .unwrap();

        assert_eq!(
            seed.coupling_order_policy(),
            OnTheFlyCouplingOrderPolicyV1::Minimal
        );
        assert_eq!(seed.coupling_hierarchies(), [1]);
        assert_eq!(seed.explicit_coupling_limits(), [None]);
    }

    #[test]
    fn ordered_process_seed_batch_is_deterministic_and_builds_one_template_catalog() {
        let (first_source, templates, direct) = scalar_inputs();
        let mut second_source = first_source.clone();
        second_source["process_digest"] = json!(digest(93).to_string());
        let pack_digest = templates.summary().prepared_kernel_pack_digest;
        let parse = |value: &Value| {
            parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(value).unwrap())
                .unwrap()
        };
        let singleton_bytes = |value: &Value| {
            encode_on_the_fly_process_seed_v1(
                &build_on_the_fly_process_seed_v1(parse(value), &templates, &direct, pack_digest)
                    .unwrap(),
            )
            .unwrap()
        };
        let expected = [
            singleton_bytes(&second_source),
            singleton_bytes(&first_source),
        ];

        PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| count.set(0));
        let guard = OnTheFlyForbiddenWorkGuardV1::begin().unwrap();
        let first = build_on_the_fly_process_seeds_v1(
            vec![parse(&second_source), parse(&first_source)],
            &templates,
            &direct,
            pack_digest,
        )
        .unwrap()
        .iter()
        .map(encode_on_the_fly_process_seed_v1)
        .collect::<RusticolResult<Vec<_>>>()
        .unwrap();
        let forbidden = guard.finish().unwrap();
        PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| assert_eq!(count.get(), 1));

        PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| count.set(0));
        let second = build_on_the_fly_process_seeds_v1(
            vec![parse(&second_source), parse(&first_source)],
            &templates,
            &direct,
            pack_digest,
        )
        .unwrap()
        .iter()
        .map(encode_on_the_fly_process_seed_v1)
        .collect::<RusticolResult<Vec<_>>>()
        .unwrap();
        PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| assert_eq!(count.get(), 1));

        assert_eq!(first, expected);
        assert_eq!(second, expected);
        assert_eq!(forbidden.direct_plan_load_attempts, 0);
        assert_eq!(forbidden.direct_plan_decode_attempts, 0);
        assert_eq!(forbidden.direct_plan_materialization_attempts, 0);
        assert_eq!(forbidden.established_builder_attempts, 0);
    }

    #[test]
    fn ordered_process_seed_batch_attributes_failure_to_index_and_identity() {
        let (source, templates, direct) = scalar_inputs();
        let mut invalid_source = source.clone();
        invalid_source["process_digest"] = json!(digest(93).to_string());
        invalid_source["external_sources"][0]["states"][0]["source_template_id"] = json!(99);
        let parse = |value: &Value| {
            parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(value).unwrap())
                .unwrap()
        };

        let error = build_on_the_fly_process_seeds_v1(
            vec![parse(&source), parse(&invalid_source)],
            &templates,
            &direct,
            templates.summary().prepared_kernel_pack_digest,
        )
        .unwrap_err();

        assert!(error.message().contains("process seed at index 1"));
        assert!(error.message().contains(&digest(93).to_string()));
        assert!(error.message().contains("source template is absent"));
    }

    #[cfg(feature = "python-generation-bridge")]
    #[test]
    fn process_seed_byte_singleton_matches_one_element_batch_exactly() {
        let (source, templates, direct) = scalar_inputs();
        let source_bytes = serde_json::to_vec(&source).unwrap();
        let pack_digest = templates.summary().prepared_kernel_pack_digest;

        let singleton = crate::__private::build_on_the_fly_process_seed_bytes_v1(
            &source_bytes,
            &templates,
            &direct,
            pack_digest,
        )
        .unwrap();
        let batch = crate::__private::build_on_the_fly_process_seed_bytes_batch_v1(
            &[source_bytes],
            &templates,
            &direct,
            pack_digest,
        )
        .unwrap();
        assert_eq!(batch, vec![singleton]);

        let malformed = b"{".to_vec();
        let singleton_error = crate::__private::build_on_the_fly_process_seed_bytes_v1(
            &malformed,
            &templates,
            &direct,
            pack_digest,
        )
        .unwrap_err();
        let batch_error = crate::__private::build_on_the_fly_process_seed_bytes_batch_v1(
            &[malformed],
            &templates,
            &direct,
            pack_digest,
        )
        .unwrap_err();
        assert_eq!(singleton_error, batch_error);
    }

    #[cfg(feature = "python-generation-bridge")]
    #[test]
    fn empty_process_seed_byte_batch_builds_one_template_catalog() {
        let (_, templates, direct) = scalar_inputs();
        PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| count.set(0));

        let batch = crate::__private::build_on_the_fly_process_seed_bytes_batch_v1(
            &[],
            &templates,
            &direct,
            templates.summary().prepared_kernel_pack_digest,
        )
        .unwrap();

        assert!(batch.is_empty());
        PROCESS_SEED_TEMPLATE_CATALOG_BUILD_COUNT.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    fn compact_projection_is_deterministic_and_tampering_fails_closed() {
        let (source, templates, direct) = scalar_inputs();
        let build = |value: &Value| {
            let projection =
                parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(value).unwrap())?;
            let seed = build_on_the_fly_process_seed_v1(
                projection,
                &templates,
                &direct,
                templates.summary().prepared_kernel_pack_digest,
            )?;
            encode_on_the_fly_process_seed_v1(&seed)
        };
        assert_eq!(build(&source).unwrap(), build(&source).unwrap());

        let mut stale_source = source.clone();
        stale_source["external_sources"][0]["states"][0]["source_template_id"] = json!(99);
        assert!(build(&stale_source).is_err());

        let mut stale_pack = templates
            .summary()
            .prepared_kernel_pack_digest
            .as_bytes()
            .to_owned();
        stale_pack[0] ^= 0xff;
        let projection =
            parse_on_the_fly_process_seed_projection_v1(&serde_json::to_vec(&source).unwrap())
                .unwrap();
        assert!(
            build_on_the_fly_process_seed_v1(
                projection,
                &templates,
                &direct,
                SemanticDigest::new(stale_pack).unwrap(),
            )
            .is_err()
        );
    }
}
