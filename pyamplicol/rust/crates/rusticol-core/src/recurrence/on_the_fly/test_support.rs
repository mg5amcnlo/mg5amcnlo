// SPDX-License-Identifier: 0BSD

#![allow(dead_code)] // This module is intentionally a non-default test-support surface.

//! Feature-gated development oracle for the compact on-the-fly builder.
//!
//! This is the only on-the-fly module allowed to consume the materialized
//! recurrence builder input.  It maps exactly one authenticated public row
//! into the compact seed/query API, then compares that query-local trace with
//! the established builder.  Release/default builds do not compile this file.

use super::source_seed::validate_permutation;
use super::trace::{contribution_proof_digest, hash_current_key, multiset_digest};
use super::*;
use crate::recurrence::construct::TemplateCatalog;
use crate::recurrence::process::{
    OwnedFermionPairingInput, OwnedRecurrenceProcessInput, ProcessLCSectorKind,
    ProcessSourceStateRow,
};
use crate::recurrence::template::{CurrentOrientation, CurrentStateRow, ParticleStatistics};
use crate::recurrence::{
    AuthenticatedRecurrenceBuilderInput, ConstructionTransitionDiagnosticRowV1, RecurrenceProgram,
    RecurrenceStrategy,
};

#[derive(Clone, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct OnTheFlyTestSupportReportV1 {
    pub seed_digest: SemanticDigest,
    pub query_digest: SemanticDigest,
    pub selector_digest: SemanticDigest,
    pub trace_digest: SemanticDigest,
    pub current_count: u32,
    pub contribution_count: u32,
    pub closure_count: u32,
    pub established_current_count: u32,
    pub established_contribution_count: u32,
    pub established_closure_count: u32,
    pub current_multiset_digest: SemanticDigest,
    pub established_current_multiset_digest: SemanticDigest,
    pub contribution_multiset_digest: SemanticDigest,
    pub established_contribution_multiset_digest: SemanticDigest,
    pub closure_multiset_digest: SemanticDigest,
    pub closure_parity_multiset_digest: SemanticDigest,
    pub established_closure_parity_multiset_digest: SemanticDigest,
    pub negative_contribution_factor_count: u32,
    pub established_negative_contribution_factor_count: u32,
    pub source_domain_equal: bool,
    pub pairing_oracle_equal: bool,
    pub pairing_fermion_parities: Vec<i32>,
    pub established_pairing_fermion_parities: Vec<i32>,
    pub workspace_capacity_independent: bool,
    pub compact_transition_candidates: Vec<ConstructionTransitionDiagnosticRowV1>,
    pub established_transition_candidates: Vec<ConstructionTransitionDiagnosticRowV1>,
}

impl OnTheFlyTestSupportReportV1 {
    pub fn structural_parity(&self) -> bool {
        self.current_count == self.established_current_count
            && self.contribution_count == self.established_contribution_count
            && self.closure_count == self.established_closure_count
            && self.current_multiset_digest == self.established_current_multiset_digest
            && self.contribution_multiset_digest == self.established_contribution_multiset_digest
            && self.closure_parity_multiset_digest
                == self.established_closure_parity_multiset_digest
            && self.source_domain_equal
            && self.pairing_oracle_equal
    }
}

struct OnePublicRowOracle {
    external_permutation: Vec<u32>,
    public_helicities: Vec<i32>,
    selector: OnTheFlyLcSelectorV1,
}

/// Test-support convenience bundle retaining one generated compact seed.
/// Production families instead keep one lane-owned seed and build borrowed
/// [`OnTheFlySelectedQueryTraceV1`] values for each selected query.
pub(crate) struct OnTheFlySelectedTraceV1 {
    pub(crate) seed: OnTheFlyProcessSeedV1,
    pub(crate) query: DecodedLcQueryV1,
    pub(crate) trace: OnTheFlyStructuralTraceV1,
    pub(crate) projection: OnTheFlyProjectionProbeV1,
}

fn process_string<'a>(
    process: &'a OwnedRecurrenceProcessInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<&'a str> {
    let range = process
        .string_ranges
        .get(string_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} string is absent")))?;
    let bytes = &process.string_bytes[range.as_usize_range(process.string_bytes.len(), label)?];
    std::str::from_utf8(bytes)
        .map_err(|error| integrity(format!("{label} string is not UTF-8: {error}")))
}

fn process_factor(
    process: &OwnedRecurrenceProcessInput,
    factor_id: u32,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    let row = process
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| integrity(format!("{label} factor is absent")))?;
    if row.id != factor_id {
        return Err(integrity(format!(
            "{label} factor catalog is not canonical"
        )));
    }
    ExactComplexRational::parse_parts(
        process_string(process, row.real_numerator_string_id, label)?,
        process_string(process, row.real_denominator_string_id, label)?,
        process_string(process, row.imag_numerator_string_id, label)?,
        process_string(process, row.imag_denominator_string_id, label)?,
    )
    .map_err(|error| integrity(format!("{label} exact factor is invalid: {error}")))
}

fn process_sequence<'a>(
    process: &'a OwnedRecurrenceProcessInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let range = process
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} sequence is absent")))?;
    Ok(&process.u32_sequence_values
        [range.as_usize_range(process.u32_sequence_values.len(), label)?])
}

fn process_digest(
    process: &OwnedRecurrenceProcessInput,
    digest_id: u32,
    label: &str,
) -> RusticolResult<SemanticDigest> {
    let row = process
        .digest_catalog
        .get(digest_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} digest is absent")))?;
    if row.id != digest_id {
        return Err(integrity(format!(
            "{label} digest catalog is not canonical"
        )));
    }
    SemanticDigest::new(row.value)
        .map_err(|error| integrity(format!("{label} digest is invalid: {error}")))
}

fn pairing_string<'a>(
    input: &'a OwnedFermionPairingInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<&'a str> {
    let range = input
        .string_ranges
        .get(string_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} pairing string is absent")))?;
    let bytes = &input.string_bytes[range.as_usize_range(input.string_bytes.len(), label)?];
    std::str::from_utf8(bytes)
        .map_err(|error| integrity(format!("{label} pairing string is not UTF-8: {error}")))
}

fn authenticated_source_family(
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
            "authenticated source family, statistics, and component dimension disagree",
        )),
    }
}

fn authenticated_source_orientation(
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

fn authenticated_prepared_mass_slot(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    catalog: &TemplateCatalog<'_>,
    source: SourceRow,
    current_state: CurrentStateRow,
) -> RusticolResult<Option<u32>> {
    super::source_builder::resolve_prepared_mass_slot(
        source,
        current_state,
        authenticated.template().input(),
        catalog,
        authenticated
            .process()
            .input()
            .parameter_projection
            .iter()
            .map(|row| {
                (
                    row.parameter_template_id,
                    row.prepared_parameter_id(),
                    row.component,
                )
            }),
    )
}

fn pairing_endpoint_contracts(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<BTreeMap<u32, (OnTheFlyExternalColorRoleV1, SemanticDigest)>> {
    let Some(pairing) = authenticated.process().input().fermion_pairing.as_ref() else {
        return Ok(BTreeMap::new());
    };
    let mut result = BTreeMap::new();
    for endpoint in &pairing.endpoints {
        let role = match endpoint.color_orientation {
            0 => OnTheFlyExternalColorRoleV1::Fundamental,
            1 => OnTheFlyExternalColorRoleV1::Antifundamental,
            _ => {
                return Err(integrity(
                    "pairing endpoint has an invalid color orientation",
                ));
            }
        };
        let digest = SemanticDigest::new(endpoint.contract_digest)
            .map_err(|error| integrity(format!("pairing endpoint digest is invalid: {error}")))?;
        if result
            .insert(endpoint.source_slot, (role, digest))
            .is_some()
        {
            return Err(integrity("pairing endpoint repeats a source slot"));
        }
    }
    Ok(result)
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
            "source color representation disagrees with the pairing endpoint catalog",
        )),
    }
}

fn authenticated_source_anchors(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<Vec<OnTheFlySourceAnchorV1>> {
    let process = authenticated.process().input();
    let selected_source_mode = authenticated.process().summary().selected_source_mode();
    let selected_states = process
        .selected_source_coverage
        .iter()
        .map(|row| (row.source_slot, row.source_state_index))
        .collect::<BTreeSet<_>>();
    let input = authenticated.template().input();
    let catalog = TemplateCatalog::new(input)?;
    let endpoint_contracts = pairing_endpoint_contracts(authenticated)?;
    let mut anchors = Vec::new();
    anchors
        .try_reserve_exact(process.external_legs.len())
        .map_err(|error| invalid(format!("source-anchor allocation failed: {error}")))?;
    for (source_slot, leg) in process.external_legs.iter().copied().enumerate() {
        if leg.source_slot as usize != source_slot {
            return Err(integrity("external source slots are not canonical"));
        }
        let range = leg.source_state_range.as_usize_range(
            process.source_states.len(),
            "on-the-fly authenticated source states",
        )?;
        let mut states = Vec::new();
        let mut role = None;
        let mut anchor_fermionic = None;
        let mut pairing_contract = None;
        for process_state in process.source_states[range]
            .iter()
            .copied()
            .filter(|state| {
                !selected_source_mode
                    || selected_states.contains(&(leg.source_slot, state.state_index))
            })
        {
            let source = *input
                .sources
                .get(process_state.source_template_id as usize)
                .ok_or_else(|| integrity("authenticated source template is absent"))?;
            let current_state = *input
                .current_states
                .get(process_state.current_state_template_id as usize)
                .ok_or_else(|| integrity("authenticated current-state template is absent"))?;
            if source.id != process_state.source_template_id
                || current_state.id != process_state.current_state_template_id
            {
                return Err(integrity(
                    "authenticated process source row differs from its template contract",
                ));
            }
            validate_crossed_source_state(leg.is_initial != 0, &process_state, source, input)?;
            let family = authenticated_source_family(&catalog, source, current_state)?;
            let (state_role, state_pairing_contract) = color_role(
                current_state.color_representation,
                endpoint_contracts.get(&leg.source_slot).copied(),
            )?;
            match role {
                None => role = Some(state_role),
                Some(previous) if previous == state_role => {}
                Some(_) => return Err(integrity("one source anchor mixes color roles")),
            }
            match anchor_fermionic {
                None => anchor_fermionic = Some(family.is_fermionic()),
                Some(previous) if previous == family.is_fermionic() => {}
                Some(_) => return Err(integrity("one source anchor mixes statistics")),
            }
            match pairing_contract {
                None => pairing_contract = state_pairing_contract,
                Some(previous) if Some(previous) == state_pairing_contract => {}
                Some(_) => return Err(integrity("one source anchor mixes pairing contracts")),
            }
            let seed = catalog.source_seed(source)?;
            states.push(OnTheFlySourceStateV1::new(
                process_state.state_index,
                process_state.public_helicity,
                process_state.public_helicity,
                source.id,
                current_state.id,
                catalog.digest(source.semantic_digest_id, "source semantic")?,
                catalog.digest(current_state.semantic_digest_id, "current-state semantic")?,
                process_state.momentum_sign,
                process_factor(
                    process,
                    process_state.crossing_phase_factor_id,
                    "source crossing phase",
                )?,
                process_state.spin_state,
                process_state.chirality,
                catalog
                    .flavour_flow(source.flavour_flow_id, "source flavour flow")?
                    .to_vec(),
                source.quantum_number_flow_id,
                seed.proof_digest(),
                family,
                authenticated_source_orientation(current_state)?,
                authenticated_prepared_mass_slot(authenticated, &catalog, source, current_state)?,
            )?);
        }
        anchors.push(OnTheFlySourceAnchorV1::new(
            leg.source_slot,
            leg.public_label,
            leg.is_initial != 0,
            role.ok_or_else(|| integrity("authenticated source anchor is empty"))?,
            anchor_fermionic.ok_or_else(|| integrity("source statistics are absent"))?,
            pairing_contract,
            states,
        )?);
    }
    Ok(anchors)
}

fn pairing_classes(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    contracts: &BTreeMap<u32, (OnTheFlyExternalColorRoleV1, SemanticDigest)>,
) -> RusticolResult<Vec<OnTheFlyPairingClassV1>> {
    let Some(input) = authenticated.process().input().fermion_pairing.as_ref() else {
        return Ok(Vec::new());
    };
    input
        .pairing_classes
        .iter()
        .map(|class| {
            let endpoints = |range: crate::recurrence::CheckedTableRange,
                             role: OnTheFlyExternalColorRoleV1,
                             label: &str|
             -> RusticolResult<Vec<OnTheFlyPairingEndpointV1>> {
                let values = match role {
                    OnTheFlyExternalColorRoleV1::Fundamental => &input.class_fundamental_slots,
                    OnTheFlyExternalColorRoleV1::Antifundamental => {
                        &input.class_antifundamental_slots
                    }
                    _ => unreachable!(),
                };
                values[range.as_usize_range(values.len(), label)?]
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
                pairing_string(input, class.species_string_id, "pairing class species")?,
                SemanticDigest::new(class.proof_digest).map_err(|error| {
                    integrity(format!("pairing class proof digest is invalid: {error}"))
                })?,
                endpoints(
                    class.fundamental_slot_range,
                    OnTheFlyExternalColorRoleV1::Fundamental,
                    "pairing class fundamental slots",
                )?,
                endpoints(
                    class.antifundamental_slot_range,
                    OnTheFlyExternalColorRoleV1::Antifundamental,
                    "pairing class antifundamental slots",
                )?,
            )
        })
        .collect()
}

fn compact_seed(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    direct_catalog_digest: SemanticDigest,
    external_permutation: Vec<u32>,
) -> RusticolResult<OnTheFlyProcessSeedV1> {
    if authenticated.process().summary().strategy() != RecurrenceStrategy::TopologyReplay {
        return Err(invalid(
            "the on-the-fly test oracle requires topology-replay input",
        ));
    }
    let process = authenticated.process().input();
    let normalization = *process
        .normalization
        .first()
        .ok_or_else(|| integrity("authenticated process normalization is absent"))?;
    if process.normalization.len() != 1 {
        return Err(integrity(
            "on-the-fly raw-amplitude input requires one normalization row",
        ));
    }
    let normalization_factor =
        process_factor(process, normalization.factor_id, "process normalization")?;
    if normalization_factor != ExactComplexRational::ONE {
        return Err(integrity(
            "on-the-fly raw-amplitude normalization must be exact one",
        ));
    }
    let contracts = pairing_endpoint_contracts(authenticated)?;
    let template_catalog = TemplateCatalog::new(authenticated.template().input())?;
    let mut process_limits = BTreeMap::new();
    for row in &process.coupling_limits {
        let name = process_string(process, row.name_string_id, "process coupling limit")?;
        process_limits.insert(name, row.maximum);
    }
    let coupling_limits = template_catalog
        .coupling_order_names()
        .iter()
        .map(|name| {
            process_limits.get(name).copied().map(Some).ok_or_else(|| {
                integrity(format!(
                    "authenticated process has no finalized coupling limit for model order {name:?} required by the prepared template catalog"
                ))
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    OnTheFlyProcessSeedV1::new(
        authenticated.process().semantic_identity().process_digest(),
        authenticated.template().summary().compiled_model_digest,
        authenticated.template().summary().catalog_digest,
        authenticated
            .template()
            .summary()
            .prepared_kernel_pack_digest,
        direct_catalog_digest,
        process_digest(
            process,
            normalization.semantic_digest_id,
            "process normalization semantic",
        )?,
        process_string(
            process,
            normalization.convention_string_id,
            "process normalization convention",
        )?,
        normalization_factor,
        authenticated_source_anchors(authenticated)?,
        external_permutation,
        OnTheFlyCouplingOrderPolicyV1::Explicit,
        vec![1; coupling_limits.len()],
        coupling_limits,
        pairing_classes(authenticated, &contracts)?,
    )
}

fn open_blocks_for_sector(
    process: &OwnedRecurrenceProcessInput,
    sector: crate::recurrence::process::ProcessPhysicalLCSectorRow,
) -> RusticolResult<Vec<Vec<u32>>> {
    let range = sector
        .open_string_range
        .as_usize_range(process.lc_open_strings.len(), "one-row open strings")?;
    process.lc_open_strings[range]
        .iter()
        .map(|row| {
            let mut block = vec![row.fundamental_source_slot];
            block.extend_from_slice(process_sequence(
                process,
                row.adjoint_sequence_id,
                "one-row open-string adjoints",
            )?);
            block.push(row.antifundamental_source_slot);
            Ok(block)
        })
        .collect()
}

fn established_pairing_rule_by_id(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    rule_id: Option<u32>,
) -> RusticolResult<Option<crate::recurrence::process::FermionPairingRuleRow>> {
    let Some(catalog) = authenticated.process().fermion_pairing_catalog() else {
        if rule_id.is_none() {
            return Ok(None);
        }
        return Err(integrity(
            "established pairing owner has no pairing catalog",
        ));
    };
    let Some(rule_id) = rule_id else {
        return Err(integrity(
            "established retained closure has no pairing owner",
        ));
    };
    catalog
        .rules()
        .iter()
        .copied()
        .find(|rule| rule.rule_id == rule_id)
        .map(Some)
        .ok_or_else(|| integrity(format!("established pairing owner {rule_id} is absent")))
}

fn one_public_row_oracle(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    selected_public_flow_id: u32,
    public_helicities: &[i32],
) -> RusticolResult<OnePublicRowOracle> {
    let process = authenticated.process().input();
    if public_helicities.len() != process.external_legs.len() {
        return Err(invalid(
            "one-row oracle requires one public helicity per external source",
        ));
    }
    let selected_mode = authenticated.process().summary().selected_flow_mode();
    if selected_mode
        && !process
            .selected_public_flow_coverage
            .iter()
            .any(|row| row.flow_id == selected_public_flow_id)
    {
        return Err(invalid(
            "requested public flow is absent from the selected generation slice",
        ));
    }
    let flow = process
        .public_lc_flows
        .iter()
        .copied()
        .find(|row| row.flow_id == selected_public_flow_id)
        .ok_or_else(|| invalid("requested public flow is absent"))?;
    let sector = *process
        .physical_lc_sectors
        .get(flow.construction_sector_id as usize)
        .ok_or_else(|| integrity("public row construction sector is absent"))?;
    if sector.sector_id != flow.construction_sector_id {
        return Err(integrity(
            "public row construction-sector catalog is not canonical",
        ));
    }
    let external_permutation = process_sequence(
        process,
        flow.source_slot_permutation_sequence_id,
        "one-row source permutation",
    )?
    .to_vec();
    validate_permutation(
        &external_permutation,
        process.external_legs.len(),
        "one-row external permutation",
    )?;
    let selector = match sector.kind()? {
        ProcessLCSectorKind::Singlet => OnTheFlyLcSelectorV1::Singlet,
        ProcessLCSectorKind::SingleTrace => OnTheFlyLcSelectorV1::single_trace(
            process_sequence(process, flow.word_sequence_id, "one-row public trace")?.to_vec(),
        ),
        ProcessLCSectorKind::OpenLines => {
            let construction_blocks = open_blocks_for_sector(process, sector)?;
            let public_blocks = construction_blocks
                .iter()
                .map(|block| {
                    block
                        .iter()
                        .map(|slot| {
                            external_permutation
                                .get(*slot as usize)
                                .copied()
                                .ok_or_else(|| integrity("one-row block source is out of range"))
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            OnTheFlyLcSelectorV1::open_lines(public_blocks)
        }
    };
    Ok(OnePublicRowOracle {
        external_permutation,
        public_helicities: public_helicities.to_vec(),
        selector,
    })
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SourceDomainRow {
    source_slot: u32,
    source_template_id: u32,
    spin_state_class: i32,
    family: u8,
    orientation: u8,
    source_helicity: i32,
    chirality: i32,
    prepared_mass_parameter_slot: Option<u32>,
}

fn compact_source_domain(seed: &OnTheFlyProcessSeedV1) -> Vec<SourceDomainRow> {
    let mut rows = seed
        .source_execution_specs()
        .map(|row| SourceDomainRow {
            source_slot: row.source_slot,
            source_template_id: row.source_template_id,
            spin_state_class: row.spin_state_class,
            family: row.family as u8,
            orientation: row.orientation as u8,
            source_helicity: row.helicity,
            chirality: row.chirality,
            prepared_mass_parameter_slot: row.prepared_mass_parameter_slot,
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

fn established_source_state_for_template(
    source_slot: u32,
    states: &[ProcessSourceStateRow],
    source_template_id: u32,
) -> RusticolResult<ProcessSourceStateRow> {
    let mut matches = states
        .iter()
        .copied()
        .filter(|state| state.source_template_id == source_template_id);
    let state = matches.next().ok_or_else(|| {
        integrity(format!(
            "established source slot {source_slot} template {source_template_id} is absent from its authenticated state domain"
        ))
    })?;
    if matches.next().is_some() {
        return Err(integrity(format!(
            "established source slot {source_slot} template {source_template_id} is ambiguous in its authenticated state domain"
        )));
    }
    Ok(state)
}

fn established_source_domain(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    program: &RecurrenceProgram,
) -> RusticolResult<Vec<SourceDomainRow>> {
    let input = authenticated.template().input();
    let catalog = TemplateCatalog::new(input)?;
    let mut rows = Vec::new();
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let key = current.key();
        let [source_slot] = key.support_source_slots() else {
            return Err(integrity(
                "established source current has non-unary support",
            ));
        };
        let source_template_id = match key.source_binding() {
            CurrentSourceBinding::FixedTemplate(value) => *value,
            _ => return Err(integrity("established source has no fixed source template")),
        };
        let source = *input
            .sources
            .get(source_template_id as usize)
            .ok_or_else(|| integrity("established source template is absent"))?;
        let current_state = *input
            .current_states
            .get(key.current_state_template_id() as usize)
            .ok_or_else(|| integrity("established source current state is absent"))?;
        let process = authenticated.process().input();
        let source_state_range = process
            .external_legs
            .get(*source_slot as usize)
            .ok_or_else(|| integrity("established source leg is absent"))?
            .source_state_range
            .as_usize_range(process.source_states.len(), "established source states")?;
        let source_helicity = established_source_state_for_template(
            *source_slot,
            &process.source_states[source_state_range],
            source_template_id,
        )?;
        if source_helicity.current_state_template_id != key.current_state_template_id() {
            return Err(integrity(format!(
                "established source state and current template disagree: source slot {}, source state {}, source template {}, authenticated current template {}, established current template {}",
                source_slot,
                source_helicity.state_index,
                source_template_id,
                source_helicity.current_state_template_id,
                key.current_state_template_id(),
            )));
        }
        rows.push(SourceDomainRow {
            source_slot: *source_slot,
            source_template_id,
            spin_state_class: key.spin_state_class(),
            family: authenticated_source_family(&catalog, source, current_state)? as u8,
            orientation: authenticated_source_orientation(current_state)? as u8,
            source_helicity: source_helicity.public_helicity,
            chirality: current_state.chirality,
            prepared_mass_parameter_slot: authenticated_prepared_mass_slot(
                authenticated,
                &catalog,
                source,
                current_state,
            )?,
        });
    }
    rows.sort_unstable();
    Ok(rows)
}

fn established_current_digests(program: &RecurrenceProgram) -> RusticolResult<Vec<SemanticDigest>> {
    program
        .currents()
        .iter()
        .map(|current| {
            let color = program
                .dynamic_color_states()
                .get(current.key().dynamic_lc_color_state_id().get() as usize)
                .ok_or_else(|| integrity("established current color state is absent"))?;
            hash_current_key(current.key(), color)
        })
        .collect()
}

fn established_contribution_digest(
    program: &RecurrenceProgram,
    current_digests: &[SemanticDigest],
) -> RusticolResult<SemanticDigest> {
    let mut rows = Vec::new();
    for contribution in program.contributions() {
        let [parent0, parent1] = contribution.parent_current_ids() else {
            return Err(integrity(
                "established selected-query contribution is not binary",
            ));
        };
        let result = *current_digests
            .get(contribution.result_current_id() as usize)
            .ok_or_else(|| integrity("established contribution result current is absent"))?;
        let parents = [
            *current_digests
                .get(*parent0 as usize)
                .ok_or_else(|| integrity("established contribution parent is absent"))?,
            *current_digests
                .get(*parent1 as usize)
                .ok_or_else(|| integrity("established contribution parent is absent"))?,
        ];
        rows.push(contribution_proof_digest(
            result,
            parents,
            contribution.key(),
            contribution.exact_factor(),
        )?);
    }
    multiset_digest(b"pyamplicol-on-the-fly-contribution-multiset-v1\0", rows)
}

fn closure_parity_row_digest(
    parent_digests: [SemanticDigest; 2],
    closure_template_id: u32,
    factor: ExactComplexRational,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-closure-parity-row-v1\0");
    hash_digest(&mut hash, parent_digests[0]);
    hash_digest(&mut hash, parent_digests[1]);
    hash.update(closure_template_id.to_le_bytes());
    hash_exact(&mut hash, factor);
    final_digest(hash)
}

fn compact_closure_parity_digest(
    trace: &OnTheFlyStructuralTraceV1,
) -> RusticolResult<SemanticDigest> {
    let mut current_by_component_base = BTreeMap::new();
    for (index, range) in trace.current_component_ranges.iter().copied().enumerate() {
        current_by_component_base.insert(range[0], trace.current_semantic_digests[index]);
    }
    let mut rows = Vec::new();
    for operation in trace.operations.iter() {
        let OnTheFlyTraceOperationV1::Closure { key, row } = operation else {
            continue;
        };
        let closure_template_id = key.operation_id().ok_or_else(|| {
            integrity("compact closure executor has no closure-template identity")
        })?;
        let factor = *trace
            .exact_factors
            .get(row.exact_factor_id as usize)
            .ok_or_else(|| integrity("compact closure factor is absent"))?;
        rows.push(closure_parity_row_digest(
            [
                *current_by_component_base
                    .get(&row.parent0_component_base)
                    .ok_or_else(|| integrity("compact closure parent 0 is absent"))?,
                *current_by_component_base
                    .get(&row.parent1_component_base_or_sentinel)
                    .ok_or_else(|| integrity("compact closure parent 1 is absent"))?,
            ],
            closure_template_id,
            factor,
        )?);
    }
    multiset_digest(b"pyamplicol-on-the-fly-closure-parity-multiset-v1\0", rows)
}

fn established_closure_parity_digest(
    program: &RecurrenceProgram,
    current_digests: &[SemanticDigest],
) -> RusticolResult<SemanticDigest> {
    let mut rows = Vec::new();
    for closure in program.closure_terms() {
        let [parent0, parent1] = closure.parent_current_ids() else {
            return Err(integrity(
                "established selected-query closure is not binary",
            ));
        };
        rows.push(closure_parity_row_digest(
            [
                *current_digests
                    .get(*parent0 as usize)
                    .ok_or_else(|| integrity("established closure parent 0 is absent"))?,
                *current_digests
                    .get(*parent1 as usize)
                    .ok_or_else(|| integrity("established closure parent 1 is absent"))?,
            ],
            closure.closure_template_id(),
            closure.exact_factor(),
        )?);
    }
    multiset_digest(b"pyamplicol-on-the-fly-closure-parity-multiset-v1\0", rows)
}

fn is_negative_one(value: ExactComplexRational) -> RusticolResult<bool> {
    Ok(value == ExactComplexRational::ONE.checked_neg()?)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PairingOracleRow {
    endpoint_pairs: Vec<[u32; 2]>,
    source_slot_permutation: Vec<u32>,
    source_lineage: Vec<u32>,
    fermion_parity: i32,
}

fn pairing_oracle_equal(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    owners: &[ResolvedPairingOwnerV1],
    rules: &[Option<crate::recurrence::process::FermionPairingRuleRow>],
) -> RusticolResult<(bool, Vec<i32>, Vec<i32>)> {
    let mut compact = owners
        .iter()
        .map(|owner| PairingOracleRow {
            endpoint_pairs: owner.endpoint_pairs.clone(),
            source_slot_permutation: owner.source_slot_permutation.clone(),
            source_lineage: owner.source_lineage.clone(),
            fermion_parity: owner.fermion_parity,
        })
        .collect::<Vec<_>>();
    let source_count = authenticated.process().input().external_legs.len();
    let identity = (0..source_count)
        .map(|value| checked_u32(value, "identity source slot"))
        .collect::<RusticolResult<Vec<_>>>()?;
    let catalog = authenticated.process().fermion_pairing_catalog();
    let mut established = rules
        .iter()
        .copied()
        .map(|rule| {
            let Some(rule) = rule else {
                return Ok(PairingOracleRow {
                    endpoint_pairs: Vec::new(),
                    source_slot_permutation: identity.clone(),
                    source_lineage: vec![MISSING_U32; source_count],
                    fermion_parity: 1,
                });
            };
            let catalog =
                catalog.ok_or_else(|| integrity("established pairing rule has no catalog"))?;
            let mut endpoint_pairs = catalog
                .endpoint_pairings(rule)?
                .iter()
                .map(|pair| {
                    [
                        pair.fundamental_source_slot,
                        pair.antifundamental_source_slot,
                    ]
                })
                .collect::<Vec<_>>();
            endpoint_pairs.sort_unstable();
            Ok(PairingOracleRow {
                endpoint_pairs,
                source_slot_permutation: catalog.source_slot_permutation(rule)?.to_vec(),
                source_lineage: catalog.lineage(rule)?.to_vec(),
                fermion_parity: rule.fermion_parity,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    compact.sort_unstable();
    established.sort_unstable();
    let compact_parities = compact.iter().map(|row| row.fermion_parity).collect();
    let established_parities = established.iter().map(|row| row.fermion_parity).collect();
    Ok((
        compact == established,
        compact_parities,
        established_parities,
    ))
}

pub(crate) fn build_on_the_fly_selected_trace_v1(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    direct: &PreparedDirectExecutorCatalog,
    selected_public_flow_id: u32,
    public_helicities: &[i32],
    enable_projection: bool,
) -> RusticolResult<OnTheFlySelectedTraceV1> {
    let oracle = one_public_row_oracle(authenticated, selected_public_flow_id, public_helicities)?;
    let seed = compact_seed(
        authenticated,
        direct.direct_template_catalog_digest(),
        oracle.external_permutation.clone(),
    )?;
    let query = DecodedLcQueryV1::new(
        &seed,
        oracle.external_permutation,
        &oracle.public_helicities,
        oracle.selector,
    )?;
    let selected = build_selected_lc_query_trace_for_probe_v1(
        authenticated.template(),
        direct,
        &seed,
        query,
        enable_projection,
        false,
    )?;
    Ok(OnTheFlySelectedTraceV1 {
        seed,
        query: selected.query,
        trace: selected.trace,
        projection: selected.projection,
    })
}

fn reconstruct_seed_in_projection_domain(
    retained_seed: OnTheFlyProcessSeedV1,
    artifact_seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<OnTheFlyProcessSeedV1> {
    let OnTheFlyProcessSeedV1 {
        process_digest,
        model_digest,
        template_catalog_digest,
        prepared_pack_digest,
        direct_catalog_digest,
        normalization_semantic_digest,
        normalization_convention,
        source_anchors,
        external_permutation: _,
        coupling_order_policy: _,
        coupling_hierarchies: _,
        coupling_limits: _,
        pairing_classes,
        semantic_digest: _,
    } = retained_seed;
    let reconstructed = OnTheFlyProcessSeedV1::new(
        process_digest,
        model_digest,
        template_catalog_digest,
        prepared_pack_digest,
        direct_catalog_digest,
        normalization_semantic_digest,
        normalization_convention,
        ExactComplexRational::ONE,
        source_anchors.into_vec(),
        artifact_seed.external_permutation().to_vec(),
        artifact_seed.coupling_order_policy(),
        artifact_seed.coupling_hierarchies().to_vec(),
        artifact_seed.explicit_coupling_limits().to_vec(),
        pairing_classes.into_vec(),
    )?;
    if reconstructed.semantic_digest() != artifact_seed.semantic_digest() {
        return Err(integrity(
            "retained diagnostic semantic constituents do not authenticate in the on-the-fly artifact seed projection domain",
        ));
    }
    Ok(reconstructed)
}

/// Build one retained recurrence row against an already-authenticated compact
/// artifact seed. The retained input supplies semantic constituents and the
/// public selector oracle; projection policy remains owned by the artifact.
pub(crate) fn build_on_the_fly_selected_trace_against_seed_v1(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    direct: &PreparedDirectExecutorCatalog,
    artifact_seed: &OnTheFlyProcessSeedV1,
    selected_public_flow_id: u32,
    public_helicities: &[i32],
    enable_projection: bool,
    enable_cyclic_trace_reflection: bool,
) -> RusticolResult<OnTheFlySelectedTraceV1> {
    let oracle = one_public_row_oracle(authenticated, selected_public_flow_id, public_helicities)?;
    let retained_seed = compact_seed(
        authenticated,
        direct.direct_template_catalog_digest(),
        oracle.external_permutation,
    )?;
    let _authenticated = reconstruct_seed_in_projection_domain(retained_seed, artifact_seed)?;
    let query = DecodedLcQueryV1::new(
        artifact_seed,
        artifact_seed.external_permutation().to_vec(),
        &oracle.public_helicities,
        oracle.selector,
    )?;
    let selected = build_selected_lc_query_trace_for_probe_v1(
        authenticated.template(),
        direct,
        artifact_seed,
        query,
        enable_projection,
        enable_cyclic_trace_reflection,
    )?;
    Ok(OnTheFlySelectedTraceV1 {
        seed: artifact_seed.clone(),
        query: selected.query,
        trace: selected.trace,
        projection: selected.projection,
    })
}

fn observed_transition_build<T>(
    selection: Option<crate::recurrence::diagnostic::ConstructionDiagnosticSelectionV1>,
    build: impl FnOnce() -> RusticolResult<T>,
) -> RusticolResult<(
    T,
    Vec<ConstructionTransitionDiagnosticRowV1>,
    Option<BTreeSet<SemanticDigest>>,
)> {
    crate::recurrence::diagnostic::begin_transition_diagnostic_observation(selection)?;
    let built = build();
    let observed = crate::recurrence::diagnostic::take_transition_diagnostic_observation();
    match (built, observed) {
        (Ok(value), Ok((rows, live))) => Ok((value, rows, live)),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

/// Build one compact selected query and compare it with the established
/// materialized builder. This API exists only under `on-the-fly-test-support`.
#[doc(hidden)]
pub fn on_the_fly_test_support_probe_v1(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    direct: &PreparedDirectExecutorCatalog,
    selected_public_flow_id: u32,
    public_helicities: &[i32],
) -> RusticolResult<OnTheFlyTestSupportReportV1> {
    let (compact_pre_projection, compact_transition_candidates, _) =
        observed_transition_build(None, || {
            build_on_the_fly_selected_trace_v1(
                authenticated,
                direct,
                selected_public_flow_id,
                public_helicities,
                false,
            )
        })?;
    let compact_live = compact_pre_projection
        .trace
        .current_semantic_digests
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let compact_transition_candidates = compact_transition_candidates
        .into_iter()
        .filter(|row| {
            compact_live.contains(&row.output_current_digest)
                && row
                    .ordered_parent_digests
                    .iter()
                    .all(|digest| compact_live.contains(digest))
        })
        .collect::<Vec<_>>();
    let selected = build_on_the_fly_selected_trace_v1(
        authenticated,
        direct,
        selected_public_flow_id,
        public_helicities,
        true,
    )?;
    let OnTheFlySelectedTraceV1 {
        seed,
        query,
        trace,
        projection: _,
    } = selected;
    crate::recurrence::construct::begin_established_pairing_owner_observation();
    let diagnostic_selection = crate::recurrence::diagnostic::ConstructionDiagnosticSelectionV1 {
        public_flow_id: selected_public_flow_id,
        public_helicities: public_helicities.to_vec(),
    };
    let (established, established_transition_candidates, established_live) =
        observed_transition_build(Some(diagnostic_selection), || authenticated.build())?;
    let established_live = established_live.ok_or_else(|| {
        integrity("established transition diagnostic did not record its selected live slice")
    })?;
    let established_materialized_sector_id = established
        .replay_targets()
        .iter()
        .find(|target| target.target_sector_id() == selected_public_flow_id)
        .map(crate::recurrence::RecurrenceReplayTarget::materialized_sector_id)
        .ok_or_else(|| {
            integrity(format!(
                "established transition diagnostic public flow {selected_public_flow_id} has no replay target"
            ))
        })?;
    let established_transition_candidates =
        crate::recurrence::diagnostic::retain_materialized_sector_rows(
            established_transition_candidates,
            established_materialized_sector_id,
            |row| row.materialized_sector_id,
        )
        .into_iter()
        .filter(|row| {
            established_live.contains(&row.output_current_digest)
                && row
                    .ordered_parent_digests
                    .iter()
                    .all(|digest| established_live.contains(digest))
        })
        .collect::<Vec<_>>();
    let established_pairing_rules =
        crate::recurrence::construct::take_established_pairing_owner_observation()?
            .into_iter()
            .map(|rule_id| established_pairing_rule_by_id(authenticated, rule_id))
            .collect::<RusticolResult<Vec<_>>>()?;
    let established_current_rows = established_current_digests(&established)?;
    let established_current_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-current-multiset-v1\0",
        established_current_rows.clone(),
    )?;
    let established_contribution_multiset_digest =
        established_contribution_digest(&established, &established_current_rows)?;
    let closure_parity_multiset_digest = compact_closure_parity_digest(&trace)?;
    let established_closure_parity_multiset_digest =
        established_closure_parity_digest(&established, &established_current_rows)?;
    let negative_contribution_factor_count = checked_u32(
        trace
            .operations
            .iter()
            .filter_map(|operation| match operation {
                OnTheFlyTraceOperationV1::Contribution { row, .. } => trace
                    .exact_factors
                    .get(row.exact_factor_id as usize)
                    .copied(),
                _ => None,
            })
            .map(is_negative_one)
            .collect::<RusticolResult<Vec<_>>>()?
            .into_iter()
            .filter(|negative| *negative)
            .count(),
        "compact negative contribution count",
    )?;
    let established_negative_contribution_factor_count = checked_u32(
        established
            .contributions()
            .iter()
            .map(|row| is_negative_one(row.exact_factor()))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_iter()
            .filter(|negative| *negative)
            .count(),
        "established negative contribution count",
    )?;
    let (pairing_oracle_equal, pairing_fermion_parities, established_pairing_fermion_parities) =
        pairing_oracle_equal(
            authenticated,
            &trace.pairing_owners,
            &established_pairing_rules,
        )?;
    let one = OnTheFlyWorkspaceV1::new(&trace, 1)?;
    let many = OnTheFlyWorkspaceV1::new(&trace, 17)?;
    let workspace_capacity_independent = one.point_stride() != many.point_stride();
    let proof = trace.proof();
    Ok(OnTheFlyTestSupportReportV1 {
        seed_digest: seed.semantic_digest(),
        query_digest: query.semantic_digest(),
        selector_digest: query.selector_digest,
        trace_digest: trace.semantic_digest(),
        current_count: proof.current_count(),
        contribution_count: proof.contribution_count(),
        closure_count: proof.closure_count(),
        established_current_count: checked_u32(
            established.currents().len(),
            "established current count",
        )?,
        established_contribution_count: checked_u32(
            established.contributions().len(),
            "established contribution count",
        )?,
        established_closure_count: checked_u32(
            established.closure_terms().len(),
            "established closure count",
        )?,
        current_multiset_digest: proof.current_multiset_digest(),
        established_current_multiset_digest,
        contribution_multiset_digest: proof.contribution_multiset_digest(),
        established_contribution_multiset_digest,
        closure_multiset_digest: proof.closure_multiset_digest(),
        closure_parity_multiset_digest,
        established_closure_parity_multiset_digest,
        negative_contribution_factor_count,
        established_negative_contribution_factor_count,
        source_domain_equal: compact_source_domain(&seed)
            == established_source_domain(authenticated, &established)?,
        pairing_oracle_equal,
        pairing_fermion_parities,
        established_pairing_fermion_parities,
        workspace_capacity_independent,
        compact_transition_candidates,
        established_transition_candidates,
    })
}

#[cfg(test)]
mod source_domain_tests {
    use super::*;

    fn digest(value: u8) -> SemanticDigest {
        SemanticDigest::new([value; 32]).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild_test_seed(
        seed: OnTheFlyProcessSeedV1,
        external_permutation: Vec<u32>,
        coupling_order_policy: OnTheFlyCouplingOrderPolicyV1,
        coupling_hierarchies: Vec<u32>,
        coupling_limits: Vec<Option<u32>>,
        normalization_semantic_digest: Option<SemanticDigest>,
        source_semantic_digest: Option<SemanticDigest>,
    ) -> OnTheFlyProcessSeedV1 {
        let OnTheFlyProcessSeedV1 {
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest: original_normalization_semantic_digest,
            normalization_convention,
            source_anchors,
            external_permutation: _,
            coupling_order_policy: _,
            coupling_hierarchies: _,
            coupling_limits: _,
            pairing_classes,
            semantic_digest: _,
        } = seed;
        let mut source_anchors = source_anchors.into_vec();
        if let Some(source_semantic_digest) = source_semantic_digest {
            source_anchors[0].states[0].source_semantic_digest = source_semantic_digest;
        }
        OnTheFlyProcessSeedV1::new(
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest.unwrap_or(original_normalization_semantic_digest),
            normalization_convention,
            ExactComplexRational::ONE,
            source_anchors,
            external_permutation,
            coupling_order_policy,
            coupling_hierarchies,
            coupling_limits,
            pairing_classes.into_vec(),
        )
        .unwrap()
    }

    fn projection_domain_fixture(
        normalization_semantic_digest: Option<SemanticDigest>,
        source_semantic_digest: Option<SemanticDigest>,
    ) -> (OnTheFlyProcessSeedV1, OnTheFlyProcessSeedV1) {
        let base = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        let artifact_seed = rebuild_test_seed(
            base.clone(),
            vec![0, 1],
            OnTheFlyCouplingOrderPolicyV1::Minimal,
            vec![1, 2],
            vec![None, None],
            None,
            None,
        );
        let retained_seed = rebuild_test_seed(
            base,
            vec![1, 0],
            OnTheFlyCouplingOrderPolicyV1::Explicit,
            vec![1, 1],
            vec![Some(4), Some(0)],
            normalization_semantic_digest,
            source_semantic_digest,
        );
        (artifact_seed, retained_seed)
    }

    fn source_state(
        state_index: u32,
        public_helicity: i32,
        current_state_template_id: u32,
        source_template_id: u32,
    ) -> ProcessSourceStateRow {
        ProcessSourceStateRow {
            source_slot: 0,
            state_index,
            public_helicity,
            chirality: public_helicity,
            spin_state: public_helicity,
            current_state_template_id,
            source_template_id,
            momentum_sign: 1,
            crossing_phase_factor_id: 0,
        }
    }

    #[test]
    fn established_source_domain_resolves_each_valid_state_by_its_own_template() {
        let states = [source_state(0, -1, 16, 57), source_state(1, 1, 76, 23)];
        let negative = established_source_state_for_template(0, &states, 57).unwrap();
        let positive = established_source_state_for_template(0, &states, 23).unwrap();
        assert_eq!(
            (negative.state_index, negative.current_state_template_id),
            (0, 16)
        );
        assert_eq!(
            (positive.state_index, positive.current_state_template_id),
            (1, 76)
        );
    }

    #[test]
    fn minimal_process_global_artifact_authenticates_explicit_flow_local_oracle_in_artifact_domain()
    {
        let (artifact_seed, retained_seed) = projection_domain_fixture(None, None);
        assert_ne!(
            retained_seed.semantic_digest(),
            artifact_seed.semantic_digest()
        );
        let oracle = OnePublicRowOracle {
            external_permutation: retained_seed.external_permutation().to_vec(),
            public_helicities: vec![0, 0],
            selector: OnTheFlyLcSelectorV1::Singlet,
        };
        assert!(
            DecodedLcQueryV1::new(
                &artifact_seed,
                oracle.external_permutation.clone(),
                &oracle.public_helicities,
                oracle.selector.clone(),
            )
            .is_err()
        );
        let reconstructed =
            reconstruct_seed_in_projection_domain(retained_seed, &artifact_seed).unwrap();
        assert_eq!(
            reconstructed.semantic_digest(),
            artifact_seed.semantic_digest()
        );
        let query = DecodedLcQueryV1::new(
            &artifact_seed,
            artifact_seed.external_permutation().to_vec(),
            &oracle.public_helicities,
            oracle.selector,
        )
        .unwrap();
        assert_eq!(query.process_seed_digest(), artifact_seed.semantic_digest());
    }

    #[test]
    fn projection_domain_reconstruction_rejects_mutated_semantic_constituents() {
        for (normalization_semantic_digest, source_semantic_digest) in
            [(Some(digest(93)), None), (None, Some(digest(94)))]
        {
            let (artifact_seed, retained_seed) =
                projection_domain_fixture(normalization_semantic_digest, source_semantic_digest);
            let error =
                reconstruct_seed_in_projection_domain(retained_seed, &artifact_seed).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("semantic constituents do not authenticate")
            );
        }
    }
}
