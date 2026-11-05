// SPDX-License-Identifier: 0BSD

use super::source_seed::validate_permutation;
use super::*;
use crate::recurrence::construct::{
    CurrentReflection, PendingReflectionCertificate, TransitionReflectionIndex,
    current_key_with_dynamic_color, current_reflection_candidate, current_reversal_masks,
    dynamic_color_identity_digest, pure_adjoint_word_is_canonical, reciprocal_reflection_proof,
    source_reflection,
};
use crate::recurrence::fermion_ordering::{FermionOrderingContext, fermion_ordering_factor};

fn on_the_fly_source_requires_exterior_sign(
    is_fermionic: bool,
    color_role: OnTheFlyExternalColorRoleV1,
) -> RusticolResult<bool> {
    if !is_fermionic {
        if color_role.is_pairing_endpoint() {
            return Err(integrity(
                "fermion-ordering endpoint role belongs to a bosonic source",
            ));
        }
        return Ok(false);
    }
    match color_role {
        OnTheFlyExternalColorRoleV1::Singlet => Ok(true),
        OnTheFlyExternalColorRoleV1::Fundamental | OnTheFlyExternalColorRoleV1::Antifundamental => {
            Ok(false)
        }
        OnTheFlyExternalColorRoleV1::Adjoint => Err(invalid(
            "external-fermion ordering does not support adjoint fermion sources",
        )),
    }
}

pub(super) fn on_the_fly_fermion_ordering_context(
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<FermionOrderingContext> {
    let mut source_requires_exterior_sign = Vec::with_capacity(seed.source_anchors.len());
    for (source_slot, anchor) in seed.source_anchors.iter().enumerate() {
        let source_slot = checked_u32(source_slot, "fermion-ordering source slot")?;
        if anchor.source_slot != source_slot {
            return Err(integrity(
                "fermion-ordering source anchor disagrees with its slot",
            ));
        }
        source_requires_exterior_sign.push(on_the_fly_source_requires_exterior_sign(
            anchor.is_fermionic,
            anchor.color_role,
        )?);
    }
    Ok(FermionOrderingContext::new(source_requires_exterior_sign))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingContributionKey {
    pub(super) parent_current_ids: [u32; 2],
    pub(super) key: ContributionKey,
}

#[derive(Debug)]
pub(super) struct PendingCurrent {
    pub(super) key: CurrentCoreKey,
    pub(super) source_factor: Option<ExactComplexRational>,
    pub(super) contributions: BTreeMap<PendingContributionKey, ExactComplexRational>,
    /// Canonical proof-only lineage alternatives.  As in the established
    /// builder's realized-rule set, exact contribution factors remain owned
    /// by `contributions`; lineage does not multiply or normalize amplitudes.
    pub(super) pairing_lineages: Vec<PendingPairingLineage>,
    pub(super) stage: u32,
    /// Exact reflection lineage is populated only for the certified cyclic
    /// pure-adjoint companion lane. Generic/direct construction leaves it
    /// unavailable and therefore cannot enter the fold.
    pub(super) reflection: CurrentReflection,
    /// Cold reciprocal-orbit proof retained only on the canonical member
    /// until reflected public closures have been resolved.
    pub(super) reflection_certificate: Option<PendingReflectionCertificate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedContactOrbitLocation {
    transition_id: u32,
    state_pair: (u32, u32),
    row_index: usize,
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(super) struct PreparedContactOrbitIndex {
    locations: Vec<PreparedContactOrbitLocation>,
}

fn reserve_on_the_fly_contact_locations<T>(
    locations: &mut Vec<T>,
    count: usize,
) -> RusticolResult<()> {
    locations.try_reserve_exact(count).map_err(|error| {
        invalid(format!(
            "contact-orbit transition-index allocation failed: {error}"
        ))
    })
}

impl PreparedContactOrbitIndex {
    pub(super) fn new(
        transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    ) -> RusticolResult<Self> {
        let transition_count = transitions.values().try_fold(0usize, |count, rows| {
            count
                .checked_add(rows.len())
                .ok_or_else(|| invalid("contact-orbit transition count exceeds usize"))
        })?;
        let contact_count = transitions
            .values()
            .flatten()
            .filter(|prepared| prepared.contact_orbit.is_some())
            .count();
        if contact_count == 0 {
            return Ok(Self::default());
        }
        let mut transition_ids = Vec::new();
        reserve_on_the_fly_contact_locations(&mut transition_ids, transition_count)?;
        let mut locations = Vec::new();
        reserve_on_the_fly_contact_locations(&mut locations, contact_count)?;
        for (state_pair, rows) in transitions {
            for (row_index, prepared) in rows.iter().enumerate() {
                transition_ids.push(prepared.row.id);
                if prepared.contact_orbit.is_some() {
                    locations.push(PreparedContactOrbitLocation {
                        transition_id: prepared.row.id,
                        state_pair: *state_pair,
                        row_index,
                    });
                }
            }
        }
        transition_ids.sort_unstable();
        if transition_ids.windows(2).any(|rows| rows[0] == rows[1]) {
            return Err(integrity(
                "contact-orbit transition index contains duplicate IDs",
            ));
        }
        locations.sort_unstable_by_key(|location| location.transition_id);
        Ok(Self { locations })
    }

    fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    fn get<'a>(
        &self,
        transitions: &'a BTreeMap<(u32, u32), Vec<PreparedTransition>>,
        transition_id: u32,
    ) -> Option<&'a PreparedContactOrbitTransition> {
        self.locations
            .binary_search_by_key(&transition_id, |location| location.transition_id)
            .ok()
            .and_then(|index| self.locations.get(index))
            .and_then(|location| {
                transitions
                    .get(&location.state_pair)
                    .and_then(|rows| rows.get(location.row_index))
            })
            .filter(|prepared| prepared.row.id == transition_id)
            .and_then(|prepared| prepared.contact_orbit.as_ref())
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.locations.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OnTheFlyContactContributionToken {
    destination_id: usize,
    contribution_ordinal: usize,
}

#[derive(Debug)]
struct OnTheFlyContactOwnerPlan {
    stage_current_start: usize,
    selected_tokens: Vec<OnTheFlyContactContributionToken>,
    staged_contribution_count: usize,
    retained_contribution_count: usize,
}

impl OnTheFlyContactOwnerPlan {
    /// Apply a completely selected plan without allocating or failing.
    fn commit(self, currents: &mut [PendingCurrent]) {
        let mut selected_index = 0usize;
        let mut observed_count = 0usize;
        let mut retained_count = 0usize;
        for (destination_id, current) in currents
            .iter_mut()
            .enumerate()
            .skip(self.stage_current_start)
        {
            let mut contribution_ordinal = 0usize;
            current.contributions.retain(|_, _| {
                let token = OnTheFlyContactContributionToken {
                    destination_id,
                    contribution_ordinal,
                };
                contribution_ordinal += 1;
                observed_count += 1;
                let selected = self.selected_tokens.get(selected_index) == Some(&token);
                if selected {
                    selected_index += 1;
                    retained_count += 1;
                }
                selected
            });
        }
        debug_assert_eq!(observed_count, self.staged_contribution_count);
        debug_assert_eq!(retained_count, self.retained_contribution_count);
        debug_assert_eq!(selected_index, self.selected_tokens.len());
    }
}

fn reserve_on_the_fly_contact_candidates<T>(
    candidates: &mut Vec<T>,
    count: usize,
) -> RusticolResult<()> {
    candidates.try_reserve_exact(count).map_err(|error| {
        invalid(format!(
            "contact-orbit candidate allocation failed: {error}"
        ))
    })
}

fn plan_on_the_fly_contact_orbit_owners(
    stage_current_start: usize,
    transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    contact_orbits: &PreparedContactOrbitIndex,
    currents: &[PendingCurrent],
) -> RusticolResult<Option<OnTheFlyContactOwnerPlan>> {
    plan_on_the_fly_contact_orbit_owners_with_resolver(
        stage_current_start,
        currents,
        |transition_id| contact_orbits.get(transitions, transition_id),
    )
}

fn plan_on_the_fly_contact_orbit_owners_with_resolver<'a, F>(
    stage_current_start: usize,
    currents: &'a [PendingCurrent],
    mut contact_orbit: F,
) -> RusticolResult<Option<OnTheFlyContactOwnerPlan>>
where
    F: FnMut(u32) -> Option<&'a PreparedContactOrbitTransition>,
{
    if stage_current_start > currents.len() {
        return Err(invalid(
            "contact-orbit stage current boundary exceeds current storage",
        ));
    }
    let has_certified_contact = currents[stage_current_start..].iter().any(|current| {
        current
            .contributions
            .keys()
            .any(|pending| contact_orbit(pending.key.transition_template_id()).is_some())
    });
    if !has_certified_contact {
        return Ok(None);
    }
    let staged_contribution_count =
        currents[stage_current_start..]
            .iter()
            .try_fold(0usize, |count, current| {
                count
                    .checked_add(current.contributions.len())
                    .ok_or_else(|| invalid("contact-orbit staged contribution count exceeds usize"))
            })?;
    let mut candidates = Vec::new();
    reserve_on_the_fly_contact_candidates(&mut candidates, staged_contribution_count)?;
    for (destination_id, current) in currents.iter().enumerate().skip(stage_current_start) {
        for (contribution_ordinal, pending) in current.contributions.keys().enumerate() {
            let token = OnTheFlyContactContributionToken {
                destination_id,
                contribution_ordinal,
            };
            let candidate = contact_orbit(pending.key.transition_template_id())
                .map(|contact_orbit| {
                    let left = currents
                        .get(pending.parent_current_ids[0] as usize)
                        .map(|parent| &parent.key)
                        .ok_or_else(|| integrity("contact-orbit left parent is absent"))?;
                    let right = currents
                        .get(pending.parent_current_ids[1] as usize)
                        .map(|parent| &parent.key)
                        .ok_or_else(|| integrity("contact-orbit right parent is absent"))?;
                    contact_orbit.owner_candidate(
                        &current.key,
                        [left, right],
                        pending.key.color_witness_term_id(),
                    )
                })
                .transpose()?;
            candidates.push((token, candidate));
        }
    }
    if candidates.len() != staged_contribution_count {
        return Err(integrity(
            "contact-orbit staged contribution snapshot changed length",
        ));
    }
    let selected_tokens = selected_contact_orbit_owner_tokens(candidates)?;
    let retained_contribution_count = selected_tokens.len();
    if retained_contribution_count > staged_contribution_count {
        return Err(integrity(
            "contact-orbit retained contribution count exceeds snapshot",
        ));
    }
    Ok(Some(OnTheFlyContactOwnerPlan {
        stage_current_start,
        selected_tokens,
        staged_contribution_count,
        retained_contribution_count,
    }))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingPairingLineage {
    pub(super) completed_pairs: Vec<[u32; 2]>,
    pub(super) unmatched_endpoint: Option<u32>,
}

impl PendingPairingLineage {
    pub(super) fn source(seed: &OnTheFlyProcessSeedV1, source_slot: u32) -> Self {
        let unmatched_endpoint = seed.source_anchors[source_slot as usize]
            .color_role
            .is_pairing_endpoint()
            .then_some(source_slot);
        Self {
            completed_pairs: Vec::new(),
            unmatched_endpoint,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingClosureKey {
    pub(super) closure_template_id: u32,
    pub(super) quantum_flow_template_id: Option<u32>,
    pub(super) parent_current_ids: [u32; 2],
    pub(super) color_witness_term_id: LCColorWitnessTermId,
}

#[derive(Debug)]
pub(super) struct PendingClosure {
    pub(super) key: PendingClosureKey,
    pub(super) factor: ExactComplexRational,
    pub(super) component_coefficients: Box<[ExactComplexRational]>,
    pub(super) pairing_lineages: Vec<PendingPairingLineage>,
}

/// Query-local Wick-lineage proof selected only after physical closure and
/// complete-rectangle color-alias projection have identified a canonical
/// root-bearing representative.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct ResolvedPairingOwnerV1 {
    pub(super) endpoint_pairs: Vec<[u32; 2]>,
    pub(super) proof_digest: Option<SemanticDigest>,
    pub(super) source_slot_permutation: Vec<u32>,
    pub(super) source_lineage: Vec<u32>,
    pub(super) fermion_parity: i32,
}

fn checked_pairing_capacity(left: usize, right: usize, extra: usize) -> RusticolResult<usize> {
    left.checked_add(right)
        .and_then(|combined| combined.checked_add(extra))
        .ok_or_else(|| invalid("pairing-lineage length exceeds usize"))
}

fn try_copy_pairing_pairs(source: &[[u32; 2]]) -> RusticolResult<Vec<[u32; 2]>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(source.len())
        .map_err(|error| invalid(format!("pairing-lineage allocation failed: {error}")))?;
    copied.extend_from_slice(source);
    Ok(copied)
}

fn try_clone_pairing_lineage(
    source: &PendingPairingLineage,
) -> RusticolResult<PendingPairingLineage> {
    Ok(PendingPairingLineage {
        completed_pairs: try_copy_pairing_pairs(&source.completed_pairs)?,
        unmatched_endpoint: source.unmatched_endpoint,
    })
}

pub(super) fn try_clone_pairing_lineages(
    source: &[PendingPairingLineage],
) -> RusticolResult<Vec<PendingPairingLineage>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(source.len())
        .map_err(|error| invalid(format!("pairing-lineage set allocation failed: {error}")))?;
    for lineage in source {
        copied.push(try_clone_pairing_lineage(lineage)?);
    }
    Ok(copied)
}

fn pairing_endpoint_class(
    seed: &OnTheFlyProcessSeedV1,
    source_slot: u32,
) -> RusticolResult<(OnTheFlyExternalColorRoleV1, SemanticDigest)> {
    let anchor = seed
        .source_anchors
        .get(source_slot as usize)
        .ok_or_else(|| integrity("pairing-lineage endpoint is outside the source domain"))?;
    let contract = anchor
        .pairing_source_contract_digest
        .ok_or_else(|| integrity("pairing-lineage endpoint lacks its source contract"))?;
    let class = seed
        .pairing_classes
        .iter()
        .find(|class| {
            class
                .fundamental_endpoints
                .iter()
                .chain(class.antifundamental_endpoints.iter())
                .any(|endpoint| endpoint.source_slot == source_slot)
        })
        .ok_or_else(|| integrity("pairing-lineage endpoint lacks its compact class"))?;
    let class_endpoint = class
        .fundamental_endpoints
        .iter()
        .chain(class.antifundamental_endpoints.iter())
        .find(|endpoint| endpoint.source_slot == source_slot)
        .ok_or_else(|| integrity("pairing-lineage endpoint disappeared from its class"))?;
    if class_endpoint.source_contract_digest != contract {
        return Err(integrity(
            "pairing-lineage endpoint source contract differs from its class",
        ));
    }
    Ok((anchor.color_role, class.semantic_digest))
}

fn close_pairing_endpoints(
    seed: &OnTheFlyProcessSeedV1,
    left: u32,
    right: u32,
) -> RusticolResult<Option<[u32; 2]>> {
    let (left_role, left_class) = pairing_endpoint_class(seed, left)?;
    let (right_role, right_class) = pairing_endpoint_class(seed, right)?;
    if left_class != right_class {
        return Ok(None);
    }
    Ok(match (left_role, right_role) {
        (
            OnTheFlyExternalColorRoleV1::Fundamental,
            OnTheFlyExternalColorRoleV1::Antifundamental,
        ) => Some([left, right]),
        (
            OnTheFlyExternalColorRoleV1::Antifundamental,
            OnTheFlyExternalColorRoleV1::Fundamental,
        ) => Some([right, left]),
        _ => None,
    })
}

fn combine_pairing_lineage(
    seed: &OnTheFlyProcessSeedV1,
    left: &PendingPairingLineage,
    right: &PendingPairingLineage,
    carries_colored_fermion_line: bool,
) -> RusticolResult<Option<PendingPairingLineage>> {
    let possible_new_pair = usize::from(
        left.unmatched_endpoint.is_some()
            && right.unmatched_endpoint.is_some()
            && !carries_colored_fermion_line,
    );
    let capacity = checked_pairing_capacity(
        left.completed_pairs.len(),
        right.completed_pairs.len(),
        possible_new_pair,
    )?;
    let mut completed_pairs = Vec::new();
    completed_pairs
        .try_reserve_exact(capacity)
        .map_err(|error| invalid(format!("pairing-lineage allocation failed: {error}")))?;
    completed_pairs.extend_from_slice(&left.completed_pairs);
    completed_pairs.extend_from_slice(&right.completed_pairs);
    let unmatched_endpoint = match (
        left.unmatched_endpoint,
        right.unmatched_endpoint,
        carries_colored_fermion_line,
    ) {
        (Some(endpoint), None, true) | (None, Some(endpoint), true) => Some(endpoint),
        (None, None, false) => None,
        (Some(left), Some(right), false) => {
            let Some(pair) = close_pairing_endpoints(seed, left, right)? else {
                return Ok(None);
            };
            completed_pairs.push(pair);
            None
        }
        _ => return Ok(None),
    };
    completed_pairs.sort_unstable();
    if completed_pairs.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(integrity(
            "pairing lineage repeats a completed endpoint pair",
        ));
    }
    Ok(Some(PendingPairingLineage {
        completed_pairs,
        unmatched_endpoint,
    }))
}

pub(super) fn combine_pairing_lineage_sets(
    seed: &OnTheFlyProcessSeedV1,
    left: &[PendingPairingLineage],
    right: &[PendingPairingLineage],
    carries_colored_fermion_line: bool,
) -> RusticolResult<Vec<PendingPairingLineage>> {
    let capacity = left
        .len()
        .checked_mul(right.len())
        .ok_or_else(|| invalid("pairing-lineage product exceeds usize"))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|error| invalid(format!("pairing-lineage allocation failed: {error}")))?;
    for left in left {
        for right in right {
            if let Some(lineage) =
                combine_pairing_lineage(seed, left, right, carries_colored_fermion_line)?
            {
                result.push(lineage);
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

pub(super) fn extend_pairing_lineages(
    target: &mut Vec<PendingPairingLineage>,
    source: &[PendingPairingLineage],
) -> RusticolResult<()> {
    let mut additions = try_clone_pairing_lineages(source)?;
    target
        .try_reserve(additions.len())
        .map_err(|error| invalid(format!("pairing-lineage merge allocation failed: {error}")))?;
    target.append(&mut additions);
    target.sort_unstable();
    target.dedup();
    Ok(())
}

fn retain_canonical_pairing_lineage(
    lineages: &mut Vec<PendingPairingLineage>,
) -> RusticolResult<()> {
    if lineages.is_empty() {
        return Err(integrity(
            "physical closure has no complete Wick-lineage owner",
        ));
    }
    // Pairing lineage is cold proof state, not an additive amplitude term.
    // Equal runtime closures therefore retain one semantic witness without
    // duplicating or rescaling their already aggregated numerical factor.
    lineages.sort_unstable();
    lineages.dedup();
    lineages.truncate(1);
    Ok(())
}

pub(super) fn complete_pairing_lineage(
    seed: &OnTheFlyProcessSeedV1,
    lineage: &PendingPairingLineage,
) -> RusticolResult<bool> {
    if lineage.unmatched_endpoint.is_some() {
        return Ok(false);
    }
    let expected_endpoint_count = seed
        .source_anchors
        .iter()
        .filter(|anchor| anchor.color_role.is_pairing_endpoint())
        .count();
    let observed_endpoint_count = lineage
        .completed_pairs
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("pairing endpoint count exceeds usize"))?;
    if observed_endpoint_count != expected_endpoint_count {
        return Ok(false);
    }
    for (index, pair) in lineage.completed_pairs.iter().copied().enumerate() {
        if close_pairing_endpoints(seed, pair[0], pair[1])? != Some(pair) {
            return Ok(false);
        }
        if lineage.completed_pairs[..index]
            .iter()
            .flatten()
            .any(|endpoint| *endpoint == pair[0] || *endpoint == pair[1])
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn identity_source_permutation(source_count: usize) -> RusticolResult<Vec<u32>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source_count)
        .map_err(|error| invalid(format!("pairing identity allocation failed: {error}")))?;
    for value in 0..source_count {
        result.push(checked_u32(value, "pairing identity source slot")?);
    }
    Ok(result)
}

fn lehmer_digits(reference: &[u32], selected: &[u32]) -> RusticolResult<Vec<u32>> {
    if reference.len() != selected.len() {
        return Err(invalid(
            "discovered pairing length differs from its compact class",
        ));
    }
    let mut remaining = Vec::new();
    remaining
        .try_reserve_exact(reference.len())
        .map_err(|error| invalid(format!("pairing permutation allocation failed: {error}")))?;
    remaining.extend_from_slice(reference);
    let mut digits = Vec::new();
    digits
        .try_reserve_exact(selected.len())
        .map_err(|error| invalid(format!("pairing Lehmer allocation failed: {error}")))?;
    for selected_slot in selected {
        let position = remaining
            .iter()
            .position(|candidate| candidate == selected_slot)
            .ok_or_else(|| {
                invalid("discovered pairing is not a permutation of its compact class")
            })?;
        digits.push(checked_u32(position, "pairing Lehmer digit")?);
        remaining.remove(position);
    }
    if !remaining.is_empty() {
        return Err(invalid("discovered pairing omits a compact-class endpoint"));
    }
    Ok(digits)
}

fn resolve_pairing_owner(
    seed: &OnTheFlyProcessSeedV1,
    owner: &PendingPairingLineage,
) -> RusticolResult<ResolvedPairingOwnerV1> {
    let mut source_slot_permutation = identity_source_permutation(seed.source_anchors.len())?;
    let mut source_lineage = Vec::new();
    source_lineage
        .try_reserve_exact(seed.source_anchors.len())
        .map_err(|error| invalid(format!("pairing source-lineage allocation failed: {error}")))?;
    source_lineage.resize(seed.source_anchors.len(), MISSING_U32);
    if seed.pairing_classes.is_empty() {
        if owner
            != &(PendingPairingLineage {
                completed_pairs: Vec::new(),
                unmatched_endpoint: None,
            })
        {
            return Err(integrity(
                "pairing-free query retained a nontrivial closure lineage",
            ));
        }
        return Ok(ResolvedPairingOwnerV1 {
            endpoint_pairs: Vec::new(),
            proof_digest: None,
            source_slot_permutation,
            source_lineage,
            fermion_parity: 1,
        });
    }
    if !complete_pairing_lineage(seed, owner)? {
        return Err(integrity(
            "canonical projected closure has an incomplete Wick lineage",
        ));
    }
    let endpoint_pairs = try_copy_pairing_pairs(&owner.completed_pairs)?;
    if endpoint_pairs.iter().enumerate().any(|(index, pair)| {
        endpoint_pairs[..index]
            .iter()
            .any(|other| other[0] == pair[0])
    }) {
        return Err(integrity(
            "canonical projected closure repeats a fundamental endpoint",
        ));
    }
    let mut parity = 1_i32;
    let mut proof_hash = Sha256::new();
    proof_hash.update(b"pyamplicol-on-the-fly-discovered-pairing-owner-v1\0");
    for pairing_class in &seed.pairing_classes {
        let mut reference = Vec::new();
        reference
            .try_reserve_exact(pairing_class.antifundamental_endpoints.len())
            .map_err(|error| invalid(format!("pairing reference allocation failed: {error}")))?;
        reference.extend(
            pairing_class
                .antifundamental_endpoints
                .iter()
                .map(|endpoint| endpoint.source_slot),
        );
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(pairing_class.fundamental_endpoints.len())
            .map_err(|error| invalid(format!("selected pairing allocation failed: {error}")))?;
        for endpoint in &pairing_class.fundamental_endpoints {
            let selected_slot = endpoint_pairs
                .iter()
                .find(|pair| pair[0] == endpoint.source_slot)
                .map(|pair| pair[1])
                .ok_or_else(|| {
                    integrity("canonical Wick lineage omits a pairing-class fundamental")
                })?;
            selected.push(selected_slot);
        }
        let digits = lehmer_digits(&reference, &selected)?;
        if digits.iter().map(|digit| u64::from(*digit)).sum::<u64>() % 2 == 1 {
            parity = -parity;
        }
        hash_digest(&mut proof_hash, pairing_class.semantic_digest);
        hash_len(&mut proof_hash, selected.len(), "discovered pairing class")?;
        for (reference_slot, selected_slot) in
            reference.iter().copied().zip(selected.iter().copied())
        {
            source_slot_permutation[reference_slot as usize] = selected_slot;
            proof_hash.update(reference_slot.to_le_bytes());
            proof_hash.update(selected_slot.to_le_bytes());
        }
        for digit in digits {
            proof_hash.update(digit.to_le_bytes());
        }
    }
    validate_permutation(
        &source_slot_permutation,
        seed.source_anchors.len(),
        "discovered pairing source permutation",
    )?;
    for (line_id, pair) in endpoint_pairs.iter().copied().enumerate() {
        let line_id = checked_u32(line_id, "discovered pairing line ID")?;
        for source_slot in pair {
            let entry = source_lineage
                .get_mut(source_slot as usize)
                .ok_or_else(|| integrity("discovered pairing endpoint is out of range"))?;
            if *entry != MISSING_U32 {
                return Err(integrity(
                    "discovered pairing endpoint belongs to multiple lines",
                ));
            }
            *entry = line_id;
        }
    }
    proof_hash.update(parity.to_le_bytes());
    Ok(ResolvedPairingOwnerV1 {
        endpoint_pairs,
        proof_digest: Some(final_digest(proof_hash)?),
        source_slot_permutation,
        source_lineage,
        fermion_parity: parity,
    })
}

fn projected_pairing_lineages(
    closures: &[PendingClosure],
) -> RusticolResult<Vec<&PendingPairingLineage>> {
    if closures.is_empty() {
        return Err(integrity("canonical projected closure has no Wick lineage"));
    }
    let mut lineages = Vec::new();
    lineages
        .try_reserve_exact(closures.len())
        .map_err(|error| {
            invalid(format!(
                "projected pairing-owner allocation failed: {error}"
            ))
        })?;
    for closure in closures {
        let [lineage] = closure.pairing_lineages.as_slice() else {
            return Err(integrity(format!(
                "canonical projected closure has {} Wick lineages, expected exactly one",
                closure.pairing_lineages.len(),
            )));
        };
        lineages.push(lineage);
    }
    Ok(lineages)
}

pub(super) fn resolve_projected_pairing_owners(
    seed: &OnTheFlyProcessSeedV1,
    closures: &[PendingClosure],
) -> RusticolResult<Vec<ResolvedPairingOwnerV1>> {
    projected_pairing_lineages(closures)?
        .into_iter()
        .map(|lineage| resolve_pairing_owner(seed, lineage))
        .collect()
}

pub(super) fn supports_are_disjoint(left: &[u32], right: &[u32]) -> bool {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => return false,
        }
    }
    true
}

pub(super) fn merge_disjoint_support(left: &[u32], right: &[u32]) -> RusticolResult<Vec<u32>> {
    if !supports_are_disjoint(left, right) {
        return Err(invalid(
            "query-local parents have overlapping source support",
        ));
    }
    let mut result = left.iter().chain(right).copied().collect::<Vec<_>>();
    result.sort_unstable();
    Ok(result)
}

fn selected_source_state(
    seed: &OnTheFlyProcessSeedV1,
    selected: OnTheFlySelectedSourceV1,
) -> RusticolResult<(&OnTheFlySourceAnchorV1, &OnTheFlySourceStateV1)> {
    let anchor = seed
        .source_anchors
        .get(selected.source_slot as usize)
        .ok_or_else(|| invalid("selected source slot is absent from the compact seed"))?;
    let state = anchor.selected(selected.state_index, selected.public_helicity)?;
    Ok((anchor, state))
}

pub(super) fn validate_seed_against_templates<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<TemplateCatalog<'a>> {
    let summary = templates.summary();
    if summary.catalog_digest != seed.template_catalog_digest
        || summary.compiled_model_digest != seed.model_digest
        || summary.prepared_kernel_pack_digest != seed.prepared_pack_digest
    {
        return Err(integrity(
            "compact seed belongs to a different model/template/prepared catalog",
        ));
    }
    let catalog = TemplateCatalog::new(templates.input())?;
    if catalog.coupling_order_dimension() != seed.explicit_coupling_limits().len() {
        return Err(integrity(
            "compact coupling-limit dimension differs from the template catalog",
        ));
    }
    Ok(catalog)
}

pub(super) fn validate_source_contract(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    anchor: &OnTheFlySourceAnchorV1,
    state: &OnTheFlySourceStateV1,
) -> RusticolResult<(SourceRow, crate::recurrence::template::CurrentStateRow)> {
    let input = templates.input();
    let source = *input
        .sources
        .get(state.source_template_id as usize)
        .ok_or_else(|| integrity("compact source template is absent"))?;
    let current_state = *input
        .current_states
        .get(state.current_state_template_id as usize)
        .ok_or_else(|| integrity("compact current-state template is absent"))?;
    if source.id != state.source_template_id
        || current_state.id != state.current_state_template_id
        || source.flavour_flow_id as usize >= input.flavour_flow_ranges.len()
        || source.quantum_number_flow_id != state.quantum_number_flow_id
        || current_state.chirality != state.chirality
    {
        return Err(integrity("compact source-state contract is stale"));
    }
    if catalog.flavour_flow(source.flavour_flow_id, "source flavour flow")?
        != state.flavour_flow.as_ref()
        || catalog.digest(source.semantic_digest_id, "source semantic")?
            != state.source_semantic_digest
        || catalog.digest(current_state.semantic_digest_id, "current-state semantic")?
            != state.current_state_semantic_digest
        || catalog.source_seed(source)?.proof_digest() != state.color_seed_proof_digest
    {
        return Err(integrity(
            "compact source-state semantic evidence differs from the template catalog",
        ));
    }
    validate_crossed_source_state(
        anchor.is_initial,
        &ProcessSourceStateRow {
            source_slot: anchor.source_slot,
            state_index: state.state_index,
            public_helicity: state.public_helicity,
            chirality: state.chirality,
            spin_state: state.spin_state,
            current_state_template_id: state.current_state_template_id,
            source_template_id: state.source_template_id,
            momentum_sign: state.momentum_sign,
            // The helper does not consume the process-owned factor ID; the
            // compact seed authenticates the exact crossing phase directly.
            crossing_phase_factor_id: 0,
        },
        source,
        input,
    )?;
    let binding = input
        .evaluator_bindings
        .get(source.evaluator_binding_id as usize)
        .ok_or_else(|| integrity("source evaluator binding is absent"))?;
    if binding.id != source.evaluator_binding_id
        || EvaluatorContractKind::try_from(binding.contract_kind)? != EvaluatorContractKind::Source
    {
        return Err(integrity("source evaluator binding has the wrong role"));
    }
    Ok((source, current_state))
}

fn query_target_matches(closed: &[LCColorComponent], query: &DecodedLcQueryV1) -> bool {
    let mut closed = closed.to_vec();
    closed.sort_unstable();
    closed.as_slice() == query.target_components.as_ref()
}

fn reflected_query_closure_factor(
    query: &DecodedLcQueryV1,
    closed: &[LCColorComponent],
    colors: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    parent_ids: [u32; 2],
) -> RusticolResult<Option<ExactComplexRational>> {
    let mut certified_parent = None;
    for parent_id in parent_ids {
        let current = currents
            .get(parent_id as usize)
            .ok_or_else(|| integrity("query-local closure reflection parent is absent"))?;
        let Some(certificate) = current.reflection_certificate.as_ref() else {
            continue;
        };
        if certified_parent.replace((current, certificate)).is_some() {
            return Err(integrity(
                "one query-local closure references multiple folded reflection orbits",
            ));
        }
    }
    let Some((current, certificate)) = certified_parent else {
        return Ok(None);
    };
    let proof = current.reflection.proof().ok_or_else(|| {
        integrity("certified query-local closure parent has no reflection lineage")
    })?;
    if proof.proof_digest() != certificate.canonical_lineage_digest() {
        return Err(integrity(
            "query-local closure reflection lineage differs from its certificate",
        ));
    }
    let color = colors
        .get(current.key.dynamic_lc_color_state_id())
        .ok_or_else(|| integrity("query-local closure reflection color is absent"))?;
    if dynamic_color_identity_digest(color)? != certificate.canonical_color_identity() {
        return Err(integrity(
            "query-local closure reflection color differs from its certificate",
        ));
    }
    if !certificate.is_reciprocal_two_cycle() {
        return Err(integrity(
            "query-local closure reflection certificate is not a reciprocal two-cycle",
        ));
    }
    let [closed_trace] = closed else {
        return Ok(None);
    };
    let [target_trace] = query.target_components.as_ref() else {
        return Ok(None);
    };
    let source_count = query.selected_sources.len();
    if source_count <= 2
        || closed_trace.kind() != LCColorComponentKind::Trace
        || target_trace.kind() != LCColorComponentKind::Trace
        || closed_trace.source_slots().len() != source_count
        || target_trace.source_slots().len() != source_count
    {
        return Ok(None);
    }
    validate_permutation(
        closed_trace.source_slots(),
        source_count,
        "query-local reflected closure word",
    )?;
    let permutation = certificate.source_permutation();
    validate_permutation(
        permutation,
        source_count,
        "query-local closure reflection permutation",
    )?;
    if permutation.get(query.closure_anchor_slot as usize).copied()
        != Some(query.closure_anchor_slot)
    {
        return Err(integrity(
            "query-local closure reflection does not fix its certified anchor",
        ));
    }
    let mapped_word = closed_trace
        .source_slots()
        .iter()
        .map(|source_slot| {
            permutation
                .get(*source_slot as usize)
                .copied()
                .ok_or_else(|| integrity("closure reflection omits a source slot"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mapped_trace = LCColorComponent::new(LCColorComponentKind::Trace, mapped_word)?;
    if &mapped_trace == closed_trace {
        return Err(integrity(
            "query-local closure reflection maps a trace to a cyclic fixed point",
        ));
    }
    Ok((&mapped_trace == target_trace).then_some(certificate.canonical_phase()))
}

#[allow(clippy::too_many_arguments)] // Explicitly mirrors the query-local sweep state.
pub(super) fn insert_selected_sources(
    grammar: &PreparedOnTheFlyGrammarV1,
    seed: &OnTheFlyProcessSeedV1,
    coupling_limits: &[Option<u32>],
    query: &DecodedLcQueryV1,
    enable_cyclic_trace_reflection: bool,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    if coupling_limits.len() != seed.explicit_coupling_limits().len() {
        return Err(integrity(
            "resolved coupling-limit dimension differs from the compact seed",
        ));
    }
    let zero_orders = vec![0_u32; coupling_limits.len()];
    for selected in query.selected_sources.iter().copied() {
        let (_, state) = selected_source_state(seed, selected)?;
        let contract = grammar
            .sources
            .get(&(state.source_template_id, state.current_state_template_id))
            .ok_or_else(|| integrity("selected source has no prepared source contract"))?;
        let color = contract.color_seed.instantiate(
            selected.source_slot,
            contract.current_state.color_representation,
        )?;
        let color_id = colors.intern(color)?;
        let reflection = if enable_cyclic_trace_reflection {
            source_reflection(
                colors,
                color_id,
                selected.source_slot,
                state.color_seed_proof_digest,
            )?
        } else {
            CurrentReflection::Unavailable
        };
        let key = CurrentCoreKey::new(
            seed.template_catalog_digest,
            RecurrenceNodeKind::Source,
            state.current_state_template_id,
            color_id,
            vec![selected.source_slot],
            CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                source_slot: selected.source_slot,
                coefficient: state.momentum_sign,
            }])?,
            CurrentHelicityIdentity::topology_replay(
                state.spin_state,
                vec![SourceStateAssignment::new(
                    selected.source_slot,
                    selected.state_index,
                )],
            )?,
            state.flavour_flow.to_vec(),
            state.quantum_number_flow_id,
            zero_orders.clone(),
            CurrentSourceBinding::FixedTemplate(state.source_template_id),
            None,
        )?;
        let id = checked_u32(currents.len(), "query-local current count")?;
        if current_ids.insert(key.clone(), id).is_some() {
            return Err(integrity(
                "selected source construction produced a duplicate current",
            ));
        }
        currents.push(PendingCurrent {
            key,
            source_factor: Some(state.crossing_phase),
            contributions: BTreeMap::new(),
            pairing_lineages: vec![PendingPairingLineage::source(seed, selected.source_slot)],
            stage: 0,
            reflection,
            reflection_certificate: None,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn include_transition(
    templates: &ValidatedRecurrenceTemplateInput,
    prepared: &PreparedTransition,
    transition_reflections: &TransitionReflectionIndex,
    enable_cyclic_trace_reflection: bool,
    concrete_parent_ids: [u32; 2],
    source_count: usize,
    seed: &OnTheFlyProcessSeedV1,
    fermion_ordering: &FermionOrderingContext,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    let parents = [
        &currents[concrete_parent_ids[0] as usize].key,
        &currents[concrete_parent_ids[1] as usize].key,
    ];
    if !(0..2).all(|index| {
        prepared.input_states[index] == parents[index].current_state_template_id()
            && quantum_parent_spin_matches(prepared.input_spins[index], parents[index])
    }) {
        return Ok(());
    }
    if let Some(contact_orbit) = prepared.contact_orbit.as_ref()
        && !contact_orbit.accepts_parent_domain(parents)?
    {
        return Ok(());
    }
    let Some(coupling_orders) = combined_coupling_orders(
        parents[0].coupling_orders(),
        parents[1].coupling_orders(),
        &prepared.local_orders,
        coupling_limits,
    )?
    else {
        return Ok(());
    };
    let support = merge_disjoint_support(
        parents[0].support_source_slots(),
        parents[1].support_source_slots(),
    )?;
    if support.len() >= source_count {
        return Ok(());
    }
    let helicity = merged_helicity_identity(
        parents[0].helicity_identity(),
        parents[1].helicity_identity(),
        prepared.quantum.result_spin_state,
    )?;
    let flavour = prepared.flavour.apply(parents);
    let momentum = merged_momentum(parents[0].momentum(), parents[1].momentum())?;
    let parent_colors = [
        colors
            .get(parents[0].dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("left query-local color state disappeared"))?
            .clone(),
        colors
            .get(parents[1].dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("right query-local color state disappeared"))?
            .clone(),
    ];
    let (evaluator_parent_ids, exchange_factor) = prepared.evaluator_parents(concrete_parent_ids);
    let fermion_factor = fermion_ordering_factor(
        &templates.input().current_states,
        [
            parents[0].current_state_template_id(),
            parents[1].current_state_template_id(),
        ],
        [
            parents[0].support_source_slots(),
            parents[1].support_source_slots(),
        ],
        fermion_ordering,
    )?;
    let result_state = templates
        .input()
        .current_states
        .get(prepared.row.result_state_template_id as usize)
        .ok_or_else(|| integrity("pairing-lineage result state is absent"))?;
    let carries_colored_fermion_line =
        result_state.statistics == 1 && result_state.color_representation != 1;
    let pairing_lineages = combine_pairing_lineage_sets(
        seed,
        &currents[concrete_parent_ids[0] as usize].pairing_lineages,
        &currents[concrete_parent_ids[1] as usize].pairing_lineages,
        carries_colored_fermion_line,
    )?;
    if pairing_lineages.is_empty() {
        return Ok(());
    }
    let parent_reflections = if enable_cyclic_trace_reflection {
        [
            currents[concrete_parent_ids[0] as usize].reflection.clone(),
            currents[concrete_parent_ids[1] as usize].reflection.clone(),
        ]
    } else {
        [
            CurrentReflection::Unavailable,
            CurrentReflection::Unavailable,
        ]
    };
    let reversal_masks = if enable_cyclic_trace_reflection {
        current_reversal_masks(&parent_colors, &parent_reflections)
    } else {
        vec![0]
    };
    let local_reflection_proof = enable_cyclic_trace_reflection
        .then(|| transition_reflections.proof(prepared.row.id))
        .flatten();
    let output_factor = prepared.output_factor()?;
    for prepared_witness in &prepared.witnesses {
        if prepared_witness.row.left_shape_string_id != parent_colors[0].output_color_shape_id()
            || prepared_witness.row.right_shape_string_id
                != parent_colors[1].output_color_shape_id()
        {
            continue;
        }
        for reversal_mask in reversal_masks.iter().copied() {
            let mut variant_colors = parent_colors.clone();
            let mut reversal_factor = ExactComplexRational::ONE;
            for index in 0..2 {
                if reversal_mask & (1 << index) == 0 {
                    continue;
                }
                variant_colors[index] = variant_colors[index].reversed()?;
                reversal_factor = reversal_factor.checked_mul(
                    parent_reflections[index]
                        .phase()
                        .expect("reversal mask requires a proven parent phase"),
                )?;
            }
            let Some(result_color) = prepared_witness
                .witness
                .apply(&variant_colors[0], &variant_colors[1])?
            else {
                continue;
            };
            let result_reflection = current_reflection_candidate(
                &result_color,
                &parent_reflections,
                local_reflection_proof,
            )?;
            #[cfg(feature = "on-the-fly-test-support")]
            let diagnostic_result_reflection = result_reflection.clone();
            let color_id = colors.intern(result_color)?;
            let propagator_template_id = if support.len() + 1 < source_count {
                propagators
                    .get(&prepared.row.result_state_template_id)
                    .copied()
                    .flatten()
            } else {
                None
            };
            let key = CurrentCoreKey::new(
                seed.template_catalog_digest,
                RecurrenceNodeKind::Current,
                prepared.row.result_state_template_id,
                color_id,
                support.clone(),
                momentum.clone(),
                helicity.clone(),
                flavour.clone(),
                prepared.quantum.result_quantum_number_flow_id,
                coupling_orders.clone(),
                CurrentSourceBinding::None,
                propagator_template_id,
            )?;
            let result_id = if let Some(id) = current_ids.get(&key).copied() {
                currents[id as usize]
                    .reflection
                    .include(result_reflection)?;
                id
            } else {
                let id = checked_u32(currents.len(), "query-local current count")?;
                let map_key = key.clone();
                let pending = PendingCurrent {
                    key,
                    source_factor: None,
                    contributions: BTreeMap::new(),
                    pairing_lineages: try_clone_pairing_lineages(&pairing_lineages)?,
                    stage: checked_u32(support.len() - 1, "query-local current stage")?,
                    reflection: result_reflection
                        .map_or(CurrentReflection::Unavailable, CurrentReflection::Proven),
                    reflection_certificate: None,
                };
                current_ids.insert(map_key, id);
                currents.push(pending);
                id
            };
            extend_pairing_lineages(
                &mut currents[result_id as usize].pairing_lineages,
                &pairing_lineages,
            )?;
            let contribution_key = ContributionKey::new(
                prepared.row.id,
                evaluator_parent_ids.to_vec(),
                evaluator_parent_ids
                    .iter()
                    .map(|id| currents[*id as usize].key.current_state_template_id())
                    .collect(),
                evaluator_parent_ids
                    .iter()
                    .map(|id| currents[*id as usize].key.momentum().clone())
                    .collect(),
                prepared.row.result_state_template_id,
                prepared.quantum.id,
                LCColorWitnessTermId::new(
                    prepared.row.color_contraction_template_id,
                    prepared_witness.row.ordinal,
                ),
                prepared.quantum_semantic_digest,
                prepared.row.output_projection_string_id,
            )?;
            let pending_key = PendingContributionKey {
                parent_current_ids: evaluator_parent_ids,
                key: contribution_key,
            };
            let witness_factor = prepared_witness.witness.exact_factor();
            let factor = multiply_factors(&[
                prepared.base_factor,
                output_factor,
                exchange_factor,
                witness_factor,
                fermion_factor,
                reversal_factor,
            ])?;
            let aggregate = currents[result_id as usize]
                .contributions
                .entry(pending_key)
                .or_insert(ExactComplexRational::ZERO);
            aggregate_factor(aggregate, factor)?;
            #[cfg(feature = "on-the-fly-test-support")]
            let aggregate_factor_after = *aggregate;
            #[cfg(feature = "on-the-fly-test-support")]
            'transition_diagnostic: {
                if !crate::recurrence::diagnostic::transition_diagnostic_observation_active() {
                    break 'transition_diagnostic;
                }
                use crate::recurrence::diagnostic::{
                    ConstructionTransitionDiagnosticRowV1, observe_transition_diagnostic,
                };

                let digest = |current_id: u32| -> RusticolResult<SemanticDigest> {
                    let current = currents
                        .get(current_id as usize)
                        .ok_or_else(|| integrity("diagnostic current is absent"))?;
                    let color = colors
                        .get(current.key.dynamic_lc_color_state_id())
                        .ok_or_else(|| integrity("diagnostic current color is absent"))?;
                    super::trace::hash_current_key(&current.key, color)
                };
                let result_color = colors
                    .get(currents[result_id as usize].key.dynamic_lc_color_state_id())
                    .ok_or_else(|| integrity("diagnostic result color is absent"))?;
                observe_transition_diagnostic(ConstructionTransitionDiagnosticRowV1 {
                    materialized_sector_id: None,
                    output_current_digest: digest(result_id)?,
                    ordered_parent_digests: [
                        digest(evaluator_parent_ids[0])?,
                        digest(evaluator_parent_ids[1])?,
                    ],
                    transition_template_id: prepared.row.id,
                    transition_semantic_digest: prepared.transition_semantic_digest,
                    evaluator_binding_semantic_digest: prepared.evaluator_binding_digest,
                    result_state_template_id: prepared.row.result_state_template_id,
                    quantum_flow_witness_id: prepared.quantum.id,
                    quantum_semantic_digest: prepared.quantum_semantic_digest,
                    color_contraction_template_id: prepared.row.color_contraction_template_id,
                    color_witness_ordinal: prepared_witness.row.ordinal,
                    color_witness_proof_digest: prepared_witness.witness.proof_digest(),
                    output_projection_id: prepared.row.output_projection_string_id,
                    transition_factor: prepared.transition_factor,
                    contraction_factor: prepared.contraction_factor,
                    output_factor,
                    exchange_factor,
                    witness_factor,
                    reversal_mask,
                    reversal_factor,
                    candidate_factor: factor,
                    aggregate_factor_after,
                    parent_reflection_proof_digests: [
                        parent_reflections[0]
                            .proof()
                            .map(|proof| proof.proof_digest()),
                        parent_reflections[1]
                            .proof()
                            .map(|proof| proof.proof_digest()),
                    ],
                    parent_reflection_phases: [
                        parent_reflections[0].phase(),
                        parent_reflections[1].phase(),
                    ],
                    local_reflection_proof_digest: local_reflection_proof
                        .map(|proof| proof.proof_digest()),
                    local_reflection_phase: local_reflection_proof.map(|proof| proof.phase()),
                    result_reflection_proof_digest: diagnostic_result_reflection
                        .as_ref()
                        .map(|proof| proof.proof_digest()),
                    result_reflection_phase: diagnostic_result_reflection
                        .as_ref()
                        .map(|proof| proof.phase()),
                    output_color_orientation: format!("{result_color:?}"),
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct QuerySupportSizeIndex {
    support_size_by_current: Vec<usize>,
    composite_parent_eligible_by_current: Vec<bool>,
    current_ids_by_size: Vec<Vec<usize>>,
}

impl QuerySupportSizeIndex {
    fn new(
        currents: &[PendingCurrent],
        source_count: usize,
        closure_anchor_slot: Option<u32>,
    ) -> RusticolResult<Self> {
        if closure_anchor_slot.is_some_and(|slot| slot as usize >= source_count) {
            return Err(invalid(
                "query closure anchor is outside the forward source domain",
            ));
        }
        let mut index = Self {
            support_size_by_current: Vec::with_capacity(currents.len()),
            composite_parent_eligible_by_current: Vec::with_capacity(currents.len()),
            current_ids_by_size: (0..source_count).map(|_| Vec::new()).collect(),
        };
        let mut anchor_source_count = 0usize;
        for (current_id, current) in currents.iter().enumerate() {
            let support = current.key.support_source_slots();
            let contains_anchor =
                closure_anchor_slot.is_some_and(|anchor| support.binary_search(&anchor).is_ok());
            if contains_anchor {
                let anchor = closure_anchor_slot.expect("tested present closure anchor");
                if current.key.node_kind() != RecurrenceNodeKind::Source || support != [anchor] {
                    return Err(integrity(
                        "initial forward domain contains a non-singleton closure-anchor current",
                    ));
                }
                anchor_source_count += 1;
            }
            index.append(current_id, support.len(), !contains_anchor)?;
        }
        if closure_anchor_slot.is_some() && anchor_source_count != 1 {
            return Err(integrity(
                "initial forward domain must contain exactly one singleton closure-anchor source",
            ));
        }
        Ok(index)
    }

    fn append(
        &mut self,
        current_id: usize,
        support_size: usize,
        composite_parent_eligible: bool,
    ) -> RusticolResult<()> {
        if current_id != self.support_size_by_current.len()
            || current_id != self.composite_parent_eligible_by_current.len()
        {
            return Err(integrity(
                "query-local support-size index lost current-ID order",
            ));
        }
        let bucket = self
            .current_ids_by_size
            .get_mut(support_size)
            .filter(|_| support_size > 0)
            .ok_or_else(|| integrity("query-local current has invalid forward support size"))?;
        self.support_size_by_current.push(support_size);
        self.composite_parent_eligible_by_current
            .push(composite_parent_eligible);
        if composite_parent_eligible {
            bucket.push(current_id);
        }
        Ok(())
    }

    fn append_stage(
        &mut self,
        currents: &[PendingCurrent],
        stage_current_start: usize,
        target_size: usize,
        closure_anchor_slot: Option<u32>,
    ) -> RusticolResult<()> {
        if stage_current_start != self.support_size_by_current.len()
            || stage_current_start != self.composite_parent_eligible_by_current.len()
            || stage_current_start > currents.len()
        {
            return Err(integrity(
                "query-local support-size index differs from the stage prefix",
            ));
        }
        for (current_id, current) in currents.iter().enumerate().skip(stage_current_start) {
            let support = current.key.support_source_slots();
            let support_size = support.len();
            if support_size != target_size {
                return Err(integrity(
                    "query-local sweep produced a current in the wrong support-size stage",
                ));
            }
            if closure_anchor_slot.is_some_and(|anchor| support.binary_search(&anchor).is_ok()) {
                return Err(integrity(
                    "query-local sweep produced a composite containing the closure anchor",
                ));
            }
            self.append(current_id, support_size, true)?;
        }
        Ok(())
    }

    fn for_each_parent_pair(
        &self,
        target_size: usize,
        stage_current_end: usize,
        mut visit: impl FnMut([usize; 2]) -> RusticolResult<()>,
    ) -> RusticolResult<()> {
        if stage_current_end != self.support_size_by_current.len()
            || stage_current_end != self.composite_parent_eligible_by_current.len()
        {
            return Err(integrity(
                "query-local support-size schedule differs from its current prefix",
            ));
        }
        for left_id in 0..stage_current_end {
            if !self.composite_parent_eligible_by_current[left_id] {
                continue;
            }
            let left_size = self.support_size_by_current[left_id];
            if left_size >= target_size {
                continue;
            }
            let right_size = target_size - left_size;
            let right_ids = self
                .current_ids_by_size
                .get(right_size)
                .ok_or_else(|| integrity("query-local right support-size bucket is absent"))?;
            let first_right = right_ids.partition_point(|right_id| *right_id <= left_id);
            for right_id in right_ids[first_right..].iter().copied() {
                visit([left_id, right_id])?;
            }
        }
        Ok(())
    }
}

/// Fold only a completely certified reciprocal pure-adjoint orbit after the
/// stage fan-in and contact-owner selection are final.  Any missing, stale, or
/// contradictory proof retains both orientations.
fn reconcile_on_the_fly_stage_reflections(
    stage_start: usize,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    source_count: usize,
) -> RusticolResult<()> {
    if stage_start > currents.len() {
        return Err(invalid(
            "query-local reflection stage starts beyond current storage",
        ));
    }
    let stage_end = currents.len();
    let mut visited = vec![false; stage_end - stage_start];
    let mut prune = vec![false; stage_end - stage_start];

    for current_index in stage_start..stage_end {
        let local_index = current_index - stage_start;
        if visited[local_index] {
            continue;
        }
        let key = currents[current_index].key.clone();
        let color = colors
            .get(key.dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("query-local reflection color disappeared"))?
            .clone();
        let Some(word) = color.pure_adjoint_word() else {
            continue;
        };
        if word.len() < 2 {
            continue;
        }
        let canonical = pure_adjoint_word_is_canonical(word);
        let reversed_color_id = colors.intern(color.reversed()?)?;
        if reversed_color_id == key.dynamic_lc_color_state_id() {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let reversed_key = current_key_with_dynamic_color(&key, reversed_color_id)?;
        let Some(reversed_id) = current_ids.get(&reversed_key).copied() else {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        };
        let reversed_index = reversed_id as usize;
        if !(stage_start..stage_end).contains(&reversed_index) {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let reversed_local_index = reversed_index - stage_start;
        visited[local_index] = true;
        visited[reversed_local_index] = true;

        let reversed_color = colors
            .get(reversed_color_id)
            .ok_or_else(|| integrity("query-local reversed reflection color disappeared"))?;
        let Some(reversed_word) = reversed_color.pure_adjoint_word() else {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            currents[reversed_index].reflection = CurrentReflection::Unavailable;
            continue;
        };
        let reciprocal = reciprocal_reflection_proof(
            &currents[current_index].reflection,
            &currents[reversed_index].reflection,
        )?;
        if !reciprocal || canonical == pure_adjoint_word_is_canonical(reversed_word) {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            currents[reversed_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let (canonical_index, reflected_index, pruned_local_index) = if canonical {
            (current_index, reversed_index, reversed_local_index)
        } else {
            (reversed_index, current_index, local_index)
        };
        let canonical_proof = currents[canonical_index]
            .reflection
            .proof()
            .cloned()
            .ok_or_else(|| integrity("canonical query-local reflection proof disappeared"))?;
        let reflected_proof = currents[reflected_index]
            .reflection
            .proof()
            .cloned()
            .ok_or_else(|| integrity("reflected query-local reflection proof disappeared"))?;
        for (label, index, proof) in [
            ("canonical", canonical_index, &canonical_proof),
            ("reflected", reflected_index, &reflected_proof),
        ] {
            let current_color = colors
                .get(currents[index].key.dynamic_lc_color_state_id())
                .ok_or_else(|| integrity(format!("{label} query-local color disappeared")))?;
            if dynamic_color_identity_digest(current_color)? != proof.result_color_identity() {
                return Err(integrity(format!(
                    "{label} query-local reflection proof has a stale color identity"
                )));
            }
        }
        let canonical_color = colors
            .get(currents[canonical_index].key.dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("canonical query-local reflection color disappeared"))?;
        let reflected_color = colors
            .get(currents[reflected_index].key.dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("reflected query-local reflection color disappeared"))?;
        currents[canonical_index].reflection_certificate =
            Some(PendingReflectionCertificate::reciprocal_pair(
                checked_u32(canonical_index, "query-local reflection certificate ID")?,
                checked_u32(
                    canonical_index,
                    "canonical query-local reflection current ID",
                )?,
                checked_u32(
                    reflected_index,
                    "reflected query-local reflection current ID",
                )?,
                &canonical_proof,
                &reflected_proof,
                canonical_color,
                reflected_color,
                source_count,
            )?);
        prune[pruned_local_index] = true;
    }

    for current in &currents[stage_start..] {
        current_ids.remove(&current.key);
    }
    let stage_currents = currents.split_off(stage_start);
    for (local_index, current) in stage_currents.into_iter().enumerate() {
        if prune[local_index] {
            continue;
        }
        if current
            .contributions
            .keys()
            .flat_map(|contribution| contribution.parent_current_ids)
            .any(|parent_id| parent_id as usize >= stage_start)
        {
            return Err(integrity(
                "query-local reflection stage depends on another current in the same stage",
            ));
        }
        let current_id = checked_u32(currents.len(), "query-local reflected current ID")?;
        if current_ids
            .insert(current.key.clone(), current_id)
            .is_some()
        {
            return Err(integrity(
                "query-local reflection reconciliation produced a duplicate current",
            ));
        }
        currents.push(current);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Explicitly mirrors the query-local sweep state.
pub(super) fn build_forward_currents(
    templates: &ValidatedRecurrenceTemplateInput,
    transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    transition_reflections: &TransitionReflectionIndex,
    contact_orbits: &PreparedContactOrbitIndex,
    enable_cyclic_trace_reflection: bool,
    seed: &OnTheFlyProcessSeedV1,
    fermion_ordering: &FermionOrderingContext,
    closure_anchor_slot: u32,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    build_forward_currents_with_anchor_exclusion(
        templates,
        transitions,
        transition_reflections,
        contact_orbits,
        enable_cyclic_trace_reflection,
        seed,
        fermion_ordering,
        Some(closure_anchor_slot),
        coupling_limits,
        propagators,
        colors,
        currents,
        current_ids,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_forward_currents_with_anchor_exclusion(
    templates: &ValidatedRecurrenceTemplateInput,
    transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    transition_reflections: &TransitionReflectionIndex,
    contact_orbits: &PreparedContactOrbitIndex,
    enable_cyclic_trace_reflection: bool,
    seed: &OnTheFlyProcessSeedV1,
    fermion_ordering: &FermionOrderingContext,
    closure_anchor_slot: Option<u32>,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    let source_count = seed.source_anchors.len();
    // The selected closure always combines its singleton anchor source with
    // the exact complementary support. Supports only grow by disjoint union,
    // so a composite containing the anchor can never feed that complement.
    // Keep the source current itself for closure, but exclude it from the
    // ordered composite-parent schedule before any transition is considered.
    let mut support_size_index =
        QuerySupportSizeIndex::new(currents, source_count, closure_anchor_slot)?;
    for target_size in 2..source_count {
        let stage_current_start = currents.len();
        support_size_index.for_each_parent_pair(
            target_size,
            stage_current_start,
            |[left_index, right_index]| {
                let left = &currents[left_index].key;
                let right = &currents[right_index].key;
                debug_assert_eq!(
                    left.support_source_slots().len() + right.support_source_slots().len(),
                    target_size,
                );
                if !supports_are_disjoint(left.support_source_slots(), right.support_source_slots())
                {
                    return Ok(());
                }
                let left_state = left.current_state_template_id();
                let right_state = right.current_state_template_id();
                let Some(rows) = transitions.get(&canonical_state_pair(left_state, right_state))
                else {
                    return Ok(());
                };
                let left_id = checked_u32(left_index, "query-local parent ID")?;
                let right_id = checked_u32(right_index, "query-local parent ID")?;
                for prepared in rows {
                    let Some(parent_ids) =
                        prepared.parent_ids(left_state, right_state, left_id, right_id)
                    else {
                        continue;
                    };
                    include_transition(
                        templates,
                        prepared,
                        transition_reflections,
                        enable_cyclic_trace_reflection,
                        parent_ids,
                        source_count,
                        seed,
                        fermion_ordering,
                        coupling_limits,
                        propagators,
                        colors,
                        currents,
                        current_ids,
                    )?;
                }
                Ok(())
            },
        )?;
        if !contact_orbits.is_empty()
            && let Some(plan) = plan_on_the_fly_contact_orbit_owners(
                stage_current_start,
                transitions,
                contact_orbits,
                currents,
            )?
        {
            plan.commit(currents);
        }
        if enable_cyclic_trace_reflection {
            reconcile_on_the_fly_stage_reflections(
                stage_current_start,
                colors,
                currents,
                current_ids,
                source_count,
            )?;
        }
        support_size_index.append_stage(
            currents,
            stage_current_start,
            target_size,
            closure_anchor_slot,
        )?;
    }
    Ok(())
}

fn closure_quantum_matches(
    quantum: &PreparedClosureQuantum,
    parents: [&CurrentCoreKey; 2],
) -> bool {
    match (quantum.input_states, quantum.input_spins) {
        (None, None) => true,
        (Some(states), Some(spins)) => (0..2).all(|index| {
            states[index] == parents[index].current_state_template_id()
                && quantum_parent_spin_matches(spins[index], parents[index])
        }),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)] // Explicitly mirrors the query-local sweep state.
pub(super) fn build_selected_closures(
    templates: &ValidatedRecurrenceTemplateInput,
    closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    seed: &OnTheFlyProcessSeedV1,
    fermion_ordering: &FermionOrderingContext,
    query: &DecodedLcQueryV1,
    enable_cyclic_trace_reflection: bool,
    colors: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
) -> RusticolResult<Option<Vec<PendingClosure>>> {
    let anchor_support = [query.closure_anchor_slot];
    let complement_support = (0..query.selected_sources.len() as u32)
        .filter(|slot| *slot != query.closure_anchor_slot)
        .collect::<Vec<_>>();
    let anchor_ids = currents
        .iter()
        .enumerate()
        .filter(|(_, current)| {
            current.key.node_kind() == RecurrenceNodeKind::Source
                && current.key.support_source_slots() == anchor_support
        })
        .map(|(id, _)| checked_u32(id, "closure anchor current ID"))
        .collect::<RusticolResult<Vec<_>>>()?;
    if anchor_ids.len() != 1 {
        return Err(invalid(
            "decoded closure anchor does not identify exactly one source",
        ));
    }
    let complement_ids = currents
        .iter()
        .enumerate()
        .filter(|(_, current)| current.key.support_source_slots() == complement_support)
        .map(|(id, _)| checked_u32(id, "closure complement current ID"))
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut retained = BTreeMap::<PendingClosureKey, PendingClosure>::new();
    let anchor_id = anchor_ids[0];
    let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
    for complement_id in complement_ids {
        let complement_state = currents[complement_id as usize]
            .key
            .current_state_template_id();
        let Some(rows) = closures.get(&canonical_state_pair(anchor_state, complement_state)) else {
            continue;
        };
        for closure in rows {
            let Some(concrete_parent_ids) =
                closure.parent_ids(anchor_state, complement_state, anchor_id, complement_id)
            else {
                continue;
            };
            let parents = [
                &currents[concrete_parent_ids[0] as usize].key,
                &currents[concrete_parent_ids[1] as usize].key,
            ];
            let pairing_lineages = combine_pairing_lineage_sets(
                // A physical closure is bosonic and must close the final
                // unmatched fundamental/antifundamental pair, if any.
                // Completed pairs remain query-local proof state only.
                seed,
                &currents[concrete_parent_ids[0] as usize].pairing_lineages,
                &currents[concrete_parent_ids[1] as usize].pairing_lineages,
                false,
            )?;
            let mut complete_lineages = Vec::new();
            complete_lineages
                .try_reserve_exact(pairing_lineages.len())
                .map_err(|error| {
                    invalid(format!(
                        "complete pairing-lineage allocation failed: {error}"
                    ))
                })?;
            for lineage in pairing_lineages {
                if complete_pairing_lineage(seed, &lineage)? {
                    complete_lineages.push(lineage);
                }
            }
            let pairing_lineages = complete_lineages;
            if pairing_lineages.is_empty() {
                continue;
            }
            let parent_colors = [
                colors
                    .get(parents[0].dynamic_lc_color_state_id())
                    .ok_or_else(|| integrity("closure left color state disappeared"))?,
                colors
                    .get(parents[1].dynamic_lc_color_state_id())
                    .ok_or_else(|| integrity("closure right color state disappeared"))?,
            ];
            let (evaluator_parent_ids, exchange_factor) =
                closure.evaluator_parents(concrete_parent_ids);
            let fermion_factor = fermion_ordering_factor(
                &templates.input().current_states,
                [
                    parents[0].current_state_template_id(),
                    parents[1].current_state_template_id(),
                ],
                [
                    parents[0].support_source_slots(),
                    parents[1].support_source_slots(),
                ],
                fermion_ordering,
            )?;
            for quantum in closure
                .quantum_flows
                .iter()
                .filter(|quantum| closure_quantum_matches(quantum, parents))
            {
                for witness in &closure.witnesses {
                    if witness.row.left_shape_string_id != parent_colors[0].output_color_shape_id()
                        || witness.row.right_shape_string_id
                            != parent_colors[1].output_color_shape_id()
                    {
                        continue;
                    }
                    let closed = witness
                        .witness
                        .closed_components(parent_colors[0], parent_colors[1])?;
                    let reflection_factor = if query_target_matches(&closed, query) {
                        ExactComplexRational::ONE
                    } else if enable_cyclic_trace_reflection {
                        let Some(factor) = reflected_query_closure_factor(
                            query,
                            &closed,
                            colors,
                            currents,
                            concrete_parent_ids,
                        )?
                        else {
                            continue;
                        };
                        factor
                    } else {
                        continue;
                    };
                    let key = PendingClosureKey {
                        closure_template_id: closure.row.id,
                        quantum_flow_template_id: quantum.row.map(|row| row.id),
                        parent_current_ids: evaluator_parent_ids,
                        color_witness_term_id: LCColorWitnessTermId::new(
                            closure.row.color_contraction_template_id,
                            witness.row.ordinal,
                        ),
                    };
                    let factor = multiply_factors(&[
                        closure.base_factor,
                        quantum.output_factor()?,
                        exchange_factor,
                        witness.witness.exact_factor(),
                        fermion_factor,
                        reflection_factor,
                    ])?;
                    let coefficients = closure.component_coefficients.clone();
                    match retained.entry(key.clone()) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(PendingClosure {
                                key,
                                factor,
                                component_coefficients: coefficients,
                                pairing_lineages: try_clone_pairing_lineages(&pairing_lineages)?,
                            });
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            if entry.get().component_coefficients != coefficients {
                                return Err(integrity(
                                    "equal closure identities have different component coefficients",
                                ));
                            }
                            aggregate_factor(&mut entry.get_mut().factor, factor)?;
                            extend_pairing_lineages(
                                &mut entry.get_mut().pairing_lineages,
                                &pairing_lineages,
                            )?;
                        }
                    }
                }
            }
        }
    }
    retained.retain(|_, closure| !closure.factor.is_zero());
    for closure in retained.values_mut() {
        retain_canonical_pairing_lineage(&mut closure.pairing_lineages)?;
    }
    if retained.is_empty() {
        // The process-global immutable grammar proved this closure domain
        // complete before it entered the runtime cache.
        return Ok(None);
    }
    Ok(Some(retained.into_values().collect()))
}

pub(super) fn validate_prepared_closure_domain(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
) -> RusticolResult<()> {
    let expected = &templates.input().closures;
    let observed_count = closures.values().try_fold(0usize, |count, rows| {
        count
            .checked_add(rows.len())
            .ok_or_else(|| invalid("prepared closure-domain row count exceeds usize"))
    })?;
    if observed_count != expected.len() {
        return Err(integrity(format!(
            "prepared closure domain has {observed_count} rows, expected {}",
            expected.len()
        )));
    }

    let mut observed = vec![false; expected.len()];
    for (state_pair, rows) in closures {
        for closure in rows {
            let expected_row = expected
                .get(closure.row.id as usize)
                .filter(|row| row.id == closure.row.id)
                .ok_or_else(|| integrity("prepared closure domain contains an unknown row"))?;
            if closure.row != *expected_row {
                return Err(integrity(
                    "prepared closure row differs from its authenticated template",
                ));
            }
            let expected_states: [u32; 2] = catalog
                .u32_sequence(expected_row.input_state_sequence_id, "closure input states")?
                .try_into()
                .map_err(|_| integrity("authenticated closure input-state pair is not binary"))?;
            if closure.input_states != expected_states
                || *state_pair != canonical_state_pair(expected_states[0], expected_states[1])
            {
                return Err(integrity(
                    "prepared closure parent-state pair differs from its authenticated template",
                ));
            }
            let seen = observed
                .get_mut(closure.row.id as usize)
                .ok_or_else(|| integrity("prepared closure row ID exceeds its domain"))?;
            if std::mem::replace(seen, true) {
                return Err(integrity(
                    "prepared closure domain repeats an authenticated row",
                ));
            }
        }
    }
    if observed.iter().any(|seen| !seen) {
        return Err(integrity(
            "prepared closure domain omits an authenticated row",
        ));
    }
    Ok(())
}

pub(super) fn live_current_ids(
    currents: &[PendingCurrent],
    closures: &[PendingClosure],
) -> RusticolResult<BTreeSet<u32>> {
    let mut live = BTreeSet::new();
    let mut queue = VecDeque::new();
    for closure in closures {
        for parent in closure.key.parent_current_ids {
            if live.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    while let Some(current_id) = queue.pop_front() {
        let current = currents
            .get(current_id as usize)
            .ok_or_else(|| integrity("liveness queue references an absent current"))?;
        for (contribution, factor) in &current.contributions {
            if factor.is_zero() {
                continue;
            }
            for parent in contribution.parent_current_ids {
                if live.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use crate::recurrence::DynamicLCColorStateId;
    use crate::recurrence::construct::CurrentReflectionProof;
    use crate::recurrence::contact_orbit_owner::{
        ContactOrbitStepProof, ContactOrbitTestBinding, contact_orbit_application_for_test,
        contact_orbit_test_template, final_contact_orbit_step_for_test,
        partial_contact_orbit_step_for_test, prepared_contact_orbit_transition_for_test,
    };
    use crate::recurrence::{ExactRational, LCColorEndpoint, LCColorPortBinding};

    use super::*;

    #[test]
    fn fermion_ordering_filter_uses_authenticated_source_color_role() {
        assert!(
            on_the_fly_source_requires_exterior_sign(true, OnTheFlyExternalColorRoleV1::Singlet,)
                .unwrap()
        );
        assert!(
            !on_the_fly_source_requires_exterior_sign(
                true,
                OnTheFlyExternalColorRoleV1::Fundamental,
            )
            .unwrap()
        );
        assert!(
            !on_the_fly_source_requires_exterior_sign(
                true,
                OnTheFlyExternalColorRoleV1::Antifundamental,
            )
            .unwrap()
        );
        assert!(
            !on_the_fly_source_requires_exterior_sign(false, OnTheFlyExternalColorRoleV1::Adjoint,)
                .unwrap()
        );
    }

    #[test]
    fn fermion_ordering_filter_rejects_unsupported_or_unauthenticated_roles() {
        let adjoint =
            on_the_fly_source_requires_exterior_sign(true, OnTheFlyExternalColorRoleV1::Adjoint)
                .unwrap_err();
        assert!(adjoint.to_string().contains("adjoint fermion"));
        let bosonic_endpoint = on_the_fly_source_requires_exterior_sign(
            false,
            OnTheFlyExternalColorRoleV1::Fundamental,
        )
        .unwrap_err();
        assert!(bosonic_endpoint.to_string().contains("bosonic source"));
    }

    fn contact_digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
    }

    fn reflection_test_states() -> (
        DynamicLCColorStateInterner,
        DynamicLCColorStateId,
        DynamicLCColorStateId,
    ) {
        let canonical = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 3, 2]).unwrap(),
            ],
        )
        .unwrap();
        let reversed = canonical.reversed().unwrap();
        let mut colors = DynamicLCColorStateInterner::default();
        let canonical_id = colors.intern(canonical).unwrap();
        let reversed_id = colors.intern(reversed).unwrap();
        (colors, canonical_id, reversed_id)
    }

    fn proven_reflection(
        colors: &DynamicLCColorStateInterner,
        color_id: DynamicLCColorStateId,
        phase: ExactComplexRational,
        lineage: u8,
    ) -> CurrentReflection {
        let color = colors.get(color_id).unwrap();
        CurrentReflection::Proven(
            CurrentReflectionProof::new(
                phase,
                [contact_digest(lineage)],
                dynamic_color_identity_digest(color).unwrap(),
            )
            .unwrap(),
        )
    }

    fn reflection_current(
        node_kind: RecurrenceNodeKind,
        color_id: DynamicLCColorStateId,
        support: &[u32],
        reflection: CurrentReflection,
    ) -> PendingCurrent {
        PendingCurrent {
            key: CurrentCoreKey::new(
                contact_digest(230),
                node_kind,
                0,
                color_id,
                support.to_vec(),
                CanonicalMomentumLinearForm::new(
                    support
                        .iter()
                        .copied()
                        .map(|source_slot| MomentumTerm {
                            source_slot,
                            coefficient: 1,
                        })
                        .collect(),
                )
                .unwrap(),
                CurrentHelicityIdentity::topology_replay(
                    0,
                    support
                        .iter()
                        .copied()
                        .map(|source_slot| SourceStateAssignment::new(source_slot, 0))
                        .collect(),
                )
                .unwrap(),
                vec![1],
                0,
                vec![0],
                if node_kind == RecurrenceNodeKind::Source {
                    CurrentSourceBinding::FixedTemplate(0)
                } else {
                    CurrentSourceBinding::None
                },
                None,
            )
            .unwrap(),
            source_factor: (node_kind == RecurrenceNodeKind::Source)
                .then_some(ExactComplexRational::ONE),
            contributions: BTreeMap::new(),
            pairing_lineages: vec![PendingPairingLineage {
                completed_pairs: Vec::new(),
                unmatched_endpoint: None,
            }],
            stage: u32::try_from(support.len().saturating_sub(1)).unwrap(),
            reflection,
            reflection_certificate: None,
        }
    }

    #[test]
    fn query_local_reflection_fold_is_insertion_invariant_and_fail_closed() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let run = |reverse_insertion: bool, prove_both: bool| {
            let (mut colors, canonical_id, reversed_id) = reflection_test_states();
            let mut currents = vec![
                reflection_current(
                    RecurrenceNodeKind::Current,
                    canonical_id,
                    &[1, 2, 3],
                    proven_reflection(&colors, canonical_id, minus_one, 201),
                ),
                reflection_current(
                    RecurrenceNodeKind::Current,
                    reversed_id,
                    &[1, 2, 3],
                    if prove_both {
                        proven_reflection(&colors, reversed_id, minus_one, 202)
                    } else {
                        let mut late_unproved =
                            proven_reflection(&colors, reversed_id, minus_one, 202);
                        late_unproved.include(None).unwrap();
                        late_unproved
                    },
                ),
            ];
            if reverse_insertion {
                currents.reverse();
            }
            let mut current_ids = currents
                .iter()
                .enumerate()
                .map(|(id, current)| (current.key.clone(), id as u32))
                .collect::<BTreeMap<_, _>>();
            reconcile_on_the_fly_stage_reflections(
                0,
                &mut colors,
                &mut currents,
                &mut current_ids,
                4,
            )
            .unwrap();
            (colors, currents, current_ids)
        };

        let (_, folded, folded_ids) = run(false, true);
        let (_, folded_reordered, folded_reordered_ids) = run(true, true);
        assert_eq!(folded.len(), 1);
        assert_eq!(folded_ids.len(), 1);
        assert_eq!(folded[0].key, folded_reordered[0].key);
        assert_eq!(folded_ids, folded_reordered_ids);
        for current in [&folded[0], &folded_reordered[0]] {
            let certificate = current.reflection_certificate.as_ref().unwrap();
            assert!(certificate.is_reciprocal_two_cycle());
            assert_eq!(certificate.source_permutation(), [0, 2, 1, 3]);
            assert_eq!(certificate.canonical_phase(), minus_one);
        }

        let (_, residual, residual_ids) = run(false, false);
        assert_eq!(residual.len(), 2);
        assert_eq!(residual_ids.len(), 2);
        assert!(
            residual
                .iter()
                .all(|current| current.reflection_certificate.is_none())
        );
    }

    #[test]
    fn reflected_query_closure_uses_only_the_certified_exact_phase() {
        let canonical_phase =
            ExactComplexRational::new(ExactRational::new(2, 1).unwrap(), ExactRational::ZERO);
        let reflected_phase =
            ExactComplexRational::new(ExactRational::new(1, 2).unwrap(), ExactRational::ZERO);
        let (mut colors, canonical_id, reversed_id) = reflection_test_states();
        let mut currents = vec![
            reflection_current(
                RecurrenceNodeKind::Source,
                canonical_id,
                &[0],
                CurrentReflection::Unavailable,
            ),
            reflection_current(
                RecurrenceNodeKind::Current,
                canonical_id,
                &[1, 2, 3],
                proven_reflection(&colors, canonical_id, canonical_phase, 203),
            ),
            reflection_current(
                RecurrenceNodeKind::Current,
                reversed_id,
                &[1, 2, 3],
                proven_reflection(&colors, reversed_id, reflected_phase, 204),
            ),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<BTreeMap<_, _>>();
        reconcile_on_the_fly_stage_reflections(1, &mut colors, &mut currents, &mut current_ids, 4)
            .unwrap();
        assert_eq!(currents.len(), 2);

        let seed = adjoint_reflection_seed();
        let query = DecodedLcQueryV1::new(
            &seed,
            vec![0, 1, 2, 3],
            &[0, 0, 0, 0],
            OnTheFlyLcSelectorV1::single_trace(vec![0, 2, 3, 1]),
        )
        .unwrap();
        let query = OnTheFlyClosureAnchorPolicyV1::certified_cyclic_minimum(0)
            .canonicalize_query(&seed, query)
            .unwrap();
        let closed =
            vec![LCColorComponent::new(LCColorComponentKind::Trace, vec![0, 1, 3, 2]).unwrap()];
        assert!(!query_target_matches(&closed, &query));
        assert_eq!(
            reflected_query_closure_factor(&query, &closed, &colors, &currents, [0, 1]).unwrap(),
            Some(canonical_phase),
        );

        let certificate = currents[1].reflection_certificate.take();
        assert_eq!(
            reflected_query_closure_factor(&query, &closed, &colors, &currents, [0, 1]).unwrap(),
            None,
        );
        currents[1].reflection_certificate = certificate;
        currents[1].reflection = proven_reflection(&colors, canonical_id, canonical_phase, 205);
        assert!(
            reflected_query_closure_factor(&query, &closed, &colors, &currents, [0, 1]).is_err()
        );
    }

    fn contact_current(
        node_kind: RecurrenceNodeKind,
        state: u32,
        color: u32,
        support: &[u32],
    ) -> PendingCurrent {
        contact_current_with_source_template(
            node_kind,
            state,
            color,
            support,
            support.first().copied().unwrap_or_default(),
        )
    }

    fn contact_current_with_source_template(
        node_kind: RecurrenceNodeKind,
        state: u32,
        color: u32,
        support: &[u32],
        source_template_id: u32,
    ) -> PendingCurrent {
        let key = CurrentCoreKey::new(
            contact_digest(230),
            node_kind,
            state,
            DynamicLCColorStateId::from_interner(color),
            support.to_vec(),
            CanonicalMomentumLinearForm::new(
                support
                    .iter()
                    .copied()
                    .map(|source_slot| MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    })
                    .collect(),
            )
            .unwrap(),
            CurrentHelicityIdentity::topology_replay(
                0,
                support
                    .iter()
                    .copied()
                    .map(|source_slot| SourceStateAssignment::new(source_slot, 0))
                    .collect(),
            )
            .unwrap(),
            vec![state as i32],
            0,
            vec![0],
            if node_kind == RecurrenceNodeKind::Source {
                CurrentSourceBinding::FixedTemplate(source_template_id)
            } else {
                CurrentSourceBinding::None
            },
            None,
        )
        .unwrap();
        PendingCurrent {
            key,
            source_factor: (node_kind == RecurrenceNodeKind::Source)
                .then_some(ExactComplexRational::ONE),
            contributions: BTreeMap::new(),
            pairing_lineages: vec![PendingPairingLineage {
                completed_pairs: Vec::new(),
                unmatched_endpoint: None,
            }],
            stage: u32::try_from(support.len().saturating_sub(1)).unwrap(),
            reflection: CurrentReflection::Unavailable,
            reflection_certificate: None,
        }
    }

    fn legacy_support_size_pairs(support_sizes: &[usize], target_size: usize) -> Vec<[usize; 2]> {
        let mut pairs = Vec::new();
        for left_id in 0..support_sizes.len() {
            if support_sizes[left_id] >= target_size {
                continue;
            }
            for right_id in (left_id + 1)..support_sizes.len() {
                if support_sizes[right_id] < target_size
                    && support_sizes[left_id] + support_sizes[right_id] == target_size
                {
                    pairs.push([left_id, right_id]);
                }
            }
        }
        pairs
    }

    fn legacy_anchor_filtered_pairs(
        supports: &[Vec<u32>],
        target_size: usize,
        closure_anchor_slot: u32,
    ) -> Vec<[usize; 2]> {
        legacy_support_size_pairs(
            &supports.iter().map(Vec::len).collect::<Vec<_>>(),
            target_size,
        )
        .into_iter()
        .filter(|[left, right]| {
            !supports[*left].contains(&closure_anchor_slot)
                && !supports[*right].contains(&closure_anchor_slot)
        })
        .collect()
    }

    #[test]
    fn query_support_size_index_preserves_legacy_pair_order() {
        let mut random_state = 0x6a09_e667_f3bc_c909_u64;
        for source_count in 3..=12 {
            for case_index in 0..64 {
                let current_count = source_count + case_index % (source_count * 3);
                let mut support_sizes = Vec::with_capacity(current_count);
                for _ in 0..current_count {
                    random_state = random_state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    support_sizes.push(
                        1 + usize::try_from(random_state % (source_count as u64 - 1)).unwrap(),
                    );
                }

                let mut index = QuerySupportSizeIndex {
                    support_size_by_current: Vec::new(),
                    composite_parent_eligible_by_current: Vec::new(),
                    current_ids_by_size: (0..source_count).map(|_| Vec::new()).collect(),
                };
                for (current_id, support_size) in support_sizes.iter().copied().enumerate() {
                    index.append(current_id, support_size, true).unwrap();
                }

                for target_size in 2..source_count {
                    let mut indexed_pairs = Vec::new();
                    index
                        .for_each_parent_pair(target_size, support_sizes.len(), |pair| {
                            indexed_pairs.push(pair);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(
                        indexed_pairs,
                        legacy_support_size_pairs(&support_sizes, target_size),
                        "source_count={source_count}, case_index={case_index}, target_size={target_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn all_flow_queries_use_distinct_anchor_filters_without_reordering_pairs() {
        // An all-flow family constructs each decoded query independently.
        // Rebuild the same support domain for every possible query anchor to
        // prove that no anchor eligibility leaks from one query to the next.
        for source_count in 2_u32..=8 {
            let mut supports = (1_u32..(1_u32 << source_count))
                .filter(|mask| mask.count_ones() < source_count)
                .map(|mask| {
                    (0..source_count)
                        .filter(|slot| mask & (1 << slot) != 0)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            supports.sort_by_key(|support| (support.len(), support.clone()));

            for closure_anchor_slot in 0..source_count {
                let mut index = QuerySupportSizeIndex {
                    support_size_by_current: Vec::new(),
                    composite_parent_eligible_by_current: Vec::new(),
                    current_ids_by_size: (0..source_count as usize).map(|_| Vec::new()).collect(),
                };
                for (current_id, support) in supports.iter().enumerate() {
                    index
                        .append(
                            current_id,
                            support.len(),
                            !support.contains(&closure_anchor_slot),
                        )
                        .unwrap();
                }

                for target_size in 2..source_count as usize {
                    let mut filtered_pairs = Vec::new();
                    index
                        .for_each_parent_pair(target_size, supports.len(), |pair| {
                            filtered_pairs.push(pair);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(
                        filtered_pairs,
                        legacy_anchor_filtered_pairs(&supports, target_size, closure_anchor_slot,),
                        "source_count={source_count}, closure_anchor_slot={closure_anchor_slot}, target_size={target_size}",
                    );
                }
            }
        }
    }

    #[test]
    fn two_source_anchor_is_retained_but_cannot_enter_composite_schedule() {
        let currents = vec![
            contact_current(RecurrenceNodeKind::Source, 0, 0, &[0]),
            contact_current(RecurrenceNodeKind::Source, 0, 0, &[1]),
        ];
        let index = QuerySupportSizeIndex::new(&currents, 2, Some(1)).unwrap();
        assert_eq!(index.support_size_by_current, [1, 1]);
        assert_eq!(index.composite_parent_eligible_by_current, [true, false]);
        let mut pairs = Vec::new();
        index
            .for_each_parent_pair(2, currents.len(), |pair| {
                pairs.push(pair);
                Ok(())
            })
            .unwrap();
        assert!(pairs.is_empty());

        let error = QuerySupportSizeIndex::new(&currents, 2, Some(2)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("outside the forward source domain")
        );

        let missing_anchor = vec![contact_current(RecurrenceNodeKind::Source, 0, 0, &[0])];
        let error = QuerySupportSizeIndex::new(&missing_anchor, 2, Some(1)).unwrap_err();
        assert!(error.to_string().contains("exactly one singleton"));

        let malformed_anchor = vec![
            contact_current(RecurrenceNodeKind::Source, 0, 0, &[0]),
            contact_current(RecurrenceNodeKind::Current, 0, 0, &[0, 1]),
        ];
        let error = QuerySupportSizeIndex::new(&malformed_anchor, 2, Some(1)).unwrap_err();
        assert!(error.to_string().contains("non-singleton closure-anchor"));
    }

    #[test]
    fn query_support_size_index_enforces_stage_prefix() {
        let mut index = QuerySupportSizeIndex {
            support_size_by_current: Vec::new(),
            composite_parent_eligible_by_current: Vec::new(),
            current_ids_by_size: (0..4).map(|_| Vec::new()).collect(),
        };
        index.append(0, 1, true).unwrap();
        index.append(1, 1, true).unwrap();
        index.append(2, 1, true).unwrap();

        let error = index
            .for_each_parent_pair(2, 2, |_| Ok(()))
            .expect_err("a size bucket extending past the stage prefix must be rejected");
        assert!(
            error
                .to_string()
                .contains("differs from its current prefix")
        );
    }

    fn contact_transition(
        step: ContactOrbitStepProof,
        input_state_template_ids: [u32; 2],
        transition_digest: u8,
    ) -> PreparedContactOrbitTransition {
        contact_transition_with_witness(
            step,
            input_state_template_ids,
            transition_digest,
            LCColorWitnessTermId::new(4, 0),
        )
    }

    fn contact_transition_with_witness(
        step: ContactOrbitStepProof,
        input_state_template_ids: [u32; 2],
        transition_digest: u8,
        color_witness_term_id: LCColorWitnessTermId,
    ) -> PreparedContactOrbitTransition {
        let mut application = contact_orbit_application_for_test();
        application.color_witness_term_id = color_witness_term_id;
        prepared_contact_orbit_transition_for_test(
            step,
            input_state_template_ids,
            contact_digest(transition_digest),
            application,
        )
    }

    fn add_contact_contribution(
        currents: &mut [PendingCurrent],
        destination: u32,
        transition: u32,
        parents: [u32; 2],
    ) {
        let parent_keys = parents.map(|id| &currents[id as usize].key);
        let destination_key = &currents[destination as usize].key;
        let key = ContributionKey::new(
            transition,
            parents.to_vec(),
            parent_keys
                .iter()
                .map(|key| key.current_state_template_id())
                .collect(),
            parent_keys
                .iter()
                .map(|key| key.momentum().clone())
                .collect(),
            destination_key.current_state_template_id(),
            0,
            LCColorWitnessTermId::new(4, 0),
            contact_digest(2),
            8,
        )
        .unwrap();
        assert!(
            currents[destination as usize]
                .contributions
                .insert(
                    PendingContributionKey {
                        parent_current_ids: parents,
                        key,
                    },
                    ExactComplexRational::ONE,
                )
                .is_none()
        );
    }

    fn contact_transition_ids(current: &PendingCurrent) -> Vec<u32> {
        current
            .contributions
            .keys()
            .map(|pending| pending.key.transition_template_id())
            .collect()
    }

    fn pending_contact_counts(currents: &[PendingCurrent]) -> (usize, usize) {
        (
            currents.len(),
            currents
                .iter()
                .map(|current| current.contributions.len())
                .sum(),
        )
    }

    fn scalar_contact_seed(source_count: u32) -> OnTheFlyProcessSeedV1 {
        let state = OnTheFlySourceStateV1::new(
            0,
            0,
            0,
            0,
            0,
            contact_digest(231),
            contact_digest(232),
            1,
            ExactComplexRational::ONE,
            0,
            0,
            vec![1],
            0,
            contact_digest(233),
            OnTheFlySourceWavefunctionFamilyV1::Scalar,
            OnTheFlySourceOrientationV1::SelfConjugate,
            None,
        )
        .unwrap();
        let anchors = (0_u32..source_count)
            .map(|source_slot| {
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    source_slot,
                    false,
                    OnTheFlyExternalColorRoleV1::Singlet,
                    false,
                    None,
                    vec![state.clone()],
                )
                .unwrap()
            })
            .collect();
        OnTheFlyProcessSeedV1::new(
            contact_digest(234),
            contact_digest(235),
            contact_digest(230),
            contact_digest(236),
            contact_digest(237),
            contact_digest(238),
            "raw-amplitude-contact-test",
            ExactComplexRational::ONE,
            anchors,
            (0_u32..source_count).collect(),
            OnTheFlyCouplingOrderPolicyV1::Explicit,
            vec![1],
            vec![Some(0)],
            Vec::new(),
        )
        .unwrap()
    }

    fn adjoint_reflection_seed() -> OnTheFlyProcessSeedV1 {
        let state = OnTheFlySourceStateV1::new(
            0,
            0,
            0,
            0,
            0,
            contact_digest(241),
            contact_digest(242),
            1,
            ExactComplexRational::ONE,
            0,
            0,
            vec![1],
            0,
            contact_digest(243),
            OnTheFlySourceWavefunctionFamilyV1::Vector,
            OnTheFlySourceOrientationV1::SelfConjugate,
            None,
        )
        .unwrap();
        let anchors = (0_u32..4)
            .map(|source_slot| {
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    source_slot,
                    source_slot < 2,
                    OnTheFlyExternalColorRoleV1::Adjoint,
                    false,
                    None,
                    vec![state.clone()],
                )
                .unwrap()
            })
            .collect();
        OnTheFlyProcessSeedV1::new(
            contact_digest(244),
            contact_digest(245),
            contact_digest(230),
            contact_digest(246),
            contact_digest(247),
            contact_digest(248),
            "raw-amplitude-reflection-test",
            ExactComplexRational::ONE,
            anchors,
            vec![0, 1, 2, 3],
            OnTheFlyCouplingOrderPolicyV1::Explicit,
            vec![1],
            vec![Some(0)],
            Vec::new(),
        )
        .unwrap()
    }

    fn production_contact_barrier_currents(
        reverse_transition_order: bool,
        include_ordinary_scalar_transition: bool,
        closure_anchor_slot: Option<u32>,
    ) -> Vec<PendingCurrent> {
        let mut template_input = contact_orbit_test_template(ContactOrbitTestBinding::One);
        let mut intermediate_state = template_input.current_states[0];
        intermediate_state.id = 1;
        let mut append_canonical_string = |value: &[u8]| {
            let previous_string = template_input.string_ranges.last().unwrap();
            let previous_range = previous_string
                .as_usize_range(template_input.string_bytes.len(), "test string")
                .unwrap();
            assert!(
                &template_input.string_bytes[previous_range] < value,
                "the intermediate-state fixture must extend the canonical string catalog"
            );
            let id = u32::try_from(template_input.string_ranges.len()).unwrap();
            template_input
                .string_ranges
                .push(crate::recurrence::CheckedTableRange::new(
                    u64::try_from(template_input.string_bytes.len()).unwrap(),
                    u64::try_from(value.len()).unwrap(),
                ));
            template_input.string_bytes.extend_from_slice(value);
            id
        };
        let intermediate_propagator_template_id =
            append_canonical_string(b"zz-on-the-fly-intermediate-propagator");
        let intermediate_state_template_id =
            append_canonical_string(b"zz-on-the-fly-intermediate-state");
        intermediate_state.template_string_id = intermediate_state_template_id;
        let intermediate_digest_id = u32::try_from(template_input.digest_catalog.len()).unwrap();
        template_input
            .digest_catalog
            .push(crate::recurrence::template::DigestCatalogRow {
                id: intermediate_digest_id,
                value: [240; 32],
            });
        intermediate_state.semantic_digest_id = intermediate_digest_id;
        template_input.current_states.push(intermediate_state);
        let intermediate_propagator_digest_id =
            u32::try_from(template_input.digest_catalog.len()).unwrap();
        template_input
            .digest_catalog
            .push(crate::recurrence::template::DigestCatalogRow {
                id: intermediate_propagator_digest_id,
                value: [241; 32],
            });
        let mut intermediate_propagator = template_input.propagators[0];
        intermediate_propagator.id = 1;
        intermediate_propagator.template_string_id = intermediate_propagator_template_id;
        intermediate_propagator.state_template_id = 1;
        intermediate_propagator.applies_propagator = 0;
        intermediate_propagator.evaluator_binding_id = MISSING_U32;
        intermediate_propagator.numerator_expression_digest_id = MISSING_U32;
        intermediate_propagator.denominator_expression_digest_id = MISSING_U32;
        intermediate_propagator.semantic_digest_id = intermediate_propagator_digest_id;
        template_input.propagators.push(intermediate_propagator);
        template_input.catalog_header[0].current_state_count = 2;
        template_input.catalog_header[0].propagator_count = 2;
        let templates = template_input.validate().unwrap();
        let catalog = TemplateCatalog::new(templates.input()).unwrap();
        let prepared = prepared_transitions(&templates, &catalog).unwrap();
        let base = prepared.values().next().unwrap()[0].clone();
        let partial_steps = [
            partial_contact_orbit_step_for_test(0, 1, 2, 20, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(1, 0, 2, 30, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(0, 2, 1, 21, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(2, 0, 1, 31, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(0, 3, 1, 22, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(3, 0, 1, 32, [0, 0, 0, 0]),
        ];
        let final_steps = [
            final_contact_orbit_step_for_test(&[0], &[1, 2], 3, 60, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[1, 2], &[0], 3, 70, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[0], &[1, 3], 2, 61, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[1, 3], &[0], 2, 71, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[0], &[2, 3], 1, 62, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[2, 3], &[0], 1, 72, [0, 0, 0, 0]),
        ];
        let transition_order = if reverse_transition_order {
            (0_usize..partial_steps.len()).rev().collect::<Vec<_>>()
        } else {
            (0_usize..partial_steps.len()).collect::<Vec<_>>()
        };
        let build_row = |index: usize,
                         id_offset: u32,
                         step: ContactOrbitStepProof,
                         input_states: [u32; 2],
                         result_state: u32| {
            let id = id_offset + u32::try_from(index).unwrap();
            let digest_byte = 40 + u8::try_from(id).unwrap();
            let mut row = base.clone();
            row.row.id = id;
            row.input_states = input_states;
            row.row.result_state_template_id = result_state;
            row.quantum.result_state_template_id = result_state;
            row.local_orders = vec![0].into_boxed_slice();
            row.transition_semantic_digest = contact_digest(digest_byte);
            row.contact_orbit = Some(contact_transition_with_witness(
                step,
                input_states,
                digest_byte,
                LCColorWitnessTermId::new(
                    row.row.color_contraction_template_id,
                    row.witnesses[0].row.ordinal,
                ),
            ));
            row
        };
        let mut partial_rows = transition_order
            .iter()
            .copied()
            .map(|index| build_row(index, 0, partial_steps[index].clone(), [0, 0], 1))
            .collect::<Vec<_>>();
        let final_rows = transition_order
            .into_iter()
            .map(|index| {
                let input_states = if index % 2 == 0 { [0, 1] } else { [1, 0] };
                build_row(index, 6, final_steps[index].clone(), input_states, 1)
            })
            .collect::<Vec<_>>();
        if include_ordinary_scalar_transition {
            let mut ordinary = base.clone();
            ordinary.row.id = 12;
            ordinary.input_states = [0, 0];
            ordinary.row.result_state_template_id = 0;
            ordinary.quantum.result_state_template_id = 0;
            ordinary.local_orders = vec![0].into_boxed_slice();
            ordinary.transition_semantic_digest = contact_digest(120);
            ordinary.contact_orbit = None;
            partial_rows.push(ordinary);
        }
        let transitions = BTreeMap::from([((0, 0), partial_rows), ((0, 1), final_rows)]);
        let propagators = propagator_by_state(&templates).unwrap();
        let seed = scalar_contact_seed(4);
        let mut colors = DynamicLCColorStateInterner::default();
        let color_id = colors
            .intern(
                DynamicLCColorState::new(
                    base.witnesses[0].row.left_shape_string_id,
                    None,
                    Vec::new(),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(color_id, DynamicLCColorStateId::from_interner(0));
        let mut currents = (0_u32..4)
            .map(|source_slot| {
                contact_current_with_source_template(
                    RecurrenceNodeKind::Source,
                    0,
                    0,
                    &[source_slot],
                    0,
                )
            })
            .collect::<Vec<_>>();
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), u32::try_from(id).unwrap()))
            .collect::<BTreeMap<_, _>>();
        let contact_orbits = PreparedContactOrbitIndex::new(&transitions).unwrap();
        let fermion_ordering = on_the_fly_fermion_ordering_context(&seed).unwrap();

        build_forward_currents_with_anchor_exclusion(
            &templates,
            &transitions,
            &TransitionReflectionIndex::default(),
            &contact_orbits,
            false,
            &seed,
            &fermion_ordering,
            closure_anchor_slot,
            seed.explicit_coupling_limits(),
            &propagators,
            &mut colors,
            &mut currents,
            &mut current_ids,
        )
        .unwrap();

        currents
    }

    #[allow(clippy::type_complexity)] // Compact assertion fixture returned only inside tests.
    fn production_contact_barrier_case(
        reverse_transition_order: bool,
        include_ordinary_scalar_transition: bool,
    ) -> (Vec<(Vec<u32>, u32, Vec<u32>)>, (usize, usize), bool) {
        let currents = production_contact_barrier_currents(
            reverse_transition_order,
            include_ordinary_scalar_transition,
            None,
        );

        assert!(
            currents[4..].iter().all(|current| {
                let expected = if include_ordinary_scalar_transition
                    && current.stage == 2
                    && current.key.current_state_template_id() == 1
                {
                    6
                } else if current.stage == 1 {
                    1
                } else {
                    3
                };
                current.contributions.len() == expected
            }),
            "certified pair/final destinations retain one owner per physical assignment",
        );
        assert!(
            currents[4..].iter().all(|current| {
                current.contributions.keys().all(|contribution| {
                    contribution.parent_current_ids.iter().all(|parent_id| {
                        let parent = &currents[*parent_id as usize];
                        parent.source_factor.is_some() || !parent.contributions.is_empty()
                    })
                })
            }),
            "a selected contact transition must not reference an empty parent",
        );

        let staged = currents[4..]
            .iter()
            .map(|current| {
                (
                    current.key.support_source_slots().to_vec(),
                    current.stage,
                    contact_transition_ids(current),
                )
            })
            .collect();
        let retained_partial_is_consumed = currents[10..].iter().all(|current| {
            current.contributions.keys().all(|contribution| {
                let mut partial_parents = contribution
                    .parent_current_ids
                    .iter()
                    .copied()
                    .filter(|parent| (4..10).contains(&usize::try_from(*parent).unwrap()));
                let Some(partial_parent) = partial_parents.next() else {
                    return false;
                };
                partial_parents.next().is_none()
                    && currents[partial_parent as usize]
                        .key
                        .current_state_template_id()
                        == 1
                    && contribution
                        .parent_current_ids
                        .iter()
                        .copied()
                        .any(|parent| {
                            parent != partial_parent
                                && currents[parent as usize].key.current_state_template_id() == 0
                        })
            })
        });
        (
            staged,
            pending_contact_counts(&currents),
            retained_partial_is_consumed,
        )
    }

    type OrderedExactForwardSignature = Vec<(
        Vec<u32>,
        RecurrenceNodeKind,
        u32,
        Vec<(u32, [Vec<u32>; 2], ExactComplexRational)>,
    )>;

    fn ordered_exact_forward_signature(
        currents: &[PendingCurrent],
    ) -> OrderedExactForwardSignature {
        currents
            .iter()
            .map(|current| {
                (
                    current.key.support_source_slots().to_vec(),
                    current.key.node_kind(),
                    current.stage,
                    current
                        .contributions
                        .iter()
                        .map(|(pending, factor)| {
                            (
                                pending.key.transition_template_id(),
                                pending.parent_current_ids.map(|parent_id| {
                                    currents[parent_id as usize]
                                        .key
                                        .support_source_slots()
                                        .to_vec()
                                }),
                                *factor,
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    fn commit_contact_plan(
        currents: &mut [PendingCurrent],
        contacts: &BTreeMap<u32, PreparedContactOrbitTransition>,
        stage_current_start: usize,
    ) -> RusticolResult<()> {
        if let Some(plan) = plan_on_the_fly_contact_orbit_owners_with_resolver(
            stage_current_start,
            currents,
            |transition_id| contacts.get(&transition_id),
        )? {
            plan.commit(currents);
        }
        Ok(())
    }

    fn contact_0000_case(final_stage: bool, reverse_insertion: bool) -> (Vec<u32>, (usize, usize)) {
        let (mut currents, steps, parent_ids, destination_id) = if final_stage {
            (
                vec![
                    contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
                    contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
                    contact_current(RecurrenceNodeKind::Source, 10, 2, &[12]),
                    contact_current(RecurrenceNodeKind::Current, 11, 3, &[11, 12]),
                    contact_current(RecurrenceNodeKind::Current, 11, 4, &[10, 12]),
                    contact_current(RecurrenceNodeKind::Current, 11, 5, &[10, 11]),
                    contact_current(RecurrenceNodeKind::Current, 20, 6, &[10, 11, 12]),
                ],
                vec![
                    final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 60, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 70, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 61, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 71, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 62, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 72, [0, 0, 0, 0]),
                ],
                vec![[3, 0], [0, 3], [4, 1], [1, 4], [5, 2], [2, 5]],
                6_u32,
            )
        } else {
            (
                vec![
                    contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
                    contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
                    contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
                ],
                vec![
                    partial_contact_orbit_step_for_test(0, 1, 2, 20, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(1, 0, 2, 30, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(0, 2, 1, 21, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(2, 0, 1, 31, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(0, 3, 1, 22, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(3, 0, 1, 32, [0, 0, 0, 0]),
                ],
                vec![[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
                2_u32,
            )
        };
        let contacts = steps
            .into_iter()
            .enumerate()
            .map(|(transition, step)| {
                let transition = u32::try_from(transition).unwrap();
                let parents = parent_ids[transition as usize];
                (
                    transition,
                    contact_transition(
                        step,
                        parents.map(|parent| {
                            currents[parent as usize].key.current_state_template_id()
                        }),
                        40 + u8::try_from(transition).unwrap(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let order = if reverse_insertion {
            (0_u32..6).rev().collect::<Vec<_>>()
        } else {
            (0_u32..6).collect::<Vec<_>>()
        };
        for transition in order {
            add_contact_contribution(
                &mut currents,
                destination_id,
                transition,
                parent_ids[transition as usize],
            );
        }
        commit_contact_plan(&mut currents, &contacts, destination_id as usize).unwrap();
        (
            contact_transition_ids(&currents[destination_id as usize]),
            pending_contact_counts(&currents),
        )
    }

    fn closure(lineages: Vec<PendingPairingLineage>) -> PendingClosure {
        PendingClosure {
            key: PendingClosureKey {
                closure_template_id: 0,
                quantum_flow_template_id: None,
                parent_current_ids: [0, 1],
                color_witness_term_id: LCColorWitnessTermId::new(0, 0),
            },
            factor: ExactComplexRational::ONE,
            component_coefficients: vec![ExactComplexRational::ONE].into_boxed_slice(),
            pairing_lineages: lineages,
        }
    }

    fn lineage(pair: [u32; 2]) -> PendingPairingLineage {
        PendingPairingLineage {
            completed_pairs: vec![pair],
            unmatched_endpoint: None,
        }
    }

    #[test]
    fn contact_0000_fan_in_keeps_exact_assignment_multiplicity_deterministically() {
        let partial_forward = contact_0000_case(false, false);
        let partial_reverse = contact_0000_case(false, true);
        assert_eq!(partial_forward, (vec![0], (3, 1)));
        assert_eq!(partial_reverse, partial_forward);

        let final_forward = contact_0000_case(true, false);
        let final_reverse = contact_0000_case(true, true);
        assert_eq!(final_reverse, final_forward);
        assert_eq!(final_forward.0.len(), 3);
        assert_eq!(final_forward.1, (7, 3));
    }

    #[test]
    fn forward_sweep_barrier_prunes_complete_contact_fan_in_before_next_stage() {
        let forward = production_contact_barrier_case(false, false);
        let reverse = production_contact_barrier_case(true, false);
        assert_eq!(reverse, forward);
        let (staged, counts, retained_partial_is_consumed) = forward;
        assert_eq!(
            staged
                .iter()
                .map(|(support, stage, _)| (support.clone(), *stage))
                .collect::<Vec<_>>(),
            vec![
                (vec![0, 1], 1),
                (vec![0, 2], 1),
                (vec![0, 3], 1),
                (vec![1, 2], 1),
                (vec![1, 3], 1),
                (vec![2, 3], 1),
                (vec![0, 1, 2], 2),
                (vec![0, 1, 3], 2),
                (vec![0, 2, 3], 2),
                (vec![1, 2, 3], 2),
            ],
        );
        assert!(staged.iter().all(|(_, stage, transition_ids)| {
            transition_ids.len() == if *stage == 1 { 1 } else { 3 }
        }));
        assert_eq!(counts, (14, 18));
        assert!(retained_partial_is_consumed);
    }

    #[test]
    fn anchor_exclusion_preserves_order_exact_factors_and_contact_owners() {
        let baseline = production_contact_barrier_currents(false, false, None);
        let baseline_signature = ordered_exact_forward_signature(&baseline);

        for closure_anchor_slot in 0_u32..4 {
            let pruned =
                production_contact_barrier_currents(false, false, Some(closure_anchor_slot));
            assert_eq!(
                pruned
                    .iter()
                    .filter(|current| {
                        current.key.node_kind() == RecurrenceNodeKind::Source
                            && current.key.support_source_slots() == [closure_anchor_slot]
                    })
                    .count(),
                1,
                "the singleton closure anchor must remain materialized",
            );
            assert!(pruned.iter().all(|current| {
                current.key.node_kind() == RecurrenceNodeKind::Source
                    || !current
                        .key
                        .support_source_slots()
                        .contains(&closure_anchor_slot)
            }));

            let expected = baseline_signature
                .iter()
                .filter(|(support, node_kind, _, _)| {
                    *node_kind == RecurrenceNodeKind::Source
                        || !support.contains(&closure_anchor_slot)
                })
                .cloned()
                .collect::<Vec<_>>();
            assert_eq!(
                ordered_exact_forward_signature(&pruned),
                expected,
                "anchor filtering must preserve surviving current order, exact factors, ordered parent supports, and contact-orbit owners",
            );

            let anchor_id = pruned
                .iter()
                .position(|current| {
                    current.key.node_kind() == RecurrenceNodeKind::Source
                        && current.key.support_source_slots() == [closure_anchor_slot]
                })
                .unwrap() as u32;
            let complement = (0_u32..4)
                .filter(|slot| *slot != closure_anchor_slot)
                .collect::<Vec<_>>();
            let complement_id = pruned
                .iter()
                .position(|current| current.key.support_source_slots() == complement)
                .unwrap() as u32;
            let closure = PendingClosure {
                key: PendingClosureKey {
                    closure_template_id: 0,
                    quantum_flow_template_id: None,
                    parent_current_ids: [anchor_id, complement_id],
                    color_witness_term_id: LCColorWitnessTermId::new(0, 0),
                },
                factor: ExactComplexRational::ONE,
                component_coefficients: vec![ExactComplexRational::ONE].into_boxed_slice(),
                pairing_lineages: vec![PendingPairingLineage {
                    completed_pairs: Vec::new(),
                    unmatched_endpoint: None,
                }],
            };
            let live = live_current_ids(&pruned, &[closure]).unwrap();
            assert!(live.contains(&anchor_id));
            assert!(live.iter().all(|current_id| {
                let current = &pruned[*current_id as usize];
                current.key.node_kind() == RecurrenceNodeKind::Source
                    || !current
                        .key
                        .support_source_slots()
                        .contains(&closure_anchor_slot)
            }));
        }
    }

    #[test]
    fn anchor_exclusion_preserves_structural_zero_when_no_complement_can_be_built() {
        let templates = contact_orbit_test_template(ContactOrbitTestBinding::None)
            .validate()
            .unwrap();
        let propagators = propagator_by_state(&templates).unwrap();
        let seed = scalar_contact_seed(3);
        let query = DecodedLcQueryV1::new(
            &seed,
            vec![0, 1, 2],
            &[0, 0, 0],
            OnTheFlyLcSelectorV1::Singlet,
        )
        .unwrap();
        let transitions = BTreeMap::new();
        let closures = BTreeMap::new();
        let contact_orbits = PreparedContactOrbitIndex::new(&transitions).unwrap();
        let fermion_ordering = on_the_fly_fermion_ordering_context(&seed).unwrap();
        let build = |closure_anchor_slot| {
            let mut colors = DynamicLCColorStateInterner::default();
            let mut currents = (0_u32..3)
                .map(|slot| {
                    contact_current_with_source_template(
                        RecurrenceNodeKind::Source,
                        0,
                        0,
                        &[slot],
                        0,
                    )
                })
                .collect::<Vec<_>>();
            let mut current_ids = currents
                .iter()
                .enumerate()
                .map(|(id, current)| (current.key.clone(), id as u32))
                .collect::<BTreeMap<_, _>>();
            build_forward_currents_with_anchor_exclusion(
                &templates,
                &transitions,
                &TransitionReflectionIndex::default(),
                &contact_orbits,
                false,
                &seed,
                &fermion_ordering,
                closure_anchor_slot,
                seed.explicit_coupling_limits(),
                &propagators,
                &mut colors,
                &mut currents,
                &mut current_ids,
            )
            .unwrap();
            assert!(
                build_selected_closures(
                    &templates,
                    &closures,
                    &seed,
                    &fermion_ordering,
                    &query,
                    false,
                    &colors,
                    &currents,
                )
                .unwrap()
                .is_none()
            );
            currents
        };
        let baseline = build(None);
        let pruned = build(Some(query.closure_anchor_slot));
        assert_eq!(
            ordered_exact_forward_signature(&pruned),
            ordered_exact_forward_signature(&baseline),
        );
    }

    #[test]
    fn forward_sweep_accepts_contact_partial_in_its_certified_internal_parent_domain() {
        let forward = production_contact_barrier_case(false, true);
        let reverse = production_contact_barrier_case(true, true);
        assert_eq!(reverse, forward);
        let (staged, _, _) = forward;

        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 1)
                .any(|(_, _, transition_ids)| {
                    transition_ids
                        .iter()
                        .any(|transition_id| *transition_id < 6)
                }),
            "one-plus-one source parents must retain a certified contact partial",
        );
        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 1)
                .any(|(_, _, transition_ids)| transition_ids.as_slice() == [12]),
            "the ordinary scalar transition must construct a composite scalar parent",
        );
        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 2)
                .flat_map(|(_, _, transition_ids)| transition_ids)
                .any(|transition_id| *transition_id < 6),
            "a certified contact partial must accept a matching internal scalar parent",
        );
        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 2)
                .flat_map(|(_, _, transition_ids)| transition_ids)
                .any(|transition_id| (6..12).contains(transition_id)),
            "a one-plus-two certified contact final must remain available",
        );
    }

    #[test]
    fn contact_0012_partial_and_final_fan_in_each_keep_one_owner() {
        let mut partial_currents = vec![
            contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let partial_contacts = BTreeMap::from([
            (
                10,
                contact_transition(
                    partial_contact_orbit_step_for_test(0, 1, 2, 50, [0, 0, 1, 2]),
                    [10, 10],
                    52,
                ),
            ),
            (
                11,
                contact_transition(
                    partial_contact_orbit_step_for_test(1, 0, 2, 51, [0, 0, 1, 2]),
                    [10, 10],
                    53,
                ),
            ),
        ]);
        add_contact_contribution(&mut partial_currents, 2, 11, [0, 1]);
        add_contact_contribution(&mut partial_currents, 2, 10, [0, 1]);
        commit_contact_plan(&mut partial_currents, &partial_contacts, 2).unwrap();
        assert_eq!(contact_transition_ids(&partial_currents[2]), [10]);
        assert_eq!(pending_contact_counts(&partial_currents), (3, 1));

        let mut final_currents = vec![
            contact_current(RecurrenceNodeKind::Current, 20, 0, &[10, 11]),
            contact_current(RecurrenceNodeKind::Source, 30, 1, &[12]),
            contact_current(RecurrenceNodeKind::Current, 40, 2, &[10, 11, 12]),
        ];
        let final_contacts = BTreeMap::from([
            (
                20,
                contact_transition(
                    final_contact_orbit_step_for_test(&[0, 1], &[2], 3, 54, [0, 0, 1, 2]),
                    [20, 30],
                    56,
                ),
            ),
            (
                21,
                contact_transition(
                    final_contact_orbit_step_for_test(&[2], &[0, 1], 3, 55, [0, 0, 1, 2]),
                    [30, 20],
                    57,
                ),
            ),
        ]);
        add_contact_contribution(&mut final_currents, 2, 20, [0, 1]);
        add_contact_contribution(&mut final_currents, 2, 21, [1, 0]);
        commit_contact_plan(&mut final_currents, &final_contacts, 2).unwrap();
        assert_eq!(contact_transition_ids(&final_currents[2]), [21]);
        assert_eq!(pending_contact_counts(&final_currents), (3, 1));
    }

    #[test]
    fn contact_owner_planning_rolls_back_conflict_decode_and_allocation_errors() {
        let mut currents = vec![
            contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let duplicate_step = partial_contact_orbit_step_for_test(0, 1, 2, 60, [0, 0, 0, 0]);
        let contacts = BTreeMap::from([
            (30, contact_transition(duplicate_step.clone(), [10, 10], 61)),
            (31, contact_transition(duplicate_step, [10, 10], 61)),
        ]);
        add_contact_contribution(&mut currents, 2, 30, [0, 1]);
        add_contact_contribution(&mut currents, 2, 31, [0, 1]);
        let before = currents[2].contributions.clone();
        assert!(
            plan_on_the_fly_contact_orbit_owners_with_resolver(2, &currents, |transition_id| {
                contacts.get(&transition_id)
            })
            .unwrap_err()
            .to_string()
            .contains("conflicting exact rank")
        );
        assert_eq!(currents[2].contributions, before);

        let mut malformed = currents;
        let (pending, factor) = malformed[2]
            .contributions
            .first_key_value()
            .map(|(pending, factor)| (pending.clone(), *factor))
            .unwrap();
        malformed[2].contributions.remove(&pending);
        malformed[2].contributions.insert(
            PendingContributionKey {
                parent_current_ids: [u32::MAX, pending.parent_current_ids[1]],
                key: pending.key,
            },
            factor,
        );
        let malformed_before = malformed[2].contributions.clone();
        assert!(
            plan_on_the_fly_contact_orbit_owners_with_resolver(2, &malformed, |transition_id| {
                contacts.get(&transition_id)
            },)
            .unwrap_err()
            .to_string()
            .contains("left parent is absent")
        );
        assert_eq!(malformed[2].contributions, malformed_before);

        let mut candidate_reservation = Vec::<OnTheFlyContactContributionToken>::new();
        assert!(
            reserve_on_the_fly_contact_candidates(&mut candidate_reservation, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("candidate allocation failed")
        );
        assert!(candidate_reservation.is_empty());
        let mut location_reservation = Vec::<u8>::new();
        assert!(
            reserve_on_the_fly_contact_locations(&mut location_reservation, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("transition-index allocation failed")
        );
        assert!(location_reservation.is_empty());
    }

    #[test]
    fn contact_index_and_uncertified_controls_use_production_lookup_and_fast_path() {
        let none = contact_orbit_test_template(ContactOrbitTestBinding::None)
            .validate()
            .unwrap();
        let none_catalog = TemplateCatalog::new(none.input()).unwrap();
        let none_transitions = prepared_transitions(&none, &none_catalog).unwrap();
        assert!(
            PreparedContactOrbitIndex::new(&none_transitions)
                .unwrap()
                .is_empty()
        );

        let one = contact_orbit_test_template(ContactOrbitTestBinding::One)
            .validate()
            .unwrap();
        let one_catalog = TemplateCatalog::new(one.input()).unwrap();
        let mut one_transitions = prepared_transitions(&one, &one_catalog).unwrap();
        let mut ordinary = one_transitions.values().flatten().next().unwrap().clone();
        ordinary.row.id = u32::MAX - 1;
        ordinary.contact_orbit = None;
        assert!(
            one_transitions
                .values()
                .flatten()
                .all(|prepared| prepared.row.id != ordinary.row.id)
        );
        one_transitions.values_mut().next().unwrap().push(ordinary);
        let index = PreparedContactOrbitIndex::new(&one_transitions).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
        for prepared in one_transitions.values().flatten() {
            assert_eq!(
                index.get(&one_transitions, prepared.row.id),
                prepared.contact_orbit.as_ref(),
            );
        }
        assert!(index.get(&one_transitions, u32::MAX).is_none());

        let mut currents = vec![
            contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            contact_current(RecurrenceNodeKind::Source, 11, 1, &[11]),
            contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        // Distinct uncertified rows represent ordinary V3, vector, fermion,
        // and QCD controls. No model-level class is inspected by this path.
        for transition in 70..74 {
            add_contact_contribution(&mut currents, 2, transition, [0, 1]);
        }
        let before = currents[2].contributions.clone();
        assert!(
            plan_on_the_fly_contact_orbit_owners(
                2,
                &BTreeMap::new(),
                &PreparedContactOrbitIndex::default(),
                &currents
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(currents[2].contributions, before);
        assert_eq!(pending_contact_counts(&currents), (3, 4));
    }

    #[test]
    fn pairing_lineage_capacity_is_checked_without_an_arbitrary_limit() {
        assert_eq!(
            checked_pairing_capacity(1 << 20, 1 << 20, 1).unwrap(),
            2_097_153
        );
        assert!(checked_pairing_capacity(usize::MAX, 1, 0).is_err());
    }

    #[test]
    fn large_pairing_lineage_clone_is_fallible_and_exact() {
        let pair_count = 16_384usize;
        let mut completed_pairs = Vec::new();
        completed_pairs.try_reserve_exact(pair_count).unwrap();
        for index in 0..pair_count {
            completed_pairs.push([index as u32, (index + pair_count) as u32]);
        }
        let lineage = PendingPairingLineage {
            completed_pairs,
            unmatched_endpoint: None,
        };
        let copied = try_clone_pairing_lineage(&lineage).unwrap();
        assert_eq!(copied, lineage);
    }

    #[test]
    fn projected_pairing_owners_are_exactly_one_per_closure_not_global() {
        assert!(
            projected_pairing_lineages(&[closure(vec![lineage([0, 1]), lineage([0, 3])])]).is_err()
        );
        let closures = [
            closure(vec![lineage([0, 1])]),
            closure(vec![lineage([0, 3])]),
        ];
        let owners = projected_pairing_lineages(&closures).unwrap();
        assert_eq!(owners, [&lineage([0, 1]), &lineage([0, 3])]);
        assert!(projected_pairing_lineages(&[]).is_err());
    }

    #[test]
    fn multi_realized_closure_keeps_one_semantic_owner_without_rescaling() {
        let mut pending = closure(vec![
            lineage([4, 7]),
            lineage([0, 3]),
            lineage([2, 5]),
            lineage([0, 3]),
        ]);
        pending.factor = ExactComplexRational::ONE.checked_neg().unwrap();
        let expected_factor = pending.factor;

        retain_canonical_pairing_lineage(&mut pending.pairing_lineages).unwrap();

        assert_eq!(pending.pairing_lineages, vec![lineage([0, 3])]);
        assert_eq!(pending.factor, expected_factor);
    }
}
