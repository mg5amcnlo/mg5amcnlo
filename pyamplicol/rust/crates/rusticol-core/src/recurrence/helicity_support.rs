// SPDX-License-Identifier: 0BSD

//! Exact resolved-helicity support for one physical all-flow recurrence.
//!
//! All-flow construction deliberately erases source-helicity ancestry from
//! current identity.  That makes the semantic graph compact, but executing
//! every retained row for every resolved helicity loses the principal warm
//! benefit of the concrete on-the-fly schedules.  This module recovers the
//! exact row support without rebuilding one graph per helicity:
//!
//! 1. enumerate the all-flow source-dispatch Cartesian product;
//! 2. propagate availability forward, admitting a transition/closure only
//!    when every dynamic source parent has the selected crossed spin required
//!    by its authenticated quantum-flow witness; and
//! 3. propagate demand backwards from physically materialized closure rows.
//!
//! The result is semantic and independent of DirectPlan row ordering.  Direct
//! lowering maps these masks through its deterministic row reordering and
//! interns the final sidecar domains once.

use std::collections::BTreeMap;

use super::template::{MISSING_U32, ValidatedRecurrenceTemplateInput};
use super::{
    CurrentSourceBinding, RecurrenceCurrent, RecurrenceNodeKind, RecurrenceProgram,
    RecurrenceStrategy,
};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!(
        "recurrence all-flow helicity support: {}",
        message.into()
    ))
}

/// One temporary fixed-width mask over the auxiliary all-flow plan's own
/// canonical resolved-helicity axis.
///
/// These masks intentionally retain equal widths while semantic projection is
/// in progress.  The direct helicity-dispatch lowering is the sole authority
/// which compacts and interns the persisted domains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllFlowHelicityMask {
    words: Box<[u64]>,
}

impl AllFlowHelicityMask {
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn contains(&self, resolved_helicity_id: u32) -> bool {
        let word = resolved_helicity_id as usize / u64::BITS as usize;
        let bit = resolved_helicity_id % u64::BITS;
        self.words
            .get(word)
            .is_some_and(|value| value & (1_u64 << bit) != 0)
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }
}

/// Semantic support masks for one physical [`RecurrenceStrategy::AllFlowUnion`]
/// program.
///
/// Table indices are the corresponding dense semantic IDs.  `current_demand`
/// is the final live-current support after backward projection; for source
/// currents it is also the exact source-row support.  Contribution masks have
/// already been intersected with result demand, and finalization masks equal
/// their owning current's final demand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllFlowHelicitySupportProjection {
    resolved_helicity_count: u32,
    current_availability: Box<[AllFlowHelicityMask]>,
    current_demand: Box<[AllFlowHelicityMask]>,
    contribution_support: Box<[AllFlowHelicityMask]>,
    finalization_support: Box<[AllFlowHelicityMask]>,
    closure_support: Box<[AllFlowHelicityMask]>,
}

impl AllFlowHelicitySupportProjection {
    pub const fn resolved_helicity_count(&self) -> u32 {
        self.resolved_helicity_count
    }

    pub fn current_availability(&self) -> &[AllFlowHelicityMask] {
        &self.current_availability
    }

    pub fn current_demand(&self) -> &[AllFlowHelicityMask] {
        &self.current_demand
    }

    pub fn contribution_support(&self) -> &[AllFlowHelicityMask] {
        &self.contribution_support
    }

    pub fn finalization_support(&self) -> &[AllFlowHelicityMask] {
        &self.finalization_support
    }

    pub fn closure_support(&self) -> &[AllFlowHelicityMask] {
        &self.closure_support
    }
}

#[derive(Clone, Copy, Debug)]
struct SourceChoice {
    source_state_index: u32,
    crossed_spin_state_class: i32,
}

#[derive(Debug)]
struct SourceAxis {
    current_id: u32,
    choices: Vec<SourceChoice>,
    spin_domains: BTreeMap<i32, Vec<u64>>,
}

fn zero_mask(word_count: usize) -> Vec<u64> {
    vec![0; word_count]
}

fn full_mask(resolved_helicity_count: u32, word_count: usize) -> Vec<u64> {
    let mut words = vec![u64::MAX; word_count];
    if let Some(last) = words.last_mut()
        && !resolved_helicity_count.is_multiple_of(u64::BITS)
    {
        *last = (1_u64 << (resolved_helicity_count % u64::BITS)) - 1;
    }
    words
}

fn mask_and_assign(destination: &mut [u64], source: &[u64]) {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination &= *source;
    }
}

fn mask_or_assign(destination: &mut [u64], source: &[u64]) {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination |= *source;
    }
}

fn mask_intersection(left: &[u64], right: &[u64]) -> Vec<u64> {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left & right)
        .collect()
}

fn boxed_masks(rows: Vec<Vec<u64>>) -> Box<[AllFlowHelicityMask]> {
    rows.into_iter()
        .map(|words| AllFlowHelicityMask {
            words: words.into_boxed_slice(),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn only_source_slot(current: &RecurrenceCurrent) -> RusticolResult<u32> {
    match current.key().support_source_slots() {
        [source_slot] => Ok(*source_slot),
        _ => Err(invalid(format!(
            "source current {} does not own exactly one source slot",
            current.id()
        ))),
    }
}

fn source_axes(program: &RecurrenceProgram) -> RusticolResult<(Vec<SourceAxis>, u32, usize)> {
    let source_count = program
        .currents()
        .iter()
        .filter(|current| current.is_source())
        .count();
    if source_count == 0 {
        return Err(invalid("all-flow program has no source currents"));
    }
    let mut axes = (0..source_count)
        .map(|_| None)
        .collect::<Vec<Option<SourceAxis>>>();
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let source_slot = only_source_slot(current)?;
        let axis = axes.get_mut(source_slot as usize).ok_or_else(|| {
            invalid(format!(
                "source current {} uses non-dense source slot {source_slot}",
                current.id()
            ))
        })?;
        if axis.is_some() {
            return Err(invalid(format!(
                "source slot {source_slot} has more than one all-flow dispatch current"
            )));
        }
        let CurrentSourceBinding::RuntimeDispatch {
            variant_bindings, ..
        } = current.key().source_binding()
        else {
            return Err(invalid(format!(
                "source current {} has no runtime dispatch binding",
                current.id()
            )));
        };
        if variant_bindings.is_empty() {
            return Err(invalid(format!(
                "source current {} has no runtime source choices",
                current.id()
            )));
        }
        let mut choices = variant_bindings
            .iter()
            .map(|binding| SourceChoice {
                source_state_index: binding.source_state_index(),
                crossed_spin_state_class: binding.crossed_spin_state_class(),
            })
            .collect::<Vec<_>>();
        if choices.iter().any(|choice| {
            choice.crossed_spin_state_class == super::DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS
        }) {
            return Err(invalid(format!(
                "source slot {source_slot} has a non-concrete crossed-spin choice"
            )));
        }
        choices.sort_by_key(|choice| choice.source_state_index);
        if choices
            .windows(2)
            .any(|pair| pair[0].source_state_index == pair[1].source_state_index)
        {
            return Err(invalid(format!(
                "source slot {source_slot} repeats a source-state choice"
            )));
        }
        *axis = Some(SourceAxis {
            current_id: current.id(),
            choices,
            spin_domains: BTreeMap::new(),
        });
    }
    let mut axes = axes
        .into_iter()
        .enumerate()
        .map(|(source_slot, axis)| {
            axis.ok_or_else(|| invalid(format!("source slot {source_slot} is absent")))
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let cartesian_count = axes.iter().try_fold(1_u64, |count, axis| {
        count
            .checked_mul(
                u64::try_from(axis.choices.len())
                    .map_err(|_| invalid("runtime source-choice count exceeds u64"))?,
            )
            .ok_or_else(|| invalid("resolved-helicity Cartesian product exceeds u64"))
    })?;
    if cartesian_count != program.retained_helicity_count() {
        return Err(invalid(format!(
            "runtime source choices span {cartesian_count} helicities, expected {}",
            program.retained_helicity_count()
        )));
    }
    let resolved_helicity_count = u32::try_from(cartesian_count)
        .map_err(|_| invalid("resolved-helicity count exceeds the u32 sidecar domain"))?;
    let word_count = usize::try_from(resolved_helicity_count.div_ceil(u64::BITS))
        .map_err(|_| invalid("resolved-helicity mask width exceeds usize"))?;

    for axis in &mut axes {
        for choice in &axis.choices {
            axis.spin_domains
                .entry(choice.crossed_spin_state_class)
                .or_insert_with(|| zero_mask(word_count));
        }
    }
    let mut indices = vec![0usize; axes.len()];
    for helicity_id in 0..resolved_helicity_count {
        for (axis, choice_index) in axes.iter_mut().zip(&indices) {
            let choice = axis.choices[*choice_index];
            let words = axis
                .spin_domains
                .get_mut(&choice.crossed_spin_state_class)
                .expect("source spin domain was reserved");
            words[helicity_id as usize / u64::BITS as usize] |= 1_u64 << (helicity_id % u64::BITS);
        }
        for source_slot in (0..indices.len()).rev() {
            indices[source_slot] += 1;
            if indices[source_slot] < axes[source_slot].choices.len() {
                break;
            }
            indices[source_slot] = 0;
        }
    }
    Ok((axes, resolved_helicity_count, word_count))
}

fn template_u32_sequence<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let input = templates.input();
    let range = input
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .filter(|range| range.id == sequence_id)
        .ok_or_else(|| invalid(format!("{label} sequence {sequence_id} is absent")))?;
    let range = range
        .range
        .as_usize_range(input.u32_sequence_values.len(), label)?;
    Ok(&input.u32_sequence_values[range])
}

fn template_i32_sequence<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [i32]> {
    let input = templates.input();
    let range = input
        .i32_sequence_ranges
        .get(sequence_id as usize)
        .filter(|range| range.id == sequence_id)
        .ok_or_else(|| invalid(format!("{label} sequence {sequence_id} is absent")))?;
    let range = range
        .range
        .as_usize_range(input.i32_sequence_values.len(), label)?;
    Ok(&input.i32_sequence_values[range])
}

fn quantum_flow_contract(
    templates: &ValidatedRecurrenceTemplateInput,
    quantum_flow_id: u32,
) -> RusticolResult<(&[u32], &[i32])> {
    let flow = templates
        .input()
        .quantum_flows
        .get(quantum_flow_id as usize)
        .filter(|flow| flow.id == quantum_flow_id)
        .ok_or_else(|| invalid(format!("quantum flow {quantum_flow_id} is absent")))?;
    let states = template_u32_sequence(
        templates,
        flow.input_state_sequence_id,
        "quantum-flow input state",
    )?;
    let spins = template_i32_sequence(
        templates,
        flow.input_spin_sequence_id,
        "quantum-flow input spin",
    )?;
    if states.len() != 2 || spins.len() != 2 {
        return Err(invalid(format!(
            "quantum flow {quantum_flow_id} is not binary"
        )));
    }
    Ok((states, spins))
}

fn construction_parent_orders_for_evaluator_tuple(
    evaluator_parents: [u32; 2],
    parent_states: [u32; 2],
    transition_input_states: [u32; 2],
    canonical_input_order: [u32; 2],
    exchange_is_proven: bool,
) -> Vec<[u32; 2]> {
    let mut orders = Vec::new();
    for construction in [
        evaluator_parents,
        [evaluator_parents[1], evaluator_parents[0]],
    ] {
        let construction_states = construction.map(|parent_id| {
            if parent_id == evaluator_parents[0] {
                parent_states[0]
            } else {
                parent_states[1]
            }
        });
        if construction_states != transition_input_states {
            continue;
        }
        let mut evaluator = [
            construction[canonical_input_order[0] as usize],
            construction[canonical_input_order[1] as usize],
        ];
        if exchange_is_proven && evaluator[1] < evaluator[0] {
            evaluator.swap(0, 1);
        }
        if evaluator == evaluator_parents && !orders.contains(&construction) {
            orders.push(construction);
        }
    }
    orders
}

fn contribution_construction_parent_orders(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    contribution_id: u32,
) -> RusticolResult<Vec<[u32; 2]>> {
    let contribution = program
        .contributions()
        .get(contribution_id as usize)
        .filter(|row| row.id() == contribution_id)
        .ok_or_else(|| invalid(format!("contribution {contribution_id} is absent")))?;
    let [first, second] = contribution.parent_current_ids() else {
        return Err(invalid(format!(
            "contribution {contribution_id} is not binary"
        )));
    };
    let transition_id = contribution.key().transition_template_id();
    let transition = templates
        .input()
        .transitions
        .get(transition_id as usize)
        .filter(|transition| transition.id == transition_id)
        .ok_or_else(|| invalid(format!("transition {transition_id} is absent")))?;
    if transition.quantum_flow_template_id != contribution.key().quantum_flow_witness_id() {
        return Err(invalid(format!(
            "contribution {contribution_id} quantum-flow witness disagrees with transition {transition_id}"
        )));
    }
    let input_states = template_u32_sequence(
        templates,
        transition.input_state_sequence_id,
        "transition input state",
    )?;
    let [left_state, right_state] = input_states else {
        return Err(invalid(format!("transition {transition_id} is not binary")));
    };
    let canonical_input_order = template_u32_sequence(
        templates,
        transition.canonical_input_order_sequence_id,
        "transition canonical input order",
    )?;
    let canonical_input_order = match canonical_input_order {
        [0, 1] => [0, 1],
        [1, 0] => [1, 0],
        _ => {
            return Err(invalid(format!(
                "transition {transition_id} canonical input order is not binary"
            )));
        }
    };
    let first_state = program.currents()[*first as usize]
        .key()
        .current_state_template_id();
    let second_state = program.currents()[*second as usize]
        .key()
        .current_state_template_id();
    let orders = construction_parent_orders_for_evaluator_tuple(
        [*first, *second],
        [first_state, second_state],
        [*left_state, *right_state],
        canonical_input_order,
        transition.input_exchange_factor_id != MISSING_U32,
    );
    if orders.is_empty() {
        return Err(invalid(format!(
            "contribution {contribution_id} parents cannot reconstruct transition {transition_id} input order"
        )));
    }
    Ok(orders)
}

fn flow_gate(
    program: &RecurrenceProgram,
    source_axis_by_current: &[Option<usize>],
    source_axes: &[SourceAxis],
    parent_ids: [u32; 2],
    required_states: &[u32],
    required_spins: &[i32],
    word_count: usize,
) -> RusticolResult<Vec<u64>> {
    let mut gate = vec![u64::MAX; word_count];
    for (ordinal, parent_id) in parent_ids.into_iter().enumerate() {
        let parent = program
            .currents()
            .get(parent_id as usize)
            .ok_or_else(|| invalid(format!("flow parent current {parent_id} is absent")))?;
        if parent.key().current_state_template_id() != required_states[ordinal] {
            gate.fill(0);
            return Ok(gate);
        }
        if parent.key().node_kind() == RecurrenceNodeKind::Source
            && parent.key().spin_state_class() == super::DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS
        {
            let axis_id = source_axis_by_current
                .get(parent_id as usize)
                .copied()
                .flatten()
                .ok_or_else(|| invalid(format!("source current {parent_id} has no source axis")))?;
            let axis = &source_axes[axis_id];
            if let Some(spin_domain) = axis.spin_domains.get(&required_spins[ordinal]) {
                mask_and_assign(&mut gate, spin_domain);
            } else {
                gate.fill(0);
            }
        } else if parent.key().spin_state_class() != required_spins[ordinal] {
            gate.fill(0);
            return Ok(gate);
        }
    }
    Ok(gate)
}

fn parent_availability(
    parent_ids: &[u32],
    current_availability: &[Vec<u64>],
    word_count: usize,
) -> RusticolResult<Vec<u64>> {
    let mut availability = vec![u64::MAX; word_count];
    for parent_id in parent_ids {
        let parent = current_availability
            .get(*parent_id as usize)
            .ok_or_else(|| invalid(format!("parent current {parent_id} is absent")))?;
        mask_and_assign(&mut availability, parent);
    }
    Ok(availability)
}

fn closure_construction_parent_orders(
    program: &RecurrenceProgram,
    closure_id: u32,
) -> RusticolResult<Vec<[u32; 2]>> {
    let group = program
        .closure_proofs()
        .groups()
        .iter()
        .find(|group| group.emitted_runtime_closure_term_id() == Some(closure_id))
        .ok_or_else(|| invalid(format!("closure term {closure_id} has no proof group")))?;
    let range = group.contribution_range().as_usize_range(
        program.closure_proofs().contributions().len(),
        "closure support proof contribution",
    )?;
    let mut orders = Vec::new();
    for contribution in &program.closure_proofs().contributions()[range] {
        let [left, right] = contribution.construction_parent_runtime_ids() else {
            return Err(invalid(format!(
                "closure term {closure_id} has a non-binary construction proof"
            )));
        };
        let (Some(left), Some(right)) = (*left, *right) else {
            return Err(invalid(format!(
                "closure term {closure_id} proof lacks runtime parents"
            )));
        };
        let order = [left, right];
        if !orders.contains(&order) {
            orders.push(order);
        }
    }
    if orders.is_empty() {
        return Err(invalid(format!(
            "closure term {closure_id} proof group is empty"
        )));
    }
    Ok(orders)
}

/// Compute exact per-resolved-helicity support for a physical all-flow graph.
///
/// This function is deliberately pure and lowering-order independent.  It
/// performs no topology replay, no numerical evaluation, and no per-sector
/// recurrence construction.
pub fn project_all_flow_helicity_support(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<AllFlowHelicitySupportProjection> {
    if program.strategy() != RecurrenceStrategy::AllFlowUnion {
        return Err(invalid("projection requires an all-flow-union program"));
    }
    if !program.replay_targets().is_empty() || !program.resolved_helicities().is_empty() {
        return Err(invalid(
            "physical all-flow support must not use topology replay or fixed-helicity destinations",
        ));
    }
    if program
        .amplitude_destinations()
        .iter()
        .any(|destination| destination.target_helicity_id().is_some())
    {
        return Err(invalid(
            "physical all-flow amplitude destinations must remain helicity-parametric",
        ));
    }

    let (source_axes, resolved_helicity_count, word_count) = source_axes(program)?;
    let all_helicities = full_mask(resolved_helicity_count, word_count);
    let mut source_axis_by_current = vec![None; program.currents().len()];
    for (axis_id, axis) in source_axes.iter().enumerate() {
        source_axis_by_current[axis.current_id as usize] = Some(axis_id);
    }

    let mut current_availability = vec![zero_mask(word_count); program.currents().len()];
    let mut contribution_support = vec![zero_mask(word_count); program.contributions().len()];
    for current in program.currents() {
        if current.is_source() {
            current_availability[current.id() as usize].clone_from(&all_helicities);
            continue;
        }
        let range = current.contribution_range().as_usize_range(
            program.contributions().len(),
            "current helicity-support contribution",
        )?;
        let mut result_availability = zero_mask(word_count);
        for contribution in &program.contributions()[range] {
            let construction_parent_orders =
                contribution_construction_parent_orders(program, templates, contribution.id())?;
            let (required_states, required_spins) =
                quantum_flow_contract(templates, contribution.key().quantum_flow_witness_id())?;
            let mut availability = parent_availability(
                contribution.parent_current_ids(),
                &current_availability,
                word_count,
            )?;
            let mut aggregate_gate = zero_mask(word_count);
            for construction_parents in construction_parent_orders {
                let gate = flow_gate(
                    program,
                    &source_axis_by_current,
                    &source_axes,
                    construction_parents,
                    required_states,
                    required_spins,
                    word_count,
                )?;
                mask_or_assign(&mut aggregate_gate, &gate);
            }
            mask_and_assign(&mut availability, &aggregate_gate);
            contribution_support[contribution.id() as usize].clone_from(&availability);
            mask_or_assign(&mut result_availability, &availability);
        }
        current_availability[current.id() as usize] = result_availability;
    }

    let mut closure_support = vec![zero_mask(word_count); program.closure_terms().len()];
    let mut current_demand = vec![zero_mask(word_count); program.currents().len()];
    for closure in program.closure_terms() {
        let mut support = parent_availability(
            closure.parent_current_ids(),
            &current_availability,
            word_count,
        )?;
        if let Some(flow_id) = closure.quantum_flow_template_id() {
            let (required_states, required_spins) = quantum_flow_contract(templates, flow_id)?;
            let mut aggregate_gate = zero_mask(word_count);
            for construction_parents in closure_construction_parent_orders(program, closure.id())? {
                let gate = flow_gate(
                    program,
                    &source_axis_by_current,
                    &source_axes,
                    construction_parents,
                    required_states,
                    required_spins,
                    word_count,
                )?;
                mask_or_assign(&mut aggregate_gate, &gate);
            }
            mask_and_assign(&mut support, &aggregate_gate);
        }
        closure_support[closure.id() as usize].clone_from(&support);
        for parent_id in closure.parent_current_ids() {
            mask_or_assign(&mut current_demand[*parent_id as usize], &support);
        }
    }

    // Current IDs are topological, so reverse traversal always propagates
    // demand to parents which have not yet been finalized.
    for current in program.currents().iter().rev() {
        let current_id = current.id() as usize;
        let live = mask_intersection(
            &current_demand[current_id],
            &current_availability[current_id],
        );
        current_demand[current_id] = live.clone();
        if current.is_source() {
            continue;
        }
        let range = current.contribution_range().as_usize_range(
            program.contributions().len(),
            "backward helicity-support contribution",
        )?;
        for contribution in &program.contributions()[range] {
            let contribution_id = contribution.id() as usize;
            let live_contribution =
                mask_intersection(&contribution_support[contribution_id], &live);
            contribution_support[contribution_id] = live_contribution.clone();
            for parent_id in contribution.parent_current_ids() {
                mask_or_assign(&mut current_demand[*parent_id as usize], &live_contribution);
            }
        }
    }

    let mut finalization_support = vec![zero_mask(word_count); program.finalizations().len()];
    for finalization in program.finalizations() {
        finalization_support[finalization.id() as usize]
            .clone_from(&current_demand[finalization.current_id() as usize]);
    }

    Ok(AllFlowHelicitySupportProjection {
        resolved_helicity_count,
        current_availability: boxed_masks(current_availability),
        current_demand: boxed_masks(current_demand),
        contribution_support: boxed_masks(contribution_support),
        finalization_support: boxed_masks(finalization_support),
        closure_support: boxed_masks(closure_support),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::layout::RuntimeSourceVariantBinding;
    use crate::recurrence::{
        CanonicalMomentumLinearForm, CheckedTableRange, ContributionKey, CurrentCoreKey,
        CurrentHelicityIdentity, DynamicLCColorState, DynamicLCColorStateId, ExactComplexRational,
        LCColorWitnessTermId, MomentumTerm, RecurrenceAmplitudeDestination, RecurrenceClosureTerm,
        RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization, SemanticDigest,
    };

    fn digest(seed: u8) -> SemanticDigest {
        SemanticDigest::new([seed; 32]).unwrap()
    }

    fn momentum(slots: &[u32]) -> CanonicalMomentumLinearForm {
        CanonicalMomentumLinearForm::new(
            slots
                .iter()
                .map(|source_slot| MomentumTerm {
                    source_slot: *source_slot,
                    coefficient: 1,
                })
                .collect(),
        )
        .unwrap()
    }

    fn source_key_with_spins(
        slot: u32,
        source_spin_state_class: i32,
        crossed_spin_state_classes: [i32; 2],
    ) -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(3),
            RecurrenceNodeKind::Source,
            0,
            DynamicLCColorStateId::from_interner(slot),
            vec![slot],
            momentum(&[slot]),
            CurrentHelicityIdentity::all_flow_union(source_spin_state_class),
            vec![1],
            0,
            vec![],
            CurrentSourceBinding::runtime_dispatch_with_variants(
                0,
                vec![
                    RuntimeSourceVariantBinding::new(
                        0,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        crossed_spin_state_classes[0],
                        ExactComplexRational::ONE,
                    )
                    .unwrap(),
                    RuntimeSourceVariantBinding::new(
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        crossed_spin_state_classes[1],
                        ExactComplexRational::ONE,
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
            None,
        )
        .unwrap()
    }

    fn source_key(slot: u32) -> CurrentCoreKey {
        source_key_with_spins(
            slot,
            super::super::DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS,
            [0, 1],
        )
    }

    fn internal_key(id: u32, slots: &[u32]) -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(3),
            RecurrenceNodeKind::Current,
            0,
            DynamicLCColorStateId::from_interner(id),
            slots.to_vec(),
            momentum(slots),
            CurrentHelicityIdentity::all_flow_union(0),
            vec![1],
            0,
            vec![],
            CurrentSourceBinding::None,
            Some(0),
        )
        .unwrap()
    }

    fn contribution(
        id: u32,
        result_id: u32,
        parents: [u32; 2],
        currents: &[RecurrenceCurrent],
    ) -> RecurrenceContribution {
        RecurrenceContribution::new(
            id,
            result_id,
            parents.to_vec(),
            ContributionKey::new(
                0,
                parents.to_vec(),
                vec![0, 0],
                parents
                    .iter()
                    .map(|parent| currents[*parent as usize].key().momentum().clone())
                    .collect(),
                0,
                0,
                LCColorWitnessTermId::new(0, 0),
                digest(6),
                0,
            )
            .unwrap(),
            ExactComplexRational::ONE,
        )
        .unwrap()
    }

    fn physical_union_program_with_source_keys(
        source_keys: [CurrentCoreKey; 3],
    ) -> RecurrenceProgram {
        let sources = source_keys
            .into_iter()
            .enumerate()
            .map(|(id, key)| {
                RecurrenceCurrent::new(id as u32, key, None, CheckedTableRange::new(0, 0), None)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut currents = sources;
        currents.push(
            RecurrenceCurrent::new(
                3,
                internal_key(3, &[0, 1]),
                None,
                CheckedTableRange::new(0, 1),
                Some(0),
            )
            .unwrap(),
        );
        currents.push(
            RecurrenceCurrent::new(
                4,
                internal_key(4, &[0, 2]),
                None,
                CheckedTableRange::new(1, 1),
                Some(1),
            )
            .unwrap(),
        );
        let contributions = vec![
            contribution(0, 3, [0, 1], &currents),
            contribution(1, 4, [0, 2], &currents),
        ];
        RecurrenceProgram::new(
            RecurrenceStrategy::AllFlowUnion,
            1,
            8,
            (0..5)
                .map(|id| DynamicLCColorState::new(id, None, vec![]).unwrap())
                .collect(),
            currents,
            contributions,
            vec![
                RecurrenceFinalization::new(0, 3, Some(0), ExactComplexRational::ONE).unwrap(),
                RecurrenceFinalization::new(1, 4, Some(0), ExactComplexRational::ONE).unwrap(),
            ],
            vec![],
            vec![],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, None, CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, Some(0), vec![3, 2], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    fn physical_union_program() -> RecurrenceProgram {
        physical_union_program_with_source_keys([source_key(0), source_key(1), source_key(2)])
    }

    fn bits(mask: &AllFlowHelicityMask) -> Vec<u32> {
        (0..8).filter(|id| mask.contains(*id)).collect()
    }

    #[test]
    fn forward_availability_and_backward_demand_prune_exact_helicity_rows() {
        let templates = super::super::validated_template_fixture();
        let projection =
            project_all_flow_helicity_support(&physical_union_program(), &templates).unwrap();

        assert_eq!(projection.resolved_helicity_count(), 8);
        assert_eq!(bits(&projection.current_availability()[3]), [0, 1]);
        assert_eq!(bits(&projection.current_availability()[4]), [0, 2]);
        assert_eq!(bits(&projection.closure_support()[0]), [0]);

        assert_eq!(bits(&projection.current_demand()[0]), [0]);
        assert_eq!(bits(&projection.current_demand()[1]), [0]);
        assert_eq!(bits(&projection.current_demand()[2]), [0]);
        assert_eq!(bits(&projection.current_demand()[3]), [0]);
        assert!(projection.current_demand()[4].is_empty());
        assert_eq!(bits(&projection.contribution_support()[0]), [0]);
        assert!(projection.contribution_support()[1].is_empty());
        assert_eq!(bits(&projection.finalization_support()[0]), [0]);
        assert!(projection.finalization_support()[1].is_empty());
    }

    #[test]
    fn fixed_and_dynamic_sources_share_exact_transition_and_open_line_support() {
        let templates = super::super::validated_template_fixture();
        let program = physical_union_program_with_source_keys([
            source_key_with_spins(0, 0, [0, 1]),
            source_key(1),
            source_key_with_spins(2, 0, [0, 1]),
        ]);
        let projection = project_all_flow_helicity_support(&program, &templates).unwrap();

        assert_eq!(projection.resolved_helicity_count(), 8);
        // Source 0 is a fixed full-state representation even though its
        // embedded source-evaluator variants retain distinct crossed spins.
        // Source 1 is dynamically gated by the transition witness, while
        // source 2 is unconstrained until closure.
        assert_eq!(bits(&projection.current_availability()[3]), [0, 1, 4, 5]);
        // Both source parents of this otherwise dead current are fixed.
        assert_eq!(
            bits(&projection.current_availability()[4]),
            (0..8).collect::<Vec<_>>()
        );
        // The physical closure owns source 2 directly, exercising the open-line
        // source-parent path without applying a dynamic mask to its fixed spin.
        assert_eq!(bits(&projection.closure_support()[0]), [0, 1, 4, 5]);
        assert_eq!(bits(&projection.current_demand()[0]), [0, 1, 4, 5]);
        assert_eq!(bits(&projection.current_demand()[1]), [0, 1, 4, 5]);
        assert_eq!(bits(&projection.current_demand()[2]), [0, 1, 4, 5]);
        assert_eq!(bits(&projection.current_demand()[3]), [0, 1, 4, 5]);
        assert!(projection.current_demand()[4].is_empty());
    }

    #[test]
    fn source_axis_rejects_a_reserved_dynamic_crossed_spin_choice() {
        let templates = super::super::validated_template_fixture();
        let program = physical_union_program_with_source_keys([
            source_key_with_spins(
                0,
                super::super::DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS,
                [0, super::super::DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS],
            ),
            source_key(1),
            source_key(2),
        ]);
        let error = project_all_flow_helicity_support(&program, &templates)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("source slot 0 has a non-concrete crossed-spin choice"),
            "{error}"
        );
    }

    #[test]
    fn equal_state_evaluator_order_is_inverted_before_unequal_spin_gating() {
        let program = physical_union_program();
        let (source_axes, resolved_helicity_count, word_count) = source_axes(&program).unwrap();
        let mut source_axis_by_current = vec![None; program.currents().len()];
        for (axis_id, axis) in source_axes.iter().enumerate() {
            source_axis_by_current[axis.current_id as usize] = Some(axis_id);
        }

        // Construction [0, 1] is persisted as evaluator [1, 0] by a reversed
        // canonical input order.  Quantum-flow spins remain in construction
        // order and therefore must not be applied directly to the stored row.
        let orders =
            construction_parent_orders_for_evaluator_tuple([1, 0], [0, 0], [0, 0], [1, 0], false);
        assert_eq!(orders, [[0, 1]]);
        let gate = flow_gate(
            &program,
            &source_axis_by_current,
            &source_axes,
            orders[0],
            &[0, 0],
            &[0, 1],
            word_count,
        )
        .unwrap();
        let gate = AllFlowHelicityMask {
            words: gate.into_boxed_slice(),
        };
        assert_eq!(resolved_helicity_count, 8);
        assert_eq!(bits(&gate), [2, 3]);

        // A certified exchange collapses both reciprocal construction orders
        // to the same ascending evaluator tuple.  The stored row is live on
        // the union of both exact spin assignments.
        let exchange_orders =
            construction_parent_orders_for_evaluator_tuple([0, 1], [0, 0], [0, 0], [0, 1], true);
        assert_eq!(exchange_orders, [[0, 1], [1, 0]]);
        let mut exchange_gate = zero_mask(word_count);
        for order in exchange_orders {
            let gate = flow_gate(
                &program,
                &source_axis_by_current,
                &source_axes,
                order,
                &[0, 0],
                &[0, 1],
                word_count,
            )
            .unwrap();
            mask_or_assign(&mut exchange_gate, &gate);
        }
        let exchange_gate = AllFlowHelicityMask {
            words: exchange_gate.into_boxed_slice(),
        };
        assert_eq!(bits(&exchange_gate), [2, 3, 4, 5]);
    }

    #[test]
    fn projection_rejects_a_noncanonical_resolved_helicity_axis() {
        let templates = super::super::validated_template_fixture();
        let base = physical_union_program();
        let program = RecurrenceProgram::new(
            RecurrenceStrategy::AllFlowUnion,
            base.physical_sector_count(),
            7,
            base.dynamic_color_states().to_vec(),
            base.currents().to_vec(),
            base.contributions().to_vec(),
            base.finalizations().to_vec(),
            vec![],
            vec![],
            base.amplitude_destinations().to_vec(),
            base.closure_terms().to_vec(),
        )
        .unwrap();
        let error = project_all_flow_helicity_support(&program, &templates)
            .unwrap_err()
            .to_string();
        assert!(error.contains("span 8 helicities, expected 7"), "{error}");
    }
}
