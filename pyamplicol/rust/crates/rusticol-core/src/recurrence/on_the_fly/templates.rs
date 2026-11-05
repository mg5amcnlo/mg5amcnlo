// SPDX-License-Identifier: 0BSD

use super::*;
use crate::recurrence::construct::TransitionReflectionIndex;
use crate::recurrence::fermion_ordering::FermionOrderingContext;
use crate::recurrence::template::{
    CurrentOrientation, EvaluatorCallableKind, OutputFactorSource, ParticleStatistics,
};

/// Immutable grammar prepared once for one loaded on-the-fly runtime.
/// Query families borrow these exact rows and query-invariant contexts; they
/// never rebuild transition, closure, contact-orbit, propagator, or fermion
/// ordering state per selected query.
#[derive(Debug)]
pub(crate) struct PreparedOnTheFlyGrammarV1 {
    pub(super) transitions: BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    pub(super) transition_reflections: TransitionReflectionIndex,
    pub(super) contact_orbits: PreparedContactOrbitIndex,
    pub(super) closures: BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    pub(super) propagators: BTreeMap<u32, Option<u32>>,
    pub(super) sources: BTreeMap<(u32, u32), PreparedOnTheFlySourceContractV1>,
    pub(super) propagator_executors: BTreeMap<u32, PreparedOnTheFlyExecutorContractV1>,
    pub(super) fermion_ordering: FermionOrderingContext,
    pub(super) closure_anchor_policy: OnTheFlyClosureAnchorPolicyV1,
}

impl PreparedOnTheFlyGrammarV1 {
    pub(crate) fn canonicalize_query(
        &self,
        seed: &OnTheFlyProcessSeedV1,
        query: DecodedLcQueryV1,
    ) -> RusticolResult<DecodedLcQueryV1> {
        self.closure_anchor_policy.canonicalize_query(seed, query)
    }

    pub(crate) fn authenticate_canonical_query(
        &self,
        seed: &OnTheFlyProcessSeedV1,
        query: &DecodedLcQueryV1,
    ) -> RusticolResult<()> {
        self.closure_anchor_policy
            .authenticate_canonical_query(seed, query)
    }

    pub(super) fn closure_anchor_for_components(
        &self,
        seed: &OnTheFlyProcessSeedV1,
        components: &[LCColorComponent],
    ) -> Option<u32> {
        self.closure_anchor_policy
            .closure_anchor_for_components(seed, components)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PreparedOnTheFlyExecutorContractV1 {
    pub(super) operation_id: u32,
    pub(super) operation_semantic_digest: SemanticDigest,
    pub(super) evaluator_binding_id: u32,
    pub(super) evaluator_binding_semantic_digest: SemanticDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PreparedOnTheFlySourceContractV1 {
    pub(super) current_state: CurrentStateRow,
    pub(super) color_seed: LCColorSourceSeed,
    pub(super) executor: PreparedOnTheFlyExecutorContractV1,
}

pub(super) fn prepare_on_the_fly_grammar_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<PreparedOnTheFlyGrammarV1> {
    let transitions = prepared_transitions(templates, catalog)?;
    let transition_reflections = TransitionReflectionIndex::new(templates.input(), catalog)?;
    let contact_orbits = PreparedContactOrbitIndex::new(&transitions)?;
    let closures = prepared_closures(templates, catalog)?;
    // Structural-zero queries may trust this immutable domain only because it
    // is proved complete once, before the grammar enters the runtime cache.
    validate_prepared_closure_domain(templates, catalog, &closures)?;
    let propagators = propagator_by_state(templates)?;
    let fermion_ordering = on_the_fly_fermion_ordering_context(seed)?;
    let closure_anchor_policy =
        certified_cyclic_trace_closure_anchor_policy(templates.input(), catalog, seed)?;
    Ok(PreparedOnTheFlyGrammarV1 {
        transitions,
        transition_reflections,
        contact_orbits,
        closures,
        sources: prepared_source_contracts(templates, catalog, seed)?,
        propagator_executors: prepared_propagator_executors(templates, catalog, &propagators)?,
        propagators,
        fermion_ordering,
        closure_anchor_policy,
    })
}

/// Infer the v4 cyclic-anchor capability from the compact process seed and
/// the already-authenticated template catalog.
///
/// This deliberately mirrors the generation-side v4 admission proof.  A
/// false predicate retains the established selector-endpoint policy; malformed
/// catalog references remain hard errors rather than silently widening the
/// optimization domain.
fn certified_cyclic_trace_closure_anchor_policy(
    input: &crate::recurrence::template::OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<OnTheFlyClosureAnchorPolicyV1> {
    let fallback = || Ok(OnTheFlyClosureAnchorPolicyV1::selector_endpoint());
    if !seed.pairing_classes.is_empty()
        || seed.source_anchors.iter().any(|anchor| {
            anchor.is_fermionic || anchor.color_role != OnTheFlyExternalColorRoleV1::Adjoint
        })
    {
        return fallback();
    }
    let state_template_ids = seed
        .source_anchors
        .iter()
        .flat_map(|anchor| anchor.states.iter())
        .map(|state| state.current_state_template_id)
        .collect::<BTreeSet<_>>();
    if state_template_ids.len() != 1 {
        return fallback();
    }
    let state_template_id = *state_template_ids
        .iter()
        .next()
        .expect("tested one cyclic-anchor state template");
    let state = input
        .current_states
        .get(state_template_id as usize)
        .copied()
        .ok_or_else(|| integrity("cyclic-anchor current-state template is absent"))?;
    if state.id != state_template_id
        || CurrentOrientation::try_from(state.orientation)? != CurrentOrientation::SelfConjugate
        || ParticleStatistics::try_from(state.statistics)? != ParticleStatistics::Boson
        || state.particle_id != state.anti_particle_id
        || state.color_representation != 8
        || catalog.string(
            state.lc_color_shape_string_id,
            "cyclic-anchor current-state color shape",
        )? != "adjoint-segment"
        || state.auxiliary_kind_string_id != MISSING_U32
    {
        return fallback();
    }

    let mut has_direct_closure = false;
    for closure in input.closures.iter().copied() {
        let coupling_order_set = input
            .coupling_order_ranges
            .get(closure.coupling_order_set_id as usize)
            .ok_or_else(|| integrity("cyclic-anchor coupling-order set is absent"))?;
        if catalog.u32_sequence(
            closure.input_state_sequence_id,
            "cyclic-anchor closure input states",
        )? != [state_template_id, state_template_id]
            || closure.result_state_template_id != MISSING_U32
            || !catalog
                .u32_sequence(
                    closure.coupling_parameter_sequence_id,
                    "cyclic-anchor closure coupling parameters",
                )?
                .is_empty()
            || coupling_order_set.range.count != 0
            || !catalog
                .u32_sequence(
                    closure.eligible_quantum_flow_sequence_id,
                    "cyclic-anchor closure quantum flows",
                )?
                .is_empty()
            || OutputFactorSource::try_from(closure.output_factor_source)?
                != OutputFactorSource::None
            || catalog.string(
                closure.equivalence_class_string_id,
                "cyclic-anchor closure equivalence class",
            )? != "direct-contraction"
        {
            continue;
        }
        let binding = input
            .evaluator_bindings
            .get(closure.evaluator_binding_id as usize)
            .copied()
            .ok_or_else(|| integrity("cyclic-anchor closure evaluator is absent"))?;
        if binding.id != closure.evaluator_binding_id
            || EvaluatorCallableKind::try_from(binding.callable_kind)?
                != EvaluatorCallableKind::RusticolTemplate
            || EvaluatorContractKind::try_from(binding.contract_kind)?
                != EvaluatorContractKind::Closure
            || catalog.u32_sequence(
                binding.input_state_sequence_id,
                "cyclic-anchor evaluator input states",
            )? != [state_template_id, state_template_id]
            || binding.output_state_template_id != MISSING_U32
        {
            continue;
        }
        let color = input
            .color_contractions
            .get(closure.color_contraction_template_id as usize)
            .copied()
            .ok_or_else(|| integrity("cyclic-anchor color contraction is absent"))?;
        if color.id == closure.color_contraction_template_id
            && catalog.i32_sequence(
                color.input_representation_sequence_id,
                "cyclic-anchor color input representations",
            )? == [8, 8]
            && color.has_output_representation == 0
        {
            has_direct_closure = true;
            break;
        }
    }
    if !has_direct_closure {
        return fallback();
    }
    let minimum = seed
        .source_anchors
        .iter()
        .map(|anchor| anchor.source_slot)
        .min()
        .ok_or_else(|| integrity("cyclic-anchor source domain is empty"))?;
    Ok(OnTheFlyClosureAnchorPolicyV1::certified_cyclic_minimum(
        minimum,
    ))
}

#[derive(Clone, Debug)]
pub(super) struct PreparedFlavourFlow {
    operation: PreparedFlavourFlowOperation,
    result_particle: i32,
    static_result: Box<[i32]>,
}

#[derive(Clone, Copy, Debug)]
enum PreparedFlavourFlowOperation {
    Constant,
    AppendLeft,
    AppendRight,
    ConcatLeftRight,
}

impl PreparedFlavourFlow {
    fn new(quantum: QuantumFlowRow, catalog: &TemplateCatalog<'_>) -> RusticolResult<Self> {
        let operation = match catalog.string(
            quantum.flavour_flow_operation_string_id,
            "quantum-flow flavour operation",
        )? {
            "constant-result" => PreparedFlavourFlowOperation::Constant,
            "append-left-result" => PreparedFlavourFlowOperation::AppendLeft,
            "append-right-result" => PreparedFlavourFlowOperation::AppendRight,
            "concat-left-right-result" => PreparedFlavourFlowOperation::ConcatLeftRight,
            value => {
                return Err(invalid(format!(
                    "unsupported quantum-flow flavour operation {value:?}"
                )));
            }
        };
        let static_result = catalog
            .flavour_flow(quantum.result_flavour_flow_id, "quantum result flavour")?
            .to_vec();
        let result_particle = *static_result
            .last()
            .ok_or_else(|| invalid("quantum result flavour ancestry is empty"))?;
        Ok(Self {
            operation,
            result_particle,
            static_result: static_result.into_boxed_slice(),
        })
    }

    pub(super) fn apply(&self, parents: [&CurrentCoreKey; 2]) -> Vec<i32> {
        self.apply_flows(parents[0].flavour_flow(), parents[1].flavour_flow())
    }

    pub(super) fn apply_flows(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        let append = |parent: &[i32], result_particle| {
            let mut result = parent.to_vec();
            if result.last().copied() != Some(result_particle) {
                result.push(result_particle);
            }
            result
        };
        match self.operation {
            PreparedFlavourFlowOperation::Constant => self.static_result.to_vec(),
            PreparedFlavourFlowOperation::AppendLeft => append(left, self.result_particle),
            PreparedFlavourFlowOperation::AppendRight => append(right, self.result_particle),
            PreparedFlavourFlowOperation::ConcatLeftRight => {
                let mut result = left.to_vec();
                result.extend_from_slice(right);
                result.push(self.result_particle);
                result
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedWitness {
    pub(super) row: LCColorTransitionWitnessRow,
    pub(super) witness: crate::recurrence::LCColorTransitionWitness,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedTransition {
    pub(super) row: TransitionRow,
    pub(super) input_states: [u32; 2],
    pub(super) quantum: QuantumFlowRow,
    pub(super) input_spins: [i32; 2],
    pub(super) local_orders: Box<[u32]>,
    canonical_input_order: [u32; 2],
    input_exchange_factor: Option<ExactComplexRational>,
    pub(super) base_factor: ExactComplexRational,
    #[cfg(feature = "on-the-fly-test-support")]
    pub(super) transition_factor: ExactComplexRational,
    #[cfg(feature = "on-the-fly-test-support")]
    pub(super) contraction_factor: ExactComplexRational,
    coupling_authenticated: bool,
    binding_coupling: ExactComplexRational,
    output_factor_source: u8,
    pub(super) flavour: PreparedFlavourFlow,
    pub(super) quantum_semantic_digest: SemanticDigest,
    pub(super) transition_semantic_digest: SemanticDigest,
    pub(super) evaluator_binding_digest: SemanticDigest,
    pub(super) contact_orbit: Option<PreparedContactOrbitTransition>,
    pub(super) witnesses: Box<[PreparedWitness]>,
}

impl PreparedTransition {
    fn new(
        row: TransitionRow,
        templates: &ValidatedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input = templates.input();
        let input_states: [u32; 2] = catalog
            .u32_sequence(row.input_state_sequence_id, "transition input states")?
            .try_into()
            .map_err(|_| invalid("on-the-fly transition must be binary"))?;
        let quantum = *input
            .quantum_flows
            .get(row.quantum_flow_template_id as usize)
            .ok_or_else(|| invalid("transition quantum flow is absent"))?;
        let input_spins: [i32; 2] = catalog
            .i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?
            .try_into()
            .map_err(|_| invalid("on-the-fly quantum flow must be binary"))?;
        let quantum_states =
            catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
        if quantum_states != input_states {
            return Err(invalid("transition and quantum-flow input states disagree"));
        }
        let canonical_input_order = match catalog.u32_sequence(
            row.canonical_input_order_sequence_id,
            "transition canonical input order",
        )? {
            [0, 1] => [0, 1],
            [1, 0] => [1, 0],
            _ => return Err(invalid("transition canonical input order is not binary")),
        };
        let input_exchange_factor = (row.input_exchange_factor_id != MISSING_U32)
            .then(|| catalog.factor(row.input_exchange_factor_id, "transition input exchange"))
            .transpose()?;
        let contraction = input
            .color_contractions
            .get(row.color_contraction_template_id as usize)
            .ok_or_else(|| invalid("transition color contraction is absent"))?;
        let quantum_coupling =
            catalog.factor(quantum.exact_coupling_factor_id, "quantum coupling")?;
        let binding_coupling = catalog.factor(
            row.binding_coupling_factor_id,
            "transition binding coupling",
        )?;
        let transition_exact_factor = catalog.factor(row.exact_factor_id, "transition exact")?;
        let contraction_exact_factor = catalog.factor(
            contraction.exact_coefficient_factor_id,
            "transition color contraction",
        )?;
        let base_factor = multiply_factors(&[transition_exact_factor, contraction_exact_factor])?;
        let binding = input
            .evaluator_bindings
            .get(row.evaluator_binding_id as usize)
            .ok_or_else(|| invalid("transition evaluator binding is absent"))?;
        if binding.id != row.evaluator_binding_id
            || EvaluatorContractKind::try_from(binding.contract_kind)?
                != EvaluatorContractKind::Vertex
        {
            return Err(invalid("transition evaluator binding has the wrong role"));
        }
        let local_orders = catalog
            .coupling_orders(row.coupling_order_set_id)?
            .into_boxed_slice();
        let quantum_semantic_digest =
            catalog.digest(quantum.semantic_digest_id, "quantum semantic")?;
        let witness_rows = catalog.witness_rows(row.color_contraction_template_id)?;
        let contact_orbit = prepare_contact_orbit_transition(
            input,
            catalog,
            row,
            quantum_semantic_digest,
            &local_orders,
            binding_coupling,
            transition_exact_factor,
            contraction_exact_factor,
            input_exchange_factor,
            witness_rows,
        )?;
        let witnesses = witness_rows
            .iter()
            .copied()
            .map(|witness_row| {
                Ok(PreparedWitness {
                    row: witness_row,
                    witness: catalog.witness(witness_row)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            row,
            input_states,
            quantum,
            input_spins,
            local_orders,
            canonical_input_order,
            input_exchange_factor,
            base_factor,
            #[cfg(feature = "on-the-fly-test-support")]
            transition_factor: transition_exact_factor,
            #[cfg(feature = "on-the-fly-test-support")]
            contraction_factor: contraction_exact_factor,
            coupling_authenticated: quantum_coupling == binding_coupling,
            binding_coupling,
            output_factor_source: row.output_factor_source,
            flavour: PreparedFlavourFlow::new(quantum, catalog)?,
            quantum_semantic_digest,
            transition_semantic_digest: catalog
                .digest(row.semantic_digest_id, "transition semantic")?,
            evaluator_binding_digest: catalog
                .digest(binding.semantic_digest_id, "transition evaluator binding")?,
            contact_orbit,
            witnesses: witnesses.into_boxed_slice(),
        })
    }

    pub(super) fn parent_ids(
        &self,
        left_state: u32,
        right_state: u32,
        left_id: u32,
        right_id: u32,
    ) -> Option<[u32; 2]> {
        if self.input_states == [left_state, right_state] {
            Some([left_id, right_id])
        } else if left_state != right_state && self.input_states == [right_state, left_state] {
            Some([right_id, left_id])
        } else {
            None
        }
    }

    pub(super) fn evaluator_parents(&self, concrete: [u32; 2]) -> ([u32; 2], ExactComplexRational) {
        let mut ordered = match self.canonical_input_order {
            [0, 1] => concrete,
            [1, 0] => [concrete[1], concrete[0]],
            _ => unreachable!("validated transition input order"),
        };
        let mut factor = ExactComplexRational::ONE;
        if let Some(exchange) = self.input_exchange_factor
            && ordered[1] < ordered[0]
        {
            ordered.swap(0, 1);
            factor = exchange;
        }
        (ordered, factor)
    }

    pub(super) fn output_factor(&self) -> RusticolResult<ExactComplexRational> {
        if !self.coupling_authenticated {
            return Err(invalid(
                "transition binding coupling does not match its quantum-flow coupling witness",
            ));
        }
        output_factor_from_binding(
            self.binding_coupling,
            self.output_factor_source,
            "on-the-fly transition",
        )
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedClosureQuantum {
    pub(super) row: Option<QuantumFlowRow>,
    pub(super) input_states: Option<[u32; 2]>,
    pub(super) input_spins: Option<[i32; 2]>,
    coupling_authenticated: bool,
    binding_coupling: ExactComplexRational,
    output_factor_source: u8,
}

impl PreparedClosureQuantum {
    pub(super) fn output_factor(&self) -> RusticolResult<ExactComplexRational> {
        if !self.coupling_authenticated {
            return Err(invalid(
                "closure binding coupling does not match its quantum-flow coupling witness",
            ));
        }
        output_factor_from_binding(
            self.binding_coupling,
            self.output_factor_source,
            "on-the-fly closure quantum flow",
        )
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedClosure {
    pub(super) row: ClosureRow,
    pub(super) input_states: [u32; 2],
    pub(super) local_orders: Box<[u32]>,
    pub(super) component_coefficients: Box<[ExactComplexRational]>,
    canonical_input_order: [u32; 2],
    input_exchange_factor: Option<ExactComplexRational>,
    pub(super) base_factor: ExactComplexRational,
    pub(super) closure_semantic_digest: SemanticDigest,
    pub(super) evaluator_binding_digest: SemanticDigest,
    pub(super) quantum_flows: Box<[PreparedClosureQuantum]>,
    pub(super) witnesses: Box<[PreparedWitness]>,
}

impl PreparedClosure {
    fn new(
        row: ClosureRow,
        templates: &ValidatedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input = templates.input();
        let input_states: [u32; 2] = catalog
            .u32_sequence(row.input_state_sequence_id, "closure input states")?
            .try_into()
            .map_err(|_| invalid("on-the-fly closure must be binary"))?;
        let local_orders = catalog
            .coupling_orders(row.coupling_order_set_id)?
            .into_boxed_slice();
        let component_coefficients = catalog
            .u32_sequence(
                row.component_coefficient_sequence_id,
                "closure component coefficients",
            )?
            .iter()
            .map(|factor_id| catalog.factor(*factor_id, "closure component coefficient"))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        let canonical_input_order = match catalog.u32_sequence(
            row.canonical_input_order_sequence_id,
            "closure canonical input order",
        )? {
            [0, 1] => [0, 1],
            [1, 0] => [1, 0],
            _ => return Err(invalid("closure canonical input order is not binary")),
        };
        let input_exchange_factor = (row.input_exchange_factor_id != MISSING_U32)
            .then(|| catalog.factor(row.input_exchange_factor_id, "closure input exchange"))
            .transpose()?;
        let contraction = input
            .color_contractions
            .get(row.color_contraction_template_id as usize)
            .ok_or_else(|| invalid("closure color contraction is absent"))?;
        let binding_coupling =
            catalog.factor(row.binding_coupling_factor_id, "closure binding coupling")?;
        let base_factor = multiply_factors(&[
            catalog.factor(row.exact_factor_id, "closure exact")?,
            catalog.factor(
                contraction.exact_coefficient_factor_id,
                "closure color contraction",
            )?,
        ])?;
        let binding = input
            .evaluator_bindings
            .get(row.evaluator_binding_id as usize)
            .ok_or_else(|| invalid("closure evaluator binding is absent"))?;
        if binding.id != row.evaluator_binding_id
            || EvaluatorContractKind::try_from(binding.contract_kind)?
                != EvaluatorContractKind::Closure
        {
            return Err(invalid("closure evaluator binding has the wrong role"));
        }
        let eligible = catalog.u32_sequence(
            row.eligible_quantum_flow_sequence_id,
            "closure eligible quantum flows",
        )?;
        let quantum_flows = if eligible.is_empty() {
            vec![PreparedClosureQuantum {
                row: None,
                input_states: None,
                input_spins: None,
                coupling_authenticated: true,
                binding_coupling,
                output_factor_source: row.output_factor_source,
            }]
        } else {
            eligible
                .iter()
                .copied()
                .map(|id| {
                    let quantum = *input
                        .quantum_flows
                        .get(id as usize)
                        .ok_or_else(|| invalid("closure quantum flow is absent"))?;
                    let states: [u32; 2] = catalog
                        .u32_sequence(quantum.input_state_sequence_id, "closure quantum states")?
                        .try_into()
                        .map_err(|_| invalid("closure quantum flow must be binary"))?;
                    let spins: [i32; 2] = catalog
                        .i32_sequence(quantum.input_spin_sequence_id, "closure quantum spins")?
                        .try_into()
                        .map_err(|_| invalid("closure quantum flow must be binary"))?;
                    let quantum_coupling = catalog
                        .factor(quantum.exact_coupling_factor_id, "closure quantum coupling")?;
                    Ok(PreparedClosureQuantum {
                        row: Some(quantum),
                        input_states: Some(states),
                        input_spins: Some(spins),
                        coupling_authenticated: quantum_coupling == binding_coupling,
                        binding_coupling,
                        output_factor_source: row.output_factor_source,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?
        };
        let witnesses = catalog
            .witness_rows(row.color_contraction_template_id)?
            .iter()
            .copied()
            .map(|witness_row| {
                Ok(PreparedWitness {
                    row: witness_row,
                    witness: catalog.witness(witness_row)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            row,
            input_states,
            local_orders,
            component_coefficients,
            canonical_input_order,
            input_exchange_factor,
            base_factor,
            closure_semantic_digest: catalog.digest(row.semantic_digest_id, "closure semantic")?,
            evaluator_binding_digest: catalog
                .digest(binding.semantic_digest_id, "closure evaluator binding")?,
            quantum_flows: quantum_flows.into_boxed_slice(),
            witnesses: witnesses.into_boxed_slice(),
        })
    }

    pub(super) fn parent_ids(
        &self,
        anchor_state: u32,
        complement_state: u32,
        anchor_id: u32,
        complement_id: u32,
    ) -> Option<[u32; 2]> {
        if self.input_states == [complement_state, anchor_state] {
            Some([complement_id, anchor_id])
        } else if anchor_state != complement_state
            && self.input_states == [anchor_state, complement_state]
        {
            Some([anchor_id, complement_id])
        } else {
            None
        }
    }

    pub(super) fn evaluator_parents(&self, concrete: [u32; 2]) -> ([u32; 2], ExactComplexRational) {
        let mut ordered = match self.canonical_input_order {
            [0, 1] => concrete,
            [1, 0] => [concrete[1], concrete[0]],
            _ => unreachable!("validated closure input order"),
        };
        let mut factor = ExactComplexRational::ONE;
        if let Some(exchange) = self.input_exchange_factor
            && ordered[1] < ordered[0]
        {
            ordered.swap(0, 1);
            factor = exchange;
        }
        (ordered, factor)
    }
}

fn prepared_source_contracts(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<BTreeMap<(u32, u32), PreparedOnTheFlySourceContractV1>> {
    let mut result = BTreeMap::new();
    for anchor in &seed.source_anchors {
        for state in &anchor.states {
            let (source, current_state) =
                validate_source_contract(templates, catalog, anchor, state)?;
            let binding = templates
                .input()
                .evaluator_bindings
                .get(source.evaluator_binding_id as usize)
                .ok_or_else(|| integrity("source evaluator binding is absent"))?;
            let contract = PreparedOnTheFlySourceContractV1 {
                current_state,
                color_seed: catalog.source_seed(source)?,
                executor: PreparedOnTheFlyExecutorContractV1 {
                    operation_id: source.id,
                    operation_semantic_digest: catalog
                        .digest(source.semantic_digest_id, "source semantic")?,
                    evaluator_binding_id: binding.id,
                    evaluator_binding_semantic_digest: catalog
                        .digest(binding.semantic_digest_id, "source evaluator semantic")?,
                },
            };
            let key = (state.source_template_id, state.current_state_template_id);
            if let Some(previous) = result.insert(key, contract)
                && previous != contract
            {
                return Err(integrity(
                    "equal compact source identities have different prepared contracts",
                ));
            }
        }
    }
    Ok(result)
}

fn prepared_propagator_executors(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    propagators: &BTreeMap<u32, Option<u32>>,
) -> RusticolResult<BTreeMap<u32, PreparedOnTheFlyExecutorContractV1>> {
    let mut result = BTreeMap::new();
    for propagator_id in propagators.values().flatten().copied() {
        let row = templates
            .input()
            .propagators
            .get(propagator_id as usize)
            .ok_or_else(|| integrity("propagator executor template is absent"))?;
        let binding = templates
            .input()
            .evaluator_bindings
            .get(row.evaluator_binding_id as usize)
            .ok_or_else(|| integrity("propagator executor binding is absent"))?;
        let contract = PreparedOnTheFlyExecutorContractV1 {
            operation_id: row.id,
            operation_semantic_digest: catalog
                .digest(row.semantic_digest_id, "propagator semantic")?,
            evaluator_binding_id: binding.id,
            evaluator_binding_semantic_digest: catalog
                .digest(binding.semantic_digest_id, "propagator evaluator semantic")?,
        };
        if let Some(previous) = result.insert(propagator_id, contract)
            && previous != contract
        {
            return Err(integrity(
                "equal propagator identities have different prepared contracts",
            ));
        }
    }
    Ok(result)
}

pub(super) fn canonical_state_pair(left: u32, right: u32) -> (u32, u32) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

pub(super) fn propagator_by_state(
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<BTreeMap<u32, Option<u32>>> {
    let mut result = BTreeMap::new();
    for row in &templates.input().propagators {
        let value = (row.applies_propagator != 0).then_some(row.id);
        if result.insert(row.state_template_id, value).is_some() {
            return Err(invalid(format!(
                "current state {} has multiple propagator contracts",
                row.state_template_id
            )));
        }
    }
    Ok(result)
}

pub(super) fn prepared_transitions(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<BTreeMap<(u32, u32), Vec<PreparedTransition>>> {
    let mut result = BTreeMap::<(u32, u32), Vec<PreparedTransition>>::new();
    for row in templates.input().transitions.iter().copied() {
        let prepared = PreparedTransition::new(row, templates, catalog)?;
        result
            .entry(canonical_state_pair(
                prepared.input_states[0],
                prepared.input_states[1],
            ))
            .or_default()
            .push(prepared);
    }
    Ok(result)
}

pub(super) fn prepared_closures(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<BTreeMap<(u32, u32), Vec<PreparedClosure>>> {
    let mut result = BTreeMap::<(u32, u32), Vec<PreparedClosure>>::new();
    for row in templates.input().closures.iter().copied() {
        let prepared = PreparedClosure::new(row, templates, catalog)?;
        result
            .entry(canonical_state_pair(
                prepared.input_states[0],
                prepared.input_states[1],
            ))
            .or_default()
            .push(prepared);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::contact_orbit_owner::{
        ContactOrbitTestBinding, contact_orbit_test_template,
    };
    use crate::recurrence::fermion_ordering::fermion_ordering_factor;
    use crate::recurrence::validated_template_fixture;

    fn append_string(
        input: &mut crate::recurrence::template::OwnedRecurrenceTemplateInput,
        value: &str,
    ) -> u32 {
        let id = u32::try_from(input.string_ranges.len()).unwrap();
        let start = u64::try_from(input.string_bytes.len()).unwrap();
        input.string_bytes.extend_from_slice(value.as_bytes());
        input
            .string_ranges
            .push(crate::recurrence::CheckedTableRange::new(
                start,
                u64::try_from(value.len()).unwrap(),
            ));
        id
    }

    fn four_adjoint_seed(
        summary: crate::recurrence::template::RecurrenceTemplateValidationSummary,
    ) -> OnTheFlyProcessSeedV1 {
        let anchors = (0..4_u32)
            .map(|source_slot| {
                let state = OnTheFlySourceStateV1::new(
                    0,
                    1,
                    1,
                    0,
                    0,
                    SemanticDigest::new([5; 32]).unwrap(),
                    SemanticDigest::new([4; 32]).unwrap(),
                    1,
                    ExactComplexRational::ONE,
                    50_000,
                    0,
                    vec![21],
                    0,
                    SemanticDigest::new([17; 32]).unwrap(),
                    OnTheFlySourceWavefunctionFamilyV1::Vector,
                    OnTheFlySourceOrientationV1::SelfConjugate,
                    None,
                )
                .unwrap();
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    source_slot,
                    source_slot < 2,
                    OnTheFlyExternalColorRoleV1::Adjoint,
                    false,
                    None,
                    vec![state],
                )
                .unwrap()
            })
            .collect();
        OnTheFlyProcessSeedV1::new(
            SemanticDigest::new([91; 32]).unwrap(),
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            SemanticDigest::new([0x7d; 32]).unwrap(),
            SemanticDigest::new([92; 32]).unwrap(),
            "raw-amplitude-test",
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

    fn cyclic_anchor_template_fixture() -> (
        crate::recurrence::template::OwnedRecurrenceTemplateInput,
        OnTheFlyProcessSeedV1,
    ) {
        let templates = validated_template_fixture();
        let seed = four_adjoint_seed(templates.summary());
        let mut input = templates.into_input();
        let adjoint_shape = append_string(&mut input, "adjoint-segment");
        let direct_contraction = append_string(&mut input, "direct-contraction");
        let empty_sequence = input
            .u32_sequence_ranges
            .iter()
            .find(|row| row.range.count == 0)
            .unwrap()
            .id;

        input.current_states[0].color_representation = 8;
        input.current_states[0].lc_color_shape_string_id = adjoint_shape;
        let closure = &mut input.closures[0];
        closure.result_state_template_id = MISSING_U32;
        closure.eligible_quantum_flow_sequence_id = empty_sequence;
        closure.equivalence_class_string_id = direct_contraction;
        input.evaluator_bindings[closure.evaluator_binding_id as usize].callable_kind =
            EvaluatorCallableKind::RusticolTemplate as u8;
        let color = &mut input.color_contractions[closure.color_contraction_template_id as usize];
        let representations = input.i32_sequence_ranges
            [color.input_representation_sequence_id as usize]
            .range
            .as_usize_range(input.i32_sequence_values.len(), "test color inputs")
            .unwrap();
        input.i32_sequence_values[representations].copy_from_slice(&[8, 8]);
        (input, seed)
    }

    #[test]
    fn on_the_fly_prepared_transition_binds_only_certified_contact_metadata() {
        let none = contact_orbit_test_template(ContactOrbitTestBinding::None)
            .validate()
            .unwrap();
        let none_catalog = TemplateCatalog::new(none.input()).unwrap();
        let none_prepared = prepared_transitions(&none, &none_catalog).unwrap();
        assert!(
            none_prepared
                .values()
                .flatten()
                .next()
                .unwrap()
                .contact_orbit
                .is_none()
        );

        let one = contact_orbit_test_template(ContactOrbitTestBinding::One)
            .validate()
            .unwrap();
        let one_catalog = TemplateCatalog::new(one.input()).unwrap();
        let one_prepared = prepared_transitions(&one, &one_catalog).unwrap();
        assert!(
            one_prepared
                .values()
                .flatten()
                .next()
                .unwrap()
                .contact_orbit
                .is_some()
        );
    }

    #[test]
    fn prepared_grammar_caches_exact_query_invariant_contexts() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            SemanticDigest::new([0x7d; 32]).unwrap(),
        )
        .unwrap();
        let catalog = TemplateCatalog::new(templates.input()).unwrap();
        let grammar = prepare_on_the_fly_grammar_v1(&templates, &catalog, &seed).unwrap();

        assert_eq!(
            grammar.contact_orbits,
            PreparedContactOrbitIndex::new(&grammar.transitions).unwrap(),
        );
        assert_eq!(
            grammar.closure_anchor_policy,
            OnTheFlyClosureAnchorPolicyV1::selector_endpoint(),
        );
        let fresh_fermion_ordering = on_the_fly_fermion_ordering_context(&seed).unwrap();
        assert_eq!(
            fermion_ordering_factor(
                &templates.input().current_states,
                [0, 0],
                [&[0], &[1]],
                &grammar.fermion_ordering,
            )
            .unwrap(),
            fermion_ordering_factor(
                &templates.input().current_states,
                [0, 0],
                [&[0], &[1]],
                &fresh_fermion_ordering,
            )
            .unwrap(),
        );
    }

    #[test]
    fn cyclic_anchor_policy_requires_the_exact_v4_template_contract() {
        let (input, seed) = cyclic_anchor_template_fixture();
        let catalog = TemplateCatalog::new(&input).unwrap();
        assert_eq!(
            certified_cyclic_trace_closure_anchor_policy(&input, &catalog, &seed).unwrap(),
            OnTheFlyClosureAnchorPolicyV1::certified_cyclic_minimum(0),
        );

        let mut ineligible = input.clone();
        let closure = ineligible.closures[0];
        ineligible.evaluator_bindings[closure.evaluator_binding_id as usize].callable_kind =
            EvaluatorCallableKind::PreparedKernel as u8;
        let catalog = TemplateCatalog::new(&ineligible).unwrap();
        assert_eq!(
            certified_cyclic_trace_closure_anchor_policy(&ineligible, &catalog, &seed).unwrap(),
            OnTheFlyClosureAnchorPolicyV1::selector_endpoint(),
        );

        let mut auxiliary = input;
        auxiliary.current_states[0].auxiliary_kind_string_id =
            auxiliary.current_states[0].lc_color_shape_string_id;
        let catalog = TemplateCatalog::new(&auxiliary).unwrap();
        assert_eq!(
            certified_cyclic_trace_closure_anchor_policy(&auxiliary, &catalog, &seed).unwrap(),
            OnTheFlyClosureAnchorPolicyV1::selector_endpoint(),
        );
    }
}
