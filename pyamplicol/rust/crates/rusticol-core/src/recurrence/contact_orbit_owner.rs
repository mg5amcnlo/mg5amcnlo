// SPDX-License-Identifier: 0BSD

//! Exact owner selection for compiler-certified scalar contact orbits.
//!
//! The helper is deliberately process-blind. It forgets only binary
//! traversal direction and the concrete evaluator-binding row while retaining
//! every physical, selector, color, coupling, and numerical evaluator witness.
//! Owner selection uses fallibly reserved vector scratch: it does not clone
//! current/application graphs or grow map/set nodes per transition.

use std::{cmp::Ordering, mem::size_of};

use super::{
    CurrentCoreKey, CurrentSourceBinding, DynamicLCColorStateId, ExactComplexRational,
    LCColorWitnessTermId, RecurrenceNodeKind, RecurrenceStrategy, SemanticDigest,
    construct::TemplateCatalog,
    template::{
        ContactOrbitStage, EvaluatorCallableKind, EvaluatorContractKind,
        LCColorTransitionWitnessRow, OwnedRecurrenceTemplateInput, TransitionRow,
    },
};
use crate::{RusticolError, RusticolResult};

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message)
}

fn allocation(message: impl Into<String>) -> RusticolError {
    RusticolError::internal(message)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitCoveredLegs {
    len: u8,
    values: [u32; 2],
}

impl ContactOrbitCoveredLegs {
    pub(super) fn new(values: &[u32]) -> RusticolResult<Self> {
        let len = u8::try_from(values.len())
            .map_err(|_| integrity("contact-orbit covered-leg count exceeds u8"))?;
        if !(1..=2).contains(&len) {
            return Err(integrity(
                "contact-orbit covered-leg count must be one or two",
            ));
        }
        let mut canonical = [0_u32; 2];
        canonical[..values.len()].copy_from_slice(values);
        canonical[..values.len()].sort_unstable();
        if values.len() == 2 && canonical[0] == canonical[1] {
            return Err(integrity("contact-orbit covered legs are not unique"));
        }
        Ok(Self {
            len,
            values: canonical,
        })
    }

    fn values(&self) -> &[u32] {
        &self.values[..usize::from(self.len)]
    }

    const fn len(self) -> usize {
        self.len as usize
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitStepProof {
    pub(super) certificate_id: u32,
    pub(super) certificate_semantic_digest: SemanticDigest,
    pub(super) step_semantic_digest: SemanticDigest,
    pub(super) stage: ContactOrbitStage,
    pub(super) result_leg: u32,
    pub(super) physical_leg_equivalence_classes: Vec<u32>,
    pub(super) left_covered_legs: ContactOrbitCoveredLegs,
    pub(super) right_covered_legs: ContactOrbitCoveredLegs,
    pub(super) source_particle_legs: [i32; 3],
    pub(super) certificate_reconstruction_factor: ExactComplexRational,
    pub(super) step_reconstruction_factor: ExactComplexRational,
    pub(super) evaluator_class: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitApplicationWitness {
    pub(super) transition_equivalence_class: String,
    pub(super) quantum_semantic_digest: SemanticDigest,
    pub(super) color_contraction_semantic_digest: SemanticDigest,
    pub(super) color_witness_term_id: LCColorWitnessTermId,
    pub(super) color_witness_proof_digest: SemanticDigest,
    pub(super) coupling_orders: Vec<u32>,
    pub(super) coupling_parameter_ids: Vec<u32>,
    pub(super) momentum_convention: Vec<u32>,
    pub(super) binding_coupling: ExactComplexRational,
    pub(super) transition_exact_factor: ExactComplexRational,
    pub(super) color_exact_factor: ExactComplexRational,
    pub(super) witness_exact_factor: ExactComplexRational,
    pub(super) input_exchange_factor: Option<ExactComplexRational>,
    pub(super) output_factor_source: u8,
    pub(super) evaluator_contract_kind: u8,
    pub(super) evaluator_callable_kind: u8,
    pub(super) evaluator_callable_signature: SemanticDigest,
    pub(super) evaluator_input_layout: Vec<u32>,
    pub(super) evaluator_output_layout: Vec<u32>,
    pub(super) evaluator_exact_expression_digests: Vec<SemanticDigest>,
    pub(super) evaluator_runtime_template: Option<String>,
    pub(super) output_projection_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreparedContactOrbitTransition {
    step: ContactOrbitStepProof,
    input_state_template_ids: [u32; 2],
    transition_semantic_digest: SemanticDigest,
    applications: Vec<ContactOrbitApplicationWitness>,
}

/// Minimal parent facts needed by the contact-orbit construction-domain
/// predicate.  The process-global on-the-fly coupling sweep deliberately has
/// no `CurrentCoreKey`; this keeps its admission decision identical to the
/// query-local sweep without manufacturing a recurrence current.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ContactOrbitParentTopologyDomain<'a> {
    current_state_template_id: u32,
    support_source_slots: &'a [u32],
    is_source: bool,
    is_current: bool,
}

impl<'a> ContactOrbitParentTopologyDomain<'a> {
    pub(super) const fn source(
        current_state_template_id: u32,
        support_source_slots: &'a [u32],
    ) -> Self {
        Self {
            current_state_template_id,
            support_source_slots,
            is_source: true,
            is_current: false,
        }
    }

    pub(super) const fn current(
        current_state_template_id: u32,
        support_source_slots: &'a [u32],
    ) -> Self {
        Self {
            current_state_template_id,
            support_source_slots,
            is_source: false,
            is_current: true,
        }
    }

    fn from_current_key(current: &'a CurrentCoreKey) -> Self {
        Self {
            current_state_template_id: current.current_state_template_id(),
            support_source_slots: current.support_source_slots(),
            is_source: current.node_kind() == RecurrenceNodeKind::Source
                && !matches!(current.source_binding(), &CurrentSourceBinding::None),
            is_current: current.node_kind() == RecurrenceNodeKind::Current
                && matches!(current.source_binding(), &CurrentSourceBinding::None),
        }
    }
}

fn try_copy_u32(values: &[u32], label: &str) -> RusticolResult<Vec<u32>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    copied.extend_from_slice(values);
    Ok(copied)
}

fn try_copy_digests(
    catalog: &TemplateCatalog<'_>,
    digest_ids: &[u32],
    label: &str,
) -> RusticolResult<Vec<SemanticDigest>> {
    let mut digests = Vec::new();
    digests
        .try_reserve_exact(digest_ids.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    for digest_id in digest_ids {
        digests.push(catalog.digest(*digest_id, label)?);
    }
    Ok(digests)
}

fn try_clone_digests(
    values: &[SemanticDigest],
    label: &str,
) -> RusticolResult<Vec<SemanticDigest>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    copied.extend_from_slice(values);
    Ok(copied)
}

fn try_copy_string(value: &str, label: &str) -> RusticolResult<String> {
    let mut copied = String::new();
    copied
        .try_reserve_exact(value.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    copied.push_str(value);
    Ok(copied)
}

fn strict_contact_orbit_step(
    input: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    transition: TransitionRow,
) -> RusticolResult<Option<ContactOrbitStepProof>> {
    let step_ids = catalog.u32_sequence(
        transition.contact_orbit_step_sequence_id,
        "transition contact-orbit steps",
    )?;
    let step_digest_ids = catalog.u32_sequence(
        transition.contact_orbit_step_semantic_digest_sequence_id,
        "transition contact-orbit step digests",
    )?;
    if step_ids.len() != step_digest_ids.len() {
        return Err(integrity(
            "transition contact-orbit step bindings are incomplete",
        ));
    }
    match step_ids {
        [] => return Ok(None),
        [_] => {}
        _ => {
            return Err(integrity(
                "certified transition must bind exactly one contact-orbit step",
            ));
        }
    }
    let step_id = step_ids[0];
    let step = input
        .contact_orbit_steps
        .get(step_id as usize)
        .copied()
        .filter(|step| step.id == step_id)
        .ok_or_else(|| integrity("transition contact-orbit step is absent"))?;
    if step.semantic_digest_id != step_digest_ids[0] {
        return Err(integrity(
            "transition contact-orbit step digest binding is stale",
        ));
    }
    let certificate = input
        .contact_orbit_certificates
        .get(step.certificate_id as usize)
        .copied()
        .filter(|certificate| certificate.id == step.certificate_id)
        .ok_or_else(|| integrity("contact-orbit certificate is absent"))?;
    let physical_leg_equivalence_classes = try_copy_u32(
        catalog.u32_sequence(
            certificate.physical_leg_equivalence_sequence_id,
            "contact-orbit physical-leg equivalence classes",
        )?,
        "contact-orbit physical-leg equivalence classes",
    )?;
    if !matches!(physical_leg_equivalence_classes.len(), 4 | 5) {
        return Err(integrity(
            "contact-orbit certificate does not describe four or five legs",
        ));
    }
    let source_particle_legs: [i32; 3] = catalog
        .i32_sequence(
            step.source_particle_leg_sequence_id,
            "contact-orbit source-particle legs",
        )?
        .try_into()
        .map_err(|_| integrity("contact-orbit step does not describe three source legs"))?;
    let proof = ContactOrbitStepProof {
        certificate_id: certificate.id,
        certificate_semantic_digest: catalog.digest(
            certificate.semantic_digest_id,
            "contact-orbit certificate semantic",
        )?,
        step_semantic_digest: catalog
            .digest(step.semantic_digest_id, "contact-orbit step semantic")?,
        stage: ContactOrbitStage::try_from(step.stage)?,
        result_leg: u32::from(step.result_leg),
        physical_leg_equivalence_classes,
        left_covered_legs: ContactOrbitCoveredLegs::new(catalog.u32_sequence(
            step.left_covered_leg_sequence_id,
            "contact-orbit left covered legs",
        )?)?,
        right_covered_legs: ContactOrbitCoveredLegs::new(catalog.u32_sequence(
            step.right_covered_leg_sequence_id,
            "contact-orbit right covered legs",
        )?)?,
        source_particle_legs,
        certificate_reconstruction_factor: catalog.factor(
            certificate.reconstruction_factor_id,
            "contact-orbit certificate reconstruction",
        )?,
        step_reconstruction_factor: catalog.factor(
            step.reconstruction_factor_id,
            "contact-orbit step reconstruction",
        )?,
        evaluator_class: try_copy_string(
            catalog.string(
                certificate.evaluator_class_string_id,
                "contact-orbit evaluator class",
            )?,
            "contact-orbit evaluator class",
        )?,
    };
    validate_step(&proof)?;
    Ok(Some(proof))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_contact_orbit_transition(
    input: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    transition: TransitionRow,
    quantum_semantic_digest: SemanticDigest,
    local_coupling_orders: &[u32],
    binding_coupling: ExactComplexRational,
    transition_exact_factor: ExactComplexRational,
    contraction_exact_factor: ExactComplexRational,
    input_exchange_factor: Option<ExactComplexRational>,
    witness_rows: &[LCColorTransitionWitnessRow],
) -> RusticolResult<Option<PreparedContactOrbitTransition>> {
    let Some(step) = strict_contact_orbit_step(input, catalog, transition)? else {
        return Ok(None);
    };
    let input_state_template_ids = <[u32; 2]>::try_from(catalog.u32_sequence(
        transition.input_state_sequence_id,
        "contact-orbit transition input states",
    )?)
    .map_err(|_| integrity("certified contact-orbit transition is not binary"))?;
    let contraction = input
        .color_contractions
        .get(transition.color_contraction_template_id as usize)
        .copied()
        .filter(|row| row.id == transition.color_contraction_template_id)
        .ok_or_else(|| integrity("contact-orbit color contraction is absent"))?;
    let binding = input
        .evaluator_bindings
        .get(transition.evaluator_binding_id as usize)
        .copied()
        .filter(|row| row.id == transition.evaluator_binding_id)
        .ok_or_else(|| integrity("contact-orbit evaluator binding is absent"))?;
    if EvaluatorContractKind::try_from(binding.contract_kind)? != EvaluatorContractKind::Vertex {
        return Err(integrity("contact-orbit evaluator binding is not a vertex"));
    }
    let _ = EvaluatorCallableKind::try_from(binding.callable_kind)?;
    let evaluator_input_layout = try_copy_u32(
        catalog.u32_sequence(
            binding.input_layout_sequence_id,
            "contact-orbit evaluator input layout",
        )?,
        "contact-orbit evaluator input layout",
    )?;
    let evaluator_output_layout = try_copy_u32(
        catalog.u32_sequence(
            binding.output_layout_sequence_id,
            "contact-orbit evaluator output layout",
        )?,
        "contact-orbit evaluator output layout",
    )?;
    let evaluator_exact_expression_digests = try_copy_digests(
        catalog,
        catalog.u32_sequence(
            binding.exact_expression_digest_sequence_id,
            "contact-orbit evaluator exact expressions",
        )?,
        "contact-orbit evaluator exact expression",
    )?;
    let coupling_orders =
        try_copy_u32(local_coupling_orders, "contact-orbit local coupling orders")?;
    let coupling_parameter_ids = try_copy_u32(
        catalog.u32_sequence(
            transition.coupling_parameter_sequence_id,
            "contact-orbit coupling parameters",
        )?,
        "contact-orbit coupling parameters",
    )?;
    let momentum_convention = try_copy_u32(
        catalog.u32_sequence(
            transition.momentum_convention_sequence_id,
            "contact-orbit momentum convention",
        )?,
        "contact-orbit momentum convention",
    )?;
    let transition_equivalence_class = try_copy_string(
        catalog.string(
            transition.equivalence_class_string_id,
            "contact-orbit transition equivalence class",
        )?,
        "contact-orbit transition equivalence class",
    )?;
    let evaluator_runtime_template = if binding.runtime_template_string_id == u32::MAX {
        None
    } else {
        Some(try_copy_string(
            catalog.string(
                binding.runtime_template_string_id,
                "contact-orbit evaluator runtime template",
            )?,
            "contact-orbit evaluator runtime template",
        )?)
    };
    let color_contraction_semantic_digest = catalog.digest(
        contraction.semantic_digest_id,
        "contact-orbit color-contraction semantic",
    )?;
    let evaluator_callable_signature = catalog.digest(
        binding.callable_signature_digest_id,
        "contact-orbit evaluator callable signature",
    )?;
    let mut applications = Vec::new();
    applications
        .try_reserve_exact(witness_rows.len())
        .map_err(|error| {
            allocation(format!(
                "contact-orbit application allocation failed: {error}"
            ))
        })?;
    for witness in witness_rows {
        if witness.color_contraction_id != contraction.id {
            return Err(integrity(
                "contact-orbit witness belongs to another color contraction",
            ));
        }
        applications.push(ContactOrbitApplicationWitness {
            transition_equivalence_class: try_copy_string(
                &transition_equivalence_class,
                "contact-orbit transition equivalence class",
            )?,
            quantum_semantic_digest,
            color_contraction_semantic_digest,
            color_witness_term_id: LCColorWitnessTermId::new(contraction.id, witness.ordinal),
            color_witness_proof_digest: catalog
                .digest(witness.proof_digest_id, "contact-orbit color witness proof")?,
            coupling_orders: try_copy_u32(&coupling_orders, "contact-orbit local coupling orders")?,
            coupling_parameter_ids: try_copy_u32(
                &coupling_parameter_ids,
                "contact-orbit coupling parameters",
            )?,
            momentum_convention: try_copy_u32(
                &momentum_convention,
                "contact-orbit momentum convention",
            )?,
            binding_coupling,
            transition_exact_factor,
            color_exact_factor: contraction_exact_factor,
            witness_exact_factor: catalog
                .factor(witness.exact_factor_id, "contact-orbit color witness")?,
            input_exchange_factor,
            output_factor_source: transition.output_factor_source,
            evaluator_contract_kind: binding.contract_kind,
            evaluator_callable_kind: binding.callable_kind,
            evaluator_callable_signature,
            evaluator_input_layout: try_copy_u32(
                &evaluator_input_layout,
                "contact-orbit evaluator input layout",
            )?,
            evaluator_output_layout: try_copy_u32(
                &evaluator_output_layout,
                "contact-orbit evaluator output layout",
            )?,
            evaluator_exact_expression_digests: try_clone_digests(
                &evaluator_exact_expression_digests,
                "contact-orbit evaluator exact expressions",
            )?,
            evaluator_runtime_template: evaluator_runtime_template
                .as_deref()
                .map(|value| try_copy_string(value, "contact-orbit evaluator runtime template"))
                .transpose()?,
            output_projection_id: transition.output_projection_string_id,
        });
    }
    Ok(Some(PreparedContactOrbitTransition {
        step,
        input_state_template_ids,
        transition_semantic_digest: catalog.digest(
            transition.semantic_digest_id,
            "contact-orbit transition semantic",
        )?,
        applications,
    }))
}

impl PreparedContactOrbitTransition {
    /// Topology-only counterpart of [`Self::accepts_parent_domain`].  It is
    /// used by the process-global on-the-fly coupling sweep, which retains no
    /// recurrence-current payload beyond the exact admission facts below.
    pub(super) fn accepts_parent_topology_domain(
        &self,
        parents: [ContactOrbitParentTopologyDomain<'_>; 2],
    ) -> RusticolResult<bool> {
        validate_step(&self.step)?;
        Ok(contact_orbit_parent_topology_mapping(
            &self.step,
            parents,
            self.input_state_template_ids,
        )?
        .is_some())
    }

    /// Whether these concrete parents belong to the certified contact step's
    /// physical construction domain.
    ///
    /// The established builder reaches this point only after its process-wide
    /// structural-demand filter, so a missing mapping remains an integrity
    /// error in [`Self::owner_candidate`].  The query-local on-the-fly sweep
    /// deliberately has no such global demand graph and uses this predicate to
    /// avoid instantiating a certified auxiliary step on wider, otherwise
    /// state-compatible currents.
    pub(super) fn accepts_parent_domain(
        &self,
        parents: [&CurrentCoreKey; 2],
    ) -> RusticolResult<bool> {
        self.accepts_parent_topology_domain(
            parents.map(ContactOrbitParentTopologyDomain::from_current_key),
        )
    }

    pub(super) fn owner_candidate<'a>(
        &'a self,
        destination: &'a CurrentCoreKey,
        parents: [&'a CurrentCoreKey; 2],
        color_witness_term_id: LCColorWitnessTermId,
    ) -> RusticolResult<ContactOrbitOwnerCandidate<'a>> {
        let mut matches = self
            .applications
            .iter()
            .filter(|application| application.color_witness_term_id == color_witness_term_id);
        let application = matches
            .next()
            .ok_or_else(|| integrity("certified contact-orbit color witness is absent"))?;
        if matches.next().is_some() {
            return Err(integrity(
                "certified contact-orbit color witness is ambiguous",
            ));
        }
        contact_orbit_owner_candidate(
            Some(&self.step),
            destination,
            parents,
            self.input_state_template_ids,
            application,
            self.transition_semantic_digest,
        )?
        .ok_or_else(|| integrity("certified contact-orbit owner candidate disappeared"))
    }
}

#[cfg(test)]
pub(super) fn prepared_contact_orbit_transition_for_test(
    step: ContactOrbitStepProof,
    input_state_template_ids: [u32; 2],
    transition_semantic_digest: SemanticDigest,
    application: ContactOrbitApplicationWitness,
) -> PreparedContactOrbitTransition {
    PreparedContactOrbitTransition {
        step,
        input_state_template_ids,
        transition_semantic_digest,
        applications: vec![application],
    }
}

#[cfg(test)]
fn contact_orbit_test_digest(byte: u8) -> SemanticDigest {
    SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
}

#[cfg(test)]
pub(super) fn contact_orbit_application_for_test() -> ContactOrbitApplicationWitness {
    ContactOrbitApplicationWitness {
        transition_equivalence_class: "constant-scalar-contact".into(),
        quantum_semantic_digest: contact_orbit_test_digest(2),
        color_contraction_semantic_digest: contact_orbit_test_digest(3),
        color_witness_term_id: LCColorWitnessTermId::new(4, 0),
        color_witness_proof_digest: contact_orbit_test_digest(5),
        coupling_orders: vec![1],
        coupling_parameter_ids: vec![2],
        momentum_convention: vec![0, 1],
        binding_coupling: ExactComplexRational::ONE,
        transition_exact_factor: ExactComplexRational::ONE,
        color_exact_factor: ExactComplexRational::ONE,
        witness_exact_factor: ExactComplexRational::ONE,
        input_exchange_factor: None,
        output_factor_source: 0,
        evaluator_contract_kind: EvaluatorContractKind::Vertex as u8,
        evaluator_callable_kind: EvaluatorCallableKind::PreparedKernel as u8,
        evaluator_callable_signature: contact_orbit_test_digest(6),
        evaluator_input_layout: vec![10, 11],
        evaluator_output_layout: vec![12],
        evaluator_exact_expression_digests: vec![contact_orbit_test_digest(7)],
        evaluator_runtime_template: None,
        output_projection_id: 8,
    }
}

#[cfg(test)]
pub(super) fn partial_contact_orbit_step_for_test(
    left: u32,
    right: u32,
    result: u32,
    digest_byte: u8,
    equivalence_classes: [u32; 4],
) -> ContactOrbitStepProof {
    ContactOrbitStepProof {
        certificate_id: 0,
        certificate_semantic_digest: contact_orbit_test_digest(9),
        step_semantic_digest: contact_orbit_test_digest(digest_byte),
        stage: ContactOrbitStage::Partial,
        result_leg: result,
        physical_leg_equivalence_classes: equivalence_classes.to_vec(),
        left_covered_legs: ContactOrbitCoveredLegs::new(&[left]).unwrap(),
        right_covered_legs: ContactOrbitCoveredLegs::new(&[right]).unwrap(),
        source_particle_legs: [left as i32, right as i32, -1],
        certificate_reconstruction_factor: ExactComplexRational::ONE,
        step_reconstruction_factor: ExactComplexRational::ONE,
        evaluator_class: "constant-scalar-contact-v1".into(),
    }
}

#[cfg(test)]
pub(super) fn final_contact_orbit_step_for_test(
    left: &[u32],
    right: &[u32],
    result: u32,
    digest_byte: u8,
    equivalence_classes: [u32; 4],
) -> ContactOrbitStepProof {
    ContactOrbitStepProof {
        certificate_id: 0,
        certificate_semantic_digest: contact_orbit_test_digest(9),
        step_semantic_digest: contact_orbit_test_digest(digest_byte),
        stage: ContactOrbitStage::Final,
        result_leg: result,
        physical_leg_equivalence_classes: equivalence_classes.to_vec(),
        left_covered_legs: ContactOrbitCoveredLegs::new(left).unwrap(),
        right_covered_legs: ContactOrbitCoveredLegs::new(right).unwrap(),
        source_particle_legs: [
            if left.len() == 2 { -1 } else { left[0] as i32 },
            if right.len() == 2 {
                -1
            } else {
                right[0] as i32
            },
            result as i32,
        ],
        certificate_reconstruction_factor: ExactComplexRational::ONE,
        step_reconstruction_factor: ExactComplexRational::ONE,
        evaluator_class: "constant-scalar-contact-v1".into(),
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PhysicalLegAssignments {
    len: u8,
    values: [u32; 2],
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitPhysicalAssignment {
    physical_leg_assignments: PhysicalLegAssignments,
    source_particle_equivalence_class: Option<u32>,
}

/// Physics identity retained while quotienting a certified contact traversal orbit.
///
/// A certified physical source leg remains bound to its concrete external
/// source slot.  Auxiliary support must also remain in the key: at five points
/// the final HEFT node can join two auxiliaries, and their support partition is
/// the physical identical-particle assignment whose multiplicity is carried by
/// the compiler-generated kernel.  Forgetting it collapses distinct Hgggg
/// assignments.  Every other current distinction stays in the owner key too.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitCurrentPhysics<'a> {
    catalog_digest: SemanticDigest,
    node_kind: RecurrenceNodeKind,
    current_state_template_id: u32,
    dynamic_lc_color_state_id: DynamicLCColorStateId,
    recurrence_strategy: RecurrenceStrategy,
    spin_state_class: i32,
    flavour_flow: &'a [i32],
    quantum_number_flow_id: u32,
    coupling_orders: &'a [u32],
    source_binding: &'a CurrentSourceBinding,
    propagator_template_id: Option<u32>,
    support_source_slots: &'a [u32],
}

impl<'a> ContactOrbitCurrentPhysics<'a> {
    fn from_current(current: &'a CurrentCoreKey) -> Self {
        Self {
            catalog_digest: current.catalog_digest(),
            node_kind: current.node_kind(),
            current_state_template_id: current.current_state_template_id(),
            dynamic_lc_color_state_id: current.dynamic_lc_color_state_id(),
            recurrence_strategy: current.helicity_identity().strategy(),
            spin_state_class: current.spin_state_class(),
            flavour_flow: current.flavour_flow(),
            quantum_number_flow_id: current.quantum_number_flow_id(),
            coupling_orders: current.coupling_orders(),
            source_binding: current.source_binding(),
            propagator_template_id: current.propagator_template_id(),
            support_source_slots: current.support_source_slots(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitParentGroup<'a> {
    current: ContactOrbitCurrentPhysics<'a>,
    assignment: ContactOrbitPhysicalAssignment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ContactOrbitParentSide {
    expected_state_template_id: u32,
    covered_legs: ContactOrbitCoveredLegs,
    source_particle_leg: i32,
    assignment: ContactOrbitPhysicalAssignment,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitOwnerGroupKey<'a> {
    certificate_id: u32,
    certificate_semantic_digest: SemanticDigest,
    stage: ContactOrbitStage,
    result_leg_equivalence_class: u32,
    destination: &'a CurrentCoreKey,
    parents: [ContactOrbitParentGroup<'a>; 2],
    output_source_particle_equivalence_class: Option<u32>,
    evaluator_class: &'a str,
    coupling_parameter_ids: &'a [u32],
    binding_coupling: ExactComplexRational,
    application_exact_factor: ExactComplexRational,
    input_exchange_factor: Option<ExactComplexRational>,
    certificate_reconstruction_factor: ExactComplexRational,
    step_reconstruction_factor: ExactComplexRational,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitOwnerRank<'a> {
    destination: &'a CurrentCoreKey,
    parents: [&'a CurrentCoreKey; 2],
    oriented_covered_legs: [ContactOrbitCoveredLegs; 2],
    oriented_source_particle_legs: [i32; 3],
    application: &'a ContactOrbitApplicationWitness,
    step_semantic_digest: SemanticDigest,
    transition_semantic_digest: SemanticDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ContactOrbitOwnerCandidate<'a> {
    group: ContactOrbitOwnerGroupKey<'a>,
    rank: ContactOrbitOwnerRank<'a>,
}

fn canonical_physical_assignment(
    covered_legs: ContactOrbitCoveredLegs,
    equivalence_classes: &[u32],
    source_particle_leg: i32,
) -> RusticolResult<ContactOrbitPhysicalAssignment> {
    let mut assignments = [0_u32; 2];
    for (index, leg) in covered_legs.values().iter().copied().enumerate() {
        assignments[index] = *equivalence_classes
            .get(leg as usize)
            .ok_or_else(|| integrity("contact-orbit covered leg is outside arity"))?;
    }
    assignments[..covered_legs.len()].sort_unstable();
    let source_particle_equivalence_class = match source_particle_leg {
        -1 => None,
        leg if leg >= 0 => Some(
            *equivalence_classes
                .get(leg as usize)
                .ok_or_else(|| integrity("contact-orbit source particle leg is outside arity"))?,
        ),
        _ => {
            return Err(integrity(
                "contact-orbit source particle leg uses an unknown sentinel",
            ));
        }
    };
    Ok(ContactOrbitPhysicalAssignment {
        physical_leg_assignments: PhysicalLegAssignments {
            len: covered_legs.len,
            values: assignments,
        },
        source_particle_equivalence_class,
    })
}

fn physical_leg_equivalence_class(
    leg: i32,
    equivalence_classes: &[u32],
    label: &str,
) -> RusticolResult<Option<u32>> {
    match leg {
        -1 => Ok(None),
        leg if leg >= 0 => equivalence_classes
            .get(leg as usize)
            .copied()
            .map(Some)
            .ok_or_else(|| integrity(format!("contact-orbit {label} is outside arity"))),
        _ => Err(integrity(format!(
            "contact-orbit {label} uses an unknown sentinel"
        ))),
    }
}

fn validate_step(step: &ContactOrbitStepProof) -> RusticolResult<()> {
    if step.certificate_reconstruction_factor != ExactComplexRational::ONE
        || step.step_reconstruction_factor != ExactComplexRational::ONE
    {
        return Err(integrity(
            "contact-orbit owner requires exact unit reconstruction",
        ));
    }
    let arity = step.physical_leg_equivalence_classes.len();
    if !matches!(arity, 4 | 5) {
        return Err(integrity(
            "contact-orbit certificate does not describe four or five legs",
        ));
    }
    let result_index = usize::try_from(step.result_leg)
        .ok()
        .filter(|index| *index < arity)
        .ok_or_else(|| integrity("contact-orbit result leg is outside arity"))?;
    let mut covered = vec![false; arity];
    for leg in step
        .left_covered_legs
        .values()
        .iter()
        .chain(step.right_covered_legs.values())
        .copied()
    {
        let index = usize::try_from(leg)
            .ok()
            .filter(|index| *index < arity)
            .ok_or_else(|| integrity("contact-orbit covered leg is outside arity"))?;
        if index == result_index || covered[index] {
            return Err(integrity("contact-orbit covered-leg partition is invalid"));
        }
        covered[index] = true;
    }
    match step.stage {
        ContactOrbitStage::Partial => {
            if step.left_covered_legs.len() != 1
                || step.right_covered_legs.len() != 1
                || step.source_particle_legs
                    != [
                        step.left_covered_legs.values()[0] as i32,
                        step.right_covered_legs.values()[0] as i32,
                        -1,
                    ]
            {
                return Err(integrity(
                    "contact-orbit partial owner lineage is inconsistent",
                ));
            }
        }
        ContactOrbitStage::Final => {
            let expected_sources = [
                if step.left_covered_legs.len() > 1 {
                    -1
                } else {
                    step.left_covered_legs.values()[0] as i32
                },
                if step.right_covered_legs.len() > 1 {
                    -1
                } else {
                    step.right_covered_legs.values()[0] as i32
                },
                step.result_leg as i32,
            ];
            if step.left_covered_legs.len() > 2
                || step.right_covered_legs.len() > 2
                || step.source_particle_legs != expected_sources
                || covered
                    .iter()
                    .enumerate()
                    .any(|(index, value)| *value != (index != result_index))
            {
                return Err(integrity(
                    "contact-orbit final owner lineage is inconsistent",
                ));
            }
        }
    }
    Ok(())
}

fn validate_support_union(
    destination: &CurrentCoreKey,
    parents: [&CurrentCoreKey; 2],
) -> RusticolResult<()> {
    let left = parents[0].support_source_slots();
    let right = parents[1].support_source_slots();
    let destination_support = destination.support_source_slots();
    if left
        .len()
        .checked_add(right.len())
        .filter(|combined| *combined == destination_support.len())
        .is_none()
    {
        return Err(integrity(
            "contact-orbit parent supports do not reproduce the destination",
        ));
    }
    let (mut left_index, mut right_index) = (0_usize, 0_usize);
    for expected in destination_support.iter().copied() {
        let next = match (left.get(left_index), right.get(right_index)) {
            (Some(left_value), Some(right_value)) => match left_value.cmp(right_value) {
                Ordering::Less => {
                    left_index += 1;
                    *left_value
                }
                Ordering::Greater => {
                    right_index += 1;
                    *right_value
                }
                Ordering::Equal => {
                    return Err(integrity("contact-orbit parent supports are not disjoint"));
                }
            },
            (Some(left_value), None) => {
                left_index += 1;
                *left_value
            }
            (None, Some(right_value)) => {
                right_index += 1;
                *right_value
            }
            (None, None) => {
                return Err(integrity(
                    "contact-orbit parent supports do not reproduce the destination",
                ));
            }
        };
        if next != expected {
            return Err(integrity(
                "contact-orbit parent supports do not reproduce the destination",
            ));
        }
    }
    Ok(())
}

fn topology_domain_matches_contact_orbit_side(
    parent: ContactOrbitParentTopologyDomain<'_>,
    side: ContactOrbitParentSide,
) -> bool {
    if parent.current_state_template_id != side.expected_state_template_id
        || parent.support_source_slots.is_empty()
    {
        return false;
    }
    match side.source_particle_leg {
        -1 => parent.is_current,
        // A physical contact leg may be fed either by an external source or by
        // an ordinary off-shell current.  Its certified leg number identifies
        // a vertex slot, not an external-process source slot.
        leg if leg >= 0 => parent.is_source || parent.is_current,
        _ => false,
    }
}

fn contact_orbit_parent_topology_mapping(
    step: &ContactOrbitStepProof,
    parents: [ContactOrbitParentTopologyDomain<'_>; 2],
    input_state_template_ids: [u32; 2],
) -> RusticolResult<Option<([ContactOrbitParentSide; 2], [usize; 2])>> {
    let sides = [
        ContactOrbitParentSide {
            expected_state_template_id: input_state_template_ids[0],
            covered_legs: step.left_covered_legs,
            source_particle_leg: step.source_particle_legs[0],
            assignment: canonical_physical_assignment(
                step.left_covered_legs,
                &step.physical_leg_equivalence_classes,
                step.source_particle_legs[0],
            )?,
        },
        ContactOrbitParentSide {
            expected_state_template_id: input_state_template_ids[1],
            covered_legs: step.right_covered_legs,
            source_particle_leg: step.source_particle_legs[1],
            assignment: canonical_physical_assignment(
                step.right_covered_legs,
                &step.physical_leg_equivalence_classes,
                step.source_particle_legs[1],
            )?,
        },
    ];
    let direct = topology_domain_matches_contact_orbit_side(parents[0], sides[0])
        && topology_domain_matches_contact_orbit_side(parents[1], sides[1]);
    let exchanged = topology_domain_matches_contact_orbit_side(parents[1], sides[0])
        && topology_domain_matches_contact_orbit_side(parents[0], sides[1]);
    let parent_by_side = match (direct, exchanged) {
        (true, false) => [0_usize, 1_usize],
        (false, true) => [1_usize, 0_usize],
        (true, true)
            if sides[0].expected_state_template_id == sides[1].expected_state_template_id
                && sides[0].assignment == sides[1].assignment =>
        {
            [0_usize, 1_usize]
        }
        (true, true) => {
            return Err(integrity(
                "contact-orbit parent-to-side mapping is ambiguous across distinct physical assignments",
            ));
        }
        (false, false) => return Ok(None),
    };
    Ok(Some((sides, parent_by_side)))
}

fn contact_orbit_parent_mapping(
    step: &ContactOrbitStepProof,
    parents: [&CurrentCoreKey; 2],
    input_state_template_ids: [u32; 2],
) -> RusticolResult<Option<([ContactOrbitParentSide; 2], [usize; 2])>> {
    contact_orbit_parent_topology_mapping(
        step,
        parents.map(ContactOrbitParentTopologyDomain::from_current_key),
        input_state_template_ids,
    )
}

pub(super) fn contact_orbit_owner_candidate<'a>(
    step: Option<&'a ContactOrbitStepProof>,
    destination: &'a CurrentCoreKey,
    parents: [&'a CurrentCoreKey; 2],
    input_state_template_ids: [u32; 2],
    application: &'a ContactOrbitApplicationWitness,
    transition_semantic_digest: SemanticDigest,
) -> RusticolResult<Option<ContactOrbitOwnerCandidate<'a>>> {
    let Some(step) = step else {
        return Ok(None);
    };
    validate_step(step)?;
    validate_support_union(destination, parents)?;
    let (sides, parent_by_side) =
        contact_orbit_parent_mapping(step, parents, input_state_template_ids)?
            .ok_or_else(|| integrity("contact-orbit parent-to-side mapping is absent"))?;
    let mut canonical_parents = [
        ContactOrbitParentGroup {
            current: ContactOrbitCurrentPhysics::from_current(parents[parent_by_side[0]]),
            assignment: sides[0].assignment,
        },
        ContactOrbitParentGroup {
            current: ContactOrbitCurrentPhysics::from_current(parents[parent_by_side[1]]),
            assignment: sides[1].assignment,
        },
    ];
    canonical_parents.sort_unstable();
    let result_leg_equivalence_class =
        step.physical_leg_equivalence_classes[step.result_leg as usize];
    let application_exact_factor = application
        .transition_exact_factor
        .checked_mul(application.color_exact_factor)?
        .checked_mul(application.witness_exact_factor)?;
    Ok(Some(ContactOrbitOwnerCandidate {
        group: ContactOrbitOwnerGroupKey {
            certificate_id: step.certificate_id,
            certificate_semantic_digest: step.certificate_semantic_digest,
            stage: step.stage,
            result_leg_equivalence_class,
            destination,
            parents: canonical_parents,
            output_source_particle_equivalence_class: physical_leg_equivalence_class(
                step.source_particle_legs[2],
                &step.physical_leg_equivalence_classes,
                "output source particle leg",
            )?,
            evaluator_class: &step.evaluator_class,
            coupling_parameter_ids: &application.coupling_parameter_ids,
            binding_coupling: application.binding_coupling,
            application_exact_factor,
            input_exchange_factor: application.input_exchange_factor,
            certificate_reconstruction_factor: step.certificate_reconstruction_factor,
            step_reconstruction_factor: step.step_reconstruction_factor,
        },
        rank: ContactOrbitOwnerRank {
            destination,
            parents,
            oriented_covered_legs: [step.left_covered_legs, step.right_covered_legs],
            oriented_source_particle_legs: step.source_particle_legs,
            application,
            step_semantic_digest: step.step_semantic_digest,
            transition_semantic_digest,
        },
    }))
}

fn compare_items<T: Ord>(
    left: &(T, Option<ContactOrbitOwnerCandidate<'_>>),
    right: &(T, Option<ContactOrbitOwnerCandidate<'_>>),
) -> Ordering {
    match (&left.1, &right.1) {
        (None, None) => left.0.cmp(&right.0),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left_candidate), Some(right_candidate)) => left_candidate
            .group
            .cmp(&right_candidate.group)
            .then_with(|| left_candidate.rank.cmp(&right_candidate.rank))
            .then_with(|| left.0.cmp(&right.0)),
    }
}

/// Select one exact owner per certified group without mutating caller state.
///
/// The exact-size iterator is copied into one fallibly reserved scratch vector,
/// then sorted in place. The returned token vector is separately reserved
/// before any selection, so every allocation failure occurs before caller-owned
/// construction state can be replaced.
pub(super) fn selected_contact_orbit_owner_tokens<'a, T, I>(items: I) -> RusticolResult<Vec<T>>
where
    T: Ord,
    I: IntoIterator<Item = (T, Option<ContactOrbitOwnerCandidate<'a>>)>,
    I::IntoIter: ExactSizeIterator,
{
    let iterator = items.into_iter();
    let item_count = iterator.len();
    item_count
        .checked_mul(size_of::<(T, Option<ContactOrbitOwnerCandidate<'a>>)>())
        .ok_or_else(|| integrity("contact-orbit owner scratch byte size overflows usize"))?;
    let mut scratch = Vec::new();
    scratch.try_reserve_exact(item_count).map_err(|error| {
        allocation(format!(
            "contact-orbit owner scratch allocation failed: {error}"
        ))
    })?;
    for item in iterator {
        if scratch.len() == item_count {
            return Err(integrity(
                "contact-orbit exact-size owner iterator exceeded its declared length",
            ));
        }
        scratch.push(item);
    }
    if scratch.len() != item_count {
        return Err(integrity(
            "contact-orbit exact-size owner iterator changed length",
        ));
    }

    let mut selected = Vec::new();
    selected.try_reserve_exact(item_count).map_err(|error| {
        allocation(format!(
            "contact-orbit selected-owner allocation failed: {error}"
        ))
    })?;
    scratch.sort_unstable_by(compare_items);
    let mut iterator = scratch.into_iter().peekable();
    while let Some((token, candidate)) = iterator.next() {
        let Some(candidate) = candidate else {
            selected.push(token);
            continue;
        };
        while let Some((next_token, Some(next_candidate))) = iterator.peek() {
            if next_candidate.group != candidate.group {
                break;
            }
            if next_candidate.rank == candidate.rank && next_token != &token {
                return Err(integrity(
                    "contact-orbit owner candidates have a conflicting exact rank",
                ));
            }
            iterator.next();
        }
        selected.push(token);
    }
    selected.sort_unstable();
    if selected.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(integrity(
            "contact-orbit owner selection produced duplicate tokens",
        ));
    }
    Ok(selected)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ContactOrbitTestBinding {
    None,
    One,
    Two,
    MissingStep,
    DigestMismatch,
}

#[cfg(test)]
pub(super) fn contact_orbit_test_template(
    binding: ContactOrbitTestBinding,
) -> OwnedRecurrenceTemplateInput {
    use super::{
        template::{ContactOrbitCertificateRow, ContactOrbitStepRow, DigestCatalogRow},
        validated_template_fixture,
    };

    fn string_id(input: &OwnedRecurrenceTemplateInput, value: &str) -> u32 {
        input
            .string_ranges
            .iter()
            .position(|range| {
                let range = range
                    .as_usize_range(input.string_bytes.len(), "test string")
                    .unwrap();
                input.string_bytes[range] == *value.as_bytes()
            })
            .unwrap() as u32
    }

    fn u32_sequence_id(input: &OwnedRecurrenceTemplateInput, values: &[u32]) -> u32 {
        input
            .u32_sequence_ranges
            .iter()
            .position(|row| {
                let range = row
                    .range
                    .as_usize_range(input.u32_sequence_values.len(), "test u32 sequence")
                    .unwrap();
                &input.u32_sequence_values[range] == values
            })
            .unwrap() as u32
    }

    fn i32_sequence_id(input: &OwnedRecurrenceTemplateInput, values: &[i32]) -> u32 {
        input
            .i32_sequence_ranges
            .iter()
            .position(|row| {
                let range = row
                    .range
                    .as_usize_range(input.i32_sequence_values.len(), "test i32 sequence")
                    .unwrap();
                &input.i32_sequence_values[range] == values
            })
            .unwrap() as u32
    }

    fn append_digest(input: &mut OwnedRecurrenceTemplateInput, byte: u8) -> u32 {
        let id = input.digest_catalog.len() as u32;
        input.digest_catalog.push(DigestCatalogRow {
            id,
            value: [byte; 32],
        });
        id
    }

    let mut input = validated_template_fixture().into_input();
    if binding == ContactOrbitTestBinding::None {
        return input;
    }

    let algorithm = string_id(&input, "compiler-certified-contact-orbit");
    let evaluator_class = string_id(
        &input,
        "constant-scalar-literal-singlet-self-conjugate-boson",
    );
    let certificate_template = string_id(&input, "any");
    let first_step_template = string_id(&input, "basis");
    let second_step_template = string_id(&input, "component");
    let particle = string_id(&input, "scalar");
    let vertex = particle;
    let particles = u32_sequence_id(&input, &[particle, particle, particle, particle]);
    let equivalence = u32_sequence_id(&input, &[0, 0, 0, 0]);
    let left_first = u32_sequence_id(&input, &[0]);
    let right_first = u32_sequence_id(&input, &[1]);
    let left_second = right_first;
    let right_second = left_first;
    let source_first = i32_sequence_id(&input, &[0, 1, -1]);
    let source_second = i32_sequence_id(&input, &[1, 0, -1]);
    let certificate_digest = append_digest(&mut input, 26);
    let first_step_digest = append_digest(&mut input, 27);
    let second_step_digest = append_digest(&mut input, 28);
    input.contact_orbit_certificates = vec![ContactOrbitCertificateRow {
        id: 0,
        template_string_id: certificate_template,
        algorithm_string_id: algorithm,
        algorithm_version: 2,
        term_id: 0,
        vertex_string_id: vertex,
        particle_string_sequence_id: particles,
        evaluator_class_string_id: evaluator_class,
        physical_leg_equivalence_sequence_id: equivalence,
        reconstruction_factor_id: 0,
        semantic_digest_id: certificate_digest,
    }];
    input.contact_orbit_steps = vec![
        ContactOrbitStepRow {
            id: 0,
            template_string_id: first_step_template,
            certificate_id: 0,
            stage: ContactOrbitStage::Partial as u8,
            result_leg: 2,
            left_covered_leg_sequence_id: left_first,
            right_covered_leg_sequence_id: right_first,
            source_particle_leg_sequence_id: source_first,
            reconstruction_factor_id: 0,
            semantic_digest_id: first_step_digest,
        },
        ContactOrbitStepRow {
            id: 1,
            template_string_id: second_step_template,
            certificate_id: 0,
            stage: ContactOrbitStage::Partial as u8,
            result_leg: 2,
            left_covered_leg_sequence_id: left_second,
            right_covered_leg_sequence_id: right_second,
            source_particle_leg_sequence_id: source_second,
            reconstruction_factor_id: 0,
            semantic_digest_id: second_step_digest,
        },
    ];
    input.catalog_header[0].contact_orbit_certificate_count = 1;
    input.catalog_header[0].contact_orbit_step_count = 2;
    let (step_ids, digest_ids) = match binding {
        ContactOrbitTestBinding::None => unreachable!(),
        ContactOrbitTestBinding::One => (vec![0], vec![first_step_digest]),
        ContactOrbitTestBinding::Two => (vec![0, 1], vec![first_step_digest, second_step_digest]),
        ContactOrbitTestBinding::MissingStep => (vec![2], vec![first_step_digest]),
        ContactOrbitTestBinding::DigestMismatch => (vec![0], vec![second_step_digest]),
    };
    input.transitions[0].contact_orbit_step_sequence_id = u32_sequence_id(&input, &step_ids);
    input.transitions[0].contact_orbit_step_semantic_digest_sequence_id =
        u32_sequence_id(&input, &digest_ids);
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        CanonicalMomentumLinearForm, CurrentHelicityIdentity, CurrentSourceBinding,
        DynamicLCColorStateId, ExactRational, MomentumTerm, RecurrenceNodeKind,
        SourceStateAssignment,
    };

    fn digest(value: u8) -> SemanticDigest {
        SemanticDigest::new([value; 32]).unwrap()
    }

    fn exact(value: i128) -> ExactComplexRational {
        ExactComplexRational::new(ExactRational::new(value, 1).unwrap(), ExactRational::ZERO)
    }

    fn exact_ratio(numerator: i128, denominator: i128) -> ExactComplexRational {
        ExactComplexRational::new(
            ExactRational::new(numerator, denominator).unwrap(),
            ExactRational::ZERO,
        )
    }

    fn current(state: u32, color: u32, support: &[u32]) -> CurrentCoreKey {
        let is_source = support.len() == 1;
        current_with_physics(
            if is_source {
                RecurrenceNodeKind::Source
            } else {
                RecurrenceNodeKind::Current
            },
            state,
            color,
            support,
            if is_source {
                RecurrenceStrategy::TopologyReplay
            } else {
                RecurrenceStrategy::AllFlowUnion
            },
            0,
            vec![state as i32],
            0,
            vec![0],
            if is_source {
                CurrentSourceBinding::FixedTemplate(support[0])
            } else {
                CurrentSourceBinding::None
            },
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn current_with_physics(
        node_kind: RecurrenceNodeKind,
        state: u32,
        color: u32,
        support: &[u32],
        strategy: RecurrenceStrategy,
        spin_state_class: i32,
        flavour_flow: Vec<i32>,
        quantum_number_flow_id: u32,
        coupling_orders: Vec<u32>,
        source_binding: CurrentSourceBinding,
        propagator_template_id: Option<u32>,
    ) -> CurrentCoreKey {
        let helicity_identity = match strategy {
            RecurrenceStrategy::TopologyReplay => CurrentHelicityIdentity::topology_replay(
                spin_state_class,
                support
                    .iter()
                    .copied()
                    .map(|source_slot| SourceStateAssignment::new(source_slot, 0))
                    .collect(),
            )
            .unwrap(),
            RecurrenceStrategy::AllFlowUnion => {
                CurrentHelicityIdentity::all_flow_union(spin_state_class)
            }
            RecurrenceStrategy::ContractedColorUnion => {
                CurrentHelicityIdentity::contracted_color_union(
                    spin_state_class,
                    support
                        .iter()
                        .copied()
                        .map(|source_slot| SourceStateAssignment::new(source_slot, 0))
                        .collect(),
                )
                .unwrap()
            }
        };
        CurrentCoreKey::new(
            digest(1),
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
            helicity_identity,
            flavour_flow,
            quantum_number_flow_id,
            coupling_orders,
            source_binding,
            propagator_template_id,
        )
        .unwrap()
    }

    fn topology_current(state: u32, color: u32, support: &[u32]) -> CurrentCoreKey {
        let is_source = support.len() == 1;
        current_with_physics(
            if is_source {
                RecurrenceNodeKind::Source
            } else {
                RecurrenceNodeKind::Current
            },
            state,
            color,
            support,
            RecurrenceStrategy::TopologyReplay,
            0,
            vec![state as i32],
            0,
            vec![0],
            if is_source {
                CurrentSourceBinding::FixedTemplate(0)
            } else {
                CurrentSourceBinding::None
            },
            None,
        )
    }

    fn topology_source(source_slot: u32, source_template_id: u32) -> CurrentCoreKey {
        current_with_physics(
            RecurrenceNodeKind::Source,
            1,
            1,
            &[source_slot],
            RecurrenceStrategy::TopologyReplay,
            0,
            vec![1],
            0,
            vec![0],
            CurrentSourceBinding::FixedTemplate(source_template_id),
            None,
        )
    }

    fn application() -> ContactOrbitApplicationWitness {
        ContactOrbitApplicationWitness {
            transition_equivalence_class: "constant-scalar-contact".into(),
            quantum_semantic_digest: digest(2),
            color_contraction_semantic_digest: digest(3),
            color_witness_term_id: LCColorWitnessTermId::new(4, 0),
            color_witness_proof_digest: digest(5),
            coupling_orders: vec![1],
            coupling_parameter_ids: vec![2],
            momentum_convention: vec![0, 1],
            binding_coupling: ExactComplexRational::ONE,
            transition_exact_factor: ExactComplexRational::ONE,
            color_exact_factor: ExactComplexRational::ONE,
            witness_exact_factor: ExactComplexRational::ONE,
            input_exchange_factor: None,
            output_factor_source: 0,
            evaluator_contract_kind: EvaluatorContractKind::Vertex as u8,
            evaluator_callable_kind: EvaluatorCallableKind::PreparedKernel as u8,
            evaluator_callable_signature: digest(6),
            evaluator_input_layout: vec![10, 11],
            evaluator_output_layout: vec![12],
            evaluator_exact_expression_digests: vec![digest(7)],
            evaluator_runtime_template: None,
            output_projection_id: 8,
        }
    }

    fn prepare_test_contact_orbit(
        input: &OwnedRecurrenceTemplateInput,
    ) -> RusticolResult<Option<PreparedContactOrbitTransition>> {
        let catalog = TemplateCatalog::new(input)?;
        let transition = input.transitions[0];
        let quantum = input.quantum_flows[transition.quantum_flow_template_id as usize];
        let contraction =
            input.color_contractions[transition.color_contraction_template_id as usize];
        prepare_contact_orbit_transition(
            input,
            &catalog,
            transition,
            catalog.digest(quantum.semantic_digest_id, "test quantum semantic")?,
            &catalog.coupling_orders(transition.coupling_order_set_id)?,
            catalog.factor(
                transition.binding_coupling_factor_id,
                "test binding coupling",
            )?,
            catalog.factor(transition.exact_factor_id, "test transition exact")?,
            catalog.factor(
                contraction.exact_coefficient_factor_id,
                "test contraction exact",
            )?,
            None,
            catalog.witness_rows(transition.color_contraction_template_id)?,
        )
    }

    #[test]
    fn strict_prepared_contact_orbit_decode_is_zero_or_one_and_fail_closed() {
        let none = contact_orbit_test_template(ContactOrbitTestBinding::None);
        assert!(prepare_test_contact_orbit(&none).unwrap().is_none());

        let one = contact_orbit_test_template(ContactOrbitTestBinding::One)
            .validate()
            .unwrap();
        assert!(prepare_test_contact_orbit(one.input()).unwrap().is_some());

        let two = contact_orbit_test_template(ContactOrbitTestBinding::Two)
            .validate()
            .unwrap();
        assert!(
            prepare_test_contact_orbit(two.input())
                .unwrap_err()
                .to_string()
                .contains("exactly one contact-orbit step")
        );

        let missing = contact_orbit_test_template(ContactOrbitTestBinding::MissingStep);
        assert!(
            prepare_test_contact_orbit(&missing)
                .unwrap_err()
                .to_string()
                .contains("contact-orbit step is absent")
        );

        let mismatch = contact_orbit_test_template(ContactOrbitTestBinding::DigestMismatch);
        assert!(
            prepare_test_contact_orbit(&mismatch)
                .unwrap_err()
                .to_string()
                .contains("digest binding is stale")
        );
    }

    fn partial_step(left: u32, right: u32, result: u32, digest_byte: u8) -> ContactOrbitStepProof {
        ContactOrbitStepProof {
            certificate_id: 0,
            certificate_semantic_digest: digest(9),
            step_semantic_digest: digest(digest_byte),
            stage: ContactOrbitStage::Partial,
            result_leg: result,
            physical_leg_equivalence_classes: vec![0, 0, 0, 0],
            left_covered_legs: ContactOrbitCoveredLegs::new(&[left]).unwrap(),
            right_covered_legs: ContactOrbitCoveredLegs::new(&[right]).unwrap(),
            source_particle_legs: [left as i32, right as i32, -1],
            certificate_reconstruction_factor: ExactComplexRational::ONE,
            step_reconstruction_factor: ExactComplexRational::ONE,
            evaluator_class: "constant-scalar-contact-v1".into(),
        }
    }

    fn final_step(
        left: &[u32],
        right: &[u32],
        result: u32,
        digest_byte: u8,
        equivalence_classes: [u32; 4],
    ) -> ContactOrbitStepProof {
        ContactOrbitStepProof {
            certificate_id: 0,
            certificate_semantic_digest: digest(9),
            step_semantic_digest: digest(digest_byte),
            stage: ContactOrbitStage::Final,
            result_leg: result,
            physical_leg_equivalence_classes: equivalence_classes.to_vec(),
            left_covered_legs: ContactOrbitCoveredLegs::new(left).unwrap(),
            right_covered_legs: ContactOrbitCoveredLegs::new(right).unwrap(),
            source_particle_legs: [
                if left.len() == 2 { -1 } else { left[0] as i32 },
                if right.len() == 2 {
                    -1
                } else {
                    right[0] as i32
                },
                result as i32,
            ],
            certificate_reconstruction_factor: ExactComplexRational::ONE,
            step_reconstruction_factor: ExactComplexRational::ONE,
            evaluator_class: "constant-scalar-contact-v1".into(),
        }
    }

    fn five_leg_final_step(
        left: &[u32],
        right: &[u32],
        result: u32,
        digest_byte: u8,
    ) -> ContactOrbitStepProof {
        ContactOrbitStepProof {
            certificate_id: 0,
            certificate_semantic_digest: digest(9),
            step_semantic_digest: digest(digest_byte),
            stage: ContactOrbitStage::Final,
            result_leg: result,
            physical_leg_equivalence_classes: vec![0, 0, 0, 0, 1],
            left_covered_legs: ContactOrbitCoveredLegs::new(left).unwrap(),
            right_covered_legs: ContactOrbitCoveredLegs::new(right).unwrap(),
            source_particle_legs: [-1, -1, result as i32],
            certificate_reconstruction_factor: ExactComplexRational::ONE,
            step_reconstruction_factor: ExactComplexRational::ONE,
            evaluator_class: "scalar-heft-colored-contact-tree".into(),
        }
    }

    fn candidate<'a>(
        step: &'a ContactOrbitStepProof,
        destination: &'a CurrentCoreKey,
        parents: [&'a CurrentCoreKey; 2],
        application: &'a ContactOrbitApplicationWitness,
        transition_digest: u8,
    ) -> ContactOrbitOwnerCandidate<'a> {
        contact_orbit_owner_candidate(
            Some(step),
            destination,
            parents,
            parents.map(|parent| parent.current_state_template_id()),
            application,
            digest(transition_digest),
        )
        .unwrap()
        .unwrap()
    }

    fn candidate_with_input_states<'a>(
        step: &'a ContactOrbitStepProof,
        destination: &'a CurrentCoreKey,
        parents: [&'a CurrentCoreKey; 2],
        input_state_template_ids: [u32; 2],
        application: &'a ContactOrbitApplicationWitness,
        transition_digest: u8,
    ) -> RusticolResult<Option<ContactOrbitOwnerCandidate<'a>>> {
        contact_orbit_owner_candidate(
            Some(step),
            destination,
            parents,
            input_state_template_ids,
            application,
            digest(transition_digest),
        )
    }

    #[test]
    fn owner_group_forgets_only_left_right_traversal() {
        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let application = application();
        let forward_step = partial_step(0, 1, 2, 10);
        let reverse_step = partial_step(1, 0, 2, 11);
        let forward = candidate(
            &forward_step,
            &destination,
            [&left, &right],
            &application,
            12,
        );
        let reverse = candidate(
            &reverse_step,
            &destination,
            [&right, &left],
            &application,
            13,
        );

        assert_eq!(forward.group, reverse.group);
        assert_ne!(forward.rank, reverse.rank);
        assert_eq!(
            selected_contact_orbit_owner_tokens([(1_u32, Some(reverse)), (0_u32, Some(forward)),])
                .unwrap(),
            vec![0],
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens([(0_u32, Some(forward)), (1_u32, Some(reverse)),])
                .unwrap(),
            vec![0],
        );
    }

    #[test]
    fn parent_assignment_binds_evaluator_swaps_to_certified_final_sides() {
        let pair = current(20, 2, &[10, 11]);
        let source = current(30, 1, &[12]);
        let destination = current(40, 3, &[10, 11, 12]);
        let application = application();
        let forward_step = final_step(&[0, 1], &[2], 3, 20, [0, 0, 1, 2]);
        let mirror_step = final_step(&[2], &[0, 1], 3, 21, [0, 0, 1, 2]);

        let forward = candidate_with_input_states(
            &forward_step,
            &destination,
            [&source, &pair],
            [20, 30],
            &application,
            22,
        )
        .unwrap()
        .unwrap();
        let mirror = candidate_with_input_states(
            &mirror_step,
            &destination,
            [&pair, &source],
            [30, 20],
            &application,
            23,
        )
        .unwrap()
        .unwrap();

        assert_eq!(forward.group, mirror.group);
        assert_ne!(forward.rank, mirror.rank);
    }

    #[test]
    fn parent_assignment_handles_distinct_states_and_identical_ambiguity() {
        let left = current(10, 1, &[10]);
        let right = current(20, 1, &[11]);
        let destination = current(30, 2, &[10, 11]);
        let application = application();
        let distinct_step = partial_step(0, 1, 2, 24);
        let direct = candidate_with_input_states(
            &distinct_step,
            &destination,
            [&left, &right],
            [10, 20],
            &application,
            25,
        )
        .unwrap()
        .unwrap();
        let swapped = candidate_with_input_states(
            &distinct_step,
            &destination,
            [&right, &left],
            [10, 20],
            &application,
            26,
        )
        .unwrap()
        .unwrap();
        assert_eq!(direct.group, swapped.group);

        let identical_right = current(10, 1, &[11]);
        let identical_step = partial_step(0, 1, 2, 27);
        assert!(
            candidate_with_input_states(
                &identical_step,
                &destination,
                [&left, &identical_right],
                [10, 10],
                &application,
                28,
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn parent_assignment_rejects_ambiguous_or_invalid_side_semantics() {
        let left = current(10, 1, &[10]);
        let right = current(10, 1, &[11]);
        let destination = current(30, 2, &[10, 11]);
        let application = application();
        let mut non_equivalent = partial_step(0, 1, 2, 29);
        non_equivalent.physical_leg_equivalence_classes = vec![0, 1, 2, 3];
        assert!(
            candidate_with_input_states(
                &non_equivalent,
                &destination,
                [&left, &right],
                [10, 10],
                &application,
                30,
            )
            .unwrap_err()
            .to_string()
            .contains("mapping is ambiguous")
        );

        let step = partial_step(0, 1, 2, 31);
        assert!(
            candidate_with_input_states(
                &step,
                &destination,
                [&left, &right],
                [90, 91],
                &application,
                32,
            )
            .unwrap_err()
            .to_string()
            .contains("mapping is absent")
        );
        let internal_leg = current_with_physics(
            RecurrenceNodeKind::Current,
            10,
            1,
            &[10],
            RecurrenceStrategy::TopologyReplay,
            0,
            vec![10],
            0,
            vec![0],
            CurrentSourceBinding::None,
            None,
        );
        assert!(
            candidate_with_input_states(
                &step,
                &destination,
                [&internal_leg, &right],
                [10, 10],
                &application,
                33,
            )
            .unwrap()
            .is_some()
        );
        let pair = current(10, 1, &[11, 12]);
        let wide_destination = current(30, 2, &[10, 11, 12]);
        assert!(
            candidate_with_input_states(
                &step,
                &wide_destination,
                [&left, &pair],
                [10, 10],
                &application,
                34,
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn prepared_parent_domain_accepts_sources_and_internal_physical_legs() {
        let source_a = current(10, 1, &[10]);
        let source_b = current(10, 1, &[11]);
        let source_c = current(10, 1, &[12]);
        let composite_scalar = current(10, 1, &[10, 11]);
        let auxiliary_pair = current(20, 1, &[10, 11]);

        let partial = prepared_contact_orbit_transition_for_test(
            partial_step(0, 1, 2, 35),
            [10, 10],
            digest(36),
            application(),
        );
        assert!(
            partial
                .accepts_parent_domain([&source_a, &source_b])
                .unwrap()
        );
        assert!(
            partial
                .accepts_parent_domain([&composite_scalar, &source_c])
                .unwrap()
        );
        for parents in [[&source_a, &source_b], [&composite_scalar, &source_c]] {
            assert_eq!(
                partial.accepts_parent_domain(parents).unwrap(),
                partial
                    .accepts_parent_topology_domain(
                        parents.map(ContactOrbitParentTopologyDomain::from_current_key),
                    )
                    .unwrap(),
                "query-local and process-global topology admission diverged"
            );
        }

        let final_transition = prepared_contact_orbit_transition_for_test(
            final_step(&[0, 1], &[2], 3, 37, [0, 0, 0, 0]),
            [20, 10],
            digest(38),
            application(),
        );
        assert!(
            final_transition
                .accepts_parent_domain([&auxiliary_pair, &source_c])
                .unwrap()
        );

        let mut ambiguous_step = partial_step(0, 1, 2, 39);
        ambiguous_step.physical_leg_equivalence_classes = vec![0, 1, 2, 3];
        let ambiguous = prepared_contact_orbit_transition_for_test(
            ambiguous_step,
            [10, 10],
            digest(40),
            application(),
        );
        assert!(
            ambiguous
                .accepts_parent_domain([&source_a, &source_b])
                .unwrap_err()
                .to_string()
                .contains("mapping is ambiguous")
        );
    }

    #[test]
    fn certified_four_identical_scalar_channels_keep_assignment_multiplicity() {
        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let application = application();
        let forward_steps = [
            partial_step(0, 1, 2, 20),
            partial_step(0, 2, 1, 21),
            partial_step(0, 3, 1, 22),
        ];
        let reverse_steps = [
            partial_step(1, 0, 2, 30),
            partial_step(2, 0, 1, 31),
            partial_step(3, 0, 1, 32),
        ];
        let candidates = [
            (
                0_u32,
                Some(candidate(
                    &forward_steps[0],
                    &destination,
                    [&left, &right],
                    &application,
                    40,
                )),
            ),
            (
                1_u32,
                Some(candidate(
                    &reverse_steps[0],
                    &destination,
                    [&right, &left],
                    &application,
                    41,
                )),
            ),
            (
                2_u32,
                Some(candidate(
                    &forward_steps[1],
                    &destination,
                    [&left, &right],
                    &application,
                    42,
                )),
            ),
            (
                3_u32,
                Some(candidate(
                    &reverse_steps[1],
                    &destination,
                    [&right, &left],
                    &application,
                    43,
                )),
            ),
            (
                4_u32,
                Some(candidate(
                    &forward_steps[2],
                    &destination,
                    [&left, &right],
                    &application,
                    44,
                )),
            ),
            (
                5_u32,
                Some(candidate(
                    &reverse_steps[2],
                    &destination,
                    [&right, &left],
                    &application,
                    45,
                )),
            ),
        ];
        assert_eq!(
            candidates
                .iter()
                .map(|(_, value)| value.unwrap().group)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1,
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens(candidates).unwrap(),
            vec![0],
        );

        let singles = [
            current(3, 3, &[10]),
            current(3, 3, &[11]),
            current(3, 3, &[12]),
        ];
        let pairs = [
            current(4, 4, &[11, 12]),
            current(4, 4, &[10, 12]),
            current(4, 4, &[10, 11]),
        ];
        let final_destination = current(5, 5, &[10, 11, 12]);
        let final_forward = final_step(&[3], &[1, 2], 0, 60, [0, 0, 0, 0]);
        let final_reverse = final_step(&[1, 2], &[3], 0, 70, [0, 0, 0, 0]);
        let final_candidates = singles
            .iter()
            .zip(pairs.iter())
            .enumerate()
            .flat_map(|(index, (single, pair))| {
                let index = u32::try_from(index).unwrap();
                [
                    (
                        2 * index,
                        Some(candidate(
                            &final_forward,
                            &final_destination,
                            [single, pair],
                            &application,
                            80 + index as u8,
                        )),
                    ),
                    (
                        2 * index + 1,
                        Some(candidate(
                            &final_reverse,
                            &final_destination,
                            [pair, single],
                            &application,
                            90 + index as u8,
                        )),
                    ),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(
            final_candidates[0].1.unwrap().group,
            final_candidates[1].1.unwrap().group
        );
        assert_eq!(
            final_candidates[2].1.unwrap().group,
            final_candidates[3].1.unwrap().group
        );
        assert_eq!(
            final_candidates[4].1.unwrap().group,
            final_candidates[5].1.unwrap().group
        );
        assert_ne!(
            final_candidates[0].1.unwrap().group,
            final_candidates[2].1.unwrap().group
        );
        assert_ne!(
            final_candidates[0].1.unwrap().group,
            final_candidates[4].1.unwrap().group
        );
        assert_ne!(
            final_candidates[2].1.unwrap().group,
            final_candidates[4].1.unwrap().group
        );
        let forward_selection =
            selected_contact_orbit_owner_tokens(final_candidates.iter().copied()).unwrap();
        let reverse_selection =
            selected_contact_orbit_owner_tokens(final_candidates.iter().rev().copied()).unwrap();
        assert_eq!(forward_selection, reverse_selection);
        assert_eq!(forward_selection.len(), 3,);
    }

    #[test]
    fn five_leg_auxiliary_partitions_keep_physical_assignment_multiplicity() {
        let application = application();
        let destination = current(40, 4, &[10, 11, 12, 13]);
        let first_left = current(20, 2, &[10, 11]);
        let first_right = current(30, 2, &[12, 13]);
        let second_left = current(20, 2, &[10, 12]);
        let second_right = current(30, 2, &[11, 13]);
        let forward_step = five_leg_final_step(&[0, 1], &[2, 3], 4, 100);
        let mirror_step = five_leg_final_step(&[2, 3], &[0, 1], 4, 101);

        let first = candidate_with_input_states(
            &forward_step,
            &destination,
            [&first_left, &first_right],
            [20, 30],
            &application,
            102,
        )
        .unwrap()
        .unwrap();
        let first_mirror = candidate_with_input_states(
            &mirror_step,
            &destination,
            [&first_right, &first_left],
            [30, 20],
            &application,
            103,
        )
        .unwrap()
        .unwrap();
        let second = candidate_with_input_states(
            &forward_step,
            &destination,
            [&second_left, &second_right],
            [20, 30],
            &application,
            104,
        )
        .unwrap()
        .unwrap();

        assert_eq!(first.group, first_mirror.group);
        assert_ne!(first.group, second.group);
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (0_u32, Some(first)),
                (1_u32, Some(first_mirror)),
                (2_u32, Some(second)),
            ])
            .unwrap(),
            vec![0, 2],
        );
    }

    #[test]
    fn certified_owner_preserves_concrete_destination_across_contact_stages() {
        let application = application();
        let partial = partial_step(0, 1, 2, 10);
        let first_left = topology_current(1, 1, &[10]);
        let first_right = topology_current(1, 1, &[11]);
        let first_destination = topology_current(2, 2, &[10, 11]);
        let second_left = topology_current(1, 1, &[20]);
        let second_right = topology_current(1, 1, &[21]);
        let second_destination = topology_current(2, 2, &[20, 21]);
        let first = candidate(
            &partial,
            &first_destination,
            [&first_left, &first_right],
            &application,
            12,
        );
        let second = candidate(
            &partial,
            &second_destination,
            [&second_left, &second_right],
            &application,
            12,
        );

        assert_ne!(first.group, second.group);
        assert_ne!(first.rank, second.rank);
        assert_eq!(
            selected_contact_orbit_owner_tokens([(0_u32, Some(first)), (1_u32, Some(second)),])
                .unwrap(),
            vec![0, 1],
        );

        let final_step = final_step(&[0], &[1, 2], 3, 20, [0, 0, 0, 0]);
        let first_pair = topology_current(3, 3, &[11, 12]);
        let first_final_destination = topology_current(4, 4, &[10, 11, 12]);
        let second_pair = topology_current(3, 3, &[21, 22]);
        let second_final_destination = topology_current(4, 4, &[20, 21, 22]);
        let first_final = candidate(
            &final_step,
            &first_final_destination,
            [&first_left, &first_pair],
            &application,
            22,
        );
        let second_final = candidate(
            &final_step,
            &second_final_destination,
            [&second_left, &second_pair],
            &application,
            22,
        );

        assert_ne!(first_final.group, second_final.group);
        assert_ne!(first_final.rank, second_final.rank);
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (0_u32, Some(first_final)),
                (1_u32, Some(second_final)),
            ])
            .unwrap(),
            vec![0, 1],
        );
    }

    #[test]
    fn same_order_singleton_rows_collapse_partial_and_final_scalar_0012_aliases() {
        let scalar0_left = current(10, 1, &[10]);
        let scalar0_right = current(10, 1, &[11]);
        let partial_destination = current(20, 2, &[10, 11]);
        let application = application();
        let mut partial_forward = partial_step(0, 1, 2, 50);
        partial_forward.physical_leg_equivalence_classes = vec![0, 0, 1, 2];
        let mut partial_reverse = partial_step(1, 0, 2, 51);
        partial_reverse.physical_leg_equivalence_classes = vec![0, 0, 1, 2];
        let partial_forward = candidate(
            &partial_forward,
            &partial_destination,
            [&scalar0_left, &scalar0_right],
            &application,
            52,
        );
        let partial_reverse = candidate(
            &partial_reverse,
            &partial_destination,
            [&scalar0_right, &scalar0_left],
            &application,
            53,
        );
        assert_eq!(partial_forward.group, partial_reverse.group);
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (0_u32, Some(partial_forward)),
                (1_u32, Some(partial_reverse)),
            ])
            .unwrap(),
            vec![0],
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (1_u32, Some(partial_reverse)),
                (0_u32, Some(partial_forward)),
            ])
            .unwrap(),
            vec![0],
        );

        let pair = current(20, 2, &[10, 11]);
        let scalar1 = current(30, 3, &[12]);
        let final_destination = current(40, 4, &[10, 11, 12]);
        let final_forward = final_step(&[0, 1], &[2], 3, 54, [0, 0, 1, 2]);
        let final_reverse = final_step(&[2], &[0, 1], 3, 55, [0, 0, 1, 2]);
        let final_forward = candidate(
            &final_forward,
            &final_destination,
            [&pair, &scalar1],
            &application,
            56,
        );
        let final_reverse = candidate(
            &final_reverse,
            &final_destination,
            [&scalar1, &pair],
            &application,
            57,
        );
        assert_eq!(final_forward.group, final_reverse.group);
        // The concrete rank orders the source-first mirror traversal first;
        // owner choice therefore stays deterministic independently of insertion.
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (0_u32, Some(final_forward)),
                (1_u32, Some(final_reverse)),
            ])
            .unwrap(),
            vec![1],
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (1_u32, Some(final_reverse)),
                (0_u32, Some(final_forward)),
            ])
            .unwrap(),
            vec![1],
        );
    }

    #[test]
    fn owner_current_projection_retains_non_traversal_physics() {
        let destination = current(2, 2, &[10, 11]);
        let base = ContactOrbitCurrentPhysics::from_current(&destination);
        let mut changed = base;
        changed.current_state_template_id = 3;
        assert_ne!(base, changed);
        changed = base;
        changed.dynamic_lc_color_state_id = DynamicLCColorStateId::from_interner(3);
        assert_ne!(base, changed);
        changed = base;
        changed.recurrence_strategy = RecurrenceStrategy::TopologyReplay;
        assert_ne!(base, changed);
        changed = base;
        changed.spin_state_class = 1;
        assert_ne!(base, changed);
        let different_flavour = [3];
        changed = base;
        changed.flavour_flow = &different_flavour;
        assert_ne!(base, changed);
        changed = base;
        changed.quantum_number_flow_id = 1;
        assert_ne!(base, changed);
        let different_orders = [1];
        changed = base;
        changed.coupling_orders = &different_orders;
        assert_ne!(base, changed);
        changed = base;
        changed.propagator_template_id = Some(1);
        assert_ne!(base, changed);

        let source_a = topology_source(10, 7);
        let source_b = topology_source(20, 8);
        assert_ne!(
            ContactOrbitCurrentPhysics::from_current(&source_a),
            ContactOrbitCurrentPhysics::from_current(&source_b),
        );
    }

    #[test]
    fn owner_group_quotients_application_metadata_but_keeps_physics_boundaries() {
        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let step = partial_step(0, 1, 2, 10);
        let base_application = application();
        let base = candidate(&step, &destination, [&left, &right], &base_application, 12);

        let mut traversal_application = base_application.clone();
        traversal_application.transition_equivalence_class = "alternate-traversal".into();
        traversal_application.quantum_semantic_digest = digest(20);
        traversal_application.color_contraction_semantic_digest = digest(21);
        traversal_application.color_witness_term_id = LCColorWitnessTermId::new(4, 1);
        traversal_application.color_witness_proof_digest = digest(22);
        traversal_application.momentum_convention = vec![1, 0];
        traversal_application.output_factor_source = 1;
        traversal_application.evaluator_contract_kind = EvaluatorContractKind::Closure as u8;
        traversal_application.evaluator_callable_kind =
            EvaluatorCallableKind::RusticolTemplate as u8;
        traversal_application.evaluator_callable_signature = digest(23);
        traversal_application.evaluator_input_layout = vec![13, 14];
        traversal_application.evaluator_output_layout = vec![15];
        traversal_application.evaluator_exact_expression_digests = vec![digest(24)];
        traversal_application.evaluator_runtime_template =
            Some("alternate-runtime-template".into());
        traversal_application.output_projection_id = 9;
        let traversal = candidate(
            &step,
            &destination,
            [&left, &right],
            &traversal_application,
            13,
        );

        assert_eq!(base.group, traversal.group);
        assert_ne!(base.rank, traversal.rank);
        let forward =
            selected_contact_orbit_owner_tokens([(0_u32, Some(base)), (1_u32, Some(traversal))])
                .unwrap();
        let reverse =
            selected_contact_orbit_owner_tokens([(1_u32, Some(traversal)), (0_u32, Some(base))])
                .unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.len(), 1);

        let mut redistributed_factors = base_application.clone();
        redistributed_factors.transition_exact_factor = exact(2);
        redistributed_factors.color_exact_factor = exact(3);
        redistributed_factors.witness_exact_factor = exact_ratio(1, 6);
        let redistributed = candidate(
            &step,
            &destination,
            [&left, &right],
            &redistributed_factors,
            12,
        );
        assert_eq!(base.group, redistributed.group);
        assert_ne!(base.rank, redistributed.rank);
        let forward = selected_contact_orbit_owner_tokens([
            (0_u32, Some(base)),
            (1_u32, Some(redistributed)),
        ])
        .unwrap();
        let reverse = selected_contact_orbit_owner_tokens([
            (1_u32, Some(redistributed)),
            (0_u32, Some(base)),
        ])
        .unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.len(), 1);

        let mut different_product = redistributed_factors.clone();
        different_product.witness_exact_factor = exact_ratio(1, 5);
        assert_ne!(
            base.group,
            candidate(&step, &destination, [&left, &right], &different_product, 12,).group,
        );
        let mut different_binding = base_application.clone();
        different_binding.binding_coupling = exact(2);
        assert_ne!(
            base.group,
            candidate(&step, &destination, [&left, &right], &different_binding, 12,).group,
        );
        let mut different_exchange = base_application.clone();
        different_exchange.input_exchange_factor = Some(exact(-1));
        assert_ne!(
            base.group,
            candidate(
                &step,
                &destination,
                [&left, &right],
                &different_exchange,
                12,
            )
            .group,
        );

        let mut different_parameters = base_application.clone();
        different_parameters.coupling_parameter_ids = vec![99];
        assert_ne!(
            base.group,
            candidate(
                &step,
                &destination,
                [&left, &right],
                &different_parameters,
                12,
            )
            .group,
        );

        let mut certificate_row = step.clone();
        certificate_row.certificate_id = 1;
        assert_ne!(
            base.group,
            candidate(
                &certificate_row,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );
        let mut certificate = step.clone();
        certificate.certificate_semantic_digest = digest(30);
        assert_ne!(
            base.group,
            candidate(
                &certificate,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );
        let mut equivalence = step.clone();
        equivalence.physical_leg_equivalence_classes[step.result_leg as usize] = 1;
        assert_ne!(
            base.group,
            candidate(
                &equivalence,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );
        let mut parent_equivalence = step.clone();
        parent_equivalence.physical_leg_equivalence_classes[1] = 1;
        assert!(
            candidate_with_input_states(
                &parent_equivalence,
                &destination,
                [&left, &right],
                [
                    left.current_state_template_id(),
                    right.current_state_template_id(),
                ],
                &base_application,
                12,
            )
            .unwrap_err()
            .to_string()
            .contains("mapping is ambiguous"),
        );
        let mut evaluator_class = step.clone();
        evaluator_class.evaluator_class = "different-contact-class".into();
        assert_ne!(
            base.group,
            candidate(
                &evaluator_class,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );

        let mut invalid_reconstruction = step.clone();
        invalid_reconstruction.certificate_reconstruction_factor = exact(2);
        assert!(
            contact_orbit_owner_candidate(
                Some(&invalid_reconstruction),
                &destination,
                [&left, &right],
                [
                    left.current_state_template_id(),
                    right.current_state_template_id()
                ],
                &base_application,
                digest(12),
            )
            .unwrap_err()
            .to_string()
            .contains("exact unit reconstruction")
        );
        invalid_reconstruction.certificate_reconstruction_factor = ExactComplexRational::ONE;
        invalid_reconstruction.step_reconstruction_factor = exact(2);
        assert!(
            contact_orbit_owner_candidate(
                Some(&invalid_reconstruction),
                &destination,
                [&left, &right],
                [
                    left.current_state_template_id(),
                    right.current_state_template_id()
                ],
                &base_application,
                digest(12),
            )
            .unwrap_err()
            .to_string()
            .contains("exact unit reconstruction")
        );

        let mut changed_stage = base;
        changed_stage.group.stage = ContactOrbitStage::Final;
        assert_ne!(base.group, changed_stage.group);
        let mut changed_output_source = base;
        changed_output_source
            .group
            .output_source_particle_equivalence_class = Some(2);
        assert_ne!(base.group, changed_output_source.group);
    }

    #[test]
    fn passthrough_rank_conflict_duplicate_token_and_capacity_fail_closed() {
        assert_eq!(
            selected_contact_orbit_owner_tokens([(7_u32, None)]).unwrap(),
            vec![7],
        );

        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let application = application();
        let step = partial_step(0, 1, 2, 10);
        let duplicate = candidate(&step, &destination, [&left, &right], &application, 12);
        let error = selected_contact_orbit_owner_tokens([
            (0_u32, Some(duplicate)),
            (1_u32, Some(duplicate)),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("conflicting exact rank"));

        let mut different_group_step = step.clone();
        different_group_step.evaluator_class = "different-contact-class".into();
        let different_group = candidate(
            &different_group_step,
            &destination,
            [&left, &right],
            &application,
            13,
        );
        let error = selected_contact_orbit_owner_tokens([
            (0_u32, Some(duplicate)),
            (0_u32, Some(different_group)),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("duplicate tokens"));

        struct ImpossibleExactSize<'a> {
            remaining: usize,
            marker: std::marker::PhantomData<ContactOrbitOwnerCandidate<'a>>,
        }
        impl<'a> Iterator for ImpossibleExactSize<'a> {
            type Item = (u64, Option<ContactOrbitOwnerCandidate<'a>>);

            fn next(&mut self) -> Option<Self::Item> {
                None
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.remaining, Some(self.remaining))
            }
        }
        impl ExactSizeIterator for ImpossibleExactSize<'_> {
            fn len(&self) -> usize {
                self.remaining
            }
        }
        let error = selected_contact_orbit_owner_tokens(ImpossibleExactSize {
            remaining: usize::MAX,
            marker: std::marker::PhantomData,
        })
        .unwrap_err();
        assert!(error.to_string().contains("byte size overflows usize"));

        struct UnderreportedExactSize {
            emitted: u8,
        }
        impl Iterator for UnderreportedExactSize {
            type Item = (u64, Option<ContactOrbitOwnerCandidate<'static>>);

            fn next(&mut self) -> Option<Self::Item> {
                if self.emitted == 2 {
                    return None;
                }
                let token = u64::from(self.emitted);
                self.emitted += 1;
                Some((token, None))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (1, Some(1))
            }
        }
        impl ExactSizeIterator for UnderreportedExactSize {
            fn len(&self) -> usize {
                1
            }
        }
        let error =
            selected_contact_orbit_owner_tokens(UnderreportedExactSize { emitted: 0 }).unwrap_err();
        assert!(error.to_string().contains("exceeded its declared length"));
    }
}
