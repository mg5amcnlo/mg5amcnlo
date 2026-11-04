// SPDX-License-Identifier: 0BSD

#![allow(dead_code)] // Query-local proof inspection is retained for opt-in support builds.

use super::sweep::*;
use super::*;

#[cfg(feature = "on-the-fly-test-support")]
pub(crate) const ON_THE_FLY_WORK_CENSUS_BASIS_V1: &str = "fully-resident-query-local-trace-v1";

/// Authenticated model operation addressed by one semantic executor key.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, bincode::Encode, bincode::Decode)]
#[repr(u8)]
pub(crate) enum OnTheFlyOperationKindV1 {
    Source = 0,
    Transition = 1,
    Propagator = 2,
    Closure = 3,
    IdentityFinalizer = 4,
}

/// Plan-independent semantic address of one prepared Direct-Arena executor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, bincode::Encode, bincode::Decode)]
pub(crate) enum OnTheFlyExecutorKeyV1 {
    PreparedOperation {
        direct_catalog_digest: SemanticDigest,
        role: DirectExecutorRole,
        operation_kind: OnTheFlyOperationKindV1,
        operation_id: u32,
        operation_semantic_digest: SemanticDigest,
        evaluator_binding_id: u32,
        evaluator_binding_semantic_digest: SemanticDigest,
    },
    /// The prepared direct catalog authenticates this one typed synthetic
    /// binding as a whole.  It has no recurrence-operation or evaluator row,
    /// so no sentinel semantic digests are fabricated for it.
    IdentityFinalizer {
        direct_catalog_digest: SemanticDigest,
    },
}

impl OnTheFlyExecutorKeyV1 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        direct_catalog_digest: SemanticDigest,
        role: DirectExecutorRole,
        operation_kind: OnTheFlyOperationKindV1,
        operation_id: u32,
        operation_semantic_digest: SemanticDigest,
        evaluator_binding_id: u32,
        evaluator_binding_semantic_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        let expected_role = match operation_kind {
            OnTheFlyOperationKindV1::Source => DirectExecutorRole::Source,
            OnTheFlyOperationKindV1::Transition => DirectExecutorRole::Contribution,
            OnTheFlyOperationKindV1::Propagator => DirectExecutorRole::Finalization,
            OnTheFlyOperationKindV1::Closure => DirectExecutorRole::Closure,
            OnTheFlyOperationKindV1::IdentityFinalizer => {
                return Err(invalid(
                    "identity finalizer must use its typed semantic key",
                ));
            }
        };
        if role != expected_role {
            return Err(invalid("semantic executor key has an inconsistent role"));
        }
        Ok(Self::PreparedOperation {
            direct_catalog_digest,
            role,
            operation_kind,
            operation_id,
            operation_semantic_digest,
            evaluator_binding_id,
            evaluator_binding_semantic_digest,
        })
    }

    pub(crate) const fn identity_finalizer(direct_catalog_digest: SemanticDigest) -> Self {
        Self::IdentityFinalizer {
            direct_catalog_digest,
        }
    }

    pub(crate) const fn direct_catalog_digest(self) -> SemanticDigest {
        match self {
            Self::PreparedOperation {
                direct_catalog_digest,
                ..
            }
            | Self::IdentityFinalizer {
                direct_catalog_digest,
            } => direct_catalog_digest,
        }
    }

    pub(crate) const fn role(self) -> DirectExecutorRole {
        match self {
            Self::PreparedOperation { role, .. } => role,
            Self::IdentityFinalizer { .. } => DirectExecutorRole::Finalization,
        }
    }

    pub(crate) const fn operation_kind(self) -> OnTheFlyOperationKindV1 {
        match self {
            Self::PreparedOperation { operation_kind, .. } => operation_kind,
            Self::IdentityFinalizer { .. } => OnTheFlyOperationKindV1::IdentityFinalizer,
        }
    }

    pub(crate) const fn operation_id(self) -> Option<u32> {
        match self {
            Self::PreparedOperation { operation_id, .. } => Some(operation_id),
            Self::IdentityFinalizer { .. } => None,
        }
    }

    pub(crate) const fn operation_semantic_digest(self) -> Option<SemanticDigest> {
        match self {
            Self::PreparedOperation {
                operation_semantic_digest,
                ..
            } => Some(operation_semantic_digest),
            Self::IdentityFinalizer { .. } => None,
        }
    }

    pub(crate) const fn evaluator_binding_id(self) -> Option<u32> {
        match self {
            Self::PreparedOperation {
                evaluator_binding_id,
                ..
            } => Some(evaluator_binding_id),
            Self::IdentityFinalizer { .. } => None,
        }
    }

    pub(crate) const fn evaluator_binding_semantic_digest(self) -> Option<SemanticDigest> {
        match self {
            Self::PreparedOperation {
                evaluator_binding_semantic_digest,
                ..
            } => Some(evaluator_binding_semantic_digest),
            Self::IdentityFinalizer { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum OnTheFlyTraceOperationV1 {
    Source {
        key: OnTheFlyExecutorKeyV1,
        row: DirectSourceRow,
    },
    Contribution {
        key: OnTheFlyExecutorKeyV1,
        row: DirectContributionRow,
    },
    Finalization {
        key: OnTheFlyExecutorKeyV1,
        row: DirectFinalizationRow,
    },
    Closure {
        key: OnTheFlyExecutorKeyV1,
        row: DirectClosureRow,
    },
}

/// Cold semantic contribution row retained only by the development feature.
///
/// Direct rows carry trace-local component bases and, after prepared binding,
/// may use a physical parent permutation.  A query-family union instead needs
/// the exact construction identity before those physical choices.  Keeping
/// this row beside the trace lets the prototype remap parent current IDs
/// without materializing an established recurrence program or DirectPlan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnTheFlyTraceContributionProofRowV1 {
    pub(super) result_current_id: u32,
    pub(super) parent_current_ids: [u32; 2],
    pub(super) key: ContributionKey,
    pub(super) exact_factor: ExactComplexRational,
}

impl OnTheFlyTraceOperationV1 {
    pub(super) const fn key(&self) -> OnTheFlyExecutorKeyV1 {
        match self {
            Self::Source { key, .. }
            | Self::Contribution { key, .. }
            | Self::Finalization { key, .. }
            | Self::Closure { key, .. } => *key,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyWorkspaceLayoutV1 {
    pub(super) source_count: u32,
    pub(super) lorentz_component_count: u16,
    pub(super) parameter_count: u32,
    pub(super) current_component_count: u32,
    pub(super) amplitude_component_count: u32,
    pub(super) momentum_form_count: u32,
    pub(super) exact_factor_count: u32,
}

impl OnTheFlyWorkspaceLayoutV1 {
    pub(crate) const fn source_count(self) -> u32 {
        self.source_count
    }

    pub(crate) const fn current_component_count(self) -> u32 {
        self.current_component_count
    }

    pub(crate) const fn amplitude_component_count(self) -> u32 {
        self.amplitude_component_count
    }

    #[cfg(feature = "on-the-fly-test-support")]
    pub(crate) const fn lorentz_component_count(self) -> u16 {
        self.lorentz_component_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyStructuralProofV1 {
    current_count: u32,
    contribution_count: u32,
    closure_count: u32,
    constructed_current_count: u32,
    constructed_contribution_count: u32,
    current_multiset_digest: SemanticDigest,
    contribution_multiset_digest: SemanticDigest,
    closure_multiset_digest: SemanticDigest,
    owner_digest: SemanticDigest,
    semantic_digest: SemanticDigest,
}

/// Warming-independent work represented by one selected-query trace.
///
/// The current arena is fully resident for the lifetime of the trace.  These
/// current counts therefore describe allocated resident work, not a temporal
/// liveness estimate.  Operation counts describe one execution of the trace;
/// the probe must not multiply them by warm-up or benchmark repetitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) struct OnTheFlyExecutionWorkCensusV1 {
    pub(crate) logical_current_count: u32,
    pub(crate) resident_current_count: u32,
    pub(crate) resident_current_component_count: u32,
    pub(crate) source_operation_count: u32,
    pub(crate) contribution_operation_count: u32,
    pub(crate) finalization_operation_count: u32,
    pub(crate) closure_operation_count: u32,
    pub(crate) total_kernel_application_count: u32,
}

impl OnTheFlyStructuralProofV1 {
    pub(crate) const fn current_count(self) -> u32 {
        self.current_count
    }

    pub(crate) const fn contribution_count(self) -> u32 {
        self.contribution_count
    }

    pub(crate) const fn closure_count(self) -> u32 {
        self.closure_count
    }

    pub(crate) const fn semantic_digest(self) -> SemanticDigest {
        self.semantic_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn current_multiset_digest(self) -> SemanticDigest {
        self.current_multiset_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn contribution_multiset_digest(self) -> SemanticDigest {
        self.contribution_multiset_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn closure_multiset_digest(self) -> SemanticDigest {
        self.closure_multiset_digest
    }
}

/// Immutable selected-query trace.  Exact current keys remain available for
/// development parity joins; numeric IDs are deliberately query-local.
#[derive(Debug)]
pub(crate) struct OnTheFlyStructuralTraceV1 {
    pub(super) seed_digest: SemanticDigest,
    pub(super) query_digest: SemanticDigest,
    pub(super) current_keys: Box<[CurrentCoreKey]>,
    pub(super) current_colors: Box<[DynamicLCColorState]>,
    pub(super) operations: Box<[OnTheFlyTraceOperationV1]>,
    pub(super) momentum_forms: Box<[CanonicalMomentumLinearForm]>,
    pub(super) exact_factors: Box<[ExactComplexRational]>,
    pub(super) pairing_owners: Box<[ResolvedPairingOwnerV1]>,
    pub(super) layout: OnTheFlyWorkspaceLayoutV1,
    pub(super) proof: OnTheFlyStructuralProofV1,
    pub(super) semantic_digest: SemanticDigest,
    prepared_executor_rows_bound: bool,
    pub(super) current_component_ranges: Box<[[u32; 2]]>,
    pub(super) current_semantic_digests: Box<[SemanticDigest]>,
    pub(super) contribution_proof_rows: Box<[OnTheFlyTraceContributionProofRowV1]>,
}

impl OnTheFlyStructuralTraceV1 {
    pub(crate) fn executor_keys(&self) -> impl Iterator<Item = OnTheFlyExecutorKeyV1> + '_ {
        self.operations.iter().map(OnTheFlyTraceOperationV1::key)
    }

    /// Bind authenticated prepared-executor parent order into the trace rows
    /// once, before any executor can cache descriptors for their addresses.
    /// The semantic trace identity is unchanged: this is a private physical
    /// executor binding, not a different selected recurrence graph.
    pub(crate) fn bind_prepared_executor_rows(
        &mut self,
        parent_permutations: &BTreeMap<OnTheFlyExecutorKeyV1, [u8; 2]>,
    ) -> RusticolResult<()> {
        if self.prepared_executor_rows_bound {
            return Err(integrity(
                "prepared executor parent order is already bound into the trace",
            ));
        }

        let mut contribution_permutations = Vec::new();
        for (operation_index, operation) in self.operations.iter().enumerate() {
            let key = operation.key();
            let permutation = parent_permutations.get(&key).copied().ok_or_else(|| {
                integrity("trace operation has no authenticated prepared parent order")
            })?;
            match operation {
                OnTheFlyTraceOperationV1::Contribution { .. } => match permutation {
                    [0, 1] | [1, 0] => {
                        contribution_permutations.push((operation_index, permutation));
                    }
                    _ => {
                        return Err(integrity(
                            "prepared contribution executor has a non-binary parent order",
                        ));
                    }
                },
                OnTheFlyTraceOperationV1::Source { .. }
                | OnTheFlyTraceOperationV1::Finalization { .. }
                | OnTheFlyTraceOperationV1::Closure { .. } => {
                    if permutation != [0, 1] {
                        return Err(integrity(
                            "non-contribution executor unexpectedly permutes parent rows",
                        ));
                    }
                }
            }
        }

        // The first pass made every remaining operation infallible. Mutate the
        // existing boxed rows in place so their stable storage is preserved.
        for (operation_index, permutation) in contribution_permutations {
            if permutation == [0, 1] {
                continue;
            }
            let Some(OnTheFlyTraceOperationV1::Contribution { row, .. }) =
                self.operations.get_mut(operation_index)
            else {
                unreachable!("validated contribution operation changed during parent binding");
            };
            std::mem::swap(
                &mut row.parent0_component_base,
                &mut row.parent1_component_base_or_sentinel,
            );
            std::mem::swap(
                &mut row.parent0_momentum_form_id,
                &mut row.parent1_momentum_form_id_or_sentinel,
            );
        }
        self.prepared_executor_rows_bound = true;
        Ok(())
    }

    pub(crate) const fn prepared_executor_rows_bound(&self) -> bool {
        self.prepared_executor_rows_bound
    }

    pub(crate) const fn proof(&self) -> OnTheFlyStructuralProofV1 {
        self.proof
    }

    pub(crate) const fn seed_digest(&self) -> SemanticDigest {
        self.seed_digest
    }

    pub(crate) fn current_keys(&self) -> &[CurrentCoreKey] {
        &self.current_keys
    }

    pub(crate) const fn layout(&self) -> OnTheFlyWorkspaceLayoutV1 {
        self.layout
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(crate) fn execution_work_census(&self) -> RusticolResult<OnTheFlyExecutionWorkCensusV1> {
        let mut source_operation_count = 0_u32;
        let mut contribution_operation_count = 0_u32;
        let mut finalization_operation_count = 0_u32;
        let mut closure_operation_count = 0_u32;
        for operation in &self.operations {
            let count = match operation.key().role() {
                DirectExecutorRole::Source => &mut source_operation_count,
                DirectExecutorRole::Contribution => &mut contribution_operation_count,
                DirectExecutorRole::Finalization => &mut finalization_operation_count,
                DirectExecutorRole::Closure => &mut closure_operation_count,
            };
            *count = count
                .checked_add(1)
                .ok_or_else(|| invalid("execution operation count overflows u32"))?;
        }
        let total_kernel_application_count = checked_u32(
            self.operations.len(),
            "selected-query kernel application count",
        )?;
        let counted = source_operation_count
            .checked_add(contribution_operation_count)
            .and_then(|value| value.checked_add(finalization_operation_count))
            .and_then(|value| value.checked_add(closure_operation_count))
            .ok_or_else(|| invalid("execution operation total overflows u32"))?;
        if counted != total_kernel_application_count {
            return Err(integrity(
                "execution operation census does not cover the selected-query trace",
            ));
        }
        let logical_current_count = self.proof.current_count();
        Ok(OnTheFlyExecutionWorkCensusV1 {
            logical_current_count,
            resident_current_count: logical_current_count,
            resident_current_component_count: self.layout.current_component_count(),
            source_operation_count,
            contribution_operation_count,
            finalization_operation_count,
            closure_operation_count,
            total_kernel_application_count,
        })
    }

    pub(crate) fn momentum_forms(&self) -> &[CanonicalMomentumLinearForm] {
        &self.momentum_forms
    }

    pub(crate) fn exact_factors(&self) -> &[ExactComplexRational] {
        &self.exact_factors
    }

    pub(crate) const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    #[cfg(feature = "on-the-fly-test-support")]
    pub(crate) fn tamper_first_prepared_executor_key_for_probe(&mut self) -> RusticolResult<()> {
        let replacement = SemanticDigest::new([0xa5; 32])?;
        for operation in self.operations.iter_mut() {
            let key = operation.key();
            let OnTheFlyExecutorKeyV1::PreparedOperation {
                direct_catalog_digest,
                role,
                operation_kind,
                operation_id,
                operation_semantic_digest,
                evaluator_binding_id,
                evaluator_binding_semantic_digest,
            } = key
            else {
                continue;
            };
            let replacement = if replacement != operation_semantic_digest {
                replacement
            } else {
                SemanticDigest::new([0x5a; 32])?
            };
            let tampered = OnTheFlyExecutorKeyV1::PreparedOperation {
                direct_catalog_digest,
                role,
                operation_kind,
                operation_id,
                operation_semantic_digest: replacement,
                evaluator_binding_id,
                evaluator_binding_semantic_digest,
            };
            match operation {
                OnTheFlyTraceOperationV1::Source { key, .. }
                | OnTheFlyTraceOperationV1::Contribution { key, .. }
                | OnTheFlyTraceOperationV1::Finalization { key, .. }
                | OnTheFlyTraceOperationV1::Closure { key, .. } => *key = tampered,
            }
            return Ok(());
        }
        Err(integrity(
            "selected trace has no prepared executor key to tamper",
        ))
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(crate) fn current_component_range(&self, current_id: u32) -> RusticolResult<[u32; 2]> {
        self.current_component_ranges
            .get(current_id as usize)
            .copied()
            .ok_or_else(|| invalid("observed current ID is outside the trace"))
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(crate) fn current_semantic_digest(
        &self,
        current_id: u32,
    ) -> RusticolResult<SemanticDigest> {
        self.current_semantic_digests
            .get(current_id as usize)
            .copied()
            .ok_or_else(|| invalid("observed current ID is outside the trace"))
    }
}

/// Join one query-local operation key to the exact authenticated recurrence
/// row and prepared direct-template binding that implements it.  No global
/// operation map or direct recurrence plan participates in the address.
pub(crate) fn authenticated_prepared_executor_binding<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    direct: &'a PreparedDirectExecutorCatalog,
) -> RusticolResult<
    impl Fn(OnTheFlyExecutorKeyV1) -> RusticolResult<PreparedDirectExecutorBinding> + 'a,
> {
    let direct_digest = direct.direct_template_catalog_digest();
    Ok(move |key: OnTheFlyExecutorKeyV1| {
        if key.direct_catalog_digest() != direct_digest {
            return Err(integrity(
                "operation key belongs to a different direct-template catalog",
            ));
        }
        if matches!(key, OnTheFlyExecutorKeyV1::IdentityFinalizer { .. }) {
            return Ok(PreparedDirectExecutorBinding::identity_finalizer(
                direct.resolve_identity_finalizer()?,
            ));
        }
        let operation_id = key
            .operation_id()
            .ok_or_else(|| integrity("prepared operation key has no operation ID"))?;
        let (row_id, semantic_digest_id, evaluator_binding_id, expected_role, label) =
            match key.operation_kind() {
                OnTheFlyOperationKindV1::Source => {
                    let row = templates
                        .input()
                        .sources
                        .get(operation_id as usize)
                        .ok_or_else(|| integrity("source operation row is absent"))?;
                    (
                        row.id,
                        row.semantic_digest_id,
                        row.evaluator_binding_id,
                        DirectExecutorRole::Source,
                        "source",
                    )
                }
                OnTheFlyOperationKindV1::Transition => {
                    let row = templates
                        .input()
                        .transitions
                        .get(operation_id as usize)
                        .ok_or_else(|| integrity("transition operation row is absent"))?;
                    (
                        row.id,
                        row.semantic_digest_id,
                        row.evaluator_binding_id,
                        DirectExecutorRole::Contribution,
                        "transition",
                    )
                }
                OnTheFlyOperationKindV1::Propagator => {
                    let row = templates
                        .input()
                        .propagators
                        .get(operation_id as usize)
                        .ok_or_else(|| integrity("propagator operation row is absent"))?;
                    (
                        row.id,
                        row.semantic_digest_id,
                        row.evaluator_binding_id,
                        DirectExecutorRole::Finalization,
                        "propagator",
                    )
                }
                OnTheFlyOperationKindV1::Closure => {
                    let row = templates
                        .input()
                        .closures
                        .get(operation_id as usize)
                        .ok_or_else(|| integrity("closure operation row is absent"))?;
                    (
                        row.id,
                        row.semantic_digest_id,
                        row.evaluator_binding_id,
                        DirectExecutorRole::Closure,
                        "closure",
                    )
                }
                OnTheFlyOperationKindV1::IdentityFinalizer => {
                    return Err(integrity(
                        "identity finalizer is not a prepared-operation key",
                    ));
                }
            };
        if row_id != operation_id || key.role() != expected_role {
            return Err(integrity(format!(
                "{label} operation catalog or executor role is not canonical"
            )));
        }
        let evaluator = templates
            .input()
            .evaluator_bindings
            .get(evaluator_binding_id as usize)
            .ok_or_else(|| integrity(format!("{label} evaluator binding is absent")))?;
        if evaluator.id != evaluator_binding_id
            || key.operation_semantic_digest()
                != Some(authenticated_digest_row(
                    templates,
                    semantic_digest_id,
                    &format!("{label} semantic"),
                )?)
            || key.evaluator_binding_id() != Some(evaluator.id)
            || key.evaluator_binding_semantic_digest()
                != Some(authenticated_digest_row(
                    templates,
                    evaluator.semantic_digest_id,
                    &format!("{label} evaluator semantic"),
                )?)
        {
            return Err(integrity(format!(
                "{label} operation key disagrees with authenticated template semantics"
            )));
        }
        let binding = match expected_role {
            DirectExecutorRole::Contribution => {
                let (executor_id, parent_permutation) =
                    direct.resolve_contribution(evaluator.id)?;
                PreparedDirectExecutorBinding::evaluator_with_parent_permutation(
                    expected_role,
                    evaluator.id,
                    executor_id,
                    parent_permutation,
                )
            }
            role => PreparedDirectExecutorBinding::evaluator(
                role,
                evaluator.id,
                direct.resolve_evaluator(role, evaluator.id)?,
            ),
        };
        Ok(binding)
    })
}

fn authenticated_digest_row(
    templates: &ValidatedRecurrenceTemplateInput,
    digest_id: u32,
    label: &str,
) -> RusticolResult<SemanticDigest> {
    let row = templates
        .input()
        .digest_catalog
        .get(digest_id as usize)
        .ok_or_else(|| integrity(format!("{label} digest is absent")))?;
    if row.id != digest_id {
        return Err(integrity(format!(
            "{label} digest catalog is not canonical"
        )));
    }
    SemanticDigest::new(row.value)
        .map_err(|error| integrity(format!("{label} digest is invalid: {error}")))
}

#[cfg(test)]
pub(crate) fn scalar_adapter_test_trace(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<OnTheFlyStructuralTraceV1> {
    let source = templates
        .input()
        .sources
        .first()
        .ok_or_else(|| integrity("scalar adapter fixture has no source"))?;
    let source_evaluator = templates
        .input()
        .evaluator_bindings
        .get(source.evaluator_binding_id as usize)
        .ok_or_else(|| integrity("scalar adapter source evaluator is absent"))?;
    let source_key = OnTheFlyExecutorKeyV1::new(
        seed.direct_catalog_digest(),
        DirectExecutorRole::Source,
        OnTheFlyOperationKindV1::Source,
        source.id,
        authenticated_digest_row(templates, source.semantic_digest_id, "source semantic")?,
        source_evaluator.id,
        authenticated_digest_row(
            templates,
            source_evaluator.semantic_digest_id,
            "source evaluator semantic",
        )?,
    )?;
    let closure = templates
        .input()
        .closures
        .first()
        .ok_or_else(|| integrity("scalar adapter fixture has no closure"))?;
    let closure_evaluator = templates
        .input()
        .evaluator_bindings
        .get(closure.evaluator_binding_id as usize)
        .ok_or_else(|| integrity("scalar adapter closure evaluator is absent"))?;
    let closure_key = OnTheFlyExecutorKeyV1::new(
        seed.direct_catalog_digest(),
        DirectExecutorRole::Closure,
        OnTheFlyOperationKindV1::Closure,
        closure.id,
        authenticated_digest_row(templates, closure.semantic_digest_id, "closure semantic")?,
        closure_evaluator.id,
        authenticated_digest_row(
            templates,
            closure_evaluator.semantic_digest_id,
            "closure evaluator semantic",
        )?,
    )?;
    let momentum_forms = (0..2)
        .map(|source_slot| {
            CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                source_slot,
                coefficient: 1,
            }])
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let test_digest = SemanticDigest::new([93; 32])?;
    Ok(OnTheFlyStructuralTraceV1 {
        seed_digest: seed.semantic_digest(),
        query_digest: SemanticDigest::new([94; 32])?,
        current_keys: Vec::new().into_boxed_slice(),
        current_colors: Vec::new().into_boxed_slice(),
        operations: vec![
            OnTheFlyTraceOperationV1::Source {
                key: source_key,
                row: DirectSourceRow {
                    source_slot: 0,
                    destination_component_base: 0,
                    momentum_form_id: 0,
                    source_template_or_dispatch_domain: 0,
                    spin_state_class: 50_000,
                    exact_factor_id: 0,
                    selector_domain_id: 0,
                },
            },
            OnTheFlyTraceOperationV1::Source {
                key: source_key,
                row: DirectSourceRow {
                    source_slot: 1,
                    destination_component_base: 1,
                    momentum_form_id: 1,
                    source_template_or_dispatch_domain: 0,
                    spin_state_class: 50_000,
                    exact_factor_id: 0,
                    selector_domain_id: 0,
                },
            },
            OnTheFlyTraceOperationV1::Closure {
                key: closure_key,
                row: DirectClosureRow {
                    parent0_component_base: 0,
                    parent1_component_base_or_sentinel: 1,
                    parent0_momentum_form_id: 0,
                    parent1_momentum_form_id_or_sentinel: 1,
                    amplitude_destination_id: 0,
                    exact_factor_id: 0,
                    component_factor_start: 1,
                    component_count: 1,
                    selector_domain_id: 0,
                    flags: 0,
                },
            },
        ]
        .into_boxed_slice(),
        momentum_forms: momentum_forms.into_boxed_slice(),
        exact_factors: vec![
            ExactComplexRational::ONE,
            ExactComplexRational::ONE,
            ExactComplexRational::new(
                crate::recurrence::ExactRational::new(2, 1)?,
                crate::recurrence::ExactRational::ZERO,
            ),
        ]
        .into_boxed_slice(),
        pairing_owners: vec![ResolvedPairingOwnerV1 {
            endpoint_pairs: Vec::new(),
            proof_digest: None,
            source_slot_permutation: vec![0, 1],
            source_lineage: vec![0, 1],
            fermion_parity: 1,
        }]
        .into_boxed_slice(),
        layout: OnTheFlyWorkspaceLayoutV1 {
            source_count: 2,
            lorentz_component_count: 4,
            parameter_count: 0,
            current_component_count: 2,
            amplitude_component_count: 1,
            momentum_form_count: 2,
            exact_factor_count: 3,
        },
        proof: OnTheFlyStructuralProofV1 {
            current_count: 2,
            contribution_count: 0,
            closure_count: 1,
            constructed_current_count: 2,
            constructed_contribution_count: 0,
            current_multiset_digest: test_digest,
            contribution_multiset_digest: test_digest,
            closure_multiset_digest: test_digest,
            owner_digest: test_digest,
            semantic_digest: test_digest,
        },
        semantic_digest: test_digest,
        prepared_executor_rows_bound: false,
        current_component_ranges: vec![[0, 1], [1, 1]].into_boxed_slice(),
        current_semantic_digests: vec![test_digest; 2].into_boxed_slice(),
        contribution_proof_rows: Box::new([]),
    })
}

#[cfg(test)]
impl OnTheFlyStructuralTraceV1 {
    pub(crate) fn test_two_contribution_rows()
    -> (Self, OnTheFlyExecutorKeyV1, OnTheFlyExecutorKeyV1) {
        let test_digest = SemanticDigest::new([0x5a; 32]).unwrap();
        let direct_digest = SemanticDigest::new([0x61; 32]).unwrap();
        let source_key = OnTheFlyExecutorKeyV1::new(
            direct_digest,
            DirectExecutorRole::Source,
            OnTheFlyOperationKindV1::Source,
            0,
            SemanticDigest::new([0x62; 32]).unwrap(),
            0,
            SemanticDigest::new([0x63; 32]).unwrap(),
        )
        .unwrap();
        let contribution_key = OnTheFlyExecutorKeyV1::new(
            direct_digest,
            DirectExecutorRole::Contribution,
            OnTheFlyOperationKindV1::Transition,
            1,
            SemanticDigest::new([0x64; 32]).unwrap(),
            1,
            SemanticDigest::new([0x65; 32]).unwrap(),
        )
        .unwrap();
        let momentum_forms = (0..4)
            .map(|source_slot| {
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot,
                    coefficient: 1,
                }])
                .unwrap()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let trace = Self {
            seed_digest: SemanticDigest::new([0x51; 32]).unwrap(),
            query_digest: SemanticDigest::new([0x52; 32]).unwrap(),
            current_keys: Box::new([]),
            current_colors: Box::new([]),
            operations: vec![
                OnTheFlyTraceOperationV1::Source {
                    key: source_key,
                    row: DirectSourceRow {
                        source_slot: 0,
                        destination_component_base: 1,
                        momentum_form_id: 1,
                        source_template_or_dispatch_domain: 0,
                        spin_state_class: 50_000,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                    },
                },
                OnTheFlyTraceOperationV1::Source {
                    key: source_key,
                    row: DirectSourceRow {
                        source_slot: 1,
                        destination_component_base: 3,
                        momentum_form_id: 3,
                        source_template_or_dispatch_domain: 0,
                        spin_state_class: 50_000,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                    },
                },
                OnTheFlyTraceOperationV1::Contribution {
                    key: contribution_key,
                    row: DirectContributionRow {
                        parent0_component_base: 0,
                        parent1_component_base_or_sentinel: 1,
                        parent0_momentum_form_id: 0,
                        parent1_momentum_form_id_or_sentinel: 1,
                        destination_component_base: 2,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                        flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
                    },
                },
                OnTheFlyTraceOperationV1::Contribution {
                    key: contribution_key,
                    row: DirectContributionRow {
                        parent0_component_base: 0,
                        parent1_component_base_or_sentinel: 3,
                        parent0_momentum_form_id: 0,
                        parent1_momentum_form_id_or_sentinel: 3,
                        destination_component_base: 4,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                        flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
                    },
                },
            ]
            .into_boxed_slice(),
            momentum_forms,
            exact_factors: vec![ExactComplexRational::ONE].into_boxed_slice(),
            pairing_owners: Box::new([]),
            layout: OnTheFlyWorkspaceLayoutV1 {
                source_count: 4,
                lorentz_component_count: 4,
                parameter_count: 0,
                current_component_count: 5,
                amplitude_component_count: 1,
                momentum_form_count: 4,
                exact_factor_count: 1,
            },
            proof: OnTheFlyStructuralProofV1 {
                current_count: 5,
                contribution_count: 2,
                closure_count: 0,
                constructed_current_count: 5,
                constructed_contribution_count: 2,
                current_multiset_digest: test_digest,
                contribution_multiset_digest: test_digest,
                closure_multiset_digest: test_digest,
                owner_digest: test_digest,
                semantic_digest: test_digest,
            },
            semantic_digest: test_digest,
            prepared_executor_rows_bound: false,
            current_component_ranges: (0..5)
                .map(|base| [base, 1])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            current_semantic_digests: vec![test_digest; 5].into_boxed_slice(),
            contribution_proof_rows: Box::new([]),
        };
        (trace, source_key, contribution_key)
    }

    pub(crate) fn test_bound_contribution_rows(&self) -> Vec<DirectContributionRow> {
        self.operations
            .iter()
            .filter_map(|operation| match operation {
                OnTheFlyTraceOperationV1::Contribution { row, .. } => Some(*row),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn test_query_family_trace(
        query_digest_byte: u8,
        contribution_factor: ExactComplexRational,
    ) -> (Self, OnTheFlyProjectionProbeV1) {
        let digest = |byte| SemanticDigest::new([byte; 32]).unwrap();
        let direct_digest = digest(0x71);
        let color_id = crate::recurrence::DynamicLCColorStateId::from_interner(0);
        let momentum = |slots: &[u32]| {
            CanonicalMomentumLinearForm::new(
                slots
                    .iter()
                    .copied()
                    .map(|source_slot| MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    })
                    .collect(),
            )
            .unwrap()
        };
        let helicity = |slots: &[u32]| {
            CurrentHelicityIdentity::topology_replay(
                0,
                slots
                    .iter()
                    .copied()
                    .map(|source_slot| SourceStateAssignment::new(source_slot, 0))
                    .collect(),
            )
            .unwrap()
        };
        let current_key = |node_kind, state, support: &[u32], source_binding| {
            CurrentCoreKey::new(
                digest(0x72),
                node_kind,
                state,
                color_id,
                support.to_vec(),
                momentum(support),
                helicity(support),
                vec![state as i32],
                0,
                vec![0],
                source_binding,
                (node_kind == RecurrenceNodeKind::Current).then_some(0),
            )
            .unwrap()
        };
        let current_keys = vec![
            current_key(
                RecurrenceNodeKind::Source,
                0,
                &[0],
                CurrentSourceBinding::FixedTemplate(0),
            ),
            current_key(
                RecurrenceNodeKind::Source,
                1,
                &[1],
                CurrentSourceBinding::FixedTemplate(1),
            ),
            current_key(
                RecurrenceNodeKind::Current,
                2,
                &[0, 1],
                CurrentSourceBinding::None,
            ),
        ];
        let source_key = |id| {
            OnTheFlyExecutorKeyV1::new(
                direct_digest,
                DirectExecutorRole::Source,
                OnTheFlyOperationKindV1::Source,
                id,
                digest(0x73 + id as u8),
                id,
                digest(0x75 + id as u8),
            )
            .unwrap()
        };
        let contribution_executor = OnTheFlyExecutorKeyV1::new(
            direct_digest,
            DirectExecutorRole::Contribution,
            OnTheFlyOperationKindV1::Transition,
            2,
            digest(0x77),
            2,
            digest(0x78),
        )
        .unwrap();
        let finalization_executor = OnTheFlyExecutorKeyV1::new(
            direct_digest,
            DirectExecutorRole::Finalization,
            OnTheFlyOperationKindV1::Propagator,
            0,
            digest(0x79),
            3,
            digest(0x7a),
        )
        .unwrap();
        let closure_executor = OnTheFlyExecutorKeyV1::new(
            direct_digest,
            DirectExecutorRole::Closure,
            OnTheFlyOperationKindV1::Closure,
            0,
            digest(0x7b),
            4,
            digest(0x7c),
        )
        .unwrap();
        let contribution_key = ContributionKey::new(
            2,
            vec![0, 1],
            vec![0, 1],
            vec![momentum(&[0]), momentum(&[1])],
            2,
            0,
            LCColorWitnessTermId::new(0, 0),
            digest(0x7d),
            0,
        )
        .unwrap();
        let color = DynamicLCColorState::new(0, None, vec![]).unwrap();
        let test_digest = digest(0x7e);
        let current_semantic_digests = current_keys
            .iter()
            .map(|key| hash_current_key(key, &color).unwrap())
            .collect::<Vec<_>>();
        let trace = Self {
            seed_digest: digest(0x70),
            query_digest: digest(query_digest_byte),
            current_keys: current_keys.into_boxed_slice(),
            current_colors: vec![color; 3].into_boxed_slice(),
            operations: vec![
                OnTheFlyTraceOperationV1::Source {
                    key: source_key(0),
                    row: DirectSourceRow {
                        source_slot: 0,
                        destination_component_base: 0,
                        momentum_form_id: 0,
                        source_template_or_dispatch_domain: 0,
                        spin_state_class: 0,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                    },
                },
                OnTheFlyTraceOperationV1::Source {
                    key: source_key(1),
                    row: DirectSourceRow {
                        source_slot: 1,
                        destination_component_base: 1,
                        momentum_form_id: 1,
                        source_template_or_dispatch_domain: 1,
                        spin_state_class: 0,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                    },
                },
                OnTheFlyTraceOperationV1::Contribution {
                    key: contribution_executor,
                    row: DirectContributionRow {
                        parent0_component_base: 0,
                        parent1_component_base_or_sentinel: 1,
                        parent0_momentum_form_id: 0,
                        parent1_momentum_form_id_or_sentinel: 1,
                        destination_component_base: 2,
                        exact_factor_id: 1,
                        selector_domain_id: 0,
                        flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
                    },
                },
                OnTheFlyTraceOperationV1::Finalization {
                    key: finalization_executor,
                    row: DirectFinalizationRow {
                        component_base: 2,
                        component_count: 1,
                        momentum_form_id: 2,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                        flags: 0,
                    },
                },
                OnTheFlyTraceOperationV1::Closure {
                    key: closure_executor,
                    row: DirectClosureRow {
                        parent0_component_base: 0,
                        parent1_component_base_or_sentinel: 2,
                        parent0_momentum_form_id: 0,
                        parent1_momentum_form_id_or_sentinel: 2,
                        amplitude_destination_id: 0,
                        exact_factor_id: 0,
                        component_factor_start: 2,
                        component_count: 1,
                        selector_domain_id: 0,
                        flags: 0,
                    },
                },
            ]
            .into_boxed_slice(),
            momentum_forms: vec![momentum(&[0]), momentum(&[1]), momentum(&[0, 1])]
                .into_boxed_slice(),
            exact_factors: vec![
                ExactComplexRational::ONE,
                contribution_factor,
                ExactComplexRational::ONE,
            ]
            .into_boxed_slice(),
            pairing_owners: vec![ResolvedPairingOwnerV1 {
                endpoint_pairs: Vec::new(),
                proof_digest: None,
                source_slot_permutation: vec![0, 1],
                source_lineage: vec![0, 1],
                fermion_parity: 1,
            }]
            .into_boxed_slice(),
            layout: OnTheFlyWorkspaceLayoutV1 {
                source_count: 2,
                lorentz_component_count: 4,
                parameter_count: 0,
                current_component_count: 3,
                amplitude_component_count: 1,
                momentum_form_count: 3,
                exact_factor_count: 3,
            },
            proof: OnTheFlyStructuralProofV1 {
                current_count: 3,
                contribution_count: 1,
                closure_count: 1,
                constructed_current_count: 3,
                constructed_contribution_count: 1,
                current_multiset_digest: test_digest,
                contribution_multiset_digest: test_digest,
                closure_multiset_digest: test_digest,
                owner_digest: test_digest,
                semantic_digest: test_digest,
            },
            semantic_digest: test_digest,
            prepared_executor_rows_bound: true,
            current_component_ranges: vec![[0, 1], [1, 1], [2, 1]].into_boxed_slice(),
            current_semantic_digests: current_semantic_digests.into_boxed_slice(),
            contribution_proof_rows: vec![OnTheFlyTraceContributionProofRowV1 {
                result_current_id: 2,
                parent_current_ids: [0, 1],
                key: contribution_key,
                exact_factor: contribution_factor,
            }]
            .into_boxed_slice(),
        };
        (
            trace,
            OnTheFlyProjectionProbeV1 {
                enabled: true,
                applied: false,
                pre: [3, 1, 1],
                post: [3, 1, 1],
            },
        )
    }

    pub(crate) fn test_remap_dynamic_color_id(&mut self, id: u32) {
        let color_id = crate::recurrence::DynamicLCColorStateId::from_interner(id);
        for ((key, color), digest) in self
            .current_keys
            .iter_mut()
            .zip(self.current_colors.iter())
            .zip(self.current_semantic_digests.iter_mut())
        {
            *key = crate::recurrence::construct::current_key_with_dynamic_color(key, color_id)
                .unwrap();
            *digest = hash_current_key(key, color).unwrap();
        }
    }

    pub(crate) fn test_add_query_family_contribution(
        &mut self,
        operation_id: u32,
        evaluator_binding_id: u32,
        exact_factor: ExactComplexRational,
    ) {
        let digest = |byte| SemanticDigest::new([byte; 32]).unwrap();
        let existing = self.contribution_proof_rows[0].clone();
        let executor = OnTheFlyExecutorKeyV1::new(
            self.executor_keys().next().unwrap().direct_catalog_digest(),
            DirectExecutorRole::Contribution,
            OnTheFlyOperationKindV1::Transition,
            operation_id,
            digest(0x90_u8.wrapping_add(operation_id as u8)),
            evaluator_binding_id,
            digest(0xa0_u8.wrapping_add(evaluator_binding_id as u8)),
        )
        .unwrap();
        let proof_key = ContributionKey::new(
            operation_id,
            existing.key.parent_value_class_ids().to_vec(),
            existing.key.parent_state_template_ids().to_vec(),
            existing.key.parent_momenta().to_vec(),
            existing.key.result_state_template_id(),
            existing.key.quantum_flow_witness_id(),
            existing.key.color_witness_term_id(),
            digest(0xb0_u8.wrapping_add(evaluator_binding_id as u8)),
            existing.key.output_projection_id(),
        )
        .unwrap();
        let mut factors = std::mem::take(&mut self.exact_factors).into_vec();
        let exact_factor_id = factors.len() as u32;
        factors.push(exact_factor);
        self.exact_factors = factors.into_boxed_slice();
        self.layout.exact_factor_count += 1;
        let mut operations = std::mem::take(&mut self.operations).into_vec();
        let insertion = operations
            .iter()
            .position(|operation| {
                matches!(operation, OnTheFlyTraceOperationV1::Finalization { .. })
            })
            .unwrap();
        operations.insert(
            insertion,
            OnTheFlyTraceOperationV1::Contribution {
                key: executor,
                row: DirectContributionRow {
                    parent0_component_base: 0,
                    parent1_component_base_or_sentinel: 1,
                    parent0_momentum_form_id: 0,
                    parent1_momentum_form_id_or_sentinel: 1,
                    destination_component_base: 2,
                    exact_factor_id,
                    selector_domain_id: 0,
                    flags: 0,
                },
            },
        );
        self.operations = operations.into_boxed_slice();
        let mut proofs = std::mem::take(&mut self.contribution_proof_rows).into_vec();
        proofs.push(OnTheFlyTraceContributionProofRowV1 {
            result_current_id: 2,
            parent_current_ids: [0, 1],
            key: proof_key,
            exact_factor,
        });
        self.contribution_proof_rows = proofs.into_boxed_slice();
        self.proof.contribution_count += 1;
        self.proof.constructed_contribution_count += 1;
        let replacement = digest(0xc0_u8.wrapping_add(evaluator_binding_id as u8));
        self.semantic_digest = replacement;
        self.proof.semantic_digest = replacement;
    }

    pub(crate) fn test_set_last_contribution_selector_domain(&mut self, selector_domain_id: u32) {
        let row = self
            .operations
            .iter_mut()
            .rev()
            .find_map(|operation| match operation {
                OnTheFlyTraceOperationV1::Contribution { row, .. } => Some(row),
                _ => None,
            })
            .expect("test trace has no contribution row");
        row.selector_domain_id = selector_domain_id;
        let replacement = SemanticDigest::new([0xd1; 32]).unwrap();
        self.semantic_digest = replacement;
        self.proof.semantic_digest = replacement;
    }

    pub(crate) fn test_insert_identity_finalizer(&mut self, direct_catalog_digest: SemanticDigest) {
        let operation = OnTheFlyTraceOperationV1::Finalization {
            key: OnTheFlyExecutorKeyV1::identity_finalizer(direct_catalog_digest),
            row: DirectFinalizationRow {
                component_base: 0,
                component_count: 1,
                momentum_form_id: 0,
                exact_factor_id: 2,
                selector_domain_id: 0,
                flags: 0,
            },
        };
        let mut operations = std::mem::take(&mut self.operations).into_vec();
        operations.insert(2, operation);
        self.operations = operations.into_boxed_slice();
    }

    pub(crate) fn test_tamper_first_operation_semantic_digest(&mut self, digest: SemanticDigest) {
        if let Some(OnTheFlyTraceOperationV1::Source {
            key:
                OnTheFlyExecutorKeyV1::PreparedOperation {
                    operation_semantic_digest,
                    ..
                },
            ..
        }) = self.operations.first_mut()
        {
            *operation_semantic_digest = digest;
        }
    }
}

pub(crate) fn hash_current_key(
    key: &CurrentCoreKey,
    color: &DynamicLCColorState,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-current-key-v1\0");
    hash_digest(&mut hash, key.catalog_digest());
    hash.update([match key.node_kind() {
        RecurrenceNodeKind::Source => 0,
        RecurrenceNodeKind::Current => 1,
    }]);
    hash.update(key.current_state_template_id().to_le_bytes());
    hash_len(
        &mut hash,
        key.support_source_slots().len(),
        "current support",
    )?;
    for slot in key.support_source_slots() {
        hash.update(slot.to_le_bytes());
    }
    hash_len(&mut hash, key.momentum().terms().len(), "current momentum")?;
    for term in key.momentum().terms() {
        hash.update(term.source_slot.to_le_bytes());
        hash.update(term.coefficient.to_le_bytes());
    }
    hash.update(key.spin_state_class().to_le_bytes());
    hash_len(
        &mut hash,
        key.helicity_identity().local_source_states().len(),
        "current source ancestry",
    )?;
    for assignment in key.helicity_identity().local_source_states() {
        hash.update(assignment.source_slot().to_le_bytes());
        hash.update(assignment.state_index().to_le_bytes());
    }
    hash_len(
        &mut hash,
        key.flavour_flow().len(),
        "current flavour ancestry",
    )?;
    for value in key.flavour_flow() {
        hash.update(value.to_le_bytes());
    }
    hash.update(key.quantum_number_flow_id().to_le_bytes());
    hash_len(
        &mut hash,
        key.coupling_orders().len(),
        "current coupling orders",
    )?;
    for value in key.coupling_orders() {
        hash.update(value.to_le_bytes());
    }
    match key.source_binding() {
        CurrentSourceBinding::None => hash.update([0]),
        CurrentSourceBinding::FixedTemplate(id) => {
            hash.update([1]);
            hash.update(id.to_le_bytes());
        }
        CurrentSourceBinding::RuntimeDispatch { .. } => {
            return Err(integrity(
                "selected on-the-fly current unexpectedly uses runtime source dispatch",
            ));
        }
    }
    hash.update(
        key.propagator_template_id()
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    hash.update(color.output_color_shape_id().to_le_bytes());
    hash_len(
        &mut hash,
        color.components().len(),
        "current color components",
    )?;
    for component in color.components() {
        hash.update([component.kind() as u8]);
        hash_len(&mut hash, component.source_slots().len(), "color component")?;
        for slot in component.source_slots() {
            hash.update(slot.to_le_bytes());
        }
    }
    hash_len(
        &mut hash,
        color.result_port_bindings().len(),
        "current color ports",
    )?;
    for binding in color.result_port_bindings() {
        hash.update(binding.component_index().to_le_bytes());
        hash.update([binding.endpoint() as u8]);
    }
    final_digest(hash)
}

pub(super) fn multiset_digest(
    domain: &[u8],
    mut rows: Vec<SemanticDigest>,
) -> RusticolResult<SemanticDigest> {
    rows.sort_unstable();
    let mut hash = Sha256::new();
    hash.update(domain);
    hash_len(&mut hash, rows.len(), "proof rows")?;
    for row in rows {
        hash_digest(&mut hash, row);
    }
    final_digest(hash)
}

pub(super) fn contribution_proof_digest(
    result_digest: SemanticDigest,
    parent_digests: [SemanticDigest; 2],
    key: &ContributionKey,
    factor: ExactComplexRational,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-contribution-row-v1\0");
    hash_digest(&mut hash, result_digest);
    hash_digest(&mut hash, parent_digests[0]);
    hash_digest(&mut hash, parent_digests[1]);
    hash.update(key.transition_template_id().to_le_bytes());
    hash.update(key.result_state_template_id().to_le_bytes());
    hash.update(key.quantum_flow_witness_id().to_le_bytes());
    hash.update(
        key.color_witness_term_id()
            .color_contraction_template_id()
            .to_le_bytes(),
    );
    hash.update(key.color_witness_term_id().witness_ordinal().to_le_bytes());
    hash_digest(&mut hash, key.runtime_coupling_binding_digest());
    hash.update(key.output_projection_id().to_le_bytes());
    hash_exact(&mut hash, factor);
    final_digest(hash)
}

fn closure_proof_digest(
    parent_digests: [SemanticDigest; 2],
    closure: &PendingClosure,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-closure-row-v1\0");
    hash_digest(&mut hash, parent_digests[0]);
    hash_digest(&mut hash, parent_digests[1]);
    hash.update(closure.key.closure_template_id.to_le_bytes());
    hash.update(
        closure
            .key
            .quantum_flow_template_id
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    hash.update(
        closure
            .key
            .color_witness_term_id
            .color_contraction_template_id()
            .to_le_bytes(),
    );
    hash.update(
        closure
            .key
            .color_witness_term_id
            .witness_ordinal()
            .to_le_bytes(),
    );
    hash_exact(&mut hash, closure.factor);
    hash_len(
        &mut hash,
        closure.component_coefficients.len(),
        "closure component factors",
    )?;
    for coefficient in &closure.component_coefficients {
        hash_exact(&mut hash, *coefficient);
    }
    final_digest(hash)
}

fn pairing_owner_association_multiset_digest(
    closure_proof_rows: &[SemanticDigest],
    pairing_owners: &[ResolvedPairingOwnerV1],
) -> RusticolResult<SemanticDigest> {
    if closure_proof_rows.len() != pairing_owners.len() {
        return Err(integrity(
            "closure proof and pairing-owner domains have different sizes",
        ));
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(pairing_owners.len())
        .map_err(|error| invalid(format!("pairing-owner proof allocation failed: {error}")))?;
    for (closure_proof, owner) in closure_proof_rows.iter().zip(pairing_owners) {
        let mut hash = Sha256::new();
        hash.update(b"pyamplicol-on-the-fly-closure-pairing-owner-row-v1\0");
        hash_digest(&mut hash, *closure_proof);
        match owner.proof_digest {
            None => hash.update([0]),
            Some(digest) => {
                hash.update([1]);
                hash_digest(&mut hash, digest);
            }
        }
        rows.push(final_digest(hash)?);
    }
    multiset_digest(
        b"pyamplicol-on-the-fly-closure-pairing-owner-multiset-v1\0",
        rows,
    )
}

fn intern_factor(
    factor: ExactComplexRational,
    factors: &mut Vec<ExactComplexRational>,
    ids: &mut BTreeMap<ExactComplexRational, u32>,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(&factor).copied() {
        return Ok(id);
    }
    let id = checked_u32(factors.len(), "exact factor count")?;
    factors.push(factor);
    ids.insert(factor, id);
    Ok(id)
}

fn intern_momentum(
    form: &CanonicalMomentumLinearForm,
    forms: &mut Vec<CanonicalMomentumLinearForm>,
    ids: &mut BTreeMap<CanonicalMomentumLinearForm, u32>,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(form).copied() {
        return Ok(id);
    }
    let id = checked_u32(forms.len(), "momentum form count")?;
    forms.push(form.clone());
    ids.insert(form.clone(), id);
    Ok(id)
}

fn operation_key(
    seed: &OnTheFlyProcessSeedV1,
    role: DirectExecutorRole,
    kind: OnTheFlyOperationKindV1,
    operation_id: u32,
    operation_semantic_digest: SemanticDigest,
    evaluator_binding_id: u32,
    evaluator_binding_semantic_digest: SemanticDigest,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    OnTheFlyExecutorKeyV1::new(
        seed.direct_catalog_digest,
        role,
        kind,
        operation_id,
        operation_semantic_digest,
        evaluator_binding_id,
        evaluator_binding_semantic_digest,
    )
}

fn source_executor_key(
    grammar: &PreparedOnTheFlyGrammarV1,
    seed: &OnTheFlyProcessSeedV1,
    source_template_id: u32,
    current_state_template_id: u32,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    let executor = grammar
        .sources
        .get(&(source_template_id, current_state_template_id))
        .ok_or_else(|| integrity("source executor contract is absent"))?
        .executor;
    operation_key(
        seed,
        DirectExecutorRole::Source,
        OnTheFlyOperationKindV1::Source,
        executor.operation_id,
        executor.operation_semantic_digest,
        executor.evaluator_binding_id,
        executor.evaluator_binding_semantic_digest,
    )
}

fn transition_executor_key(
    seed: &OnTheFlyProcessSeedV1,
    transition: &PreparedTransition,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    operation_key(
        seed,
        DirectExecutorRole::Contribution,
        OnTheFlyOperationKindV1::Transition,
        transition.row.id,
        transition.transition_semantic_digest,
        transition.row.evaluator_binding_id,
        transition.evaluator_binding_digest,
    )
}

fn propagator_executor_key(
    grammar: &PreparedOnTheFlyGrammarV1,
    seed: &OnTheFlyProcessSeedV1,
    propagator_template_id: u32,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    let executor = grammar
        .propagator_executors
        .get(&propagator_template_id)
        .ok_or_else(|| integrity("propagator executor contract is absent"))?;
    operation_key(
        seed,
        DirectExecutorRole::Finalization,
        OnTheFlyOperationKindV1::Propagator,
        executor.operation_id,
        executor.operation_semantic_digest,
        executor.evaluator_binding_id,
        executor.evaluator_binding_semantic_digest,
    )
}

fn closure_executor_key(
    seed: &OnTheFlyProcessSeedV1,
    closure: &PreparedClosure,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    operation_key(
        seed,
        DirectExecutorRole::Closure,
        OnTheFlyOperationKindV1::Closure,
        closure.row.id,
        closure.closure_semantic_digest,
        closure.row.evaluator_binding_id,
        closure.evaluator_binding_digest,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_trace(
    templates: &ValidatedRecurrenceTemplateInput,
    grammar: &PreparedOnTheFlyGrammarV1,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    colors: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    closures: &[PendingClosure],
    pairing_owners: Vec<ResolvedPairingOwnerV1>,
    live: &BTreeSet<u32>,
    constructed_contribution_count: usize,
) -> RusticolResult<OnTheFlyStructuralTraceV1> {
    if pairing_owners.len() != closures.len() {
        return Err(integrity(
            "selected closures and resolved pairing owners have different sizes",
        ));
    }
    let input = templates.input();
    let mut old_to_new = BTreeMap::new();
    let mut current_keys = Vec::with_capacity(live.len());
    let mut current_colors = Vec::with_capacity(live.len());
    let mut component_bases = BTreeMap::new();
    let mut component_counts = BTreeMap::new();
    let mut current_component_count = 0u32;
    for old_id in live.iter().copied() {
        let current = currents
            .get(old_id as usize)
            .ok_or_else(|| integrity("live current is absent"))?;
        let state = input
            .current_states
            .get(current.key.current_state_template_id() as usize)
            .ok_or_else(|| integrity("live current state template is absent"))?;
        let count = u16::try_from(state.dimension)
            .map_err(|_| invalid("on-the-fly current component count exceeds u16"))?;
        let new_id = checked_u32(current_keys.len(), "live current count")?;
        old_to_new.insert(old_id, new_id);
        component_bases.insert(old_id, current_component_count);
        component_counts.insert(old_id, count);
        current_component_count = current_component_count
            .checked_add(u32::from(count))
            .ok_or_else(|| invalid("on-the-fly current arena exceeds u32 components"))?;
        current_keys.push(current.key.clone());
        current_colors.push(
            colors
                .get(current.key.dynamic_lc_color_state_id())
                .ok_or_else(|| integrity("live current color state is absent"))?
                .clone(),
        );
    }

    let mut momentum_forms = Vec::new();
    let mut momentum_ids = BTreeMap::new();
    for old_id in live.iter().copied() {
        intern_momentum(
            currents[old_id as usize].key.momentum(),
            &mut momentum_forms,
            &mut momentum_ids,
        )?;
    }
    let mut exact_factors = Vec::new();
    let mut factor_ids = BTreeMap::new();
    intern_factor(
        ExactComplexRational::ONE,
        &mut exact_factors,
        &mut factor_ids,
    )?;
    let mut operations = Vec::new();

    for old_id in live
        .iter()
        .copied()
        .filter(|old_id| currents[*old_id as usize].key.node_kind() == RecurrenceNodeKind::Source)
    {
        let current = &currents[old_id as usize];
        let source_template_id = match current.key.source_binding() {
            CurrentSourceBinding::FixedTemplate(id) => *id,
            _ => return Err(integrity("selected source has no fixed template binding")),
        };
        operations.push(OnTheFlyTraceOperationV1::Source {
            key: source_executor_key(
                grammar,
                seed,
                source_template_id,
                current.key.current_state_template_id(),
            )?,
            row: DirectSourceRow {
                source_slot: current.key.support_source_slots()[0],
                destination_component_base: component_bases[&old_id],
                momentum_form_id: momentum_ids[current.key.momentum()],
                source_template_or_dispatch_domain: source_template_id,
                spin_state_class: current.key.spin_state_class(),
                exact_factor_id: intern_factor(
                    current
                        .source_factor
                        .ok_or_else(|| integrity("source current has no exact factor"))?,
                    &mut exact_factors,
                    &mut factor_ids,
                )?,
                selector_domain_id: 0,
            },
        });
    }

    let maximum_stage = live
        .iter()
        .map(|id| currents[*id as usize].stage)
        .max()
        .unwrap_or(0);
    let mut contribution_proof_rows = Vec::new();
    let mut family_contribution_rows = Vec::new();
    let current_digest_by_old = live
        .iter()
        .copied()
        .map(|old_id| {
            let current = &currents[old_id as usize];
            Ok((
                old_id,
                hash_current_key(
                    &current.key,
                    colors
                        .get(current.key.dynamic_lc_color_state_id())
                        .ok_or_else(|| integrity("proof current color state is absent"))?,
                )?,
            ))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;

    for stage in 1..=maximum_stage {
        for old_id in live
            .iter()
            .copied()
            .filter(|old_id| currents[*old_id as usize].stage == stage)
        {
            let current = &currents[old_id as usize];
            let mut initialized = false;
            for (pending, factor) in current.contributions.iter().filter(|(pending, factor)| {
                !factor.is_zero()
                    && pending
                        .parent_current_ids
                        .iter()
                        .all(|parent| live.contains(parent))
            }) {
                let transition_id = pending.key.transition_template_id();
                let prepared = grammar
                    .transitions
                    .values()
                    .flatten()
                    .find(|prepared| prepared.row.id == transition_id)
                    .ok_or_else(|| integrity("prepared live transition disappeared"))?;
                let parent_ids = pending.parent_current_ids;
                operations.push(OnTheFlyTraceOperationV1::Contribution {
                    key: transition_executor_key(seed, prepared)?,
                    row: DirectContributionRow {
                        parent0_component_base: component_bases[&parent_ids[0]],
                        parent1_component_base_or_sentinel: component_bases[&parent_ids[1]],
                        parent0_momentum_form_id: momentum_ids
                            [currents[parent_ids[0] as usize].key.momentum()],
                        parent1_momentum_form_id_or_sentinel: momentum_ids
                            [currents[parent_ids[1] as usize].key.momentum()],
                        destination_component_base: component_bases[&old_id],
                        exact_factor_id: intern_factor(
                            *factor,
                            &mut exact_factors,
                            &mut factor_ids,
                        )?,
                        selector_domain_id: 0,
                        flags: if initialized {
                            0
                        } else {
                            DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
                        },
                    },
                });
                initialized = true;
                contribution_proof_rows.push(contribution_proof_digest(
                    current_digest_by_old[&old_id],
                    [
                        current_digest_by_old[&parent_ids[0]],
                        current_digest_by_old[&parent_ids[1]],
                    ],
                    &pending.key,
                    *factor,
                )?);
                {
                    let local_parent_ids = [old_to_new[&parent_ids[0]], old_to_new[&parent_ids[1]]];
                    family_contribution_rows.push(OnTheFlyTraceContributionProofRowV1 {
                        result_current_id: old_to_new[&old_id],
                        parent_current_ids: local_parent_ids,
                        key: ContributionKey::new(
                            pending.key.transition_template_id(),
                            local_parent_ids.to_vec(),
                            pending.key.parent_state_template_ids().to_vec(),
                            pending.key.parent_momenta().to_vec(),
                            pending.key.result_state_template_id(),
                            pending.key.quantum_flow_witness_id(),
                            pending.key.color_witness_term_id(),
                            pending.key.runtime_coupling_binding_digest(),
                            pending.key.output_projection_id(),
                        )?,
                        exact_factor: *factor,
                    });
                }
            }
            if !initialized {
                return Err(integrity(
                    "live non-source current has no live contribution",
                ));
            }
            if let Some(propagator_id) = current.key.propagator_template_id() {
                let component_count = component_counts[&old_id];
                operations.push(OnTheFlyTraceOperationV1::Finalization {
                    key: propagator_executor_key(grammar, seed, propagator_id)?,
                    row: DirectFinalizationRow {
                        component_base: component_bases[&old_id],
                        component_count,
                        momentum_form_id: momentum_ids[current.key.momentum()],
                        exact_factor_id: factor_ids[&ExactComplexRational::ONE],
                        selector_domain_id: 0,
                        flags: 0,
                    },
                });
            }
        }
    }

    let mut closure_proof_rows = Vec::new();
    for closure in closures {
        let prepared = grammar
            .closures
            .values()
            .flatten()
            .find(|prepared| prepared.row.id == closure.key.closure_template_id)
            .ok_or_else(|| integrity("prepared live closure disappeared"))?;
        let parent_ids = closure.key.parent_current_ids;
        let parent0_count = component_counts[&parent_ids[0]];
        let parent1_count = component_counts[&parent_ids[1]];
        if parent0_count != parent1_count
            || closure.component_coefficients.len() != usize::from(parent0_count)
        {
            return Err(integrity(
                "closure component coefficients do not match parent dimensions",
            ));
        }
        let component_factor_start = checked_u32(exact_factors.len(), "closure factor start")?;
        exact_factors.extend_from_slice(&closure.component_coefficients);
        operations.push(OnTheFlyTraceOperationV1::Closure {
            key: closure_executor_key(seed, prepared)?,
            row: DirectClosureRow {
                parent0_component_base: component_bases[&parent_ids[0]],
                parent1_component_base_or_sentinel: component_bases[&parent_ids[1]],
                parent0_momentum_form_id: momentum_ids
                    [currents[parent_ids[0] as usize].key.momentum()],
                parent1_momentum_form_id_or_sentinel: momentum_ids
                    [currents[parent_ids[1] as usize].key.momentum()],
                amplitude_destination_id: 0,
                exact_factor_id: intern_factor(
                    closure.factor,
                    &mut exact_factors,
                    &mut factor_ids,
                )?,
                component_factor_start,
                component_count: parent0_count,
                selector_domain_id: 0,
                flags: 0,
            },
        });
        closure_proof_rows.push(closure_proof_digest(
            [
                current_digest_by_old[&parent_ids[0]],
                current_digest_by_old[&parent_ids[1]],
            ],
            closure,
        )?);
    }

    let current_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-current-multiset-v1\0",
        current_digest_by_old.values().copied().collect(),
    )?;
    let contribution_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-contribution-multiset-v1\0",
        contribution_proof_rows,
    )?;
    let pairing_owner_multiset_digest =
        pairing_owner_association_multiset_digest(&closure_proof_rows, &pairing_owners)?;
    let closure_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-closure-multiset-v1\0",
        closure_proof_rows,
    )?;
    let mut owner_hash = Sha256::new();
    owner_hash.update(b"pyamplicol-on-the-fly-owner-v2\0");
    hash_digest(&mut owner_hash, seed.semantic_digest());
    hash_digest(&mut owner_hash, query.semantic_digest());
    owner_hash.update(query.closure_anchor_slot.to_le_bytes());
    hash_digest(&mut owner_hash, query.selector_digest);
    hash_digest(&mut owner_hash, pairing_owner_multiset_digest);
    let owner_digest = final_digest(owner_hash)?;
    let mut proof_hash = Sha256::new();
    proof_hash.update(ON_THE_FLY_PROOF_DOMAIN);
    for digest in [
        current_multiset_digest,
        contribution_multiset_digest,
        closure_multiset_digest,
        owner_digest,
    ] {
        hash_digest(&mut proof_hash, digest);
    }
    let live_contribution_count = operations
        .iter()
        .filter(|operation| matches!(operation, OnTheFlyTraceOperationV1::Contribution { .. }))
        .count();
    for value in [
        live.len(),
        live_contribution_count,
        closures.len(),
        currents.len(),
        constructed_contribution_count,
    ] {
        proof_hash.update(checked_u32(value, "structural proof count")?.to_le_bytes());
    }
    let proof = OnTheFlyStructuralProofV1 {
        current_count: checked_u32(live.len(), "live current count")?,
        contribution_count: checked_u32(live_contribution_count, "live contribution count")?,
        closure_count: checked_u32(closures.len(), "live closure count")?,
        constructed_current_count: checked_u32(currents.len(), "constructed current count")?,
        constructed_contribution_count: checked_u32(
            constructed_contribution_count,
            "constructed contribution count",
        )?,
        current_multiset_digest,
        contribution_multiset_digest,
        closure_multiset_digest,
        owner_digest,
        semantic_digest: final_digest(proof_hash)?,
    };
    let exact_factor_count = checked_u32(exact_factors.len(), "exact factor count")?;
    let layout = OnTheFlyWorkspaceLayoutV1 {
        source_count: checked_u32(query.selected_sources.len(), "selected source count")?,
        lorentz_component_count: 4,
        parameter_count: checked_u32(templates.input().parameters.len(), "parameter count")?,
        current_component_count,
        amplitude_component_count: 1,
        momentum_form_count: checked_u32(momentum_ids.len(), "momentum form count")?,
        exact_factor_count,
    };
    let mut trace_hash = Sha256::new();
    trace_hash.update(b"pyamplicol-on-the-fly-structural-trace-v1\0");
    hash_digest(&mut trace_hash, seed.semantic_digest());
    hash_digest(&mut trace_hash, query.semantic_digest());
    hash_digest(&mut trace_hash, proof.semantic_digest());
    for value in [
        layout.source_count,
        u32::from(layout.lorentz_component_count),
        layout.parameter_count,
        layout.current_component_count,
        layout.amplitude_component_count,
        layout.momentum_form_count,
        layout.exact_factor_count,
    ] {
        trace_hash.update(value.to_le_bytes());
    }
    let semantic_digest = final_digest(trace_hash)?;
    let current_component_ranges = live
        .iter()
        .copied()
        .map(|old_id| {
            [
                component_bases[&old_id],
                u32::from(component_counts[&old_id]),
            ]
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let current_semantic_digests = live
        .iter()
        .copied()
        .map(|old_id| current_digest_by_old[&old_id])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(OnTheFlyStructuralTraceV1 {
        seed_digest: seed.semantic_digest(),
        query_digest: query.semantic_digest(),
        current_keys: current_keys.into_boxed_slice(),
        current_colors: current_colors.into_boxed_slice(),
        operations: operations.into_boxed_slice(),
        momentum_forms: momentum_forms.into_boxed_slice(),
        exact_factors: exact_factors.into_boxed_slice(),
        pairing_owners: pairing_owners.into_boxed_slice(),
        layout,
        proof,
        semantic_digest,
        prepared_executor_rows_bound: false,
        current_component_ranges,
        current_semantic_digests,
        contribution_proof_rows: family_contribution_rows.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn pairing_owner(byte: u8) -> ResolvedPairingOwnerV1 {
        ResolvedPairingOwnerV1 {
            endpoint_pairs: Vec::new(),
            proof_digest: Some(digest(byte)),
            source_slot_permutation: Vec::new(),
            source_lineage: Vec::new(),
            fermion_parity: 1,
        }
    }

    #[test]
    fn pairing_owner_proof_binds_closure_associations_order_independently() {
        let closures = [digest(1), digest(2)];
        let owners = [pairing_owner(3), pairing_owner(4)];
        let expected = pairing_owner_association_multiset_digest(&closures, &owners).unwrap();
        assert_eq!(
            expected,
            pairing_owner_association_multiset_digest(
                &[closures[1], closures[0]],
                &[pairing_owner(4), pairing_owner(3)],
            )
            .unwrap(),
        );
        assert_ne!(
            expected,
            pairing_owner_association_multiset_digest(
                &closures,
                &[pairing_owner(4), pairing_owner(3)],
            )
            .unwrap(),
        );
        assert_ne!(
            pairing_owner_association_multiset_digest(&[closures[0]], &[pairing_owner(3)],)
                .unwrap(),
            pairing_owner_association_multiset_digest(
                &[closures[0], closures[0]],
                &[pairing_owner(3), pairing_owner(3)],
            )
            .unwrap(),
        );
        assert!(pairing_owner_association_multiset_digest(&closures, &owners[..1]).is_err());
    }

    #[test]
    fn logical_batch_capacity_belongs_to_workspace_not_structural_trace() {
        let trace = OnTheFlyStructuralTraceV1 {
            seed_digest: digest(1),
            query_digest: digest(2),
            current_keys: Box::new([]),
            current_colors: Box::new([]),
            operations: Box::new([]),
            momentum_forms: vec![
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: 0,
                    coefficient: 1,
                }])
                .unwrap(),
            ]
            .into_boxed_slice(),
            exact_factors: vec![ExactComplexRational::ONE].into_boxed_slice(),
            pairing_owners: Box::new([]),
            layout: OnTheFlyWorkspaceLayoutV1 {
                source_count: 1,
                lorentz_component_count: 4,
                parameter_count: 0,
                current_component_count: 1,
                amplitude_component_count: 1,
                momentum_form_count: 1,
                exact_factor_count: 1,
            },
            proof: OnTheFlyStructuralProofV1 {
                current_count: 0,
                contribution_count: 0,
                closure_count: 0,
                constructed_current_count: 0,
                constructed_contribution_count: 0,
                current_multiset_digest: digest(3),
                contribution_multiset_digest: digest(4),
                closure_multiset_digest: digest(5),
                owner_digest: digest(6),
                semantic_digest: digest(7),
            },
            semantic_digest: digest(8),
            prepared_executor_rows_bound: false,
            current_component_ranges: Box::new([]),
            current_semantic_digests: Box::new([]),
            contribution_proof_rows: Box::new([]),
        };
        let small = OnTheFlyWorkspaceV1::new(&trace, 1).unwrap();
        let large = OnTheFlyWorkspaceV1::new(&trace, 65).unwrap();
        assert_ne!(small.point_stride(), large.point_stride());
        assert_eq!(trace.semantic_digest(), digest(8));
    }
}
