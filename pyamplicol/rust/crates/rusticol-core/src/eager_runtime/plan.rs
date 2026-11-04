// SPDX-License-Identifier: 0BSD

use super::{
    EagerComplex64, EagerDirectClosureSpec, EagerKernelInput, EagerKernelRole, EagerKernelSpec,
    EagerPlanDefinition, EagerPlanPayloads, EagerReductionEntry, EagerReductionGroup,
    EagerSelectorPayloads, EagerStagePayload,
};
use crate::{
    EAGER_OUTPUT_FACTOR_NONE, EagerAttachmentRow, EagerClosureRow, EagerCouplingRow,
    EagerFinalizationRow, EagerInvocationRow, EagerSelectorDomainIdRow, EagerSelectorDomainRow,
    EagerSelectorGroupRow, MISSING_U32, RusticolError, RusticolResult,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ComponentRange {
    pub(super) start: usize,
    pub(super) len: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ComponentLayout {
    slots: Vec<ComponentRange>,
    pub(super) component_count: usize,
}

impl ComponentLayout {
    pub(super) fn new(name: &str, component_counts: &[u32]) -> RusticolResult<Self> {
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(component_counts.len())
            .map_err(|error| {
                RusticolError::artifact(format!(
                    "could not reserve eager {name} slot layout: {error}"
                ))
            })?;
        let mut start = 0usize;
        for (slot_id, count) in component_counts.iter().copied().enumerate() {
            let len = usize::try_from(count).map_err(|_| {
                RusticolError::artifact(format!(
                    "eager {name} slot {slot_id} component count does not fit usize"
                ))
            })?;
            if len == 0 {
                return Err(RusticolError::artifact(format!(
                    "eager {name} slot {slot_id} has zero components"
                )));
            }
            slots.push(ComponentRange { start, len });
            start = start.checked_add(len).ok_or_else(|| {
                RusticolError::artifact(format!("eager {name} component layout overflows usize"))
            })?;
        }
        Ok(Self {
            slots,
            component_count: start,
        })
    }

    pub(super) fn get(&self, id: u32, context: &str) -> RusticolResult<ComponentRange> {
        let index = usize::try_from(id).map_err(|_| {
            RusticolError::artifact(format!("{context} id {id} does not fit usize"))
        })?;
        self.slots.get(index).copied().ok_or_else(|| {
            RusticolError::artifact(format!("{context} references unknown slot {id}"))
        })
    }

    pub(super) fn id_for_range(&self, range: ComponentRange, context: &str) -> RusticolResult<u32> {
        self.slots
            .iter()
            .position(|candidate| *candidate == range)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| {
                RusticolError::internal(format!(
                    "{context} has no matching semantic component range"
                ))
            })
    }
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledInvocation {
    pub(super) row: EagerInvocationRow,
    pub(super) left_values: ComponentRange,
    pub(super) right_values: ComponentRange,
    pub(super) left_momenta: ComponentRange,
    pub(super) right_momenta: ComponentRange,
    pub(super) attachment_range: Range<usize>,
    pub(super) selector_domain_id: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ScheduledAttachment {
    pub(super) row: EagerAttachmentRow,
    pub(super) current: ComponentRange,
    pub(super) selector_domain_id: Option<u32>,
    pub(super) initializes_current: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FinalizationCopy {
    pub(super) current: ComponentRange,
    pub(super) unpropagated: ComponentRange,
    pub(super) selector_domain_id: Option<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledFinalization {
    pub(super) row: EagerFinalizationRow,
    pub(super) current: ComponentRange,
    pub(super) propagated: Option<ComponentRange>,
    pub(super) momentum: ComponentRange,
    pub(super) selector_domain_id: Option<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledClosure {
    pub(super) row: EagerClosureRow,
    pub(super) left_values: ComponentRange,
    pub(super) right_values: ComponentRange,
    pub(super) selector_domain_id: Option<u32>,
    pub(super) initializes_amplitude: bool,
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledDirectClosure {
    pub(super) row: EagerClosureRow,
    pub(super) left_values: ComponentRange,
    pub(super) right_values: ComponentRange,
    pub(super) coefficients: Vec<EagerComplex64>,
    pub(super) selector_domain_id: Option<u32>,
    pub(super) initializes_amplitude: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ClosureExecutionRows<'a> {
    pub(super) closures: &'a [ScheduledClosure],
    pub(super) direct_closures: &'a [ScheduledDirectClosure],
    pub(super) kernels: &'a BTreeMap<u32, EagerKernelSpec>,
}

#[derive(Clone, Debug)]
pub(super) struct SelectorDomainPlan {
    pub(super) memberships: Vec<Vec<u32>>,
    pub(super) group_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
struct DecodedSelectorStage {
    stage_index: u32,
    invocation_domains: Vec<u32>,
    attachment_domains: Vec<u32>,
    unpropagated_finalization_domains: Vec<u32>,
    propagated_finalization_domains: Vec<u32>,
}

#[derive(Clone, Debug)]
struct DecodedSelectorPayload {
    plan: SelectorDomainPlan,
    stages: Vec<DecodedSelectorStage>,
    closure_domains: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct EagerStagePlan {
    pub(super) stage_index: u32,
    pub(super) current_component_count: usize,
    pub(super) invocations: Vec<ScheduledInvocation>,
    pub(super) attachments: Vec<ScheduledAttachment>,
    pub(super) finalization_copies: Vec<FinalizationCopy>,
    pub(super) finalizations: Vec<ScheduledFinalization>,
    pub(super) zero_current_ranges: Vec<ComponentRange>,
}

#[derive(Clone, Debug)]
pub struct EagerExecutionPlan {
    pub(super) values: ComponentLayout,
    pub(super) momenta: ComponentLayout,
    pub(super) currents: ComponentLayout,
    pub(super) parameter_count: usize,
    pub(super) amplitude_count: usize,
    pub(super) kernels: BTreeMap<u32, EagerKernelSpec>,
    pub(super) couplings: Vec<EagerCouplingRow>,
    pub(super) stages: Vec<EagerStagePlan>,
    pub(super) closures: Vec<ScheduledClosure>,
    pub(super) direct_closures: Vec<ScheduledDirectClosure>,
    pub(super) reduction_groups: Vec<EagerReductionGroup>,
    pub(super) reduction_entries: Vec<EagerReductionEntry>,
    pub(super) selector_domains: Option<SelectorDomainPlan>,
    pub(super) initial_value_ranges: Vec<ComponentRange>,
    pub(super) zero_amplitude_indices: Vec<usize>,
}

impl EagerExecutionPlan {
    pub fn from_payloads(
        definition: EagerPlanDefinition,
        payloads: EagerPlanPayloads<'_>,
    ) -> RusticolResult<Self> {
        let values =
            ComponentLayout::new("value", &definition.dimensions.value_slot_component_counts)?;
        let momenta = ComponentLayout::new(
            "momentum",
            &definition.dimensions.momentum_slot_component_counts,
        )?;
        let currents =
            ComponentLayout::new("current", &definition.dimensions.current_component_counts)?;
        let parameter_count = usize::try_from(definition.dimensions.parameter_count)
            .map_err(|_| RusticolError::artifact("eager parameter count does not fit usize"))?;
        let amplitude_count = usize::try_from(definition.dimensions.amplitude_count)
            .map_err(|_| RusticolError::artifact("eager amplitude count does not fit usize"))?;
        if amplitude_count == 0 {
            return Err(RusticolError::artifact(
                "eager execution plan has no amplitude outputs",
            ));
        }

        let kernels = validate_kernel_specs(&definition.kernels, parameter_count)?;
        let couplings = EagerCouplingRow::decode_table(payloads.couplings)?;
        validate_couplings(&couplings, parameter_count)?;
        let selector_payload =
            decode_selector_payloads(payloads.selector_domains, payloads.stages)?;

        let mut stages = Vec::new();
        stages
            .try_reserve_exact(payloads.stages.len())
            .map_err(|error| {
                RusticolError::artifact(format!("could not reserve eager stages: {error}"))
            })?;
        let mut previous_stage = None;
        let mut finalized_currents = BTreeSet::new();
        let mut stored_value_slots = BTreeSet::new();
        for (stage_position, payload) in payloads.stages.iter().enumerate() {
            if previous_stage.is_some_and(|previous| payload.stage_index <= previous) {
                return Err(RusticolError::artifact(
                    "eager stage indices must be strictly increasing",
                ));
            }
            previous_stage = Some(payload.stage_index);
            stages.push(load_stage(
                *payload,
                selector_payload
                    .as_ref()
                    .map(|selector| &selector.stages[stage_position]),
                selector_payload.as_ref().map(|selector| &selector.plan),
                &values,
                &momenta,
                &currents,
                &kernels,
                couplings.len(),
                parameter_count,
                &mut finalized_currents,
                &mut stored_value_slots,
            )?);
        }

        let closure_rows = EagerClosureRow::decode_table(payloads.closures)?;
        let closure_domains = selector_payload
            .as_ref()
            .map(|selector| selector.closure_domains.as_slice());
        if closure_domains.is_some_and(|domains| domains.len() != closure_rows.len()) {
            return Err(RusticolError::artifact(
                "eager closure selector-domain count does not match the closure table",
            ));
        }
        let (mut closures, mut direct_closures) = load_closures(
            &closure_rows,
            closure_domains,
            &definition.direct_closures,
            &values,
            &kernels,
            couplings.len(),
            amplitude_count,
            parameter_count,
        )?;
        validate_reduction_plan(
            &definition.reduction_groups,
            &definition.reduction_entries,
            amplitude_count,
        )?;
        if let Some(selector) = selector_payload.as_ref() {
            validate_selector_dependency_proof(
                selector,
                payloads.stages,
                &closure_rows,
                &definition.reduction_groups,
                amplitude_count,
            )?;
        }

        let initial_value_ranges = values
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot_id, range)| {
                (!stored_value_slots.contains(&(slot_id as u32))).then_some(*range)
            })
            .collect();
        let zero_amplitude_indices =
            mark_initial_amplitude_writes(&mut closures, &mut direct_closures, amplitude_count);

        Ok(Self {
            values,
            momenta,
            currents,
            parameter_count,
            amplitude_count,
            kernels,
            couplings,
            stages,
            closures,
            direct_closures,
            reduction_groups: definition.reduction_groups,
            reduction_entries: definition.reduction_entries,
            selector_domains: selector_payload.map(|selector| selector.plan),
            initial_value_ranges,
            zero_amplitude_indices,
        })
    }

    pub fn value_component_count(&self) -> usize {
        self.values.component_count
    }

    pub fn momentum_component_count(&self) -> usize {
        self.momenta.component_count
    }

    pub fn current_component_count(&self) -> usize {
        self.currents.component_count
    }

    pub(super) fn current_workspace_component_count(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.current_component_count)
            .max()
            .unwrap_or(0)
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub fn amplitude_count(&self) -> usize {
        self.amplitude_count
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn stage_indices(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.stages.iter().map(|stage| stage.stage_index)
    }

    pub fn invocation_count(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.invocations.len())
            .sum()
    }

    pub fn attachment_count(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.attachments.len())
            .sum()
    }

    pub fn closure_count(&self) -> usize {
        self.closures.len() + self.direct_closures.len()
    }

    pub fn reduction_group_count(&self) -> usize {
        self.reduction_groups.len()
    }

    pub fn reduction_entry_count(&self) -> usize {
        self.reduction_entries.len()
    }

    pub fn has_selector_domains(&self) -> bool {
        self.selector_domains.is_some()
    }

    pub(crate) fn selector_group_ids(&self) -> Option<&[u32]> {
        Some(&self.selector_domains.as_ref()?.group_ids)
    }
}

fn decode_selector_payloads(
    payloads: Option<EagerSelectorPayloads<'_>>,
    execution_stages: &[EagerStagePayload<'_>],
) -> RusticolResult<Option<DecodedSelectorPayload>> {
    let Some(payloads) = payloads else {
        return Ok(None);
    };
    if payloads.stages.len() != execution_stages.len() {
        return Err(RusticolError::artifact(format!(
            "eager selector domains cover {} stages, expected {}",
            payloads.stages.len(),
            execution_stages.len()
        )));
    }

    let domain_rows = EagerSelectorDomainRow::decode_table(payloads.domains)?;
    let group_rows = EagerSelectorGroupRow::decode_table(payloads.domain_group_ids)?;
    if domain_rows.is_empty() {
        return Err(RusticolError::artifact(
            "eager selector-domain table is empty",
        ));
    }
    let mut memberships = Vec::new();
    memberships
        .try_reserve_exact(domain_rows.len())
        .map_err(|error| {
            RusticolError::artifact(format!("could not reserve eager selector domains: {error}"))
        })?;
    let mut cursor = 0usize;
    let mut unique_memberships = BTreeSet::new();
    for (domain_id, row) in domain_rows.into_iter().enumerate() {
        let start = usize::try_from(row.member_start).map_err(|_| {
            RusticolError::artifact(format!(
                "eager selector domain {domain_id} start does not fit usize"
            ))
        })?;
        let count = usize::try_from(row.member_count).map_err(|_| {
            RusticolError::artifact(format!(
                "eager selector domain {domain_id} count does not fit usize"
            ))
        })?;
        if start != cursor {
            return Err(RusticolError::artifact(format!(
                "eager selector domain {domain_id} starts at {start}, expected {cursor}"
            )));
        }
        let stop = start.checked_add(count).ok_or_else(|| {
            RusticolError::artifact(format!(
                "eager selector domain {domain_id} range overflows usize"
            ))
        })?;
        if stop > group_rows.len() {
            return Err(RusticolError::artifact(format!(
                "eager selector domain {domain_id} exceeds its membership table"
            )));
        }
        let members = group_rows[start..stop]
            .iter()
            .map(|row| row.coherent_group_id)
            .collect::<Vec<_>>();
        if members.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RusticolError::artifact(format!(
                "eager selector domain {domain_id} members are not sorted and unique"
            )));
        }
        if !unique_memberships.insert(members.clone()) {
            return Err(RusticolError::artifact(format!(
                "eager selector domain {domain_id} duplicates an earlier membership"
            )));
        }
        memberships.push(members);
        cursor = stop;
    }
    if cursor != group_rows.len() {
        return Err(RusticolError::artifact(
            "eager selector domains do not cover their membership table",
        ));
    }
    if !memberships.iter().any(Vec::is_empty) {
        return Err(RusticolError::artifact(
            "eager selector domains do not define the empty domain",
        ));
    }
    let domain_count = memberships.len();
    let mut group_ids = memberships
        .iter()
        .flat_map(|members| members.iter().copied())
        .collect::<Vec<_>>();
    group_ids.sort_unstable();
    group_ids.dedup();

    let mut stages = Vec::new();
    stages
        .try_reserve_exact(payloads.stages.len())
        .map_err(|error| {
            RusticolError::artifact(format!("could not reserve eager selector stages: {error}"))
        })?;
    for (position, (selector, execution)) in
        payloads.stages.iter().zip(execution_stages).enumerate()
    {
        if selector.stage_index != execution.stage_index {
            return Err(RusticolError::artifact(format!(
                "eager selector stage {} at position {position} does not match execution stage {}",
                selector.stage_index, execution.stage_index
            )));
        }
        stages.push(DecodedSelectorStage {
            stage_index: selector.stage_index,
            invocation_domains: decode_domain_ids(
                selector.invocation_domains,
                domain_count,
                "invocation",
            )?,
            attachment_domains: decode_domain_ids(
                selector.attachment_domains,
                domain_count,
                "attachment",
            )?,
            unpropagated_finalization_domains: decode_domain_ids(
                selector.unpropagated_finalization_domains,
                domain_count,
                "unpropagated finalization",
            )?,
            propagated_finalization_domains: decode_domain_ids(
                selector.propagated_finalization_domains,
                domain_count,
                "propagated finalization",
            )?,
        });
    }
    let closure_domains = decode_domain_ids(payloads.closure_domains, domain_count, "closure")?;
    Ok(Some(DecodedSelectorPayload {
        plan: SelectorDomainPlan {
            memberships,
            group_ids,
        },
        stages,
        closure_domains,
    }))
}

fn decode_domain_ids(
    payload: &[u8],
    domain_count: usize,
    context: &str,
) -> RusticolResult<Vec<u32>> {
    let rows = EagerSelectorDomainIdRow::decode_table(payload)?;
    let mut result = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        if usize::try_from(row.domain_id)
            .ok()
            .is_none_or(|domain_id| domain_id >= domain_count)
        {
            return Err(RusticolError::artifact(format!(
                "eager {context} selector row {index} references unknown domain {}",
                row.domain_id
            )));
        }
        result.push(row.domain_id);
    }
    Ok(result)
}

fn validate_selector_dependency_proof(
    selector: &DecodedSelectorPayload,
    stage_payloads: &[EagerStagePayload<'_>],
    closures: &[EagerClosureRow],
    reduction_groups: &[EagerReductionGroup],
    amplitude_count: usize,
) -> RusticolResult<()> {
    let mut group_by_amplitude = vec![None; amplitude_count];
    let known_groups = reduction_groups
        .iter()
        .map(|group| group.coherent_group_id)
        .collect::<BTreeSet<_>>();
    for group in reduction_groups {
        for amplitude_index in &group.amplitude_indices {
            let index = usize::try_from(*amplitude_index).map_err(|_| {
                RusticolError::artifact("eager reduction amplitude index does not fit usize")
            })?;
            group_by_amplitude[index] = Some(group.coherent_group_id);
        }
    }
    for members in &selector.plan.memberships {
        if let Some(unknown) = members.iter().find(|group| !known_groups.contains(group)) {
            return Err(RusticolError::artifact(format!(
                "eager selector domain references unknown coherent group {unknown}"
            )));
        }
    }

    let mut value_domains = BTreeMap::<u32, BTreeSet<u32>>::new();
    for (index, closure) in closures.iter().enumerate() {
        let amplitude_index = required_index(
            closure.amplitude_index,
            amplitude_count,
            "eager closure amplitude",
        )?;
        let group_id = group_by_amplitude[amplitude_index].ok_or_else(|| {
            RusticolError::artifact(format!(
                "eager closure amplitude {amplitude_index} has no coherent group"
            ))
        })?;
        let expected = BTreeSet::from([group_id]);
        validate_selector_domain_members(
            &selector.plan,
            selector.closure_domains[index],
            &expected,
            &format!("eager closure {index}"),
        )?;
        value_domains
            .entry(closure.left_value_slot_id)
            .or_default()
            .insert(group_id);
        value_domains
            .entry(closure.right_value_slot_id)
            .or_default()
            .insert(group_id);
    }

    for (payload, selector_stage) in stage_payloads.iter().zip(&selector.stages).rev() {
        let invocations = EagerInvocationRow::decode_table(payload.invocations)?;
        let attachments = EagerAttachmentRow::decode_table(payload.attachments)?;
        let finalizations = EagerFinalizationRow::decode_table(payload.finalizations)?;
        let mut current_domains = BTreeMap::<u32, BTreeSet<u32>>::new();

        for (index, finalization) in finalizations.iter().enumerate() {
            let unpropagated = if finalization.stores_unpropagated() {
                value_domains
                    .get(&finalization.unpropagated_value_slot_id)
                    .cloned()
                    .unwrap_or_default()
            } else {
                BTreeSet::new()
            };
            validate_selector_domain_members(
                &selector.plan,
                selector_stage.unpropagated_finalization_domains[index],
                &unpropagated,
                &format!(
                    "eager stage {} unpropagated finalization {index}",
                    payload.stage_index
                ),
            )?;
            let propagated = if finalization.stores_propagated() {
                value_domains
                    .get(&finalization.propagated_value_slot_id)
                    .cloned()
                    .unwrap_or_default()
            } else {
                BTreeSet::new()
            };
            validate_selector_domain_members(
                &selector.plan,
                selector_stage.propagated_finalization_domains[index],
                &propagated,
                &format!(
                    "eager stage {} propagated finalization {index}",
                    payload.stage_index
                ),
            )?;
            let mut current = unpropagated;
            current.extend(propagated);
            current_domains.insert(finalization.current_id, current);
        }

        let mut attachment_domains = Vec::with_capacity(attachments.len());
        for (index, attachment) in attachments.iter().enumerate() {
            let expected = current_domains
                .get(&attachment.result_current_id)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "eager stage {} attachment {index} has no finalized current",
                        payload.stage_index
                    ))
                })?;
            validate_selector_domain_members(
                &selector.plan,
                selector_stage.attachment_domains[index],
                expected,
                &format!("eager stage {} attachment {index}", payload.stage_index),
            )?;
            attachment_domains.push(expected.clone());
        }

        for (index, invocation) in invocations.iter().enumerate() {
            let start = usize::try_from(invocation.attachment_start).map_err(|_| {
                RusticolError::artifact("eager invocation attachment start does not fit usize")
            })?;
            let count = usize::try_from(invocation.attachment_count).map_err(|_| {
                RusticolError::artifact("eager invocation attachment count does not fit usize")
            })?;
            let stop = start.checked_add(count).ok_or_else(|| {
                RusticolError::artifact("eager invocation attachment range overflows usize")
            })?;
            let mut expected = BTreeSet::new();
            for domain in attachment_domains.get(start..stop).ok_or_else(|| {
                RusticolError::artifact("eager invocation attachment range exceeds its table")
            })? {
                expected.extend(domain);
            }
            validate_selector_domain_members(
                &selector.plan,
                selector_stage.invocation_domains[index],
                &expected,
                &format!("eager stage {} invocation {index}", payload.stage_index),
            )?;
            value_domains
                .entry(invocation.left_value_slot_id)
                .or_default()
                .extend(&expected);
            value_domains
                .entry(invocation.right_value_slot_id)
                .or_default()
                .extend(expected);
        }
    }
    Ok(())
}

fn validate_selector_domain_members(
    plan: &SelectorDomainPlan,
    domain_id: u32,
    expected: &BTreeSet<u32>,
    context: &str,
) -> RusticolResult<()> {
    let actual = plan
        .memberships
        .get(domain_id as usize)
        .ok_or_else(|| RusticolError::internal("selector domain escaped payload validation"))?;
    if !actual.iter().copied().eq(expected.iter().copied()) {
        return Err(RusticolError::artifact(format!(
            "{context} selector domain does not match its proven dependency closure"
        )));
    }
    Ok(())
}

fn validate_kernel_specs(
    specs: &[EagerKernelSpec],
    parameter_count: usize,
) -> RusticolResult<BTreeMap<u32, EagerKernelSpec>> {
    let mut kernels = BTreeMap::new();
    for (index, spec) in specs.iter().cloned().enumerate() {
        if spec.kernel_id == MISSING_U32 {
            return Err(RusticolError::artifact(format!(
                "eager kernel spec {index} uses the reserved missing id"
            )));
        }
        if spec.inputs.is_empty() || spec.output_component_count == 0 {
            return Err(RusticolError::artifact(format!(
                "eager kernel {} has a zero input or output width",
                spec.kernel_id
            )));
        }
        if spec.independent_block_size != 1 {
            if spec.independent_block_size != super::EAGER_INDEPENDENT_BLOCK_SIZE {
                return Err(RusticolError::artifact(format!(
                    "eager kernel {} has unsupported independent block size {}",
                    spec.kernel_id, spec.independent_block_size
                )));
            }
            if spec.role != EagerKernelRole::Vertex
                || spec.inputs.iter().any(|input| {
                    !matches!(
                        input,
                        EagerKernelInput::FirstCurrentComponent(_)
                            | EagerKernelInput::SecondCurrentComponent(_)
                    )
                })
            {
                return Err(RusticolError::artifact(format!(
                    "eager kernel {} applies an independent block to a non-current vertex contract",
                    spec.kernel_id
                )));
            }
        }
        let mut unique_inputs = BTreeSet::new();
        for input in &spec.inputs {
            if !unique_inputs.insert(*input) {
                return Err(RusticolError::artifact(format!(
                    "eager kernel {} repeats input descriptor {input:?}",
                    spec.kernel_id
                )));
            }
            if let EagerKernelInput::ModelParameter(parameter_id) = input {
                required_index(
                    *parameter_id,
                    parameter_count,
                    &format!("eager kernel {} model parameter", spec.kernel_id),
                )?;
            }
        }
        let kernel_id = spec.kernel_id;
        if kernels.insert(kernel_id, spec).is_some() {
            return Err(RusticolError::artifact(format!(
                "duplicate eager kernel id {kernel_id}"
            )));
        }
    }
    Ok(kernels)
}

fn validate_couplings(rows: &[EagerCouplingRow], parameter_count: usize) -> RusticolResult<()> {
    for (index, row) in rows.iter().enumerate() {
        validate_optional_index(
            row.real_parameter_id,
            parameter_count,
            &format!("eager coupling {index} real parameter"),
        )?;
        validate_optional_index(
            row.imag_parameter_id,
            parameter_count,
            &format!("eager coupling {index} imaginary parameter"),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_stage(
    payload: EagerStagePayload<'_>,
    selector_stage: Option<&DecodedSelectorStage>,
    selector_plan: Option<&SelectorDomainPlan>,
    values: &ComponentLayout,
    momenta: &ComponentLayout,
    currents: &ComponentLayout,
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    coupling_count: usize,
    parameter_count: usize,
    globally_finalized_currents: &mut BTreeSet<u32>,
    globally_stored_value_slots: &mut BTreeSet<u32>,
) -> RusticolResult<EagerStagePlan> {
    let rows = EagerInvocationRow::decode_table(payload.invocations)?;
    let attachments = EagerAttachmentRow::decode_table(payload.attachments)?;
    let finalization_rows = EagerFinalizationRow::decode_table(payload.finalizations)?;
    if let Some(selector) = selector_stage {
        if selector.stage_index != payload.stage_index {
            return Err(RusticolError::artifact(format!(
                "eager selector stage {} does not match execution stage {}",
                selector.stage_index, payload.stage_index
            )));
        }
        for (kind, actual, expected) in [
            ("invocation", selector.invocation_domains.len(), rows.len()),
            (
                "attachment",
                selector.attachment_domains.len(),
                attachments.len(),
            ),
            (
                "unpropagated finalization",
                selector.unpropagated_finalization_domains.len(),
                finalization_rows.len(),
            ),
            (
                "propagated finalization",
                selector.propagated_finalization_domains.len(),
                finalization_rows.len(),
            ),
        ] {
            if actual != expected {
                return Err(RusticolError::artifact(format!(
                    "eager stage {} {kind} selector-domain count is {actual}, expected {expected}",
                    payload.stage_index
                )));
            }
        }
    }
    let mut invocations = Vec::new();
    invocations.try_reserve_exact(rows.len()).map_err(|error| {
        RusticolError::artifact(format!("could not reserve eager invocations: {error}"))
    })?;
    let mut attachment_cursor = 0usize;
    let mut attached_currents = BTreeSet::new();
    let mut scheduled_attachments = Vec::new();
    scheduled_attachments
        .try_reserve_exact(attachments.len())
        .map_err(|error| {
            RusticolError::artifact(format!("could not reserve eager attachments: {error}"))
        })?;

    for (index, row) in rows.into_iter().enumerate() {
        let kernel = require_kernel(
            kernels,
            row.kernel_id,
            EagerKernelRole::Vertex,
            "invocation",
        )?;
        let left_values = values.get(row.left_value_slot_id, "eager invocation left value")?;
        let right_values = values.get(row.right_value_slot_id, "eager invocation right value")?;
        let left_momenta =
            momenta.get(row.left_momentum_slot_id, "eager invocation left momentum")?;
        let right_momenta = momenta.get(
            row.right_momentum_slot_id,
            "eager invocation right momentum",
        )?;
        let coupling_index = required_index(
            row.coupling_slot_id,
            coupling_count,
            "eager invocation coupling",
        )?;
        let _ = coupling_index;
        let attachment_start = usize::try_from(row.attachment_start).map_err(|_| {
            RusticolError::artifact(format!(
                "eager invocation {index} attachment start does not fit usize"
            ))
        })?;
        let attachment_count = usize::try_from(row.attachment_count).map_err(|_| {
            RusticolError::artifact(format!(
                "eager invocation {index} attachment count does not fit usize"
            ))
        })?;
        if attachment_count == 0 {
            return Err(RusticolError::artifact(format!(
                "eager invocation {index} has no attachments"
            )));
        }
        if attachment_start != attachment_cursor {
            return Err(RusticolError::artifact(format!(
                "eager invocation {index} attachment range is not contiguous"
            )));
        }
        let attachment_end = attachment_start
            .checked_add(attachment_count)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "eager invocation {index} attachment range overflows usize"
                ))
            })?;
        if attachment_end > attachments.len() {
            return Err(RusticolError::artifact(format!(
                "eager invocation {index} attachment range exceeds the table"
            )));
        }
        validate_kernel_inputs(
            kernel,
            KernelInputBounds {
                first_current: left_values.len,
                second_current: right_values.len,
                first_momentum: left_momenta.len,
                second_momentum: right_momenta.len,
                has_coupling: true,
                parameter_count,
            },
        )?;
        if let (Some(selector), Some(domains)) = (selector_stage, selector_plan) {
            let expected_members = selector.attachment_domains[attachment_start..attachment_end]
                .iter()
                .flat_map(|domain_id| domains.memberships[*domain_id as usize].iter().copied())
                .collect::<BTreeSet<_>>();
            let invocation_members = domains.memberships
                [selector.invocation_domains[index] as usize]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            if invocation_members != expected_members {
                return Err(RusticolError::artifact(format!(
                    "eager invocation {index} selector domain does not equal its attachment union"
                )));
            }
        }
        for (attachment_index, attachment) in attachments[attachment_start..attachment_end]
            .iter()
            .enumerate()
        {
            let attachment_index = attachment_start + attachment_index;
            let current = currents.get(
                attachment.result_current_id,
                "eager attachment result current",
            )?;
            if current.len
                != usize::try_from(kernel.output_component_count).map_err(|_| {
                    RusticolError::artifact("eager kernel output width does not fit usize")
                })?
            {
                return Err(RusticolError::artifact(format!(
                    "eager invocation kernel {} output width does not match current {}",
                    kernel.kernel_id, attachment.result_current_id
                )));
            }
            attached_currents.insert(attachment.result_current_id);
            scheduled_attachments.push(ScheduledAttachment {
                row: *attachment,
                current,
                selector_domain_id: selector_stage
                    .map(|selector| selector.attachment_domains[attachment_index]),
                initializes_current: false,
            });
        }
        invocations.push(ScheduledInvocation {
            row,
            left_values,
            right_values,
            left_momenta,
            right_momenta,
            attachment_range: attachment_start..attachment_end,
            selector_domain_id: selector_stage.map(|selector| selector.invocation_domains[index]),
        });
        attachment_cursor = attachment_end;
    }
    if attachment_cursor != attachments.len() {
        return Err(RusticolError::artifact(
            "eager invocation ranges do not cover the attachment table",
        ));
    }

    let mut finalizations = Vec::new();
    let mut finalization_copies = Vec::new();
    let mut stage_current_ranges = BTreeMap::new();
    let mut current_component_count = 0usize;
    finalizations
        .try_reserve_exact(finalization_rows.len())
        .map_err(|error| {
            RusticolError::artifact(format!("could not reserve eager finalizations: {error}"))
        })?;
    let mut finalized_currents = BTreeSet::new();
    for (index, row) in finalization_rows.into_iter().enumerate() {
        if !finalized_currents.insert(row.current_id) {
            return Err(RusticolError::artifact(format!(
                "eager current {} is finalized more than once in stage {}",
                row.current_id, payload.stage_index
            )));
        }
        if !globally_finalized_currents.insert(row.current_id) {
            return Err(RusticolError::artifact(format!(
                "eager current {} is finalized in more than one stage",
                row.current_id
            )));
        }
        if row.unpropagated_value_slot_id != MISSING_U32
            && row.unpropagated_value_slot_id == row.propagated_value_slot_id
        {
            return Err(RusticolError::artifact(format!(
                "eager finalization {index} aliases propagated and unpropagated outputs"
            )));
        }
        for value_slot_id in [row.unpropagated_value_slot_id, row.propagated_value_slot_id] {
            if value_slot_id != MISSING_U32 && !globally_stored_value_slots.insert(value_slot_id) {
                return Err(RusticolError::artifact(format!(
                    "eager value slot {value_slot_id} is written by more than one finalization"
                )));
            }
        }
        let current = currents.get(row.current_id, "eager finalization current")?;
        let local_current = ComponentRange {
            start: current_component_count,
            len: current.len,
        };
        current_component_count = current_component_count
            .checked_add(current.len)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "eager stage {} current workspace overflows usize",
                    payload.stage_index
                ))
            })?;
        stage_current_ranges.insert(row.current_id, local_current);
        let unpropagated = optional_component_range(
            values,
            row.unpropagated_value_slot_id,
            "eager unpropagated value",
        )?;
        let propagated = optional_component_range(
            values,
            row.propagated_value_slot_id,
            "eager propagated value",
        )?;
        if unpropagated.is_none() && propagated.is_none() {
            return Err(RusticolError::artifact(format!(
                "eager finalization {index} stores no current value"
            )));
        }
        for output in [unpropagated, propagated].into_iter().flatten() {
            if output.len != current.len {
                return Err(RusticolError::artifact(format!(
                    "eager finalization {index} output width does not match current width"
                )));
            }
        }
        let momentum = momenta.get(row.momentum_slot_id, "eager finalization momentum")?;
        if let Some(unpropagated) = unpropagated {
            let selector_domain_id =
                selector_stage.map(|selector| selector.unpropagated_finalization_domains[index]);
            finalization_copies.push(FinalizationCopy {
                current: local_current,
                unpropagated,
                selector_domain_id,
            });
        } else if let (Some(selector), Some(domains)) = (selector_stage, selector_plan)
            && !domains.memberships[selector.unpropagated_finalization_domains[index] as usize]
                .is_empty()
        {
            return Err(RusticolError::artifact(format!(
                "missing eager unpropagated output {index} has a nonempty selector domain"
            )));
        }
        if row.applies_kernel() {
            let kernel = require_kernel(
                kernels,
                row.kernel_id,
                EagerKernelRole::Finalization,
                "finalization",
            )?;
            if propagated.is_none() {
                return Err(RusticolError::artifact(format!(
                    "eager finalization {index} applies a kernel without a propagated output"
                )));
            }
            validate_kernel_inputs(
                kernel,
                KernelInputBounds {
                    first_current: current.len,
                    second_current: 0,
                    first_momentum: momentum.len,
                    second_momentum: 0,
                    has_coupling: false,
                    parameter_count,
                },
            )?;
            if usize::try_from(kernel.output_component_count).ok() != Some(current.len) {
                return Err(RusticolError::artifact(format!(
                    "eager finalization kernel {} output width does not match current width",
                    kernel.kernel_id
                )));
            }
            finalizations.push(ScheduledFinalization {
                row,
                current: local_current,
                propagated,
                momentum,
                selector_domain_id: selector_stage
                    .map(|selector| selector.propagated_finalization_domains[index]),
            });
        } else if propagated.is_some() {
            return Err(RusticolError::artifact(format!(
                "eager finalization {index} has a propagated output but no kernel"
            )));
        } else if let (Some(selector), Some(domains)) = (selector_stage, selector_plan)
            && !domains.memberships[selector.propagated_finalization_domains[index] as usize]
                .is_empty()
        {
            return Err(RusticolError::artifact(format!(
                "missing eager propagated output {index} has a nonempty selector domain"
            )));
        }
    }
    if !attached_currents.is_subset(&finalized_currents) {
        let missing = attached_currents
            .difference(&finalized_currents)
            .next()
            .copied()
            .unwrap_or(MISSING_U32);
        return Err(RusticolError::artifact(format!(
            "eager stage {} does not finalize attached current {missing}",
            payload.stage_index
        )));
    }
    if !finalized_currents.is_subset(&attached_currents) {
        let unwritten = finalized_currents
            .difference(&attached_currents)
            .next()
            .copied()
            .unwrap_or(MISSING_U32);
        return Err(RusticolError::artifact(format!(
            "eager stage {} finalizes unwritten current {unwritten}",
            payload.stage_index
        )));
    }

    for attachment in &mut scheduled_attachments {
        attachment.current = *stage_current_ranges
            .get(&attachment.row.result_current_id)
            .ok_or_else(|| {
                RusticolError::internal(format!(
                    "eager stage {} lost current {} while compacting its workspace",
                    payload.stage_index, attachment.row.result_current_id
                ))
            })?;
    }

    // Stable sorting makes equal-kernel rows contiguous for packetization while
    // preserving artifact order, which defines deterministic lane ordering.
    invocations.sort_by_key(|item| item.row.kernel_id);
    finalizations.sort_by_key(|item| item.row.kernel_id);
    let zero_current_ranges =
        mark_initial_current_writes(&invocations, &mut scheduled_attachments, &finalizations);
    Ok(EagerStagePlan {
        stage_index: payload.stage_index,
        current_component_count,
        invocations,
        attachments: scheduled_attachments,
        finalization_copies,
        finalizations,
        zero_current_ranges,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_closures(
    rows: &[EagerClosureRow],
    selector_domain_ids: Option<&[u32]>,
    direct_specs: &[EagerDirectClosureSpec],
    values: &ComponentLayout,
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    coupling_count: usize,
    amplitude_count: usize,
    parameter_count: usize,
) -> RusticolResult<(Vec<ScheduledClosure>, Vec<ScheduledDirectClosure>)> {
    let mut direct_by_index = BTreeMap::new();
    for spec in direct_specs {
        let index = usize::try_from(spec.closure_index).map_err(|_| {
            RusticolError::artifact("eager direct closure index does not fit usize")
        })?;
        if spec.coefficients.is_empty() {
            return Err(RusticolError::artifact(format!(
                "eager direct closure {index} has no coefficients"
            )));
        }
        if spec
            .coefficients
            .iter()
            .any(|coefficient| !complex_is_finite(*coefficient))
        {
            return Err(RusticolError::artifact(format!(
                "eager direct closure {index} has a non-finite coefficient"
            )));
        }
        if direct_by_index.insert(index, spec).is_some() {
            return Err(RusticolError::artifact(format!(
                "duplicate eager direct closure specification {index}"
            )));
        }
    }

    let mut closures = Vec::new();
    let mut direct_closures = Vec::new();
    for (index, row) in rows.iter().copied().enumerate() {
        let left_values = values.get(row.left_value_slot_id, "eager closure left value")?;
        let right_values = values.get(row.right_value_slot_id, "eager closure right value")?;
        required_index(
            row.amplitude_index,
            amplitude_count,
            "eager closure amplitude",
        )?;
        if row.kernel_id == MISSING_U32 {
            if row.coupling_slot_id != MISSING_U32 {
                return Err(RusticolError::artifact(format!(
                    "eager direct closure {index} unexpectedly references a coupling"
                )));
            }
            if row.output_factor_source != EAGER_OUTPUT_FACTOR_NONE {
                return Err(RusticolError::artifact(format!(
                    "eager direct closure {index} has a dynamic output factor"
                )));
            }
            let spec = direct_by_index.remove(&index).ok_or_else(|| {
                RusticolError::artifact(format!(
                    "eager direct closure {index} lacks contraction coefficients"
                ))
            })?;
            if left_values.len != right_values.len || left_values.len != spec.coefficients.len() {
                return Err(RusticolError::artifact(format!(
                    "eager direct closure {index} component widths do not match"
                )));
            }
            direct_closures.push(ScheduledDirectClosure {
                row,
                left_values,
                right_values,
                coefficients: spec.coefficients.clone(),
                selector_domain_id: selector_domain_ids.map(|domains| domains[index]),
                initializes_amplitude: false,
            });
            continue;
        }
        if direct_by_index.contains_key(&index) {
            return Err(RusticolError::artifact(format!(
                "eager kernel closure {index} has direct-contraction metadata"
            )));
        }
        let kernel = require_kernel(kernels, row.kernel_id, EagerKernelRole::Closure, "closure")?;
        required_index(
            row.coupling_slot_id,
            coupling_count,
            "eager closure coupling",
        )?;
        validate_kernel_inputs(
            kernel,
            KernelInputBounds {
                first_current: left_values.len,
                second_current: right_values.len,
                first_momentum: 0,
                second_momentum: 0,
                has_coupling: true,
                parameter_count,
            },
        )?;
        if kernel.output_component_count != 1 {
            return Err(RusticolError::artifact(format!(
                "eager closure kernel {} must produce one component",
                kernel.kernel_id
            )));
        }
        closures.push(ScheduledClosure {
            row,
            left_values,
            right_values,
            selector_domain_id: selector_domain_ids.map(|domains| domains[index]),
            initializes_amplitude: false,
        });
    }
    if let Some(index) = direct_by_index.keys().next() {
        return Err(RusticolError::artifact(format!(
            "eager direct closure specification {index} has no matching row"
        )));
    }
    // Keep the same stable equal-kernel ordering contract as stage invocations.
    closures.sort_by_key(|item| item.row.kernel_id);
    Ok((closures, direct_closures))
}

pub(super) fn mark_initial_current_writes(
    invocations: &[ScheduledInvocation],
    attachments: &mut [ScheduledAttachment],
    finalizations: &[ScheduledFinalization],
) -> Vec<ComponentRange> {
    let mut written = BTreeSet::new();
    for invocation in invocations {
        for attachment in &mut attachments[invocation.attachment_range.clone()] {
            attachment.initializes_current = written.insert(attachment.row.result_current_id);
        }
    }
    finalizations
        .iter()
        .filter_map(|item| (!written.contains(&item.row.current_id)).then_some(item.current))
        .collect()
}

pub(super) fn mark_initial_amplitude_writes(
    closures: &mut [ScheduledClosure],
    direct_closures: &mut [ScheduledDirectClosure],
    amplitude_count: usize,
) -> Vec<usize> {
    let mut written = BTreeSet::new();
    for closure in closures {
        closure.initializes_amplitude = written.insert(closure.row.amplitude_index);
    }
    for closure in direct_closures {
        closure.initializes_amplitude = written.insert(closure.row.amplitude_index);
    }
    (0..amplitude_count)
        .filter(|index| !written.contains(&(*index as u32)))
        .collect()
}

fn validate_reduction_plan(
    groups: &[EagerReductionGroup],
    entries: &[EagerReductionEntry],
    amplitude_count: usize,
) -> RusticolResult<()> {
    if groups.is_empty() || entries.is_empty() {
        return Err(RusticolError::artifact(
            "eager reduction requires nonempty groups and entries",
        ));
    }
    let mut covered = BTreeSet::new();
    let mut coherent_group_ids = BTreeSet::new();
    for (group_index, group) in groups.iter().enumerate() {
        if !coherent_group_ids.insert(group.coherent_group_id) {
            return Err(RusticolError::artifact(format!(
                "eager coherent reduction-group ID {} is duplicated",
                group.coherent_group_id
            )));
        }
        if group.amplitude_indices.is_empty() {
            return Err(RusticolError::artifact(format!(
                "eager reduction group {group_index} is empty"
            )));
        }
        for amplitude_index in &group.amplitude_indices {
            required_index(
                *amplitude_index,
                amplitude_count,
                "eager reduction amplitude",
            )?;
            if !covered.insert(*amplitude_index) {
                return Err(RusticolError::artifact(format!(
                    "eager amplitude {amplitude_index} belongs to more than one reduction group"
                )));
            }
        }
    }
    if covered.len() != amplitude_count {
        return Err(RusticolError::artifact(format!(
            "eager reduction groups cover {} of {amplitude_count} amplitudes",
            covered.len()
        )));
    }
    for (index, entry) in entries.iter().enumerate() {
        required_index(
            entry.left_group_index,
            groups.len(),
            "eager reduction left group",
        )?;
        required_index(
            entry.right_group_index,
            groups.len(),
            "eager reduction right group",
        )?;
        if !complex_is_finite(entry.coefficient) {
            return Err(RusticolError::artifact(format!(
                "eager reduction entry {index} has a non-finite coefficient"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct KernelInputBounds {
    first_current: usize,
    second_current: usize,
    first_momentum: usize,
    second_momentum: usize,
    has_coupling: bool,
    parameter_count: usize,
}

fn validate_kernel_inputs(
    kernel: &EagerKernelSpec,
    bounds: KernelInputBounds,
) -> RusticolResult<()> {
    for input in &kernel.inputs {
        let (allowed, index, count) = match *input {
            EagerKernelInput::FirstCurrentComponent(index) => {
                (true, Some(index), bounds.first_current)
            }
            EagerKernelInput::SecondCurrentComponent(index) => (
                kernel.role != EagerKernelRole::Finalization,
                Some(index),
                bounds.second_current,
            ),
            EagerKernelInput::FirstMomentumComponent(index) => (
                kernel.role != EagerKernelRole::Closure,
                Some(index),
                bounds.first_momentum,
            ),
            EagerKernelInput::SecondMomentumComponent(index) => (
                kernel.role == EagerKernelRole::Vertex,
                Some(index),
                bounds.second_momentum,
            ),
            EagerKernelInput::CouplingReal | EagerKernelInput::CouplingImag => {
                (bounds.has_coupling, None, 0)
            }
            EagerKernelInput::ModelParameter(index) => (true, Some(index), bounds.parameter_count),
        };
        if !allowed {
            return Err(RusticolError::artifact(format!(
                "eager {:?} kernel {} cannot use input descriptor {input:?}",
                kernel.role, kernel.kernel_id
            )));
        }
        if let Some(index) = index {
            let index = usize::try_from(index).map_err(|_| {
                RusticolError::artifact(format!(
                    "eager kernel {} input descriptor {input:?} does not fit usize",
                    kernel.kernel_id
                ))
            })?;
            if index >= count {
                return Err(RusticolError::artifact(format!(
                    "eager kernel {} input descriptor {input:?} is outside 0..{count}",
                    kernel.kernel_id
                )));
            }
        }
    }
    Ok(())
}

fn require_kernel<'a>(
    kernels: &'a BTreeMap<u32, EagerKernelSpec>,
    kernel_id: u32,
    role: EagerKernelRole,
    context: &str,
) -> RusticolResult<&'a EagerKernelSpec> {
    let kernel = kernels.get(&kernel_id).ok_or_else(|| {
        RusticolError::artifact(format!(
            "eager {context} references unknown kernel {kernel_id}"
        ))
    })?;
    if kernel.role != role {
        return Err(RusticolError::artifact(format!(
            "eager {context} kernel {kernel_id} has role {:?}, expected {role:?}",
            kernel.role
        )));
    }
    Ok(kernel)
}

fn optional_component_range(
    layout: &ComponentLayout,
    id: u32,
    context: &str,
) -> RusticolResult<Option<ComponentRange>> {
    if id == MISSING_U32 {
        return Ok(None);
    }
    layout.get(id, context).map(Some)
}

fn validate_optional_index(id: u32, count: usize, context: &str) -> RusticolResult<()> {
    if id == MISSING_U32 {
        return Ok(());
    }
    required_index(id, count, context).map(|_| ())
}

fn required_index(id: u32, count: usize, context: &str) -> RusticolResult<usize> {
    if id == MISSING_U32 {
        return Err(RusticolError::artifact(format!(
            "{context} uses the reserved missing id"
        )));
    }
    let index = usize::try_from(id)
        .map_err(|_| RusticolError::artifact(format!("{context} id does not fit usize")))?;
    if index >= count {
        return Err(RusticolError::artifact(format!(
            "{context} index {index} is outside 0..{count}"
        )));
    }
    Ok(index)
}

pub(super) fn complex_is_finite(value: EagerComplex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
