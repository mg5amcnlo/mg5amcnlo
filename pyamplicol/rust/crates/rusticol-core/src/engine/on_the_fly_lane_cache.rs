// SPDX-License-Identifier: 0BSD

use super::*;
use crate::recurrence::on_the_fly::{
    OnTheFlyFamilySnapshotRefV1, OnTheFlyFamilySnapshotV1, restore_on_the_fly_grammar_v1,
};

#[derive(bincode::Encode, bincode::Decode)]
struct LcProjectionV1 {
    amplitude_destinations: Box<[Option<usize>]>,
    census: Option<OnTheFlyQueryFamilyCensusV1>,
    logical_point_capacity: u32,
}

#[derive(bincode::Encode)]
struct LcProjectionRefV1<'a> {
    amplitude_destinations: &'a [Option<usize>],
    census: Option<OnTheFlyQueryFamilyCensusV1>,
    logical_point_capacity: u32,
}

#[derive(bincode::Encode, bincode::Decode)]
struct ContractedProjectionV1 {
    helicity_ordinals: Box<[usize]>,
    structural_color_count: usize,
    projection: LcProjectionV1,
}

#[derive(bincode::Encode)]
struct ContractedProjectionRefV1<'a> {
    helicity_ordinals: &'a [usize],
    structural_color_count: usize,
    projection: LcProjectionRefV1<'a>,
}

#[derive(bincode::Decode)]
pub(in super::super) struct OnTheFlyLaneSnapshotV1 {
    policy: Option<OnTheFlyResolvedCouplingPolicyV1>,
    process_preparation_count: u64,
    lc: Option<LcProjectionV1>,
    contracted: Option<ContractedProjectionV1>,
    executor: Option<OnTheFlyFamilySnapshotV1>,
}

#[derive(bincode::Encode)]
pub(in super::super) struct OnTheFlyLaneSnapshotRefV1<'a> {
    policy: Option<&'a OnTheFlyResolvedCouplingPolicyV1>,
    process_preparation_count: u64,
    lc: Option<LcProjectionRefV1<'a>>,
    contracted: Option<ContractedProjectionRefV1<'a>>,
    executor: Option<OnTheFlyFamilySnapshotRefV1<'a>>,
}

impl OnTheFlyLaneSnapshotV1 {
    pub(in super::super) fn has_family(&self) -> bool {
        self.lc.is_some() || self.contracted.is_some()
    }
}

impl OnTheFlyNativeRuntime {
    pub(in super::super) fn snapshot(&self) -> RusticolResult<OnTheFlyLaneSnapshotRefV1<'_>> {
        if self.families.len() > 1 || (self.last_family.is_some() && self.last_family != Some(0)) {
            return Err(RusticolError::internal(
                "OTF retention policy changed; update the cache format",
            ));
        }
        Ok(OnTheFlyLaneSnapshotRefV1 {
            policy: self.coupling_policy.as_ref(),
            process_preparation_count: self.process_preparation_count,
            lc: self.families.first().map(|family| LcProjectionRefV1 {
                amplitude_destinations: &family.amplitude_destinations,
                census: family.census,
                logical_point_capacity: family.logical_point_capacity,
            }),
            contracted: self
                .contracted_family
                .as_ref()
                .map(|family| ContractedProjectionRefV1 {
                    helicity_ordinals: &family.helicity_ordinals,
                    structural_color_count: family.structural_color_count,
                    projection: LcProjectionRefV1 {
                        amplitude_destinations: &family.amplitude_destinations,
                        census: family.census,
                        logical_point_capacity: family.logical_point_capacity,
                    },
                }),
            executor: self.executor.snapshot()?,
        })
    }

    pub(in super::super) fn restore_snapshot(
        &mut self,
        snapshot: OnTheFlyLaneSnapshotV1,
        requests: Option<Vec<OnTheFlyLcQueryRequestV1>>,
        contracted_selection: Option<(&[usize], usize)>,
    ) -> RusticolResult<()> {
        let OnTheFlyLaneSnapshotV1 {
            policy,
            process_preparation_count,
            lc,
            contracted,
            executor,
        } = snapshot;
        if lc.is_some() && contracted.is_some()
            || lc.is_some() != requests.is_some()
            || contracted.is_some() != contracted_selection.is_some()
            || (lc.is_some() || contracted.is_some()) && policy.is_none()
        {
            return Err(RusticolError::integrity(
                "saved OTF family and selected workload disagree",
            ));
        }
        let grammar = policy
            .as_ref()
            .map(|policy| restore_on_the_fly_grammar_v1(&self.templates, &self.seed, policy))
            .transpose()?;
        let projection = lc
            .as_ref()
            .or_else(|| contracted.as_ref().map(|value| &value.projection));
        if let Some(projection) = projection {
            let count = executor
                .as_ref()
                .map_or(0, OnTheFlyFamilySnapshotV1::amplitude_destination_count);
            if projection.logical_point_capacity == 0
                || projection.census != executor.as_ref().map(OnTheFlyFamilySnapshotV1::census)
                || projection
                    .amplitude_destinations
                    .iter()
                    .flatten()
                    .any(|&id| id >= count)
                || executor.is_none()
                    && projection
                        .amplitude_destinations
                        .iter()
                        .any(Option::is_some)
            {
                return Err(RusticolError::integrity(
                    "saved OTF amplitude projection is inconsistent",
                ));
            }
        } else if executor.is_some() {
            return Err(RusticolError::integrity(
                "saved OTF executor has no selected workload",
            ));
        }
        if let (Some(lc), Some(requests)) = (&lc, &requests)
            && lc.amplitude_destinations.len() != requests.len()
        {
            return Err(RusticolError::integrity(
                "saved OTF LC projection has the wrong query count",
            ));
        }
        if let (Some(contracted), Some((helicities, colors))) = (&contracted, contracted_selection)
            && (contracted.helicity_ordinals.as_ref() != helicities
                || contracted.structural_color_count != colors
                || helicities.len().checked_mul(colors)
                    != Some(contracted.projection.amplitude_destinations.len()))
        {
            return Err(RusticolError::integrity(
                "saved OTF contracted projection has the wrong selected axes",
            ));
        }
        // Allocate all replacement metadata before committing a new executor.
        let mut restored_lc = Vec::new();
        if lc.is_some() {
            restored_lc.try_reserve_exact(1).map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "OTF restored selection allocation failed: {error}"
                ))
            })?;
        }
        let requests = requests.map(Vec::into_boxed_slice);
        let capacity = projection.map_or(1, |value| value.logical_point_capacity);
        let handle = if let Some(executor) = executor {
            let restored = (|| {
                self.executor.resolver_mut().bind_on_the_fly_saved_family(
                    &self.templates,
                    &self.direct_catalog,
                    executor.executor_keys(),
                )?;
                self.executor.restore_snapshot(executor, capacity)
            })();
            match restored {
                Ok(handle) => {
                    self.executor
                        .resolver_mut()
                        .commit_pending_resolved_bindings()?;
                    Some(handle)
                }
                Err(error) => {
                    self.executor
                        .resolver_mut()
                        .discard_pending_resolved_bindings();
                    return Err(error);
                }
            }
        } else {
            self.executor.clear_families()?;
            self.executor.resolver_mut().clear_resolved_bindings();
            None
        };
        if let Some(lc) = lc {
            restored_lc.push(PreparedOnTheFlyLcFamilyV1 {
                requests: requests.expect("validated saved LC requests disappeared"),
                amplitude_destinations: lc.amplitude_destinations,
                executor_handle: handle,
                census: lc.census,
                logical_point_capacity: lc.logical_point_capacity,
            });
        }
        self.last_family = (!restored_lc.is_empty()).then_some(0);
        self.families = restored_lc;
        self.pending_family = None;
        self.contracted_family = contracted.map(|value| PreparedOnTheFlyContractedFamilyV1 {
            helicity_ordinals: value.helicity_ordinals,
            structural_color_count: value.structural_color_count,
            amplitude_destinations: value.projection.amplitude_destinations,
            executor_handle: handle,
            census: value.projection.census,
            logical_point_capacity: value.projection.logical_point_capacity,
        });
        self.pending_contracted_family = None;
        self.prepared_grammar = grammar;
        self.coupling_policy = policy;
        self.coupling_policy_resolution = self.coupling_policy.as_ref().map(|_| Duration::ZERO);
        self.process_preparation_count = process_preparation_count;
        self.source_momenta_scratch = Vec::new();
        self.amplitude_scratch = Vec::new();
        Ok(())
    }
}
