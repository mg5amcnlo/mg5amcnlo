// SPDX-License-Identifier: 0BSD

//! Portable rows only: no kernel pointers, descriptor caches, or numeric workspaces.

use super::*;

#[derive(bincode::Encode, bincode::Decode)]
pub(crate) struct OnTheFlyFamilySnapshotV1 {
    strategy: u32,
    census: OnTheFlyQueryFamilyCensusV1,
    source_count: u32,
    lorentz_component_count: u16,
    parameter_count: u32,
    current_component_count: u32,
    amplitude_destination_count: u32,
    momentum_forms: Box<[CanonicalMomentumLinearForm]>,
    exact_factors: Box<[ExactComplexRational]>,
    row_groups: Box<[OnTheFlyFamilyRowGroupV1]>,
}

/// Same wire layout as the owned decoder, borrowing the potentially large rows.
#[derive(bincode::Encode)]
pub(crate) struct OnTheFlyFamilySnapshotRefV1<'a> {
    strategy: u32,
    census: OnTheFlyQueryFamilyCensusV1,
    source_count: u32,
    lorentz_component_count: u16,
    parameter_count: u32,
    current_component_count: u32,
    amplitude_destination_count: u32,
    momentum_forms: &'a [CanonicalMomentumLinearForm],
    exact_factors: &'a [ExactComplexRational],
    row_groups: &'a [OnTheFlyFamilyRowGroupV1],
}

impl OnTheFlyFamilySnapshotV1 {
    pub(crate) fn executor_keys(&self) -> impl Iterator<Item = OnTheFlyExecutorKeyV1> + '_ {
        self.row_groups.iter().flat_map(|group| {
            std::iter::once(group.representative_key)
                .chain(group.member_bindings.iter().map(|(key, _)| *key))
        })
    }

    pub(crate) const fn census(&self) -> OnTheFlyQueryFamilyCensusV1 {
        self.census
    }

    pub(crate) const fn amplitude_destination_count(&self) -> usize {
        self.amplitude_destination_count as usize
    }

    fn into_family(self) -> RusticolResult<(RecurrenceStrategy, OnTheFlyBuiltQueryFamilyV1)> {
        let strategy = match self.strategy {
            0 => RecurrenceStrategy::TopologyReplay,
            2 => RecurrenceStrategy::ContractedColorUnion,
            _ => return Err(integrity("saved OTF family has an unsupported strategy")),
        };
        let family = OnTheFlyBuiltQueryFamilyV1 {
            census: self.census,
            source_count: self.source_count,
            lorentz_component_count: self.lorentz_component_count,
            parameter_count: self.parameter_count,
            current_component_count: self.current_component_count,
            amplitude_destination_count: self.amplitude_destination_count,
            momentum_forms: self.momentum_forms,
            exact_factors: self.exact_factors,
            row_groups: self.row_groups,
            #[cfg(any(test, feature = "on-the-fly-test-support"))]
            observed_currents: Box::new([]),
        };
        validate_ordered_family_schedule(&family.row_groups)?;
        if family.census.source_frame_partition_count != 1
            || family.census.union_amplitude_destination_count != family.amplitude_destination_count
            || family.census.union_unique_current_component_count != family.current_component_count
        {
            return Err(integrity(
                "saved OTF family dimensions disagree with its census",
            ));
        }
        Ok((strategy, family))
    }
}

impl<R: OnTheFlyPreparedExecutorResolver> OnTheFlyQueryFamilyExecutorV1<R> {
    pub(crate) fn snapshot(&self) -> RusticolResult<Option<OnTheFlyFamilySnapshotRefV1<'_>>> {
        if self.families.len() > 1 {
            return Err(integrity(
                "OTF family retention policy changed; update the cache format",
            ));
        }
        Ok(self.families.first().map(|bound| {
            let family = &bound.family;
            OnTheFlyFamilySnapshotRefV1 {
                strategy: bound.strategy.as_u32(),
                census: family.census,
                source_count: family.source_count,
                lorentz_component_count: family.lorentz_component_count,
                parameter_count: family.parameter_count,
                current_component_count: family.current_component_count,
                amplitude_destination_count: family.amplitude_destination_count,
                momentum_forms: &family.momentum_forms,
                exact_factors: &family.exact_factors,
                row_groups: &family.row_groups,
            }
        }))
    }

    /// All fallible allocation and binding happens before replacing the old
    /// committed owner. No execution or numeric input is required to restore it.
    pub(crate) fn restore_snapshot(
        &mut self,
        snapshot: OnTheFlyFamilySnapshotV1,
        logical_point_capacity: u32,
    ) -> RusticolResult<OnTheFlyQueryFamilyHandleV1> {
        let (strategy, family) = snapshot.into_family()?;
        if family.parameter_count as usize != self.parameter_state.len() {
            return Err(integrity(
                "saved OTF parameter layout differs from the loaded model",
            ));
        }
        let resolved_groups = self.bind_groups(&family)?;
        let packed_singleton_capable =
            Self::resolved_groups_are_packed_singleton_capable(&family, &resolved_groups);
        let workspace = OnTheFlyFamilyWorkspaceV1::new(
            &family,
            logical_point_capacity,
            packed_singleton_capable,
        )?;
        let interaction_program =
            Self::bind_interaction_program(&family, &resolved_groups, strategy)?;
        let singleton_fanout_program =
            Self::bind_singleton_fanout_program(&family, &resolved_groups)?;
        if self.families.capacity() == 0 {
            self.families.try_reserve_exact(1).map_err(|error| {
                invalid(format!("OTF restored-family allocation failed: {error}"))
            })?;
        }
        let candidate = BoundOnTheFlyQueryFamilyV1 {
            identity: QueryFamilyCacheIdentityV1::Restored,
            strategy,
            family,
            workspace,
            resolved_groups,
            packed_singleton_capable,
            interaction_program,
            singleton_fanout_program,
            applied_parameter_version: u64::MAX,
            descriptor_exposed: false,
        };
        self.invalidate_exposed_row_tables()?;
        self.pending = Some(candidate);
        if let Err(error) = self.commit_pending_family() {
            // The fresh candidate has never exposed a row pointer. In
            // particular, generation-counter exhaustion must not leave it
            // selected after a failed restore of a previously warm runner.
            self.pending = None;
            return Err(error);
        }
        self.active_retained_handle()
            .ok_or_else(|| integrity("restored OTF family has no handle"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_decode_keeps_canonical_type_invariants() {
        let config = bincode::config::standard();
        for denominator in [0_i128, -1] {
            let bytes = bincode::encode_to_vec((1_i128, denominator), config).unwrap();
            assert!(
                bincode::decode_from_slice::<crate::recurrence::ExactRational, _>(&bytes, config)
                    .is_err()
            );
        }
        let bytes = bincode::encode_to_vec([0_u8; 32], config).unwrap();
        assert!(bincode::decode_from_slice::<SemanticDigest, _>(&bytes, config).is_err());
        let bytes = bincode::encode_to_vec(
            vec![crate::recurrence::MomentumTerm {
                source_slot: 0,
                coefficient: 0,
            }],
            config,
        )
        .unwrap();
        assert!(
            bincode::decode_from_slice::<CanonicalMomentumLinearForm, _>(&bytes, config).is_err()
        );
    }
}
