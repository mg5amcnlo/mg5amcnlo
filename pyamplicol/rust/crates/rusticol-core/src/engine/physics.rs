// SPDX-License-Identifier: 0BSD

use super::*;

static NEXT_PHYSICS_RUNTIME_BINDING_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

impl PhysicsRuntime {
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub(super) fn new(manifest: ProcessPhysicsV1) -> RusticolResult<Self> {
        manifest.validate()?;
        let helicity_index_by_id = manifest
            .helicities
            .iter()
            .enumerate()
            .map(|(index, helicity)| (helicity.id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let helicity_representative_index = manifest
            .helicities
            .iter()
            .map(|helicity| {
                helicity_index_by_id
                    .get(&helicity.representative_id)
                    .copied()
                    .ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "helicity {} references absent representative {}",
                            helicity.id, helicity.representative_id
                        ))
                    })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut helicity_members_by_representative = vec![Vec::new(); manifest.helicities.len()];
        for (physical_index, representative_index) in
            helicity_representative_index.iter().copied().enumerate()
        {
            if !manifest.helicities[physical_index].structural_zero {
                helicity_members_by_representative[representative_index].push(physical_index);
            }
        }
        let color_index_by_id = manifest
            .color_components
            .iter()
            .enumerate()
            .map(|(index, color)| (color.id().to_string(), index))
            .collect::<BTreeMap<_, _>>();
        let mut reduction_by_group_id = BTreeMap::new();
        let mut numeric_reduction_by_group_id = BTreeMap::new();
        for group in &manifest.reduction.groups {
            let group_id = parse_reduction_group_id(&group.id)?;
            let mut expected_helicity_weight = None;
            for id in &group.physical_helicity_ids {
                let index = helicity_index_by_id[id];
                let coefficient = manifest.helicities[index].coefficient;
                if let Some(expected) = expected_helicity_weight {
                    if coefficient.to_bits() != expected {
                        return Err(RusticolError::artifact(format!(
                            "resolved reduction group {} has inconsistent helicity weights",
                            group.id
                        )));
                    }
                } else {
                    expected_helicity_weight = Some(coefficient.to_bits());
                }
            }
            if reduction_by_group_id
                .insert(group_id, group.clone())
                .is_some()
            {
                return Err(RusticolError::artifact(format!(
                    "resolved reduction groups map to duplicate evaluator id {group_id}"
                )));
            }
            if numeric_reduction_by_group_id
                .insert(
                    group_id,
                    numeric_reduction_group(
                        group,
                        &manifest,
                        &helicity_index_by_id,
                        &color_index_by_id,
                    )?,
                )
                .is_some()
            {
                return Err(RusticolError::artifact(format!(
                    "resolved reduction groups map to duplicate numeric evaluator id {group_id}"
                )));
            }
        }
        Ok(Self {
            binding_id: NEXT_PHYSICS_RUNTIME_BINDING_ID
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            manifest,
            helicity_index_by_id,
            helicity_members_by_representative,
            color_index_by_id,
            reduction_by_group_id,
            numeric_reduction_by_group_id,
        })
    }

    pub(super) fn selected_helicity_indices(
        &self,
        ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Vec<usize>> {
        let mut indices = Vec::new();
        self.select_indices_into(ids, &self.helicity_index_by_id, "helicity", &mut indices)?;
        Ok(indices)
    }

    pub(super) fn selected_color_indices(
        &self,
        ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Vec<usize>> {
        let mut indices = Vec::new();
        self.select_indices_into(
            ids,
            &self.color_index_by_id,
            "color component",
            &mut indices,
        )?;
        Ok(indices)
    }

    pub(super) fn canonical_total_helicity_selector<'a>(
        &self,
        ids: Option<&'a BTreeSet<String>>,
    ) -> Option<&'a BTreeSet<String>> {
        canonical_total_axis_selector(ids, &self.helicity_index_by_id)
    }

    pub(super) fn canonical_total_color_selector<'a>(
        &self,
        ids: Option<&'a BTreeSet<String>>,
    ) -> Option<&'a BTreeSet<String>> {
        canonical_total_axis_selector(ids, &self.color_index_by_id)
    }

    pub(super) fn validate_helicity_id_slice(&self, ids: Option<&[String]>) -> RusticolResult<()> {
        self.validate_id_slice(ids, &self.helicity_index_by_id, "helicity")
    }

    pub(super) fn validate_color_id_slice(&self, ids: Option<&[String]>) -> RusticolResult<()> {
        self.validate_id_slice(ids, &self.color_index_by_id, "color component")
    }

    fn validate_id_slice(
        &self,
        ids: Option<&[String]>,
        available: &BTreeMap<String, usize>,
        kind: &str,
    ) -> RusticolResult<()> {
        if let Some(ids) = ids {
            for id in ids {
                if !available.contains_key(id) {
                    return Err(RusticolError::selector(format!(
                        "unknown resolved {kind} id {id:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn select_indices_into(
        &self,
        ids: Option<&BTreeSet<String>>,
        available: &BTreeMap<String, usize>,
        kind: &str,
        indices: &mut Vec<usize>,
    ) -> RusticolResult<()> {
        indices.clear();
        if let Some(ids) = ids {
            for id in ids {
                let index = available.get(id).ok_or_else(|| {
                    RusticolError::selector(format!("unknown resolved {kind} id {id:?}"))
                })?;
                indices.push(*index);
            }
            indices.sort_unstable();
        } else {
            indices.extend(0..available.len());
        }
        Ok(())
    }

    pub(super) fn selected_helicity_indices_into(
        &self,
        ids: Option<&BTreeSet<String>>,
        indices: &mut Vec<usize>,
    ) -> RusticolResult<()> {
        self.select_indices_into(ids, &self.helicity_index_by_id, "helicity", indices)
    }

    pub(super) fn selected_color_indices_into(
        &self,
        ids: Option<&BTreeSet<String>>,
        indices: &mut Vec<usize>,
    ) -> RusticolResult<()> {
        self.select_indices_into(ids, &self.color_index_by_id, "color component", indices)
    }

    pub(super) fn has_contracted_color_axis(&self) -> bool {
        matches!(
            self.manifest.color_components.as_slice(),
            [PhysicsColorComponentV1::ContractedColor(_)]
        )
    }

    pub(super) fn color_is_computed(&self, index: usize) -> bool {
        match &self.manifest.color_components[index] {
            PhysicsColorComponentV1::LcFlow(flow) => flow.computed,
            PhysicsColorComponentV1::ContractedColor(_) => true,
        }
    }

    #[inline(always)]
    pub(super) fn helicity_orbit_members(&self, representative: usize) -> &[usize] {
        self.helicity_members_by_representative
            .get(representative)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(super) fn normalized_helicity_weights(
        &self,
        group: &crate::ReductionGroup,
    ) -> RusticolResult<Vec<(usize, f64)>> {
        let mut weights = Vec::new();
        self.normalized_helicity_weights_into(group, &mut weights)?;
        Ok(weights)
    }

    pub(super) fn normalized_helicity_weights_into(
        &self,
        group: &crate::ReductionGroup,
        weights: &mut Vec<(usize, f64)>,
    ) -> RusticolResult<()> {
        weights.clear();
        weights
            .try_reserve(group.physical_helicity_ids.len())
            .map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "could not reserve normalized helicity weights: {error}"
                ))
            })?;
        let mut total = 0.0;
        for id in &group.physical_helicity_ids {
            let index = self.helicity_index_by_id[id];
            let coefficient = self.manifest.helicities[index].coefficient;
            total += coefficient;
            weights.push((index, coefficient));
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(RusticolError::artifact(format!(
                "resolved reduction group {} has no positive helicity weight",
                group.id
            )));
        }
        for (_, weight) in weights.iter_mut() {
            *weight /= total;
        }
        Ok(())
    }

    pub(super) fn normalized_member_weights(
        &self,
        group: &crate::ReductionGroup,
    ) -> RusticolResult<Vec<(usize, usize, f64)>> {
        let mut weights = Vec::new();
        self.normalized_member_weights_into(group, &mut weights)?;
        Ok(weights)
    }

    pub(super) fn normalized_member_weights_into(
        &self,
        group: &crate::ReductionGroup,
        weights: &mut Vec<(usize, usize, f64)>,
    ) -> RusticolResult<()> {
        weights.clear();
        let mut total = 0.0;
        for helicity_id in &group.physical_helicity_ids {
            let helicity_index = self.helicity_index_by_id[helicity_id];
            let helicity_weight = self.manifest.helicities[helicity_index].coefficient;
            for color_id in &group.physical_color_ids {
                let color_index = self.color_index_by_id[color_id];
                let weight =
                    helicity_weight * self.manifest.color_components[color_index].coefficient();
                total += weight;
                weights.push((helicity_index, color_index, weight));
            }
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(RusticolError::artifact(format!(
                "resolved reduction group {} has no positive member weight",
                group.id
            )));
        }
        for (_, _, weight) in weights.iter_mut() {
            *weight /= total;
        }
        Ok(())
    }

    pub(super) fn lc_resolved_replay_plan(
        &self,
        mappings: &LcTopologyReplayMappings,
        sector_routes: &[Vec<LcTopologyReplaySectorRoute>],
        materialized_sector_by_id: &BTreeMap<i64, LcMaterializedSector>,
    ) -> RusticolResult<LcResolvedReplayPlan> {
        if self.manifest.color_accuracy.as_str() != "lc" {
            return Err(RusticolError::artifact(
                "LC topology replay requires LC resolved physics metadata",
            ));
        }
        if mappings.len() != sector_routes.len() {
            return Err(RusticolError::artifact(
                "LC topology replay mappings and sector routes have inconsistent lengths",
            ));
        }

        let helicity_count = self.manifest.helicities.len();
        let color_count = self.manifest.color_components.len();
        let mut relevant_helicities = BTreeSet::new();
        let mut relevant_colors = BTreeSet::new();
        for reduction in self.reduction_by_group_id.values() {
            for id in &reduction.physical_helicity_ids {
                relevant_helicities.insert(self.helicity_index_by_id[id]);
            }
            for id in &reduction.physical_color_ids {
                relevant_colors.insert(self.color_index_by_id[id]);
            }
        }
        if relevant_helicities.is_empty() || relevant_colors.is_empty() {
            return Err(RusticolError::artifact(
                "LC topology replay has no materialized resolved members",
            ));
        }

        let helicity_index_by_values = self
            .manifest
            .helicities
            .iter()
            .enumerate()
            .map(|(index, helicity)| (helicity.values.clone(), index))
            .collect::<BTreeMap<_, _>>();
        if helicity_index_by_values.len() != helicity_count {
            return Err(RusticolError::artifact(
                "resolved physics contains duplicate helicity vectors",
            ));
        }
        let mut color_index_by_word = BTreeMap::new();
        for (index, color) in self.manifest.color_components.iter().enumerate() {
            let PhysicsColorComponentV1::LcFlow(flow) = color else {
                return Err(RusticolError::artifact(
                    "LC topology replay encountered a contracted color component",
                ));
            };
            if color_index_by_word
                .insert(flow.word.clone(), index)
                .is_some()
            {
                return Err(RusticolError::artifact(
                    "resolved physics contains duplicate LC flow words",
                ));
            }
        }

        let mut entries = Vec::with_capacity(mappings.len());
        let mut target_sector_by_color = BTreeMap::new();
        for (mapping, mapping_routes) in mappings.iter().zip(sector_routes) {
            validate_public_label_permutation(mapping, self.manifest.external_particles.len())?;
            let mut helicity_targets = BTreeMap::new();
            for source_index in &relevant_helicities {
                let source = &self.manifest.helicities[*source_index];
                let target_values = permute_public_helicity(&source.values, mapping);
                let target_index = *helicity_index_by_values.get(&target_values).ok_or_else(|| {
                    RusticolError::compatibility(format!(
                        "resolved physics is missing replayed helicity vector {target_values:?}; regenerate the artifact with complete topology-replay reductions"
                    ))
                })?;
                let target = &self.manifest.helicities[target_index];
                if source.coefficient.to_bits() != target.coefficient.to_bits() {
                    return Err(RusticolError::artifact(format!(
                        "LC topology replay maps helicity {} to {} with a different reduction coefficient",
                        source.id, target.id
                    )));
                }
                helicity_targets.insert(*source_index, target_index);
            }

            let mut routes = Vec::new();
            for sector_route in mapping_routes {
                let source_sector = *materialized_sector_by_id
                    .get(&sector_route.materialized_sector_id)
                    .ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "LC topology replay materialized sector {} has no computed physics component",
                            sector_route.materialized_sector_id
                        ))
                    })?;
                let source_index = source_sector.color_index;
                let PhysicsColorComponentV1::LcFlow(source) =
                    &self.manifest.color_components[source_index]
                else {
                    return Err(RusticolError::artifact(
                        "LC topology replay encountered a contracted materialized color component",
                    ));
                };
                if !source.computed || source.representative_id != source.id {
                    return Err(RusticolError::artifact(format!(
                        "LC topology replay sector {} is not bound to a self-representing computed color component",
                        sector_route.materialized_sector_id
                    )));
                }
                let target_word = permute_public_color_word(&source.word, mapping);
                let reduction_weight = if sector_route.residual {
                    source_sector.reduction_weight
                } else {
                    sector_route.squared_reduction_weight()
                };
                let target_words = replay_public_color_words(&target_word, reduction_weight)?;
                let mut target_coefficients = BTreeMap::new();
                for word in target_words {
                    let target_index = *color_index_by_word.get(&word).ok_or_else(|| {
                        RusticolError::compatibility(format!(
                            "resolved physics is missing replayed LC flow word {word:?}; regenerate the artifact with replay-to-public-flow reductions"
                        ))
                    })?;
                    let coefficient = self.manifest.color_components[target_index].coefficient();
                    if !coefficient.is_finite() || coefficient <= 0.0 {
                        return Err(RusticolError::artifact(format!(
                            "replayed LC flow {} has no positive reduction coefficient",
                            self.manifest.color_components[target_index].id()
                        )));
                    }
                    let PhysicsColorComponentV1::LcFlow(target) =
                        &self.manifest.color_components[target_index]
                    else {
                        unreachable!("LC color axes checked above");
                    };
                    if target.representative_id != source.id {
                        return Err(RusticolError::artifact(format!(
                            "LC topology replay physical sector {} targets color {} represented by {}, expected {}",
                            sector_route.physical_sector_id,
                            target.id,
                            target.representative_id,
                            source.id
                        )));
                    }
                    if let Some(previous_sector) =
                        target_sector_by_color.insert(target_index, sector_route.physical_sector_id)
                        && previous_sector != sector_route.physical_sector_id
                    {
                        return Err(RusticolError::artifact(format!(
                            "LC topology replay physical sectors {previous_sector} and {} overlap color {}",
                            sector_route.physical_sector_id, target.id
                        )));
                    }
                    target_coefficients
                        .entry(target_index)
                        .or_insert(coefficient);
                }
                let total = target_coefficients.values().sum::<f64>();
                if !total.is_finite() || total <= 0.0 {
                    return Err(RusticolError::artifact(
                        "LC topology replay has no positive public-flow reduction weight",
                    ));
                }
                for (source_helicity, target_helicity) in &helicity_targets {
                    for (target_color, coefficient) in &target_coefficients {
                        routes.push(LcResolvedReplayRoute {
                            source_index: source_helicity * color_count + source_index,
                            target_index: target_helicity * color_count + target_color,
                            weight: reduction_weight * coefficient / total,
                        });
                    }
                }
            }
            if routes.is_empty() {
                return Err(RusticolError::artifact(
                    "LC topology replay mapping contains no resolved component routes",
                ));
            }
            entries.push(LcResolvedReplayEntry { routes });
        }
        if target_sector_by_color.len() != color_count {
            return Err(RusticolError::artifact(format!(
                "LC topology replay covers {} of {color_count} physical color components",
                target_sector_by_color.len()
            )));
        }
        let mut routes_by_target = vec![Vec::new(); helicity_count * color_count];
        for (mapping_index, entry) in entries.iter().enumerate() {
            for route in &entry.routes {
                routes_by_target[route.target_index].push(LcResolvedReplayTargetRoute {
                    mapping_index,
                    source_index: route.source_index,
                    weight: route.weight,
                });
            }
        }
        Ok(LcResolvedReplayPlan {
            #[cfg(test)]
            entries,
            routes_by_target,
            color_count,
        })
    }

    #[cfg(test)]
    pub(super) fn select_lc_resolved_replay_plan(
        &self,
        plan: &LcResolvedReplayPlan,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<LcResolvedReplaySelection> {
        let key =
            self.lc_resolved_replay_selection_key(selected_helicity_ids, selected_color_ids)?;
        self.select_lc_resolved_replay_plan_for_key(plan, &key)
    }

    pub(super) fn lc_resolved_replay_selection_key(
        &self,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<LcResolvedReplaySelectionKey> {
        Ok(LcResolvedReplaySelectionKey {
            helicity_indices: selected_helicity_ids
                .map(|ids| self.selected_helicity_indices(Some(ids)))
                .transpose()?,
            color_indices: selected_color_ids
                .map(|ids| self.selected_color_indices(Some(ids)))
                .transpose()?,
        })
    }

    pub(super) fn select_lc_resolved_replay_plan_for_key(
        &self,
        plan: &LcResolvedReplayPlan,
        key: &LcResolvedReplaySelectionKey,
    ) -> RusticolResult<LcResolvedReplaySelection> {
        let helicity_indices = key
            .helicity_indices
            .clone()
            .unwrap_or_else(|| (0..self.manifest.helicities.len()).collect());
        let color_indices = key
            .color_indices
            .clone()
            .unwrap_or_else(|| (0..self.manifest.color_components.len()).collect());
        let mut compact_target_by_full = BTreeMap::new();
        for (helicity_position, helicity_index) in helicity_indices.iter().enumerate() {
            for (color_position, color_index) in color_indices.iter().enumerate() {
                compact_target_by_full.insert(
                    helicity_index * plan.color_count + color_index,
                    helicity_position * color_indices.len() + color_position,
                );
            }
        }

        let mut routes_by_mapping = BTreeMap::<usize, Vec<LcResolvedReplayRoute>>::new();
        for (target_full_index, target_compact_index) in &compact_target_by_full {
            let routes = plan
                .routes_by_target
                .get(*target_full_index)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "LC topology replay target index is outside its route index",
                    )
                })?;
            for route in routes {
                routes_by_mapping
                    .entry(route.mapping_index)
                    .or_default()
                    .push(LcResolvedReplayRoute {
                        source_index: route.source_index,
                        target_index: *target_compact_index,
                        weight: route.weight,
                    });
            }
        }

        let mut mapping_indices = Vec::with_capacity(routes_by_mapping.len());
        let mut entries = Vec::with_capacity(routes_by_mapping.len());
        let mut source_helicity_indices = Vec::new();
        let mut source_color_indices = Vec::new();
        for (mapping_index, selected_routes) in routes_by_mapping {
            let source_helicities = selected_routes
                .iter()
                .map(|route| route.source_index / plan.color_count)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let source_colors = selected_routes
                .iter()
                .map(|route| route.source_index % plan.color_count)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let compact_helicity = source_helicities
                .iter()
                .copied()
                .enumerate()
                .map(|(position, index)| (index, position))
                .collect::<BTreeMap<_, _>>();
            let compact_color = source_colors
                .iter()
                .copied()
                .enumerate()
                .map(|(position, index)| (index, position))
                .collect::<BTreeMap<_, _>>();
            let routes = selected_routes
                .into_iter()
                .map(|route| {
                    let source_helicity = route.source_index / plan.color_count;
                    let source_color = route.source_index % plan.color_count;
                    LcResolvedReplayRoute {
                        source_index: compact_helicity[&source_helicity] * source_colors.len()
                            + compact_color[&source_color],
                        target_index: route.target_index,
                        weight: route.weight,
                    }
                })
                .collect();
            mapping_indices.push(mapping_index);
            entries.push(LcResolvedReplayEntry { routes });
            source_helicity_indices.push(source_helicities);
            source_color_indices.push(source_colors);
        }
        if entries.len() != mapping_indices.len()
            || entries.len() != source_helicity_indices.len()
            || entries.len() != source_color_indices.len()
        {
            return Err(RusticolError::integrity(
                "LC topology replay selection has inconsistent mappings and entries",
            ));
        }
        let mut entry_indices_by_source = BTreeMap::<(Vec<usize>, Vec<usize>), Vec<usize>>::new();
        for entry_index in 0..entries.len() {
            entry_indices_by_source
                .entry((
                    source_helicity_indices[entry_index].clone(),
                    source_color_indices[entry_index].clone(),
                ))
                .or_default()
                .push(entry_index);
        }
        let source_groups = entry_indices_by_source
            .into_iter()
            .map(|((helicity_indices, color_indices), entry_indices)| {
                let helicity_ids = helicity_indices
                    .iter()
                    .map(|index| self.manifest.helicities[*index].id.clone())
                    .collect();
                let color_ids = color_indices
                    .iter()
                    .map(|index| self.manifest.color_components[*index].id().to_string())
                    .collect();
                LcResolvedReplaySourceGroup {
                    mapping_indices: entry_indices
                        .iter()
                        .map(|entry_index| mapping_indices[*entry_index])
                        .collect(),
                    entries: entry_indices
                        .iter()
                        .map(|entry_index| entries[*entry_index].clone())
                        .collect(),
                    helicity_ids,
                    color_ids,
                    materialized_sector_ids: None,
                    direct_total_plan: None,
                    source_component_count: helicity_indices.len() * color_indices.len(),
                }
            })
            .collect();
        Ok(LcResolvedReplaySelection {
            #[cfg(test)]
            mapping_indices,
            #[cfg(test)]
            entries,
            #[cfg(test)]
            source_helicity_indices,
            #[cfg(test)]
            source_color_indices,
            source_groups,
            helicity_indices,
            color_indices,
        })
    }
}

fn canonical_total_axis_selector<'a>(
    ids: Option<&'a BTreeSet<String>>,
    available: &BTreeMap<String, usize>,
) -> Option<&'a BTreeSet<String>> {
    ids.filter(|ids| {
        ids.len() != available.len() || ids.iter().any(|id| !available.contains_key(id))
    })
}

fn numeric_reduction_group(
    group: &crate::ReductionGroup,
    manifest: &ProcessPhysicsV1,
    helicity_index_by_id: &BTreeMap<String, usize>,
    color_index_by_id: &BTreeMap<String, usize>,
) -> RusticolResult<NumericReductionGroup> {
    let mut physical_helicity_indices = Vec::with_capacity(group.physical_helicity_ids.len());
    let mut helicity_membership = vec![false; manifest.helicities.len()];
    let mut normalized_helicity_weights = Vec::with_capacity(group.physical_helicity_ids.len());
    let mut total_helicity_weight = 0.0;
    for id in &group.physical_helicity_ids {
        let index = *helicity_index_by_id.get(id).ok_or_else(|| {
            RusticolError::artifact(format!(
                "resolved reduction group {} references unknown helicity {id:?}",
                group.id
            ))
        })?;
        let coefficient = manifest.helicities[index].coefficient;
        physical_helicity_indices.push(index);
        helicity_membership[index] = true;
        normalized_helicity_weights.push((index, coefficient));
        total_helicity_weight += coefficient;
    }
    if !total_helicity_weight.is_finite() || total_helicity_weight <= 0.0 {
        return Err(RusticolError::artifact(format!(
            "resolved reduction group {} has no positive helicity weight",
            group.id
        )));
    }
    for (_, weight) in &mut normalized_helicity_weights {
        *weight /= total_helicity_weight;
    }

    let mut physical_color_weights = Vec::with_capacity(group.physical_color_ids.len());
    let mut total_color_weight = 0.0;
    for id in &group.physical_color_ids {
        let index = *color_index_by_id.get(id).ok_or_else(|| {
            RusticolError::artifact(format!(
                "resolved reduction group {} references unknown color {id:?}",
                group.id
            ))
        })?;
        let coefficient = manifest.color_components[index].coefficient();
        physical_color_weights.push((index, coefficient));
        total_color_weight += coefficient;
    }
    if !total_color_weight.is_finite() || total_color_weight <= 0.0 {
        return Err(RusticolError::artifact(format!(
            "resolved reduction group {} has no positive color weight",
            group.id
        )));
    }
    let mut normalized_color_weights = physical_color_weights.clone();
    for (_, weight) in &mut normalized_color_weights {
        *weight /= total_color_weight;
    }

    let member_count = group
        .physical_helicity_ids
        .len()
        .checked_mul(group.physical_color_ids.len())
        .ok_or_else(|| RusticolError::artifact("resolved reduction member count overflows"))?;
    let mut normalized_member_weights = Vec::with_capacity(member_count);
    let mut total_member_weight = 0.0;
    for helicity_id in &group.physical_helicity_ids {
        let helicity_index = helicity_index_by_id[helicity_id];
        let helicity_weight = manifest.helicities[helicity_index].coefficient;
        for (color_index, color_weight) in &physical_color_weights {
            let weight = helicity_weight * *color_weight;
            total_member_weight += weight;
            normalized_member_weights.push((helicity_index, *color_index, weight));
        }
    }
    if !total_member_weight.is_finite() || total_member_weight <= 0.0 {
        return Err(RusticolError::artifact(format!(
            "resolved reduction group {} has no positive member weight",
            group.id
        )));
    }
    for (_, _, weight) in &mut normalized_member_weights {
        *weight /= total_member_weight;
    }

    Ok(NumericReductionGroup {
        physical_helicity_indices,
        helicity_membership,
        normalized_helicity_weights,
        normalized_color_weights,
        normalized_member_weights,
    })
}

impl ExecutionRuntime {
    pub(super) fn set_external_pdg_order_recursive(&mut self, pdgs: &[i32]) {
        self.external_pdg_order = pdgs.to_vec();
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            sum_runtime.set_external_pdg_order_recursive(pdgs);
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            selector_runtime.set_external_pdg_order_recursive(pdgs);
        }
        for selector_runtime in self.color_selector_runtimes.values_mut() {
            selector_runtime.set_external_pdg_order_recursive(pdgs);
        }
    }

    pub(super) fn remap_physics_reduction_overrides(
        &mut self,
        helicity_id_map: &BTreeMap<String, String>,
        color_id_map: &BTreeMap<String, String>,
    ) -> RusticolResult<()> {
        self.lc_resolved_replay_plan = None;
        self.lc_resolved_replay_selection_cache = None;
        if let Some(reduction) = self.physics_reduction_override.as_mut() {
            remap_reduction_ids(reduction, helicity_id_map, color_id_map)?;
        }
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            sum_runtime.remap_physics_reduction_overrides(helicity_id_map, color_id_map)?;
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            selector_runtime.remap_physics_reduction_overrides(helicity_id_map, color_id_map)?;
        }
        for selector_runtime in self.color_selector_runtimes.values_mut() {
            selector_runtime.remap_physics_reduction_overrides(helicity_id_map, color_id_map)?;
        }
        Ok(())
    }

    pub(super) fn attach_physics(&mut self, physics: Arc<PhysicsRuntime>) -> RusticolResult<()> {
        self.lc_resolved_replay_plan = None;
        self.lc_resolved_replay_selection_cache = None;
        self.physics = Some(Arc::clone(&physics));
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            let sum_physics =
                if let Some(reduction) = sum_runtime.physics_reduction_override.as_ref() {
                    let mut manifest = physics.manifest.clone();
                    manifest.reduction = reduction.clone();
                    Arc::new(PhysicsRuntime::new(manifest)?)
                } else {
                    Arc::clone(&physics)
                };
            sum_runtime.attach_physics(sum_physics)?;
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            let selector_physics =
                if let Some(reduction) = selector_runtime.physics_reduction_override.as_ref() {
                    let mut manifest = physics.manifest.clone();
                    manifest.reduction = reduction.clone();
                    Arc::new(PhysicsRuntime::new(manifest)?)
                } else {
                    Arc::clone(&physics)
                };
            selector_runtime.attach_physics(selector_physics)?;
        }
        for selector_runtime in self.color_selector_runtimes.values_mut() {
            let selector_physics =
                if let Some(reduction) = selector_runtime.physics_reduction_override.as_ref() {
                    let mut manifest = physics.manifest.clone();
                    manifest.reduction = reduction.clone();
                    Arc::new(PhysicsRuntime::new(manifest)?)
                } else {
                    Arc::clone(&physics)
                };
            selector_runtime.attach_physics(selector_physics)?;
        }
        Ok(())
    }

    pub(super) fn cached_lc_resolved_replay_plan(
        &mut self,
        physics: &PhysicsRuntime,
    ) -> RusticolResult<Arc<LcResolvedReplayPlan>> {
        if let Some(plan) = &self.lc_resolved_replay_plan {
            return Ok(Arc::clone(plan));
        }
        let materialized_sector_by_id = self.lc_materialized_sectors_by_id(physics)?;
        let plan = Arc::new(physics.lc_resolved_replay_plan(
            &self.lc_topology_replay_public_mappings,
            &self.lc_topology_replay_routes,
            &materialized_sector_by_id,
        )?);
        self.lc_resolved_replay_plan = Some(Arc::clone(&plan));
        Ok(plan)
    }

    pub(super) fn cached_lc_resolved_replay_selection(
        &mut self,
        physics: &PhysicsRuntime,
        plan: &LcResolvedReplayPlan,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Arc<LcResolvedReplaySelection>> {
        if let Some((cached_key, selection)) = &self.lc_resolved_replay_selection_cache
            && selected_helicity_ids.is_none()
            && cached_key.helicity_indices.is_none()
            && let Some(color_ids) = selected_color_ids
            && color_ids.len() == 1
            && let Some(color_id) = color_ids.iter().next()
            && let Some(color_index) = physics.color_index_by_id.get(color_id)
            && cached_key.color_indices.as_deref() == Some(std::slice::from_ref(color_index))
        {
            return Ok(Arc::clone(selection));
        }
        if let Some((cached_key, selection)) = &self.lc_resolved_replay_selection_cache
            && selected_color_ids.is_none()
            && cached_key.color_indices.is_none()
            && let Some(helicity_ids) = selected_helicity_ids
            && helicity_ids.len() == 1
            && let Some(helicity_id) = helicity_ids.iter().next()
            && let Some(helicity_index) = physics.helicity_index_by_id.get(helicity_id)
            && cached_key.helicity_indices.as_deref() == Some(std::slice::from_ref(helicity_index))
        {
            return Ok(Arc::clone(selection));
        }
        let key =
            physics.lc_resolved_replay_selection_key(selected_helicity_ids, selected_color_ids)?;
        if let Some((cached_key, selection)) = &self.lc_resolved_replay_selection_cache
            && cached_key == &key
        {
            return Ok(Arc::clone(selection));
        }
        let mut selection = physics.select_lc_resolved_replay_plan_for_key(plan, &key)?;
        let sectors_by_id = self.lc_materialized_sectors_by_id(physics)?;
        let sector_by_color_index = sectors_by_id
            .iter()
            .map(|(sector_id, sector)| (sector.color_index, *sector_id))
            .collect::<BTreeMap<_, _>>();
        let target_component_count = selection
            .helicity_indices
            .len()
            .checked_mul(selection.color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        for source_group in &mut selection.source_groups {
            let materialized_sector_ids = source_group
                .color_ids
                .iter()
                .map(|color_id| {
                    let color_index = physics.color_index_by_id.get(color_id).ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "LC topology replay selected unknown materialized color {color_id:?}"
                        ))
                    })?;
                    sector_by_color_index.get(color_index).copied().ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "LC topology replay has no materialized sector for color {color_id:?}"
                        ))
                    })
                })
                .collect::<RusticolResult<BTreeSet<_>>>()?;
            source_group.direct_total_plan = None;
            if selected_helicity_ids.is_none()
                && let ([mapping_index], [replay_entry]) = (
                    source_group.mapping_indices.as_slice(),
                    source_group.entries.as_slice(),
                )
                && materialized_sector_ids.len() == 1
                && let Some(materialized_sector_id) = materialized_sector_ids.iter().next().copied()
                && let Some(scale) = uniform_lc_replay_total_scale(
                    replay_entry,
                    source_group.source_component_count,
                    target_component_count,
                )
                && self
                    .color_selector_runtimes
                    .get(&materialized_sector_id)
                    .is_some_and(|runtime| {
                        certifies_lc_direct_total_source(
                            runtime,
                            &source_group.helicity_ids,
                            &source_group.color_ids,
                        )
                    })
            {
                source_group.direct_total_plan = Some(LcDirectTotalPlan {
                    mapping_index: *mapping_index,
                    materialized_sector_id,
                    scale,
                });
            }
            source_group.materialized_sector_ids = Some(materialized_sector_ids);
        }
        let selection = Arc::new(selection);
        self.lc_resolved_replay_selection_cache = Some((key, Arc::clone(&selection)));
        Ok(selection)
    }

    pub(super) fn lc_materialized_sectors_by_id(
        &self,
        physics: &PhysicsRuntime,
    ) -> RusticolResult<BTreeMap<i64, LcMaterializedSector>> {
        let amplitude = self.amplitude_stage.as_ref().ok_or_else(|| {
            RusticolError::artifact("LC topology replay requires a loaded compiled amplitude stage")
        })?;
        self.lc_materialized_sectors_by_id_from_groups(physics, &amplitude.raw_sum_groups)
    }

    pub(super) fn lc_materialized_sector_ids_for_color_ids(
        &self,
        physics: &PhysicsRuntime,
        color_ids: &BTreeSet<String>,
    ) -> RusticolResult<BTreeSet<i64>> {
        let sectors_by_id = self.lc_materialized_sectors_by_id(physics)?;
        let sector_by_color_index = sectors_by_id
            .iter()
            .map(|(sector_id, sector)| (sector.color_index, *sector_id))
            .collect::<BTreeMap<_, _>>();
        color_ids
            .iter()
            .map(|color_id| {
                let color_index = physics.color_index_by_id.get(color_id).ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "LC topology replay selected unknown materialized color {color_id:?}"
                    ))
                })?;
                sector_by_color_index
                    .get(color_index)
                    .copied()
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "LC topology replay color {color_id:?} has no materialized sector"
                        ))
                    })
            })
            .collect()
    }

    pub(super) fn lc_materialized_sectors_by_id_from_groups(
        &self,
        physics: &PhysicsRuntime,
        groups: &[RawSumGroup],
    ) -> RusticolResult<BTreeMap<i64, LcMaterializedSector>> {
        let mut result = BTreeMap::new();
        for group in groups {
            if group.sector_ids.is_empty() {
                continue;
            }
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "LC topology replay is missing physics reduction group {}",
                        group.id
                    ))
                })?;
            let computed = reduction
                .physical_color_ids
                .iter()
                .map(|id| physics.color_index_by_id[id])
                .filter(|index| physics.color_is_computed(*index))
                .collect::<BTreeSet<_>>();
            if computed.len() != 1 {
                return Err(RusticolError::artifact(format!(
                    "LC topology replay coherent group {} has {} computed color representatives, expected one",
                    group.id,
                    computed.len()
                )));
            }
            let color_index = *computed.iter().next().expect("one computed color checked");
            let sector = LcMaterializedSector {
                color_index,
                // all_sector_weight folds independent helicity and color
                // multiplicities. Resolved public-flow expansion must only
                // distribute the color factor; helicity replay is handled by
                // its own selector/reduction axis.
                reduction_weight: lc_color_replay_weight(group)?,
            };
            for sector_id in &group.sector_ids {
                if let Some(previous) = result.insert(*sector_id, sector)
                    && (previous.color_index != sector.color_index
                        || previous.reduction_weight.to_bits() != sector.reduction_weight.to_bits())
                {
                    return Err(RusticolError::artifact(format!(
                        "LC topology replay materialized sector {sector_id} maps to inconsistent computed colors or weights"
                    )));
                }
            }
        }
        for sector_id in &self.lc_topology_replay_materialized_sector_ids {
            if !result.contains_key(sector_id) {
                return Err(RusticolError::artifact(format!(
                    "LC topology replay materialized sector {sector_id} is absent from the amplitude groups"
                )));
            }
        }
        Ok(result)
    }

    pub(super) fn remap_lc_topology_replay_public_labels(
        &mut self,
        representative_to_public: &[usize],
    ) -> RusticolResult<()> {
        if representative_to_public.len() != self.external_count
            || representative_to_public
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != (0..self.external_count).collect::<BTreeSet<_>>()
        {
            return Err(RusticolError::artifact(
                "process alias has an invalid public-label permutation for LC topology replay",
            ));
        }
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            sum_runtime.remap_lc_topology_replay_public_labels(representative_to_public)?;
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            selector_runtime.remap_lc_topology_replay_public_labels(representative_to_public)?;
        }
        if !self.lc_topology_replay_enabled {
            return Ok(());
        }
        self.lc_topology_replay_public_mappings = self
            .lc_topology_replay_mappings
            .iter()
            .map(|mapping| {
                mapping
                    .iter()
                    .map(|(representative, sector)| {
                        (
                            representative_to_public[*representative],
                            representative_to_public[*sector],
                        )
                    })
                    .collect()
            })
            .collect();
        self.lc_resolved_replay_plan = None;
        self.lc_resolved_replay_selection_cache = None;
        Ok(())
    }
}

pub(super) fn certifies_lc_direct_total_source(
    runtime: &ExecutionRuntime,
    helicity_ids: &BTreeSet<String>,
    color_ids: &BTreeSet<String>,
) -> bool {
    if helicity_ids.is_empty()
        || color_ids.is_empty()
        || runtime.helicity_sum_runtime.is_some()
        || runtime.lc_topology_replay_enabled
        || runtime.has_compiled_helicity_execution_plan()
    {
        return false;
    }
    let Some(physics) = runtime.physics.as_deref() else {
        return false;
    };
    let Some(amplitude) = runtime.amplitude_stage.as_ref() else {
        return false;
    };
    if amplitude.color_contraction.is_some()
        || physics.manifest.reduction.kind != crate::ReductionKind::LcDiagonal
    {
        return false;
    }
    let Some(expected_component_count) = helicity_ids.len().checked_mul(color_ids.len()) else {
        return false;
    };
    let groups = &physics.manifest.reduction.groups;
    if groups.is_empty()
        || amplitude.raw_sum_groups.len() != groups.len()
        || runtime.amplitude_output_count != groups.len()
    {
        return false;
    }
    let raw_group_ids = amplitude
        .raw_sum_groups
        .iter()
        .map(|group| group.id)
        .collect::<BTreeSet<_>>();
    if raw_group_ids != physics.reduction_by_group_id.keys().copied().collect() {
        return false;
    }
    let mut covered_members = BTreeSet::new();
    for group in groups {
        if group.physical_helicity_ids.is_empty()
            || group.physical_color_ids.is_empty()
            || !group
                .physical_helicity_ids
                .contains(&group.representative_helicity_id)
            || !group
                .physical_color_ids
                .contains(&group.representative_color_id)
        {
            return false;
        }
        for helicity_id in &group.physical_helicity_ids {
            for color_id in &group.physical_color_ids {
                if !helicity_ids.contains(helicity_id)
                    || !color_ids.contains(color_id)
                    || !covered_members.insert((helicity_id.as_str(), color_id.as_str()))
                {
                    return false;
                }
            }
        }
    }
    covered_members.len() == expected_component_count
}

fn remap_reduction_ids(
    reduction: &mut crate::Reduction,
    helicity_id_map: &BTreeMap<String, String>,
    color_id_map: &BTreeMap<String, String>,
) -> RusticolResult<()> {
    for group in &mut reduction.groups {
        group.representative_helicity_id = helicity_id_map
            .get(&group.representative_helicity_id)
            .cloned()
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "reduction override representative helicity {:?} is absent from alias remapping",
                    group.representative_helicity_id
                ))
            })?;
        group.representative_color_id = color_id_map
            .get(&group.representative_color_id)
            .cloned()
            .ok_or_else(|| {
            RusticolError::integrity(format!(
                "reduction override representative color {:?} is absent from alias remapping",
                group.representative_color_id
            ))
        })?;
        for id in &mut group.physical_helicity_ids {
            *id = helicity_id_map.get(id).cloned().ok_or_else(|| {
                RusticolError::integrity(format!(
                    "reduction override physical helicity {id:?} is absent from alias remapping"
                ))
            })?;
        }
        for id in &mut group.physical_color_ids {
            *id = color_id_map.get(id).cloned().ok_or_else(|| {
                RusticolError::integrity(format!(
                    "reduction override physical color {id:?} is absent from alias remapping"
                ))
            })?;
        }
    }
    Ok(())
}

fn lc_color_replay_weight(group: &RawSumGroup) -> RusticolResult<f64> {
    if !group.weight.is_finite()
        || group.weight <= 0.0
        || !group.all_sector_weight.is_finite()
        || group.all_sector_weight <= 0.0
    {
        return Err(RusticolError::artifact(format!(
            "LC topology replay coherent group {} has no positive helicity/color weight",
            group.id
        )));
    }
    let color_replay_weight = group.all_sector_weight / group.weight;
    if !color_replay_weight.is_finite() || color_replay_weight <= 0.0 {
        return Err(RusticolError::artifact(format!(
            "LC topology replay coherent group {} has no positive color-only weight",
            group.id
        )));
    }
    Ok(color_replay_weight)
}

fn validate_public_label_permutation(
    mapping: &[(usize, usize)],
    external_count: usize,
) -> RusticolResult<()> {
    let mut representatives = BTreeSet::new();
    let mut sectors = BTreeSet::new();
    for (representative, sector) in mapping {
        if *representative >= external_count || *sector >= external_count {
            return Err(RusticolError::artifact(
                "LC topology replay public-label mapping references an out-of-range external leg",
            ));
        }
        if !representatives.insert(*representative) || !sectors.insert(*sector) {
            return Err(RusticolError::artifact(
                "LC topology replay public-label mapping is not one-to-one",
            ));
        }
    }
    if representatives != sectors {
        return Err(RusticolError::artifact(
            "LC topology replay public-label mapping is not a permutation of its support",
        ));
    }
    Ok(())
}

fn permute_public_helicity(values: &[i32], mapping: &[(usize, usize)]) -> Vec<i32> {
    let mut target = values.to_vec();
    for (representative, sector) in mapping {
        target[*sector] = values[*representative];
    }
    target
}

fn permute_public_color_word(word: &[usize], mapping: &[(usize, usize)]) -> Vec<usize> {
    let labels = mapping.iter().copied().collect::<BTreeMap<_, _>>();
    word.iter()
        .map(|label| {
            label
                .checked_sub(1)
                .and_then(|index| labels.get(&index).copied())
                .map(|index| index + 1)
                .unwrap_or(*label)
        })
        .collect()
}

fn replay_public_color_words(
    word: &[usize],
    replay_weight: f64,
) -> RusticolResult<Vec<Vec<usize>>> {
    const WEIGHT_TOLERANCE: f64 = 1.0e-12;
    if (replay_weight - 1.0).abs() <= WEIGHT_TOLERANCE {
        return Ok(vec![word.to_vec()]);
    }
    if (replay_weight - 2.0).abs() > WEIGHT_TOLERANCE {
        return Err(RusticolError::compatibility(format!(
            "resolved LC topology replay cannot expand sector weight {replay_weight}; schema-v3 replay metadata defines public-flow expansion only for unit sectors and folded trace-reflection weight two"
        )));
    }
    if word.len() < 2 {
        return Ok(vec![word.to_vec()]);
    }
    let mut reflected = Vec::with_capacity(word.len());
    reflected.push(word[0]);
    reflected.extend(word[1..].iter().rev().copied());
    if reflected == word {
        Ok(vec![word.to_vec()])
    } else {
        Ok(vec![word.to_vec(), reflected])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_replay_weight_excludes_helicity_multiplicity() {
        let group = RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 2.0,
            all_sector_weight: 4.0,
            sector_ids: vec![0],
        };

        assert_eq!(lc_color_replay_weight(&group).unwrap(), 2.0);
    }

    #[test]
    fn color_replay_weight_rejects_invalid_axes() {
        let group = RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 0.0,
            all_sector_weight: 2.0,
            sector_ids: vec![0],
        };

        assert!(lc_color_replay_weight(&group).is_err());
    }
}
