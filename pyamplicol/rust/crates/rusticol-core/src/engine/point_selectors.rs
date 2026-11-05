// SPDX-License-Identifier: 0BSD

use super::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct PointSelectorKey {
    pub(super) helicity_index: Option<usize>,
    pub(super) color_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PointSelectorPartition {
    pub(super) key: PointSelectorKey,
    pub(super) rows: PointSelectorRows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PointSelectorRows {
    Contiguous { start: usize, end: usize },
    Gathered { start: usize, end: usize },
}

impl PointSelectorRows {
    pub(super) fn len(self) -> usize {
        match self {
            Self::Contiguous { start, end } | Self::Gathered { start, end } => end - start,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PointSelectorPlan {
    None,
    Homogeneous(PointSelectorKey),
    Partitioned,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PointSelectorPlanProfile {
    pub(super) kind: &'static str,
    pub(super) group_sizes: Vec<usize>,
    pub(super) reordered_point_count: usize,
    pub(super) simd_lane_width: usize,
    pub(super) simd_occupancy: f64,
}

/// Reusable storage for stable per-point selector grouping.
///
/// Partition order follows first occurrence and row order within a gathered
/// partition follows caller order. `HashMap` iteration order therefore never
/// affects execution order. Clearing the map and vectors retains their backing
/// allocations for subsequent calls of the same or smaller shape.
#[derive(Default)]
pub(super) struct PointSelectorPlanner {
    partition_index_by_key: HashMap<PointSelectorKey, usize>,
    partitions: Vec<PointSelectorPartition>,
    partition_counts: Vec<usize>,
    partition_cursors: Vec<usize>,
    gathered_rows: Vec<usize>,
}

impl PointSelectorPlanner {
    pub(super) fn build(
        &mut self,
        point_count: usize,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
        helicity_count: usize,
        color_count: usize,
    ) -> RusticolResult<PointSelectorPlan> {
        self.clear_plan();
        if helicity_by_point.is_none() && color_by_point.is_none() {
            return Ok(PointSelectorPlan::None);
        }
        validate_selector_axis(
            helicity_by_point,
            point_count,
            helicity_count,
            "helicity_by_point",
        )?;
        validate_selector_axis(
            color_by_point,
            point_count,
            color_count,
            "color_flow_by_point",
        )?;

        let selector_key = |point_index: usize| PointSelectorKey {
            helicity_index: helicity_by_point.map(|values| values[point_index] as usize),
            color_index: color_by_point.map(|values| values[point_index] as usize),
        };
        if point_count == 0 {
            return Ok(PointSelectorPlan::Partitioned);
        }

        let first = selector_key(0);
        if (1..point_count).all(|point_index| selector_key(point_index) == first) {
            return Ok(PointSelectorPlan::Homogeneous(first));
        }

        if self.build_contiguous_runs(point_count, &selector_key) {
            return Ok(PointSelectorPlan::Partitioned);
        }

        self.build_stable_gathered_partitions(point_count, &selector_key);
        Ok(PointSelectorPlan::Partitioned)
    }

    pub(super) fn partitions(&self) -> &[PointSelectorPartition] {
        &self.partitions
    }

    pub(super) fn gathered_rows(&self, rows: PointSelectorRows) -> &[usize] {
        match rows {
            PointSelectorRows::Gathered { start, end } => &self.gathered_rows[start..end],
            PointSelectorRows::Contiguous { .. } => &[],
        }
    }

    pub(super) fn profile(
        &self,
        plan: PointSelectorPlan,
        point_count: usize,
        simd_lane_width: usize,
    ) -> PointSelectorPlanProfile {
        let (kind, group_sizes, reordered_point_count) = match plan {
            PointSelectorPlan::None => ("none", Vec::new(), 0),
            PointSelectorPlan::Homogeneous(_) => ("homogeneous", vec![point_count], 0),
            PointSelectorPlan::Partitioned
                if self.partitions.iter().all(|partition| {
                    matches!(partition.rows, PointSelectorRows::Contiguous { .. })
                }) =>
            {
                (
                    "contiguous",
                    self.partitions
                        .iter()
                        .map(|partition| partition.rows.len())
                        .collect(),
                    0,
                )
            }
            PointSelectorPlan::Partitioned => {
                let reordered_point_count = self
                    .gathered_rows
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(grouped_index, source_index)| grouped_index != source_index)
                    .count();
                (
                    "stable-grouped",
                    self.partitions
                        .iter()
                        .map(|partition| partition.rows.len())
                        .collect(),
                    reordered_point_count,
                )
            }
        };
        let simd_lane_width = simd_lane_width.max(1);
        let padded_point_count = group_sizes
            .iter()
            .map(|size| size.div_ceil(simd_lane_width) * simd_lane_width)
            .sum::<usize>();
        let simd_occupancy = if point_count == 0 || padded_point_count == 0 {
            1.0
        } else {
            point_count as f64 / padded_point_count as f64
        };
        PointSelectorPlanProfile {
            kind,
            group_sizes,
            reordered_point_count,
            simd_lane_width,
            simd_occupancy,
        }
    }

    fn clear_plan(&mut self) {
        self.partition_index_by_key.clear();
        self.partitions.clear();
        self.partition_counts.clear();
        self.partition_cursors.clear();
        self.gathered_rows.clear();
    }

    fn build_contiguous_runs(
        &mut self,
        point_count: usize,
        selector_key: &impl Fn(usize) -> PointSelectorKey,
    ) -> bool {
        let mut run_start = 0;
        let mut run_key = selector_key(0);
        for point_index in 1..=point_count {
            let next_key = (point_index < point_count).then(|| selector_key(point_index));
            if next_key == Some(run_key) {
                continue;
            }
            if self.partition_index_by_key.contains_key(&run_key) {
                return false;
            }
            self.partition_index_by_key
                .insert(run_key, self.partitions.len());
            self.partitions.push(PointSelectorPartition {
                key: run_key,
                rows: PointSelectorRows::Contiguous {
                    start: run_start,
                    end: point_index,
                },
            });
            if let Some(key) = next_key {
                run_start = point_index;
                run_key = key;
            }
        }
        true
    }

    fn build_stable_gathered_partitions(
        &mut self,
        point_count: usize,
        selector_key: &impl Fn(usize) -> PointSelectorKey,
    ) {
        self.partition_index_by_key.clear();
        self.partitions.clear();
        self.partition_counts.clear();

        for point_index in 0..point_count {
            let key = selector_key(point_index);
            let partition_index = match self.partition_index_by_key.get(&key).copied() {
                Some(index) => index,
                None => {
                    let index = self.partitions.len();
                    self.partition_index_by_key.insert(key, index);
                    self.partitions.push(PointSelectorPartition {
                        key,
                        rows: PointSelectorRows::Gathered { start: 0, end: 0 },
                    });
                    self.partition_counts.push(0);
                    index
                }
            };
            self.partition_counts[partition_index] += 1;
        }

        self.partition_cursors.resize(self.partitions.len(), 0);
        let mut offset = 0;
        for (partition_index, partition) in self.partitions.iter_mut().enumerate() {
            let end = offset + self.partition_counts[partition_index];
            partition.rows = PointSelectorRows::Gathered { start: offset, end };
            self.partition_cursors[partition_index] = offset;
            offset = end;
        }
        self.gathered_rows.resize(point_count, 0);
        for point_index in 0..point_count {
            let partition_index = self.partition_index_by_key[&selector_key(point_index)];
            let cursor = &mut self.partition_cursors[partition_index];
            self.gathered_rows[*cursor] = point_index;
            *cursor += 1;
        }
    }
}

#[derive(Default)]
pub(super) struct SelectorSetCache {
    entries: Vec<SelectorSetCacheEntry>,
}

struct SelectorSetCacheEntry {
    ids: Vec<String>,
    selected: BTreeSet<String>,
}

impl SelectorSetCache {
    const MAX_ENTRIES: usize = 64;

    pub(super) fn resolve(
        &mut self,
        ids: Option<&[String]>,
        kind: &str,
    ) -> RusticolResult<Option<&BTreeSet<String>>> {
        let Some(ids) = ids else {
            return Ok(None);
        };
        if ids.is_empty() {
            return Err(RusticolError::selector(format!(
                "resolved {kind} selection must not be empty"
            )));
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.ids.as_slice() == ids)
        {
            return Ok(Some(&self.entries[index].selected));
        }
        let selected = ids.iter().cloned().collect::<BTreeSet<_>>();
        if selected.len() != ids.len() {
            return Err(RusticolError::selector(format!(
                "resolved {kind} selection contains duplicate ids"
            )));
        }
        if self.entries.len() == Self::MAX_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push(SelectorSetCacheEntry {
            ids: ids.to_vec(),
            selected,
        });
        Ok(Some(
            &self
                .entries
                .last()
                .expect("selector cache entry was just appended")
                .selected,
        ))
    }
}

#[cfg(test)]
mod selector_set_cache_tests {
    use super::*;

    #[test]
    fn selector_set_cache_is_bounded_and_rejects_duplicates() {
        let mut cache = SelectorSetCache::default();
        for index in 0..=SelectorSetCache::MAX_ENTRIES {
            let ids = vec![format!("id-{index}")];
            cache.resolve(Some(&ids), "test").unwrap();
        }
        assert_eq!(cache.entries.len(), SelectorSetCache::MAX_ENTRIES);
        assert_eq!(cache.entries[0].ids, vec!["id-1"]);

        let duplicate = vec!["same".to_string(), "same".to_string()];
        assert!(cache.resolve(Some(&duplicate), "test").is_err());
        assert_eq!(cache.entries.len(), SelectorSetCache::MAX_ENTRIES);
    }
}

#[derive(Default)]
pub(super) struct PointSelectorExecutionScratch {
    pub(super) planner: PointSelectorPlanner,
    pub(super) gathered_batch: Vec<Vec<[f64; 4]>>,
    pub(super) partition_totals: Vec<f64>,
    pub(super) output_totals: Vec<f64>,
    pub(super) helicity_selector_sets: SelectorSetCache,
    pub(super) color_selector_sets: SelectorSetCache,
    pub(super) helicity_singletons: Vec<BTreeSet<String>>,
    pub(super) color_singletons: Vec<BTreeSet<String>>,
}

impl PointSelectorExecutionScratch {
    pub(super) fn prepare_singletons(&mut self, physics: &PhysicsRuntime) {
        if self.helicity_singletons.len() != physics.manifest.helicities.len() {
            self.helicity_singletons.clear();
            self.helicity_singletons
                .reserve(physics.manifest.helicities.len());
            self.helicity_singletons.extend(
                physics
                    .manifest
                    .helicities
                    .iter()
                    .map(|helicity| BTreeSet::from([helicity.id.clone()])),
            );
        }
        if self.color_singletons.len() != physics.manifest.color_components.len() {
            self.color_singletons.clear();
            self.color_singletons
                .reserve(physics.manifest.color_components.len());
            self.color_singletons.extend(
                physics
                    .manifest
                    .color_components
                    .iter()
                    .map(|color| BTreeSet::from([color.id().to_string()])),
            );
        }
    }
}

pub(super) fn fill_gathered_batch<'a>(
    gathered_batch: &'a mut Vec<Vec<[f64; 4]>>,
    batch: &[Vec<[f64; 4]>],
    point_indices: &[usize],
) -> &'a [Vec<[f64; 4]>] {
    if gathered_batch.len() < point_indices.len() {
        gathered_batch.resize_with(point_indices.len(), Vec::new);
    }
    for (target, source_index) in gathered_batch.iter_mut().zip(point_indices.iter().copied()) {
        target.clear();
        target.extend_from_slice(&batch[source_index]);
    }
    &gathered_batch[..point_indices.len()]
}

pub(super) fn fill_gathered_batch_from_view<'a>(
    gathered_batch: &'a mut Vec<Vec<[f64; 4]>>,
    batch: F64MomentumBatchView<'_>,
    point_indices: &[usize],
) -> RusticolResult<&'a [Vec<[f64; 4]>]> {
    if gathered_batch.len() < point_indices.len() {
        gathered_batch.resize_with(point_indices.len(), Vec::new);
    }
    for (target, source_index) in gathered_batch.iter_mut().zip(point_indices.iter().copied()) {
        if source_index >= batch.point_count() {
            return Err(RusticolError::integrity(
                "point-selector gather references an out-of-range input row",
            ));
        }
        let point = batch.point(source_index);
        target.clear();
        target.reserve(batch.external_count());
        for external_index in 0..batch.external_count() {
            target.push(point.momentum(external_index).ok_or_else(|| {
                RusticolError::integrity(
                    "validated momentum view is missing an external leg during selector gather",
                )
            })?);
        }
    }
    Ok(&gathered_batch[..point_indices.len()])
}

#[cfg(test)]
pub(super) fn write_partition_totals(
    partition_totals: &mut Vec<f64>,
    resolved: &ResolvedValues<f64>,
) {
    let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
    partition_totals.clear();
    if component_count == 0 {
        partition_totals.resize(resolved.point_count, 0.0);
    } else {
        partition_totals.reserve(resolved.point_count);
        partition_totals.extend(
            resolved
                .values
                .chunks(component_count)
                .map(|point| point.iter().sum::<f64>()),
        );
    }
}

pub(super) fn scatter_partition_totals(
    values: &mut [f64],
    partition_totals: &[f64],
    rows: PointSelectorRows,
    planner: &PointSelectorPlanner,
) {
    match rows {
        PointSelectorRows::Contiguous { start, end } => {
            values[start..end].copy_from_slice(partition_totals);
        }
        rows @ PointSelectorRows::Gathered { .. } => {
            for (source, target) in partition_totals
                .iter()
                .copied()
                .zip(planner.gathered_rows(rows).iter().copied())
            {
                values[target] = source;
            }
        }
    }
}

fn validate_selector_axis(
    values: Option<&[u32]>,
    point_count: usize,
    available_count: usize,
    name: &str,
) -> RusticolResult<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.len() != point_count {
        return Err(RusticolError::selector(format!(
            "{name} contains {} entries, expected one selector for each of {point_count} points",
            values.len()
        )));
    }
    if let Some((point_index, selector)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, selector)| *selector as usize >= available_count)
    {
        return Err(RusticolError::selector(format!(
            "{name}[{point_index}]={selector} is out of range; the artifact exposes {available_count} selectors on this axis"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded_rows(planner: &PointSelectorPlanner) -> Vec<(PointSelectorKey, Vec<usize>, bool)> {
        planner
            .partitions()
            .iter()
            .map(|partition| match partition.rows {
                PointSelectorRows::Contiguous { start, end } => {
                    (partition.key, (start..end).collect(), true)
                }
                rows @ PointSelectorRows::Gathered { .. } => {
                    (partition.key, planner.gathered_rows(rows).to_vec(), false)
                }
            })
            .collect()
    }

    #[test]
    fn stable_random_grouping_preserves_first_occurrence_and_caller_order() {
        let colors = [2, 1, 2, 0, 1, 2, 0];
        let mut planner = PointSelectorPlanner::default();
        assert_eq!(
            planner
                .build(7, None, Some(&colors), 1, 3)
                .expect("selector plan"),
            PointSelectorPlan::Partitioned
        );
        assert_eq!(
            expanded_rows(&planner),
            vec![
                (
                    PointSelectorKey {
                        helicity_index: None,
                        color_index: Some(2),
                    },
                    vec![0, 2, 5],
                    false,
                ),
                (
                    PointSelectorKey {
                        helicity_index: None,
                        color_index: Some(1),
                    },
                    vec![1, 4],
                    false,
                ),
                (
                    PointSelectorKey {
                        helicity_index: None,
                        color_index: Some(0),
                    },
                    vec![3, 6],
                    false,
                ),
            ]
        );
    }

    #[test]
    fn profile_distinguishes_zero_reorder_and_stable_grouped_plans() {
        let mut planner = PointSelectorPlanner::default();
        let pooled = [0, 0, 1, 1];
        let plan = planner
            .build(4, None, Some(&pooled), 1, 2)
            .expect("pooled selector plan");
        let profile = planner.profile(plan, 4, 2);
        assert_eq!(profile.kind, "contiguous");
        assert_eq!(profile.group_sizes, vec![2, 2]);
        assert_eq!(profile.reordered_point_count, 0);

        let alternating = [0, 1, 0, 1];
        let plan = planner
            .build(4, None, Some(&alternating), 1, 2)
            .expect("alternating selector plan");
        let profile = planner.profile(plan, 4, 2);
        assert_eq!(profile.kind, "stable-grouped");
        assert_eq!(profile.group_sizes, vec![2, 2]);
        assert_eq!(profile.reordered_point_count, 2);
        assert!((0.0..=1.0).contains(&profile.simd_occupancy));
    }

    #[test]
    fn pooled_selector_runs_are_borrowed_without_gathering() {
        let colors = [2, 2, 1, 1, 0];
        let mut planner = PointSelectorPlanner::default();
        assert_eq!(
            planner
                .build(5, None, Some(&colors), 1, 3)
                .expect("selector plan"),
            PointSelectorPlan::Partitioned
        );
        assert_eq!(
            expanded_rows(&planner),
            vec![
                (
                    PointSelectorKey {
                        helicity_index: None,
                        color_index: Some(2),
                    },
                    vec![0, 1],
                    true,
                ),
                (
                    PointSelectorKey {
                        helicity_index: None,
                        color_index: Some(1),
                    },
                    vec![2, 3],
                    true,
                ),
                (
                    PointSelectorKey {
                        helicity_index: None,
                        color_index: Some(0),
                    },
                    vec![4],
                    true,
                ),
            ]
        );
        assert!(planner.gathered_rows.is_empty());
    }

    #[test]
    fn recognizes_homogeneous_selectors_without_partition_storage() {
        let helicities = [1, 1, 1, 1];
        let colors = [2, 2, 2, 2];
        let mut planner = PointSelectorPlanner::default();
        assert_eq!(
            planner
                .build(4, Some(&helicities), Some(&colors), 2, 3)
                .expect("selector plan"),
            PointSelectorPlan::Homogeneous(PointSelectorKey {
                helicity_index: Some(1),
                color_index: Some(2),
            })
        );
        assert!(planner.partitions().is_empty());
    }

    #[test]
    fn reuses_planner_gather_and_partition_result_buffers() {
        let colors = [2, 1, 2, 0, 1, 2, 0];
        let batch = (0..colors.len())
            .map(|point| vec![[point as f64, 0.0, 0.0, 0.0]; 3])
            .collect::<Vec<_>>();
        let mut scratch = PointSelectorExecutionScratch::default();

        scratch
            .planner
            .build(colors.len(), None, Some(&colors), 1, 3)
            .expect("first selector plan");
        let partition_storage = scratch.planner.partitions.as_ptr();
        let gathered_row_storage = scratch.planner.gathered_rows.as_ptr();
        let first_rows = scratch.planner.partitions[0].rows;
        let first_indices = scratch.planner.gathered_rows(first_rows).to_vec();
        fill_gathered_batch(&mut scratch.gathered_batch, &batch, &first_indices);
        let gathered_batch_storage = scratch.gathered_batch.as_ptr();
        let gathered_point_storage = scratch.gathered_batch[0].as_ptr();
        let resolved = ResolvedValues {
            values: vec![1.0, 2.0, 3.0],
            point_count: 3,
            helicity_indices: vec![0],
            color_indices: vec![0],
        };
        write_partition_totals(&mut scratch.partition_totals, &resolved);
        let partition_total_storage = scratch.partition_totals.as_ptr();

        scratch
            .planner
            .build(colors.len(), None, Some(&colors), 1, 3)
            .expect("second selector plan");
        assert_eq!(scratch.planner.partitions.as_ptr(), partition_storage);
        assert_eq!(scratch.planner.gathered_rows.as_ptr(), gathered_row_storage);
        let first_rows = scratch.planner.partitions[0].rows;
        let first_indices = scratch.planner.gathered_rows(first_rows).to_vec();
        fill_gathered_batch(&mut scratch.gathered_batch, &batch, &first_indices);
        assert_eq!(scratch.gathered_batch.as_ptr(), gathered_batch_storage);
        assert_eq!(scratch.gathered_batch[0].as_ptr(), gathered_point_storage);
        write_partition_totals(&mut scratch.partition_totals, &resolved);
        assert_eq!(scratch.partition_totals.as_ptr(), partition_total_storage);
    }

    #[test]
    fn gathered_partition_scatter_restores_caller_order() {
        let colors = [2, 1, 2, 0, 1, 2, 0];
        let mut planner = PointSelectorPlanner::default();
        planner
            .build(colors.len(), None, Some(&colors), 1, 3)
            .expect("selector plan");

        let mut values = vec![0.0; colors.len()];
        for partition in planner.partitions().iter().copied() {
            let point_indices = planner.gathered_rows(partition.rows);
            let partition_values = point_indices
                .iter()
                .map(|point_index| 100.0 + *point_index as f64)
                .collect::<Vec<_>>();
            scatter_partition_totals(&mut values, &partition_values, partition.rows, &planner);
        }

        assert_eq!(
            values,
            (0..colors.len())
                .map(|point_index| 100.0 + point_index as f64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn validates_lengths_and_bounds() {
        let mut planner = PointSelectorPlanner::default();
        let error = planner
            .build(2, Some(&[0]), None, 1, 1)
            .expect_err("length mismatch");
        assert!(
            error
                .to_string()
                .contains("one selector for each of 2 points")
        );

        let error = planner
            .build(2, None, Some(&[0, 1]), 1, 1)
            .expect_err("out-of-range selector");
        assert!(error.to_string().contains("color_flow_by_point[1]=1"));
    }
}
