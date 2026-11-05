// SPDX-License-Identifier: 0BSD

use super::super::*;

pub(crate) fn apply_input_crossing_map(
    batch: Vec<Vec<[f64; 4]>>,
    expected_legs: usize,
    input_crossing_map: Option<&[InputCrossingMapEntry]>,
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    let Some(map) = input_crossing_map else {
        return Ok(batch);
    };
    if map.len() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "input crossing map has {} entries, expected {expected_legs}",
            map.len()
        )));
    }
    let mut seen = vec![false; expected_legs];
    for entry in map {
        if entry.target_index >= expected_legs || entry.source_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "input crossing map references an out-of-range external leg",
            ));
        }
        if seen[entry.target_index] {
            return Err(RusticolError::invalid_argument(
                "input crossing map contains a duplicate target index",
            ));
        }
        seen[entry.target_index] = true;
    }
    if seen.iter().any(|value| !*value) {
        return Err(RusticolError::invalid_argument(
            "input crossing map does not cover every target index",
        ));
    }
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        let mut mapped = vec![[0.0; 4]; expected_legs];
        for entry in map {
            let source = point[entry.source_index];
            mapped[entry.target_index] = [
                entry.sign * source[0],
                entry.sign * source[1],
                entry.sign * source[2],
                entry.sign * source[3],
            ];
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn validate_input_crossing_map(
    expected_legs: usize,
    input_crossing_map: Option<&[InputCrossingMapEntry]>,
) -> RusticolResult<Option<&[InputCrossingMapEntry]>> {
    let Some(map) = input_crossing_map else {
        return Ok(None);
    };
    if map.len() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "input crossing map has {} entries, expected {expected_legs}",
            map.len()
        )));
    }
    let mut seen = vec![false; expected_legs];
    for entry in map {
        if entry.target_index >= expected_legs || entry.source_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "input crossing map references an out-of-range external leg",
            ));
        }
        if seen[entry.target_index] {
            return Err(RusticolError::invalid_argument(
                "input crossing map contains a duplicate target index",
            ));
        }
        seen[entry.target_index] = true;
    }
    if seen.iter().any(|value| !*value) {
        return Err(RusticolError::invalid_argument(
            "input crossing map does not cover every target index",
        ));
    }
    Ok(Some(map))
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn apply_input_crossing_map_generic<T>(
    batch: &[Vec<[T; 4]>],
    expected_legs: usize,
    input_crossing_map: Option<&[InputCrossingMapEntry]>,
) -> RusticolResult<Vec<Vec<[T; 4]>>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let Some(map) = validate_input_crossing_map(expected_legs, input_crossing_map)? else {
        return Ok(batch.to_vec());
    };
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        let mut mapped = vec![std::array::from_fn(|_| T::new_zero()); expected_legs];
        for entry in map {
            let source = &point[entry.source_index];
            for (target, source_component) in mapped[entry.target_index].iter_mut().zip(source) {
                *target = T::from(entry.sign) * source_component.clone();
            }
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

pub(crate) fn apply_lc_topology_label_permutation(
    batch: &[Vec<[f64; 4]>],
    expected_legs: usize,
    mapping: &[(usize, usize)],
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    let mut seen = vec![false; expected_legs];
    for (representative_index, sector_index) in mapping {
        if *representative_index >= expected_legs || *sector_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation references an out-of-range external leg",
            ));
        }
        if seen[*representative_index] {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation contains a duplicate representative label",
            ));
        }
        seen[*representative_index] = true;
    }
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        if point.len() != expected_legs {
            return Err(RusticolError::invalid_argument(format!(
                "LC topology replay point has {} external legs, expected {expected_legs}",
                point.len(),
            )));
        }
        let mut mapped = point.clone();
        for (representative_index, sector_index) in mapping {
            mapped[*representative_index] = point[*sector_index];
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

pub(crate) fn apply_lc_topology_label_permutations(
    batch: &[Vec<[f64; 4]>],
    expected_legs: usize,
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    let mut expanded_batch = Vec::with_capacity(batch.len() * mappings.len());
    for mapping in mappings {
        expanded_batch.extend(apply_lc_topology_label_permutation(
            batch,
            expected_legs,
            mapping,
        )?);
    }
    Ok(expanded_batch)
}

pub(crate) fn apply_lc_topology_label_permutations_from_view(
    batch: F64MomentumBatchView<'_>,
    expected_legs: usize,
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    if batch.external_count() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "LC topology replay input has {} external legs, expected {expected_legs}",
            batch.external_count()
        )));
    }
    let capacity = batch
        .point_count()
        .checked_mul(mappings.len())
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    let mut expanded_batch = Vec::with_capacity(capacity);
    for mapping in mappings {
        let mut seen = vec![false; expected_legs];
        for (representative_index, sector_index) in mapping {
            if *representative_index >= expected_legs || *sector_index >= expected_legs {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay label permutation references an out-of-range external leg",
                ));
            }
            if seen[*representative_index] {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay label permutation contains a duplicate representative label",
                ));
            }
            seen[*representative_index] = true;
        }
        for point_index in 0..batch.point_count() {
            let point = batch.point(point_index);
            let mut mapped = (0..expected_legs)
                .map(|external_index| {
                    point.momentum(external_index).ok_or_else(|| {
                        RusticolError::integrity(
                            "validated momentum view is missing an external leg during topology replay",
                        )
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            for (representative_index, sector_index) in mapping {
                mapped[*representative_index] = point.momentum(*sector_index).ok_or_else(|| {
                    RusticolError::integrity(
                        "validated momentum view is missing a permuted external leg",
                    )
                })?;
            }
            expanded_batch.push(mapped);
        }
    }
    Ok(expanded_batch)
}

/// Materialize one validated topology-replay permutation into reusable flat
/// row-major storage.
///
/// This is the totals-only Direct-Arena crossing lane: it avoids both the
/// nested per-point vectors and the mapping-by-batch expansion used by the
/// resolved compatibility path. The caller owns and reuses `output`.
pub(crate) fn apply_lc_topology_label_permutation_from_view_into_flat(
    batch: F64MomentumBatchView<'_>,
    expected_legs: usize,
    mapping: &[(usize, usize)],
    output: &mut Vec<f64>,
) -> RusticolResult<()> {
    if batch.external_count() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "LC topology replay input has {} external legs, expected {expected_legs}",
            batch.external_count()
        )));
    }
    for (mapping_index, (representative_index, sector_index)) in mapping.iter().enumerate() {
        if *representative_index >= expected_legs || *sector_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation references an out-of-range external leg",
            ));
        }
        if mapping[..mapping_index]
            .iter()
            .any(|(previous, _)| previous == representative_index)
        {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation contains a duplicate representative label",
            ));
        }
    }
    let values_per_point = expected_legs
        .checked_mul(4)
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    let scalar_count = batch
        .point_count()
        .checked_mul(values_per_point)
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    output.resize(scalar_count, 0.0);
    for point_index in 0..batch.point_count() {
        let point = batch.point(point_index);
        let row_start = point_index * values_per_point;
        for external_index in 0..expected_legs {
            let momentum = point.momentum(external_index).ok_or_else(|| {
                RusticolError::integrity(
                    "validated momentum view is missing an external leg during topology replay",
                )
            })?;
            output[row_start + external_index * 4..row_start + (external_index + 1) * 4]
                .copy_from_slice(&momentum);
        }
        for (representative_index, sector_index) in mapping {
            let momentum = point.momentum(*sector_index).ok_or_else(|| {
                RusticolError::integrity(
                    "validated momentum view is missing a permuted external leg",
                )
            })?;
            let start = row_start + representative_index * 4;
            output[start..start + 4].copy_from_slice(&momentum);
        }
    }
    Ok(())
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn apply_lc_topology_label_permutation_generic<T>(
    batch: &[Vec<[T; 4]>],
    expected_legs: usize,
    mapping: &[(usize, usize)],
) -> RusticolResult<Vec<Vec<[T; 4]>>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let mut seen = vec![false; expected_legs];
    for (representative_index, sector_index) in mapping {
        if *representative_index >= expected_legs || *sector_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation references an out-of-range external leg",
            ));
        }
        if seen[*representative_index] {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation contains a duplicate representative label",
            ));
        }
        seen[*representative_index] = true;
    }
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        if point.len() != expected_legs {
            return Err(RusticolError::invalid_argument(format!(
                "LC topology replay point has {} external legs, expected {expected_legs}",
                point.len(),
            )));
        }
        let mut mapped = point.clone();
        for (representative_index, sector_index) in mapping {
            mapped[*representative_index] = point[*sector_index].clone();
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn apply_lc_topology_label_permutations_generic<T>(
    batch: &[Vec<[T; 4]>],
    expected_legs: usize,
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<Vec<Vec<[T; 4]>>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let mut expanded_batch = Vec::with_capacity(batch.len() * mappings.len());
    for mapping in mappings {
        expanded_batch.extend(apply_lc_topology_label_permutation_generic(
            batch,
            expected_legs,
            mapping,
        )?);
    }
    Ok(expanded_batch)
}

pub(crate) fn replay_mappings_per_expanded_batch(n_points: usize) -> usize {
    if n_points == 0 {
        return 1;
    }
    (MAX_LC_TOPOLOGY_REPLAY_EXPANDED_POINTS / n_points).max(1)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn decimal_digits_to_bits(decimal_digits: u32) -> u32 {
    (decimal_digits as f64 * std::f64::consts::LOG2_10).ceil() as u32
}
