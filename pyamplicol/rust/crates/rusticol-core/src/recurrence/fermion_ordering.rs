// SPDX-License-Identifier: 0BSD

//! Cold construction-time exterior-algebra signs for external fermions.

use super::ExactComplexRational;
use super::template::{CurrentOrientation, CurrentStateRow, ParticleStatistics};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[derive(Clone, Debug)]
pub(super) struct FermionOrderingContext {
    source_requires_exterior_sign: Box<[bool]>,
    active: bool,
}

impl FermionOrderingContext {
    pub(super) fn new(source_requires_exterior_sign: Vec<bool>) -> Self {
        let active = source_requires_exterior_sign
            .iter()
            .copied()
            .any(|value| value);
        Self {
            source_requires_exterior_sign: source_requires_exterior_sign.into_boxed_slice(),
            active,
        }
    }

    fn source_requires_exterior_sign(&self, source_slot: u32) -> RusticolResult<bool> {
        self.source_requires_exterior_sign
            .get(source_slot as usize)
            .copied()
            .ok_or_else(|| invalid("fermion-ordering source slot is absent"))
    }
}

pub(super) fn authenticated_source_requires_exterior_sign(
    is_fermionic: bool,
    is_authenticated_colored_endpoint: bool,
    color_representations: impl IntoIterator<Item = i32>,
) -> RusticolResult<bool> {
    if !is_fermionic {
        if is_authenticated_colored_endpoint {
            return Err(invalid(
                "fermion-ordering endpoint authentication selected a bosonic source",
            ));
        }
        return Ok(false);
    }
    let mut expected_ownership = None;
    for color_representation in color_representations {
        let ownership = match color_representation {
            1 => true,
            3 | -3 => false,
            value => {
                return Err(invalid(format!(
                    "external-fermion ordering does not support fermion color representation \
                     {value}"
                )));
            }
        };
        if expected_ownership.is_some_and(|expected| expected != ownership) {
            return Err(invalid(
                "external-fermion ordering source states have mixed color-sign ownership",
            ));
        }
        expected_ownership = Some(ownership);
    }
    let requires_exterior_sign = expected_ownership.ok_or_else(|| {
        invalid("external-fermion ordering fermion source has no authenticated state")
    })?;
    if is_authenticated_colored_endpoint == requires_exterior_sign {
        return Err(invalid(if is_authenticated_colored_endpoint {
            "fermion-ordering authenticated colored endpoint has singlet source states"
        } else {
            "fermion-ordering fundamental source lacks authenticated endpoint ownership"
        }));
    }
    Ok(requires_exterior_sign)
}

fn external_support_sign(
    left_support: &[u32],
    right_support: &[u32],
    context: &FermionOrderingContext,
) -> RusticolResult<i32> {
    let mut odd = false;
    for left in left_support.iter().copied() {
        if !context.source_requires_exterior_sign(left)? {
            continue;
        }
        for right in right_support.iter().copied() {
            if left > right && context.source_requires_exterior_sign(right)? {
                odd = !odd;
            }
        }
    }
    Ok(if odd { -1 } else { 1 })
}

fn input_orientation_sign(left: CurrentStateRow, right: CurrentStateRow) -> RusticolResult<i32> {
    let left_statistics = ParticleStatistics::try_from(left.statistics)?;
    let right_statistics = ParticleStatistics::try_from(right.statistics)?;
    if left_statistics != ParticleStatistics::Fermion
        || right_statistics != ParticleStatistics::Fermion
    {
        return Ok(1);
    }
    let left_requires_exterior_sign = match left.color_representation {
        1 => true,
        3 | -3 => false,
        value => {
            return Err(invalid(format!(
                "external-fermion ordering does not support fermion color representation {value}"
            )));
        }
    };
    let right_requires_exterior_sign = match right.color_representation {
        1 => true,
        3 | -3 => false,
        value => {
            return Err(invalid(format!(
                "external-fermion ordering does not support fermion color representation {value}"
            )));
        }
    };
    if left_requires_exterior_sign != right_requires_exterior_sign {
        return Err(invalid(
            "external-fermion ordering encountered mixed color-sign ownership",
        ));
    }
    if !left_requires_exterior_sign {
        return Ok(1);
    }
    let left_orientation = CurrentOrientation::try_from(left.orientation)?;
    let right_orientation = CurrentOrientation::try_from(right.orientation)?;
    match (left_orientation, right_orientation) {
        (CurrentOrientation::Antiparticle, CurrentOrientation::Particle) => Ok(1),
        (CurrentOrientation::Particle, CurrentOrientation::Antiparticle) => Ok(-1),
        (CurrentOrientation::SelfConjugate, _) | (_, CurrentOrientation::SelfConjugate) => Err(
            invalid("external-fermion ordering does not support self-conjugate fermion currents"),
        ),
        _ => Err(invalid(
            "external-fermion ordering requires opposite Dirac orientations",
        )),
    }
}

/// Return the exact sign for one construction-ordered binary current merge.
pub(super) fn fermion_ordering_factor(
    current_states: &[CurrentStateRow],
    parent_state_ids: [u32; 2],
    parent_supports: [&[u32]; 2],
    context: &FermionOrderingContext,
) -> RusticolResult<ExactComplexRational> {
    if !context.active {
        return Ok(ExactComplexRational::ONE);
    }
    let left_state = current_states
        .get(parent_state_ids[0] as usize)
        .copied()
        .ok_or_else(|| invalid("fermion-ordering left current state is absent"))?;
    let right_state = current_states
        .get(parent_state_ids[1] as usize)
        .copied()
        .ok_or_else(|| invalid("fermion-ordering right current state is absent"))?;
    let sign = external_support_sign(parent_supports[0], parent_supports[1], context)?
        * input_orientation_sign(left_state, right_state)?;
    match sign {
        1 => Ok(ExactComplexRational::ONE),
        -1 => ExactComplexRational::ONE.checked_neg(),
        _ => unreachable!("the product of two parities is a parity"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        id: u32,
        statistics: ParticleStatistics,
        orientation: CurrentOrientation,
        color_representation: i32,
    ) -> CurrentStateRow {
        CurrentStateRow {
            id,
            template_string_id: id,
            particle_id: id as i32,
            anti_particle_id: id as i32,
            species_string_id: id,
            orientation: orientation as u8,
            statistics: statistics as u8,
            color_representation,
            basis_string_id: id,
            tensor_ordering_sequence_id: id,
            dimension: 1,
            chirality: 0,
            lc_color_shape_string_id: id,
            auxiliary_kind_string_id: u32::MAX,
            mass_parameter_id: u32::MAX,
            width_parameter_id: u32::MAX,
            semantic_digest_id: id,
        }
    }

    #[test]
    fn direct_and_exchanged_dirac_pairings_have_opposite_cached_products() {
        let anti = state(
            0,
            ParticleStatistics::Fermion,
            CurrentOrientation::Antiparticle,
            1,
        );
        let particle = state(
            1,
            ParticleStatistics::Fermion,
            CurrentOrientation::Particle,
            1,
        );
        let boson = state(
            2,
            ParticleStatistics::Boson,
            CurrentOrientation::SelfConjugate,
            1,
        );
        let states = [anti, particle, boson];
        let context = FermionOrderingContext::new(vec![true; 4]);

        let direct_first =
            fermion_ordering_factor(&states, [0, 1], [&[0], &[1]], &context).unwrap();
        let direct_attach =
            fermion_ordering_factor(&states, [2, 0], [&[0, 1], &[2]], &context).unwrap();
        let exchange_first =
            fermion_ordering_factor(&states, [1, 0], [&[1], &[2]], &context).unwrap();
        let exchange_attach =
            fermion_ordering_factor(&states, [0, 2], [&[0], &[1, 2]], &context).unwrap();

        assert_eq!(
            direct_first.checked_mul(direct_attach).unwrap(),
            ExactComplexRational::ONE
        );
        assert_eq!(
            exchange_first.checked_mul(exchange_attach).unwrap(),
            ExactComplexRational::ONE.checked_neg().unwrap()
        );
    }

    #[test]
    fn bosonic_sources_do_not_participate_in_support_parity() {
        let boson = state(
            0,
            ParticleStatistics::Boson,
            CurrentOrientation::SelfConjugate,
            1,
        );
        let context = FermionOrderingContext::new(vec![true, true, true, false]);
        let factor = fermion_ordering_factor(&[boson], [0, 0], [&[3], &[0]], &context).unwrap();
        assert_eq!(factor, ExactComplexRational::ONE);
    }

    #[test]
    fn authenticated_source_filter_separates_singlets_from_colored_endpoints() {
        assert!(authenticated_source_requires_exterior_sign(true, false, [1, 1]).unwrap());
        assert!(!authenticated_source_requires_exterior_sign(true, true, [3, -3]).unwrap());
        assert!(!authenticated_source_requires_exterior_sign(false, false, [8]).unwrap());
    }

    #[test]
    fn authenticated_source_filter_fails_closed_on_mixed_or_unauthenticated_color() {
        for error in [
            authenticated_source_requires_exterior_sign(true, false, [1, 3]).unwrap_err(),
            authenticated_source_requires_exterior_sign(true, false, [3, 3]).unwrap_err(),
            authenticated_source_requires_exterior_sign(true, true, [1, 1]).unwrap_err(),
            authenticated_source_requires_exterior_sign(true, false, [8, 8]).unwrap_err(),
        ] {
            assert!(
                error.to_string().contains("fermion-ordering")
                    || error.to_string().contains("external-fermion ordering")
            );
        }
    }

    #[test]
    fn colored_open_line_orientation_is_not_counted_twice() {
        let anti = state(
            0,
            ParticleStatistics::Fermion,
            CurrentOrientation::Antiparticle,
            -3,
        );
        let particle = state(
            1,
            ParticleStatistics::Fermion,
            CurrentOrientation::Particle,
            3,
        );
        let context = FermionOrderingContext::new(vec![false, false]);
        let factor =
            fermion_ordering_factor(&[anti, particle], [1, 0], [&[1], &[0]], &context).unwrap();
        assert_eq!(factor, ExactComplexRational::ONE);
    }

    #[test]
    fn mixed_current_ownership_fails_during_construction() {
        let singlet = state(
            0,
            ParticleStatistics::Fermion,
            CurrentOrientation::Antiparticle,
            1,
        );
        let colored = state(
            1,
            ParticleStatistics::Fermion,
            CurrentOrientation::Particle,
            3,
        );
        let context = FermionOrderingContext::new(vec![true, false]);
        let error = fermion_ordering_factor(&[singlet, colored], [0, 1], [&[0], &[1]], &context)
            .unwrap_err();
        assert!(error.to_string().contains("mixed color-sign ownership"));
    }

    #[test]
    fn same_dirac_orientation_fails_during_construction() {
        for orientation in [
            CurrentOrientation::Particle,
            CurrentOrientation::Antiparticle,
        ] {
            let state = state(0, ParticleStatistics::Fermion, orientation, 1);
            let context = FermionOrderingContext::new(vec![true, true]);
            let error =
                fermion_ordering_factor(&[state], [0, 0], [&[0], &[1]], &context).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("requires opposite Dirac orientations")
            );
        }
    }
}
