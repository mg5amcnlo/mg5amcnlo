// SPDX-License-Identifier: 0BSD

use super::*;

#[derive(Clone, Debug)]
pub(super) struct RuntimeSourceState {
    pub(super) state: GenericSourceStateIrManifest,
    pub(super) factor: Complex<f64>,
}

#[derive(Clone, Copy)]
enum ResolvedSourceKind {
    Scalar,
    Weyl { antiparticle: bool },
    Dirac { antiparticle: bool, mass: f64 },
    MasslessVector,
    MassiveVector { mass: f64 },
    Spin2 { mass: f64 },
}

/// Resolve the already-generated SourceIR once for a source's point loop.
/// Masses are read at this boundary, so parameter changes apply to the next
/// tile without a new process plan or a second set of wavefunction formulas.
pub(super) struct ResolvedSourceWavefunction<'a> {
    source: &'a GenericSourceRecordManifest,
    state: &'a GenericSourceStateIrManifest,
    external_index: usize,
    kind: ResolvedSourceKind,
}

impl<'a> ResolvedSourceWavefunction<'a> {
    #[inline]
    pub(super) fn new(
        source: &'a GenericSourceRecordManifest,
        state: &'a GenericSourceStateIrManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        output_len: usize,
    ) -> RusticolResult<Self> {
        if source.source_kind != "external-wavefunction" {
            return Err(RusticolError::invalid_argument(format!(
                "generic source kind {:?} is not implemented",
                source.source_kind
            )));
        }
        let external_index = source.leg_label.checked_sub(1).ok_or_else(|| {
            RusticolError::invalid_argument("generic source leg labels are one-based")
        })?;
        if external_index >= external_count {
            return Err(RusticolError::invalid_argument(format!(
                "generic source {} refers to unknown external label {}",
                source.source_id, source.leg_label
            )));
        }
        let family = source.source_ir.wavefunction_family;
        let dimension = source.source_ir.component_dimension;
        if output_len != dimension {
            return Err(RusticolError::invalid_argument(format!(
                "generic source {} expected dimension {} but slot has length {}",
                source.source_id, dimension, output_len
            )));
        }
        let mass = || {
            particle_mass_from_map(
                particle_masses,
                source.source_ir.identity.pdg_label,
                source.source_ir.identity.anti_pdg_label,
            )
        };
        let kind = match (family, dimension) {
            (GenericWavefunctionFamilyManifest::Scalar, 1) => ResolvedSourceKind::Scalar,
            (GenericWavefunctionFamilyManifest::Fermion, 2) => ResolvedSourceKind::Weyl {
                antiparticle: source_is_antiparticle(source)?,
            },
            (GenericWavefunctionFamilyManifest::Fermion, 4) => ResolvedSourceKind::Dirac {
                antiparticle: source_is_antiparticle(source)?,
                mass: mass(),
            },
            (GenericWavefunctionFamilyManifest::Vector, 4) => match mass() {
                0.0 => ResolvedSourceKind::MasslessVector,
                mass => ResolvedSourceKind::MassiveVector { mass },
            },
            (GenericWavefunctionFamilyManifest::Spin2, 16) => {
                ResolvedSourceKind::Spin2 { mass: mass() }
            }
            _ => return Err(unsupported_source_wavefunction(source)),
        };
        Ok(Self {
            source,
            state,
            external_index,
            kind,
        })
    }

    #[inline]
    pub(super) fn write<P>(&self, point: &P, out: &mut [Complex<f64>]) -> RusticolResult<()>
    where
        P: F64MomentumPoint + ?Sized,
    {
        debug_assert_eq!(out.len(), self.source.source_ir.component_dimension);
        let input = point.momentum(self.external_index).ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "generic source {} refers to absent external label {}",
                self.source.source_id, self.source.leg_label
            ))
        })?;
        let momentum = match self.source.applied_crossing.momentum_transform {
            GenericMomentumTransformManifest::Identity => input,
            GenericMomentumTransformManifest::NegateFourMomentum => negate(input),
        };
        match self.kind {
            ResolvedSourceKind::Scalar => out[0] = c64(1.0, 0.0),
            ResolvedSourceKind::Weyl { antiparticle } => {
                let wave = if antiparticle {
                    ext_antiquark_weyl_array(momentum, self.state.helicity, self.state.chirality)
                } else {
                    ext_quark_weyl_array(momentum, self.state.helicity, self.state.chirality)
                };
                out.copy_from_slice(&wave);
            }
            ResolvedSourceKind::Dirac { antiparticle, mass } => {
                let wave = if antiparticle {
                    ext_antiquark_dirac_massive(momentum, self.state.helicity, mass)
                } else {
                    ext_quark_dirac_massive(momentum, self.state.helicity, mass)
                };
                out.copy_from_slice(&wave);
            }
            ResolvedSourceKind::MasslessVector => {
                out.copy_from_slice(&ext_gluon(momentum, self.state.helicity));
            }
            ResolvedSourceKind::MassiveVector { mass } => {
                out.copy_from_slice(&ext_massive_vector(momentum, self.state.helicity, mass));
            }
            ResolvedSourceKind::Spin2 { mass } => {
                out.copy_from_slice(&ext_spin2(momentum, self.state.helicity, mass)?);
            }
        }
        // Do not fold this phase into a runtime-state factor: their existing
        // two complex multiplications have a defined floating-point order.
        apply_source_phase_f64(&self.source.applied_crossing, out);
        Ok(())
    }
}

impl ExecutionRuntime {
    pub(super) fn default_runtime_source_state(
        source: &GenericSourceRecordManifest,
    ) -> RusticolResult<GenericSourceStateIrManifest> {
        default_source_state(source)
    }

    pub(super) fn fill_sources_row_with_states<P>(
        sources: &[GenericSourceRecordManifest],
        source_states: &[RuntimeSourceState],
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        row: &mut [Complex<f64>],
        point: &P,
    ) -> RusticolResult<()>
    where
        P: F64MomentumPoint + ?Sized,
    {
        if source_states.len() != sources.len() {
            return Err(RusticolError::invalid_argument(format!(
                "runtime source-state count {} does not match source count {}",
                source_states.len(),
                sources.len()
            )));
        }
        for (source, runtime_state) in sources.iter().zip(source_states) {
            let start = source.value_slot.component_start;
            let stop = source.value_slot.component_stop;
            if stop > row.len() || stop < start {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} has an invalid value-slot range",
                    source.source_id
                )));
            }
            if runtime_state.factor == c64(0.0, 0.0) {
                row[start..stop].fill(c64(0.0, 0.0));
                continue;
            }
            Self::write_source_wavefunction_with_state(
                source,
                &runtime_state.state,
                external_count,
                particle_masses,
                point,
                &mut row[start..stop],
            )?;
            if runtime_state.factor != c64(1.0, 0.0) {
                for component in &mut row[start..stop] {
                    *component *= runtime_state.factor;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_sources_row_generic_with_states<T>(
        sources: &[GenericSourceRecordManifest],
        source_states: &[RuntimeSourceState],
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        row: &mut [Complex<T>],
        point: &[[T; 4]],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if source_states.len() != sources.len() {
            return Err(RusticolError::invalid_argument(format!(
                "runtime source-state count {} does not match source count {}",
                source_states.len(),
                sources.len()
            )));
        }
        for (source, runtime_state) in sources.iter().zip(source_states) {
            let start = source.value_slot.component_start;
            let stop = source.value_slot.component_stop;
            if stop > row.len() || stop < start {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} has an invalid value-slot range",
                    source.source_id
                )));
            }
            if runtime_state.factor == c64(0.0, 0.0) {
                row[start..stop].fill(complex_zero::<T>());
                continue;
            }
            Self::write_source_wavefunction_generic_with_state(
                source,
                &runtime_state.state,
                external_count,
                particle_masses,
                point,
                &mut row[start..stop],
            )?;
            if runtime_state.factor != c64(1.0, 0.0) {
                let factor = c_generic(
                    T::from(runtime_state.factor.re),
                    T::from(runtime_state.factor.im),
                );
                for component in &mut row[start..stop] {
                    *component *= factor.clone();
                }
            }
        }
        Ok(())
    }

    pub(super) fn runtime_source_state(
        source: &GenericSourceRecordManifest,
        declared_state_index: usize,
        factor: Complex<f64>,
    ) -> RusticolResult<RuntimeSourceState> {
        let declared = source
            .source_ir
            .states
            .get(declared_state_index)
            .ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "generic source {} has no declared state {}",
                    source.source_id, declared_state_index
                ))
            })?;
        let state = declared
            .transformed(&source.applied_crossing)
            .map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "generic source {} runtime state is invalid: {error}",
                    source.source_id
                ))
            })?;
        Ok(RuntimeSourceState { state, factor })
    }

    pub(super) fn runtime_source_state_index_for_helicity(
        source: &GenericSourceRecordManifest,
        helicity: i32,
    ) -> RusticolResult<Option<usize>> {
        let mut matching =
            source
                .source_ir
                .states
                .iter()
                .enumerate()
                .filter_map(|(index, state)| {
                    state
                        .transformed(&source.applied_crossing)
                        .ok()
                        .filter(|state| state.helicity == helicity)
                        .map(|_| index)
                });
        let result = matching.next();
        if matching.next().is_some() {
            return Err(RusticolError::artifact(format!(
                "generic source {} declares multiple runtime states for helicity {}",
                source.source_id, helicity
            )));
        }
        Ok(result)
    }

    pub(super) fn inactive_runtime_source_state(
        source: &GenericSourceRecordManifest,
    ) -> RusticolResult<RuntimeSourceState> {
        Ok(RuntimeSourceState {
            state: default_source_state(source)?,
            factor: c64(0.0, 0.0),
        })
    }

    pub(super) fn fill_sources_row<P>(
        sources: &[GenericSourceRecordManifest],
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        row: &mut [Complex<f64>],
        point: &P,
    ) -> RusticolResult<()>
    where
        P: F64MomentumPoint + ?Sized,
    {
        for source in sources {
            let start = source.value_slot.component_start;
            let stop = source.value_slot.component_stop;
            if stop > row.len() || stop < start {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} has an invalid value-slot range",
                    source.source_id
                )));
            }
            Self::write_source_wavefunction(
                source,
                external_count,
                particle_masses,
                point,
                &mut row[start..stop],
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_sources_row_generic<T>(
        sources: &[GenericSourceRecordManifest],
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        row: &mut [Complex<T>],
        point: &[[T; 4]],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        for source in sources {
            let start = source.value_slot.component_start;
            let stop = source.value_slot.component_stop;
            if stop > row.len() || stop < start {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} has an invalid value-slot range",
                    source.source_id
                )));
            }
            Self::write_source_wavefunction_generic(
                source,
                external_count,
                particle_masses,
                point,
                &mut row[start..stop],
            )?;
        }
        Ok(())
    }

    pub(super) fn fill_momenta_row<P>(
        momentum_slots: &[GenericMomentumSlotManifest],
        value_parameter_count: usize,
        external_count: usize,
        external_is_initial: &[bool],
        row: &mut [Complex<f64>],
        point: &P,
    ) -> RusticolResult<()>
    where
        P: F64MomentumPoint + ?Sized,
    {
        for slot in momentum_slots {
            let start = value_parameter_count + slot.component_start;
            let stop = value_parameter_count + slot.component_stop;
            if stop > row.len() || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum = [0.0; 4];
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= external_count || index >= external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                let sign = if external_is_initial[index] {
                    -1.0
                } else {
                    1.0
                };
                let point_momentum = point.momentum(index).ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to absent external label {}",
                        slot.momentum_slot_id, label
                    ))
                })?;
                for (momentum_component, point_component) in momentum.iter_mut().zip(point_momentum)
                {
                    *momentum_component += sign * point_component;
                }
            }
            for (output, component) in row[start..stop].iter_mut().zip(momentum) {
                *output = c64(component, 0.0);
            }
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_momenta_row_generic<T>(
        momentum_slots: &[GenericMomentumSlotManifest],
        value_parameter_count: usize,
        external_count: usize,
        external_is_initial: &[bool],
        row: &mut [Complex<T>],
        point: &[[T; 4]],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        for slot in momentum_slots {
            let start = value_parameter_count + slot.component_start;
            let stop = value_parameter_count + slot.component_stop;
            if stop > row.len() || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum: [T; 4] = std::array::from_fn(|_| T::new_zero());
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= external_count || index >= external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                for (momentum_component, point_component) in momentum.iter_mut().zip(&point[index])
                {
                    if external_is_initial[index] {
                        *momentum_component -= point_component.clone();
                    } else {
                        *momentum_component += point_component.clone();
                    }
                }
            }
            for (output, component) in row[start..stop].iter_mut().zip(momentum) {
                *output = c_generic(component, T::new_zero());
            }
        }
        Ok(())
    }

    pub(super) fn fill_model_parameters_row(
        model_parameter_start: usize,
        model_parameter_values: &[f64],
        row: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        let stop = model_parameter_start + model_parameter_values.len();
        if stop > row.len() {
            return Err(RusticolError::invalid_argument(
                "generic model-parameter block exceeds runtime row length",
            ));
        }
        for (index, value) in model_parameter_values.iter().enumerate() {
            row[model_parameter_start + index] = c64(*value, 0.0);
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_model_parameters_row_generic<T>(
        model_parameter_start: usize,
        model_parameter_values: &[f64],
        row: &mut [Complex<T>],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let stop = model_parameter_start + model_parameter_values.len();
        if stop > row.len() {
            return Err(RusticolError::invalid_argument(
                "generic model-parameter block exceeds runtime row length",
            ));
        }
        for (index, value) in model_parameter_values.iter().enumerate() {
            row[model_parameter_start + index] = c_generic(T::from(*value), T::new_zero());
        }
        Ok(())
    }

    pub(super) fn write_source_wavefunction<P>(
        source: &GenericSourceRecordManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &P,
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()>
    where
        P: F64MomentumPoint + ?Sized,
    {
        let state = default_source_state(source)?;
        Self::write_source_wavefunction_with_state(
            source,
            &state,
            external_count,
            particle_masses,
            point,
            out,
        )
    }

    pub(super) fn write_source_wavefunction_with_state<P>(
        source: &GenericSourceRecordManifest,
        state: &GenericSourceStateIrManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &P,
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()>
    where
        P: F64MomentumPoint + ?Sized,
    {
        ResolvedSourceWavefunction::new(source, state, external_count, particle_masses, out.len())?
            .write(point, out)
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn write_source_wavefunction_generic<T>(
        source: &GenericSourceRecordManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &[[T; 4]],
        out: &mut [Complex<T>],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let state = default_source_state(source)?;
        Self::write_source_wavefunction_generic_with_state(
            source,
            &state,
            external_count,
            particle_masses,
            point,
            out,
        )
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn write_source_wavefunction_generic_with_state<T>(
        source: &GenericSourceRecordManifest,
        state: &GenericSourceStateIrManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &[[T; 4]],
        out: &mut [Complex<T>],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        Self::write_source_wavefunction_generic_unphased(
            source,
            state,
            external_count,
            particle_masses,
            point,
            out,
        )?;
        apply_source_phase_generic(&source.applied_crossing, out);
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    fn write_source_wavefunction_generic_unphased<T>(
        source: &GenericSourceRecordManifest,
        state: &GenericSourceStateIrManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &[[T; 4]],
        out: &mut [Complex<T>],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if source.source_kind != "external-wavefunction" {
            return Err(RusticolError::invalid_argument(format!(
                "generic source kind {:?} is not implemented",
                source.source_kind
            )));
        }
        let index = source.leg_label.checked_sub(1).ok_or_else(|| {
            RusticolError::invalid_argument("generic source leg labels are one-based")
        })?;
        if index >= external_count {
            return Err(RusticolError::invalid_argument(format!(
                "generic source {} refers to unknown external label {}",
                source.source_id, source.leg_label
            )));
        }
        let source_ir = &source.source_ir;
        let identity = &source_ir.identity;
        let family = source_ir.wavefunction_family;
        let dimension = source_ir.component_dimension;
        let momentum = match source.applied_crossing.momentum_transform {
            GenericMomentumTransformManifest::Identity => point[index].clone(),
            GenericMomentumTransformManifest::NegateFourMomentum => negate_generic(&point[index]),
        };
        if dimension == 1 && family == GenericWavefunctionFamilyManifest::Scalar {
            if out.len() != 1 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 1 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            out[0] = c_generic(T::from(1.0), T::new_zero());
            return Ok(());
        }
        if dimension == 2 && family == GenericWavefunctionFamilyManifest::Fermion {
            if out.len() != 2 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 2 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let chirality = state.chirality;
            let wave = if source_is_antiparticle(source)? {
                ext_antiquark_weyl_generic(&momentum, state.helicity, chirality)
            } else {
                ext_quark_weyl_generic(&momentum, state.helicity, chirality)
            };
            out.clone_from_slice(&wave);
            return Ok(());
        }
        if dimension == 4 {
            if out.len() != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 4 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let wave = if family == GenericWavefunctionFamilyManifest::Fermion {
                let mass = particle_mass_from_map(
                    particle_masses,
                    identity.pdg_label,
                    identity.anti_pdg_label,
                );
                if mass != 0.0 {
                    return Err(RusticolError::invalid_argument(
                        "high-precision generic massive fermion sources are not implemented",
                    ));
                }
                if source_is_antiparticle(source)? {
                    ext_antiquark_dirac_generic(&momentum, state.helicity)
                } else {
                    ext_quark_dirac_generic(&momentum, state.helicity)
                }
            } else if family == GenericWavefunctionFamilyManifest::Vector {
                let mass = particle_mass_from_map(
                    particle_masses,
                    identity.pdg_label,
                    identity.anti_pdg_label,
                );
                if mass == 0.0 {
                    ext_gluon_generic(&momentum, state.helicity)
                } else {
                    ext_massive_vector_generic(&momentum, state.helicity, T::from(mass))
                }
            } else {
                return Err(unsupported_source_wavefunction(source));
            };
            out.clone_from_slice(&wave);
            return Ok(());
        }
        if dimension == 16 && family == GenericWavefunctionFamilyManifest::Spin2 {
            if out.len() != 16 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 16 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let wave = ext_spin2_generic(
                &momentum,
                state.helicity,
                T::from(particle_mass_from_map(
                    particle_masses,
                    identity.pdg_label,
                    identity.anti_pdg_label,
                )),
            )?;
            out.clone_from_slice(&wave);
            return Ok(());
        }
        Err(RusticolError::invalid_argument(format!(
            "generic source kind {:?} with dimension {} is not implemented",
            family.as_str(),
            dimension
        )))
    }
}

fn default_source_state(
    source: &GenericSourceRecordManifest,
) -> RusticolResult<GenericSourceStateIrManifest> {
    source
        .source_ir
        .states
        .iter()
        .filter_map(|state| state.transformed(&source.applied_crossing).ok())
        .find(|state| {
            state.helicity == source.source_helicity && state.chirality == source.chirality
        })
        .ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "generic source {} default state is absent from its SourceIR",
                source.source_id
            ))
        })
}

fn apply_source_phase_f64(crossing: &GenericCrossingIrManifest, out: &mut [Complex<f64>]) {
    if crossing.phase == [1.0, 0.0] {
        return;
    }
    let phase = c64(crossing.phase[0], crossing.phase[1]);
    for component in out {
        *component *= phase;
    }
}

#[cfg(feature = "symbolica-runtime")]
fn apply_source_phase_generic<T>(crossing: &GenericCrossingIrManifest, out: &mut [Complex<T>])
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    if crossing.phase == [1.0, 0.0] {
        return;
    }
    let phase = c_generic(T::from(crossing.phase[0]), T::from(crossing.phase[1]));
    for component in out {
        *component *= &phase;
    }
}

fn source_is_antiparticle(source: &GenericSourceRecordManifest) -> RusticolResult<bool> {
    match source.source_ir.identity.orientation {
        GenericSourceOrientationManifest::Particle => Ok(false),
        GenericSourceOrientationManifest::Antiparticle => Ok(true),
        GenericSourceOrientationManifest::SelfConjugate => {
            Err(RusticolError::invalid_argument(format!(
                "generic source {} uses an unsupported self-conjugate fermion wavefunction",
                source.source_id
            )))
        }
    }
}

fn unsupported_source_wavefunction(source: &GenericSourceRecordManifest) -> RusticolError {
    RusticolError::invalid_argument(format!(
        "generic source kind {:?} with dimension {} is not implemented",
        source.source_ir.wavefunction_family.as_str(),
        source.source_ir.component_dimension
    ))
}

fn particle_mass_from_map(
    particle_masses: &BTreeMap<i32, f64>,
    particle_id: i32,
    anti_particle_id: i32,
) -> f64 {
    particle_masses
        .get(&particle_id)
        .copied()
        .or_else(|| particle_masses.get(&anti_particle_id).copied())
        .unwrap_or(0.0)
}
