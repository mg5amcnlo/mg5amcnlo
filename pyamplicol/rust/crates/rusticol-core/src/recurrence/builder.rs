// SPDX-License-Identifier: 0BSD

//! Composite authentication and compact recurrence schedule construction.

use super::process::{ProcessSemanticTemplateReference, ValidatedRecurrenceProcessInput};
use super::template::{RecurrenceSemanticTemplateKind, ValidatedRecurrenceTemplateInput};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Process and prepared-model inputs authenticated as one semantic unit.
#[derive(Clone, Debug)]
pub struct AuthenticatedRecurrenceBuilderInput {
    process: ValidatedRecurrenceProcessInput,
    template: ValidatedRecurrenceTemplateInput,
}

impl AuthenticatedRecurrenceBuilderInput {
    pub fn new(
        process: ValidatedRecurrenceProcessInput,
        template: ValidatedRecurrenceTemplateInput,
    ) -> RusticolResult<Self> {
        let process_identity = process.semantic_identity();
        let template_summary = template.summary();
        if process_identity.model_catalog_digest() != template_summary.compiled_model_digest {
            return Err(invalid(format!(
                "recurrence process model-catalog digest {} does not match prepared model {}",
                process_identity.model_catalog_digest(),
                template_summary.compiled_model_digest,
            )));
        }
        if process_identity.prepared_catalog_digest() != template_summary.catalog_digest {
            return Err(invalid(format!(
                "recurrence process prepared-catalog digest {} does not match template catalog {}",
                process_identity.prepared_catalog_digest(),
                template_summary.catalog_digest,
            )));
        }

        let template_index = template.semantic_index()?;
        for reference in process.template_references() {
            authenticate_template_reference(reference, &template_index)?;
        }
        authenticate_singlet_closure_anchors(&process, &template)?;

        Ok(Self { process, template })
    }

    pub const fn process(&self) -> &ValidatedRecurrenceProcessInput {
        &self.process
    }

    pub const fn template(&self) -> &ValidatedRecurrenceTemplateInput {
        &self.template
    }

    pub fn into_parts(
        self,
    ) -> (
        ValidatedRecurrenceProcessInput,
        ValidatedRecurrenceTemplateInput,
    ) {
        (self.process, self.template)
    }

    pub fn build(&self) -> RusticolResult<super::RecurrenceProgram> {
        super::construct::build_recurrence_program(self)
    }

    pub fn build_with_progress(
        &self,
        progress: &mut dyn FnMut(super::RecurrenceBuildProgress) -> RusticolResult<()>,
    ) -> RusticolResult<super::RecurrenceProgram> {
        super::construct::build_recurrence_program_with_progress(self, progress)
    }

    /// Build while returning construction-only generation diagnostics.
    #[doc(hidden)]
    pub fn build_with_progress_and_telemetry(
        &self,
        progress: &mut dyn FnMut(super::RecurrenceBuildProgress) -> RusticolResult<()>,
    ) -> RusticolResult<(
        super::RecurrenceProgram,
        super::RecurrenceGenerationTelemetry,
    )> {
        super::construct::build_recurrence_program_with_progress_and_telemetry(self, progress)
    }
}

fn authenticate_singlet_closure_anchors(
    process: &ValidatedRecurrenceProcessInput,
    template: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<()> {
    let process_input = process.input();
    let template_input = template.input();
    let mut fermionic_source_slots = Vec::new();

    for leg in &process_input.external_legs {
        let declared_fermionic = match leg.is_fermionic {
            0 => false,
            1 => true,
            value => {
                return Err(invalid(format!(
                    "source slot {} has invalid fermionic marker {value}",
                    leg.source_slot
                )));
            }
        };
        let state_range = leg.source_state_range.as_usize_range(
            process_input.source_states.len(),
            "external source states during closure-anchor authentication",
        )?;
        let mut statistics = None;
        for source_state in &process_input.source_states[state_range] {
            let state = template_input
                .current_states
                .get(source_state.current_state_template_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "source slot {} references absent current-state template {}",
                        leg.source_slot, source_state.current_state_template_id
                    ))
                })?;
            match statistics {
                None => statistics = Some(state.statistics),
                Some(previous) if previous == state.statistics => {}
                Some(previous) => {
                    return Err(invalid(format!(
                        "source slot {} mixes particle statistics {previous} and {}",
                        leg.source_slot, state.statistics
                    )));
                }
            }
        }
        let template_fermionic = statistics == Some(1);
        if template_fermionic != declared_fermionic {
            return Err(invalid(format!(
                "source slot {} fermionic marker does not match its authenticated current-state templates",
                leg.source_slot
            )));
        }
        if declared_fermionic {
            fermionic_source_slots.push(leg.source_slot);
        }
    }

    let expected_singlet_anchor = fermionic_source_slots.first().copied().unwrap_or(0);
    for sector in &process_input.physical_lc_sectors {
        let word = process_u32_sequence(
            process_input,
            sector.word_sequence_id,
            "physical LC color word during closure-anchor authentication",
        )?;
        if word.is_empty() && sector.closure_source_slot != expected_singlet_anchor {
            return Err(invalid(format!(
                "all-singlet LC sector {} uses closure source slot {}, expected first fermionic source slot (or first source) {expected_singlet_anchor}",
                sector.sector_id, sector.closure_source_slot
            )));
        }
    }
    Ok(())
}

fn process_u32_sequence<'a>(
    input: &'a super::process::OwnedRecurrenceProcessInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let range = input
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .copied()
        .ok_or_else(|| invalid(format!("{label} references absent sequence {sequence_id}")))?;
    let range = range.as_usize_range(input.u32_sequence_values.len(), label)?;
    Ok(&input.u32_sequence_values[range])
}

fn authenticate_template_reference(
    reference: &ProcessSemanticTemplateReference,
    template_index: &super::template::RecurrenceTemplateSemanticIndex,
) -> RusticolResult<()> {
    let kind = RecurrenceSemanticTemplateKind::parse(reference.typed_id.kind.as_str())?;
    let Some(identity) = template_index.record(kind, reference.typed_id.template_id) else {
        return Err(invalid(format!(
            "recurrence process references absent prepared {} template {}",
            kind.as_str(),
            reference.typed_id.template_id,
        )));
    };
    if identity.semantic_digest != reference.semantic_digest {
        return Err(invalid(format!(
            "recurrence process {} template {} has semantic digest {}, expected {}",
            kind.as_str(),
            reference.typed_id.template_id,
            reference.semantic_digest,
            identity.semantic_digest,
        )));
    }
    if identity.prepared_kernel_id != reference.prepared_kernel_id {
        return Err(invalid(format!(
            "recurrence process {} template {} has prepared kernel {:?}, expected {:?}",
            kind.as_str(),
            reference.typed_id.template_id,
            reference.prepared_kernel_id,
            identity.prepared_kernel_id,
        )));
    }
    Ok(())
}
