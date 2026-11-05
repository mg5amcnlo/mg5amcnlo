// SPDX-License-Identifier: 0BSD

use std::fmt;

use super::ExactComplexRational;
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// A nonzero SHA-256 semantic digest.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, bincode::Encode)]
pub struct SemanticDigest([u8; 32]);

impl<Context> bincode::Decode<Context> for SemanticDigest {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Self::new(bincode::Decode::decode(decoder)?)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))
    }
}

bincode::impl_borrow_decode!(SemanticDigest);

impl SemanticDigest {
    pub fn new(bytes: [u8; 32]) -> RusticolResult<Self> {
        if bytes == [0; 32] {
            return Err(invalid("semantic digest must not be all zero"));
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for SemanticDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Workload-specific recurrence schedule strategy.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum RecurrenceStrategy {
    TopologyReplay = 0,
    AllFlowUnion = 1,
    ContractedColorUnion = 2,
}

impl RecurrenceStrategy {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopologyReplay => "topology-replay",
            Self::AllFlowUnion => "all-flow-union",
            Self::ContractedColorUnion => "contracted-color-union",
        }
    }

    pub const fn uses_runtime_source_dispatch(self) -> bool {
        matches!(self, Self::AllFlowUnion)
    }

    pub const fn retains_local_source_ancestry(self) -> bool {
        !self.uses_runtime_source_dispatch()
    }

    pub const fn materializes_all_color_sectors(self) -> bool {
        !matches!(self, Self::TopologyReplay)
    }

    pub const fn uses_topology_replay_targets(self) -> bool {
        matches!(self, Self::TopologyReplay | Self::ContractedColorUnion)
    }
}

impl TryFrom<u32> for RecurrenceStrategy {
    type Error = RusticolError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::TopologyReplay),
            1 => Ok(Self::AllFlowUnion),
            2 => Ok(Self::ContractedColorUnion),
            _ => Err(invalid(format!(
                "unknown recurrence strategy discriminant {value}"
            ))),
        }
    }
}

impl fmt::Display for RecurrenceStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Semantic kind of recurrence state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum RecurrenceNodeKind {
    Source = 0,
    Current = 1,
}

/// Identifier issued only by the dynamic LC color-state interner.
///
/// This newtype prevents physical sector IDs from accidentally becoming
/// recurrence current identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DynamicLCColorStateId(u32);

impl DynamicLCColorStateId {
    pub(crate) const fn from_interner(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// One exact term of one prepared LC color-transition witness.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LCColorWitnessTermId {
    color_contraction_template_id: u32,
    witness_ordinal: u32,
}

impl LCColorWitnessTermId {
    pub const fn new(color_contraction_template_id: u32, witness_ordinal: u32) -> Self {
        Self {
            color_contraction_template_id,
            witness_ordinal,
        }
    }

    pub const fn color_contraction_template_id(self) -> u32 {
        self.color_contraction_template_id
    }

    pub const fn witness_ordinal(self) -> u32 {
        self.witness_ordinal
    }
}

/// One source-state choice in a topology-replay current's local ancestry.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceStateAssignment {
    source_slot: u32,
    state_index: u32,
}

impl SourceStateAssignment {
    pub const fn new(source_slot: u32, state_index: u32) -> Self {
        Self {
            source_slot,
            state_index,
        }
    }

    pub const fn source_slot(self) -> u32 {
        self.source_slot
    }

    pub const fn state_index(self) -> u32 {
        self.state_index
    }
}

/// Layout-specific helicity identity of one recurrence current.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CurrentHelicityIdentity {
    /// Static spin class plus one retained source state for every local source.
    TopologyReplay {
        spin_state_class: i32,
        local_source_states: Box<[SourceStateAssignment]>,
    },
    /// Static transition spin class with source helicities selected at runtime.
    AllFlowUnion { spin_state_class: i32 },
    /// Static spin class and local source ancestry across the complete color basis.
    ContractedColorUnion {
        spin_state_class: i32,
        local_source_states: Box<[SourceStateAssignment]>,
    },
}

impl CurrentHelicityIdentity {
    pub fn topology_replay(
        spin_state_class: i32,
        local_source_states: Vec<SourceStateAssignment>,
    ) -> RusticolResult<Self> {
        validate_sequence_len("local source-state ancestry", local_source_states.len())?;
        let source_slots = local_source_states
            .iter()
            .map(|assignment| assignment.source_slot)
            .collect::<Vec<_>>();
        validate_strict_u32_sequence("local source-state ancestry", &source_slots)?;
        Ok(Self::TopologyReplay {
            spin_state_class,
            local_source_states: local_source_states.into_boxed_slice(),
        })
    }

    pub const fn all_flow_union(spin_state_class: i32) -> Self {
        Self::AllFlowUnion { spin_state_class }
    }

    pub fn contracted_color_union(
        spin_state_class: i32,
        local_source_states: Vec<SourceStateAssignment>,
    ) -> RusticolResult<Self> {
        validate_sequence_len("local source-state ancestry", local_source_states.len())?;
        let source_slots = local_source_states
            .iter()
            .map(|assignment| assignment.source_slot)
            .collect::<Vec<_>>();
        validate_strict_u32_sequence("local source-state ancestry", &source_slots)?;
        Ok(Self::ContractedColorUnion {
            spin_state_class,
            local_source_states: local_source_states.into_boxed_slice(),
        })
    }

    pub const fn strategy(&self) -> RecurrenceStrategy {
        match self {
            Self::TopologyReplay { .. } => RecurrenceStrategy::TopologyReplay,
            Self::AllFlowUnion { .. } => RecurrenceStrategy::AllFlowUnion,
            Self::ContractedColorUnion { .. } => RecurrenceStrategy::ContractedColorUnion,
        }
    }

    pub const fn spin_state_class(&self) -> i32 {
        match self {
            Self::TopologyReplay {
                spin_state_class, ..
            }
            | Self::AllFlowUnion { spin_state_class }
            | Self::ContractedColorUnion {
                spin_state_class, ..
            } => *spin_state_class,
        }
    }

    pub fn local_source_states(&self) -> &[SourceStateAssignment] {
        match self {
            Self::TopologyReplay {
                local_source_states,
                ..
            } => local_source_states,
            Self::AllFlowUnion { .. } => &[],
            Self::ContractedColorUnion {
                local_source_states,
                ..
            } => local_source_states,
        }
    }
}

/// One process-bound runtime source variant for an all-flow-union source.
///
/// The complete variant catalog belongs to the source dispatch domain. No
/// selected numerical helicity enters the current identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeSourceVariantBinding {
    source_state_index: u32,
    public_helicity: i32,
    runtime_variant_id: u32,
    source_template_id: u32,
    source_state_template_id: u32,
    crossed_state_template_id: u32,
    crossed_spin_state_class: i32,
    crossing_factor: ExactComplexRational,
}

impl RuntimeSourceVariantBinding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_state_index: u32,
        public_helicity: i32,
        runtime_variant_id: u32,
        source_template_id: u32,
        source_state_template_id: u32,
        crossed_state_template_id: u32,
        crossed_spin_state_class: i32,
        crossing_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if runtime_variant_id == u32::MAX
            || source_template_id == u32::MAX
            || source_state_template_id == u32::MAX
            || crossed_state_template_id == u32::MAX
        {
            return Err(invalid(
                "runtime source variant reserves the u32 missing-value sentinel",
            ));
        }
        if crossing_factor.is_zero() {
            return Err(invalid(
                "runtime source variant requires a nonzero crossing factor",
            ));
        }
        Ok(Self {
            source_state_index,
            public_helicity,
            runtime_variant_id,
            source_template_id,
            source_state_template_id,
            crossed_state_template_id,
            crossed_spin_state_class,
            crossing_factor,
        })
    }

    pub const fn source_state_index(self) -> u32 {
        self.source_state_index
    }

    pub const fn public_helicity(self) -> i32 {
        self.public_helicity
    }

    pub const fn runtime_variant_id(self) -> u32 {
        self.runtime_variant_id
    }

    pub const fn source_template_id(self) -> u32 {
        self.source_template_id
    }

    pub const fn source_state_template_id(self) -> u32 {
        self.source_state_template_id
    }

    pub const fn crossed_state_template_id(self) -> u32 {
        self.crossed_state_template_id
    }

    pub const fn crossed_spin_state_class(self) -> i32 {
        self.crossed_spin_state_class
    }

    pub const fn crossing_factor(self) -> ExactComplexRational {
        self.crossing_factor
    }
}

/// Source semantics referenced by a recurrence key.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CurrentSourceBinding {
    None,
    FixedTemplate(u32),
    RuntimeDispatch {
        domain: u32,
        source_template_ids: Box<[u32]>,
        variant_bindings: Box<[RuntimeSourceVariantBinding]>,
    },
}

impl CurrentSourceBinding {
    pub fn runtime_dispatch(domain: u32, source_template_ids: Vec<u32>) -> RusticolResult<Self> {
        validate_sequence_len(
            "runtime source-template variants",
            source_template_ids.len(),
        )?;
        validate_strict_u32_sequence("runtime source-template variants", &source_template_ids)?;
        if source_template_ids.contains(&u32::MAX) {
            return Err(invalid(
                "runtime source-template variants reserve the u32 sentinel",
            ));
        }
        Ok(Self::RuntimeDispatch {
            domain,
            source_template_ids: source_template_ids.into_boxed_slice(),
            variant_bindings: Box::new([]),
        })
    }

    pub fn runtime_dispatch_with_variants(
        domain: u32,
        variants: Vec<RuntimeSourceVariantBinding>,
    ) -> RusticolResult<Self> {
        if domain == u32::MAX {
            return Err(invalid(
                "runtime source-dispatch domain reserves the u32 sentinel",
            ));
        }
        validate_sequence_len("runtime source variants", variants.len())?;
        if variants.is_empty() {
            return Err(invalid(
                "runtime source-dispatch domain requires concrete variants",
            ));
        }
        let state_indices = variants
            .iter()
            .map(|variant| variant.source_state_index)
            .collect::<Vec<_>>();
        validate_strict_u32_sequence("runtime source-state variants", &state_indices)?;
        let mut runtime_variant_ids = variants
            .iter()
            .map(|variant| variant.runtime_variant_id)
            .collect::<Vec<_>>();
        runtime_variant_ids.sort_unstable();
        validate_strict_u32_sequence("runtime-helicity variant IDs", &runtime_variant_ids)?;
        let mut source_template_ids = variants
            .iter()
            .map(|variant| variant.source_template_id)
            .collect::<Vec<_>>();
        source_template_ids.sort_unstable();
        source_template_ids.dedup();
        validate_strict_u32_sequence("runtime source-template variants", &source_template_ids)?;
        Ok(Self::RuntimeDispatch {
            domain,
            source_template_ids: source_template_ids.into_boxed_slice(),
            variant_bindings: variants.into_boxed_slice(),
        })
    }
}

impl TryFrom<u32> for RecurrenceNodeKind {
    type Error = RusticolError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Source),
            1 => Ok(Self::Current),
            _ => Err(invalid(format!(
                "unknown recurrence node-kind discriminant {value}"
            ))),
        }
    }
}

/// One canonical term in a momentum linear form.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, bincode::Encode, bincode::Decode,
)]
pub struct MomentumTerm {
    pub source_slot: u32,
    pub coefficient: i32,
}

/// Canonical source-slot-ordered momentum linear form.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, bincode::Encode)]
pub struct CanonicalMomentumLinearForm(Box<[MomentumTerm]>);

impl<Context> bincode::Decode<Context> for CanonicalMomentumLinearForm {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Self::new(bincode::Decode::decode(decoder)?)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))
    }
}

bincode::impl_borrow_decode!(CanonicalMomentumLinearForm);

impl CanonicalMomentumLinearForm {
    pub fn new(terms: Vec<MomentumTerm>) -> RusticolResult<Self> {
        validate_sequence_len("momentum terms", terms.len())?;
        let mut previous = None;
        for (index, term) in terms.iter().enumerate() {
            if term.coefficient == 0 {
                return Err(invalid(format!(
                    "momentum term {index} has zero coefficient"
                )));
            }
            if let Some(previous) = previous
                && previous >= term.source_slot
            {
                return Err(invalid(format!(
                    "momentum terms are not strictly source-slot ordered at row {index}"
                )));
            }
            previous = Some(term.source_slot);
        }
        Ok(Self(terms.into_boxed_slice()))
    }

    pub fn terms(&self) -> &[MomentumTerm] {
        &self.0
    }
}

/// Exact semantic identity before contribution-vector comparison.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CurrentCoreKey {
    catalog_digest: SemanticDigest,
    node_kind: RecurrenceNodeKind,
    current_state_template_id: u32,
    dynamic_lc_color_state_id: DynamicLCColorStateId,
    support_source_slots: Box<[u32]>,
    momentum: CanonicalMomentumLinearForm,
    helicity_identity: CurrentHelicityIdentity,
    flavour_flow: Box<[i32]>,
    quantum_number_flow_id: u32,
    coupling_orders: Box<[u32]>,
    source_binding: CurrentSourceBinding,
    propagator_template_id: Option<u32>,
}

impl CurrentCoreKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        catalog_digest: SemanticDigest,
        node_kind: RecurrenceNodeKind,
        current_state_template_id: u32,
        dynamic_lc_color_state_id: DynamicLCColorStateId,
        support_source_slots: Vec<u32>,
        momentum: CanonicalMomentumLinearForm,
        helicity_identity: CurrentHelicityIdentity,
        flavour_flow: Vec<i32>,
        quantum_number_flow_id: u32,
        coupling_orders: Vec<u32>,
        source_binding: CurrentSourceBinding,
        propagator_template_id: Option<u32>,
    ) -> RusticolResult<Self> {
        validate_sequence_len("support source slots", support_source_slots.len())?;
        validate_sequence_len("flavour flow", flavour_flow.len())?;
        validate_sequence_len("coupling orders", coupling_orders.len())?;
        validate_strict_u32_sequence("support source slots", &support_source_slots)?;
        let ancestry_slots = helicity_identity
            .local_source_states()
            .iter()
            .map(|assignment| assignment.source_slot)
            .collect::<Vec<_>>();
        if helicity_identity.strategy().retains_local_source_ancestry()
            && ancestry_slots != support_source_slots
        {
            return Err(invalid(
                "topology-replay helicity ancestry must cover every local source slot",
            ));
        }
        match (node_kind, helicity_identity.strategy(), &source_binding) {
            (
                RecurrenceNodeKind::Source,
                RecurrenceStrategy::TopologyReplay,
                CurrentSourceBinding::FixedTemplate(_),
            )
            | (
                RecurrenceNodeKind::Source,
                RecurrenceStrategy::ContractedColorUnion,
                CurrentSourceBinding::FixedTemplate(_),
            )
            | (
                RecurrenceNodeKind::Source,
                RecurrenceStrategy::AllFlowUnion,
                CurrentSourceBinding::RuntimeDispatch { .. },
            )
            | (RecurrenceNodeKind::Current, _, CurrentSourceBinding::None) => {}
            (RecurrenceNodeKind::Source, RecurrenceStrategy::TopologyReplay, _) => {
                return Err(invalid(
                    "topology-replay source requires one fixed source template",
                ));
            }
            (RecurrenceNodeKind::Source, RecurrenceStrategy::ContractedColorUnion, _) => {
                return Err(invalid(
                    "contracted-color-union source requires one fixed source template",
                ));
            }
            (RecurrenceNodeKind::Source, RecurrenceStrategy::AllFlowUnion, _) => {
                return Err(invalid(
                    "all-flow-union source requires a runtime dispatch domain",
                ));
            }
            (RecurrenceNodeKind::Current, _, _) => {
                return Err(invalid(
                    "non-source recurrence key must not carry source binding state",
                ));
            }
        }
        if node_kind == RecurrenceNodeKind::Source && support_source_slots.len() != 1 {
            return Err(invalid(
                "source recurrence key must cover exactly one external source slot",
            ));
        }
        Ok(Self {
            catalog_digest,
            node_kind,
            current_state_template_id,
            dynamic_lc_color_state_id,
            support_source_slots: support_source_slots.into_boxed_slice(),
            momentum,
            helicity_identity,
            flavour_flow: flavour_flow.into_boxed_slice(),
            quantum_number_flow_id,
            coupling_orders: coupling_orders.into_boxed_slice(),
            source_binding,
            propagator_template_id,
        })
    }

    pub const fn catalog_digest(&self) -> SemanticDigest {
        self.catalog_digest
    }

    pub const fn node_kind(&self) -> RecurrenceNodeKind {
        self.node_kind
    }

    pub const fn current_state_template_id(&self) -> u32 {
        self.current_state_template_id
    }

    pub const fn dynamic_lc_color_state_id(&self) -> DynamicLCColorStateId {
        self.dynamic_lc_color_state_id
    }

    pub fn support_source_slots(&self) -> &[u32] {
        &self.support_source_slots
    }

    pub const fn momentum(&self) -> &CanonicalMomentumLinearForm {
        &self.momentum
    }

    pub const fn helicity_identity(&self) -> &CurrentHelicityIdentity {
        &self.helicity_identity
    }

    pub const fn spin_state_class(&self) -> i32 {
        self.helicity_identity.spin_state_class()
    }

    pub fn flavour_flow(&self) -> &[i32] {
        &self.flavour_flow
    }

    pub const fn quantum_number_flow_id(&self) -> u32 {
        self.quantum_number_flow_id
    }

    pub fn coupling_orders(&self) -> &[u32] {
        &self.coupling_orders
    }

    pub const fn source_binding(&self) -> &CurrentSourceBinding {
        &self.source_binding
    }

    pub const fn propagator_template_id(&self) -> Option<u32> {
        self.propagator_template_id
    }
}

/// Exact identity of one contribution before coefficient aggregation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContributionKey {
    transition_template_id: u32,
    parent_value_class_ids: Box<[u32]>,
    parent_state_template_ids: Box<[u32]>,
    parent_momenta: Box<[CanonicalMomentumLinearForm]>,
    result_state_template_id: u32,
    quantum_flow_witness_id: u32,
    color_witness_term_id: LCColorWitnessTermId,
    runtime_coupling_binding_digest: SemanticDigest,
    output_projection_id: u32,
}

impl ContributionKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transition_template_id: u32,
        parent_value_class_ids: Vec<u32>,
        parent_state_template_ids: Vec<u32>,
        parent_momenta: Vec<CanonicalMomentumLinearForm>,
        result_state_template_id: u32,
        quantum_flow_witness_id: u32,
        color_witness_term_id: LCColorWitnessTermId,
        runtime_coupling_binding_digest: SemanticDigest,
        output_projection_id: u32,
    ) -> RusticolResult<Self> {
        let parent_count = parent_value_class_ids.len();
        if parent_count == 0 {
            return Err(invalid(
                "recurrence contribution requires at least one parent",
            ));
        }
        if parent_state_template_ids.len() != parent_count || parent_momenta.len() != parent_count {
            return Err(invalid(format!(
                "recurrence contribution parent columns have lengths {parent_count}, {}, and {}",
                parent_state_template_ids.len(),
                parent_momenta.len()
            )));
        }
        u32::try_from(parent_count)
            .map_err(|_| invalid("recurrence contribution parent count exceeds u32"))?;
        Ok(Self {
            transition_template_id,
            parent_value_class_ids: parent_value_class_ids.into_boxed_slice(),
            parent_state_template_ids: parent_state_template_ids.into_boxed_slice(),
            parent_momenta: parent_momenta.into_boxed_slice(),
            result_state_template_id,
            quantum_flow_witness_id,
            color_witness_term_id,
            runtime_coupling_binding_digest,
            output_projection_id,
        })
    }

    pub const fn transition_template_id(&self) -> u32 {
        self.transition_template_id
    }

    pub fn parent_value_class_ids(&self) -> &[u32] {
        &self.parent_value_class_ids
    }

    pub fn parent_state_template_ids(&self) -> &[u32] {
        &self.parent_state_template_ids
    }

    pub fn parent_momenta(&self) -> &[CanonicalMomentumLinearForm] {
        &self.parent_momenta
    }

    pub const fn result_state_template_id(&self) -> u32 {
        self.result_state_template_id
    }

    pub const fn quantum_flow_witness_id(&self) -> u32 {
        self.quantum_flow_witness_id
    }

    pub const fn color_witness_term_id(&self) -> LCColorWitnessTermId {
        self.color_witness_term_id
    }

    pub const fn runtime_coupling_binding_digest(&self) -> SemanticDigest {
        self.runtime_coupling_binding_digest
    }

    pub const fn output_projection_id(&self) -> u32 {
        self.output_projection_id
    }
}

fn validate_strict_u32_sequence(label: &str, values: &[u32]) -> RusticolResult<()> {
    let mut previous = None;
    for (index, value) in values.iter().copied().enumerate() {
        if let Some(previous) = previous
            && previous >= value
        {
            return Err(invalid(format!(
                "{label} are not strictly ordered at row {index}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_sequence_len(label: &str, length: usize) -> RusticolResult<u32> {
    u32::try_from(length).map_err(|_| invalid(format!("{label} length {length} exceeds u32")))
}
