// SPDX-License-Identifier: 0BSD

use std::collections::{BTreeSet, HashMap};

use crate::{RusticolError, RusticolResult};

use super::{DynamicLCColorStateId, ExactComplexRational, SemanticDigest};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// One ordered component in a dynamic partial LC color forest.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorComponentKind {
    OpenString = 0,
    AdjointSegment = 1,
    Trace = 2,
}

impl TryFrom<u8> for LCColorComponentKind {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::OpenString),
            1 => Ok(Self::AdjointSegment),
            2 => Ok(Self::Trace),
            _ => Err(invalid(format!(
                "unsupported LC color component kind {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorComponentRole {
    Active = 0,
    Passive = 1,
    None = 2,
}

impl TryFrom<u8> for LCColorComponentRole {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Active),
            1 => Ok(Self::Passive),
            2 => Ok(Self::None),
            _ => Err(invalid(format!(
                "unsupported LC color component role {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorSourceSeedOperation {
    Empty = 0,
    Singleton = 1,
}

impl TryFrom<u8> for LCColorSourceSeedOperation {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Singleton),
            _ => Err(invalid(format!(
                "unsupported LC source seed operation {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LCColorSourceSeed {
    operation: LCColorSourceSeedOperation,
    output_color_shape_id: u32,
    component_kind: Option<LCColorComponentKind>,
    component_role: LCColorComponentRole,
    proof_digest: SemanticDigest,
}

impl LCColorSourceSeed {
    pub fn new(
        operation: LCColorSourceSeedOperation,
        output_color_shape_id: u32,
        component_kind: Option<LCColorComponentKind>,
        component_role: LCColorComponentRole,
        proof_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        match operation {
            LCColorSourceSeedOperation::Empty
                if component_kind.is_some() || component_role != LCColorComponentRole::None =>
            {
                return Err(invalid(
                    "an empty LC source seed cannot declare a component",
                ));
            }
            LCColorSourceSeedOperation::Singleton
                if component_kind.is_none() || component_role == LCColorComponentRole::None =>
            {
                return Err(invalid(
                    "a singleton LC source seed requires component kind and role",
                ));
            }
            _ => {}
        }
        Ok(Self {
            operation,
            output_color_shape_id,
            component_kind,
            component_role,
            proof_digest,
        })
    }

    pub fn instantiate(
        self,
        source_slot: u32,
        color_representation: i32,
    ) -> RusticolResult<DynamicLCColorState> {
        match self.operation {
            LCColorSourceSeedOperation::Empty => {
                if color_representation != 1 {
                    return Err(invalid(
                        "an empty LC source seed requires the singlet representation",
                    ));
                }
                DynamicLCColorState::new(self.output_color_shape_id, None, Vec::new())
            }
            LCColorSourceSeedOperation::Singleton => {
                let component = LCColorComponent::new(
                    self.component_kind
                        .expect("validated singleton component kind"),
                    vec![source_slot],
                )?;
                let result_port_bindings = match color_representation {
                    3 => vec![LCColorPortBinding::new(0, LCColorEndpoint::Back)],
                    -3 => vec![LCColorPortBinding::new(0, LCColorEndpoint::Front)],
                    8 => vec![
                        LCColorPortBinding::new(0, LCColorEndpoint::Back),
                        LCColorPortBinding::new(0, LCColorEndpoint::Front),
                    ],
                    value => {
                        return Err(invalid(format!(
                            "a singleton LC source seed has unsupported representation {value}",
                        )));
                    }
                };
                if self.component_role != LCColorComponentRole::Active {
                    return Err(invalid(
                        "a colored LC source seed must expose active result ports",
                    ));
                }
                DynamicLCColorState::new_port_wired(
                    self.output_color_shape_id,
                    result_port_bindings,
                    vec![component],
                )
            }
        }
    }

    pub const fn proof_digest(self) -> SemanticDigest {
        self.proof_digest
    }
}

/// Canonical ordered colored-source word for one LC component.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LCColorComponent {
    kind: LCColorComponentKind,
    source_slots: Box<[u32]>,
}

/// One endpoint of an ordered LC strand.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorEndpoint {
    Front = 0,
    Back = 1,
}

impl LCColorEndpoint {
    const fn reversed(self) -> Self {
        match self {
            Self::Front => Self::Back,
            Self::Back => Self::Front,
        }
    }

    fn source_slot(self, component: &LCColorComponent) -> RusticolResult<u32> {
        match self {
            Self::Front => component.source_slots().first().copied(),
            Self::Back => component.source_slots().last().copied(),
        }
        .ok_or_else(|| invalid("an LC color result port references an empty component"))
    }
}

/// Binding from one ordered current-result color port to a strand endpoint.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LCColorPortBinding {
    component_index: u32,
    endpoint: LCColorEndpoint,
}

impl LCColorPortBinding {
    pub const fn new(component_index: u32, endpoint: LCColorEndpoint) -> Self {
        Self {
            component_index,
            endpoint,
        }
    }

    pub const fn component_index(self) -> u32 {
        self.component_index
    }

    pub const fn endpoint(self) -> LCColorEndpoint {
        self.endpoint
    }
}

/// Reference to one result port of one of the two transition parents.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LCColorParentPort {
    parent_index: u8,
    port_index: u8,
}

impl LCColorParentPort {
    pub fn new(parent_index: u8, port_index: u8) -> RusticolResult<Self> {
        if parent_index > 1 {
            return Err(invalid(
                "LC color parent-port parent index must be zero or one",
            ));
        }
        Ok(Self {
            parent_index,
            port_index,
        })
    }

    const fn parent_index(self) -> usize {
        self.parent_index as usize
    }

    const fn port_index(self) -> usize {
        self.port_index as usize
    }
}

/// Exact port wiring certified for one local LC tensor term.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LCColorPortWiring {
    component_parent_order: [u8; 2],
    input_pairings: Box<[[LCColorParentPort; 2]]>,
    result_port_bindings: Box<[LCColorParentPort]>,
}

impl LCColorPortWiring {
    pub fn new(
        component_parent_order: [u8; 2],
        input_pairings: Vec<[LCColorParentPort; 2]>,
        result_port_bindings: Vec<LCColorParentPort>,
    ) -> RusticolResult<Self> {
        if component_parent_order != [0, 1] && component_parent_order != [1, 0] {
            return Err(invalid(
                "LC color component parent order must be [0, 1] or [1, 0]",
            ));
        }
        let mut consumed = BTreeSet::new();
        for pairing in &input_pairings {
            if pairing[0] == pairing[1] {
                return Err(invalid("an LC color port cannot be paired with itself"));
            }
            for port in pairing {
                if !consumed.insert(*port) {
                    return Err(invalid(
                        "an LC color parent port is consumed by more than one pairing",
                    ));
                }
            }
        }
        for port in &result_port_bindings {
            if !consumed.insert(*port) {
                return Err(invalid(
                    "an LC color parent port is both paired and exposed as a result port",
                ));
            }
        }
        Ok(Self {
            component_parent_order,
            input_pairings: input_pairings.into_boxed_slice(),
            result_port_bindings: result_port_bindings.into_boxed_slice(),
        })
    }

    pub fn apply(
        &self,
        left: &DynamicLCColorState,
        right: &DynamicLCColorState,
        result_color_shape_id: u32,
    ) -> RusticolResult<DynamicLCColorState> {
        let parents = [left, right];
        let mut components = Vec::<Option<LCColorComponent>>::new();
        let mut parent_ports = [Vec::new(), Vec::new()];
        for parent_index in self.component_parent_order {
            let parent = parents[parent_index as usize];
            let offset = components.len() as u32;
            components.extend(parent.components().iter().cloned().map(Some));
            parent_ports[parent_index as usize] = parent
                .result_port_bindings()
                .iter()
                .map(|binding| {
                    Some(LCColorPortBinding::new(
                        binding.component_index() + offset,
                        binding.endpoint(),
                    ))
                })
                .collect();
        }

        let expected_parent_ports = parent_ports[0].len() + parent_ports[1].len();
        if expected_parent_ports != self.input_pairings.len() * 2 + self.result_port_bindings.len()
        {
            return Err(invalid(
                "LC color wiring does not consume every parent result port exactly once",
            ));
        }
        for pairing in self.input_pairings.iter().copied() {
            let left_binding = take_parent_port(pairing[0], &mut parent_ports)?;
            let right_binding = take_parent_port(pairing[1], &mut parent_ports)?;
            join_color_endpoints(
                &mut components,
                &mut parent_ports,
                left_binding,
                right_binding,
            )?;
        }
        let mut result_ports = Vec::with_capacity(self.result_port_bindings.len());
        for reference in self.result_port_bindings.iter().copied() {
            result_ports.push(take_parent_port(reference, &mut parent_ports)?);
        }
        if parent_ports.iter().flatten().any(Option::is_some) {
            return Err(invalid(
                "LC color wiring leaves an unconsumed parent result port",
            ));
        }

        let mut remap = vec![u32::MAX; components.len()];
        let mut compact = Vec::new();
        for (old_index, component) in components.into_iter().enumerate() {
            if let Some(component) = component {
                remap[old_index] = compact.len() as u32;
                compact.push(component);
            }
        }
        for binding in &mut result_ports {
            binding.component_index = *remap
                .get(binding.component_index as usize)
                .ok_or_else(|| invalid("LC color result port references an absent component"))?;
            if binding.component_index == u32::MAX {
                return Err(invalid(
                    "LC color result port references a consumed component",
                ));
            }
        }
        DynamicLCColorState::new_port_wired(result_color_shape_id, result_ports, compact)
    }
}

fn take_parent_port(
    reference: LCColorParentPort,
    parent_ports: &mut [Vec<Option<LCColorPortBinding>>; 2],
) -> RusticolResult<LCColorPortBinding> {
    parent_ports
        .get_mut(reference.parent_index())
        .and_then(|ports| ports.get_mut(reference.port_index()))
        .ok_or_else(|| invalid("LC color wiring references an absent parent result port"))?
        .take()
        .ok_or_else(|| invalid("LC color wiring consumes one parent result port twice"))
}

fn join_color_endpoints(
    components: &mut [Option<LCColorComponent>],
    parent_ports: &mut [Vec<Option<LCColorPortBinding>>; 2],
    left_binding: LCColorPortBinding,
    right_binding: LCColorPortBinding,
) -> RusticolResult<()> {
    let left_index = left_binding.component_index() as usize;
    let right_index = right_binding.component_index() as usize;
    if left_index == right_index {
        if left_binding.endpoint() == right_binding.endpoint() {
            return Err(invalid(
                "LC color wiring cannot join the same endpoint of one strand",
            ));
        }
        if parent_ports.iter().flatten().any(|binding| {
            binding
                .as_ref()
                .is_some_and(|binding| binding.component_index() as usize == left_index)
        }) {
            return Err(invalid(
                "LC color wiring closes a strand that still exposes a result port",
            ));
        }
        let component = components
            .get_mut(left_index)
            .and_then(Option::take)
            .ok_or_else(|| invalid("LC color wiring closes an absent strand"))?;
        components[left_index] = Some(LCColorComponent::new(
            LCColorComponentKind::Trace,
            component.source_slots().to_vec(),
        )?);
        return Ok(());
    }

    let mut left = components
        .get_mut(left_index)
        .and_then(Option::take)
        .ok_or_else(|| invalid("LC color wiring joins an absent left strand"))?;
    let mut right = components
        .get_mut(right_index)
        .and_then(Option::take)
        .ok_or_else(|| invalid("LC color wiring joins an absent right strand"))?;
    let reverse_left = left_binding.endpoint() == LCColorEndpoint::Front;
    let reverse_right = right_binding.endpoint() == LCColorEndpoint::Back;
    if reverse_left {
        left = left.reversed()?;
    }
    if reverse_right {
        right = right.reversed()?;
    }
    let result_kind = if left.kind() == LCColorComponentKind::OpenString
        || right.kind() == LCColorComponentKind::OpenString
    {
        LCColorComponentKind::OpenString
    } else {
        LCColorComponentKind::AdjointSegment
    };
    let word = left
        .source_slots()
        .iter()
        .chain(right.source_slots())
        .copied()
        .collect();
    let retained_index = left_index.min(right_index);
    components[retained_index] = Some(LCColorComponent::new(result_kind, word)?);

    for binding in parent_ports.iter_mut().flatten().flatten() {
        let component_index = binding.component_index() as usize;
        if component_index == left_index {
            binding.component_index = retained_index as u32;
            if reverse_left {
                binding.endpoint = binding.endpoint.reversed();
            }
        } else if component_index == right_index {
            binding.component_index = retained_index as u32;
            if reverse_right {
                binding.endpoint = binding.endpoint.reversed();
            }
        }
    }
    Ok(())
}

impl LCColorComponent {
    pub fn new(kind: LCColorComponentKind, source_slots: Vec<u32>) -> RusticolResult<Self> {
        if source_slots.is_empty() {
            return Err(invalid(
                "an LC color component cannot have an empty source word",
            ));
        }
        if source_slots.len() > u32::MAX as usize {
            return Err(invalid("LC color component exceeds the u32 ABI domain"));
        }
        let mut seen = BTreeSet::new();
        if source_slots.iter().any(|slot| !seen.insert(*slot)) {
            return Err(invalid(
                "an LC color component cannot contain a source slot more than once",
            ));
        }
        let source_slots = if kind == LCColorComponentKind::Trace {
            canonical_trace_rotation(source_slots)
        } else {
            source_slots
        };
        Ok(Self {
            kind,
            source_slots: source_slots.into_boxed_slice(),
        })
    }

    pub const fn kind(&self) -> LCColorComponentKind {
        self.kind
    }

    pub fn source_slots(&self) -> &[u32] {
        &self.source_slots
    }

    fn reversed(&self) -> RusticolResult<Self> {
        Self::new(self.kind, self.source_slots.iter().rev().copied().collect())
    }
}

/// Internable dynamic LC state, independent of physical sector identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DynamicLCColorState {
    output_color_shape_id: u32,
    active_component_index: Option<u32>,
    result_port_bindings: Box<[LCColorPortBinding]>,
    components: Box<[LCColorComponent]>,
}

impl DynamicLCColorState {
    pub fn new(
        output_color_shape_id: u32,
        active_component_index: Option<u32>,
        components: Vec<LCColorComponent>,
    ) -> RusticolResult<Self> {
        let result_port_bindings = active_component_index
            .map(|index| LCColorPortBinding::new(index, LCColorEndpoint::Back))
            .into_iter()
            .collect();
        Self::new_port_wired(output_color_shape_id, result_port_bindings, components)
    }

    pub const fn output_color_shape_id(&self) -> u32 {
        self.output_color_shape_id
    }

    pub fn components(&self) -> &[LCColorComponent] {
        &self.components
    }

    pub const fn active_component_index(&self) -> Option<u32> {
        self.active_component_index
    }

    pub fn active_component(&self) -> Option<&LCColorComponent> {
        self.active_component_index
            .map(|index| &self.components[index as usize])
    }

    pub(crate) fn reversed(&self) -> RusticolResult<Self> {
        let components = self
            .components
            .iter()
            .rev()
            .map(LCColorComponent::reversed)
            .collect::<RusticolResult<Vec<_>>>()?;
        let result_port_bindings = self
            .result_port_bindings
            .iter()
            .rev()
            .map(|binding| {
                LCColorPortBinding::new(
                    self.components.len() as u32 - 1 - binding.component_index,
                    binding.endpoint.reversed(),
                )
            })
            .collect();
        Self::new_port_wired(self.output_color_shape_id, result_port_bindings, components)
    }

    pub(crate) fn pure_adjoint_word(&self) -> Option<&[u32]> {
        if self.components.len() != 1
            || self.active_component_index != Some(0)
            || self.result_port_bindings.len() != 2
        {
            return None;
        }
        let exposes_front = self.result_port_bindings.iter().any(|binding| {
            binding.component_index() == 0 && binding.endpoint() == LCColorEndpoint::Front
        });
        let exposes_back = self.result_port_bindings.iter().any(|binding| {
            binding.component_index() == 0 && binding.endpoint() == LCColorEndpoint::Back
        });
        if !exposes_front || !exposes_back {
            return None;
        }
        let component = &self.components[0];
        (component.kind() == LCColorComponentKind::AdjointSegment)
            .then_some(component.source_slots())
    }

    pub fn new_port_wired(
        output_color_shape_id: u32,
        result_port_bindings: Vec<LCColorPortBinding>,
        components: Vec<LCColorComponent>,
    ) -> RusticolResult<Self> {
        if components.len() > u32::MAX as usize {
            return Err(invalid("LC color forest exceeds the u32 ABI domain"));
        }
        let mut seen_slots = BTreeSet::new();
        for component in &components {
            if component
                .source_slots()
                .iter()
                .any(|slot| !seen_slots.insert(*slot))
            {
                return Err(invalid(
                    "a dynamic LC color state cannot duplicate colored source slots",
                ));
            }
        }
        let mut seen_ports = BTreeSet::new();
        for binding in &result_port_bindings {
            let component = components
                .get(binding.component_index as usize)
                .ok_or_else(|| invalid("LC color result port is outside the color forest"))?;
            if component.kind() == LCColorComponentKind::Trace {
                return Err(invalid("a closed LC trace cannot expose a result port"));
            }
            if !seen_ports.insert(*binding) {
                return Err(invalid(
                    "two LC color result ports cannot bind the same strand endpoint",
                ));
            }
        }
        let active_component_index = result_port_bindings
            .first()
            .map(|binding| binding.component_index())
            .filter(|index| {
                result_port_bindings
                    .iter()
                    .all(|binding| binding.component_index() == *index)
            });
        Ok(Self {
            output_color_shape_id,
            active_component_index,
            result_port_bindings: result_port_bindings.into_boxed_slice(),
            components: components.into_boxed_slice(),
        })
    }

    pub fn result_port_bindings(&self) -> &[LCColorPortBinding] {
        &self.result_port_bindings
    }

    /// Source lineage exposed by this current's ordered result color ports.
    ///
    /// Closed neutral currents deliberately return an empty lineage.  Open
    /// fermion and adjoint currents retain only the source endpoints that can
    /// flow into a later recurrence, rather than the history of already closed
    /// quark lines used to construct them.
    pub fn result_port_lineage_source_slots(&self) -> RusticolResult<Vec<u32>> {
        self.result_port_bindings
            .iter()
            .map(|binding| {
                let component = self
                    .components
                    .get(binding.component_index() as usize)
                    .ok_or_else(|| invalid("LC color result-port component disappeared"))?;
                binding.endpoint().source_slot(component)
            })
            .collect()
    }
}

/// Owns the only mapping from semantic LC color states to recurrence IDs.
#[derive(Clone, Debug, Default)]
pub struct DynamicLCColorStateInterner {
    states: Vec<DynamicLCColorState>,
    ids: HashMap<DynamicLCColorState, DynamicLCColorStateId>,
}

impl DynamicLCColorStateInterner {
    pub fn intern(&mut self, state: DynamicLCColorState) -> RusticolResult<DynamicLCColorStateId> {
        if let Some(id) = self.ids.get(&state) {
            return Ok(*id);
        }
        let raw_id = u32::try_from(self.states.len())
            .map_err(|_| invalid("dynamic LC color-state interner exceeds u32"))?;
        let id = DynamicLCColorStateId::from_interner(raw_id);
        self.states.push(state.clone());
        self.ids.insert(state, id);
        Ok(id)
    }

    pub fn get(&self, id: DynamicLCColorStateId) -> Option<&DynamicLCColorState> {
        self.states.get(id.get() as usize)
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn into_states(self) -> Vec<DynamicLCColorState> {
        self.states
    }
}

/// Exact operation performed by one compiler-certified LC transition term.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorComponentOperation {
    ConcatenateJoin = 0,
    ConcatenateKeep = 1,
    InheritLeft = 2,
    InheritRight = 3,
    Empty = 4,
    Close = 5,
}

impl TryFrom<u8> for LCColorComponentOperation {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ConcatenateJoin),
            1 => Ok(Self::ConcatenateKeep),
            2 => Ok(Self::InheritLeft),
            3 => Ok(Self::InheritRight),
            4 => Ok(Self::Empty),
            5 => Ok(Self::Close),
            _ => Err(invalid(format!(
                "unsupported LC color component operation {value}"
            ))),
        }
    }
}

/// Executable color-state witness emitted by the model compiler.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LCColorTransitionWitness {
    input_permutation: [u8; 2],
    reverse_parent_mask: u8,
    operation: LCColorComponentOperation,
    result_component_kind: Option<LCColorComponentKind>,
    result_component_role: LCColorComponentRole,
    result_color_shape_id: Option<u32>,
    port_wiring: LCColorPortWiring,
    exact_factor: ExactComplexRational,
    proof_digest: SemanticDigest,
}

impl LCColorTransitionWitness {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_permutation: [u8; 2],
        reverse_parent_mask: u8,
        operation: LCColorComponentOperation,
        result_component_kind: Option<LCColorComponentKind>,
        result_component_role: LCColorComponentRole,
        result_color_shape_id: Option<u32>,
        port_wiring: LCColorPortWiring,
        exact_factor: ExactComplexRational,
        proof_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        if input_permutation != [0, 1] && input_permutation != [1, 0] {
            return Err(invalid(
                "LC color witness input permutation must be [0, 1] or [1, 0]",
            ));
        }
        if reverse_parent_mask > 0b11 {
            return Err(invalid(
                "LC color witness has an invalid parent-reversal mask",
            ));
        }
        if exact_factor.is_zero() {
            return Err(invalid(
                "an executable LC color witness cannot have zero factor",
            ));
        }
        match operation {
            LCColorComponentOperation::Close => {
                if result_component_role != LCColorComponentRole::None
                    || result_color_shape_id.is_some()
                {
                    return Err(invalid(
                        "an LC closure witness cannot declare an active result color state",
                    ));
                }
            }
            LCColorComponentOperation::ConcatenateJoin => {
                if result_component_kind.is_none()
                    || result_component_role == LCColorComponentRole::None
                    || result_color_shape_id.is_none()
                {
                    return Err(invalid(
                        "an LC join witness requires result component, role, and shape contracts",
                    ));
                }
            }
            LCColorComponentOperation::InheritLeft | LCColorComponentOperation::InheritRight => {
                if result_component_kind.is_some()
                    || result_component_role != LCColorComponentRole::Active
                    || result_color_shape_id.is_none()
                {
                    return Err(invalid(
                        "an LC inheritance witness requires an active result role and shape",
                    ));
                }
            }
            _ => {
                if result_component_kind.is_some()
                    || result_component_role != LCColorComponentRole::None
                    || result_color_shape_id.is_none()
                {
                    return Err(invalid(
                        "a passive LC witness requires only a result shape contract",
                    ));
                }
            }
        }
        Ok(Self {
            input_permutation,
            reverse_parent_mask,
            operation,
            result_component_kind,
            result_component_role,
            result_color_shape_id,
            port_wiring,
            exact_factor,
            proof_digest,
        })
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }

    pub const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }

    pub fn apply(
        &self,
        left: &DynamicLCColorState,
        right: &DynamicLCColorState,
    ) -> RusticolResult<Option<DynamicLCColorState>> {
        let mut parents = [left.clone(), right.clone()];
        for (index, parent) in parents.iter_mut().enumerate() {
            if self.reverse_parent_mask & (1 << index) != 0 {
                *parent = parent.reversed()?;
            }
        }
        if self.operation == LCColorComponentOperation::Close {
            self.closed_components(&parents[0], &parents[1])?;
            return Ok(None);
        }
        Ok(Some(
            self.port_wiring.apply(
                &parents[0],
                &parents[1],
                self.result_color_shape_id
                    .expect("validated non-closure result color shape"),
            )?,
        ))
    }

    pub fn closed_components(
        &self,
        left: &DynamicLCColorState,
        right: &DynamicLCColorState,
    ) -> RusticolResult<Vec<LCColorComponent>> {
        if self.operation != LCColorComponentOperation::Close {
            return Err(invalid(
                "closed-components is available only for an LC closure witness",
            ));
        }
        let mut parents = [left.clone(), right.clone()];
        for (index, parent) in parents.iter_mut().enumerate() {
            if self.reverse_parent_mask & (1 << index) != 0 {
                *parent = parent.reversed()?;
            }
        }
        let closed = self.port_wiring.apply(&parents[0], &parents[1], 0)?;
        if !closed.result_port_bindings().is_empty() {
            return Err(invalid("an LC closure leaves an exposed color port"));
        }
        Ok(closed.components().to_vec())
    }
}

fn canonical_trace_rotation(mut source_slots: Vec<u32>) -> Vec<u32> {
    let length = source_slots.len();
    if length < 2 {
        return source_slots;
    }

    // Booth's least-rotation scan compares every cyclic suffix at most a
    // constant number of times. This preserves the exact lexicographic
    // canonical form without materializing all O(n) rotations.
    let mut left = 0usize;
    let mut right = 1usize;
    let mut matched = 0usize;
    while left < length && right < length && matched < length {
        let left_value = source_slots[(left + matched) % length];
        let right_value = source_slots[(right + matched) % length];
        match left_value.cmp(&right_value) {
            std::cmp::Ordering::Equal => matched += 1,
            std::cmp::Ordering::Greater => {
                left += matched + 1;
                if left == right {
                    left += 1;
                }
                matched = 0;
            }
            std::cmp::Ordering::Less => {
                right += matched + 1;
                if left == right {
                    right += 1;
                }
                matched = 0;
            }
        }
    }
    source_slots.rotate_left(left.min(right) % length);
    source_slots
}

#[cfg(test)]
mod tests {
    use super::canonical_trace_rotation;

    fn reference_trace_rotation(source_slots: &[u32]) -> Vec<u32> {
        (0..source_slots.len())
            .map(|offset| {
                source_slots[offset..]
                    .iter()
                    .chain(&source_slots[..offset])
                    .copied()
                    .collect::<Vec<_>>()
            })
            .min()
            .unwrap_or_default()
    }

    #[test]
    fn linear_trace_rotation_matches_exhaustive_reference() {
        for length in 0..=8 {
            let case_count = 3_usize.pow(length as u32);
            for mut encoded in 0..case_count {
                let mut source_slots = Vec::with_capacity(length);
                for _ in 0..length {
                    source_slots.push((encoded % 3) as u32);
                    encoded /= 3;
                }
                assert_eq!(
                    canonical_trace_rotation(source_slots.clone()),
                    reference_trace_rotation(&source_slots),
                    "source slots {source_slots:?}",
                );
            }
        }
    }
}
