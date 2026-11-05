// SPDX-License-Identifier: 0BSD

use super::*;

pub(crate) const ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI: &str =
    "pyamplicol-on-the-fly-process-seed-identity-v1";

/// Compact publicable identity of one decoded process-seed source state.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OnTheFlyProcessSeedStateIdentityV1 {
    pub(crate) state_index: u32,
    pub(crate) public_helicity: i32,
    pub(crate) prepared_mass_parameter_slot: Option<u32>,
}

/// Compact publicable identity of one decoded process-seed source anchor.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OnTheFlyProcessSeedSourceIdentityV1 {
    pub(crate) source_slot: u32,
    pub(crate) public_label: u32,
    pub(crate) is_initial: bool,
    pub(crate) states: Vec<OnTheFlyProcessSeedStateIdentityV1>,
}

/// Exact compact identity embedded beside the sole native process seed.
///
/// This is derived only from a successfully decoded seed.  It deliberately
/// records source-domain facts rather than reproducing the binary seed.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OnTheFlyProcessSeedIdentityV1 {
    pub(crate) abi: String,
    pub(crate) process_digest: String,
    pub(crate) compiled_model_digest: String,
    pub(crate) recurrence_template_catalog_digest: String,
    pub(crate) prepared_kernel_pack_digest: String,
    pub(crate) recurrence_direct_template_catalog_digest: String,
    pub(crate) semantic_digest: String,
    pub(crate) external_permutation: Vec<u32>,
    pub(crate) external_sources: Vec<OnTheFlyProcessSeedSourceIdentityV1>,
}

/// Model-neutral source family needed by the private direct-source adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OnTheFlySourceWavefunctionFamilyV1 {
    Scalar = 0,
    WeylFermion = 1,
    DiracFermion = 2,
    Vector = 3,
    Spin2 = 4,
}
impl OnTheFlySourceWavefunctionFamilyV1 {
    pub(super) const fn is_fermionic(self) -> bool {
        matches!(self, Self::WeylFermion | Self::DiracFermion)
    }
}

/// Model-neutral particle orientation retained by one source state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OnTheFlySourceOrientationV1 {
    Particle = 0,
    Antiparticle = 1,
    SelfConjugate = 2,
}

/// Exact external LC role; this is structural input, not a public-flow row.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub(crate) enum OnTheFlyExternalColorRoleV1 {
    Singlet = 0,
    Fundamental = 1,
    Antifundamental = 2,
    Adjoint = 3,
}

impl OnTheFlyExternalColorRoleV1 {
    pub(super) const fn is_colored(self) -> bool {
        !matches!(self, Self::Singlet)
    }

    pub(super) const fn is_pairing_endpoint(self) -> bool {
        matches!(self, Self::Fundamental | Self::Antifundamental)
    }
}

/// One authenticated source execution row exposed to the eventual engine
/// adapter. Source-template IDs may be sparse; callers must retain them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlySourceExecutionSpecV1 {
    pub(crate) source_slot: u32,
    pub(crate) source_template_id: u32,
    pub(crate) spin_state_class: i32,
    pub(crate) family: OnTheFlySourceWavefunctionFamilyV1,
    pub(crate) orientation: OnTheFlySourceOrientationV1,
    pub(crate) helicity: i32,
    pub(crate) chirality: i32,
    pub(crate) prepared_mass_parameter_slot: Option<u32>,
}

/// One concrete selected-helicity source state retained in the compact seed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlySourceStateV1 {
    pub(super) state_index: u32,
    pub(super) public_helicity: i32,
    pub(super) source_helicity: i32,
    pub(super) source_template_id: u32,
    pub(super) current_state_template_id: u32,
    pub(super) source_semantic_digest: SemanticDigest,
    pub(super) current_state_semantic_digest: SemanticDigest,
    pub(super) momentum_sign: i32,
    pub(super) crossing_phase: ExactComplexRational,
    pub(super) spin_state: i32,
    pub(super) chirality: i32,
    pub(super) flavour_flow: Box<[i32]>,
    pub(super) quantum_number_flow_id: u32,
    pub(super) color_seed_proof_digest: SemanticDigest,
    pub(super) source_family: OnTheFlySourceWavefunctionFamilyV1,
    pub(super) source_orientation: OnTheFlySourceOrientationV1,
    pub(super) prepared_mass_parameter_slot: Option<u32>,
}

impl OnTheFlySourceStateV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        state_index: u32,
        public_helicity: i32,
        source_helicity: i32,
        source_template_id: u32,
        current_state_template_id: u32,
        source_semantic_digest: SemanticDigest,
        current_state_semantic_digest: SemanticDigest,
        momentum_sign: i32,
        crossing_phase: ExactComplexRational,
        spin_state: i32,
        chirality: i32,
        flavour_flow: Vec<i32>,
        quantum_number_flow_id: u32,
        color_seed_proof_digest: SemanticDigest,
        source_family: OnTheFlySourceWavefunctionFamilyV1,
        source_orientation: OnTheFlySourceOrientationV1,
        prepared_mass_parameter_slot: Option<u32>,
    ) -> RusticolResult<Self> {
        if source_template_id == MISSING_U32 || current_state_template_id == MISSING_U32 {
            return Err(invalid("source state reserves the missing u32 sentinel"));
        }
        if momentum_sign != -1 && momentum_sign != 1 {
            return Err(invalid("source momentum sign must be -1 or +1"));
        }
        if crossing_phase.is_zero() || flavour_flow.is_empty() {
            return Err(invalid(
                "source state requires a nonzero crossing phase and flavour ancestry",
            ));
        }
        if source_family.is_fermionic()
            && source_orientation == OnTheFlySourceOrientationV1::SelfConjugate
        {
            return Err(invalid(
                "fermion source state cannot have self-conjugate orientation",
            ));
        }
        Ok(Self {
            state_index,
            public_helicity,
            source_helicity,
            source_template_id,
            current_state_template_id,
            source_semantic_digest,
            current_state_semantic_digest,
            momentum_sign,
            crossing_phase,
            spin_state,
            chirality,
            flavour_flow: flavour_flow.into_boxed_slice(),
            quantum_number_flow_id,
            color_seed_proof_digest,
            source_family,
            source_orientation,
            prepared_mass_parameter_slot,
        })
    }

    pub(crate) const fn state_index(&self) -> u32 {
        self.state_index
    }

    pub(crate) const fn public_helicity(&self) -> i32 {
        self.public_helicity
    }

    pub(crate) const fn prepared_mass_parameter_slot(&self) -> Option<u32> {
        self.prepared_mass_parameter_slot
    }
}

/// All concrete source alternatives for one external source slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlySourceAnchorV1 {
    pub(super) source_slot: u32,
    pub(super) external_label: u32,
    pub(super) is_initial: bool,
    pub(super) color_role: OnTheFlyExternalColorRoleV1,
    pub(super) is_fermionic: bool,
    pub(super) pairing_source_contract_digest: Option<SemanticDigest>,
    pub(super) states: Box<[OnTheFlySourceStateV1]>,
}

impl OnTheFlySourceAnchorV1 {
    pub(crate) fn new(
        source_slot: u32,
        external_label: u32,
        is_initial: bool,
        color_role: OnTheFlyExternalColorRoleV1,
        is_fermionic: bool,
        pairing_source_contract_digest: Option<SemanticDigest>,
        mut states: Vec<OnTheFlySourceStateV1>,
    ) -> RusticolResult<Self> {
        if states.is_empty() {
            return Err(invalid("source anchor has no concrete state"));
        }
        states.sort_unstable_by_key(|state| state.state_index);
        if states
            .windows(2)
            .any(|pair| pair[0].state_index == pair[1].state_index)
        {
            return Err(invalid("source anchor repeats a state index"));
        }
        if states
            .iter()
            .any(|state| state.source_family.is_fermionic() != is_fermionic)
        {
            return Err(invalid(
                "source anchor statistics disagree with its execution states",
            ));
        }
        if color_role.is_pairing_endpoint() != pairing_source_contract_digest.is_some()
            || (color_role.is_pairing_endpoint() && !is_fermionic)
        {
            return Err(invalid(
                "open-line endpoint role requires one fermionic source contract",
            ));
        }
        Ok(Self {
            source_slot,
            external_label,
            is_initial,
            color_role,
            is_fermionic,
            pairing_source_contract_digest,
            states: states.into_boxed_slice(),
        })
    }

    pub(super) fn selected(
        &self,
        state_index: u32,
        public_helicity: i32,
    ) -> RusticolResult<&OnTheFlySourceStateV1> {
        self.states
            .iter()
            .find(|state| {
                state.state_index == state_index && state.public_helicity == public_helicity
            })
            .ok_or_else(|| {
                invalid(format!(
                    "source slot {} has no state {state_index} with public helicity {public_helicity}",
                    self.source_slot
                ))
            })
    }

    pub(crate) const fn source_slot(&self) -> u32 {
        self.source_slot
    }

    pub(crate) const fn external_label(&self) -> u32 {
        self.external_label
    }

    pub(crate) const fn is_initial(&self) -> bool {
        self.is_initial
    }

    pub(crate) const fn color_role(&self) -> OnTheFlyExternalColorRoleV1 {
        self.color_role
    }

    pub(crate) fn states(&self) -> &[OnTheFlySourceStateV1] {
        &self.states
    }
}

/// One explicit endpoint in a compact species pairing class.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct OnTheFlyPairingEndpointV1 {
    pub(crate) source_slot: u32,
    pub(crate) source_contract_digest: SemanticDigest,
}

/// O(external-leg) pairing-class input. Rules and permutations are never
/// enumerated: one selected permutation is validated from Lehmer digits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyPairingClassV1 {
    pub(super) species: Box<str>,
    pub(super) species_semantic_digest: SemanticDigest,
    pub(super) fundamental_endpoints: Box<[OnTheFlyPairingEndpointV1]>,
    pub(super) antifundamental_endpoints: Box<[OnTheFlyPairingEndpointV1]>,
    pub(super) semantic_digest: SemanticDigest,
}

impl OnTheFlyPairingClassV1 {
    pub(crate) fn new(
        species: impl Into<Box<str>>,
        species_semantic_digest: SemanticDigest,
        mut fundamental_endpoints: Vec<OnTheFlyPairingEndpointV1>,
        mut antifundamental_endpoints: Vec<OnTheFlyPairingEndpointV1>,
    ) -> RusticolResult<Self> {
        let species = species.into();
        if species.is_empty() {
            return Err(invalid("pairing class species identity is empty"));
        }
        fundamental_endpoints.sort_unstable_by_key(|endpoint| endpoint.source_slot);
        antifundamental_endpoints.sort_unstable_by_key(|endpoint| endpoint.source_slot);
        if fundamental_endpoints.is_empty()
            || fundamental_endpoints.len() != antifundamental_endpoints.len()
            || fundamental_endpoints
                .windows(2)
                .any(|pair| pair[0].source_slot == pair[1].source_slot)
            || antifundamental_endpoints
                .windows(2)
                .any(|pair| pair[0].source_slot == pair[1].source_slot)
        {
            return Err(invalid(
                "pairing class endpoints must be nonempty, balanced, and unique",
            ));
        }
        let mut result = Self {
            species,
            species_semantic_digest,
            fundamental_endpoints: fundamental_endpoints.into_boxed_slice(),
            antifundamental_endpoints: antifundamental_endpoints.into_boxed_slice(),
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        result.semantic_digest = result.compute_digest()?;
        Ok(result)
    }

    pub(crate) fn fundamental_endpoints(&self) -> &[OnTheFlyPairingEndpointV1] {
        &self.fundamental_endpoints
    }

    pub(crate) fn antifundamental_endpoints(&self) -> &[OnTheFlyPairingEndpointV1] {
        &self.antifundamental_endpoints
    }

    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(b"pyamplicol-on-the-fly-pairing-class-v1\0");
        hash_len(&mut hash, self.species.len(), "pairing species")?;
        hash.update(self.species.as_bytes());
        hash_digest(&mut hash, self.species_semantic_digest);
        for endpoints in [&self.fundamental_endpoints, &self.antifundamental_endpoints] {
            hash_len(&mut hash, endpoints.len(), "pairing endpoints")?;
            for endpoint in endpoints.iter() {
                hash.update(endpoint.source_slot.to_le_bytes());
                hash_digest(&mut hash, endpoint.source_contract_digest);
            }
        }
        final_digest(hash)
    }
}

/// Compact immutable process input for one on-the-fly lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OnTheFlyCouplingOrderPolicyV1 {
    Minimal = 0,
    Explicit = 1,
}

/// Compact immutable process input for one on-the-fly lane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyProcessSeedV1 {
    pub(super) process_digest: SemanticDigest,
    pub(super) model_digest: SemanticDigest,
    pub(super) template_catalog_digest: SemanticDigest,
    pub(super) prepared_pack_digest: SemanticDigest,
    pub(super) direct_catalog_digest: SemanticDigest,
    pub(super) normalization_semantic_digest: SemanticDigest,
    pub(super) normalization_convention: Box<str>,
    pub(super) source_anchors: Box<[OnTheFlySourceAnchorV1]>,
    /// Gather map: construction source slot i receives public external slot
    /// external_permutation[i].
    pub(super) external_permutation: Box<[u32]>,
    pub(super) coupling_order_policy: OnTheFlyCouplingOrderPolicyV1,
    pub(super) coupling_hierarchies: Box<[u32]>,
    /// User-supplied maxima only. `None` means that this model order has no
    /// explicit hard cap; it never means that the default minimal policy has
    /// already been resolved.
    pub(super) coupling_limits: Box<[Option<u32>]>,
    pub(super) pairing_classes: Box<[OnTheFlyPairingClassV1]>,
    pub(super) semantic_digest: SemanticDigest,
}

impl OnTheFlyProcessSeedV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        process_digest: SemanticDigest,
        model_digest: SemanticDigest,
        template_catalog_digest: SemanticDigest,
        prepared_pack_digest: SemanticDigest,
        direct_catalog_digest: SemanticDigest,
        normalization_semantic_digest: SemanticDigest,
        normalization_convention: impl Into<Box<str>>,
        normalization_factor: ExactComplexRational,
        mut source_anchors: Vec<OnTheFlySourceAnchorV1>,
        external_permutation: Vec<u32>,
        coupling_order_policy: OnTheFlyCouplingOrderPolicyV1,
        coupling_hierarchies: Vec<u32>,
        coupling_limits: Vec<Option<u32>>,
        mut pairing_classes: Vec<OnTheFlyPairingClassV1>,
    ) -> RusticolResult<Self> {
        if normalization_factor != ExactComplexRational::ONE {
            return Err(integrity(
                "raw-amplitude seed normalization factor must be exact one",
            ));
        }
        let normalization_convention = normalization_convention.into();
        if normalization_convention.is_empty() {
            return Err(invalid("normalization convention is empty"));
        }
        source_anchors.sort_unstable_by_key(|anchor| anchor.source_slot);
        if source_anchors.len() < 2
            || source_anchors
                .iter()
                .enumerate()
                .any(|(index, anchor)| anchor.source_slot as usize != index)
        {
            return Err(invalid(
                "source anchors must form a dense domain with at least two slots",
            ));
        }
        if coupling_limits.is_empty()
            || coupling_hierarchies.len() != coupling_limits.len()
            || coupling_hierarchies.contains(&0)
        {
            return Err(invalid(
                "on-the-fly coupling policy requires one positive hierarchy and optional hard cap per model order",
            ));
        }
        validate_permutation(
            &external_permutation,
            source_anchors.len(),
            "external source permutation",
        )?;
        pairing_classes.sort_unstable_by(|left, right| {
            left.species
                .cmp(&right.species)
                .then_with(|| left.semantic_digest.cmp(&right.semantic_digest))
        });
        if pairing_classes
            .windows(2)
            .any(|pair| pair[0].species == pair[1].species)
        {
            return Err(invalid("pairing class repeats a species identity"));
        }
        let mut covered_endpoints = BTreeSet::new();
        for pairing_class in &pairing_classes {
            for (endpoints, role) in [
                (
                    pairing_class.fundamental_endpoints.as_ref(),
                    OnTheFlyExternalColorRoleV1::Fundamental,
                ),
                (
                    pairing_class.antifundamental_endpoints.as_ref(),
                    OnTheFlyExternalColorRoleV1::Antifundamental,
                ),
            ] {
                for endpoint in endpoints {
                    let anchor = source_anchors
                        .get(endpoint.source_slot as usize)
                        .ok_or_else(|| invalid("pairing endpoint is outside the source domain"))?;
                    if anchor.color_role != role
                        || anchor.pairing_source_contract_digest
                            != Some(endpoint.source_contract_digest)
                        || !covered_endpoints.insert(endpoint.source_slot)
                    {
                        return Err(integrity(
                            "pairing endpoint role or source-contract identity is inconsistent",
                        ));
                    }
                }
            }
        }
        let required_endpoints = source_anchors
            .iter()
            .filter(|anchor| anchor.color_role.is_pairing_endpoint())
            .map(|anchor| anchor.source_slot)
            .collect::<BTreeSet<_>>();
        if covered_endpoints != required_endpoints {
            return Err(integrity(
                "pairing classes do not cover every open-line endpoint exactly once",
            ));
        }

        let mut result = Self {
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            source_anchors: source_anchors.into_boxed_slice(),
            external_permutation: external_permutation.into_boxed_slice(),
            coupling_order_policy,
            coupling_hierarchies: coupling_hierarchies.into_boxed_slice(),
            coupling_limits: coupling_limits.into_boxed_slice(),
            pairing_classes: pairing_classes.into_boxed_slice(),
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        result.semantic_digest = result.compute_digest()?;
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) fn with_selector_local_zero(self) -> RusticolResult<Self> {
        let Self {
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            source_anchors,
            external_permutation,
            coupling_order_policy,
            coupling_hierarchies,
            coupling_limits,
            pairing_classes,
            semantic_digest: _,
        } = self;
        let source_anchors = source_anchors
            .into_vec()
            .into_iter()
            .map(|anchor| {
                let OnTheFlySourceAnchorV1 {
                    source_slot,
                    external_label,
                    is_initial,
                    color_role,
                    is_fermionic,
                    pairing_source_contract_digest,
                    states,
                } = anchor;
                let mut states = states.into_vec();
                let mut zero = states[0].clone();
                zero.state_index = 1;
                zero.public_helicity = 1;
                zero.spin_state = 50_001;
                states.push(zero);
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    external_label,
                    is_initial,
                    color_role,
                    is_fermionic,
                    pairing_source_contract_digest,
                    states,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Self::new(
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            ExactComplexRational::ONE,
            source_anchors,
            external_permutation.into_vec(),
            coupling_order_policy,
            coupling_hierarchies.into_vec(),
            coupling_limits.into_vec(),
            pairing_classes.into_vec(),
        )
    }

    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(b"pyamplicol-on-the-fly-process-seed-v2\0");
        for digest in [
            self.process_digest,
            self.model_digest,
            self.template_catalog_digest,
            self.prepared_pack_digest,
            self.direct_catalog_digest,
            self.normalization_semantic_digest,
        ] {
            hash_digest(&mut hash, digest);
        }
        hash_len(
            &mut hash,
            self.normalization_convention.len(),
            "normalization convention",
        )?;
        hash.update(self.normalization_convention.as_bytes());
        hash_len(&mut hash, self.source_anchors.len(), "source anchors")?;
        for anchor in &self.source_anchors {
            hash.update(anchor.source_slot.to_le_bytes());
            hash.update(anchor.external_label.to_le_bytes());
            hash.update([
                anchor.color_role as u8,
                u8::from(anchor.is_fermionic),
                u8::from(anchor.is_initial),
            ]);
            match anchor.pairing_source_contract_digest {
                None => hash.update([0]),
                Some(value) => {
                    hash.update([1]);
                    hash_digest(&mut hash, value);
                }
            }
            hash_len(&mut hash, anchor.states.len(), "source states")?;
            for state in &anchor.states {
                hash.update(state.state_index.to_le_bytes());
                hash.update(state.public_helicity.to_le_bytes());
                hash.update(state.source_helicity.to_le_bytes());
                hash.update(state.source_template_id.to_le_bytes());
                hash.update(state.current_state_template_id.to_le_bytes());
                hash_digest(&mut hash, state.source_semantic_digest);
                hash_digest(&mut hash, state.current_state_semantic_digest);
                hash.update(state.momentum_sign.to_le_bytes());
                hash_exact(&mut hash, state.crossing_phase);
                hash.update(state.spin_state.to_le_bytes());
                hash.update(state.chirality.to_le_bytes());
                hash_len(&mut hash, state.flavour_flow.len(), "source flavour flow")?;
                for flavour in &state.flavour_flow {
                    hash.update(flavour.to_le_bytes());
                }
                hash.update(state.quantum_number_flow_id.to_le_bytes());
                hash_digest(&mut hash, state.color_seed_proof_digest);
                hash.update([state.source_family as u8, state.source_orientation as u8]);
                match state.prepared_mass_parameter_slot {
                    None => hash.update([0]),
                    Some(slot) => {
                        hash.update([1]);
                        hash.update(slot.to_le_bytes());
                    }
                }
            }
        }
        hash_len(
            &mut hash,
            self.external_permutation.len(),
            "external permutation",
        )?;
        for slot in &self.external_permutation {
            hash.update(slot.to_le_bytes());
        }
        hash.update([self.coupling_order_policy as u8]);
        hash_len(
            &mut hash,
            self.coupling_hierarchies.len(),
            "coupling hierarchies",
        )?;
        for (hierarchy, limit) in self
            .coupling_hierarchies
            .iter()
            .zip(self.coupling_limits.iter())
        {
            hash.update(hierarchy.to_le_bytes());
            match limit {
                None => hash.update([0]),
                Some(limit) => {
                    hash.update([1]);
                    hash.update(limit.to_le_bytes());
                }
            }
        }
        hash_len(&mut hash, self.pairing_classes.len(), "pairing classes")?;
        for pairing_class in &self.pairing_classes {
            hash_digest(&mut hash, pairing_class.semantic_digest);
        }
        final_digest(hash)
    }

    pub(crate) const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    pub(crate) const fn process_digest(&self) -> SemanticDigest {
        self.process_digest
    }

    pub(crate) fn identity(&self) -> OnTheFlyProcessSeedIdentityV1 {
        OnTheFlyProcessSeedIdentityV1 {
            abi: ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI.to_owned(),
            process_digest: self.process_digest().to_string(),
            compiled_model_digest: self.model_digest().to_string(),
            recurrence_template_catalog_digest: self.template_catalog_digest().to_string(),
            prepared_kernel_pack_digest: self.prepared_pack_digest().to_string(),
            recurrence_direct_template_catalog_digest: self.direct_catalog_digest().to_string(),
            semantic_digest: self.semantic_digest().to_string(),
            external_permutation: self.external_permutation().to_vec(),
            external_sources: self
                .source_anchors()
                .iter()
                .map(|anchor| OnTheFlyProcessSeedSourceIdentityV1 {
                    source_slot: anchor.source_slot(),
                    public_label: anchor.external_label(),
                    is_initial: anchor.is_initial(),
                    states: anchor
                        .states()
                        .iter()
                        .map(|state| OnTheFlyProcessSeedStateIdentityV1 {
                            state_index: state.state_index(),
                            public_helicity: state.public_helicity(),
                            prepared_mass_parameter_slot: state.prepared_mass_parameter_slot(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    pub(crate) fn external_permutation(&self) -> &[u32] {
        &self.external_permutation
    }

    pub(crate) fn source_anchors(&self) -> &[OnTheFlySourceAnchorV1] {
        &self.source_anchors
    }

    pub(crate) fn pairing_classes(&self) -> &[OnTheFlyPairingClassV1] {
        &self.pairing_classes
    }

    pub(crate) const fn coupling_order_policy(&self) -> OnTheFlyCouplingOrderPolicyV1 {
        self.coupling_order_policy
    }

    pub(crate) fn coupling_hierarchies(&self) -> &[u32] {
        &self.coupling_hierarchies
    }

    pub(crate) fn explicit_coupling_limits(&self) -> &[Option<u32>] {
        &self.coupling_limits
    }

    pub(crate) const fn model_digest(&self) -> SemanticDigest {
        self.model_digest
    }

    pub(crate) const fn template_catalog_digest(&self) -> SemanticDigest {
        self.template_catalog_digest
    }

    pub(crate) const fn prepared_pack_digest(&self) -> SemanticDigest {
        self.prepared_pack_digest
    }

    pub(crate) const fn direct_catalog_digest(&self) -> SemanticDigest {
        self.direct_catalog_digest
    }

    pub(crate) fn source_execution_specs(
        &self,
    ) -> impl Iterator<Item = OnTheFlySourceExecutionSpecV1> + '_ {
        self.source_anchors.iter().flat_map(|anchor| {
            anchor
                .states
                .iter()
                .map(move |state| OnTheFlySourceExecutionSpecV1 {
                    source_slot: anchor.source_slot,
                    source_template_id: state.source_template_id,
                    spin_state_class: state.spin_state,
                    family: state.source_family,
                    orientation: state.source_orientation,
                    helicity: state.source_helicity,
                    chirality: state.chirality,
                    prepared_mass_parameter_slot: state.prepared_mass_parameter_slot,
                })
        })
    }
}

#[cfg(test)]
pub(crate) fn scalar_adapter_test_seed(
    model_digest: SemanticDigest,
    template_catalog_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_catalog_digest: SemanticDigest,
) -> RusticolResult<OnTheFlyProcessSeedV1> {
    let source_semantic_digest = SemanticDigest::new([5; 32])?;
    let current_semantic_digest = SemanticDigest::new([4; 32])?;
    let color_seed_digest = SemanticDigest::new([17; 32])?;
    let anchors = (0..2)
        .map(|source_slot| {
            let state = OnTheFlySourceStateV1::new(
                0,
                0,
                0,
                0,
                0,
                source_semantic_digest,
                current_semantic_digest,
                1,
                ExactComplexRational::ONE,
                50_000,
                0,
                vec![1],
                0,
                color_seed_digest,
                OnTheFlySourceWavefunctionFamilyV1::Scalar,
                OnTheFlySourceOrientationV1::SelfConjugate,
                None,
            )?;
            OnTheFlySourceAnchorV1::new(
                source_slot,
                source_slot,
                false,
                OnTheFlyExternalColorRoleV1::Singlet,
                false,
                None,
                vec![state],
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    OnTheFlyProcessSeedV1::new(
        SemanticDigest::new([91; 32])?,
        model_digest,
        template_catalog_digest,
        prepared_pack_digest,
        direct_catalog_digest,
        SemanticDigest::new([92; 32])?,
        "raw-amplitude-test",
        ExactComplexRational::ONE,
        anchors,
        vec![0, 1],
        OnTheFlyCouplingOrderPolicyV1::Explicit,
        vec![1],
        vec![Some(0)],
        Vec::new(),
    )
}

pub(super) fn validate_permutation(
    values: &[u32],
    expected_len: usize,
    label: &str,
) -> RusticolResult<()> {
    if values.len() != expected_len {
        return Err(invalid(format!(
            "{label} has length {}, expected {expected_len}",
            values.len(),
        )));
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    if sorted
        != (0..expected_len)
            .map(|value| checked_u32(value, label))
            .collect::<RusticolResult<Vec<_>>>()?
    {
        return Err(invalid(format!("{label} is not a bijection")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rebuild_with_coupling_policy(
        seed: OnTheFlyProcessSeedV1,
        coupling_hierarchies: Vec<u32>,
        coupling_limits: Vec<Option<u32>>,
    ) -> RusticolResult<OnTheFlyProcessSeedV1> {
        let OnTheFlyProcessSeedV1 {
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            source_anchors,
            external_permutation,
            coupling_order_policy,
            coupling_hierarchies: _,
            coupling_limits: _,
            pairing_classes,
            semantic_digest: _,
        } = seed;
        OnTheFlyProcessSeedV1::new(
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            ExactComplexRational::ONE,
            source_anchors.into_vec(),
            external_permutation.into_vec(),
            coupling_order_policy,
            coupling_hierarchies,
            coupling_limits,
            pairing_classes.into_vec(),
        )
    }

    #[test]
    fn genuine_probe_explicit_model_order_limits_fail_closed() {
        let seed = scalar_adapter_test_seed(
            SemanticDigest::new([1; 32]).unwrap(),
            SemanticDigest::new([2; 32]).unwrap(),
            SemanticDigest::new([3; 32]).unwrap(),
            SemanticDigest::new([4; 32]).unwrap(),
        )
        .unwrap();
        for (hierarchies, limits) in [
            (vec![], vec![]),
            (vec![1], vec![Some(0), None]),
            (vec![1, 0], vec![Some(0), None]),
        ] {
            assert!(
                rebuild_with_coupling_policy(seed.clone(), hierarchies, limits)
                    .unwrap_err()
                    .to_string()
                    .contains("one positive hierarchy and optional hard cap")
            );
        }
        assert!(rebuild_with_coupling_policy(seed.clone(), vec![1], vec![Some(3)]).is_ok());
        assert!(
            rebuild_with_coupling_policy(seed.clone(), vec![1, 2], vec![Some(0), None]).is_ok()
        );
        assert!(rebuild_with_coupling_policy(seed, vec![1, 2], vec![Some(0), Some(3)]).is_ok());
    }
}
