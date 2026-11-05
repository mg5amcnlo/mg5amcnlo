// SPDX-License-Identifier: 0BSD

use super::source_seed::validate_permutation;
use super::*;

/// One selected source state in a decoded query.  Source slots are always in
/// construction order after applying the seed's authenticated gather map.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct OnTheFlySelectedSourceV1 {
    pub(crate) source_slot: u32,
    pub(crate) state_index: u32,
    pub(crate) public_helicity: i32,
}

/// One already-decoded public LC selector.  Slots use public external order;
/// [`DecodedLcQueryV1::new`] authenticates and maps them to construction order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OnTheFlyLcSelectorV1 {
    Singlet,
    SingleTrace { public_word: Box<[u32]> },
    OpenLines { public_blocks: Box<[Box<[u32]>]> },
}

impl OnTheFlyLcSelectorV1 {
    pub(crate) fn single_trace(public_word: Vec<u32>) -> Self {
        Self::SingleTrace {
            public_word: public_word.into_boxed_slice(),
        }
    }

    pub(crate) fn open_lines(public_blocks: Vec<Vec<u32>>) -> Self {
        Self::OpenLines {
            public_blocks: public_blocks
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

/// One exact physical LC selector.  It contains only the selected structural
/// values; selector identity is their digest and never a process-wide flow ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedLcQueryV1 {
    pub(super) seed_digest: SemanticDigest,
    pub(super) external_permutation: Box<[u32]>,
    pub(super) selected_sources: Box<[OnTheFlySelectedSourceV1]>,
    pub(super) target_components: Box<[LCColorComponent]>,
    pub(super) closure_anchor_slot: u32,
    pub(super) selector_digest: SemanticDigest,
    pub(super) semantic_digest: SemanticDigest,
}

impl DecodedLcQueryV1 {
    pub(crate) fn new(
        seed: &OnTheFlyProcessSeedV1,
        external_permutation: Vec<u32>,
        public_helicities: &[i32],
        selector: OnTheFlyLcSelectorV1,
    ) -> RusticolResult<Self> {
        if external_permutation.as_slice() != seed.external_permutation.as_ref() {
            return Err(integrity(
                "query external permutation differs from its compact seed",
            ));
        }
        validate_permutation(
            &external_permutation,
            seed.source_anchors.len(),
            "query external permutation",
        )?;
        if public_helicities.len() != seed.source_anchors.len() {
            return Err(invalid(
                "public-helicity selector does not cover every external slot exactly once",
            ));
        }
        let inverse_permutation = inverse_permutation(&external_permutation)?;
        let selected_sources = selected_sources(seed, public_helicities)?;
        let decoded = decode_selector(seed, &inverse_permutation, selector)?;
        let seed_digest = seed.semantic_digest();
        let mut query = Self {
            seed_digest,
            external_permutation: external_permutation.into_boxed_slice(),
            selected_sources: selected_sources.into_boxed_slice(),
            target_components: decoded.target_components.into_boxed_slice(),
            closure_anchor_slot: decoded.closure_anchor_slot,
            selector_digest: decoded.selector_digest,
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        query.semantic_digest = query.compute_digest()?;
        Ok(query)
    }

    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(b"pyamplicol-on-the-fly-decoded-lc-query-v2\0");
        hash_digest(&mut hash, self.seed_digest);
        hash_digest(&mut hash, self.selector_digest);
        hash_len(
            &mut hash,
            self.external_permutation.len(),
            "query external permutation",
        )?;
        for source_slot in &self.external_permutation {
            hash.update(source_slot.to_le_bytes());
        }
        hash.update(self.closure_anchor_slot.to_le_bytes());
        hash_len(&mut hash, self.selected_sources.len(), "selected sources")?;
        for selected in &self.selected_sources {
            hash.update(selected.source_slot.to_le_bytes());
            hash.update(selected.state_index.to_le_bytes());
            hash.update(selected.public_helicity.to_le_bytes());
        }
        hash_len(&mut hash, self.target_components.len(), "target components")?;
        for component in &self.target_components {
            hash.update([component.kind() as u8]);
            hash_len(&mut hash, component.source_slots().len(), "component word")?;
            for slot in component.source_slots() {
                hash.update(slot.to_le_bytes());
            }
        }
        final_digest(hash)
    }

    pub(crate) const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    pub(crate) const fn process_seed_digest(&self) -> SemanticDigest {
        self.seed_digest
    }

    fn with_closure_anchor_slot(mut self, closure_anchor_slot: u32) -> RusticolResult<Self> {
        if !self
            .selected_sources
            .iter()
            .any(|source| source.source_slot == closure_anchor_slot)
        {
            return Err(integrity(
                "certified query closure anchor is outside its selected source domain",
            ));
        }
        if self.closure_anchor_slot != closure_anchor_slot {
            self.closure_anchor_slot = closure_anchor_slot;
            self.semantic_digest = self.compute_digest()?;
        }
        Ok(self)
    }
}

/// Authenticated private choice of closure ancestry for one prepared process.
///
/// The ordinary selector endpoint remains authoritative for every ineligible
/// generic, open-line, singlet, or heterogeneous shape.  Residual/cross is a
/// downstream color-metric classification, so eligible pure-adjoint trace
/// groups use the cyclic minimum regardless of that grouping.  A prepared
/// grammar enables it only after proving the same homogeneous adjoint state
/// and direct binary closure contract used by recurrence v4 sectors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OnTheFlyClosureAnchorPolicyV1 {
    cyclic_trace_minimum_source_slot: Option<u32>,
}

impl OnTheFlyClosureAnchorPolicyV1 {
    pub(super) const fn selector_endpoint() -> Self {
        Self {
            cyclic_trace_minimum_source_slot: None,
        }
    }

    pub(super) const fn certified_cyclic_minimum(source_slot: u32) -> Self {
        Self {
            cyclic_trace_minimum_source_slot: Some(source_slot),
        }
    }

    pub(super) fn closure_anchor_for_components(
        self,
        seed: &OnTheFlyProcessSeedV1,
        components: &[LCColorComponent],
    ) -> Option<u32> {
        let selector_endpoint = physical_lc_selector_closure_anchor(seed, components)?;
        let Some(minimum) = self.cyclic_trace_minimum_source_slot else {
            return Some(selector_endpoint);
        };
        let [component] = components else {
            return Some(selector_endpoint);
        };
        if component.kind() != LCColorComponentKind::Trace
            || component.source_slots().len() != seed.source_anchors.len()
            || seed
                .source_anchors
                .iter()
                .any(|anchor| anchor.color_role != OnTheFlyExternalColorRoleV1::Adjoint)
            || component.source_slots().iter().copied().min() != Some(minimum)
        {
            return Some(selector_endpoint);
        }
        Some(minimum)
    }

    pub(super) fn canonicalize_query(
        self,
        seed: &OnTheFlyProcessSeedV1,
        query: DecodedLcQueryV1,
    ) -> RusticolResult<DecodedLcQueryV1> {
        let closure_anchor_slot = self
            .closure_anchor_for_components(seed, &query.target_components)
            .ok_or_else(|| integrity("decoded query lost its physical selector shape"))?;
        query.with_closure_anchor_slot(closure_anchor_slot)
    }

    pub(super) fn authenticate_canonical_query(
        self,
        seed: &OnTheFlyProcessSeedV1,
        query: &DecodedLcQueryV1,
    ) -> RusticolResult<()> {
        if &self.canonicalize_query(seed, query.clone())? != query {
            return Err(integrity(
                "decoded query was not canonicalized at its prepared-grammar boundary",
            ));
        }
        Ok(())
    }

    pub(super) fn certifies_cyclic_trace_reflection(
        self,
        seed: &OnTheFlyProcessSeedV1,
        query: &DecodedLcQueryV1,
    ) -> bool {
        let Some(minimum) = self.cyclic_trace_minimum_source_slot else {
            return false;
        };
        let [component] = query.target_components.as_ref() else {
            return false;
        };
        query.seed_digest == seed.semantic_digest()
            && query.closure_anchor_slot == minimum
            && component.kind() == LCColorComponentKind::Trace
            && component.source_slots().len() == seed.source_anchors.len()
            && component.source_slots().iter().copied().min() == Some(minimum)
            && seed
                .source_anchors
                .iter()
                .all(|anchor| anchor.color_role == OnTheFlyExternalColorRoleV1::Adjoint)
            && self.closure_anchor_for_components(seed, &query.target_components) == Some(minimum)
    }
}

struct DecodedSelector {
    target_components: Vec<LCColorComponent>,
    closure_anchor_slot: u32,
    selector_digest: SemanticDigest,
}

fn selected_sources(
    seed: &OnTheFlyProcessSeedV1,
    public_helicities: &[i32],
) -> RusticolResult<Vec<OnTheFlySelectedSourceV1>> {
    seed.source_anchors
        .iter()
        .map(|anchor| {
            let public_slot = *seed
                .external_permutation
                .get(anchor.source_slot as usize)
                .ok_or_else(|| integrity("external permutation source slot is absent"))?;
            let public_helicity = public_helicities[public_slot as usize];
            let matches = anchor
                .states
                .iter()
                .filter(|state| state.public_helicity == public_helicity)
                .collect::<Vec<_>>();
            let [state] = matches.as_slice() else {
                return Err(invalid(format!(
                    "source slot {} has {} states for public helicity {public_helicity}, expected exactly one",
                    anchor.source_slot,
                    matches.len(),
                )));
            };
            Ok(OnTheFlySelectedSourceV1 {
                source_slot: anchor.source_slot,
                state_index: state.state_index,
                public_helicity,
            })
        })
        .collect()
}

fn inverse_permutation(permutation: &[u32]) -> RusticolResult<Vec<u32>> {
    let mut inverse = vec![MISSING_U32; permutation.len()];
    for (construction_slot, public_slot) in permutation.iter().copied().enumerate() {
        let entry = inverse
            .get_mut(public_slot as usize)
            .ok_or_else(|| invalid("external permutation public slot is out of bounds"))?;
        if *entry != MISSING_U32 {
            return Err(invalid("external permutation repeats a public slot"));
        }
        *entry = checked_u32(construction_slot, "construction source slot")?;
    }
    if inverse.contains(&MISSING_U32) {
        return Err(invalid("external permutation omits a public slot"));
    }
    Ok(inverse)
}

fn map_public_word(word: &[u32], inverse: &[u32], label: &str) -> RusticolResult<Vec<u32>> {
    if word.is_empty() {
        return Err(invalid(format!("{label} is empty")));
    }
    word.iter()
        .map(|public_slot| {
            inverse
                .get(*public_slot as usize)
                .copied()
                .ok_or_else(|| invalid(format!("{label} public slot is out of bounds")))
        })
        .collect()
}

fn colored_source_slots(seed: &OnTheFlyProcessSeedV1) -> BTreeSet<u32> {
    seed.source_anchors
        .iter()
        .filter(|anchor| anchor.color_role.is_colored())
        .map(|anchor| anchor.source_slot)
        .collect()
}

fn require_exact_colored_coverage(
    seed: &OnTheFlyProcessSeedV1,
    words: &[Vec<u32>],
) -> RusticolResult<()> {
    let mut observed = BTreeSet::new();
    for slot in words.iter().flatten().copied() {
        if !observed.insert(slot) {
            return Err(invalid("decoded selector repeats a colored source slot"));
        }
    }
    if observed != colored_source_slots(seed) {
        return Err(invalid(
            "decoded selector does not cover every colored source exactly once",
        ));
    }
    Ok(())
}

/// Whether closed LC components form any physical selector admitted by this
/// compact process seed.  This is deliberately selector-independent: the
/// cold minimal-coupling sweep uses it to prove that a structural closure can
/// feed at least one public query without constructing a public-flow table.
pub(super) fn components_match_physical_lc_selector(
    seed: &OnTheFlyProcessSeedV1,
    components: &[LCColorComponent],
) -> bool {
    let expected = colored_source_slots(seed);
    let mut observed = BTreeSet::new();
    if components
        .iter()
        .flat_map(|component| component.source_slots().iter().copied())
        .any(|slot| !observed.insert(slot))
        || observed != expected
    {
        return false;
    }

    if expected.is_empty() {
        return components.is_empty() && seed.pairing_classes.is_empty();
    }
    if seed.pairing_classes.is_empty() {
        return components.len() == 1
            && components[0].kind() == LCColorComponentKind::Trace
            && components[0].source_slots().iter().all(|slot| {
                seed.source_anchors[*slot as usize].color_role
                    == OnTheFlyExternalColorRoleV1::Adjoint
            });
    }
    !components.is_empty()
        && components.iter().all(|component| {
            let slots = component.source_slots();
            component.kind() == LCColorComponentKind::OpenString
                && slots.len() >= 2
                && seed.source_anchors[slots[0] as usize].color_role
                    == OnTheFlyExternalColorRoleV1::Fundamental
                && seed.source_anchors[*slots.last().expect("nonempty") as usize].color_role
                    == OnTheFlyExternalColorRoleV1::Antifundamental
                && slots[1..slots.len() - 1].iter().all(|slot| {
                    seed.source_anchors[*slot as usize].color_role
                        == OnTheFlyExternalColorRoleV1::Adjoint
                })
        })
}

/// Recover the same private closure anchor that query decoding chooses for a
/// valid physical selector.  `None` means the components are not a selector
/// admitted by this process seed.
pub(super) fn physical_lc_selector_closure_anchor(
    seed: &OnTheFlyProcessSeedV1,
    components: &[LCColorComponent],
) -> Option<u32> {
    if !components_match_physical_lc_selector(seed, components) {
        return None;
    }
    if components.is_empty() {
        return Some(
            seed.source_anchors
                .iter()
                .find(|anchor| anchor.is_fermionic)
                .unwrap_or(&seed.source_anchors[0])
                .source_slot,
        );
    }
    if seed.pairing_classes.is_empty() {
        return components[0].source_slots().last().copied();
    }

    let minimum_colored_slot = colored_source_slots(seed).into_iter().next()?;
    let mut blocks = components
        .iter()
        .map(LCColorComponent::source_slots)
        .collect::<Vec<_>>();
    blocks.sort_unstable_by(|left, right| left[0].cmp(&right[0]).then_with(|| left.cmp(right)));
    let first = blocks
        .iter()
        .position(|block| block.contains(&minimum_colored_slot))?;
    blocks.rotate_left(first);
    blocks.last()?.last().copied()
}

fn decode_selector(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    selector: OnTheFlyLcSelectorV1,
) -> RusticolResult<DecodedSelector> {
    match selector {
        OnTheFlyLcSelectorV1::Singlet => decode_singlet(seed),
        OnTheFlyLcSelectorV1::SingleTrace { public_word } => {
            decode_single_trace(seed, inverse, &public_word)
        }
        OnTheFlyLcSelectorV1::OpenLines { public_blocks } => {
            decode_open_lines(seed, inverse, &public_blocks)
        }
    }
}

fn decode_singlet(seed: &OnTheFlyProcessSeedV1) -> RusticolResult<DecodedSelector> {
    if seed
        .source_anchors
        .iter()
        .any(|anchor| anchor.color_role.is_colored())
    {
        return Err(invalid(
            "singlet selector is invalid for a process with colored external sources",
        ));
    }
    if !seed.pairing_classes.is_empty() {
        return Err(integrity(
            "singlet seed unexpectedly contains pairing classes",
        ));
    }
    let closure_anchor_slot = seed
        .source_anchors
        .iter()
        .find(|anchor| anchor.is_fermionic)
        .unwrap_or(&seed.source_anchors[0])
        .source_slot;
    let selector_digest = selector_digest(&[])?;
    if !components_match_physical_lc_selector(seed, &[]) {
        return Err(integrity(
            "singlet selector failed its physical-shape proof",
        ));
    }
    Ok(DecodedSelector {
        target_components: Vec::new(),
        closure_anchor_slot,
        selector_digest,
    })
}

fn decode_single_trace(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    public_word: &[u32],
) -> RusticolResult<DecodedSelector> {
    if !seed.pairing_classes.is_empty() {
        return Err(invalid(
            "single-trace selector is invalid for a seed with open-line pairing classes",
        ));
    }
    let word = map_public_word(public_word, inverse, "single-trace word")?;
    require_exact_colored_coverage(seed, std::slice::from_ref(&word))?;
    if word.iter().any(|slot| {
        seed.source_anchors[*slot as usize].color_role != OnTheFlyExternalColorRoleV1::Adjoint
    }) {
        return Err(invalid(
            "single-trace selector contains a non-adjoint external source",
        ));
    }
    let component = LCColorComponent::new(LCColorComponentKind::Trace, word)?;
    if !components_match_physical_lc_selector(seed, std::slice::from_ref(&component)) {
        return Err(integrity(
            "single-trace selector failed its physical-shape proof",
        ));
    }
    let closure_anchor_slot = *component
        .source_slots()
        .last()
        .ok_or_else(|| invalid("single-trace word is empty"))?;
    let selector_digest = selector_digest(std::slice::from_ref(&component))?;
    Ok(DecodedSelector {
        target_components: vec![component],
        closure_anchor_slot,
        selector_digest,
    })
}

fn decode_open_lines(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    public_blocks: &[Box<[u32]>],
) -> RusticolResult<DecodedSelector> {
    if seed.pairing_classes.is_empty() || public_blocks.is_empty() {
        return Err(invalid(
            "open-line selector requires pairing classes and nonempty blocks",
        ));
    }
    let mut blocks = public_blocks
        .iter()
        .map(|word| map_public_word(word, inverse, "open-line block"))
        .collect::<RusticolResult<Vec<_>>>()?;
    require_exact_colored_coverage(seed, &blocks)?;
    for block in &blocks {
        if block.len() < 2
            || seed.source_anchors[block[0] as usize].color_role
                != OnTheFlyExternalColorRoleV1::Fundamental
            || seed.source_anchors[*block.last().expect("nonempty") as usize].color_role
                != OnTheFlyExternalColorRoleV1::Antifundamental
            || block[1..block.len() - 1].iter().any(|slot| {
                seed.source_anchors[*slot as usize].color_role
                    != OnTheFlyExternalColorRoleV1::Adjoint
            })
        {
            return Err(invalid(
                "open-line block must be fundamental, zero or more adjoints, antifundamental",
            ));
        }
    }

    // Public callers may enumerate the same tensor product in any block
    // order.  Map into construction slots first, then recover the canonical
    // product order used by the established LC builder before choosing its
    // cyclic closure anchor.  Reversing a block remains a distinct (and
    // invalid above) selector; only whole-block enumeration order is erased.
    blocks.sort_unstable_by(|left, right| left[0].cmp(&right[0]).then_with(|| left.cmp(right)));

    let minimum_colored_slot = colored_source_slots(seed)
        .into_iter()
        .next()
        .ok_or_else(|| invalid("open-line selector has no colored source"))?;
    let first_block = blocks
        .iter()
        .position(|block| block.contains(&minimum_colored_slot))
        .ok_or_else(|| integrity("minimum colored source is absent from open-line blocks"))?;
    blocks.rotate_left(first_block);
    let closure_anchor_slot = *blocks
        .last()
        .and_then(|block| block.last())
        .ok_or_else(|| invalid("open-line selector has no closure anchor"))?;
    let mut components = blocks
        .into_iter()
        .map(|block| LCColorComponent::new(LCColorComponentKind::OpenString, block))
        .collect::<RusticolResult<Vec<_>>>()?;
    components.sort_unstable();
    if !components_match_physical_lc_selector(seed, &components) {
        return Err(integrity(
            "open-line selector failed its physical-shape proof",
        ));
    }
    let selector_digest = selector_digest(&components)?;
    Ok(DecodedSelector {
        target_components: components,
        closure_anchor_slot,
        selector_digest,
    })
}

fn selector_digest(components: &[LCColorComponent]) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-structural-selector-v1\0");
    hash_len(&mut hash, components.len(), "selector components")?;
    for component in components {
        hash.update([component.kind() as u8]);
        hash_len(&mut hash, component.source_slots().len(), "selector word")?;
        for source_slot in component.source_slots() {
            hash.update(source_slot.to_le_bytes());
        }
    }
    final_digest(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn anchor(
        source_slot: u32,
        role: OnTheFlyExternalColorRoleV1,
        contract: Option<SemanticDigest>,
    ) -> OnTheFlySourceAnchorV1 {
        let fermionic = role.is_pairing_endpoint();
        let family = if fermionic {
            OnTheFlySourceWavefunctionFamilyV1::DiracFermion
        } else {
            OnTheFlySourceWavefunctionFamilyV1::Vector
        };
        let orientation = match role {
            OnTheFlyExternalColorRoleV1::Fundamental => OnTheFlySourceOrientationV1::Particle,
            OnTheFlyExternalColorRoleV1::Antifundamental => {
                OnTheFlySourceOrientationV1::Antiparticle
            }
            OnTheFlyExternalColorRoleV1::Singlet | OnTheFlyExternalColorRoleV1::Adjoint => {
                OnTheFlySourceOrientationV1::SelfConjugate
            }
        };
        let state = OnTheFlySourceStateV1::new(
            0,
            1,
            1,
            source_slot,
            source_slot,
            digest(20 + source_slot as u8),
            digest(40 + source_slot as u8),
            1,
            ExactComplexRational::ONE,
            1,
            0,
            vec![source_slot as i32 + 1],
            source_slot,
            digest(60 + source_slot as u8),
            family,
            orientation,
            None,
        )
        .unwrap();
        OnTheFlySourceAnchorV1::new(
            source_slot,
            100 + source_slot,
            false,
            role,
            fermionic,
            contract,
            vec![state],
        )
        .unwrap()
    }

    fn seed(
        roles: &[OnTheFlyExternalColorRoleV1],
        external_permutation: Vec<u32>,
    ) -> OnTheFlyProcessSeedV1 {
        let anchors = roles
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, role)| {
                let contract = role.is_pairing_endpoint().then(|| digest(100 + slot as u8));
                anchor(slot as u32, role, contract)
            })
            .collect::<Vec<_>>();
        let fundamental = anchors
            .iter()
            .filter(|anchor| anchor.color_role == OnTheFlyExternalColorRoleV1::Fundamental)
            .map(|anchor| OnTheFlyPairingEndpointV1 {
                source_slot: anchor.source_slot,
                source_contract_digest: anchor.pairing_source_contract_digest.unwrap(),
            })
            .collect::<Vec<_>>();
        let antifundamental = anchors
            .iter()
            .filter(|anchor| anchor.color_role == OnTheFlyExternalColorRoleV1::Antifundamental)
            .map(|anchor| OnTheFlyPairingEndpointV1 {
                source_slot: anchor.source_slot,
                source_contract_digest: anchor.pairing_source_contract_digest.unwrap(),
            })
            .collect::<Vec<_>>();
        let classes = if fundamental.is_empty() {
            Vec::new()
        } else {
            vec![
                OnTheFlyPairingClassV1::new("q", digest(88), fundamental, antifundamental).unwrap(),
            ]
        };
        OnTheFlyProcessSeedV1::new(
            digest(1),
            digest(2),
            digest(3),
            digest(4),
            digest(5),
            digest(6),
            "raw amplitude",
            ExactComplexRational::ONE,
            anchors,
            external_permutation,
            OnTheFlyCouplingOrderPolicyV1::Explicit,
            vec![1],
            vec![Some(2)],
            classes,
        )
        .unwrap()
    }

    #[test]
    fn inverse_permutation_binds_public_and_construction_slots_exactly() {
        assert_eq!(inverse_permutation(&[2, 0, 3, 1]).unwrap(), [1, 3, 0, 2]);
        assert!(inverse_permutation(&[0, 0]).is_err());
        assert!(inverse_permutation(&[0, 2]).is_err());
    }

    #[test]
    fn selector_domains_are_derived_from_external_roles() {
        let singlet_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Singlet,
                OnTheFlyExternalColorRoleV1::Singlet,
            ],
            vec![1, 0],
        );
        let singlet = DecodedLcQueryV1::new(
            &singlet_seed,
            vec![1, 0],
            &[1, 1],
            OnTheFlyLcSelectorV1::Singlet,
        )
        .unwrap();
        assert!(singlet.target_components.is_empty());
        assert_eq!(singlet.closure_anchor_slot, 0);

        let trace_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
            ],
            vec![1, 0],
        );
        assert!(
            DecodedLcQueryV1::new(
                &trace_seed,
                vec![1, 0],
                &[1, 1],
                OnTheFlyLcSelectorV1::Singlet,
            )
            .is_err()
        );
        let trace = DecodedLcQueryV1::new(
            &trace_seed,
            vec![1, 0],
            &[1, 1],
            OnTheFlyLcSelectorV1::single_trace(vec![1, 0]),
        )
        .unwrap();
        assert_eq!(trace.target_components[0].source_slots(), [0, 1]);
        assert_eq!(trace.closure_anchor_slot, 1);
    }

    #[test]
    fn single_trace_anchor_is_cyclic_invariant_but_reversal_remains_distinct() {
        let trace_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
            ],
            vec![0, 1, 2],
        );
        let decode = |word| {
            DecodedLcQueryV1::new(
                &trace_seed,
                vec![0, 1, 2],
                &[1, 1, 1],
                OnTheFlyLcSelectorV1::single_trace(word),
            )
            .unwrap()
        };
        let canonical = decode(vec![0, 1, 2]);
        let rotated = decode(vec![1, 2, 0]);
        let reversed = decode(vec![0, 2, 1]);
        assert_eq!(canonical.target_components, rotated.target_components);
        assert_eq!(canonical.closure_anchor_slot, rotated.closure_anchor_slot);
        assert_eq!(canonical.selector_digest, rotated.selector_digest);
        assert_eq!(canonical.semantic_digest(), rotated.semantic_digest());
        assert_ne!(canonical.target_components, reversed.target_components);
        assert_ne!(canonical.selector_digest, reversed.selector_digest);
        assert_ne!(canonical.closure_anchor_slot, reversed.closure_anchor_slot);
    }

    #[test]
    fn certified_cyclic_policy_shares_one_anchor_without_touching_generic_shapes() {
        let trace_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
            ],
            vec![0, 1, 2, 3],
        );
        let decode_trace = |word| {
            DecodedLcQueryV1::new(
                &trace_seed,
                vec![0, 1, 2, 3],
                &[1, 1, 1, 1],
                OnTheFlyLcSelectorV1::single_trace(word),
            )
            .unwrap()
        };
        let queries = [
            vec![0, 1, 2, 3],
            vec![0, 1, 3, 2],
            vec![0, 2, 1, 3],
            vec![0, 2, 3, 1],
            vec![0, 3, 1, 2],
            vec![0, 3, 2, 1],
        ]
        .map(decode_trace);
        assert_eq!(
            queries
                .iter()
                .map(|query| query.closure_anchor_slot)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 2, 3]),
        );

        let fallback = OnTheFlyClosureAnchorPolicyV1::selector_endpoint();
        for query in &queries {
            assert_eq!(
                fallback
                    .canonicalize_query(&trace_seed, query.clone())
                    .unwrap(),
                query.clone(),
            );
        }
        let fixed = OnTheFlyClosureAnchorPolicyV1::certified_cyclic_minimum(0);
        let error = fixed
            .authenticate_canonical_query(&trace_seed, &queries[0])
            .unwrap_err();
        assert!(error.to_string().contains("prepared-grammar boundary"));
        let fixed_queries =
            queries.map(|query| fixed.canonicalize_query(&trace_seed, query).unwrap());
        assert!(
            fixed_queries
                .iter()
                .all(|query| query.closure_anchor_slot == 0)
        );
        for query in &fixed_queries {
            fixed
                .authenticate_canonical_query(&trace_seed, query)
                .unwrap();
            assert!(fixed.certifies_cyclic_trace_reflection(&trace_seed, query));
            assert!(!fallback.certifies_cyclic_trace_reflection(&trace_seed, query));
        }
        assert_eq!(
            fixed_queries
                .iter()
                .map(DecodedLcQueryV1::semantic_digest)
                .collect::<BTreeSet<_>>()
                .len(),
            6,
        );

        let open_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Fundamental,
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Antifundamental,
            ],
            vec![0, 1, 2],
        );
        let open = DecodedLcQueryV1::new(
            &open_seed,
            vec![0, 1, 2],
            &[1, 1, 1],
            OnTheFlyLcSelectorV1::open_lines(vec![vec![0, 1, 2]]),
        )
        .unwrap();
        assert_eq!(open.closure_anchor_slot, 2);
        assert_eq!(
            fixed.canonicalize_query(&open_seed, open.clone()).unwrap(),
            open,
        );
        assert!(!fixed.certifies_cyclic_trace_reflection(&open_seed, &open));
    }

    #[test]
    fn open_line_selector_binds_only_the_selected_color_tensor() {
        let open_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Fundamental,
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Antifundamental,
            ],
            vec![2, 0, 1],
        );
        let query = DecodedLcQueryV1::new(
            &open_seed,
            vec![2, 0, 1],
            &[1, 1, 1],
            OnTheFlyLcSelectorV1::open_lines(vec![vec![2, 0, 1]]),
        )
        .unwrap();
        assert_eq!(query.target_components[0].source_slots(), [0, 1, 2]);
        assert_eq!(query.closure_anchor_slot, 2);
    }

    #[test]
    fn multi_open_line_block_order_is_canonical_after_public_alias_permutation() {
        let roles = [
            OnTheFlyExternalColorRoleV1::Fundamental,
            OnTheFlyExternalColorRoleV1::Adjoint,
            OnTheFlyExternalColorRoleV1::Antifundamental,
            OnTheFlyExternalColorRoleV1::Fundamental,
            OnTheFlyExternalColorRoleV1::Adjoint,
            OnTheFlyExternalColorRoleV1::Antifundamental,
        ];
        // Construction slot -> public slot. The two physical open strings are
        // [0,1,2] and [3,4,5], hence [3,0,5] and [2,1,4] publicly.
        let permutation = vec![3, 0, 5, 2, 1, 4];
        let open_seed = seed(&roles, permutation.clone());
        let decode = |blocks| {
            DecodedLcQueryV1::new(
                &open_seed,
                permutation.clone(),
                &[1; 6],
                OnTheFlyLcSelectorV1::open_lines(blocks),
            )
            .unwrap()
        };
        let forward = decode(vec![vec![3, 0, 5], vec![2, 1, 4]]);
        let reversed_blocks = decode(vec![vec![2, 1, 4], vec![3, 0, 5]]);

        assert_eq!(forward.target_components, reversed_blocks.target_components);
        assert_eq!(forward.selector_digest, reversed_blocks.selector_digest);
        assert_eq!(forward.semantic_digest(), reversed_blocks.semantic_digest());
        assert_eq!(forward.closure_anchor_slot, 5);
        assert_eq!(
            forward
                .target_components
                .iter()
                .map(|component| component.source_slots())
                .collect::<Vec<_>>(),
            vec![&[0, 1, 2][..], &[3, 4, 5][..]],
        );
    }
}
