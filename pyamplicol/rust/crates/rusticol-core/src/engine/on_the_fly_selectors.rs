// SPDX-License-Identifier: 0BSD

//! Compact public-selector adapter for the private LC on-the-fly lane.
//!
//! The established runtime API addresses helicities as `h:+1,-1,...` and LC
//! components as `flow:<external labels>` (or `flow:singlet`).  This module
//! preserves those strings without loading a process-wide physics table.  It
//! snapshots only O(external-state) seed data and derives individual axis
//! members by ordinal when a selector or explicit introspection call needs
//! them.
//!
//! Complete LC axes are factorial for general processes.  Accordingly,
//! [`OnTheFlyLazySelectionV1`] is a lazy materializer: load, total evaluation,
//! profiling, and benchmarking need not create dense helicity/color metadata.
//! A caller which genuinely requests resolved metadata may explicitly
//! materialize it through [`OnTheFlySelectorIntrospectionCacheV1`].

use super::on_the_fly_lane::{OnTheFlyLcQueryRequestV1, OnTheFlyLcReductionTargetV1};
use super::on_the_fly_manifest::OnTheFlyColorBasis;
use crate::recurrence::on_the_fly::{
    DecodedLcQueryV1, OnTheFlyExternalColorRoleV1, OnTheFlyLcSelectorV1, OnTheFlyProcessSeedV1,
};
use crate::{RusticolError, RusticolResult};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::selector(format!("on-the-fly selector: {}", message.into()))
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(format!("on-the-fly selector: {}", message.into()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactExternalV1 {
    source_slot: u32,
    public_slot: u32,
    public_label: u32,
    is_initial: bool,
    color_role: OnTheFlyExternalColorRoleV1,
    public_helicities: Box<[i32]>,
}

/// Compact seed-derived external record.  Particle names and PDGs remain
/// owned by the existing process-selection metadata; this structural record
/// supplies the selector-relevant half of that existing API on demand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnTheFlyExternalSelectorRecordV1 {
    pub(super) source_slot: u32,
    pub(super) public_slot: u32,
    pub(super) public_label: u32,
    pub(super) is_initial: bool,
    pub(super) color_role: OnTheFlyExternalColorRoleV1,
}

/// One exact established public helicity ID, materialized only on request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnTheFlyHelicitySelectorRecordV1 {
    pub(super) id: String,
    pub(super) index: usize,
    pub(super) values: Box<[i32]>,
}

/// One exact established public LC color ID, materialized only on request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnTheFlyColorSelectorRecordV1 {
    pub(super) id: String,
    pub(super) index: usize,
    pub(super) word: Box<[u32]>,
    pub(super) selector: OnTheFlyLcSelectorV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompactColorDomainV1 {
    Singlet,
    SingleTrace,
    OpenLines,
}

/// Exact compact coverage marker supplied by generation.  Incomplete color
/// coverage would require retaining the selected dense flow list, defeating
/// this lane's process-seed contract, so it is rejected at construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OnTheFlyLcColorCoverageV1 {
    Complete,
    #[expect(
        dead_code,
        reason = "retained as the explicit rejected compatibility sentinel"
    )]
    Incomplete,
}

/// Small generation-owned selector policy which cannot be reconstructed from
/// external roles.  `reference_color_word` may reorder one ordinary complete
/// flow to the front; reflection folding changes pure-trace public ordering
/// while preserving both members as public aliases. Contracted adjoint colour
/// instead retains ordinary trace amplitudes with two fixed DDM endpoints.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OnTheFlyLcSelectorPolicyV1 {
    pub(super) color_coverage: OnTheFlyLcColorCoverageV1,
    pub(super) color_basis: OnTheFlyColorBasis,
    pub(super) reference_color_word: Option<Box<[u32]>>,
    pub(super) trace_reflections_folded: bool,
}

impl OnTheFlyLcSelectorPolicyV1 {
    #[cfg(test)]
    pub(super) fn complete(
        reference_color_word: Option<Vec<u32>>,
        trace_reflections_folded: bool,
    ) -> Self {
        Self {
            color_coverage: OnTheFlyLcColorCoverageV1::Complete,
            color_basis: OnTheFlyColorBasis::Trace,
            reference_color_word: reference_color_word.map(Vec::into_boxed_slice),
            trace_reflections_folded,
        }
    }
}

type OnTheFlySelectedOrdinalsV1 = (Option<Box<[usize]>>, Option<Box<[usize]>>);

/// O(external-state) selector owner retained beside the private native lane.
#[derive(Clone, Debug)]
pub(super) struct OnTheFlyCompactSelectorAdapterV1 {
    external_permutation: Box<[u32]>,
    representative_to_public: Box<[u32]>,
    public_to_representative: Box<[u32]>,
    public_label_by_representative_label: BTreeMap<u32, u32>,
    representative_label_by_public_label: BTreeMap<u32, u32>,
    externals_by_public_slot: Box<[CompactExternalV1]>,
    public_slot_by_label: BTreeMap<u32, usize>,
    fundamental_labels: Box<[u32]>,
    antifundamental_labels: Box<[u32]>,
    adjoint_labels: Box<[u32]>,
    color_domain: CompactColorDomainV1,
    color_basis: OnTheFlyColorBasis,
    reference_unfolded_color_ordinal: Option<usize>,
    trace_reflections_folded: bool,
    helicity_count: usize,
    color_count: usize,
}

impl OnTheFlyCompactSelectorAdapterV1 {
    /// Snapshot only selector-relevant seed data.  The accessors used here are
    /// intentionally narrow; they do not expose the recurrence catalogs or a
    /// dense public-flow table.
    pub(super) fn from_seed(
        seed: &OnTheFlyProcessSeedV1,
        policy: OnTheFlyLcSelectorPolicyV1,
    ) -> RusticolResult<Self> {
        if policy.color_coverage != OnTheFlyLcColorCoverageV1::Complete {
            return Err(invalid(
                "on-the-fly LC requires complete public color coverage",
            ));
        }
        let external_permutation = seed.external_permutation();
        let anchors = seed.source_anchors();
        if external_permutation.len() != anchors.len() {
            return Err(integrity(
                "external permutation and source-anchor domains differ",
            ));
        }

        let mut by_public = vec![None; anchors.len()];
        let mut public_slot_by_label = BTreeMap::new();
        for anchor in anchors {
            let public_slot = *external_permutation
                .get(anchor.source_slot() as usize)
                .ok_or_else(|| integrity("source anchor is outside the gather map"))?;
            let destination = by_public
                .get_mut(public_slot as usize)
                .ok_or_else(|| integrity("gather map public slot is out of bounds"))?;
            if destination.is_some() {
                return Err(integrity("gather map repeats a public external slot"));
            }
            if public_slot_by_label
                .insert(anchor.external_label(), public_slot as usize)
                .is_some()
            {
                return Err(integrity("compact seed repeats a public external label"));
            }
            let mut public_helicities = Vec::new();
            public_helicities
                .try_reserve_exact(anchor.states().len())
                .map_err(|error| invalid(format!("helicity-domain allocation failed: {error}")))?;
            for state in anchor.states() {
                let helicity = state.public_helicity();
                if public_helicities.contains(&helicity) {
                    return Err(integrity(format!(
                        "source slot {} repeats public helicity {helicity}",
                        anchor.source_slot(),
                    )));
                }
                public_helicities.push(helicity);
            }
            if public_helicities.is_empty() {
                return Err(integrity("source anchor has no public helicity"));
            }
            *destination = Some(CompactExternalV1 {
                source_slot: anchor.source_slot(),
                public_slot,
                public_label: anchor.external_label(),
                is_initial: anchor.is_initial(),
                color_role: anchor.color_role(),
                public_helicities: public_helicities.into_boxed_slice(),
            });
        }
        let externals_by_public_slot = by_public
            .into_iter()
            .map(|entry| entry.ok_or_else(|| integrity("gather map omits a public slot")))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();

        // Pairing classes are not color-flow pairings: they authenticate Wick
        // compatibility.  Nevertheless, checking their endpoint coverage here
        // ensures that every fundamental/antifundamental role used to split a
        // public flow is backed by the same compact seed contract.
        let mut pairing_endpoints = BTreeSet::new();
        for pairing_class in seed.pairing_classes() {
            for endpoint in pairing_class
                .fundamental_endpoints()
                .iter()
                .chain(pairing_class.antifundamental_endpoints())
            {
                if !pairing_endpoints.insert(endpoint.source_slot) {
                    return Err(integrity("pairing classes repeat an endpoint"));
                }
            }
        }

        let mut fundamental_labels = Vec::new();
        let mut antifundamental_labels = Vec::new();
        let mut adjoint_labels = Vec::new();
        let mut required_pairing_endpoints = BTreeSet::new();
        for external in externals_by_public_slot.iter() {
            match external.color_role {
                OnTheFlyExternalColorRoleV1::Singlet => {}
                OnTheFlyExternalColorRoleV1::Fundamental => {
                    fundamental_labels.push(external.public_label);
                    required_pairing_endpoints.insert(external.source_slot);
                }
                OnTheFlyExternalColorRoleV1::Antifundamental => {
                    antifundamental_labels.push(external.public_label);
                    required_pairing_endpoints.insert(external.source_slot);
                }
                OnTheFlyExternalColorRoleV1::Adjoint => {
                    adjoint_labels.push(external.public_label);
                }
            }
        }
        if pairing_endpoints != required_pairing_endpoints {
            return Err(integrity(
                "pairing-class coverage differs from the open-line endpoint roles",
            ));
        }
        fundamental_labels.sort_unstable();
        antifundamental_labels.sort_unstable();
        adjoint_labels.sort_unstable();
        if fundamental_labels.len() != antifundamental_labels.len() {
            return Err(integrity(
                "LC selector domain has unbalanced open-line endpoints",
            ));
        }

        let color_domain = if !fundamental_labels.is_empty() {
            CompactColorDomainV1::OpenLines
        } else if !adjoint_labels.is_empty() {
            CompactColorDomainV1::SingleTrace
        } else {
            CompactColorDomainV1::Singlet
        };
        if policy.color_basis == OnTheFlyColorBasis::Adjoint
            && (adjoint_labels.len() < 2
                || !fundamental_labels.is_empty()
                || policy.trace_reflections_folded)
        {
            return Err(integrity(
                "adjoint color basis requires at least two adjoint external sources, no fundamental lines and no trace-reflection folding",
            ));
        }
        let helicity_count = checked_product(
            externals_by_public_slot
                .iter()
                .map(|external| external.public_helicities.len()),
            "public helicity count",
        )?;
        let color_count = match color_domain {
            CompactColorDomainV1::Singlet => 1,
            CompactColorDomainV1::SingleTrace => {
                single_trace_color_count(adjoint_labels.len(), policy.color_basis)?
            }
            CompactColorDomainV1::OpenLines => {
                let line_count = fundamental_labels.len();
                let pairing_count = checked_factorial(line_count, "open-line pairing count")?;
                let adjoint_order_count =
                    checked_factorial(adjoint_labels.len(), "adjoint order count")?;
                let split_count = checked_binomial(
                    adjoint_labels
                        .len()
                        .checked_add(line_count.saturating_sub(1))
                        .ok_or_else(|| invalid("open-line split domain exceeds usize"))?,
                    line_count.saturating_sub(1),
                    "open-line split count",
                )?;
                pairing_count
                    .checked_mul(adjoint_order_count)
                    .and_then(|value| value.checked_mul(split_count))
                    .ok_or_else(|| invalid("LC open-line flow count exceeds usize"))?
            }
        };
        if policy.trace_reflections_folded && color_domain != CompactColorDomainV1::SingleTrace {
            return Err(integrity(
                "trace-reflection folding is valid only for a pure single-trace LC domain",
            ));
        }
        let mut adapter = Self {
            external_permutation: external_permutation.to_vec().into_boxed_slice(),
            representative_to_public: (0..anchors.len() as u32)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            public_to_representative: (0..anchors.len() as u32)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            public_label_by_representative_label: public_slot_by_label
                .keys()
                .copied()
                .map(|label| (label, label))
                .collect(),
            representative_label_by_public_label: public_slot_by_label
                .keys()
                .copied()
                .map(|label| (label, label))
                .collect(),
            externals_by_public_slot,
            public_slot_by_label,
            fundamental_labels: fundamental_labels.into_boxed_slice(),
            antifundamental_labels: antifundamental_labels.into_boxed_slice(),
            adjoint_labels: adjoint_labels.into_boxed_slice(),
            color_domain,
            color_basis: policy.color_basis,
            reference_unfolded_color_ordinal: None,
            trace_reflections_folded: policy.trace_reflections_folded,
            helicity_count,
            color_count,
        };
        if let Some(reference) = policy.reference_color_word {
            if reference.is_empty() {
                if adapter.color_domain != CompactColorDomainV1::Singlet {
                    return Err(invalid(
                        "empty reference color word is valid only for a singlet process",
                    ));
                }
            } else {
                // Requiring a member of the ordinary complete axis excludes
                // anti-endpoint-first and whole-block traversal aliases. Such
                // references are useful generation hints, but are not distinct
                // complete LC public components and cannot be retained without
                // a dense exceptional-flow table.
                adapter.selector_from_labels(&reference)?;
                let ordinal = adapter.unfolded_color_ordinal(&reference)?;
                let (actual, _) = adapter.unfolded_color_at(ordinal)?;
                if actual.as_ref() != reference.as_ref() {
                    return Err(invalid(
                        "reference color word is not a member of the ordinary complete LC axis",
                    ));
                }
                if adapter.trace_reflections_folded {
                    let reflected = reflected_trace_word(&reference)?;
                    let reflected_ordinal = adapter.unfolded_color_ordinal(&reflected)?;
                    if ordinal > reflected_ordinal {
                        return Err(invalid(
                            "folded trace reference must be the canonical member of its reflection class",
                        ));
                    }
                }
                adapter.reference_unfolded_color_ordinal = Some(ordinal);
            }
        }
        Ok(adapter)
    }

    /// Project public selector IDs through an already authenticated artifact
    /// alias without changing the representative compact seed or its query
    /// identity. The mapping is representative public slot -> requested public
    /// slot, matching the existing artifact-selection ABI.
    pub(super) fn with_public_permutation(mut self, permutation: &[usize]) -> RusticolResult<Self> {
        if permutation.len() != self.externals_by_public_slot.len()
            || permutation.iter().copied().collect::<BTreeSet<_>>()
                != (0..permutation.len()).collect::<BTreeSet<_>>()
        {
            return Err(integrity(
                "artifact alias does not define a complete public external permutation",
            ));
        }
        if permutation
            .iter()
            .copied()
            .enumerate()
            .all(|(representative, public)| representative == public)
        {
            return Ok(self);
        }
        let mut inverse = vec![u32::MAX; permutation.len()];
        let mut public_label_by_representative_label = BTreeMap::new();
        let mut representative_label_by_public_label = BTreeMap::new();
        for (representative_slot, public_slot) in permutation.iter().copied().enumerate() {
            inverse[public_slot] = u32::try_from(representative_slot)
                .map_err(|_| integrity("representative public slot exceeds u32"))?;
            let external = self
                .externals_by_public_slot
                .get(representative_slot)
                .ok_or_else(|| integrity("representative public slot is absent"))?;
            let expected_label = u32::try_from(representative_slot + 1)
                .map_err(|_| integrity("representative public label exceeds u32"))?;
            if external.public_label != expected_label {
                return Err(integrity(
                    "permuted public selection requires canonical one-based external labels",
                ));
            }
            let public_label = u32::try_from(public_slot + 1)
                .map_err(|_| integrity("requested public label exceeds u32"))?;
            public_label_by_representative_label.insert(expected_label, public_label);
            representative_label_by_public_label.insert(public_label, expected_label);
        }
        if inverse.contains(&u32::MAX) {
            return Err(integrity("artifact alias omits a public external slot"));
        }
        self.representative_to_public = permutation
            .iter()
            .copied()
            .map(|slot| u32::try_from(slot).map_err(|_| integrity("public slot exceeds u32")))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        self.public_to_representative = inverse.into_boxed_slice();
        self.public_label_by_representative_label = public_label_by_representative_label;
        self.representative_label_by_public_label = representative_label_by_public_label;
        Ok(self)
    }

    pub(super) const fn helicity_count(&self) -> usize {
        self.helicity_count
    }

    pub(super) const fn color_count(&self) -> usize {
        self.color_count
    }

    /// Decode one structural query directly from compact ordinals.
    ///
    /// Contracted NLC/full execution has a singleton public color axis, but
    /// evaluates its authenticated structural owner basis before applying the
    /// color metric.  This cold adapter keeps those decoded queries transient
    /// instead of retaining flow strings, traces, or a dense public table.
    pub(super) fn decoded_query_at(
        &self,
        seed: &OnTheFlyProcessSeedV1,
        helicity_ordinal: usize,
        structural_color_ordinal: usize,
    ) -> RusticolResult<DecodedLcQueryV1> {
        let public_helicities = self.helicity_at(helicity_ordinal)?;
        let (_, selector) = self.color_at(structural_color_ordinal)?;
        DecodedLcQueryV1::new(
            seed,
            self.external_permutation.to_vec(),
            &public_helicities,
            selector,
        )
    }

    pub(super) fn parse_helicity_id(&self, value: &str) -> RusticolResult<Box<[i32]>> {
        let payload = value
            .strip_prefix("h:")
            .ok_or_else(|| invalid(format!("unknown helicity ID {value:?}")))?;
        if payload.is_empty() {
            return Err(invalid("helicity ID has no external values"));
        }
        let public_values = payload
            .split(',')
            .map(|part| {
                let parsed = part.parse::<i32>().map_err(|_| {
                    invalid(format!("helicity ID {value:?} has an invalid integer"))
                })?;
                if format!("{parsed:+}") != part {
                    return Err(invalid(format!("helicity ID {value:?} is not canonical",)));
                }
                Ok(parsed)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if public_values.len() != self.externals_by_public_slot.len() {
            return Err(invalid(format!(
                "helicity ID {value:?} has {} values, expected {}",
                public_values.len(),
                self.externals_by_public_slot.len(),
            )));
        }
        let mut values = vec![0; public_values.len()];
        for (public_slot, selected) in public_values.iter().copied().enumerate() {
            let representative_slot = *self
                .public_to_representative
                .get(public_slot)
                .ok_or_else(|| integrity("requested public helicity slot is absent"))?;
            values[representative_slot as usize] = selected;
        }
        for (public_slot, (external, selected)) in self
            .externals_by_public_slot
            .iter()
            .zip(values.iter().copied())
            .enumerate()
        {
            if !external.public_helicities.contains(&selected) {
                return Err(invalid(format!(
                    "helicity ID {value:?} selects unavailable helicity {selected} at public slot {public_slot}",
                )));
            }
        }
        Ok(values.into_boxed_slice())
    }

    pub(super) fn parse_color_id(
        &self,
        value: &str,
    ) -> RusticolResult<(Box<[u32]>, OnTheFlyLcSelectorV1)> {
        if value == "flow:singlet" {
            if self.color_domain != CompactColorDomainV1::Singlet {
                return Err(invalid(
                    "flow:singlet is invalid for colored external particles",
                ));
            }
            return Ok((Box::new([]), OnTheFlyLcSelectorV1::Singlet));
        }
        let payload = value
            .strip_prefix("flow:")
            .ok_or_else(|| invalid(format!("unknown LC color ID {value:?}")))?;
        if payload.is_empty() {
            return Err(invalid("LC color ID has no external labels"));
        }
        let labels = payload
            .split(',')
            .map(|part| {
                let label = part
                    .parse::<u32>()
                    .map_err(|_| invalid(format!("LC color ID {value:?} has an invalid label")))?;
                if label.to_string() != part {
                    return Err(invalid(format!("LC color ID {value:?} is not canonical",)));
                }
                Ok(label)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let canonical = canonical_color_id(&labels);
        if canonical != value {
            return Err(invalid(format!("LC color ID {value:?} is not canonical")));
        }
        let representative_labels = labels
            .iter()
            .map(|label| {
                self.representative_label_by_public_label
                    .get(label)
                    .copied()
                    .ok_or_else(|| invalid("LC color ID contains an unknown external label"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let selector = self.selector_from_labels(&representative_labels)?;
        Ok((representative_labels.into_boxed_slice(), selector))
    }

    fn selector_from_labels(&self, labels: &[u32]) -> RusticolResult<OnTheFlyLcSelectorV1> {
        let required = self
            .externals_by_public_slot
            .iter()
            .filter(|external| external.color_role != OnTheFlyExternalColorRoleV1::Singlet)
            .map(|external| external.public_label)
            .collect::<BTreeSet<_>>();
        let observed = labels.iter().copied().collect::<BTreeSet<_>>();
        if observed.len() != labels.len() || observed != required {
            return Err(invalid(
                "LC color ID must cover every colored external label exactly once",
            ));
        }
        let public_slots = labels
            .iter()
            .map(|label| {
                self.public_slot_by_label
                    .get(label)
                    .copied()
                    .ok_or_else(|| invalid("LC color ID contains an unknown external label"))
                    .and_then(|slot| {
                        u32::try_from(slot).map_err(|_| invalid("public external slot exceeds u32"))
                    })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        match self.color_domain {
            CompactColorDomainV1::Singlet => {
                Err(invalid("colored LC flow is invalid for a singlet process"))
            }
            CompactColorDomainV1::SingleTrace => {
                if labels.iter().any(|label| {
                    let slot = self.public_slot_by_label[label];
                    self.externals_by_public_slot[slot].color_role
                        != OnTheFlyExternalColorRoleV1::Adjoint
                }) {
                    return Err(invalid("single-trace LC flow contains a non-adjoint label"));
                }
                if labels.first() != self.adjoint_labels.first() {
                    return Err(invalid(
                        "single-trace LC flow must use the minimum adjoint label as cyclic anchor",
                    ));
                }
                if self.color_basis == OnTheFlyColorBasis::Adjoint
                    && labels.last() != self.adjoint_labels.last()
                {
                    return Err(invalid(
                        "adjoint color flow must preserve both fixed endpoint labels",
                    ));
                }
                Ok(OnTheFlyLcSelectorV1::single_trace(public_slots))
            }
            CompactColorDomainV1::OpenLines => {
                let mut blocks = Vec::<Vec<u32>>::new();
                let mut offset = 0;
                while offset < labels.len() {
                    let start_slot = self.public_slot_by_label[&labels[offset]];
                    if self.externals_by_public_slot[start_slot].color_role
                        != OnTheFlyExternalColorRoleV1::Fundamental
                    {
                        return Err(invalid(
                            "open-line LC flow block must start at a fundamental label",
                        ));
                    }
                    let mut end = offset + 1;
                    while end < labels.len() {
                        let role = self.externals_by_public_slot
                            [self.public_slot_by_label[&labels[end]]]
                            .color_role;
                        if role != OnTheFlyExternalColorRoleV1::Adjoint {
                            break;
                        }
                        end += 1;
                    }
                    if end >= labels.len()
                        || self.externals_by_public_slot[self.public_slot_by_label[&labels[end]]]
                            .color_role
                            != OnTheFlyExternalColorRoleV1::Antifundamental
                    {
                        return Err(invalid(
                            "open-line LC flow block must end at an antifundamental label",
                        ));
                    }
                    blocks.push(public_slots[offset..=end].to_vec());
                    offset = end + 1;
                }
                if blocks.len() != self.fundamental_labels.len() {
                    return Err(invalid("LC flow has the wrong number of open-line blocks"));
                }
                // The established LC public plan orders blocks by the sorted
                // fundamental external labels.  Other block orders are valid
                // traversal aliases internally, but are not distinct public IDs.
                let starts = blocks
                    .iter()
                    .map(|block| {
                        let public_slot = block[0] as usize;
                        self.externals_by_public_slot[public_slot].public_label
                    })
                    .collect::<Vec<_>>();
                if starts.as_slice() != self.fundamental_labels.as_ref() {
                    return Err(invalid(
                        "open-line LC flow blocks are not in canonical fundamental-label order",
                    ));
                }
                Ok(OnTheFlyLcSelectorV1::open_lines(blocks))
            }
        }
    }

    fn helicity_at(&self, mut index: usize) -> RusticolResult<Box<[i32]>> {
        if index >= self.helicity_count {
            return Err(invalid("public helicity ordinal is out of bounds"));
        }
        let mut values = vec![0; self.externals_by_public_slot.len()];
        for (slot, external) in self.externals_by_public_slot.iter().enumerate().rev() {
            let radix = external.public_helicities.len();
            values[slot] = external.public_helicities[index % radix];
            index /= radix;
        }
        Ok(values.into_boxed_slice())
    }

    fn public_helicities_from_representative(
        &self,
        representative: &[i32],
    ) -> RusticolResult<Box<[i32]>> {
        if representative.len() != self.representative_to_public.len() {
            return Err(integrity(
                "representative helicity domain differs from the public permutation",
            ));
        }
        let mut public = vec![0; representative.len()];
        for (representative_slot, value) in representative.iter().copied().enumerate() {
            let public_slot = self.representative_to_public[representative_slot] as usize;
            public[public_slot] = value;
        }
        Ok(public.into_boxed_slice())
    }

    fn public_labels_from_representative(
        &self,
        representative: &[u32],
    ) -> RusticolResult<Box<[u32]>> {
        representative
            .iter()
            .map(|label| {
                self.public_label_by_representative_label
                    .get(label)
                    .copied()
                    .ok_or_else(|| integrity("representative LC label has no public alias"))
            })
            .collect::<RusticolResult<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    fn helicity_ordinal(&self, values: &[i32]) -> RusticolResult<usize> {
        let mut ordinal = 0usize;
        for (external, selected) in self.externals_by_public_slot.iter().zip(values) {
            let digit = external
                .public_helicities
                .iter()
                .position(|candidate| candidate == selected)
                .ok_or_else(|| invalid("public helicity is outside the compact seed"))?;
            ordinal = ordinal
                .checked_mul(external.public_helicities.len())
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| invalid("public helicity ordinal exceeds usize"))?;
        }
        Ok(ordinal)
    }

    fn color_at(&self, index: usize) -> RusticolResult<(Box<[u32]>, OnTheFlyLcSelectorV1)> {
        if index >= self.color_count {
            return Err(invalid("public LC color ordinal is out of bounds"));
        }
        let unfolded = self.unfolded_color_ordinal_at(index)?;
        self.unfolded_color_at(unfolded)
    }

    fn unfolded_color_at(
        &self,
        index: usize,
    ) -> RusticolResult<(Box<[u32]>, OnTheFlyLcSelectorV1)> {
        if index >= self.color_count {
            return Err(invalid("unfolded LC color ordinal is out of bounds"));
        }
        let labels = match self.color_domain {
            CompactColorDomainV1::Singlet => Vec::new(),
            CompactColorDomainV1::SingleTrace => {
                let first = *self
                    .adjoint_labels
                    .first()
                    .ok_or_else(|| integrity("single-trace domain has no adjoint anchor"))?;
                let mut labels = vec![first];
                if self.color_basis == OnTheFlyColorBasis::Adjoint {
                    let last = self.adjoint_labels.len() - 1;
                    labels.extend(nth_permutation(&self.adjoint_labels[1..last], index)?);
                    labels.push(self.adjoint_labels[last]);
                } else {
                    labels.extend(nth_permutation(&self.adjoint_labels[1..], index)?);
                }
                labels
            }
            CompactColorDomainV1::OpenLines => self.open_line_color_at(index)?,
        };
        let selector = if labels.is_empty() {
            OnTheFlyLcSelectorV1::Singlet
        } else {
            self.selector_from_labels(&labels)?
        };
        Ok((labels.into_boxed_slice(), selector))
    }

    fn unfolded_color_ordinal_at(&self, public_index: usize) -> RusticolResult<usize> {
        if public_index >= self.color_count {
            return Err(invalid("public LC color ordinal is out of bounds"));
        }
        if self.trace_reflections_folded {
            return self.folded_unfolded_color_ordinal_at(public_index);
        }
        match self.reference_unfolded_color_ordinal {
            None => Ok(public_index),
            Some(reference) if public_index == 0 => Ok(reference),
            Some(reference) => {
                let ordinary = public_index - 1;
                Ok(if ordinary >= reference {
                    ordinary + 1
                } else {
                    ordinary
                })
            }
        }
    }

    fn folded_unfolded_color_ordinal_at(&self, public_index: usize) -> RusticolResult<usize> {
        let reference = self.reference_unfolded_color_ordinal;
        let mut next_public = 0usize;
        if let Some(reference) = reference {
            let (word, _) = self.unfolded_color_at(reference)?;
            let reflected = reflected_trace_word(&word)?;
            let reflected_ordinal = self.unfolded_color_ordinal(&reflected)?;
            for ordinal in [reference, reflected_ordinal] {
                if next_public == public_index {
                    return Ok(ordinal);
                }
                next_public += 1;
                if reflected_ordinal == reference {
                    break;
                }
            }
        }
        for ordinal in 0..self.color_count {
            let (word, _) = self.unfolded_color_at(ordinal)?;
            let reflected = reflected_trace_word(&word)?;
            let reflected_ordinal = self.unfolded_color_ordinal(&reflected)?;
            if ordinal > reflected_ordinal || reference == Some(ordinal) {
                continue;
            }
            for member in [ordinal, reflected_ordinal] {
                if next_public == public_index {
                    return Ok(member);
                }
                next_public += 1;
                if reflected_ordinal == ordinal {
                    break;
                }
            }
        }
        Err(integrity(
            "folded trace enumeration did not cover its public color axis",
        ))
    }

    fn open_line_color_at(&self, index: usize) -> RusticolResult<Vec<u32>> {
        let line_count = self.fundamental_labels.len();
        let split_count = checked_binomial(
            self.adjoint_labels
                .len()
                .checked_add(line_count.saturating_sub(1))
                .ok_or_else(|| invalid("open-line split domain exceeds usize"))?,
            line_count.saturating_sub(1),
            "open-line split count",
        )?;
        let adjoint_order_count =
            checked_factorial(self.adjoint_labels.len(), "adjoint order count")?;
        let allocation_count = adjoint_order_count
            .checked_mul(split_count)
            .ok_or_else(|| invalid("ordered adjoint allocation count exceeds usize"))?;
        let pairing_index = index / allocation_count;
        let allocation_index = index % allocation_count;
        let adjoint_order_index = allocation_index / split_count;
        let split_index = allocation_index % split_count;
        let antifundamentals = nth_permutation(&self.antifundamental_labels, pairing_index)?;
        let adjoints = nth_permutation(&self.adjoint_labels, adjoint_order_index)?;
        let lengths = nth_weak_composition(adjoint_labels_len(&adjoints), line_count, split_index)?;

        let mut labels = Vec::new();
        labels
            .try_reserve_exact(
                line_count
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(adjoints.len()))
                    .ok_or_else(|| invalid("LC color word length exceeds usize"))?,
            )
            .map_err(|error| invalid(format!("LC color word allocation failed: {error}")))?;
        let mut offset = 0_usize;
        for line in 0..line_count {
            labels.push(self.fundamental_labels[line]);
            let end = offset
                .checked_add(lengths[line])
                .ok_or_else(|| invalid("LC adjoint allocation exceeds usize"))?;
            labels.extend_from_slice(&adjoints[offset..end]);
            labels.push(antifundamentals[line]);
            offset = end;
        }
        if offset != adjoints.len() {
            return Err(integrity("LC adjoint allocation did not consume its word"));
        }
        Ok(labels)
    }

    fn unfolded_color_ordinal(&self, labels: &[u32]) -> RusticolResult<usize> {
        match self.color_domain {
            CompactColorDomainV1::Singlet => {
                if labels.is_empty() {
                    Ok(0)
                } else {
                    Err(invalid("singlet color ordinal received a colored word"))
                }
            }
            CompactColorDomainV1::SingleTrace => {
                if labels.first() != self.adjoint_labels.first() {
                    return Err(invalid("single-trace cyclic anchor differs"));
                }
                if self.color_basis == OnTheFlyColorBasis::Adjoint {
                    if labels.len() != self.adjoint_labels.len()
                        || labels.last() != self.adjoint_labels.last()
                    {
                        return Err(invalid("adjoint color flow endpoints differ"));
                    }
                    let last = self.adjoint_labels.len() - 1;
                    permutation_rank(&self.adjoint_labels[1..last], &labels[1..last])
                } else {
                    permutation_rank(&self.adjoint_labels[1..], &labels[1..])
                }
            }
            CompactColorDomainV1::OpenLines => {
                let mut antifundamentals = Vec::new();
                let mut adjoints = Vec::new();
                let mut lengths = Vec::new();
                let mut offset = 0;
                for fundamental in self.fundamental_labels.iter().copied() {
                    if labels.get(offset).copied() != Some(fundamental) {
                        return Err(invalid("open-line block order differs"));
                    }
                    offset += 1;
                    let start = adjoints.len();
                    while let Some(label) = labels.get(offset).copied() {
                        let role = self.externals_by_public_slot[self.public_slot_by_label[&label]]
                            .color_role;
                        if role != OnTheFlyExternalColorRoleV1::Adjoint {
                            break;
                        }
                        adjoints.push(label);
                        offset += 1;
                    }
                    let antifundamental = labels
                        .get(offset)
                        .copied()
                        .ok_or_else(|| invalid("open-line block has no terminal endpoint"))?;
                    antifundamentals.push(antifundamental);
                    offset += 1;
                    lengths.push(adjoints.len() - start);
                }
                if offset != labels.len() {
                    return Err(invalid("open-line flow has trailing labels"));
                }
                let pairing_index =
                    permutation_rank(&self.antifundamental_labels, &antifundamentals)?;
                let adjoint_order_index = permutation_rank(&self.adjoint_labels, &adjoints)?;
                let split_index = weak_composition_rank(&lengths)?;
                let split_count = checked_binomial(
                    self.adjoint_labels
                        .len()
                        .checked_add(self.fundamental_labels.len().saturating_sub(1))
                        .ok_or_else(|| invalid("open-line split domain exceeds usize"))?,
                    self.fundamental_labels.len().saturating_sub(1),
                    "open-line split count",
                )?;
                let allocation_count =
                    checked_factorial(self.adjoint_labels.len(), "adjoint order count")?
                        .checked_mul(split_count)
                        .ok_or_else(|| invalid("ordered adjoint allocation count exceeds usize"))?;
                pairing_index
                    .checked_mul(allocation_count)
                    .and_then(|value| {
                        value.checked_add(
                            adjoint_order_index
                                .checked_mul(split_count)
                                .and_then(|inner| inner.checked_add(split_index))?,
                        )
                    })
                    .ok_or_else(|| invalid("LC color ordinal exceeds usize"))
            }
        }
    }

    fn color_ordinal(&self, labels: &[u32]) -> RusticolResult<usize> {
        let unfolded = self.unfolded_color_ordinal(labels)?;
        if self.trace_reflections_folded {
            for public in 0..self.color_count {
                if self.folded_unfolded_color_ordinal_at(public)? == unfolded {
                    return Ok(public);
                }
            }
            return Err(integrity(
                "folded trace public axis omits an unfolded color member",
            ));
        }
        Ok(match self.reference_unfolded_color_ordinal {
            None => unfolded,
            Some(reference) if unfolded == reference => 0,
            Some(reference) if unfolded < reference => unfolded + 1,
            Some(_) => unfolded,
        })
    }

    pub(super) fn selected_ordinals(
        &self,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<OnTheFlySelectedOrdinalsV1> {
        let helicity_ordinals = match selected_helicity_ids {
            None => None,
            Some(ids) => {
                if ids.is_empty() {
                    return Err(invalid("helicity selection is empty"));
                }
                let mut ordinals = ids
                    .iter()
                    .map(|id| {
                        let values = self.parse_helicity_id(id)?;
                        self.helicity_ordinal(&values)
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                ordinals.sort_unstable();
                ordinals.dedup();
                Some(ordinals.into_boxed_slice())
            }
        };
        let color_ordinals = match selected_color_ids {
            None => None,
            Some(ids) => {
                if ids.is_empty() {
                    return Err(invalid("LC color selection is empty"));
                }
                let mut ordinals = ids
                    .iter()
                    .map(|id| {
                        let (labels, _) = self.parse_color_id(id)?;
                        self.color_ordinal(&labels)
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                ordinals.sort_unstable();
                ordinals.dedup();
                Some(ordinals.into_boxed_slice())
            }
        };
        Ok((helicity_ordinals, color_ordinals))
    }

    pub(super) fn selection<'a>(
        &'a self,
        seed: &'a OnTheFlyProcessSeedV1,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<OnTheFlyLazySelectionV1<'a>> {
        let (helicity_ordinals, color_ordinals) =
            self.selected_ordinals(selected_helicity_ids, selected_color_ids)?;
        Ok(OnTheFlyLazySelectionV1 {
            adapter: self,
            seed,
            helicity_ordinals,
            color_ordinals,
        })
    }

    pub(super) fn selection_from_ordinals<'a>(
        &'a self,
        seed: &'a OnTheFlyProcessSeedV1,
        helicity_ordinals: Option<&[usize]>,
        color_ordinals: Option<&[usize]>,
    ) -> RusticolResult<OnTheFlyLazySelectionV1<'a>> {
        Ok(OnTheFlyLazySelectionV1 {
            adapter: self,
            seed,
            helicity_ordinals: owned_selected_ordinals(
                helicity_ordinals,
                self.helicity_count,
                "helicity",
            )?,
            color_ordinals: owned_selected_ordinals(color_ordinals, self.color_count, "color")?,
        })
    }
}

/// Lazy selector Cartesian product.  This is the explicit materializer seam
/// for factorial default-all flow coverage; callers may consume one request or
/// one bounded batch without ever allocating a dense public axis.
pub(super) struct OnTheFlyLazySelectionV1<'a> {
    adapter: &'a OnTheFlyCompactSelectorAdapterV1,
    seed: &'a OnTheFlyProcessSeedV1,
    helicity_ordinals: Option<Box<[usize]>>,
    color_ordinals: Option<Box<[usize]>>,
}

impl OnTheFlyLazySelectionV1<'_> {
    pub(super) fn helicity_count(&self) -> usize {
        self.helicity_ordinals
            .as_ref()
            .map_or(self.adapter.helicity_count, |values| values.len())
    }

    pub(super) fn color_count(&self) -> usize {
        self.color_ordinals
            .as_ref()
            .map_or(self.adapter.color_count, |values| values.len())
    }

    pub(super) fn helicity_ordinal_at(&self, position: usize) -> RusticolResult<usize> {
        selected_ordinal(
            self.helicity_ordinals.as_deref(),
            self.adapter.helicity_count,
            position,
            "helicity",
        )
    }

    pub(super) fn color_ordinal_at(&self, position: usize) -> RusticolResult<usize> {
        selected_ordinal(
            self.color_ordinals.as_deref(),
            self.adapter.color_count,
            position,
            "color",
        )
    }

    pub(super) fn request_count(&self) -> RusticolResult<usize> {
        self.helicity_count()
            .checked_mul(self.color_count())
            .ok_or_else(|| invalid("selected LC query count exceeds usize"))
    }

    pub(super) fn helicity_id_at(&self, position: usize) -> RusticolResult<String> {
        let ordinal = selected_ordinal(
            self.helicity_ordinals.as_deref(),
            self.adapter.helicity_count,
            position,
            "helicity",
        )?;
        let representative = self.adapter.helicity_at(ordinal)?;
        let public = self
            .adapter
            .public_helicities_from_representative(&representative)?;
        Ok(canonical_helicity_id(&public))
    }

    pub(super) fn color_id_at(&self, position: usize) -> RusticolResult<String> {
        let ordinal = selected_ordinal(
            self.color_ordinals.as_deref(),
            self.adapter.color_count,
            position,
            "color",
        )?;
        let (representative, _) = self.adapter.color_at(ordinal)?;
        let public = self
            .adapter
            .public_labels_from_representative(&representative)?;
        Ok(canonical_color_id(&public))
    }

    pub(super) fn request_at(&self, position: usize) -> RusticolResult<OnTheFlyLcQueryRequestV1> {
        let color_count = self.color_count();
        if color_count == 0 || position >= self.request_count()? {
            return Err(invalid("selected LC query ordinal is out of bounds"));
        }
        let helicity_position = position / color_count;
        let color_position = position % color_count;
        let helicity_ordinal = selected_ordinal(
            self.helicity_ordinals.as_deref(),
            self.adapter.helicity_count,
            helicity_position,
            "helicity",
        )?;
        let color_ordinal = selected_ordinal(
            self.color_ordinals.as_deref(),
            self.adapter.color_count,
            color_position,
            "color",
        )?;
        let query = self
            .adapter
            .decoded_query_at(self.seed, helicity_ordinal, color_ordinal)?;
        OnTheFlyLcQueryRequestV1::new(
            query,
            vec![OnTheFlyLcReductionTargetV1::new(
                helicity_position,
                color_position,
                1.0,
            )?],
        )
    }

    pub(super) fn iter(&self) -> OnTheFlyLazyRequestIterV1<'_, '_> {
        OnTheFlyLazyRequestIterV1 {
            selection: self,
            next: 0,
        }
    }
}

pub(super) struct OnTheFlyLazyRequestIterV1<'selection, 'seed> {
    selection: &'selection OnTheFlyLazySelectionV1<'seed>,
    next: usize,
}

impl Iterator for OnTheFlyLazyRequestIterV1<'_, '_> {
    type Item = RusticolResult<OnTheFlyLcQueryRequestV1>;

    fn next(&mut self) -> Option<Self::Item> {
        let count = match self.selection.request_count() {
            Ok(value) => value,
            Err(error) => {
                self.next = usize::MAX;
                return Some(Err(error));
            }
        };
        if self.next >= count {
            return None;
        }
        let position = self.next;
        self.next += 1;
        Some(self.selection.request_at(position))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.selection.request_count() {
            Ok(count) => {
                let remaining = count.saturating_sub(self.next);
                (remaining, Some(remaining))
            }
            Err(_) => (1, Some(1)),
        }
    }
}

/// Lane-local cache used only by explicit metadata calls.  Evaluation and
/// profiling consume [`OnTheFlyLazySelectionV1`] directly and never fill it.
#[derive(Default)]
pub(super) struct OnTheFlySelectorIntrospectionCacheV1 {
    #[expect(
        dead_code,
        reason = "retained for the private external-selector metadata compatibility API"
    )]
    externals: Option<Box<[OnTheFlyExternalSelectorRecordV1]>>,
    helicities: Option<Box<[OnTheFlyHelicitySelectorRecordV1]>>,
    colors: Option<Box<[OnTheFlyColorSelectorRecordV1]>>,
}

impl OnTheFlySelectorIntrospectionCacheV1 {
    #[expect(
        dead_code,
        reason = "retained for the private external-selector metadata compatibility API"
    )]
    pub(super) fn externals<'a>(
        &'a mut self,
        adapter: &OnTheFlyCompactSelectorAdapterV1,
    ) -> &'a [OnTheFlyExternalSelectorRecordV1] {
        self.externals.get_or_insert_with(|| {
            adapter
                .externals_by_public_slot
                .iter()
                .map(|external| OnTheFlyExternalSelectorRecordV1 {
                    source_slot: external.source_slot,
                    public_slot: external.public_slot,
                    public_label: external.public_label,
                    is_initial: external.is_initial,
                    color_role: external.color_role,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    pub(super) fn helicities<'a>(
        &'a mut self,
        adapter: &OnTheFlyCompactSelectorAdapterV1,
    ) -> RusticolResult<&'a [OnTheFlyHelicitySelectorRecordV1]> {
        if self.helicities.is_none() {
            let mut records = Vec::new();
            records
                .try_reserve_exact(adapter.helicity_count)
                .map_err(|error| {
                    invalid(format!("helicity metadata allocation failed: {error}"))
                })?;
            for index in 0..adapter.helicity_count {
                let values = adapter.helicity_at(index)?;
                records.push(OnTheFlyHelicitySelectorRecordV1 {
                    id: canonical_helicity_id(&values),
                    index,
                    values,
                });
            }
            self.helicities = Some(records.into_boxed_slice());
        }
        Ok(self
            .helicities
            .as_deref()
            .expect("helicity cache was filled"))
    }

    pub(super) fn colors<'a>(
        &'a mut self,
        adapter: &OnTheFlyCompactSelectorAdapterV1,
    ) -> RusticolResult<&'a [OnTheFlyColorSelectorRecordV1]> {
        if self.colors.is_none() {
            let mut records = Vec::new();
            records
                .try_reserve_exact(adapter.color_count)
                .map_err(|error| invalid(format!("color metadata allocation failed: {error}")))?;
            for index in 0..adapter.color_count {
                let (word, selector) = adapter.color_at(index)?;
                records.push(OnTheFlyColorSelectorRecordV1 {
                    id: canonical_color_id(&word),
                    index,
                    word,
                    selector,
                });
            }
            self.colors = Some(records.into_boxed_slice());
        }
        Ok(self.colors.as_deref().expect("color cache was filled"))
    }
}

fn selected_ordinal(
    selected: Option<&[usize]>,
    all_count: usize,
    position: usize,
    label: &str,
) -> RusticolResult<usize> {
    match selected {
        Some(values) => values
            .get(position)
            .copied()
            .ok_or_else(|| invalid(format!("selected {label} position is out of bounds"))),
        None if position < all_count => Ok(position),
        None => Err(invalid(format!("{label} position is out of bounds"))),
    }
}

fn owned_selected_ordinals(
    selected: Option<&[usize]>,
    all_count: usize,
    label: &str,
) -> RusticolResult<Option<Box<[usize]>>> {
    let Some(selected) = selected else {
        return Ok(None);
    };
    if selected.is_empty() {
        return Err(invalid(format!("{label} selection is empty")));
    }
    if selected.iter().any(|ordinal| *ordinal >= all_count) {
        return Err(invalid(format!(
            "selected {label} ordinal is out of bounds"
        )));
    }
    if selected.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(format!(
            "selected {label} ordinals are not strictly increasing"
        )));
    }
    Ok(Some(selected.to_vec().into_boxed_slice()))
}

fn canonical_helicity_id(values: &[i32]) -> String {
    format!(
        "h:{}",
        values
            .iter()
            .map(|value| format!("{value:+}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn canonical_color_id(word: &[u32]) -> String {
    if word.is_empty() {
        "flow:singlet".to_string()
    } else {
        format!(
            "flow:{}",
            word.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

fn reflected_trace_word(word: &[u32]) -> RusticolResult<Vec<u32>> {
    let Some((first, rest)) = word.split_first() else {
        return Err(invalid("cannot reflect an empty trace word"));
    };
    let mut reflected = Vec::new();
    reflected
        .try_reserve_exact(word.len())
        .map_err(|error| invalid(format!("trace-reflection allocation failed: {error}")))?;
    reflected.push(*first);
    reflected.extend(rest.iter().rev().copied());
    Ok(reflected)
}

fn checked_product(values: impl IntoIterator<Item = usize>, label: &str) -> RusticolResult<usize> {
    values.into_iter().try_fold(1usize, |total, value| {
        total
            .checked_mul(value)
            .ok_or_else(|| invalid(format!("{label} exceeds usize")))
    })
}

fn single_trace_color_count(count: usize, basis: OnTheFlyColorBasis) -> RusticolResult<usize> {
    let anchors = match basis {
        OnTheFlyColorBasis::Trace => 1,
        OnTheFlyColorBasis::Adjoint => 2,
    };
    checked_factorial(count.saturating_sub(anchors), "single-trace count")
}

fn checked_factorial(value: usize, label: &str) -> RusticolResult<usize> {
    (2..=value).try_fold(1usize, |total, factor| {
        total
            .checked_mul(factor)
            .ok_or_else(|| invalid(format!("{label} exceeds usize")))
    })
}

fn checked_binomial(n: usize, k: usize, label: &str) -> RusticolResult<usize> {
    let k = k.min(n.saturating_sub(k));
    let mut value = 1usize;
    for index in 0..k {
        value = value
            .checked_mul(n - index)
            .ok_or_else(|| invalid(format!("{label} exceeds usize")))?
            / (index + 1);
    }
    Ok(value)
}

fn nth_permutation(values: &[u32], mut rank: usize) -> RusticolResult<Vec<u32>> {
    if rank >= checked_factorial(values.len(), "permutation count")? {
        return Err(invalid("permutation ordinal is out of bounds"));
    }
    let mut remaining = values.to_vec();
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|error| invalid(format!("permutation allocation failed: {error}")))?;
    for width in (1..=values.len()).rev() {
        let block = checked_factorial(width - 1, "permutation block")?;
        let digit = rank / block;
        rank %= block;
        result.push(remaining.remove(digit));
    }
    Ok(result)
}

fn permutation_rank(reference: &[u32], selected: &[u32]) -> RusticolResult<usize> {
    if reference.len() != selected.len() {
        return Err(invalid("selected permutation has the wrong length"));
    }
    let mut remaining = reference.to_vec();
    let mut rank = 0usize;
    for (index, selected) in selected.iter().copied().enumerate() {
        let digit = remaining
            .iter()
            .position(|candidate| *candidate == selected)
            .ok_or_else(|| invalid("selected values are not a permutation"))?;
        rank = rank
            .checked_add(
                digit
                    .checked_mul(checked_factorial(
                        reference.len() - index - 1,
                        "permutation block",
                    )?)
                    .ok_or_else(|| invalid("permutation ordinal exceeds usize"))?,
            )
            .ok_or_else(|| invalid("permutation ordinal exceeds usize"))?;
        remaining.remove(digit);
    }
    Ok(rank)
}

fn nth_weak_composition(total: usize, bins: usize, mut rank: usize) -> RusticolResult<Vec<usize>> {
    if bins == 0 {
        return if total == 0 && rank == 0 {
            Ok(Vec::new())
        } else {
            Err(invalid("nonempty sequence cannot be split into zero bins"))
        };
    }
    let count = checked_binomial(
        total
            .checked_add(bins - 1)
            .ok_or_else(|| invalid("weak-composition domain exceeds usize"))?,
        bins - 1,
        "weak-composition count",
    )?;
    if rank >= count {
        return Err(invalid("weak-composition ordinal is out of bounds"));
    }
    let mut remaining = total;
    let mut result = Vec::with_capacity(bins);
    for remaining_bins in (2..=bins).rev() {
        let mut selected = None;
        for head in 0..=remaining {
            let suffix_count = checked_binomial(
                remaining - head + remaining_bins - 2,
                remaining_bins - 2,
                "weak-composition suffix count",
            )?;
            if rank < suffix_count {
                selected = Some(head);
                break;
            }
            rank -= suffix_count;
        }
        let head = selected.ok_or_else(|| integrity("weak-composition unranking failed"))?;
        result.push(head);
        remaining -= head;
    }
    result.push(remaining);
    Ok(result)
}

fn weak_composition_rank(values: &[usize]) -> RusticolResult<usize> {
    if values.is_empty() {
        return Err(invalid("weak composition has no bins"));
    }
    let mut remaining = values
        .iter()
        .try_fold(0usize, |total, value| total.checked_add(*value))
        .ok_or_else(|| invalid("weak-composition total exceeds usize"))?;
    let mut rank = 0usize;
    for (index, selected) in values[..values.len() - 1].iter().copied().enumerate() {
        if selected > remaining {
            return Err(invalid("weak-composition value exceeds its remainder"));
        }
        let remaining_bins = values.len() - index;
        for head in 0..selected {
            rank = rank
                .checked_add(checked_binomial(
                    remaining - head + remaining_bins - 2,
                    remaining_bins - 2,
                    "weak-composition suffix count",
                )?)
                .ok_or_else(|| invalid("weak-composition ordinal exceeds usize"))?;
        }
        remaining -= selected;
    }
    if values.last().copied() != Some(remaining) {
        return Err(invalid("weak composition does not preserve its total"));
    }
    Ok(rank)
}

const fn adjoint_labels_len(values: &[u32]) -> usize {
    values.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::SemanticDigest;
    use crate::recurrence::on_the_fly::scalar_adapter_test_seed;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn adapter(
        roles: &[(u32, OnTheFlyExternalColorRoleV1, &[i32])],
    ) -> OnTheFlyCompactSelectorAdapterV1 {
        adapter_in_basis(roles, OnTheFlyColorBasis::Trace)
    }

    fn adapter_in_basis(
        roles: &[(u32, OnTheFlyExternalColorRoleV1, &[i32])],
        color_basis: OnTheFlyColorBasis,
    ) -> OnTheFlyCompactSelectorAdapterV1 {
        let externals_by_public_slot = roles
            .iter()
            .enumerate()
            .map(|(slot, (label, role, helicities))| CompactExternalV1 {
                source_slot: slot as u32,
                public_slot: slot as u32,
                public_label: *label,
                is_initial: slot < 2,
                color_role: *role,
                public_helicities: helicities.to_vec().into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let public_slot_by_label = externals_by_public_slot
            .iter()
            .enumerate()
            .map(|(slot, external)| (external.public_label, slot))
            .collect();
        let labels = |role| {
            let mut labels = externals_by_public_slot
                .iter()
                .filter(|external| external.color_role == role)
                .map(|external| external.public_label)
                .collect::<Vec<_>>();
            labels.sort_unstable();
            labels.into_boxed_slice()
        };
        let fundamental_labels = labels(OnTheFlyExternalColorRoleV1::Fundamental);
        let antifundamental_labels = labels(OnTheFlyExternalColorRoleV1::Antifundamental);
        let adjoint_labels = labels(OnTheFlyExternalColorRoleV1::Adjoint);
        let color_domain = if !fundamental_labels.is_empty() {
            CompactColorDomainV1::OpenLines
        } else if !adjoint_labels.is_empty() {
            CompactColorDomainV1::SingleTrace
        } else {
            CompactColorDomainV1::Singlet
        };
        let helicity_count = checked_product(
            externals_by_public_slot
                .iter()
                .map(|external| external.public_helicities.len()),
            "test helicities",
        )
        .unwrap();
        let color_count = match color_domain {
            CompactColorDomainV1::Singlet => 1,
            CompactColorDomainV1::SingleTrace => {
                single_trace_color_count(adjoint_labels.len(), color_basis).unwrap()
            }
            CompactColorDomainV1::OpenLines => {
                checked_factorial(fundamental_labels.len(), "test pairings").unwrap()
                    * checked_factorial(adjoint_labels.len(), "test adjoints").unwrap()
                    * checked_binomial(
                        adjoint_labels.len() + fundamental_labels.len() - 1,
                        fundamental_labels.len() - 1,
                        "test splits",
                    )
                    .unwrap()
            }
        };
        OnTheFlyCompactSelectorAdapterV1 {
            external_permutation: (0..roles.len() as u32)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            representative_to_public: (0..roles.len() as u32)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            public_to_representative: (0..roles.len() as u32)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            public_label_by_representative_label: roles
                .iter()
                .map(|(label, _, _)| (*label, *label))
                .collect(),
            representative_label_by_public_label: roles
                .iter()
                .map(|(label, _, _)| (*label, *label))
                .collect(),
            externals_by_public_slot,
            public_slot_by_label,
            fundamental_labels,
            antifundamental_labels,
            adjoint_labels,
            color_domain,
            color_basis,
            reference_unfolded_color_ordinal: None,
            trace_reflections_folded: false,
            helicity_count,
            color_count,
        }
    }

    #[test]
    fn permutation_and_split_ranks_are_exact_inverses() {
        let values = [1, 3, 5, 7];
        for rank in 0..24 {
            let selected = nth_permutation(&values, rank).unwrap();
            assert_eq!(permutation_rank(&values, &selected).unwrap(), rank);
        }
        for total in 0..6 {
            for bins in 1..5 {
                let count = checked_binomial(total + bins - 1, bins - 1, "test").unwrap();
                for rank in 0..count {
                    let selected = nth_weak_composition(total, bins, rank).unwrap();
                    assert_eq!(selected.iter().sum::<usize>(), total);
                    assert_eq!(weak_composition_rank(&selected).unwrap(), rank);
                }
            }
        }
    }

    #[test]
    fn adjoint_color_ordinals_permute_only_between_both_fixed_anchors() {
        for (total, expected_count) in [(3, 1), (4, 2), (5, 6), (6, 24), (7, 120)] {
            let roles = (1..=total)
                .map(|label| (label, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1][..]))
                .collect::<Vec<_>>();
            let selector = adapter_in_basis(&roles, OnTheFlyColorBasis::Adjoint);
            assert_eq!(selector.color_count(), expected_count);
            assert_eq!(selector.helicity_count(), 1 << total);
            let mut words = BTreeSet::new();
            for index in 0..expected_count {
                let (word, _) = selector.color_at(index).unwrap();
                assert_eq!(word.first(), Some(&1));
                assert_eq!(word.last(), Some(&total));
                assert_eq!(selector.color_ordinal(&word).unwrap(), index);
                let (parsed, _) = selector.parse_color_id(&canonical_color_id(&word)).unwrap();
                assert_eq!(parsed, word);
                assert!(words.insert(word));
            }
            assert!(selector.color_at(expected_count).is_err());
        }
    }

    #[test]
    fn adjoint_reference_reordering_and_public_aliases_preserve_the_basis() {
        let mut selector = adapter_in_basis(
            &[
                (1, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
                (2, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
                (3, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
                (4, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            ],
            OnTheFlyColorBasis::Adjoint,
        );
        assert!(selector.parse_color_id("flow:1,2,4,3").is_err());
        assert!(selector.unfolded_color_ordinal(&[1, 2, 4, 3]).is_err());
        assert!(selector.unfolded_color_ordinal(&[2, 1, 3, 4]).is_err());
        assert!(selector.unfolded_color_ordinal(&[1, 4]).is_err());
        selector.reference_unfolded_color_ordinal = Some(1);
        for (index, expected) in [(0, [1, 3, 2, 4]), (1, [1, 2, 3, 4])] {
            let (word, _) = selector.color_at(index).unwrap();
            assert_eq!(word.as_ref(), expected);
            assert_eq!(selector.color_ordinal(&word).unwrap(), index);
        }
        let aliased = selector.with_public_permutation(&[3, 1, 2, 0]).unwrap();
        let (word, _) = aliased.parse_color_id("flow:4,3,2,1").unwrap();
        assert_eq!(word.as_ref(), [1, 3, 2, 4]);
        assert_eq!(aliased.color_ordinal(&word).unwrap(), 0);
    }

    #[test]
    fn adjoint_basis_rejects_a_non_adjoint_seed_at_the_adapter_boundary() {
        let seed = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        let mut policy = OnTheFlyLcSelectorPolicyV1::complete(None, false);
        policy.color_basis = OnTheFlyColorBasis::Adjoint;
        assert!(OnTheFlyCompactSelectorAdapterV1::from_seed(&seed, policy).is_err());
    }

    #[test]
    fn adjoint_color_ordinals_allow_certified_colourless_spectators() {
        // Model/coupling-order certification is generation-side. Here only
        // the structural selector domain is validated: singlets do not take
        // part in the two-anchor colour permutation.
        for gluon_count in [2, 3, 4] {
            let mut sources: Vec<_> = (1..=gluon_count)
                .map(|label| (label, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1][..]))
                .collect();
            sources.push((gluon_count + 1, OnTheFlyExternalColorRoleV1::Singlet, &[0]));
            let selector = adapter_in_basis(&sources, OnTheFlyColorBasis::Adjoint);
            assert_eq!(selector.helicity_count(), 1 << gluon_count);
            assert_eq!(
                selector.color_count(),
                single_trace_color_count(gluon_count as usize, OnTheFlyColorBasis::Adjoint)
                    .unwrap()
            );
            for index in 0..selector.color_count() {
                let (word, _) = selector.color_at(index).unwrap();
                assert_eq!(word.len(), gluon_count as usize);
                assert_eq!(word.first(), Some(&1));
                assert_eq!(word.last(), Some(&gluon_count));
                assert_eq!(selector.color_ordinal(&word).unwrap(), index);
            }
        }
    }

    #[test]
    fn public_ids_require_the_existing_canonical_spelling() {
        assert_eq!(canonical_helicity_id(&[-1, 0, 1]), "h:-1,+0,+1");
        assert_eq!(canonical_color_id(&[]), "flow:singlet");
        assert_eq!(canonical_color_id(&[2, 5, 1]), "flow:2,5,1");
    }

    #[test]
    fn direct_ordinal_decoder_matches_the_existing_lc_query_contract() {
        let seed = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        let selector = OnTheFlyCompactSelectorAdapterV1::from_seed(
            &seed,
            OnTheFlyLcSelectorPolicyV1::complete(None, false),
        )
        .unwrap();
        let decoded = selector.decoded_query_at(&seed, 0, 0).unwrap();
        let expected =
            DecodedLcQueryV1::new(&seed, vec![0, 1], &[0, 0], OnTheFlyLcSelectorV1::Singlet)
                .unwrap();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn six_fermion_compact_census_has_sixty_four_helicities_and_six_flows() {
        let selector = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Fundamental, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Antifundamental, &[-1, 1]),
            (3, OnTheFlyExternalColorRoleV1::Fundamental, &[-1, 1]),
            (4, OnTheFlyExternalColorRoleV1::Antifundamental, &[-1, 1]),
            (5, OnTheFlyExternalColorRoleV1::Fundamental, &[-1, 1]),
            (6, OnTheFlyExternalColorRoleV1::Antifundamental, &[-1, 1]),
        ]);
        assert_eq!(selector.helicity_count(), 64);
        assert_eq!(selector.color_count(), 6);
    }

    #[test]
    fn alias_permutation_round_trips_helicity_and_color_ids_without_changing_ordinals() {
        let representative = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Fundamental, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Antifundamental, &[-1, 1]),
            (3, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (4, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
        ]);
        let representative_helicity = representative.parse_helicity_id("h:+1,-1,+1,-1").unwrap();
        let (representative_color, _) = representative.parse_color_id("flow:1,3,4,2").unwrap();
        let helicity_ordinal = representative
            .helicity_ordinal(&representative_helicity)
            .unwrap();
        let color_ordinal = representative.color_ordinal(&representative_color).unwrap();

        let aliased = representative
            .with_public_permutation(&[0, 1, 3, 2])
            .unwrap();
        let aliased_helicity = aliased.parse_helicity_id("h:+1,-1,-1,+1").unwrap();
        let (aliased_color, _) = aliased.parse_color_id("flow:1,4,3,2").unwrap();
        assert_eq!(aliased_helicity, representative_helicity);
        assert_eq!(aliased_color, representative_color);
        assert_eq!(
            aliased.helicity_ordinal(&aliased_helicity).unwrap(),
            helicity_ordinal
        );
        assert_eq!(
            aliased.color_ordinal(&aliased_color).unwrap(),
            color_ordinal
        );
        assert_eq!(
            aliased
                .public_helicities_from_representative(&representative_helicity)
                .unwrap()
                .as_ref(),
            [1, -1, -1, 1]
        );
        assert_eq!(
            aliased
                .public_labels_from_representative(&representative_color)
                .unwrap()
                .as_ref(),
            [1, 4, 3, 2]
        );
    }

    #[test]
    fn deferred_public_physics_materializes_once_and_applies_alias_permutation_once() {
        use crate::{ArtifactProcess, ArtifactSelection, ExternalParticle, ParticleRole};

        let representative = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Fundamental, &[-1]),
            (2, OnTheFlyExternalColorRoleV1::Antifundamental, &[1]),
            (3, OnTheFlyExternalColorRoleV1::Adjoint, &[0]),
            (4, OnTheFlyExternalColorRoleV1::Adjoint, &[2]),
        ]);
        let external_particles = [
            ("a", 1, ParticleRole::Initial),
            ("b", -1, ParticleRole::Initial),
            ("c", 21, ParticleRole::Final),
            ("d", 22, ParticleRole::Final),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (particle, pdg, role))| ExternalParticle {
            index,
            label: index + 1,
            particle: particle.to_string(),
            pdg,
            role,
            momentum_slot: index,
            momentum_components: ["E", "px", "py", "pz"].map(str::to_string),
        })
        .collect();
        let metadata = super::super::on_the_fly_public_metadata::OnTheFlyPublicMetadataV1::for_test(
            "a_b_to_c_d",
            "a b > c d",
            external_particles,
        );
        let selection = ArtifactSelection {
            process: ArtifactProcess {
                id: "a_b_to_c_d".to_string(),
                expression: "a b > c d".to_string(),
                color_accuracy: "lc".to_string(),
                external_pdgs: vec![1, -1, 21, 22],
                physics_path: "processes/a_b_to_c_d/physics.json".to_string(),
                required_runtime_capabilities: Vec::new(),
                aliases: Vec::new(),
            },
            requested_id: "a_b_to_d_c".to_string(),
            alias: None,
            public_expression: "a b > d c".to_string(),
            external_pdgs: vec![1, -1, 22, 21],
            external_permutation: vec![0, 1, 3, 2],
            inferred_permutation: true,
            process_group: None,
        };
        let lazy = super::super::native_runtime::LazyProcessPhysicsV1::deferred_on_the_fly(
            metadata,
            representative,
            selection,
        );

        let first = lazy.get().unwrap();
        let second = lazy.get().unwrap();
        assert!(std::ptr::eq(first, second));
        assert_eq!(first.process_id, "a_b_to_d_c");
        assert_eq!(first.process, "a b > d c");
        assert_eq!(
            first
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .collect::<Vec<_>>(),
            [1, -1, 22, 21]
        );
        assert_eq!(first.helicities.len(), 1);
        assert_eq!(first.helicities[0].id, "h:-1,+1,+2,+0");
        assert_eq!(first.helicities[0].values, [-1, 1, 2, 0]);
        assert_eq!(first.color_components[0].id(), "flow:1,4,3,2");
        let crate::ColorComponent::LcFlow(flow) = &first.color_components[0] else {
            panic!("on-the-fly public color axis changed kind")
        };
        assert_eq!(flow.word, [1, 4, 3, 2]);
    }

    #[test]
    fn helicity_ids_validate_exact_seed_domains_and_canonical_signs() {
        let adapter = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Singlet, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Singlet, &[0]),
        ]);
        assert_eq!(adapter.helicity_count(), 2);
        assert_eq!(adapter.helicity_at(0).unwrap().as_ref(), [-1, 0]);
        assert_eq!(adapter.helicity_at(1).unwrap().as_ref(), [1, 0]);
        assert_eq!(
            adapter.parse_helicity_id("h:+1,+0").unwrap().as_ref(),
            [1, 0]
        );
        assert!(adapter.parse_helicity_id("h:1,+0").is_err());
        assert!(adapter.parse_helicity_id("h:+1,+1").is_err());
        assert!(adapter.parse_helicity_id("h:+1").is_err());
    }

    #[test]
    fn pure_adjoint_flows_keep_the_minimum_label_as_cyclic_anchor() {
        let adapter = adapter(&[
            (3, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (1, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
        ]);
        assert_eq!(adapter.color_count(), 2);
        assert_eq!(adapter.color_at(0).unwrap().0.as_ref(), [1, 2, 3]);
        assert_eq!(adapter.color_at(1).unwrap().0.as_ref(), [1, 3, 2]);
        assert!(adapter.parse_color_id("flow:1,3,2").is_ok());
        assert!(adapter.parse_color_id("flow:2,3,1").is_err());
        assert!(adapter.parse_color_id("flow:01,3,2").is_err());
    }

    #[test]
    fn reference_word_reorders_one_complete_flow_without_changing_the_axis() {
        let mut adapter = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (3, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
        ]);
        adapter.reference_unfolded_color_ordinal = Some(1);
        assert_eq!(adapter.color_at(0).unwrap().0.as_ref(), [1, 3, 2]);
        assert_eq!(adapter.color_at(1).unwrap().0.as_ref(), [1, 2, 3]);
        assert_eq!(adapter.color_ordinal(&[1, 3, 2]).unwrap(), 0);
        assert_eq!(adapter.color_ordinal(&[1, 2, 3]).unwrap(), 1);
    }

    #[test]
    fn folded_trace_axis_retains_each_public_alias_in_generator_order() {
        let mut adapter = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (3, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (4, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
        ]);
        adapter.trace_reflections_folded = true;
        let expected = [
            [1, 2, 3, 4],
            [1, 4, 3, 2],
            [1, 2, 4, 3],
            [1, 3, 4, 2],
            [1, 3, 2, 4],
            [1, 4, 2, 3],
        ];
        for (index, word) in expected.iter().enumerate() {
            assert_eq!(adapter.color_at(index).unwrap().0.as_ref(), word);
            assert_eq!(adapter.color_ordinal(word).unwrap(), index);
        }

        adapter.reference_unfolded_color_ordinal = Some(2);
        let with_reference = [
            [1, 3, 2, 4],
            [1, 4, 2, 3],
            [1, 2, 3, 4],
            [1, 4, 3, 2],
            [1, 2, 4, 3],
            [1, 3, 4, 2],
        ];
        for (index, word) in with_reference.iter().enumerate() {
            assert_eq!(adapter.color_at(index).unwrap().0.as_ref(), word);
            assert_eq!(adapter.color_ordinal(word).unwrap(), index);
        }
    }

    #[test]
    fn open_line_lazy_order_matches_the_established_generator() {
        let adapter = adapter(&[
            (1, OnTheFlyExternalColorRoleV1::Antifundamental, &[-1, 1]),
            (2, OnTheFlyExternalColorRoleV1::Fundamental, &[-1, 1]),
            (3, OnTheFlyExternalColorRoleV1::Fundamental, &[-1, 1]),
            (4, OnTheFlyExternalColorRoleV1::Antifundamental, &[-1, 1]),
            (5, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
            (6, OnTheFlyExternalColorRoleV1::Adjoint, &[-1, 1]),
        ]);
        assert_eq!(adapter.color_count(), 12);
        let first = [
            [2, 1, 3, 5, 6, 4],
            [2, 5, 1, 3, 6, 4],
            [2, 5, 6, 1, 3, 4],
            [2, 1, 3, 6, 5, 4],
        ];
        for (index, expected) in first.iter().enumerate() {
            let (actual, _) = adapter.color_at(index).unwrap();
            assert_eq!(actual.as_ref(), expected);
            assert_eq!(adapter.color_ordinal(&actual).unwrap(), index);
            assert!(adapter.parse_color_id(&canonical_color_id(&actual)).is_ok());
        }
        assert!(adapter.parse_color_id("flow:3,5,6,4,2,1").is_err());
        assert!(adapter.parse_color_id("flow:2,5,1,4,6,3").is_err());
    }
}
