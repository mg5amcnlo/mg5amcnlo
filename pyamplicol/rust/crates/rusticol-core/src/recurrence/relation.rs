// SPDX-License-Identifier: 0BSD

//! Opt-in numerical nomination and exact replay for recurrence-current reuse.
//!
//! Numerical projections are deliberately only an index. A current is
//! reusable only when the complete lowered contribution vector is
//! proportional over [`ExactComplexRational`]. The resulting certificate is
//! replayed once more before lowering may emit a scale-copy row.

use std::collections::BTreeMap;

use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};

use super::{
    DIRECT_NONE_U32, DirectExecutorRole, DirectNodeKind, DirectRecurrencePlan,
    ExactComplexRational, RecurrenceStrategy,
};
use crate::{RusticolError, RusticolResult};

const PROJECTION_PRIME: u64 = (1_u64 << 61) - 1;
const RELATION_CERTIFICATE_ALGORITHM: &str = "recurrence-exact-rational-row-vector-replay-v1";
pub const NUMERICAL_RELATION_CERTIFICATE_ALGORITHM: &str =
    "authenticated-independent-recursive-decimal-raw-probes-v2";
pub const RECURRENCE_NUMERICAL_SOURCE_ABI: &str =
    "pyamplicol-recurrence-numerical-current-source-v3";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("recurrence relation discovery: {}", message.into()))
}

/// User-visible recurrence relation-discovery mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecurrenceRelationDiscoveryMode {
    Off,
    Diagnostic,
    CertifiedReuse,
}

impl RecurrenceRelationDiscoveryMode {
    pub fn parse(value: &str) -> RusticolResult<Self> {
        match value {
            "off" => Ok(Self::Off),
            "diagnostic" => Ok(Self::Diagnostic),
            "certified-reuse" => Ok(Self::CertifiedReuse),
            _ => Err(invalid(format!("unsupported mode {value:?}"))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Diagnostic => "diagnostic",
            Self::CertifiedReuse => "certified-reuse",
        }
    }

    pub const fn applies_reuse(self) -> bool {
        matches!(self, Self::CertifiedReuse)
    }
}

/// Authenticated options passed through the recurrence lowering boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct RecurrenceRelationDiscoveryOptions {
    pub mode: RecurrenceRelationDiscoveryMode,
    pub precision_digits: u32,
    pub probe_count: u32,
    pub verification_probe_count: u32,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub seed: u64,
    pub color_accuracy: String,
    pub numerical_evidence: Option<RecurrenceNumericalRelationEvidence>,
}

impl RecurrenceRelationDiscoveryOptions {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: RecurrenceRelationDiscoveryMode,
        precision_digits: u32,
        probe_count: u32,
        verification_probe_count: u32,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        seed: u64,
        color_accuracy: impl Into<String>,
    ) -> RusticolResult<Self> {
        if precision_digits < 80 {
            return Err(invalid("precision must be at least 80 decimal digits"));
        }
        if probe_count < 2 {
            return Err(invalid("at least two probes are required"));
        }
        if verification_probe_count < 2 {
            return Err(invalid("at least two verification probes are required"));
        }
        if !relative_tolerance.is_finite()
            || relative_tolerance < 0.0
            || !absolute_tolerance.is_finite()
            || absolute_tolerance < 0.0
            || (relative_tolerance == 0.0 && absolute_tolerance == 0.0)
        {
            return Err(invalid(
                "relative and absolute tolerances must be finite and nonnegative, and at least one must be positive",
            ));
        }
        let color_accuracy = color_accuracy.into();
        if !matches!(color_accuracy.as_str(), "lc" | "nlc" | "full") {
            return Err(invalid(format!(
                "unsupported color accuracy {color_accuracy:?}"
            )));
        }
        Ok(Self {
            mode,
            precision_digits,
            probe_count,
            verification_probe_count,
            relative_tolerance,
            absolute_tolerance,
            seed,
            color_accuracy,
            numerical_evidence: None,
        })
    }

    pub fn with_numerical_evidence(
        mut self,
        evidence: RecurrenceNumericalRelationEvidence,
    ) -> RusticolResult<Self> {
        if self.mode == RecurrenceRelationDiscoveryMode::Off {
            return Err(invalid("off mode cannot carry numerical relation evidence"));
        }
        if evidence.requested_mode != self.mode {
            return Err(invalid(
                "numerical evidence mode disagrees with lowering options",
            ));
        }
        if evidence.certificate_algorithm != NUMERICAL_RELATION_CERTIFICATE_ALGORITHM {
            return Err(invalid(
                "numerical evidence uses an unsupported certificate algorithm",
            ));
        }
        if evidence.precision_digits != self.precision_digits
            || evidence.probe_count != self.probe_count
            || evidence.verification_probe_count != self.verification_probe_count
            || evidence.relative_tolerance.to_bits() != self.relative_tolerance.to_bits()
            || evidence.absolute_tolerance.to_bits() != self.absolute_tolerance.to_bits()
            || evidence.seed != self.seed
        {
            return Err(invalid(
                "numerical evidence probe contract disagrees with lowering options",
            ));
        }
        let expected_candidate_count = evidence
            .mappings
            .len()
            .checked_add(evidence.verification_rejected_count)
            .ok_or_else(|| invalid("numerical evidence candidate count overflows usize"))?;
        let expected_rejected_count = evidence
            .tested_hypothesis_count
            .checked_sub(evidence.mappings.len())
            .ok_or_else(|| invalid("numerical evidence rejected-hypothesis count underflows"))?;
        if evidence.numerical_candidate_count != expected_candidate_count
            || evidence.tested_hypothesis_count < evidence.numerical_candidate_count
            || evidence.rejected_hypothesis_count != expected_rejected_count
        {
            return Err(invalid(
                "numerical evidence candidate counters are inconsistent",
            ));
        }
        self.numerical_evidence = Some(evidence);
        Ok(self)
    }

    pub fn off() -> Self {
        Self {
            mode: RecurrenceRelationDiscoveryMode::Off,
            precision_digits: 96,
            probe_count: 4,
            verification_probe_count: 4,
            relative_tolerance: 1.0e-70,
            absolute_tolerance: 1.0e-80,
            seed: 0x5059_414d,
            color_accuracy: "lc".to_owned(),
            numerical_evidence: None,
        }
    }
}

/// One Python-authenticated exact ±1/zero scale mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceNumericalCurrentMapping {
    pub current_id: u32,
    pub representative_id: Option<u32>,
    pub execution_representative_id: u32,
    pub relation_kind: String,
    pub factor: ExactComplexRational,
    pub current_dimension: u16,
    pub certificate_proof_sha256: String,
    pub candidate_observations_sha256: String,
    pub verification_observations_sha256: String,
}

/// Canonical evidence supplied only after an unpublished exact baseline pass.
#[derive(Clone, Debug, PartialEq)]
pub struct RecurrenceNumericalRelationEvidence {
    pub requested_mode: RecurrenceRelationDiscoveryMode,
    pub schedule_semantic_digest: String,
    pub baseline_runtime_layout_digest: String,
    pub source_semantics_sha256: String,
    pub runtime_parameter_schema_sha256: String,
    pub candidate_observation_batch_sha256: String,
    pub verification_observation_batch_sha256: String,
    pub decision_sha256: String,
    pub rejection_decision_sha256: String,
    pub certificate_algorithm: String,
    pub certificate_set_sha256: String,
    pub precision_digits: u32,
    pub probe_count: u32,
    pub verification_probe_count: u32,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub seed: u64,
    pub numerical_candidate_count: usize,
    pub verification_rejected_count: usize,
    pub rejected_hypothesis_count: usize,
    pub tested_hypothesis_count: usize,
    pub mappings: Vec<RecurrenceNumericalCurrentMapping>,
}

/// Recompute the source contract consumed by the exact Python probe pass.
///
/// This deliberately binds the baseline physical layout as well as every
/// current comparison contract. The numerical evidence is rejected unless
/// its claimed digest is byte-for-byte identical to this native result.
pub fn recurrence_numerical_source_semantics_sha256(
    plan: &DirectRecurrencePlan,
    process_id: &str,
) -> RusticolResult<String> {
    let mut finalization_executors = BTreeMap::<u32, u32>::new();
    for group in plan
        .row_groups()
        .iter()
        .filter(|group| group.role == DirectExecutorRole::Finalization)
    {
        let start = u32::try_from(group.row_start)
            .map_err(|_| invalid("finalization row-group start exceeds u32"))?;
        let stop = start
            .checked_add(group.row_count)
            .ok_or_else(|| invalid("finalization row-group range overflows u32"))?;
        for row_id in start..stop {
            if finalization_executors
                .insert(row_id, group.direct_executor_id)
                .is_some()
            {
                return Err(invalid(
                    "baseline finalization row belongs to multiple executor groups",
                ));
            }
        }
    }

    let currents = plan
        .currents()
        .iter()
        .map(|current| {
            let (executor_id, momentum_form_id, factor) = if current.finalization_row_or_sentinel
                == DIRECT_NONE_U32
            {
                (
                    DIRECT_NONE_U32,
                    DIRECT_NONE_U32,
                    [
                        "1".to_owned(),
                        "1".to_owned(),
                        "0".to_owned(),
                        "1".to_owned(),
                    ],
                )
            } else {
                let row = plan
                    .finalizations()
                    .get(current.finalization_row_or_sentinel as usize)
                    .ok_or_else(|| invalid("baseline current finalization row is absent"))?;
                let executor_id = *finalization_executors
                    .get(&current.finalization_row_or_sentinel)
                    .ok_or_else(|| invalid("baseline current finalization executor is absent"))?;
                let factor = plan
                    .exact_factors()
                    .get(row.exact_factor_id as usize)
                    .ok_or_else(|| invalid("baseline finalization factor is absent"))?;
                (
                    executor_id,
                    row.momentum_form_id,
                    [
                        factor.real().numerator().to_string(),
                        factor.real().denominator().to_string(),
                        factor.imag().numerator().to_string(),
                        factor.imag().denominator().to_string(),
                    ],
                )
            };
            Ok(json!({
                "current_id": current.semantic_current_id,
                "is_source": current.node_kind == DirectNodeKind::Source,
                "contract": [
                    current.node_kind as u16,
                    current.stage,
                    current.state_template_id,
                    current.component_count,
                    current.momentum_form_id,
                    current.selector_domain_id,
                    executor_id,
                    momentum_form_id,
                    factor,
                ],
            }))
        })
        .collect::<RusticolResult<Vec<JsonValue>>>()?;
    let selector_schedule = match plan.strategy() {
        RecurrenceStrategy::TopologyReplay => json!({
            "policy": "seeded-replay-target-per-physical-point-v1",
            "replay_targets": plan
                .replay_targets()
                .iter()
                .enumerate()
                .map(|(target_index, target)| json!({
                    "target_index": target_index,
                    "public_flow_id": target.public_flow_id,
                    "representative_id": target.representative_id,
                    "selector_domain_id": target.selector_domain_id,
                }))
                .collect::<Vec<_>>(),
        }),
        RecurrenceStrategy::AllFlowUnion => json!({
            "policy": "seeded-helicity-per-selector-domain-and-physical-point-v1",
            "resolved_helicities": plan
                .resolved_helicities()
                .iter()
                .enumerate()
                .map(|(helicity_index, helicity)| json!({
                    "helicity_index": helicity_index,
                    "helicity_id": helicity.id,
                    "selector_domain_id": helicity.selector_domain_id,
                }))
                .collect::<Vec<_>>(),
        }),
        RecurrenceStrategy::ContractedColorUnion => json!({
            "policy": "fixed-contracted-source-schedule-v1",
            "fixed_source_schedule": true,
        }),
    };
    let payload = json!({
        "abi": RECURRENCE_NUMERICAL_SOURCE_ABI,
        "process_id": process_id,
        "strategy": plan.strategy().as_str(),
        "schedule_semantic_digest": plan.semantic_digest().to_string(),
        "baseline_runtime_layout_digest": plan.runtime_layout_digest().to_string(),
        "selector_schedule": selector_schedule,
        "currents": currents,
    });
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| invalid(format!("could not canonicalize source semantics: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn authenticate_recurrence_numerical_relation_provenance(
    evidence: &RecurrenceNumericalRelationEvidence,
    baseline_plan: &DirectRecurrencePlan,
    process_id: &str,
) -> RusticolResult<()> {
    if evidence.baseline_runtime_layout_digest != baseline_plan.runtime_layout_digest().to_string()
    {
        return Err(RusticolError::integrity(
            "numerical relation evidence baseline runtime-layout digest does not match the native baseline",
        ));
    }
    let expected_source_semantics =
        recurrence_numerical_source_semantics_sha256(baseline_plan, process_id)?;
    if evidence.source_semantics_sha256 != expected_source_semantics {
        return Err(RusticolError::integrity(
            "numerical relation evidence source-semantics digest does not match the native baseline",
        ));
    }
    Ok(())
}

/// Runtime contract which must agree before two current values are comparable.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct RelationCurrentContract {
    pub stage: u16,
    pub state_template_id: u32,
    pub component_count: u16,
    pub momentum_form_id: u32,
    pub selector_domain_id: u32,
    pub finalization_executor_id: u32,
    pub finalization_momentum_form_id: u32,
    pub finalization_exact_factor: ExactComplexRational,
}

/// One exact lowered contribution before parent relation substitution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RelationTermInput {
    pub executor_id: u32,
    pub parent_current_ids: Box<[u32]>,
    pub parent_momentum_form_ids: Box<[u32]>,
    pub exact_factor: ExactComplexRational,
}

/// Exact schedule input for one current in topological ID order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RelationCurrentInput {
    pub current_id: u32,
    pub is_source: bool,
    pub contract: RelationCurrentContract,
    pub terms: Box<[RelationTermInput]>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RelationTermKey {
    executor_id: u32,
    parent_class_ids: Box<[u32]>,
    parent_momentum_form_ids: Box<[u32]>,
}

type ExactTermVector = BTreeMap<RelationTermKey, ExactComplexRational>;

/// Exact, independently replayable relation between two lowered currents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceCurrentRelationCertificate {
    pub current_id: u32,
    pub representative_id: u32,
    pub factor: ExactComplexRational,
    pub current_expression_sha256: String,
    pub representative_expression_sha256: String,
    pub proof_sha256: String,
}

/// Manifest-facing result of one recurrence relation-discovery pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceRelationDiscoveryReport {
    pub requested_mode: RecurrenceRelationDiscoveryMode,
    pub state: &'static str,
    pub strategy: RecurrenceStrategy,
    pub color_accuracy: String,
    pub precision_digits: u32,
    pub probe_count: u32,
    pub verification_probe_count: u32,
    pub seed: u64,
    pub certificate_algorithm: String,
    pub authenticated_certificate_set_sha256: Option<String>,
    pub authenticated_runtime_parameter_schema_sha256: Option<String>,
    pub authenticated_candidate_observation_batch_sha256: Option<String>,
    pub authenticated_verification_observation_batch_sha256: Option<String>,
    pub authenticated_decision_sha256: Option<String>,
    pub rejected_decision_sha256: String,
    pub tested_hypothesis_count: usize,
    pub verification_rejected_count: usize,
    pub rejected_hypothesis_count: usize,
    pub effective_projection_count: u32,
    pub numerical_candidate_count: usize,
    pub uncertified_candidate_count: usize,
    pub exact_certified_relation_count: usize,
    pub applied_relation_count: usize,
    pub current_count_before: usize,
    pub current_count_after: usize,
    pub contribution_count_before: usize,
    pub contribution_count_after: usize,
    pub interaction_evaluation_count_before: usize,
    pub interaction_evaluation_count_after: usize,
    pub scale_copy_row_count: usize,
    pub certificates: Vec<RecurrenceCurrentRelationCertificate>,
    pub rejected_candidates: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct RecurrenceRelationDiscoveryOutcome {
    pub relations: Vec<RecurrenceCurrentRelationCertificate>,
    pub numerical_candidate_count: usize,
    pub uncertified_candidate_count: usize,
    pub rejected_hypothesis_count: usize,
    pub rejected_candidates: Vec<String>,
    pub rejected_decision_sha256: String,
    pub effective_projection_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Fp2 {
    re: u64,
    im: u64,
}

impl Fp2 {
    const ZERO: Self = Self { re: 0, im: 0 };

    fn add(self, right: Self) -> Self {
        Self {
            re: add_mod(self.re, right.re),
            im: add_mod(self.im, right.im),
        }
    }

    fn mul(self, right: Self) -> Self {
        Self {
            re: sub_mod(mul_mod(self.re, right.re), mul_mod(self.im, right.im)),
            im: add_mod(mul_mod(self.re, right.im), mul_mod(self.im, right.re)),
        }
    }

    fn inverse(self) -> Option<Self> {
        let norm = add_mod(mul_mod(self.re, self.re), mul_mod(self.im, self.im));
        if norm == 0 {
            return None;
        }
        let inverse = pow_mod(norm, PROJECTION_PRIME - 2);
        Some(Self {
            re: mul_mod(self.re, inverse),
            im: sub_mod(0, mul_mod(self.im, inverse)),
        })
    }
}

/// Discover recurrence-current relations in topological order.
///
/// The finite-field fingerprint nominates candidates with a collision bound
/// controlled by `precision_digits` and `probe_count`. It is never accepted
/// as proof: `exact_relation_factor` checks every exact coefficient, and
/// `replay_certificate` repeats that check from the stored vectors.
pub(crate) fn discover_recurrence_current_relations(
    inputs: &[RelationCurrentInput],
    options: &RecurrenceRelationDiscoveryOptions,
) -> RusticolResult<RecurrenceRelationDiscoveryOutcome> {
    let rejection_root = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&json!({
                "abi": "rusticol-recurrence-rejected-decision-chain-v1",
                "precision_digits": options.precision_digits,
                "probe_count": options.probe_count,
                "seed": options.seed,
            }))
            .map_err(|error| invalid(format!("could not encode rejection root: {error}")))?,
        )
    );
    if options.mode == RecurrenceRelationDiscoveryMode::Off {
        return Ok(RecurrenceRelationDiscoveryOutcome {
            relations: Vec::new(),
            numerical_candidate_count: 0,
            uncertified_candidate_count: 0,
            rejected_hypothesis_count: 0,
            rejected_candidates: Vec::new(),
            rejected_decision_sha256: rejection_root,
            effective_projection_count: 0,
        });
    }
    for (index, input) in inputs.iter().enumerate() {
        if input.current_id != index as u32 {
            return Err(invalid(format!(
                "current {index} has non-canonical ID {}",
                input.current_id
            )));
        }
    }

    // Four bits per requested decimal digit is conservative relative to
    // log2(10). Each independent projection lives in F_(2^61-1)^2.
    let precision_projection_count = options.precision_digits.saturating_mul(4).div_ceil(61);
    let effective_projection_count = options.probe_count.max(precision_projection_count).max(2);

    let mut class_by_current = Vec::<u32>::with_capacity(inputs.len());
    let mut factor_by_current = Vec::<ExactComplexRational>::with_capacity(inputs.len());
    let mut vector_by_current = Vec::<ExactTermVector>::with_capacity(inputs.len());
    let mut groups = BTreeMap::<(RelationCurrentContract, Vec<Fp2>), Vec<u32>>::new();
    let mut relations = Vec::new();
    let mut numerical_candidate_count = 0usize;
    let mut uncertified_candidate_count = 0usize;
    let mut rejected_candidates = Vec::new();
    let mut rejected_decision_sha256 = rejection_root;

    for input in inputs {
        if input.is_source {
            class_by_current.push(input.current_id);
            factor_by_current.push(ExactComplexRational::ONE);
            vector_by_current.push(BTreeMap::new());
            continue;
        }
        let vector = exact_term_vector(input, &class_by_current, &factor_by_current)?;
        let Some(fingerprint) =
            numerical_fingerprint(&vector, options.seed, effective_projection_count)?
        else {
            class_by_current.push(input.current_id);
            factor_by_current.push(ExactComplexRational::ONE);
            vector_by_current.push(vector);
            continue;
        };

        let key = (input.contract.clone(), fingerprint);
        let mut relation = None;
        if let Some(representatives) = groups.get(&key) {
            for representative_id in representatives {
                numerical_candidate_count = numerical_candidate_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("candidate count overflows usize"))?;
                let representative = vector_by_current
                    .get(*representative_id as usize)
                    .ok_or_else(|| invalid("candidate representative is absent"))?;
                match exact_relation_factor(&vector, representative)? {
                    Some(factor) => {
                        let certificate = build_certificate(
                            input.current_id,
                            *representative_id,
                            factor,
                            &vector,
                            representative,
                        );
                        if replay_certificate(&certificate, &vector, representative)? {
                            relation = Some(certificate);
                            break;
                        }
                        uncertified_candidate_count += 1;
                        let reason = format!(
                            "current {} and representative {} failed exact certificate replay",
                            input.current_id, representative_id
                        );
                        let mut digest = Sha256::new();
                        digest.update(rejected_decision_sha256.as_bytes());
                        digest.update(reason.as_bytes());
                        rejected_decision_sha256 = format!("{:x}", digest.finalize());
                        if rejected_candidates.len() < 16 {
                            rejected_candidates.push(reason);
                        }
                    }
                    None => {
                        uncertified_candidate_count += 1;
                        let reason = format!(
                            "current {} and representative {} collided numerically without an exact relation",
                            input.current_id, representative_id
                        );
                        let mut digest = Sha256::new();
                        digest.update(rejected_decision_sha256.as_bytes());
                        digest.update(reason.as_bytes());
                        rejected_decision_sha256 = format!("{:x}", digest.finalize());
                        if rejected_candidates.len() < 16 {
                            rejected_candidates.push(reason);
                        }
                    }
                }
            }
        }
        if let Some(certificate) = relation {
            let representative_id = certificate.representative_id;
            class_by_current.push(class_by_current[representative_id as usize]);
            factor_by_current.push(certificate.factor);
            relations.push(certificate);
        } else {
            class_by_current.push(input.current_id);
            factor_by_current.push(ExactComplexRational::ONE);
            groups.entry(key).or_default().push(input.current_id);
        }
        vector_by_current.push(vector);
    }

    Ok(RecurrenceRelationDiscoveryOutcome {
        relations,
        numerical_candidate_count,
        uncertified_candidate_count,
        rejected_hypothesis_count: uncertified_candidate_count,
        rejected_candidates,
        rejected_decision_sha256,
        effective_projection_count,
    })
}

fn exact_term_vector(
    input: &RelationCurrentInput,
    class_by_current: &[u32],
    factor_by_current: &[ExactComplexRational],
) -> RusticolResult<ExactTermVector> {
    let mut vector = BTreeMap::<RelationTermKey, ExactComplexRational>::new();
    for term in input.terms.iter() {
        if term.parent_current_ids.len() != term.parent_momentum_form_ids.len()
            || term.parent_current_ids.is_empty()
            || term.parent_current_ids.len() > 2
        {
            return Err(invalid(format!(
                "current {} has an invalid relation term arity",
                input.current_id
            )));
        }
        let mut coefficient = term.exact_factor;
        let mut parent_class_ids = Vec::with_capacity(term.parent_current_ids.len());
        for parent_id in term.parent_current_ids.iter().copied() {
            if parent_id >= input.current_id {
                return Err(invalid(format!(
                    "current {} relation term references non-topological parent {parent_id}",
                    input.current_id
                )));
            }
            parent_class_ids.push(
                *class_by_current
                    .get(parent_id as usize)
                    .ok_or_else(|| invalid("relation parent class is absent"))?,
            );
            coefficient = coefficient.checked_mul(
                *factor_by_current
                    .get(parent_id as usize)
                    .ok_or_else(|| invalid("relation parent factor is absent"))?,
            )?;
        }
        let key = RelationTermKey {
            executor_id: term.executor_id,
            parent_class_ids: parent_class_ids.into_boxed_slice(),
            parent_momentum_form_ids: term.parent_momentum_form_ids.clone(),
        };
        let combined = vector
            .get(&key)
            .copied()
            .unwrap_or(ExactComplexRational::ZERO)
            .checked_add(coefficient)?;
        if combined.is_zero() {
            vector.remove(&key);
        } else {
            vector.insert(key, combined);
        }
    }
    Ok(vector)
}

fn numerical_fingerprint(
    vector: &ExactTermVector,
    seed: u64,
    projection_count: u32,
) -> RusticolResult<Option<Vec<Fp2>>> {
    if vector.is_empty() {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(projection_count as usize);
    for projection in 0..projection_count {
        let mut value = Fp2::ZERO;
        for (key, coefficient) in vector {
            value =
                value.add(exact_to_fp2(*coefficient)?.mul(term_projection(seed, projection, key)));
        }
        values.push(value);
    }
    let Some(pivot) = values.iter().copied().find(|value| *value != Fp2::ZERO) else {
        return Ok(None);
    };
    let Some(inverse) = pivot.inverse() else {
        return Ok(None);
    };
    Ok(Some(
        values.into_iter().map(|value| value.mul(inverse)).collect(),
    ))
}

fn term_projection(seed: u64, projection: u32, key: &RelationTermKey) -> Fp2 {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-relation-probe-v1");
    hash.update(seed.to_le_bytes());
    hash.update(projection.to_le_bytes());
    hash.update(key.executor_id.to_le_bytes());
    hash.update((key.parent_class_ids.len() as u64).to_le_bytes());
    for value in key.parent_class_ids.iter() {
        hash.update(value.to_le_bytes());
    }
    for value in key.parent_momentum_form_ids.iter() {
        hash.update(value.to_le_bytes());
    }
    let digest = hash.finalize();
    let re = u64::from_le_bytes(digest[0..8].try_into().expect("SHA slice"));
    let im = u64::from_le_bytes(digest[8..16].try_into().expect("SHA slice"));
    Fp2 {
        re: re % PROJECTION_PRIME,
        im: im % PROJECTION_PRIME,
    }
}

fn exact_to_fp2(value: ExactComplexRational) -> RusticolResult<Fp2> {
    Ok(Fp2 {
        re: rational_to_field(value.real())?,
        im: rational_to_field(value.imag())?,
    })
}

fn rational_to_field(value: super::ExactRational) -> RusticolResult<u64> {
    let numerator = field_residue(value.numerator());
    let denominator = field_residue(value.denominator());
    if denominator == 0 {
        return Err(invalid(
            "an exact denominator is zero in the numerical projection field",
        ));
    }
    Ok(mul_mod(
        numerator,
        pow_mod(denominator, PROJECTION_PRIME - 2),
    ))
}

fn field_residue(value: i128) -> u64 {
    value.rem_euclid(i128::from(PROJECTION_PRIME)) as u64
}

fn exact_relation_factor(
    current: &ExactTermVector,
    representative: &ExactTermVector,
) -> RusticolResult<Option<ExactComplexRational>> {
    if current.len() != representative.len() || current.is_empty() {
        return Ok(None);
    }
    let Some((
        (current_key, current_coefficient),
        (representative_key, representative_coefficient),
    )) = current
        .first_key_value()
        .zip(representative.first_key_value())
    else {
        return Ok(None);
    };
    if current_key != representative_key {
        return Ok(None);
    }
    let factor = current_coefficient.checked_div(*representative_coefficient)?;
    if factor.is_zero() {
        return Ok(None);
    }
    for ((current_key, current_coefficient), (representative_key, representative_coefficient)) in
        current.iter().zip(representative)
    {
        if current_key != representative_key
            || *current_coefficient != representative_coefficient.checked_mul(factor)?
        {
            return Ok(None);
        }
    }
    Ok(Some(factor))
}

fn build_certificate(
    current_id: u32,
    representative_id: u32,
    factor: ExactComplexRational,
    current: &ExactTermVector,
    representative: &ExactTermVector,
) -> RecurrenceCurrentRelationCertificate {
    let current_expression_sha256 = expression_digest(current);
    let representative_expression_sha256 = expression_digest(representative);
    let proof_sha256 = proof_digest(
        current_id,
        representative_id,
        factor,
        &current_expression_sha256,
        &representative_expression_sha256,
    );
    RecurrenceCurrentRelationCertificate {
        current_id,
        representative_id,
        factor,
        current_expression_sha256,
        representative_expression_sha256,
        proof_sha256,
    }
}

fn replay_certificate(
    certificate: &RecurrenceCurrentRelationCertificate,
    current: &ExactTermVector,
    representative: &ExactTermVector,
) -> RusticolResult<bool> {
    Ok(
        exact_relation_factor(current, representative)? == Some(certificate.factor)
            && expression_digest(current) == certificate.current_expression_sha256
            && expression_digest(representative) == certificate.representative_expression_sha256
            && proof_digest(
                certificate.current_id,
                certificate.representative_id,
                certificate.factor,
                &certificate.current_expression_sha256,
                &certificate.representative_expression_sha256,
            ) == certificate.proof_sha256,
    )
}

fn expression_digest(vector: &ExactTermVector) -> String {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-exact-row-vector-v1");
    hash.update((vector.len() as u64).to_le_bytes());
    for (key, coefficient) in vector {
        hash.update(key.executor_id.to_le_bytes());
        hash.update((key.parent_class_ids.len() as u64).to_le_bytes());
        for value in key.parent_class_ids.iter() {
            hash.update(value.to_le_bytes());
        }
        for value in key.parent_momentum_form_ids.iter() {
            hash.update(value.to_le_bytes());
        }
        update_exact_hash(&mut hash, *coefficient);
    }
    format!("{:x}", hash.finalize())
}

fn proof_digest(
    current_id: u32,
    representative_id: u32,
    factor: ExactComplexRational,
    current_digest: &str,
    representative_digest: &str,
) -> String {
    let mut hash = Sha256::new();
    hash.update(RELATION_CERTIFICATE_ALGORITHM.as_bytes());
    hash.update(current_id.to_le_bytes());
    hash.update(representative_id.to_le_bytes());
    update_exact_hash(&mut hash, factor);
    hash.update(current_digest.as_bytes());
    hash.update(representative_digest.as_bytes());
    format!("{:x}", hash.finalize())
}

fn update_exact_hash(hash: &mut Sha256, value: ExactComplexRational) {
    for part in [
        value.real().numerator(),
        value.real().denominator(),
        value.imag().numerator(),
        value.imag().denominator(),
    ] {
        let text = part.to_string();
        hash.update((text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
}

fn add_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(PROJECTION_PRIME)) as u64
}

fn sub_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(PROJECTION_PRIME) - u128::from(right))
        % u128::from(PROJECTION_PRIME)) as u64
}

fn mul_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(PROJECTION_PRIME)) as u64
}

fn pow_mod(mut value: u64, mut exponent: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod(result, value);
        }
        value = mul_mod(value, value);
        exponent >>= 1;
    }
    result
}

pub fn relation_certificate_algorithm() -> &'static str {
    RELATION_CERTIFICATE_ALGORITHM
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::ExactRational;

    fn factor(value: i128) -> ExactComplexRational {
        ExactComplexRational::new(
            ExactRational::new(value, 1).expect("integer factor"),
            ExactRational::ZERO,
        )
    }

    fn contract(stage: u16) -> RelationCurrentContract {
        RelationCurrentContract {
            stage,
            state_template_id: 11,
            component_count: 2,
            momentum_form_id: 17,
            selector_domain_id: 0,
            finalization_executor_id: 9,
            finalization_momentum_form_id: 17,
            finalization_exact_factor: ExactComplexRational::ONE,
        }
    }

    fn term(
        executor_id: u32,
        parent_current_id: u32,
        exact_factor: ExactComplexRational,
    ) -> RelationTermInput {
        RelationTermInput {
            executor_id,
            parent_current_ids: vec![parent_current_id].into_boxed_slice(),
            parent_momentum_form_ids: vec![3].into_boxed_slice(),
            exact_factor,
        }
    }

    #[test]
    fn relation_options_reject_a_zero_width_tolerance_contract() {
        let error = RecurrenceRelationDiscoveryOptions::new(
            RecurrenceRelationDiscoveryMode::CertifiedReuse,
            96,
            4,
            4,
            0.0,
            0.0,
            0x5059_414d,
            "lc",
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least one must be positive"));
    }

    #[test]
    fn numerical_nomination_requires_exact_row_vector_replay() {
        let inputs = vec![
            RelationCurrentInput {
                current_id: 0,
                is_source: true,
                contract: contract(0),
                terms: Vec::new().into_boxed_slice(),
            },
            RelationCurrentInput {
                current_id: 1,
                is_source: false,
                contract: contract(1),
                terms: vec![term(7, 0, factor(1))].into_boxed_slice(),
            },
            RelationCurrentInput {
                current_id: 2,
                is_source: false,
                contract: contract(1),
                terms: vec![term(7, 0, factor(2))].into_boxed_slice(),
            },
            // The same projected algebra under an incompatible runtime
            // contract must never be nominated for reuse.
            RelationCurrentInput {
                current_id: 3,
                is_source: false,
                contract: contract(2),
                terms: vec![term(7, 0, factor(2))].into_boxed_slice(),
            },
        ];
        let options = RecurrenceRelationDiscoveryOptions::new(
            RecurrenceRelationDiscoveryMode::Diagnostic,
            96,
            4,
            4,
            1.0e-70,
            1.0e-80,
            0x5059_414d,
            "full",
        )
        .unwrap();
        let outcome = discover_recurrence_current_relations(&inputs, &options).unwrap();
        assert_eq!(outcome.numerical_candidate_count, 1);
        assert_eq!(outcome.uncertified_candidate_count, 0);
        assert_eq!(outcome.relations.len(), 1);
        let certificate = &outcome.relations[0];
        assert_eq!(certificate.current_id, 2);
        assert_eq!(certificate.representative_id, 1);
        assert_eq!(certificate.factor, factor(2));
        assert_eq!(certificate.proof_sha256.len(), 64);
    }

    #[test]
    fn exact_parent_relations_propagate_to_later_currents() {
        let inputs = vec![
            RelationCurrentInput {
                current_id: 0,
                is_source: true,
                contract: contract(0),
                terms: Vec::new().into_boxed_slice(),
            },
            RelationCurrentInput {
                current_id: 1,
                is_source: false,
                contract: contract(1),
                terms: vec![term(7, 0, factor(1))].into_boxed_slice(),
            },
            RelationCurrentInput {
                current_id: 2,
                is_source: false,
                contract: contract(1),
                terms: vec![term(7, 0, factor(2))].into_boxed_slice(),
            },
            RelationCurrentInput {
                current_id: 3,
                is_source: false,
                contract: contract(2),
                terms: vec![term(8, 2, factor(1))].into_boxed_slice(),
            },
            RelationCurrentInput {
                current_id: 4,
                is_source: false,
                contract: contract(2),
                terms: vec![term(8, 1, factor(2))].into_boxed_slice(),
            },
        ];
        let options = RecurrenceRelationDiscoveryOptions::new(
            RecurrenceRelationDiscoveryMode::CertifiedReuse,
            96,
            4,
            4,
            1.0e-70,
            1.0e-80,
            7,
            "nlc",
        )
        .unwrap();
        let outcome = discover_recurrence_current_relations(&inputs, &options).unwrap();
        assert_eq!(
            outcome
                .relations
                .iter()
                .map(|relation| (relation.current_id, relation.representative_id))
                .collect::<Vec<_>>(),
            vec![(2, 1), (4, 3)]
        );
        assert_eq!(outcome.relations[1].factor, ExactComplexRational::ONE);
    }
}
