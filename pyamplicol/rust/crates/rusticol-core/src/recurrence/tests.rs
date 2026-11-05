// SPDX-License-Identifier: 0BSD

use std::cmp::Ordering;

use super::*;

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
}

fn momentum(slot: u32) -> CanonicalMomentumLinearForm {
    CanonicalMomentumLinearForm::new(vec![MomentumTerm {
        source_slot: slot,
        coefficient: 1,
    }])
    .expect("test momentum is canonical")
}

#[test]
fn abi_constants_are_frozen() {
    assert_eq!(RECURRENCE_TEMPLATE_ABI, "pyamplicol-recurrence-template-v1");
    assert_eq!(
        RECURRENCE_BUILDER_INPUT_ABI,
        "pyamplicol-recurrence-builder-input-v2"
    );
    assert_eq!(
        RECURRENCE_BUILDER_RESULT_ABI,
        "pyamplicol-recurrence-builder-result-v2"
    );
    assert_eq!(RECURRENCE_PLAN_ABI, "pyamplicol-recurrence-plan-v2");
    assert_eq!(
        RECURRENCE_RUNTIME_LAYOUT_ABI,
        "pyamplicol-recurrence-runtime-layout-v2"
    );
    assert_eq!(
        RECURRENCE_RUNTIME_KIND,
        "pyamplicol-runtime-recurrence-execution"
    );
    assert_eq!(
        RECURRENCE_RUNTIME_CAPABILITY,
        "rusticol.recurrence-direct-arena.complex-f64.v1"
    );
    assert_eq!(
        RECURRENCE_LC_COLOR_CAPABILITY,
        "rusticol.recurrence-color.lc.v1"
    );
}

#[test]
fn dynamic_union_spin_sentinel_is_rejected_as_concrete_input() {
    for context in [
        "process source spin",
        "template source spin",
        "quantum input spin",
        "quantum result spin",
    ] {
        assert!(validate_concrete_spin_state(-1, context).is_ok());
        let error = validate_concrete_spin_state(DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS, context)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reserved dynamic-union source-spin sentinel"));
    }
}

#[test]
fn exact_rationals_parse_reduce_and_canonicalize_zero() {
    let value = ExactRational::parse_parts("-42", "56").unwrap();
    assert_eq!(value.numerator(), -3);
    assert_eq!(value.denominator(), 4);
    assert_eq!(value.to_string(), "-3/4");
    assert_eq!(
        "000/009".parse::<ExactRational>().unwrap(),
        ExactRational::ZERO
    );
    assert!(ExactRational::parse_parts("1", "0").is_err());
    assert!(ExactRational::parse_parts("1", "-2").is_err());
    assert!("1/2/3".parse::<ExactRational>().is_err());
}

#[test]
fn exact_aggregation_distinguishes_binary64_collision() {
    let one = ExactRational::ONE;
    let epsilon = ExactRational::new(1, 1_i128 << 54).unwrap();
    let sum = one.checked_add(epsilon).unwrap();
    assert_ne!(sum, one);
    assert_eq!(sum.checked_sub(one).unwrap(), epsilon);
    assert_eq!(1.0_f64 + 2.0_f64.powi(-54), 1.0_f64);
}

#[test]
fn exact_ordering_avoids_cross_multiplication_overflow() {
    let maximum = i128::MAX;
    let left = ExactRational::new(maximum - 1, maximum).unwrap();
    let right = ExactRational::new(maximum - 2, maximum - 1).unwrap();
    assert_eq!(left.cmp(&right), Ordering::Greater);
}

#[test]
fn exact_arithmetic_fails_closed_on_overflow() {
    let large = ExactRational::new(i128::MAX, 1).unwrap();
    assert!(large.checked_add(ExactRational::ONE).is_err());
    assert!(
        large
            .checked_mul(ExactRational::new(2, 1).unwrap())
            .is_err()
    );
    assert!(ExactRational::ONE.checked_div(ExactRational::ZERO).is_err());
}

#[test]
fn binary64_conversion_is_exact_or_rejected() {
    assert_eq!(
        ExactRational::from_f64_exact(0.1).unwrap(),
        ExactRational::new(3_602_879_701_896_397, 36_028_797_018_963_968).unwrap()
    );
    assert_eq!(
        ExactRational::from_f64_exact(-0.0).unwrap(),
        ExactRational::ZERO
    );
    assert!(ExactRational::from_f64_exact(f64::INFINITY).is_err());
    assert!(ExactRational::from_f64_exact(f64::from_bits(1)).is_err());
}

#[test]
fn exact_complex_arithmetic_is_canonical() {
    let left = ExactComplexRational::parse_parts("1", "2", "1", "3").unwrap();
    let right = ExactComplexRational::parse_parts("2", "5", "-1", "7").unwrap();
    let product = left.checked_mul(right).unwrap();
    assert_eq!(product.real(), ExactRational::new(26, 105).unwrap());
    assert_eq!(product.imag(), ExactRational::new(13, 210).unwrap());
    assert_eq!(product.checked_div(right).unwrap(), left);
}

#[test]
fn checked_ranges_reject_overflow_gaps_and_bounds() {
    assert!(CheckedTableRange::new(u64::MAX, 1).end("test").is_err());
    assert!(validate_ranges_within("test", &[CheckedTableRange::new(2, 2)], 3).is_err());
    assert!(
        validate_packed_ranges(
            "test",
            &[CheckedTableRange::new(0, 1), CheckedTableRange::new(2, 1)],
            3
        )
        .is_err()
    );
    validate_packed_ranges(
        "test",
        &[CheckedTableRange::new(0, 1), CheckedTableRange::new(1, 2)],
        3,
    )
    .unwrap();
}

#[test]
fn table_validators_reject_mismatched_columns_and_references() {
    assert!(validate_equal_column_lengths("table", &[("a", 2), ("b", 1)]).is_err());
    assert!(validate_u32_references(&[0, 2], 2, "reference").is_err());
    validate_equal_column_lengths("table", &[("a", 2), ("b", 2)]).unwrap();
    validate_u32_references(&[0, 1], 2, "reference").unwrap();
}

#[test]
fn multiword_masks_retain_high_bits_and_validate_canonical_form() {
    let ranges = [CheckedTableRange::new(0, 0), CheckedTableRange::new(0, 2)];
    let populations = [0, 2];
    let words = [1, 1_u64 << 63];
    let catalog = MultiwordMaskCatalogView {
        ranges: &ranges,
        populations: &populations,
        words: &words,
    };
    catalog.validate(true).unwrap();
    assert!(catalog.contains(1, 0).unwrap());
    assert!(catalog.contains(1, 127).unwrap());
    assert!(!catalog.contains(1, 126).unwrap());

    let invalid_words = [1, 0];
    let invalid = MultiwordMaskCatalogView {
        ranges: &[CheckedTableRange::new(0, 2)],
        populations: &[1],
        words: &invalid_words,
    };
    assert!(invalid.validate(false).is_err());
}

fn header_for(
    strategy: RecurrenceStrategy,
    sections: &[CanonicalInputSection<'_>],
) -> RecurrenceBuilderInputHeader {
    let placeholder = digest(9);
    let mut header = RecurrenceBuilderInputHeader::canonical(
        strategy,
        sections.len() as u32,
        digest(1),
        digest(2),
        digest(3),
        placeholder,
    );
    header.input_digest = canonical_input_digest(&header, sections).unwrap();
    header
}

#[test]
fn input_header_digest_is_deterministic_and_authenticated() {
    let sections = [
        CanonicalInputSection {
            name: "catalogs/states",
            row_width: 2,
            row_count: 2,
            bytes: &[1, 2, 3, 4],
        },
        CanonicalInputSection {
            name: "process/legs",
            row_width: 1,
            row_count: 2,
            bytes: &[5, 6],
        },
    ];
    let header = header_for(RecurrenceStrategy::TopologyReplay, &sections);
    validate_header_and_sections(&header, &sections).unwrap();
    assert_eq!(
        canonical_input_digest(&header, &sections).unwrap(),
        header.input_digest
    );

    let modified = [
        sections[0],
        CanonicalInputSection {
            bytes: &[5, 7],
            ..sections[1]
        },
    ];
    assert!(validate_header_and_sections(&header, &modified).is_err());
}

#[test]
fn input_header_rejects_noncanonical_sections_and_abi_drift() {
    let sections = [
        CanonicalInputSection {
            name: "z",
            row_width: 1,
            row_count: 1,
            bytes: &[0],
        },
        CanonicalInputSection {
            name: "a",
            row_width: 1,
            row_count: 1,
            bytes: &[0],
        },
    ];
    let mut header = RecurrenceBuilderInputHeader::canonical(
        RecurrenceStrategy::AllFlowUnion,
        2,
        digest(1),
        digest(2),
        digest(3),
        digest(4),
    );
    assert!(validate_header_and_sections(&header, &sections).is_err());
    let canonical_section = [CanonicalInputSection {
        name: "a",
        row_width: 1,
        row_count: 1,
        bytes: &[0],
    }];
    let wrong_count_header = RecurrenceBuilderInputHeader::canonical(
        RecurrenceStrategy::AllFlowUnion,
        2,
        digest(1),
        digest(2),
        digest(3),
        digest(4),
    );
    assert!(canonical_input_digest(&wrong_count_header, &canonical_section).is_err());
    header.builder_input_abi = "future-abi".to_owned();
    assert!(header.validate_identity().is_err());
}

#[test]
fn strategy_and_digest_validation_fail_closed() {
    assert_eq!(
        RecurrenceStrategy::try_from(0).unwrap(),
        RecurrenceStrategy::TopologyReplay
    );
    assert_eq!(
        RecurrenceStrategy::try_from(1).unwrap(),
        RecurrenceStrategy::AllFlowUnion
    );
    assert_eq!(
        RecurrenceStrategy::try_from(2).unwrap(),
        RecurrenceStrategy::ContractedColorUnion
    );
    assert!(RecurrenceStrategy::try_from(3).is_err());
    assert!(SemanticDigest::new([0; 32]).is_err());
}

#[test]
fn current_core_key_preserves_every_model_visible_axis() {
    let ancestry = vec![
        SourceStateAssignment::new(0, 1),
        SourceStateAssignment::new(2, 0),
    ];
    let key = CurrentCoreKey::new(
        digest(1),
        RecurrenceNodeKind::Current,
        7,
        DynamicLCColorStateId::from_interner(13),
        vec![0, 2],
        CanonicalMomentumLinearForm::new(vec![
            MomentumTerm {
                source_slot: 0,
                coefficient: 1,
            },
            MomentumTerm {
                source_slot: 2,
                coefficient: -1,
            },
        ])
        .unwrap(),
        CurrentHelicityIdentity::topology_replay(-1, ancestry.clone()).unwrap(),
        vec![2, -2],
        3,
        vec![1, 2],
        CurrentSourceBinding::None,
        Some(9),
    )
    .unwrap();
    let changed = CurrentCoreKey::new(
        digest(1),
        RecurrenceNodeKind::Current,
        7,
        DynamicLCColorStateId::from_interner(13),
        vec![0, 2],
        key.momentum().clone(),
        CurrentHelicityIdentity::topology_replay(1, ancestry.clone()).unwrap(),
        key.flavour_flow().to_vec(),
        key.quantum_number_flow_id(),
        key.coupling_orders().to_vec(),
        CurrentSourceBinding::None,
        Some(9),
    )
    .unwrap();
    assert_ne!(key, changed);
    assert!(
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Current,
            7,
            DynamicLCColorStateId::from_interner(13),
            vec![2, 0],
            momentum(0),
            CurrentHelicityIdentity::topology_replay(0, ancestry).unwrap(),
            vec![],
            0,
            vec![],
            CurrentSourceBinding::None,
            None,
        )
        .is_err()
    );
}

#[test]
fn current_core_key_enforces_layout_specific_helicity_identity() {
    let source_momentum = momentum(0);
    let replay_source = CurrentCoreKey::new(
        digest(1),
        RecurrenceNodeKind::Source,
        7,
        DynamicLCColorStateId::from_interner(13),
        vec![0],
        source_momentum.clone(),
        CurrentHelicityIdentity::topology_replay(-1, vec![SourceStateAssignment::new(0, 2)])
            .unwrap(),
        vec![1],
        3,
        vec![],
        CurrentSourceBinding::FixedTemplate(11),
        None,
    )
    .unwrap();
    assert_eq!(replay_source.spin_state_class(), -1);
    assert_eq!(
        replay_source.helicity_identity().local_source_states()[0].state_index(),
        2
    );

    let union_source = CurrentCoreKey::new(
        digest(1),
        RecurrenceNodeKind::Source,
        7,
        DynamicLCColorStateId::from_interner(13),
        vec![0],
        source_momentum.clone(),
        CurrentHelicityIdentity::all_flow_union(0),
        vec![1],
        3,
        vec![],
        CurrentSourceBinding::runtime_dispatch(5, vec![11]).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        union_source.source_binding(),
        &CurrentSourceBinding::runtime_dispatch(5, vec![11]).unwrap()
    );
    assert!(
        union_source
            .helicity_identity()
            .local_source_states()
            .is_empty()
    );

    assert!(
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Current,
            7,
            DynamicLCColorStateId::from_interner(13),
            vec![0, 2],
            source_momentum.clone(),
            CurrentHelicityIdentity::topology_replay(0, vec![SourceStateAssignment::new(0, 1)],)
                .unwrap(),
            vec![],
            3,
            vec![],
            CurrentSourceBinding::None,
            None,
        )
        .is_err()
    );
    assert!(
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Source,
            7,
            DynamicLCColorStateId::from_interner(13),
            vec![0],
            source_momentum,
            CurrentHelicityIdentity::all_flow_union(0),
            vec![],
            3,
            vec![],
            CurrentSourceBinding::FixedTemplate(11),
            None,
        )
        .is_err()
    );
}

#[test]
fn dynamic_lc_color_states_preserve_order_and_only_fold_trace_rotations() {
    let trace = LCColorComponent::new(LCColorComponentKind::Trace, vec![4, 7, 2]).unwrap();
    assert_eq!(trace.source_slots(), &[2, 4, 7]);
    let reversed = LCColorComponent::new(LCColorComponentKind::Trace, vec![2, 7, 4]).unwrap();
    assert_ne!(trace, reversed);

    let open = LCColorComponent::new(LCColorComponentKind::OpenString, vec![7, 2]).unwrap();
    assert_eq!(open.source_slots(), &[7, 2]);
    assert!(LCColorComponent::new(LCColorComponentKind::OpenString, vec![2, 2]).is_err());
    assert!(DynamicLCColorState::new(1, None, vec![trace, open]).is_err());
}

#[test]
fn dynamic_lc_color_state_ids_are_issued_only_by_the_interner() {
    let state = DynamicLCColorState::new(
        1,
        Some(0),
        vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![0]).unwrap()],
    )
    .unwrap();
    let mut interner = DynamicLCColorStateInterner::default();
    let first = interner.intern(state.clone()).unwrap();
    let second = interner.intern(state).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.get(), 0);
    assert_eq!(interner.len(), 1);
    assert_eq!(interner.get(first).unwrap().output_color_shape_id(), 1);
}

#[test]
fn compiler_certified_lc_color_witnesses_apply_exact_ordered_operations() {
    let left = DynamicLCColorState::new(
        1,
        Some(0),
        vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![0]).unwrap()],
    )
    .unwrap();
    let right = DynamicLCColorState::new(
        1,
        Some(0),
        vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 2]).unwrap()],
    )
    .unwrap();
    let witness = LCColorTransitionWitness::new(
        [1, 0],
        0b10,
        LCColorComponentOperation::ConcatenateJoin,
        Some(LCColorComponentKind::AdjointSegment),
        LCColorComponentRole::Active,
        Some(2),
        LCColorPortWiring::new(
            [1, 0],
            vec![],
            vec![
                LCColorParentPort::new(1, 0).unwrap(),
                LCColorParentPort::new(0, 0).unwrap(),
            ],
        )
        .unwrap(),
        ExactComplexRational::new(ExactRational::ONE, ExactRational::ZERO),
        digest(9),
    )
    .unwrap();
    let result = witness.apply(&left, &right).unwrap().unwrap();
    assert_eq!(result.output_color_shape_id(), 2);
    assert_eq!(result.components().len(), 2);
    assert_eq!(result.components()[0].source_slots(), &[2, 1]);
    assert_eq!(result.components()[1].source_slots(), &[0]);
    assert_eq!(witness.proof_digest(), digest(9));

    assert!(
        LCColorTransitionWitness::new(
            [0, 0],
            0,
            LCColorComponentOperation::ConcatenateKeep,
            None,
            LCColorComponentRole::None,
            Some(2),
            LCColorPortWiring::new([0, 1], vec![], vec![]).unwrap(),
            ExactComplexRational::new(ExactRational::ONE, ExactRational::ZERO),
            digest(9),
        )
        .is_err()
    );
}

#[test]
fn lc_color_join_uses_declared_active_components_in_multi_line_forests() {
    let left = DynamicLCColorState::new(
        1,
        Some(1),
        vec![
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![7, 8]).unwrap(),
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 1]).unwrap(),
        ],
    )
    .unwrap();
    let right = DynamicLCColorState::new_port_wired(
        1,
        vec![
            LCColorPortBinding::new(0, LCColorEndpoint::Back),
            LCColorPortBinding::new(0, LCColorEndpoint::Front),
        ],
        vec![
            LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![2]).unwrap(),
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![5, 6]).unwrap(),
        ],
    )
    .unwrap();
    let witness = LCColorTransitionWitness::new(
        [0, 1],
        0,
        LCColorComponentOperation::ConcatenateJoin,
        Some(LCColorComponentKind::OpenString),
        LCColorComponentRole::Active,
        Some(2),
        LCColorPortWiring::new(
            [0, 1],
            vec![[
                LCColorParentPort::new(0, 0).unwrap(),
                LCColorParentPort::new(1, 1).unwrap(),
            ]],
            vec![LCColorParentPort::new(1, 0).unwrap()],
        )
        .unwrap(),
        ExactComplexRational::new(ExactRational::ONE, ExactRational::ZERO),
        digest(11),
    )
    .unwrap();
    let result = witness.apply(&left, &right).unwrap().unwrap();
    assert_eq!(result.active_component_index(), Some(1));
    assert_eq!(result.components()[0].source_slots(), &[7, 8]);
    assert_eq!(result.components()[1].source_slots(), &[0, 1, 2]);
    assert_eq!(result.components()[2].source_slots(), &[5, 6]);
}

#[test]
fn port_wiring_preserves_crossed_two_line_connectivity() {
    let fundamental = |slot| {
        DynamicLCColorState::new_port_wired(
            1,
            vec![LCColorPortBinding::new(0, LCColorEndpoint::Back)],
            vec![LCColorComponent::new(LCColorComponentKind::OpenString, vec![slot]).unwrap()],
        )
        .unwrap()
    };
    let antifundamental = |slot| {
        DynamicLCColorState::new_port_wired(
            2,
            vec![LCColorPortBinding::new(0, LCColorEndpoint::Front)],
            vec![LCColorComponent::new(LCColorComponentKind::OpenString, vec![slot]).unwrap()],
        )
        .unwrap()
    };

    // u(2) + ubar(3) -> g*: the two gluon result ports retain the two
    // independent quark-line endpoints rather than collapsing to one port.
    let gluon_wiring = LCColorPortWiring::new(
        [0, 1],
        vec![],
        vec![
            LCColorParentPort::new(0, 0).unwrap(),
            LCColorParentPort::new(1, 0).unwrap(),
        ],
    )
    .unwrap();
    let gluon = gluon_wiring
        .apply(&fundamental(2), &antifundamental(3), 3)
        .unwrap();
    assert_eq!(gluon.result_port_bindings().len(), 2);
    assert_eq!(gluon.result_port_lineage_source_slots().unwrap(), [2, 3]);

    // d(1) consumes the ubar-side gluon port; the u-side port becomes the
    // result current port. This creates passive [1,3] plus active [2].
    let quark_wiring = LCColorPortWiring::new(
        [0, 1],
        vec![[
            LCColorParentPort::new(0, 0).unwrap(),
            LCColorParentPort::new(1, 1).unwrap(),
        ]],
        vec![LCColorParentPort::new(1, 0).unwrap()],
    )
    .unwrap();
    let complement = quark_wiring.apply(&fundamental(1), &gluon, 1).unwrap();
    assert_eq!(
        complement
            .components()
            .iter()
            .map(LCColorComponent::source_slots)
            .collect::<Vec<_>>(),
        vec![&[1, 3][..], &[2][..]],
    );
    assert_eq!(complement.result_port_lineage_source_slots().unwrap(), [2]);

    // The remaining result port closes against dbar(0), yielding exactly the
    // crossed physical forest [1,3] + [2,0].
    let closure_wiring = LCColorPortWiring::new(
        [0, 1],
        vec![[
            LCColorParentPort::new(0, 0).unwrap(),
            LCColorParentPort::new(1, 0).unwrap(),
        ]],
        vec![],
    )
    .unwrap();
    let closed = closure_wiring
        .apply(&complement, &antifundamental(0), 0)
        .unwrap();
    assert!(closed.result_port_bindings().is_empty());
    assert!(
        closed
            .result_port_lineage_source_slots()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        closed
            .components()
            .iter()
            .map(LCColorComponent::source_slots)
            .collect::<Vec<_>>(),
        vec![&[1, 3][..], &[2, 0][..]],
    );
}

#[test]
fn contribution_key_validates_aligned_parent_contracts() {
    let key = ContributionKey::new(
        1,
        vec![2, 3],
        vec![4, 5],
        vec![momentum(0), momentum(1)],
        6,
        7,
        LCColorWitnessTermId::new(8, 0),
        digest(9),
        10,
    )
    .unwrap();
    assert_eq!(key.parent_value_class_ids(), &[2, 3]);
    assert!(
        ContributionKey::new(
            1,
            vec![2, 3],
            vec![4],
            vec![momentum(0), momentum(1)],
            6,
            7,
            LCColorWitnessTermId::new(8, 0),
            digest(9),
            10,
        )
        .is_err()
    );
}
