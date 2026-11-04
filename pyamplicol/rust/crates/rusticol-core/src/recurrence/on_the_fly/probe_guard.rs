// SPDX-License-Identifier: 0BSD

#![allow(dead_code)] // Poison hooks are exercised only by the opt-in probe harness.

//! Feature-only poison guard for the genuine on-the-fly artifact probe.

use super::*;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OnTheFlyForbiddenWorkCountsV1 {
    pub(crate) direct_plan_load_attempts: u32,
    pub(crate) direct_plan_decode_attempts: u32,
    pub(crate) direct_plan_materialization_attempts: u32,
    pub(crate) established_builder_attempts: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OnTheFlyForbiddenWorkV1 {
    DirectPlanLoad,
    DirectPlanDecode,
    DirectPlanMaterialization,
    EstablishedBuilder,
}

thread_local! {
    static ACTIVE_PROBE: RefCell<Option<OnTheFlyForbiddenWorkCountsV1>> = const {
        RefCell::new(None)
    };
}

pub(crate) struct OnTheFlyForbiddenWorkGuardV1 {
    active: bool,
}

impl OnTheFlyForbiddenWorkGuardV1 {
    pub(crate) fn begin() -> RusticolResult<Self> {
        ACTIVE_PROBE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(invalid("a forbidden-work probe is already active"));
            }
            *slot = Some(OnTheFlyForbiddenWorkCountsV1::default());
            Ok(Self { active: true })
        })
    }

    pub(crate) fn finish(mut self) -> RusticolResult<OnTheFlyForbiddenWorkCountsV1> {
        self.active = false;
        ACTIVE_PROBE.with(|slot| {
            slot.borrow_mut()
                .take()
                .ok_or_else(|| integrity("forbidden-work probe state disappeared"))
        })
    }
}

impl Drop for OnTheFlyForbiddenWorkGuardV1 {
    fn drop(&mut self) {
        if self.active {
            ACTIVE_PROBE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

pub(crate) fn reject_forbidden_work_if_probed(kind: OnTheFlyForbiddenWorkV1) -> RusticolResult<()> {
    ACTIVE_PROBE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(counts) = slot.as_mut() else {
            return Ok(());
        };
        let (counter, label) = match kind {
            OnTheFlyForbiddenWorkV1::DirectPlanLoad => (
                &mut counts.direct_plan_load_attempts,
                "DirectRecurrencePlan loading",
            ),
            OnTheFlyForbiddenWorkV1::DirectPlanDecode => (
                &mut counts.direct_plan_decode_attempts,
                "DirectRecurrencePlan deserialization",
            ),
            OnTheFlyForbiddenWorkV1::DirectPlanMaterialization => (
                &mut counts.direct_plan_materialization_attempts,
                "DirectRecurrencePlan materialization",
            ),
            OnTheFlyForbiddenWorkV1::EstablishedBuilder => (
                &mut counts.established_builder_attempts,
                "established global recurrence construction",
            ),
        };
        *counter = counter
            .checked_add(1)
            .ok_or_else(|| integrity("forbidden-work attempt counter overflowed"))?;
        Err(invalid(format!(
            "{label} is poisoned inside the genuine on-the-fly artifact probe"
        )))
    })
}
