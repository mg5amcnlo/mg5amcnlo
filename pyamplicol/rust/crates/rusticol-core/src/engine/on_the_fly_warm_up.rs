// SPDX-License-Identifier: 0BSD

//! Structured progress for the explicit one-point on-the-fly warm-up API.
//!
//! Ordinary evaluation never constructs this state.  In particular, resident
//! memory sampling is performed only while an explicit observer is installed,
//! so the warmed evaluation path acquires no clock, allocation, or syscall.

use crate::{RusticolError, RusticolResult};
use serde::Serialize;
use std::time::{Duration, Instant};

const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOnTheFlyWarmUpEventKind {
    Start,
    Update,
    End,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOnTheFlyWarmUpStage {
    ProcessPreparation,
    QueryFamily,
    FamilyFinalization,
    FirstEvaluation,
}

/// One structured warm-up progress event.  A renderer is deliberately not
/// part of rusticol-core; Python, a CLI, or another embedding chooses how to
/// display these snapshots.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NativeOnTheFlyWarmUpEvent {
    pub schema_version: u32,
    pub kind: NativeOnTheFlyWarmUpEventKind,
    pub stage: NativeOnTheFlyWarmUpStage,
    pub completed: u64,
    pub total: u64,
    pub elapsed_seconds: f64,
    pub current_rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub workers: u64,
    pub message: Option<String>,
}

/// Summary of an explicit one-point binary64 warm-up transaction.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NativeOnTheFlyWarmUpResult {
    pub schema_version: u32,
    pub elapsed_seconds: f64,
    pub query_count: u64,
    pub warmed_query_count: u64,
    pub current_rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub already_warm: bool,
    pub first_evaluation_completed: bool,
}

/// Returning `false` requests cancellation at the next observable cold-path
/// boundary. Observer errors and cancellation are propagated through the
/// same pending-family rollback used by ordinary evaluation failures. The one
/// exception is the terminal `End` event for `FirstEvaluation`: it is an
/// informational notification delivered after commit, so its return value or
/// error cannot undo a completed warm-up.
pub type NativeOnTheFlyWarmUpObserver<'a> =
    dyn FnMut(&NativeOnTheFlyWarmUpEvent) -> RusticolResult<bool> + 'a;

pub(super) struct OnTheFlyWarmUpProgress<'a> {
    observer: Option<&'a mut NativeOnTheFlyWarmUpObserver<'a>>,
    started: Instant,
    current_rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    workers: u64,
    last_update_delivery: Option<Instant>,
}

impl<'a> OnTheFlyWarmUpProgress<'a> {
    pub(super) fn new(
        observer: Option<&'a mut NativeOnTheFlyWarmUpObserver<'a>>,
        workers: usize,
    ) -> RusticolResult<Self> {
        let workers = u64::try_from(workers)
            .map_err(|_| RusticolError::invalid_argument("warm-up worker count exceeds u64"))?;
        Ok(Self {
            observer,
            started: Instant::now(),
            current_rss_bytes: None,
            peak_rss_bytes: None,
            workers,
            last_update_delivery: None,
        })
    }

    pub(super) fn elapsed_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub(super) const fn current_rss_bytes(&self) -> Option<u64> {
        self.current_rss_bytes
    }

    pub(super) const fn peak_rss_bytes(&self) -> Option<u64> {
        self.peak_rss_bytes
    }

    pub(super) fn emit(
        &mut self,
        kind: NativeOnTheFlyWarmUpEventKind,
        stage: NativeOnTheFlyWarmUpStage,
        completed: usize,
        total: usize,
        message: Option<&str>,
    ) -> RusticolResult<()> {
        if self.observer.is_none() {
            return Ok(());
        }
        if kind == NativeOnTheFlyWarmUpEventKind::Update {
            let now = Instant::now();
            if self
                .last_update_delivery
                .is_some_and(|last| now.duration_since(last) < PROGRESS_UPDATE_INTERVAL)
            {
                return Ok(());
            }
            self.last_update_delivery = Some(now);
        }
        self.deliver(kind, stage, completed, total, message, true)
    }

    /// Publish the post-commit completion notification.  The selected family
    /// and its public identity are already committed at this boundary, so a
    /// rejecting or failing observer cannot retroactively turn the completed
    /// warm-up into an error. Embeddings may report such delivery failures as
    /// unraisable diagnostics, but must not roll back physics state.
    pub(super) fn emit_terminal_notification(
        &mut self,
        stage: NativeOnTheFlyWarmUpStage,
        completed: usize,
        total: usize,
        message: Option<&str>,
    ) {
        let _ = self.deliver(
            NativeOnTheFlyWarmUpEventKind::End,
            stage,
            completed,
            total,
            message,
            false,
        );
    }

    fn deliver(
        &mut self,
        kind: NativeOnTheFlyWarmUpEventKind,
        stage: NativeOnTheFlyWarmUpStage,
        completed: usize,
        total: usize,
        message: Option<&str>,
        cancellable: bool,
    ) -> RusticolResult<()> {
        let Some(observer) = self.observer.as_deref_mut() else {
            return Ok(());
        };
        let current_rss_bytes = current_process_rss_bytes();
        if let Some(current) = current_rss_bytes {
            self.current_rss_bytes = Some(current);
            self.peak_rss_bytes = Some(self.peak_rss_bytes.unwrap_or(0).max(current));
        }
        let event = NativeOnTheFlyWarmUpEvent {
            schema_version: 1,
            kind,
            stage,
            completed: u64::try_from(completed).map_err(|_| {
                RusticolError::invalid_argument("warm-up completed count exceeds u64")
            })?,
            total: u64::try_from(total)
                .map_err(|_| RusticolError::invalid_argument("warm-up total exceeds u64"))?,
            elapsed_seconds: self.started.elapsed().as_secs_f64(),
            current_rss_bytes: self.current_rss_bytes,
            peak_rss_bytes: self.peak_rss_bytes,
            workers: self.workers,
            message: message.map(str::to_owned),
        };
        match observer(&event) {
            Ok(true) => Ok(()),
            Ok(false) if !cancellable => Ok(()),
            Ok(false) => Err(RusticolError::evaluation(
                "on-the-fly warm-up cancelled by its progress observer",
            )),
            Err(_) if !cancellable => Ok(()),
            Err(error) => Err(error),
        }
    }
}

#[cfg(target_os = "macos")]
fn current_process_rss_bytes() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
    let size = std::mem::size_of::<libc::proc_taskinfo>();
    let written = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            info.as_mut_ptr().cast(),
            i32::try_from(size).ok()?,
        )
    };
    if usize::try_from(written).ok()? != size {
        return None;
    }
    Some(unsafe { info.assume_init() }.pti_resident_size)
}

#[cfg(target_os = "linux")]
fn current_process_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page_size).ok()?;
    resident_pages.checked_mul(page_size)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
const fn current_process_rss_bytes() -> Option<u64> {
    None
}
