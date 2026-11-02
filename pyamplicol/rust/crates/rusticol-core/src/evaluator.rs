// SPDX-License-Identifier: 0BSD

#[path = "evaluator/crossing.rs"]
mod crossing;
pub(crate) use crossing::*;

#[path = "evaluator/backend.rs"]
mod backend;
pub(crate) use backend::{ensure_evaluator_capabilities_supported, evaluator_runtime_capabilities};

#[cfg(feature = "f64-compiled")]
#[path = "evaluator/compiled.rs"]
mod compiled;
#[cfg(feature = "f64-compiled")]
pub(crate) use compiled::*;

#[cfg(feature = "f64-compiled")]
#[path = "evaluator/native_direct.rs"]
pub(crate) mod native_direct;

#[cfg(feature = "f64-compiled")]
#[path = "evaluator/native_compiled_direct.rs"]
pub(crate) mod native_compiled_direct;

#[cfg(feature = "f64-symjit")]
#[path = "evaluator/symjit.rs"]
mod symjit;
#[cfg(feature = "f64-symjit")]
pub(crate) use symjit::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[path = "evaluator/symjit_plane.rs"]
pub(crate) mod symjit_plane;

#[cfg(feature = "f64-symjit")]
#[path = "evaluator/symjit_direct.rs"]
pub(crate) mod symjit_direct;
#[cfg(all(test, feature = "f64-symjit"))]
pub(crate) use symjit_direct::tests::count_allocations as count_test_allocations;

#[cfg(feature = "f64-symjit")]
#[path = "evaluator/symjit_compiled_direct.rs"]
pub(crate) mod symjit_compiled_direct;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[path = "evaluator/symjit_eager_direct.rs"]
pub(crate) mod symjit_eager_direct;

#[path = "evaluator/recurrence_source_direct.rs"]
pub(crate) mod recurrence_source_direct;

#[path = "evaluator/recurrence_closure_direct.rs"]
pub(crate) mod recurrence_closure_direct;

#[path = "evaluator/recurrence_intrinsic_direct.rs"]
pub(crate) mod recurrence_intrinsic_direct;

#[path = "evaluator/stage.rs"]
mod stage;
pub(crate) use stage::*;

#[path = "evaluator/amplitude.rs"]
mod amplitude;
#[cfg(test)]
pub(crate) use amplitude::build_compiled_symmetric_group_color_contraction_runtime;
pub(crate) use amplitude::{
    build_color_contraction_runtime, build_raw_sum_groups, generic_root_group_id,
};

pub(crate) fn native_f64_simd_lane_width() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return 8;
        }
        if std::is_x86_feature_detected!("avx2") {
            return 4;
        }
        if std::is_x86_feature_detected!("sse2") {
            return 2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return 2;
        }
    }
    1
}
