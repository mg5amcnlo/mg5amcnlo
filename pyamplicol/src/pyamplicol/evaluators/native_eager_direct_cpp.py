# SPDX-License-Identifier: 0BSD
"""Emit a genuine native eager DirectTable callable.

The ordinary compiled prepared-kernel entry accepts one dense complex row.
This producer deliberately does not call it.  Instead it lowers the retained
Symbolica instruction program directly into a row-outer, point-inner callable
whose inputs and ordered fanout destinations are persistent split-complex
planes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .._internal.physics.types import NativeEvaluationError
from .native_direct_cpp import (
    NativeDirectCppParameterKind,
    NativeDirectCppSpec,
    _lower_instruction_program,
    _ParameterAccess,
)
from .symbolica_helpers import _symbolica_instruction_program

NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI = "pyamplicol-eager-native-direct-table-v1"
_MARKER = "// pyAmpliCol genuine native eager DirectTable producer v1"
_SUPPORTED_LANES = frozenset({2, 4})


@dataclass(frozen=True, slots=True)
class NativeEagerDirectTableSource:
    source: str
    function_name: str
    evaluator_state_sha256: str
    instruction_count: int
    temporary_count: int
    input_complex_count: int
    output_complex_count: int
    invocation_stride: int
    attachment_stride: int
    simd_lane_width: int


def render_native_eager_direct_table_cpp(
    evaluator: Any,
    *,
    kernel_id: int,
    input_complex_count: int,
    output_complex_count: int,
    target_triple: str,
    simd_lane_width: int = 2,
    real_parameter_indices: tuple[int, ...] = (),
    evaluator_state_bytes: bytes | None = None,
) -> NativeEagerDirectTableSource:
    """Render one direct-only table export from retained evaluator MIR."""

    if kernel_id < 0 or kernel_id >= 2**32:
        raise NativeEvaluationError("native eager DirectTable kernel ID exceeds u32")
    if input_complex_count < 1 or output_complex_count < 1:
        raise NativeEvaluationError(
            "native eager DirectTable input/output widths must be positive"
        )
    if simd_lane_width not in _SUPPORTED_LANES:
        raise NativeEvaluationError(
            "native eager DirectTable SIMD width must be 2 or 4"
        )
    if not target_triple or "\x00" in target_triple:
        raise NativeEvaluationError("native eager DirectTable target is invalid")

    function_name = f"pyamplicol_eager_direct_table_k{kernel_id:08x}_v1"
    get_instructions = getattr(evaluator, "get_instructions", None)
    save = getattr(evaluator, "save", None)
    if not callable(get_instructions) or (
        evaluator_state_bytes is None and not callable(save)
    ):
        raise NativeEvaluationError(
            "native eager DirectTable production requires retained Symbolica "
            "instructions and evaluator state"
        )
    try:
        raw_program = _symbolica_instruction_program(get_instructions())
        evaluator_state = (
            save() if evaluator_state_bytes is None else evaluator_state_bytes
        )
    except Exception as error:
        raise NativeEvaluationError(
            "Symbolica could not expose the eager DirectTable instruction program"
        ) from error
    if not isinstance(evaluator_state, bytes) or not evaluator_state:
        raise NativeEvaluationError(
            "Symbolica returned invalid eager DirectTable evaluator state"
        )

    real_parameters = frozenset(real_parameter_indices)
    if any(index < 0 or index >= input_complex_count for index in real_parameters):
        raise NativeEvaluationError(
            "native eager DirectTable real parameter index is out of bounds"
        )
    parameter_kinds = tuple(
        (
            NativeDirectCppParameterKind.REAL_PLANE
            if index in real_parameters
            else NativeDirectCppParameterKind.COMPLEX_PLANE
        )
        for index in range(input_complex_count)
    )
    spec = NativeDirectCppSpec(
        function_name=function_name,
        parameter_kinds=parameter_kinds,
        output_count=output_complex_count,
        target_triple=target_triple,
        simd_lane_width=simd_lane_width,
    )
    accesses = tuple(
        _ParameterAccess(
            kind,
            2 * index,
            (
                2 * index + 1
                if kind is NativeDirectCppParameterKind.COMPLEX_PLANE
                else None
            ),
        )
        for index, kind in enumerate(parameter_kinds)
    )
    input_plane_count = 2 * input_complex_count
    program = _lower_instruction_program(raw_program, spec, accesses)
    invocation_stride = (input_plane_count + 2) * 4
    attachment_stride = (2 * output_complex_count + 2) * 4
    source = _render_translation_unit(
        function_name=function_name,
        statements="\n".join(program.statements),
        input_plane_count=input_plane_count,
        output_complex_count=output_complex_count,
        temporary_count=program.temporary_count,
        invocation_stride=invocation_stride,
        attachment_stride=attachment_stride,
        simd_lane_width=simd_lane_width,
        evaluator_state_sha256=hashlib.sha256(evaluator_state).hexdigest(),
        target_triple=target_triple,
    )
    return NativeEagerDirectTableSource(
        source=source,
        function_name=function_name,
        evaluator_state_sha256=hashlib.sha256(evaluator_state).hexdigest(),
        instruction_count=program.instruction_count,
        temporary_count=program.temporary_count,
        input_complex_count=input_complex_count,
        output_complex_count=output_complex_count,
        invocation_stride=invocation_stride,
        attachment_stride=attachment_stride,
        simd_lane_width=simd_lane_width,
    )


def _cpp_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_translation_unit(
    *,
    function_name: str,
    statements: str,
    input_plane_count: int,
    output_complex_count: int,
    temporary_count: int,
    invocation_stride: int,
    attachment_stride: int,
    simd_lane_width: int,
    evaluator_state_sha256: str,
    target_triple: str,
) -> str:
    vector_bytes = 8 * simd_lane_width
    temporary_extent = max(temporary_count, 1)
    output_extent = max(output_complex_count, 1)
    input_extent = max(input_plane_count, 1)
    destination_stores = "\n".join(
        f"""
      const std::uint32_t destination_re_{output} =
          read_u32(attachment + {8 * output}u);
      const std::uint32_t destination_im_{output} =
          read_u32(attachment + {8 * output + 4}u);
      if (destination_re_{output} >= view->plane_count ||
          destination_im_{output} >= view->plane_count) return 2;
      const DirectPlane destination_real_{output} =
          view->planes[destination_re_{output}];
      const DirectPlane destination_imag_{output} =
          view->planes[destination_im_{output}];
      if (destination_real_{output}.values == nullptr ||
          destination_imag_{output}.values == nullptr ||
          point_stop > destination_real_{output}.len ||
          point_stop > destination_imag_{output}.len) return 2;
      const Lane scaled_re_{output} =
          output[{output}].real * factor_re -
          output[{output}].imaginary * factor_im;
      const Lane scaled_im_{output} =
          output[{output}].real * factor_im +
          output[{output}].imaginary * factor_re;
      if (operation == 0u) {{
        store_lane(destination_real_{output}.values, point, scaled_re_{output});
        store_lane(destination_imag_{output}.values, point, scaled_im_{output});
      }} else {{
        store_lane(
            destination_real_{output}.values,
            point,
            load_lane<Lane>(destination_real_{output}.values, point) +
                scaled_re_{output});
        store_lane(
            destination_imag_{output}.values,
            point,
            load_lane<Lane>(destination_imag_{output}.values, point) +
                scaled_im_{output});
      }}"""
        for output in range(output_complex_count)
    )
    return f"""{_MARKER}
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if !defined(__clang__) && !defined(__GNUC__)
#error "pyAmpliCol native eager DirectTable requires Clang or GCC"
#endif

namespace pyamplicol_native_eager_{function_name} {{

constexpr std::uint32_t kInputPlaneCount = {input_plane_count}u;
constexpr std::uint32_t kOutputComplexCount = {output_complex_count}u;
constexpr std::uint32_t kInvocationStride = {invocation_stride}u;
constexpr std::uint32_t kAttachmentStride = {attachment_stride}u;
constexpr std::uint32_t kSimdLaneWidth = {simd_lane_width}u;

using DirectVector = double __attribute__((vector_size({vector_bytes})));

template <typename Lane>
struct DirectComplex {{
  Lane real;
  Lane imaginary;
}};

struct DirectPlane {{
  double* values;
  std::size_t len;
}};

struct DirectScalar {{
  const double* value;
}};

struct DirectTableCallViewV1 {{
  const std::uint8_t* invocations;
  std::uint32_t invocation_count;
  std::uint32_t invocation_stride;
  const std::uint8_t* attachments;
  std::uint32_t attachment_count;
  std::uint32_t attachment_stride;
  const DirectPlane* planes;
  std::uint32_t plane_count;
  std::uint32_t scalar_count;
  const DirectScalar* scalars;
  const double* factor_re;
  const double* factor_im;
  std::uint32_t factor_count;
  std::uint32_t point_start;
  std::uint32_t point_count;
}};

struct NativeEagerDirectTableMetadataV1 {{
  std::uint32_t struct_size;
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint32_t invocation_stride;
  std::uint32_t attachment_stride;
  std::uint32_t input_complex_count;
  std::uint32_t output_complex_count;
  std::uint32_t simd_lane_width;
  const char* application_abi;
  const char* function_name;
  const char* target_triple;
  const char* evaluator_state_sha256;
}};

static_assert(sizeof(void*) == 8u);
static_assert(sizeof(DirectPlane) == 16u);

inline std::uint32_t read_u32(const std::uint8_t* source) noexcept {{
  std::uint32_t value = 0u;
  std::memcpy(&value, source, sizeof(value));
  return value;
}}

template <typename Lane>
inline Lane broadcast(double value) noexcept {{
  if constexpr (std::is_same_v<Lane, double>) {{
    return value;
  }} else {{
    Lane result{{}};
    for (std::size_t lane = 0; lane < sizeof(Lane) / sizeof(double); ++lane) {{
      result[lane] = value;
    }}
    return result;
  }}
}}

template <typename Lane>
inline Lane load_lane(const double* values, std::uint32_t point) noexcept {{
  Lane result;
  std::memcpy(&result, values + point, sizeof(Lane));
  return result;
}}

template <typename Lane>
inline void store_lane(
    double* values, std::uint32_t point, Lane value) noexcept {{
  std::memcpy(values + point, &value, sizeof(Lane));
}}

template <typename Lane>
inline DirectComplex<Lane> direct_constant(
    double real, double imaginary) noexcept {{
  return {{broadcast<Lane>(real), broadcast<Lane>(imaginary)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_plane(
    const double* real,
    const double* imaginary,
    std::uint32_t point) noexcept {{
  return {{load_lane<Lane>(real, point), load_lane<Lane>(imaginary, point)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_real_plane(
    const double* real, std::uint32_t point) noexcept {{
  return {{load_lane<Lane>(real, point), broadcast<Lane>(0.0)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_scalar(
    double real, double imaginary) noexcept {{
  return direct_constant<Lane>(real, imaginary);
}}

template <typename Lane>
inline DirectComplex<Lane> direct_real_scalar(double real) noexcept {{
  return direct_constant<Lane>(real, 0.0);
}}

template <typename Lane>
inline DirectComplex<Lane> direct_add(
    DirectComplex<Lane> left, DirectComplex<Lane> right) noexcept {{
  return {{left.real + right.real, left.imaginary + right.imaginary}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_multiply(
    DirectComplex<Lane> left, DirectComplex<Lane> right) noexcept {{
  return {{
      left.real * right.real - left.imaginary * right.imaginary,
      left.real * right.imaginary + left.imaginary * right.real}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_real_reciprocal(
    DirectComplex<Lane> value) noexcept {{
  return {{broadcast<Lane>(1.0) / value.real, broadcast<Lane>(0.0)}};
}}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_reciprocal(
    DirectComplex<Lane> value) noexcept {{
  const Lane denominator =
      value.real * value.real + value.imaginary * value.imaginary;
  return {{value.real / denominator, -value.imaginary / denominator}};
}}

template <typename Lane>
int evaluate_invocation(
    const DirectTableCallViewV1* view,
    const std::array<DirectPlane, {input_extent}>& inputs,
    std::uint32_t attachment_start,
    std::uint32_t attachment_count,
    std::uint32_t point,
    std::uint32_t point_stop) noexcept {{
  std::array<DirectComplex<Lane>, {temporary_extent}> temporary;
  std::array<DirectComplex<Lane>, {output_extent}> output;
{statements}
  for (std::uint32_t attachment_index = 0u;
       attachment_index < attachment_count;
       ++attachment_index) {{
    const std::uint8_t* attachment =
        view->attachments +
        static_cast<std::size_t>(attachment_start + attachment_index) *
            kAttachmentStride;
    const std::uint32_t factor_id =
        read_u32(attachment + {8 * output_complex_count}u);
    const std::uint32_t operation =
        read_u32(attachment + {8 * output_complex_count + 4}u);
    if (factor_id >= view->factor_count || operation > 1u) return 2;
    const Lane factor_re = broadcast<Lane>(view->factor_re[factor_id]);
    const Lane factor_im = broadcast<Lane>(view->factor_im[factor_id]);
{destination_stores}
  }}
  return 0;
}}

}}  // namespace pyamplicol_native_eager_{function_name}

extern "C" const pyamplicol_native_eager_{function_name}::
    NativeEagerDirectTableMetadataV1*
{function_name}_metadata_v1() noexcept {{
  using Metadata = pyamplicol_native_eager_{function_name}::
      NativeEagerDirectTableMetadataV1;
  static const Metadata metadata{{
      static_cast<std::uint32_t>(sizeof(Metadata)),
      1u,
      0x1fu,
      {invocation_stride}u,
      {attachment_stride}u,
      {input_plane_count // 2}u,
      {output_complex_count}u,
      {simd_lane_width}u,
      "{NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI}",
      "{function_name}",
      "{_cpp_string(target_triple)}",
      "{evaluator_state_sha256}"}};
  return &metadata;
}}

extern "C" const char* {function_name}_target_triple() noexcept {{
  return "{_cpp_string(target_triple)}";
}}

extern "C" const char* {function_name}_source_sha256() noexcept {{
  return "{evaluator_state_sha256}";
}}

extern "C" int {function_name}(
    const pyamplicol_native_eager_{function_name}::
        DirectTableCallViewV1* view) noexcept {{
  using namespace pyamplicol_native_eager_{function_name};
  try {{
    if (view == nullptr || view->invocations == nullptr ||
        view->planes == nullptr || view->factor_re == nullptr ||
        view->factor_im == nullptr ||
        view->invocation_stride != kInvocationStride ||
        view->attachment_stride != kAttachmentStride ||
        view->invocation_count == 0u || view->point_count == 0u ||
        view->scalar_count != 0u ||
        (view->attachment_count != 0u && view->attachments == nullptr)) {{
      return 2;
    }}
    const std::uint64_t point_stop_wide =
        static_cast<std::uint64_t>(view->point_start) + view->point_count;
    if (point_stop_wide >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max())) return 2;
    const std::uint32_t point_stop =
        static_cast<std::uint32_t>(point_stop_wide);
    for (std::uint32_t invocation_index = 0u;
         invocation_index < view->invocation_count;
         ++invocation_index) {{
      const std::uint8_t* invocation =
          view->invocations +
          static_cast<std::size_t>(invocation_index) * kInvocationStride;
      std::array<DirectPlane, {input_extent}> inputs{{}};
      for (std::uint32_t input = 0u; input < kInputPlaneCount; ++input) {{
        const std::uint32_t plane_id = read_u32(invocation + 4u * input);
        if (plane_id >= view->plane_count) return 2;
        inputs[input] = view->planes[plane_id];
        if (inputs[input].values == nullptr ||
            point_stop > inputs[input].len) return 2;
      }}
      const std::uint32_t attachment_start =
          read_u32(invocation + 4u * kInputPlaneCount);
      const std::uint32_t attachment_count =
          read_u32(invocation + 4u * (kInputPlaneCount + 1u));
      if (static_cast<std::uint64_t>(attachment_start) + attachment_count >
          view->attachment_count) return 2;

      const std::uint32_t vector_count =
          (view->point_count / kSimdLaneWidth) * kSimdLaneWidth;
      const std::uint32_t vector_stop = view->point_start + vector_count;
      std::uint32_t point = view->point_start;
      for (; point < vector_stop; point += kSimdLaneWidth) {{
        const int status = evaluate_invocation<DirectVector>(
            view, inputs, attachment_start, attachment_count, point, point_stop);
        if (status != 0) return status;
      }}
      for (; point < point_stop; ++point) {{
        const int status = evaluate_invocation<double>(
            view, inputs, attachment_start, attachment_count, point, point_stop);
        if (status != 0) return status;
      }}
    }}
    return 0;
  }} catch (...) {{
    return 3;
  }}
}}
"""


__all__ = [
    "NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI",
    "NativeEagerDirectTableSource",
    "render_native_eager_direct_table_cpp",
]
