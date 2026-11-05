"""Unit tests for the string-formatting helpers in format.cpp/format.hpp,
exposed to Python for testing (format_si_prefix, format_with_error,
format_progress).
"""

import madspace as ms

# --- format_si_prefix ------------------------------------------------------


def test_si_prefix_below_thousand_is_bare_integer():
    assert ms.format_si_prefix(999) == "999"
    assert ms.format_si_prefix(42) == "42"


def test_si_prefix_kilo():
    assert ms.format_si_prefix(1000) == "1.00k"
    assert ms.format_si_prefix(1500) == "1.50k"


def test_si_prefix_kilo_variable_precision():
    # digits after the dot shrink as the value approaches the next power of ten,
    # to keep three significant digits
    assert ms.format_si_prefix(12345) == "12.3k"


def test_si_prefix_mega():
    assert ms.format_si_prefix(1_000_000) == "1.00M"


def test_si_prefix_rounds_up_into_next_prefix():
    # 999999 rounds to 1000 at 0-decimal precision within the "k" prefix
    assert ms.format_si_prefix(999_999) == "1000k"


def test_si_prefix_falls_back_beyond_tera():
    # no prefix covers 10^15, so it prints the plain (exponential) number
    assert ms.format_si_prefix(1e15) == "1e+15"


# --- format_with_error ------------------------------------------------------


def test_with_error_normal_case_keeps_digit_notation():
    assert ms.format_with_error(1.234, 0.056) == "1.234(56)"


def test_with_error_single_significant_digit_error():
    assert ms.format_with_error(123.456, 1.2) == "123.5(1.2)"


def test_with_error_small_error_extends_precision():
    assert ms.format_with_error(1.0, 0.001) == "1.0000(10)"


def test_with_error_large_magnitude_uses_exponential_form():
    assert ms.format_with_error(1e10, 1e6) == "1.00000(10)e+10"


def test_with_error_small_value_large_error_falls_back():
    assert ms.format_with_error(0.001, 0.05) == "0.001 ± 0.050"


def test_with_error_negative_value_large_error_falls_back():
    assert ms.format_with_error(-0.02, 0.3) == "-0.02 ± 0.30"


def test_with_error_zero_value_falls_back():
    assert ms.format_with_error(0.0, 0.1) == "0.00 ± 0.10"


def test_with_error_confidently_negative_value_keeps_digit_notation():
    # a negative integral with a small relative error must not hit the
    # log10(negative) NaN that the unguarded formula would produce
    result = ms.format_with_error(-5.0, 0.1)
    assert "nan" not in result.lower()
    assert result == "-5.00(10)"


def test_with_error_nan_error_does_not_crash():
    result = ms.format_with_error(1234.5, float("nan"))
    assert "nan" not in result.lower()


def test_with_error_zero_error_does_not_crash():
    result = ms.format_with_error(0.0, 0.0)
    assert "nan" not in result.lower()


def test_with_error_nonfinite_value_does_not_crash():
    result = ms.format_with_error(float("inf"), 1.0)
    assert isinstance(result, str)


def test_with_error_nonfinite_both_does_not_crash():
    result = ms.format_with_error(float("nan"), float("nan"))
    assert isinstance(result, str)


# --- format_progress ---------------------------------------------------------


def test_progress_zero_is_all_blank():
    assert ms.format_progress(0.0, 10) == " " * 10


def test_progress_full_is_all_blocks():
    assert ms.format_progress(1.0, 10) == "█" * 10


def test_progress_half_fills_half_the_width():
    assert ms.format_progress(0.5, 10) == "█████     "


def test_progress_uses_partial_block_characters():
    # 0.5 of a width-5 bar lands mid-cell, so the boundary cell is a partial glyph
    result = ms.format_progress(0.5, 5)
    assert result == "██▌  "
    assert len(result) == 5


def test_progress_clamps_negative_to_zero():
    assert ms.format_progress(-1.0, 5) == " " * 5


def test_progress_clamps_above_one_to_full():
    assert ms.format_progress(2.0, 5) == "█" * 5


def test_progress_output_length_matches_width():
    for progress in [0.0, 0.1, 0.37, 0.5, 0.99, 1.0]:
        assert len(ms.format_progress(progress, 20)) == 20
