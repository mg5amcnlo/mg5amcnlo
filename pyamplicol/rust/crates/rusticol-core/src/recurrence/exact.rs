// SPDX-License-Identifier: 0BSD

use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn checked_i128(value: u128, negative: bool, context: &str) -> RusticolResult<i128> {
    if value > i128::MAX as u128 {
        return Err(invalid(format!(
            "{context} exceeds the dependency-free exact i128 domain"
        )));
    }
    let value = value as i128;
    if negative {
        value
            .checked_neg()
            .ok_or_else(|| invalid(format!("{context} cannot be negated exactly")))
    } else {
        Ok(value)
    }
}

fn checked_divisor(value: u128, context: &str) -> RusticolResult<i128> {
    i128::try_from(value).map_err(|_| {
        invalid(format!(
            "{context} exceeds the dependency-free exact i128 domain"
        ))
    })
}

/// A canonical exact rational in the dependency-free, checked i128 domain.
///
/// Operations never round.  Values or intermediate results outside this
/// domain fail closed; callers must not replace such failures with binary64
/// arithmetic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, bincode::Encode)]
pub struct ExactRational {
    numerator: i128,
    denominator: i128,
}

impl<Context> bincode::Decode<Context> for ExactRational {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let numerator = bincode::Decode::decode(decoder)?;
        let denominator = bincode::Decode::decode(decoder)?;
        Self::new(numerator, denominator)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))
    }
}

bincode::impl_borrow_decode!(ExactRational);

impl ExactRational {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub fn new(numerator: i128, denominator: i128) -> RusticolResult<Self> {
        if denominator <= 0 {
            return Err(invalid("exact rational denominator must be positive"));
        }
        if numerator == 0 {
            return Ok(Self::ZERO);
        }
        if numerator == i128::MIN {
            return Err(invalid(
                "exact rational numerator exceeds the symmetric dependency-free i128 domain",
            ));
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator as u128);
        let divisor = checked_divisor(divisor, "exact rational gcd")?;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn parse_parts(numerator: &str, denominator: &str) -> RusticolResult<Self> {
        let numerator = parse_signed_decimal(numerator, "numerator")?;
        let denominator = parse_positive_decimal(denominator, "denominator")?;
        Self::new(numerator, denominator)
    }

    /// Convert a finite binary64 value to its exact dyadic rational whenever
    /// that rational fits in the checked i128 domain.
    pub fn from_f64_exact(value: f64) -> RusticolResult<Self> {
        if !value.is_finite() {
            return Err(invalid("exact rational source must be finite binary64"));
        }
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        if exponent_bits == 0 && fraction == 0 {
            return Ok(Self::ZERO);
        }

        let (mut significand, mut exponent) = if exponent_bits == 0 {
            (u128::from(fraction), -1074_i32)
        } else {
            (
                u128::from((1_u64 << 52) | fraction),
                exponent_bits - 1023 - 52,
            )
        };
        let trailing = significand.trailing_zeros() as i32;
        significand >>= trailing;
        exponent += trailing;

        if exponent >= 0 {
            let shifted = significand
                .checked_shl(exponent as u32)
                .ok_or_else(|| invalid("binary64 numerator exceeds exact i128 domain"))?;
            return Self::new(checked_i128(shifted, negative, "binary64 numerator")?, 1);
        }

        let denominator = 1_u128
            .checked_shl((-exponent) as u32)
            .ok_or_else(|| invalid("binary64 denominator exceeds exact i128 domain"))?;
        Self::new(
            checked_i128(significand, negative, "binary64 numerator")?,
            checked_divisor(denominator, "binary64 denominator")?,
        )
    }

    pub const fn numerator(self) -> i128 {
        self.numerator
    }

    pub const fn denominator(self) -> i128 {
        self.denominator
    }

    pub const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    pub const fn is_one(self) -> bool {
        self.numerator == 1 && self.denominator == 1
    }

    pub fn checked_neg(self) -> RusticolResult<Self> {
        Self::new(
            self.numerator
                .checked_neg()
                .ok_or_else(|| invalid("exact rational negation overflow"))?,
            self.denominator,
        )
    }

    pub fn checked_add(self, right: Self) -> RusticolResult<Self> {
        let common = checked_divisor(
            gcd(self.denominator as u128, right.denominator as u128),
            "exact rational denominator gcd",
        )?;
        let left_factor = right.denominator / common;
        let right_factor = self.denominator / common;
        let left = self
            .numerator
            .checked_mul(left_factor)
            .ok_or_else(|| invalid("exact rational addition numerator overflow"))?;
        let right_term = right
            .numerator
            .checked_mul(right_factor)
            .ok_or_else(|| invalid("exact rational addition numerator overflow"))?;
        let numerator = left
            .checked_add(right_term)
            .ok_or_else(|| invalid("exact rational addition numerator overflow"))?;
        let denominator = right_factor
            .checked_mul(right.denominator)
            .ok_or_else(|| invalid("exact rational addition denominator overflow"))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_sub(self, right: Self) -> RusticolResult<Self> {
        self.checked_add(right.checked_neg()?)
    }

    pub fn checked_mul(self, right: Self) -> RusticolResult<Self> {
        if self.is_zero() || right.is_zero() {
            return Ok(Self::ZERO);
        }
        let left_cross = checked_divisor(
            gcd(self.numerator.unsigned_abs(), right.denominator as u128),
            "exact rational multiplication cross-gcd",
        )?;
        let right_cross = checked_divisor(
            gcd(right.numerator.unsigned_abs(), self.denominator as u128),
            "exact rational multiplication cross-gcd",
        )?;
        let numerator = (self.numerator / left_cross)
            .checked_mul(right.numerator / right_cross)
            .ok_or_else(|| invalid("exact rational multiplication numerator overflow"))?;
        let denominator = (self.denominator / right_cross)
            .checked_mul(right.denominator / left_cross)
            .ok_or_else(|| invalid("exact rational multiplication denominator overflow"))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_div(self, right: Self) -> RusticolResult<Self> {
        if right.is_zero() {
            return Err(invalid("exact rational division by zero"));
        }
        let reciprocal_numerator = if right.numerator < 0 {
            right
                .denominator
                .checked_neg()
                .ok_or_else(|| invalid("exact rational reciprocal overflow"))?
        } else {
            right.denominator
        };
        let reciprocal_denominator = checked_divisor(
            right.numerator.unsigned_abs(),
            "exact rational reciprocal denominator",
        )?;
        self.checked_mul(Self::new(reciprocal_numerator, reciprocal_denominator)?)
    }

    pub fn checked_square(self) -> RusticolResult<Self> {
        self.checked_mul(self)
    }
}

impl Ord for ExactRational {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.numerator.signum(), other.numerator.signum()) {
            (left, right) if left != right => left.cmp(&right),
            (0, 0) => Ordering::Equal,
            (1, 1) => compare_positive(
                self.numerator as u128,
                self.denominator as u128,
                other.numerator as u128,
                other.denominator as u128,
            ),
            (-1, -1) => compare_positive(
                self.numerator.unsigned_abs(),
                self.denominator as u128,
                other.numerator.unsigned_abs(),
                other.denominator as u128,
            )
            .reverse(),
            _ => unreachable!("i128 signum has only -1, 0, and 1"),
        }
    }
}

impl PartialOrd for ExactRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for ExactRational {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

impl FromStr for ExactRational {
    type Err = RusticolError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (numerator, denominator) = value
            .split_once('/')
            .ok_or_else(|| invalid("exact rational must use numerator/denominator syntax"))?;
        if denominator.contains('/') {
            return Err(invalid(
                "exact rational must contain exactly one slash separator",
            ));
        }
        Self::parse_parts(numerator, denominator)
    }
}

/// Canonical exact complex rational used by recurrence proof coefficients.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, bincode::Encode, bincode::Decode,
)]
pub struct ExactComplexRational {
    real: ExactRational,
    imag: ExactRational,
}

impl ExactComplexRational {
    pub const ZERO: Self = Self {
        real: ExactRational::ZERO,
        imag: ExactRational::ZERO,
    };
    pub const ONE: Self = Self {
        real: ExactRational::ONE,
        imag: ExactRational::ZERO,
    };

    pub const fn new(real: ExactRational, imag: ExactRational) -> Self {
        Self { real, imag }
    }

    pub fn parse_parts(
        real_numerator: &str,
        real_denominator: &str,
        imag_numerator: &str,
        imag_denominator: &str,
    ) -> RusticolResult<Self> {
        Ok(Self::new(
            ExactRational::parse_parts(real_numerator, real_denominator)?,
            ExactRational::parse_parts(imag_numerator, imag_denominator)?,
        ))
    }

    pub const fn real(self) -> ExactRational {
        self.real
    }

    pub const fn imag(self) -> ExactRational {
        self.imag
    }

    pub const fn is_zero(self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }

    pub fn checked_neg(self) -> RusticolResult<Self> {
        Ok(Self::new(
            self.real.checked_neg()?,
            self.imag.checked_neg()?,
        ))
    }

    pub fn checked_add(self, right: Self) -> RusticolResult<Self> {
        Ok(Self::new(
            self.real.checked_add(right.real)?,
            self.imag.checked_add(right.imag)?,
        ))
    }

    pub fn checked_sub(self, right: Self) -> RusticolResult<Self> {
        Ok(Self::new(
            self.real.checked_sub(right.real)?,
            self.imag.checked_sub(right.imag)?,
        ))
    }

    pub fn checked_mul(self, right: Self) -> RusticolResult<Self> {
        let real = self
            .real
            .checked_mul(right.real)?
            .checked_sub(self.imag.checked_mul(right.imag)?)?;
        let imag = self
            .real
            .checked_mul(right.imag)?
            .checked_add(self.imag.checked_mul(right.real)?)?;
        Ok(Self::new(real, imag))
    }

    pub fn checked_div(self, right: Self) -> RusticolResult<Self> {
        if right.is_zero() {
            return Err(invalid("exact complex rational division by zero"));
        }
        let denominator = right
            .real
            .checked_square()?
            .checked_add(right.imag.checked_square()?)?;
        let real = self
            .real
            .checked_mul(right.real)?
            .checked_add(self.imag.checked_mul(right.imag)?)?
            .checked_div(denominator)?;
        let imag = self
            .imag
            .checked_mul(right.real)?
            .checked_sub(self.real.checked_mul(right.imag)?)?
            .checked_div(denominator)?;
        Ok(Self::new(real, imag))
    }

    pub fn conjugate(self) -> RusticolResult<Self> {
        Ok(Self::new(self.real, self.imag.checked_neg()?))
    }
}

fn parse_signed_decimal(value: &str, field: &str) -> RusticolResult<i128> {
    if value.is_empty() || value.trim() != value || value.as_bytes().contains(&b'_') {
        return Err(invalid(format!(
            "exact rational {field} must be an unseparated decimal integer"
        )));
    }
    let parsed = value.parse::<i128>().map_err(|_| {
        invalid(format!(
            "exact rational {field} exceeds the dependency-free exact i128 domain"
        ))
    })?;
    if parsed == i128::MIN {
        return Err(invalid(format!(
            "exact rational {field} exceeds the symmetric dependency-free i128 domain"
        )));
    }
    Ok(parsed)
}

fn parse_positive_decimal(value: &str, field: &str) -> RusticolResult<i128> {
    if value.starts_with('-') {
        return Err(invalid(format!("exact rational {field} must be positive")));
    }
    let parsed = parse_signed_decimal(value, field)?;
    if parsed <= 0 {
        return Err(invalid(format!("exact rational {field} must be positive")));
    }
    Ok(parsed)
}

/// Compare positive fractions without cross multiplication, so ordering stays
/// exact even when the cross-products exceed i128/u128.
fn compare_positive(
    mut left_numerator: u128,
    mut left_denominator: u128,
    mut right_numerator: u128,
    mut right_denominator: u128,
) -> Ordering {
    let mut reverse = false;
    loop {
        let left_quotient = left_numerator / left_denominator;
        let right_quotient = right_numerator / right_denominator;
        let quotient_order = left_quotient.cmp(&right_quotient);
        if quotient_order != Ordering::Equal {
            return if reverse {
                quotient_order.reverse()
            } else {
                quotient_order
            };
        }

        let left_remainder = left_numerator % left_denominator;
        let right_remainder = right_numerator % right_denominator;
        match (left_remainder == 0, right_remainder == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => {
                return if reverse {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
            }
            (false, true) => {
                return if reverse {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
            (false, false) => {}
        }

        left_numerator = left_denominator;
        left_denominator = left_remainder;
        right_numerator = right_denominator;
        right_denominator = right_remainder;
        reverse = !reverse;
    }
}
