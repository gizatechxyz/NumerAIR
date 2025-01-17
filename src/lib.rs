#![feature(portable_simd)]

use stwo_prover::core::fields::m31::{M31, P};

pub mod base;
pub mod eval;
pub mod packed;

// Number of bits used for decimal precision.
pub const DEFAULT_SCALE: u32 = 12;
// Scale factor = 2^DEFAULT_SCALE, used for fixed-point arithmetic.
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);
pub const SCALE_FACTOR_U32: u32 = 1 << DEFAULT_SCALE;
// Half the prime modulus.
pub const HALF_P: u32 = P / 2;

/// Trait for fixed-point arithmetic
pub trait FixedPoint {
    /// Creates a new fixed-point number from a float value.
    fn from_f64(x: f64) -> Self;

    /// Converts the fixed-point number back to float.
    fn to_f64(&self) -> f64;

    /// Returns the absolute value
    fn abs(&self) -> Self;

    /// Returns true if this represents a negative number
    /// Numbers greater than HALF_P are interpreted as negative
    fn is_negative(&self) -> bool;

    /// Adds two fixed-point numbers
    ///
    /// The addition is performed directly in the underlying field since:
    /// (a × 2^k) + (b × 2^k) = (a + b) × 2^k
    /// where k is the scaling factor
    fn fixed_add(&self, rhs: Self) -> Self;

    /// Subtracts two fixed-point numbers
    ///
    /// The subtraction is performed directly in the underlying field since:
    /// (a × 2^k) - (b × 2^k) = (a - b) × 2^k
    /// where k is the scaling factor
    fn fixed_sub(&self, rhs: Self) -> Self;

    /// Divides two fixed-point numbers
    ///
    /// To maintain precision during division, the numerator is first multiplied by
    /// an additional scaling factor:
    /// (a × 2^k) ÷ (b × 2^k) = a ÷ b
    /// result = (a × 2^k) ÷ b = (a ÷ b) × 2^k
    ///
    /// The division handles signed values by:
    /// 1. Converting inputs to absolute values
    /// 2. Performing unsigned division
    /// 3. Applying the correct sign to the result based on input signs
    ///
    /// # Panics
    /// Panics if rhs is zero
    fn fixed_div(&self, rhs: Self) -> Self;

    /// Multiply taking into account the fixed-point scale factor
    ///
    /// Since both inputs are scaled by 2^k, the product needs to be divided by 2^k:
    /// (a × 2^k) × (b × 2^k) = (a × b) × 2^2k
    /// result = ((a × b) × 2^2k) ÷ 2^k = (a × b) × 2^k
    fn fixed_mul_rem(&self, rhs: Self) -> (Self, Self)
    where
        Self: Sized;

    /// Performs signed division with remainder
    /// Returns (quotient, remainder) such that:
    /// - self = quotient * div + remainder
    /// - 0 <= remainder < |div|
    /// - quotient is rounded toward negative infinity
    fn fixed_div_rem(&self, div: Self) -> (Self, Self)
    where
        Self: Sized;
}
