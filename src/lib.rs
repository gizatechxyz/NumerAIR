use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
use stwo_prover::core::fields::m31::{M31, P};

pub mod eval;

// Half the prime modulus.
pub const HALF_P: u32 = P / 2;

/// Scale parameter marker trait
pub trait FixedScale {
    const SCALE: u32;
    const SCALE_FACTOR: i64 = 1 << Self::SCALE;
    const HALF_SCALE: i64 = 1 << (Self::SCALE - 1);
}

/// Default scale (15 bits of precision)
#[derive(Copy, Clone, Debug)]
pub struct DefaultScale;
impl FixedScale for DefaultScale {
    const SCALE: u32 = 15;
}

/// Integer representation of fixed-point Basefield with parametrized scale.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixed<S: FixedScale = DefaultScale>(pub i64, PhantomData<S>);

impl<S: FixedScale> Fixed<S> {
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        Self((value * S::SCALE_FACTOR as f64).round() as i64, PhantomData)
    }

    #[inline]
    /// Convert to a float
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / S::SCALE_FACTOR as f64
    }

    #[inline]
    /// Convert to M31 for Stwo
    pub fn to_m31(self) -> M31 {
        const MODULUS_BITS: u32 = 31;

        if self.0 >= 0 {
            // For positive numbers, use M31 reduce directly
            M31::reduce(self.0 as u64)
        } else {
            // For negative numbers, efficiently compute P - (abs(value) % P)
            // This is a fast implementation of modulo for 2^31-1
            let abs_val = (-self.0) as u64;
            let abs_mod = (((((abs_val >> MODULUS_BITS) + abs_val + 1) >> MODULUS_BITS) + abs_val)
                & (P as u64)) as u32;

            if abs_mod == 0 {
                M31::from_u32_unchecked(0) // -0 = 0
            } else {
                M31::from_u32_unchecked(P - abs_mod)
            }
        }
    }

    #[inline]
    /// Convert from M31
    pub fn from_m31(value: M31) -> Self {
        let m31_val = value.0;
        let is_negative = m31_val > HALF_P;

        if is_negative {
            Self(-((P - m31_val) as i64), PhantomData)
        } else {
            Self(m31_val as i64, PhantomData)
        }
    }

    /// Computes the reciprocal (1/x) of a fixed-point number
    ///
    /// Returns a tuple of (quotient, remainder) where:
    /// - quotient is the fixed-point representation of 1/x
    /// - remainder is the remainder after division
    #[inline]
    pub fn recip(self) -> (Self, Self) {
        assert!(self.0 != 0, "Division by zero");

        let scale_squared = S::SCALE_FACTOR * S::SCALE_FACTOR;
        let quotient = scale_squared / self.0;
        let remainder = scale_squared % self.0;

        (Self(quotient, PhantomData), Self(remainder, PhantomData))
    }

    /// Computes the fixed-point representation of the square root and its remainder.
    ///
    /// `self` represents `input_val * SCALE_FACTOR`, to compute
    /// `out` and `rem`, i.e., `out` represents `sqrt(input_val) * SCALE_FACTOR`
    /// and the following hold for their underlying integer values
    /// (`input.0`, `out.0`, `rem.0`):
    ///
    /// `out.0^2 + rem.0 = input.0 * SCALE_FACTOR`
    ///
    /// where `out.0` is the integer square root of `(input.0 * SCALE_FACTOR)`.
    /// The remainder `rem.0` is the difference `(input.0 * SCALE_FACTOR) - out.0^2`.
    pub fn sqrt(&self) -> (Self, Self) {
        // Panic for negative inputs
        assert!(self.0 >= 0, "Cannot compute square root of negative number");

        // Special case: zero input
        if self.0 == 0 {
            return (Self(0, PhantomData), Self(0, PhantomData));
        }

        // Calculate value to compute sqrt of: self * SCALE_FACTOR
        let input_scaled = (self.0 as u64) << S::SCALE;

        // Compute integer square root
        let sqrt_val = int_sqrt(input_scaled);

        // Calculate remainder (input_scaled - sqrt_val^2)
        let remainder = input_scaled - sqrt_val * sqrt_val;

        (
            Self(sqrt_val as i64, PhantomData),
            Self(remainder as i64, PhantomData),
        )
    }
}

/// Returns the floor of the square root of `n`.
#[inline]
pub fn int_sqrt(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }

    // Initial guess
    let bits = 64 - n.leading_zeros();
    let mut x = n >> (bits / 2);

    // Ensure x is not zero (which would cause division by zero)
    if x == 0 {
        x = 1;
    }

    // Newton's method with careful convergence checking
    let mut prev_x = x;
    loop {
        // Compute next iteration
        let quotient = n / x;
        let next_x = (x + quotient) / 2; // We can use regular division here since x + quotient ≤ n + 1

        // Check for convergence or oscillation
        if next_x == x || next_x == prev_x {
            return next_x;
        }

        // Update for next iteration
        prev_x = x;
        x = next_x;
    }
}

impl<S: FixedScale> Add for Fixed<S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, PhantomData)
    }
}

impl<S: FixedScale> Sub for Fixed<S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, PhantomData)
    }
}

impl<S: FixedScale> Mul for Fixed<S> {
    type Output = (Self, Self);

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let product = self.0 * rhs.0;

        let quotient = (product + S::HALF_SCALE) >> S::SCALE;

        // Calculate remainder to maintain: product = quotient * scale + remainder
        let scaled_quotient = quotient << S::SCALE;
        let remainder = product - scaled_quotient;

        (Self(quotient, PhantomData), Self(remainder, PhantomData))
    }
}

impl<S: FixedScale> Zero for Fixed<S> {
    #[inline]
    fn zero() -> Self {
        Self(0, PhantomData)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

// Define additional scale configurations as needed
#[derive(Copy, Clone, Debug)]
pub struct Scale8;
impl FixedScale for Scale8 {
    const SCALE: u32 = 8;
}

#[derive(Copy, Clone, Debug)]
pub struct Scale24;
impl FixedScale for Scale24 {
    const SCALE: u32 = 24;
}

// Implement conversions between different scales
impl<S1: FixedScale> Fixed<S1> {
    pub fn convert_to<T: FixedScale>(self) -> Fixed<T> {
        if S1::SCALE == T::SCALE {
            // Same scale, just change the type
            Fixed(self.0, PhantomData)
        } else if S1::SCALE < T::SCALE {
            // Going to higher precision
            let shift = T::SCALE - S1::SCALE;
            Fixed(self.0 << shift, PhantomData)
        } else {
            // Going to lower precision
            let shift = S1::SCALE - T::SCALE;
            Fixed(self.0 >> shift, PhantomData)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const EPSILON: f64 = 1e-3;

    fn assert_near(a: f64, b: f64) {
        assert!((a - b).abs() < EPSILON, "Expected {} to be near {}", a, b);
    }

    #[test]
    fn test_negative() {
        let a: Fixed = Fixed::<DefaultScale>::from_f64(-3.5);
        let b = Fixed::<DefaultScale>::from_f64(2.0);

        assert_near(a.to_f64(), -3.5);
        assert_near((a + b).to_f64(), -1.5);
        assert_near((a - b).to_f64(), -5.5);
    }

    #[test]
    fn test_add() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = Fixed::<DefaultScale>::from_f64(a);
            let fb = Fixed::<DefaultScale>::from_f64(b);

            assert_near((fa + fb).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = Fixed::<DefaultScale>::from_f64(a);
            let fb = Fixed::<DefaultScale>::from_f64(b);

            assert_near((fa - fb).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = Fixed::<DefaultScale>::from_f64(a);
            let fb = Fixed::<DefaultScale>::from_f64(b);

            let (q, _) = fa * fb;
            let expected = a * b;

            assert_near(q.to_f64(), expected);
        }
    }

    #[test]
    fn test_recip() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            if a.abs() < 0.1 {
                continue;
            }

            let fixed_a = Fixed::<DefaultScale>::from_f64(a);
            let (recip, _) = fixed_a.recip();
            let expected = 1.0 / a;

            assert_near(recip.to_f64(), expected);
        }

        // Test specific cases
        let test_cases = vec![
            (1.0, 1.0),   // Reciprocal of 1 is 1
            (2.0, 0.5),   // Reciprocal of 2 is 0.5
            (0.5, 2.0),   // Reciprocal of 0.5 is 2
            (4.0, 0.25),  // Reciprocal of 4 is 0.25
            (-1.0, -1.0), // Reciprocal of -1 is -1
            (-2.0, -0.5), // Reciprocal of -2 is -0.5
        ];

        for (a, expected) in test_cases {
            let fixed_a = Fixed::<DefaultScale>::from_f64(a);
            let (recip, _) = fixed_a.recip();
            assert_near(recip.to_f64(), expected);
        }
    }

    #[test]
    fn test_sqrt() {
        let mut test_cases = vec![
            0.0, 1.0, 4.0, 9.0, 10.0, 16.0, 25.0, 81.0, 100.0, 0.25, 0.0625, 0.01, 5.0, 8.0, 12.0,
            15.0, 20.0, 50.0, 10000.0, 1000000.0, // Large value
            1e-10,     // Small value
            0.001,     // rest irrationals
            0.5, 2.0, 3.0, 42.0, // Nod to Douglas Adams
        ];

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let value: f64 = rng.gen_range(0.01..50.0);
            test_cases.push(value);
        }

        for input in test_cases {
            let fixed_input = Fixed::<DefaultScale>::from_f64(input);

            if input < 0.0 {
                let (result, remainder) = fixed_input.sqrt();
                assert_eq!(result.0, 0);
                assert_eq!(remainder.0, 0);
                continue;
            }

            let (result, _) = fixed_input.sqrt();
            let result_f64 = result.to_f64();
            assert_near(result_f64, input.sqrt());
        }
    }

    #[test]
    fn test_different_scales() {
        // Test with default scale (15 bits)
        let default_scale = Fixed::<DefaultScale>::from_f64(1.5);
        assert_near(default_scale.to_f64(), 1.5);

        // Test with 8-bit scale (less precision)
        let scale8 = Fixed::<Scale8>::from_f64(1.5);
        assert_near(scale8.to_f64(), 1.5);

        // Test with 24-bit scale (more precision)
        let scale24 = Fixed::<Scale24>::from_f64(1.5);
        assert_near(scale24.to_f64(), 1.5);

        // Test conversion between scales
        let from_default_to_8 = default_scale.convert_to::<Scale8>();
        assert_near(from_default_to_8.to_f64(), 1.5);

        let from_8_to_24 = scale8.convert_to::<Scale24>();
        assert_near(from_8_to_24.to_f64(), 1.5);

        // Multiplication with different scales
        let a8 = Fixed::<Scale8>::from_f64(2.5);
        let b8 = Fixed::<Scale8>::from_f64(3.0);
        let (result8, _) = a8 * b8;
        assert_near(result8.to_f64(), 7.5);

        let a24 = Fixed::<Scale24>::from_f64(2.5);
        let b24 = Fixed::<Scale24>::from_f64(3.0);
        let (result24, _) = a24 * b24;
        assert_near(result24.to_f64(), 7.5);
    }
}
