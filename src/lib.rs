use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Rem, Sub};
use stwo::core::fields::m31::{M31, P};

pub mod eval;

// Half the prime modulus.
pub const HALF_P: u32 = P / 2;

/// Integer representation of fixed-point Basefield with parametrized scale.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixed<const SCALE: u32>(pub i64);

impl<const SCALE: u32> Fixed<SCALE> {
    const SCALE_FACTOR: i64 = 1 << SCALE;
    const HALF_SCALE_FACTOR: i64 = 1 << (SCALE - 1);

    #[inline]
    pub fn from_f64(value: f64) -> Self {
        Self((value * Self::SCALE_FACTOR as f64).round() as i64)
    }

    #[inline]
    /// Convert to a float
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE_FACTOR as f64
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
            Self(-((P - m31_val) as i64))
        } else {
            Self(m31_val as i64)
        }
    }

    /// Computes both quotient and remainder for division.
    /// Returns (quotient, remainder) where dividend = quotient * divisor + remainder
    /// Note: The quotient is stored as an unscaled integer for constraint compatibility.
    #[inline]
    pub fn div_rem(self, rhs: Self) -> (Self, Self) {
        assert!(rhs.0 != 0, "Division by zero");
        let quotient = self.0 / rhs.0;
        let remainder = self.0 % rhs.0;
        (Self(quotient), Self(remainder))
    }

    /// Computes the reciprocal (1/x) of a fixed-point number
    ///
    /// Returns a tuple of (quotient, remainder) where:
    /// - quotient is the fixed-point representation of 1/x
    /// - remainder is the remainder after division
    #[inline]
    pub fn recip(self) -> (Self, Self) {
        assert!(self.0 != 0, "Division by zero");

        let scale_factor_squared = Self::SCALE_FACTOR * Self::SCALE_FACTOR;
        let quotient = scale_factor_squared / self.0;
        let remainder = scale_factor_squared % self.0;

        (Self(quotient), Self(remainder))
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
            return (Self(0), Self(0));
        }

        // Calculate value to compute sqrt of: self * SCALE_FACTOR
        let input_scaled = (self.0 as u64) << SCALE;

        // Compute integer square root
        let sqrt_val = int_sqrt(input_scaled);

        // Calculate remainder (input_scaled - sqrt_val^2)
        let remainder = input_scaled - sqrt_val * sqrt_val;

        (Self(sqrt_val as i64), Self(remainder as i64))
    }

    /// Convert this Fixed value to a Fixed with a different scale
    pub fn convert_to<const TARGET_SCALE: u32>(self) -> Fixed<TARGET_SCALE> {
        if TARGET_SCALE == SCALE {
            // Same scale, just change the type
            Fixed(self.0)
        } else if TARGET_SCALE > SCALE {
            // Going to higher precision
            let shift = TARGET_SCALE - SCALE;
            Fixed(self.0 << shift)
        } else {
            // Going to lower precision
            let shift = SCALE - TARGET_SCALE;
            Fixed(self.0 >> shift)
        }
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

impl<const SCALE: u32> Add for Fixed<SCALE> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const SCALE: u32> Sub for Fixed<SCALE> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const SCALE: u32> Mul for Fixed<SCALE> {
    type Output = (Self, Self);

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let product = self.0 * rhs.0;

        let quotient = (product + Self::HALF_SCALE_FACTOR) >> SCALE;

        // Calculate remainder to maintain: product = quotient * scale + remainder
        let scaled_quotient = quotient << SCALE;
        let remainder = product - scaled_quotient;

        (Self(quotient), Self(remainder))
    }
}

impl<const SCALE: u32> Rem for Fixed<SCALE> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        assert!(rhs.0 != 0, "Division by zero in remainder operation");
        Self(self.0 % rhs.0)
    }
}

impl<const SCALE: u32> Zero for Fixed<SCALE> {
    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
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
        let a = Fixed::<15>::from_f64(-3.5);
        let b = Fixed::<15>::from_f64(2.0);

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

            let fa = Fixed::<15>::from_f64(a);
            let fb = Fixed::<15>::from_f64(b);

            assert_near((fa + fb).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = Fixed::<15>::from_f64(a);
            let fb = Fixed::<15>::from_f64(b);

            assert_near((fa - fb).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = Fixed::<15>::from_f64(a);
            let fb = Fixed::<15>::from_f64(b);

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

            let fixed_a = Fixed::<15>::from_f64(a);
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
            let fixed_a = Fixed::<15>::from_f64(a);
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
            let fixed_input = Fixed::<15>::from_f64(input);

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
    fn test_rem() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test random cases
        for _ in 0..100 {
            let a = (rng.gen::<f64>() - 0.5) * 20.0;
            let b = (rng.gen::<f64>() - 0.5) * 20.0;

            // Skip cases where divisor is too close to zero
            if b.abs() < 0.1 {
                continue;
            }

            let fa = Fixed::<15>::from_f64(a);
            let fb = Fixed::<15>::from_f64(b);

            let remainder = fa % fb;
            let expected = a % b;

            assert_near(remainder.to_f64(), expected);
        }

        // Test specific cases
        let test_cases = vec![
            (10.0, 3.0),   // 10 % 3 = 1
            (7.5, 2.5),    // 7.5 % 2.5 = 0
            (9.0, 4.0),    // 9 % 4 = 1
            (-10.0, 3.0),  // -10 % 3 = -1 (or 2, depending on implementation)
            (10.0, -3.0),  // 10 % -3 = 1 (or -2, depending on implementation)
            (-10.0, -3.0), // -10 % -3 = -1 (or 2, depending on implementation)
        ];

        for (a, b) in test_cases {
            let fa = Fixed::<15>::from_f64(a);
            let fb = Fixed::<15>::from_f64(b);
            let remainder = fa % fb;
            let expected = a % b;

            assert_near(remainder.to_f64(), expected);
        }
    }

    #[test]
    fn test_div_rem() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let a = (rng.gen::<f64>() - 0.5) * 20.0;
            let b = (rng.gen::<f64>() - 0.5) * 20.0;
            println!("a {:?}", a);
            println!("b {:?}", b);

            // Skip cases where divisor is too close to zero
            if b.abs() < 0.1 {
                continue;
            }

            let fa = Fixed::<15>::from_f64(a);
            let fb = Fixed::<15>::from_f64(b);

            let (quotient, remainder) = fa.div_rem(fb);

            // Verify: dividend = quotient * divisor + remainder
            let reconstructed = quotient.0 * fb.0 + remainder.0;
            assert_eq!(reconstructed, fa.0);

            // Check individual results
            let expected_quotient = (a / b).trunc();
            let expected_remainder = a % b;

            // The quotient from div_rem is stored as an unscaled integer
            assert_eq!(quotient.0 as f64, expected_quotient);
            assert_near(remainder.to_f64(), expected_remainder);
        }
    }

    #[test]
    fn test_different_scales() {
        // Test with 15-bit scale
        let scale_15 = Fixed::<15>::from_f64(1.5);
        assert_near(scale_15.to_f64(), 1.5);

        // Test with 8-bit scale (less precision)
        let scale8 = Fixed::<8>::from_f64(1.5);
        assert_near(scale8.to_f64(), 1.5);

        // Test with 24-bit scale (more precision)
        let scale24 = Fixed::<24>::from_f64(1.5);
        assert_near(scale24.to_f64(), 1.5);

        // Test conversion between scales
        let from_15_to_8 = scale_15.convert_to::<8>();
        assert_near(from_15_to_8.to_f64(), 1.5);

        let from_8_to_24 = scale8.convert_to::<24>();
        assert_near(from_8_to_24.to_f64(), 1.5);

        // Multiplication with different scales
        let a8 = Fixed::<8>::from_f64(2.5);
        let b8 = Fixed::<8>::from_f64(3.0);
        let (result8, _) = a8 * b8;
        assert_near(result8.to_f64(), 7.5);

        let a24 = Fixed::<24>::from_f64(2.5);
        let b24 = Fixed::<24>::from_f64(3.0);
        let (result24, _) = a24 * b24;
        assert_near(result24.to_f64(), 7.5);
    }
}
