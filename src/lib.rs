use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub, Rem};
use stwo_prover::core::fields::m31::{M31, P};

pub mod eval;

// Number of bits used for decimal precision.
pub const DEFAULT_SCALE: u32 = 12;
// Scale factor = 2^DEFAULT_SCALE, used for fixed-point arithmetic.
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);
// Half the prime modulus.
pub const HALF_P: u32 = P / 2;
// Mask for remainder in fixed-point operations (2^DEFAULT_SCALE - 1)
const REMAINDER_MASK: i64 = (1 << DEFAULT_SCALE) - 1;

/// Integer representation of fixed-point Basefield.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixed(pub i64);

impl Fixed {
    const SCALE_FACTOR: i64 = 1 << DEFAULT_SCALE;

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

    /// Computes the reciprocal (1/x) of a fixed-point number
    ///
    /// Returns a tuple of (quotient, remainder) where:
    /// - quotient is the fixed-point representation of 1/x
    /// - remainder is the remainder after division
    #[inline]
    pub fn recip(self) -> (Self, Self) {
        assert!(self.0 != 0, "Division by zero");

        let scale_squared = Self::SCALE_FACTOR * Self::SCALE_FACTOR;
        let quotient = scale_squared / self.0;
        let remainder = scale_squared % self.0;

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
        let input_scaled = (self.0 as u64) << DEFAULT_SCALE;

        // Compute integer square root
        let sqrt_val = int_sqrt(input_scaled);

        // Calculate remainder (input_scaled - sqrt_val^2)
        let remainder = input_scaled - sqrt_val * sqrt_val;

        (Self(sqrt_val as i64), Self(remainder as i64))
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

impl Add for Fixed {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Fixed {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for Fixed {
    type Output = (Self, Self);

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let product = self.0 * rhs.0;
        (
            Self(product >> DEFAULT_SCALE),
            Self(product & REMAINDER_MASK),
        )
    }
}

impl Rem for Fixed {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl Zero for Fixed {
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

    const EPSILON: f64 = 1e-2;

    fn assert_near(a: f64, b: f64) {
        assert!((a - b).abs() < EPSILON, "Expected {} to be near {}", a, b);
    }

    #[test]
    fn test_negative() {
        let a = Fixed::from_f64(-3.5);
        let b = Fixed::from_f64(2.0);

        assert_near(a.to_f64(), -3.5);
        assert_near((a + b.clone()).to_f64(), -1.5);
        assert_near((a - b).to_f64(), -5.5);
    }

    #[test]
    fn test_add() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);

            assert_near((fa + fb).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);

            assert_near((fa - fb).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);

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

            let fixed_a = Fixed::from_f64(a);
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
            let fixed_a = Fixed::from_f64(a);
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
            let fixed_input = Fixed::from_f64(input);

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
        let test_cases = vec![
            (5.0, 2.0, 1.0),
            (-5.0, 2.0, -1.0),
            (5.0, -2.0, 1.0),
            (-5.0, -2.0, -1.0),
            (7.5, 2.5, 0.0),
            (3.2, 1.5, 0.2),
        ];

        for (a, b, expected) in test_cases.clone() {
            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);
            let result = (fa % fb).to_f64();
            assert_near(result, expected);
        }

        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 100.0;
            let b = (rng.gen::<f64>() - 0.5) * 100.0;
            let expected = a % b;

            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);

            let result = (fa % fb).to_f64();
            assert_near(result, expected);
        }
    }

    #[test]
    fn test_zero() {
        assert!(Fixed::zero().is_zero());
        assert!(!Fixed::from_f64(1.0).is_zero());
    }
}
