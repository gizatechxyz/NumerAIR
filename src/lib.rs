use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};
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
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
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

    #[inline]
    /// Division with remainder for constraints
    pub fn div_rem(self, rhs: Self) -> (Self, Self) {
        assert!(rhs.0 != 0, "Division by zero");

        let quotient = (((self.0 as i128) << DEFAULT_SCALE) / (rhs.0 as i128)) as i64;
        let remainder = (self.0 as i128) - ((quotient as i128 * rhs.0 as i128) >> DEFAULT_SCALE);
        (Self(quotient), Self(remainder as i64))
    }

    /// Computes the square root in the fully scaled domain.
    //
    // Given that Fixed::0 = floor(real_value * SCALE_FACTOR),
    // we want to compute `out` and `rem` such that:
    //
    //     out^2 + rem = input * SCALE_FACTOR
    //
    // where `out` is the fixed-point representation of sqrt(real_value)
    // and 0 <= rem < SCALE_FACTOR.
    pub fn sqrt(&self) -> (Self, Self) {
        if self.0 <= 0 {
            return (Self(0), Self(0));
        }
        // Multiply self.0 by SCALE_FACTOR to move to the "squared" domain.
        let t = self.0 * (1 << DEFAULT_SCALE);
        // Compute the integer square root of t.
        let y = int_sqrt(t as u64) as i64;
        let squared = y * y;
        let remainder = t - squared;
        (Self(y), Self(remainder))
    }
}

/// Returns the floor of the square root of `n`.
pub fn int_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // Initial guess.
    let mut x = n / 2;
    // Newton's method iteration.
    while x * x > n {
        x = (x + n / x) / 2;
    }
    x
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
        let product = (self.0 as i128) * (rhs.0 as i128);
        let quotient = (product >> DEFAULT_SCALE) as i64;
        let remainder = (product & (REMAINDER_MASK as i128)) as i64;
        (Self(quotient), Self(remainder))
    }
}

impl Div for Fixed {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self((((self.0 as i128) << DEFAULT_SCALE) / (rhs.0 as i128)) as i64)
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
    fn test_div() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..5 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);

            let result = (fa / fb).to_f64();
            let expected = a / b;

            assert_near(result, expected);
        }
    }

    #[test]
    fn test_div_edge_cases() {
        // Test specific cases
        let test_cases = vec![
            (3.5, 2.0, 1.75),   // Simple positive division
            (-3.5, 2.0, -1.75), // Negative numerator
            (3.5, -2.0, -1.75), // Negative denominator
            (-3.5, -2.0, 1.75), // Both negative
            (1.0, 2.0, 0.5),    // Fraction less than 1
            (0.0, 2.0, 0.0),    // Zero numerator
            (1.0, 0.5, 2.0),    // Denominator less than 1
        ];

        for (a, b, expected) in test_cases {
            let fa = Fixed::from_f64(a);
            let fb = Fixed::from_f64(b);
            let result = (fa / fb).to_f64();
            assert_near(result, expected);
        }
    }

    #[test]
    fn test_sqrt() {
        let mut test_cases = vec![
            (0.0, 0.0),
            (-1.0, 0.0), // Negative input: sqrt should return 0
            (1.0, 1.0),
            (4.0, 2.0),
            (2.0, std::f64::consts::SQRT_2), // Irrational result
            (9.0, 3.0),
            (1e6, 1e3),
            (10.0, 3.162277), // Larger input
            (0.25, 0.5),      // Fractional input
        ];

        let mut rng = StdRng::seed_from_u64(42);
        // Only generate random values in a range safely representable.
        for _ in 0..100 {
            let value: f64 = rng.gen_range(1e-3..1e6);
            test_cases.push((value, value.sqrt()));
        }

        for (input, expected) in test_cases {
            let fixed_input = Fixed::from_f64(input);

            if input < 0.0 {
                println!("Input: {:.6e} (negative)", input);
                let (result, remainder) = fixed_input.sqrt();
                assert_eq!(result.0, 0);
                assert_eq!(remainder.0, 0);
                continue;
            }

            let (result, _) = fixed_input.sqrt();
            let result_f64 = result.to_f64();

            println!("Input: {:.6e}, Expected:{}, Result: {:.6e}", input, expected, fixed_input.to_f64());
            assert_near(result_f64, expected);
        }
    }
}
