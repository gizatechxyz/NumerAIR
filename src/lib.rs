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

/// Integer representation of fixed-point Basefield.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct IntRep(pub i64);

impl IntRep {
    const SCALE_FACTOR: i64 = 1 << DEFAULT_SCALE;

    /// Convert from a float
    pub fn from_f64(value: f64) -> Self {
        Self((value * Self::SCALE_FACTOR as f64).round() as i64)
    }

    /// Convert to a float
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE_FACTOR as f64
    }

    /// Convert to M31 for Stwo
    pub fn to_m31(self) -> M31 {
        const MODULUS_BITS: u32 = 31;

        if self.0 >= 0 {
            // For positive numbers, we use the M31 reduce directly
            let val = self.0 as u64;
            M31::reduce(val)
        } else {
            // For negative numbers, we need to compute P - (abs(value) % P)
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

    /// Convert from M31
    pub fn from_m31(value: M31) -> Self {
        let m31_val = value.0;
        let is_negative = m31_val > HALF_P;
        let val = if is_negative {
            (P - m31_val) as i64
        } else {
            m31_val as i64
        };
        Self(if is_negative { -val } else { val })
    }

    /// Division with remainder for constraints
    pub fn div_rem(self, rhs: Self) -> (Self, Self) {
        assert!(rhs.0 != 0, "Division by zero");

        // Handle division with proper rounding toward negative infinity
        let scaled = self.0 * Self::SCALE_FACTOR;
        let is_negative = (self.0 < 0) != (rhs.0 < 0);
        let self_negative = self.0 < 0;

        let abs_scaled = scaled.abs();
        let abs_rhs = rhs.0.abs();

        let abs_quotient = abs_scaled / abs_rhs;
        let abs_remainder = abs_scaled % abs_rhs;

        let sign_q = (-(is_negative as i64)) | 1; // -1 if negative, 1 if positive
        let sign_r = (-(self_negative as i64)) | 1; // -1 if negative, 1 if positive

        (Self(abs_quotient * sign_q), Self(abs_remainder * sign_r))
    }
}

impl Add for IntRep {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for IntRep {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

/// Multiply with remainder for constraints
impl Mul for IntRep {
    type Output = (Self, Self);
    fn mul(self, rhs: Self) -> Self::Output {
        let product = self.0 * rhs.0;
        let quotient = product >> DEFAULT_SCALE;
        let remainder = product & ((1 << DEFAULT_SCALE) - 1);
        (Self(quotient), Self(remainder))
    }
}

impl Div for IntRep {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl Zero for IntRep {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
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
        let a = IntRep::from_f64(-3.5);
        let b = IntRep::from_f64(2.0);

        assert_near(a.to_f64(), -3.5);
        assert_near((a.clone() + b.clone()).to_f64(), -1.5);
        assert_near((a - b).to_f64(), -5.5);
    }

    #[test]
    fn test_add() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = IntRep::from_f64(a);
            let fb = IntRep::from_f64(b);

            assert_near((fa + fb).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = IntRep::from_f64(a);
            let fb = IntRep::from_f64(b);

            assert_near((fa - fb).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = IntRep::from_f64(a);
            let fb = IntRep::from_f64(b);

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

            let fa = IntRep::from_f64(a);
            let fb = IntRep::from_f64(b);

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
            let fa = IntRep::from_f64(a);
            let fb = IntRep::from_f64(b);
            let result = (fa / fb).to_f64();
            assert_near(result, expected);
        }
    }
}
