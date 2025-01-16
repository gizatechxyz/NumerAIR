use std::ops::{Add, Div, Mul, Sub};
use stwo_prover::core::fields::m31::{M31, P};

/// Fixed-point number implementation using M31 field elements.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FixedM31(pub M31);

// Number of bits used for decimal precision.
pub const DEFAULT_SCALE: u32 = 12;
// Scale factor = 2^DEFAULT_SCALE, used for fixed-point arithmetic.
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);
pub const SCALE_FACTOR_U32: u32 = 1 << DEFAULT_SCALE;
// Half the prime modulus.
pub const HALF_P: u32 = P / 2;

impl FixedM31 {
    /// Creates a new fixed-point number from a float value.
    /// The value is scaled by 2^DEFAULT_SCALE and rounded to the nearest integer.
    pub fn new(x: f64) -> Self {
        let scaled = (x * (1u64 << DEFAULT_SCALE) as f64).round() as i64;
        let val = if scaled >= 0 {
            (scaled as u64 & (P as u64 - 1)) as u32
        } else {
            P - ((-scaled as u64 & (P as u64 - 1)) as u32)
        };
        FixedM31(M31(val))
    }

    /// Converts the fixed-point number back to float.
    pub fn to_f64(&self) -> f64 {
        let val = if self.0 .0 > HALF_P {
            -((P - self.0 .0) as f64)
        } else {
            self.0 .0 as f64
        };
        val / SCALE_FACTOR_U32 as f64
    }

    /// Returns the absolute value
    pub fn abs(&self) -> Self {
        if self.is_negative() {
            FixedM31(-self.0)
        } else {
            *self
        }
    }

    /// Returns true if this represents a negative number
    /// Numbers greater than HALF_P are interpreted as negative
    pub fn is_negative(&self) -> bool {
        self.0 .0 > HALF_P
    }

    /// Performs signed division with remainder
    /// Returns (quotient, remainder) such that:
    /// - self = quotient * div + remainder
    /// - 0 <= remainder < |div|
    /// - quotient is rounded toward negative infinity
    pub fn signed_div_rem(&self, div: M31) -> (FixedM31, FixedM31) {
        let value = self.0 .0;
        let divisor = div.0;

        let is_negative = value > HALF_P;
        let abs_value = if is_negative { P - value } else { value };

        let q = abs_value / divisor;
        let r = abs_value - q * divisor;

        if r == 0 {
            (
                FixedM31(M31(if is_negative { P - q } else { q })),
                FixedM31(M31(0)),
            )
        } else if is_negative {
            (FixedM31(M31(P - (q + 1))), FixedM31(M31(divisor - r)))
        } else {
            (FixedM31(M31(q)), FixedM31(M31(r)))
        }
    }
}

impl Add for FixedM31 {
    type Output = Self;

    /// Adds two fixed-point numbers
    ///
    /// The addition is performed directly in the underlying field since:
    /// (a × 2^k) + (b × 2^k) = (a + b) × 2^k
    /// where k is the scaling factor
    fn add(self, rhs: Self) -> Self {
        FixedM31(self.0 + rhs.0)
    }
}

impl Sub for FixedM31 {
    type Output = Self;

    /// Subtracts two fixed-point numbers
    ///
    /// The subtraction is performed directly in the underlying field since:
    /// (a × 2^k) - (b × 2^k) = (a - b) × 2^k
    /// where k is the scaling factor
    fn sub(self, rhs: Self) -> Self {
        FixedM31(self.0 - rhs.0)
    }
}

impl Mul for FixedM31 {
    type Output = Self;

    /// Multiplies two fixed-point numbers
    ///
    /// Since both inputs are scaled by 2^k, the product needs to be divided by 2^k:
    /// (a × 2^k) × (b × 2^k) = (a × b) × 2^2k
    /// result = ((a × b) × 2^2k) ÷ 2^k = (a × b) × 2^k
    fn mul(self, rhs: Self) -> Self {
        let prod = self.0 * rhs.0;
        let (res, _) = FixedM31(prod).signed_div_rem(SCALE_FACTOR);
        res
    }
}

impl Div for FixedM31 {
    type Output = Self;

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
    fn div(self, rhs: Self) -> Self {
        assert!(rhs.0 .0 != 0, "Division by zero");

        // Multiply numerator by scale factor to maintain precision.
        let scaled = self.0 * SCALE_FACTOR;

        // Extract absolute values and signs of operands
        let (abs_scaled, scaled_is_neg) = if scaled.0 > HALF_P {
            (M31(P - scaled.0), true)
        } else {
            (scaled, false)
        };

        let (abs_rhs, rhs_is_neg) = if rhs.0 .0 > HALF_P {
            (M31(P - rhs.0 .0), true)
        } else {
            (rhs.0, false)
        };

        // Perform division on absolute values
        let abs_result = FixedM31(abs_scaled).signed_div_rem(abs_rhs).0;

        // Apply sign based on input signs
        if scaled_is_neg ^ rhs_is_neg {
            FixedM31(M31(P - abs_result.0 .0)) // Negative result
        } else {
            abs_result // Positive result
        }
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
        let a = FixedM31::new(-3.5);
        let b = FixedM31::new(2.0);

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

            let fa = FixedM31::new(a);
            let fb = FixedM31::new(b);

            assert_near((fa + fb).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = FixedM31::new(a);
            let fb = FixedM31::new(b);

            assert_near((fa - fb).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = FixedM31::new(a);
            let fb = FixedM31::new(b);

            let result = (fa * fb).to_f64();
            let expected = a * b;

            assert_near(result, expected);
        }
    }

    #[test]
    fn test_div() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..5 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = FixedM31::new(a);
            let fb = FixedM31::new(b);

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
            let fa = FixedM31::new(a);
            let fb = FixedM31::new(b);
            let result = (fa / fb).to_f64();
            assert_near(result, expected);
        }
    }

    #[test]
    fn test_signed_div_rem() {
        // Test positive numbers
        let x = FixedM31(M31(100));
        let div = M31(7);
        let (q, r) = x.signed_div_rem(div);
        assert_eq!(q.0 .0, 14); // 100 ÷ 7 = 14
        assert_eq!(r.0 .0, 2); // 100 = 14 * 7 + 2

        // Test negative numbers
        let x = FixedM31(M31(P - 100)); // Represents -100
        let (q, r) = x.signed_div_rem(div);
        assert_eq!(q.0 .0, P - 15); // -100 ÷ 7 = -15 (represented as P - 15)
        assert_eq!(r.0 .0, 5); // -100 = -15 * 7 + 5

        // Test zero remainder
        let x = FixedM31(M31(21));
        let (q, r) = x.signed_div_rem(div);
        assert_eq!(q.0 .0, 3);
        assert_eq!(r.0 .0, 0);

        // Test negative number with zero remainder
        let x = FixedM31(M31(P - 21)); // Represents -21
        let (q, r) = x.signed_div_rem(div);
        assert_eq!(q.0 .0, P - 3); // Represents -3
        assert_eq!(r.0 .0, 0);
    }
}
