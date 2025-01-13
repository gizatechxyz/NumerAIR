use std::ops::{Add, Mul, Sub};
use stwo_prover::core::fields::m31::{M31, P};

pub mod eval;

/// Fixed point number implementation over M31 field
/// The value is stored as 2^scale * x
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FixedM31(pub M31);

pub const DEFAULT_SCALE: u32 = 12;
// Pre-compute scale factor as M31 field element
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);

impl FixedM31 {
    pub fn new(x: f64) -> Self {
        // Scale and round the float value
        let scaled = (x * (1u64 << DEFAULT_SCALE) as f64).round() as i64;

        if scaled >= 0 {
            // For positive numbers, directly convert to field element
            FixedM31(M31::from(scaled as u32))
        } else {
            // For negative numbers, add P to make it a valid M31 value
            FixedM31(M31::from((P as i64 + scaled) as u32))
        }
    }

    pub fn to_f64(&self) -> f64 {
        // Convert back to float by dividing by scale factor
        let val = if self.0 .0 > P / 2 {
            -((P - self.0 .0) as f64)
        } else {
            self.0 .0 as f64
        };
        val / (1u64 << DEFAULT_SCALE) as f64
    }

    pub fn abs(&self) -> Self {
        if self.0 .0 > P / 2 {
            FixedM31(-self.0)
        } else {
            *self
        }
    }

    pub fn is_negative(&self) -> bool {
        self.0 .0 > P / 2
    }

    pub fn signed_div_rem(&self, div: M31) -> (FixedM31, FixedM31) {
        let value = self.0 .0;
        let divisor = div.0;

        // Handle the case when value is positive (< P/2)
        if value <= P / 2 {
            let q = value / divisor;
            let r = value % divisor;
            return (FixedM31(M31(q)), FixedM31(M31(r)));
        }

        // Handle negative values (> P/2)
        // Convert to positive representation first
        let pos_value = P - value;

        // Calculate positive quotient and remainder
        let q = pos_value / divisor;
        let r = pos_value % divisor;

        if r == 0 {
            // If remainder is 0, just negate the quotient
            (FixedM31(M31(P - q)), FixedM31(M31(0)))
        } else {
            // If there's a remainder, adjust quotient and remainder
            // q = -(q + 1)
            // r = divisor - remainder
            (FixedM31(M31(P - (q + 1))), FixedM31(M31(divisor - r)))
        }
    }
}

impl Add for FixedM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Field addition preserves the scale factor
        FixedM31(self.0 + rhs.0)
    }
}

impl Sub for FixedM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // Field subtraction preserves the scale factor
        FixedM31(self.0 - rhs.0)
    }
}

impl Mul for FixedM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prod = self.0 * rhs.0;

        let (res, _) = FixedM31(prod).signed_div_rem(SCALE_FACTOR);

        res
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
    fn test_basic_ops() {
        let a = FixedM31::new(3.5);
        let b = FixedM31::new(2.0);

        assert_near(a.to_f64(), 3.5);
        assert_near(b.to_f64(), 2.0);
        assert_near((a + b).to_f64(), 5.5);
        assert_near((a - b).to_f64(), 1.5);
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
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            // Generate random values between -100 and 100
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = FixedM31::new(a);
            let fb = FixedM31::new(b);

            let result = (fa * fb).to_f64();
            let expected = a * b;

            println!(
                "a: {}, b: {}, result: {}, expected: {}",
                a, b, result, expected
            );
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
