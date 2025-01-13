use std::ops::{Add, Mul, Sub};
use stwo_prover::core::fields::m31::{M31, P};

pub mod eval;

/// Fixed point number implementation over M31 field
/// The value is stored as 2^scale * x
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FixedM31(pub M31);

pub const DEFAULT_SCALE: u32 = 12;
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);
pub const SCALE_FACTOR_U32: u32 = 1 << DEFAULT_SCALE;
pub const HALF_P: u32 = P / 2;

impl FixedM31 {
    pub fn new(x: f64) -> Self {
        let scaled = (x * (1u64 << DEFAULT_SCALE) as f64).round() as i64;
        let val = if scaled >= 0 {
            (scaled as u64 & (P as u64 - 1)) as u32
        } else {
            P - ((-scaled as u64 & (P as u64 - 1)) as u32)
        };
        FixedM31(M31(val))
    }

    pub fn to_f64(&self) -> f64 {
        let val = if self.0 .0 > HALF_P {
            -((P - self.0 .0) as f64)
        } else {
            self.0 .0 as f64
        };
        val / SCALE_FACTOR_U32 as f64
    }

    pub fn abs(&self) -> Self {
        if self.is_negative() {
            FixedM31(-self.0)
        } else {
            *self
        }
    }

    pub fn is_negative(&self) -> bool {
        self.0 .0 > HALF_P
    }

    pub fn signed_div_rem(&self, div: M31) -> (FixedM31, FixedM31) {
        let value = self.0 .0;
        let divisor = div.0;

        let is_negative = value > HALF_P;
        let abs_value = if is_negative { P - value } else { value };

        // Use fast division
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

    fn add(self, rhs: Self) -> Self {
        FixedM31(self.0 + rhs.0)
    }
}

impl Sub for FixedM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
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
