use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Rem, Sub};
use stwo::core::fields::m31::{M31, P};

pub mod eval;

// Half the prime modulus.
pub const HALF_P: u32 = P / 2;

/// Integer representation of fixed-point Basefield with runtime scale.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixed {
    pub value: i64,
    pub scale: u32,
}

impl Fixed {
    /// Create a new Fixed with the given value and scale
    pub fn new(value: i64, scale: u32) -> Self {
        Self { value, scale }
    }

    /// Get the scale factor (2^scale)
    pub fn scale_factor(&self) -> i64 {
        1i64 << self.scale
    }

    /// Get half the scale factor for rounding
    pub fn half_scale_factor(&self) -> i64 {
        1i64 << (self.scale - 1)
    }

    /// Create from f64 with specified scale
    pub fn from_f64(value: f64, scale: u32) -> Self {
        let scale_factor = 1i64 << scale;
        Self {
            value: (value * scale_factor as f64).round() as i64,
            scale,
        }
    }

    /// Convert to f64
    pub fn to_f64(self) -> f64 {
        let scale_factor = 1i64 << self.scale;
        self.value as f64 / scale_factor as f64
    }

    /// Convert to M31 for Stwo
    pub fn to_m31(self) -> M31 {
        const MODULUS_BITS: u32 = 31;

        if self.value >= 0 {
            M31::reduce(self.value as u64)
        } else {
            let abs_val = (-self.value) as u64;
            let abs_mod = (((((abs_val >> MODULUS_BITS) + abs_val + 1) >> MODULUS_BITS) + abs_val)
                & (P as u64)) as u32;

            if abs_mod == 0 {
                M31::from_u32_unchecked(0)
            } else {
                M31::from_u32_unchecked(P - abs_mod)
            }
        }
    }

    /// Convert from M31 with specified scale
    pub fn from_m31(value: M31, scale: u32) -> Self {
        let m31_val = value.0;
        let is_negative = m31_val > HALF_P;

        if is_negative {
            Self {
                value: -((P - m31_val) as i64),
                scale,
            }
        } else {
            Self {
                value: m31_val as i64,
                scale,
            }
        }
    }

    /// Computes both quotient and remainder for division
    pub fn div_rem(self, rhs: Self) -> (Self, Self) {
        assert!(rhs.value != 0, "Division by zero");
        assert_eq!(self.scale, rhs.scale, "Scales must match for division");
        
        let quotient = self.value / rhs.value;
        let remainder = self.value % rhs.value;
        
        (
            Self {
                value: quotient,
                scale: self.scale,
            },
            Self {
                value: remainder,
                scale: self.scale,
            },
        )
    }

    /// Computes the reciprocal (1/x) of a fixed-point number
    pub fn recip(self) -> (Self, Self) {
        assert!(self.value != 0, "Division by zero");

        let scale_factor = self.scale_factor();
        let scale_factor_squared = scale_factor * scale_factor;
        let quotient = scale_factor_squared / self.value;
        let remainder = scale_factor_squared % self.value;

        (
            Self {
                value: quotient,
                scale: self.scale,
            },
            Self {
                value: remainder,
                scale: self.scale,
            },
        )
    }

    /// Computes the fixed-point representation of the square root and its remainder
    pub fn sqrt(&self) -> (Self, Self) {
        assert!(self.value >= 0, "Cannot compute square root of negative number");

        if self.value == 0 {
            return (Self::zero(self.scale), Self::zero(self.scale));
        }

        let input_scaled = (self.value as u64) << self.scale;
        let sqrt_val = int_sqrt(input_scaled);
        let remainder = input_scaled - sqrt_val * sqrt_val;

        (
            Self {
                value: sqrt_val as i64,
                scale: self.scale,
            },
            Self {
                value: remainder as i64,
                scale: self.scale,
            },
        )
    }

    /// Convert this Fixed value to a Fixed with a different scale
    pub fn convert_to(self, target_scale: u32) -> Self {
        if target_scale == self.scale {
            self
        } else if target_scale > self.scale {
            let shift = target_scale - self.scale;
            Self {
                value: self.value << shift,
                scale: target_scale,
            }
        } else {
            let shift = self.scale - target_scale;
            Self {
                value: self.value >> shift,
                scale: target_scale,
            }
        }
    }

    /// Create zero with specified scale
    pub fn zero(scale: u32) -> Self {
        Self { value: 0, scale }
    }

    /// Check if the value is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Get the scale of this Fixed
    pub fn scale(&self) -> u32 {
        self.scale
    }

    /// Get the raw value
    pub fn value(&self) -> i64 {
        self.value
    }
}

/// Returns the floor of the square root of `n`
#[inline]
pub fn int_sqrt(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }

    let bits = 64 - n.leading_zeros();
    let mut x = n >> (bits / 2);

    if x == 0 {
        x = 1;
    }

    let mut prev_x = x;
    loop {
        let quotient = n / x;
        let next_x = (x + quotient) / 2;

        if next_x == x || next_x == prev_x {
            return next_x;
        }

        prev_x = x;
        x = next_x;
    }
}

// Implement arithmetic operations
impl Add for Fixed {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.scale, rhs.scale, "Scales must match for addition");
        Self {
            value: self.value + rhs.value,
            scale: self.scale,
        }
    }
}

impl Sub for Fixed {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.scale, rhs.scale, "Scales must match for subtraction");
        Self {
            value: self.value - rhs.value,
            scale: self.scale,
        }
    }
}

impl Mul for Fixed {
    type Output = (Self, Self);

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.scale, rhs.scale, "Scales must match for multiplication");
        
        let product = self.value * rhs.value;
        let half_scale_factor = self.half_scale_factor();
        let quotient = (product + half_scale_factor) >> self.scale;
        let scaled_quotient = quotient << self.scale;
        let remainder = product - scaled_quotient;

        (
            Self {
                value: quotient,
                scale: self.scale,
            },
            Self {
                value: remainder,
                scale: self.scale,
            },
        )
    }
}

impl Rem for Fixed {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        assert!(rhs.value != 0, "Division by zero in remainder operation");
        assert_eq!(self.scale, rhs.scale, "Scales must match for remainder");
        Self {
            value: self.value % rhs.value,
            scale: self.scale,
        }
    }
}

impl Zero for Fixed {
    #[inline]
    fn zero() -> Self {
        Self::zero(12) // Default scale
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

// Add convenience methods for common scales
impl Fixed {
    /// Create with 8-bit scale
    pub fn from_f64_8(value: f64) -> Self {
        Self::from_f64(value, 8)
    }

    /// Create with 12-bit scale (default)
    pub fn from_f64_12(value: f64) -> Self {
        Self::from_f64(value, 12)
    }

    /// Create with 16-bit scale
    pub fn from_f64_16(value: f64) -> Self {
        Self::from_f64(value, 16)
    }

    /// Create with 24-bit scale
    pub fn from_f64_24(value: f64) -> Self {
        Self::from_f64(value, 24)
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
        let a = Fixed::from_f64(-3.5, 15);
        let b = Fixed::from_f64(2.0, 15);

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

            let fa = Fixed::from_f64(a, 15);
            let fb = Fixed::from_f64(b, 15);

            assert_near((fa + fb).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = Fixed::from_f64(a, 15);
            let fb = Fixed::from_f64(b, 15);

            assert_near((fa - fb).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = Fixed::from_f64(a, 15);
            let fb = Fixed::from_f64(b, 15);

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

            let fixed_a = Fixed::from_f64(a, 15);
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
            let fixed_a = Fixed::from_f64(a, 15);
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
            let fixed_input = Fixed::from_f64(input, 15);

            if input < 0.0 {
                let (result, remainder) = fixed_input.sqrt();
                assert_eq!(result.value, 0);
                assert_eq!(remainder.value, 0);
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

            let fa = Fixed::from_f64(a, 15);
            let fb = Fixed::from_f64(b, 15);

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
            let fa = Fixed::from_f64(a, 15);
            let fb = Fixed::from_f64(b, 15);
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

            // Skip cases where divisor is too close to zero
            if b.abs() < 0.1 {
                continue;
            }

            let fa = Fixed::from_f64(a, 15);
            let fb = Fixed::from_f64(b, 15);

            let (quotient, remainder) = fa.div_rem(fb);

            // Verify: dividend = quotient * divisor + remainder
            let reconstructed = quotient.value * fb.value + remainder.value;
            assert_eq!(reconstructed, fa.value);

            // Check individual results
            let expected_quotient = (a / b).trunc();
            let expected_remainder = a % b;

            // The quotient from div_rem is stored as an unscaled integer
            assert_eq!(quotient.value as f64, expected_quotient);
            assert_near(remainder.to_f64(), expected_remainder);
        }
    }

    #[test]
    fn test_basic_operations() {
        let a = Fixed::from_f64(1.5, 12);
        let b = Fixed::from_f64(2.0, 12);

        assert_near(a.to_f64(), 1.5);
        assert_near(b.to_f64(), 2.0);
        assert_near((a + b).to_f64(), 3.5);
        assert_near((a - b).to_f64(), -0.5);
    }

    #[test]
    fn test_multiplication() {
        let a = Fixed::from_f64(2.5, 12);
        let b = Fixed::from_f64(3.0, 12);
        let (result, _) = a * b;
        assert_near(result.to_f64(), 7.5);
    }

    #[test]
    fn test_scale_conversion() {
        let a = Fixed::from_f64(1.5, 12);
        let b = a.convert_to(8);
        assert_near(b.to_f64(), 1.5);
        assert_eq!(b.scale(), 8);
    }

    #[test]
    fn test_different_scales() {
        let a = Fixed::from_f64_8(1.5);
        let b = Fixed::from_f64_12(1.5);
        let c = Fixed::from_f64_16(1.5);
        let d = Fixed::from_f64_24(1.5);

        assert_near(a.to_f64(), 1.5);
        assert_near(b.to_f64(), 1.5);
        assert_near(c.to_f64(), 1.5);
        assert_near(d.to_f64(), 1.5);

        assert_eq!(a.scale(), 8);
        assert_eq!(b.scale(), 12);
        assert_eq!(c.scale(), 16);
        assert_eq!(d.scale(), 24);
    }

    #[test]
    fn test_scale_mismatch() {
        let a = Fixed::from_f64(1.0, 8);
        let b = Fixed::from_f64(2.0, 12);

        // This should panic due to scale mismatch
        let result = std::panic::catch_unwind(|| a + b);
        assert!(result.is_err());
    }
}