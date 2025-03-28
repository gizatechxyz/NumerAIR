use num_traits::Zero;
use std::ops::{Add, Mul, Sub};
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
        // i128 for larger intermediate values to prevent overflow
        let self_i128 = self.0 as i128;
        // Multiply by SCALE_FACTOR to move to the "squared" domain.
        let t = self_i128 * (1 << DEFAULT_SCALE as i128);
        
        if t > u64::MAX as i128 {
            // For large values, approximate
            let approx_sqrt = (self.0 as f64).sqrt() * (Self::SCALE_FACTOR as f64).sqrt();
            let approx_sqrt_i64 = approx_sqrt as i64;
            return (Self(approx_sqrt_i64), Self(0));
        }
        
        // Compute the integer square root of t.
        let y = int_sqrt(t as u64) as i64;
        
        // use i128
        let squared = (y as i128) * (y as i128);
        let remainder = t - squared;
        
        (Self(y), Self(remainder as i64))
    }
}

/// Returns the floor of the square root of `n`.
pub fn int_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // Initial guess - use a safer initial guess for large values
    let mut x = if n > u32::MAX as u64 {
        (n >> 16).min(u32::MAX as u64)  // Safer initial guess for very large values
    } else {
        n / 2
    };
    
    // Newton's method iteration with overflow protection
    let mut prev_x = x;
    loop {
        if x == 0 {
            return 0; // don't burn CPU cycles
        }
        
        // Compute next value to avoid overflow
        let quotient = n / x;
        let sum = x.saturating_add(quotient); // Use saturating_add to handle potential overflow
        let next_x = sum / 2;
        
        // Check for convergence or oscillation
        if next_x == x || next_x == prev_x {
            return next_x;
        }
        
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
        // For 12-bit fixed-point precision, we expect
        // precision of about 2^-12 ≈ 0.000244
        let mut test_cases = vec![
            -1.0, // Negative input: sqrt should return 0
            0.0,
            1.0,
            4.0,
            9.0,
            10.0,
            16.0,
            25.0,
            81.0,
            100.0,
            0.25,
            0.0625,
            0.01,
            5.0,
            8.0,
            12.0,
            15.0,
            20.0,
            50.0,
            10000.0,
            1000000.0,
            10000000000.0,
            1e-10, // Small value
            1e10, // Large value
            0.001, // rest irrationals
            0.5,
            2.0,
            3.0,
            42.0, // Nod to Douglas Adams
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
}
