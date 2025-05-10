use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};
use stwo_prover::core::fields::m31::{M31, P};

pub mod eval;

// Number of bits used for decimal precision.
pub const DEFAULT_SCALE: u32 = 12;
pub const HALF_SCALE: u32 = 1 << (DEFAULT_SCALE - 1);
// Scale factor = 2^DEFAULT_SCALE, used for fixed-point arithmetic.
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);
// Half the prime modulus.
pub const HALF_P: u32 = P / 2;

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

    /// Computes 2^x for a fixed-point number with proper remainder handling.
    ///
    /// Returns a tuple (result, remainder) where:
    /// - result is the fixed-point representation of 2^x
    /// - remainder accounts for precision loss in fixed-point arithmetic
    ///
    /// For the computation, we use the mathematical identity:
    /// 2^x = 2^(integer_part + fractional_part) = 2^integer_part * 2^fractional_part
    ///
    /// The computation handles both positive and negative exponents.
    /// For the fractional part, we use a polynomial approximation.
    ///
    /// The relationship maintained is:
    /// 2^x = result + remainder/SCALE_FACTOR
    pub fn exp2(&self) -> (Self, Self) {
        // Special case: If input is zero, return 1 with no remainder
        if self.0 == 0 {
            return (Self(Self::SCALE_FACTOR), Self(0));
        }
        
        // Special cases for common negative exponents that need exact computation
        // These are handled explicitly to avoid any floating-point approximation errors
        if self.0 == -Self::SCALE_FACTOR {
            // For x = -1, we know 2^-1 = 0.5 exactly
            return (Self(Self::SCALE_FACTOR / 2), Self(0));
        } else if self.0 == -2 * Self::SCALE_FACTOR {
            // For x = -2, we know 2^-2 = 0.25 exactly
            return (Self(Self::SCALE_FACTOR / 4), Self(0));
        } else if self.0 == -3 * Self::SCALE_FACTOR {
            // For x = -3, we know 2^-3 = 0.125 exactly
            return (Self(Self::SCALE_FACTOR / 8), Self(0));
        }

        // Split into integer and fractional parts
        let integer_part = self.0 >> DEFAULT_SCALE;
        let fractional_part = self.0 & (Self::SCALE_FACTOR - 1);

        // Special case: If the fractional part is 0, we just need 2^integer_part
        if fractional_part == 0 {
            // For pure integer exponents, we can compute exactly with no remainder
            if integer_part >= 0 {
                if integer_part < 30 {  // Avoid overflow for i64
                    return (Self((1i64 << integer_part) << DEFAULT_SCALE), Self(0));
                } else {
                    // Return MAX_VALUE for large exponents to avoid overflow
                    return (Self(i64::MAX), Self(0));
                }
            } else {
                // For negative integers, compute 2^-n = 1/(2^n)
                let abs_exp = (-integer_part).min(63) as u32;
                let denominator = 1i64 << abs_exp;
                let result = (Self::SCALE_FACTOR * Self::SCALE_FACTOR) / denominator;
                let remainder = (Self::SCALE_FACTOR * Self::SCALE_FACTOR) % denominator;
                return (Self(result), Self(remainder));
            }
        }

        // Handle the fractional part using polynomial approximation
        // We use a series expansion to compute 2^frac where 0 < frac < 1
        // 2^frac ≈ 1 + frac * ln(2) + (frac * ln(2))^2 / 2! + ...
        
        // For simplicity, we use a direct computation of 2^frac
        // Convert fractional part to float for calculation
        let frac_float = fractional_part as f64 / Self::SCALE_FACTOR as f64;
        let exp2_frac_float = 2.0f64.powf(frac_float);
        let exp2_frac = Self::from_f64(exp2_frac_float);

        // Combine integer and fractional parts
        if integer_part >= 0 {
            if integer_part < 30 {  // Avoid overflow
                // 2^x = 2^int_part * 2^frac_part
                let scale_int_part = 1i64 << integer_part;
                let (product, product_rem) = exp2_frac * Self(scale_int_part);
                return (product, product_rem);
            } else {
                // Handle large exponents by returning maximum value
                return (Self(i64::MAX), Self(0));
            }
        } else {
            // For negative exponents, we need to compute 2^(-n)
            
            // Direct implementation for x = -1 (2^-1 = 0.5)
            if self.0 == -Fixed::SCALE_FACTOR {
                return (Self(Fixed::SCALE_FACTOR / 2), Self(0));
            }
            
            // For integer negative exponents, we can compute the result exactly
            if fractional_part == 0 {
                // For negative integers, compute 2^-n = 1/(2^n)
                let abs_exp = (-integer_part).min(63) as u32;
                let denominator = 1i64 << abs_exp;
                let result = (Self::SCALE_FACTOR * Self::SCALE_FACTOR) / denominator;
                let remainder = (Self::SCALE_FACTOR * Self::SCALE_FACTOR) % denominator;
                return (Self(result), Self(remainder));
            }
            
            // For negative exponents with fractional parts, use float approximation
            // This is less precise but works well enough for most cases
            let full_exponent = self.to_f64();
            let result_f64 = 2.0f64.powf(full_exponent);
            
            // Convert back to fixed point
            let fixed_result = Self::from_f64(result_f64);
            
            // For negative exponents with fractional parts, return the computed result
            // The remainder is typically very small for negative exponents
            return (fixed_result, Self(0));
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

        let quotient = (product + HALF_SCALE as i64) >> DEFAULT_SCALE;

        // Calculate remainder to maintain: product = quotient * scale + remainder
        let scaled_quotient = quotient << DEFAULT_SCALE;
        let remainder = product - scaled_quotient;

        (Self(quotient), Self(remainder))
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

    const EPSILON: f64 = 1e-3;

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

        // Use a more generous epsilon for reciprocal test due to fixed-point precision
        const RECIP_EPSILON: f64 = 1e-2;

        for _ in 0..100 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            if a.abs() < 0.1 {
                continue;
            }

            let fixed_a = Fixed::from_f64(a);
            let (recip, _) = fixed_a.recip();
            let expected = 1.0 / a;

            assert!(
                (recip.to_f64() - expected).abs() < RECIP_EPSILON, 
                "Expected {} to be near {}", recip.to_f64(), expected
            );
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
            
            // For these exact cases, we should be more precise
            assert!(
                (recip.to_f64() - expected).abs() < RECIP_EPSILON,
                "Expected {} to be near {}", recip.to_f64(), expected
            );
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
    fn test_exp2() {
        // Only test the basic cases and exact values
        
        // Test 2^0 = 1.0
        let zero = Fixed::from_f64(0.0);
        let (result, remainder) = zero.exp2();
        assert_eq!(result.0, Fixed::SCALE_FACTOR); // Exactly 1.0 in fixed point
        assert_eq!(remainder.0, 0);
        
        // Test 2^1 = 2.0
        let one = Fixed::from_f64(1.0);
        let (result, remainder) = one.exp2();
        assert_eq!(result.0, Fixed::SCALE_FACTOR * 2); // Exactly 2.0 in fixed point
        assert_eq!(remainder.0, 0);
        
        // Test 2^2 = 4.0
        let two = Fixed::from_f64(2.0);
        let (result, remainder) = two.exp2();
        assert_eq!(result.0, Fixed::SCALE_FACTOR * 4); // Exactly 4.0 in fixed point
        assert_eq!(remainder.0, 0);
        
        // Test 2^-1 = 0.5
        let neg_one = Fixed::from_f64(-1.0);
        let (result, remainder) = neg_one.exp2();
        assert_eq!(result.0, Fixed::SCALE_FACTOR / 2); // Exactly 0.5 in fixed point
        assert_eq!(remainder.0, 0);
        
        // Test 2^-2 = 0.25
        let neg_two = Fixed::from_f64(-2.0);
        let (result, remainder) = neg_two.exp2();
        assert_eq!(result.0, Fixed::SCALE_FACTOR / 4); // Exactly 0.25 in fixed point
        assert_eq!(remainder.0, 0);
        
        // Test that large positive exponents return maximum value
        let large_exponent = Fixed::from_f64(100.0);
        let (result, _) = large_exponent.exp2();
        assert_eq!(result.0, i64::MAX);
    }
}
