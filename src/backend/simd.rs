use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::{Add, Mul, Sub};
use stwo_prover::core::backend::simd::m31::{PackedBaseField, N_LANES};
use stwo_prover::core::fields::m31::{M31, P};

use crate::{HALF_P, SCALE_FACTOR};

use super::cpu::FixedM31;

/// Fixed-point number implementation using PackedBaseField for SIMD operations
#[derive(Copy, Clone, Debug)]
pub struct PackedFixedM31(pub PackedBaseField);

// Number of values that can be processed in parallel
pub const SIMD_WIDTH: usize = N_LANES;

impl PackedFixedM31 {
    /// Creates a new packed fixed-point number from an array of M31 values
    pub fn from_m31_array(values: [M31; SIMD_WIDTH]) -> Self {
        Self(PackedBaseField::from_array(values))
    }

    /// Creates a new packed fixed-point number broadcasting a single value to all lanes
    pub fn broadcast(value: FixedM31) -> Self {
        Self(PackedBaseField::broadcast(value.0))
    }

    /// Creates a packed fixed-point number from an array of float values
    pub fn from_f64_array(values: [f64; SIMD_WIDTH]) -> Self {
        let m31_values: [M31; SIMD_WIDTH] = std::array::from_fn(|i| FixedM31::new(values[i]).0);
        Self::from_m31_array(m31_values)
    }

    /// Extracts the values as an array of M31 elements
    pub fn to_m31_array(&self) -> [M31; SIMD_WIDTH] {
        self.0.to_array()
    }

    /// Extracts the values as an array of FixedM31
    pub fn to_fixed_array(&self) -> [FixedM31; SIMD_WIDTH] {
        let arr = self.to_m31_array();
        std::array::from_fn(|i| FixedM31(arr[i]))
    }

    /// Returns true if any lane represents a negative number
    pub fn is_negative(&self) -> [bool; SIMD_WIDTH] {
        let arr = self.to_m31_array();
        std::array::from_fn(|i| arr[i].0 > HALF_P)
    }

    /// Computes absolute values for all lanes
    pub fn abs(&self) -> Self {
        let neg_mask = self.is_negative();
        let arr = self.to_m31_array();
        let abs_arr: [M31; SIMD_WIDTH] = std::array::from_fn(|i| {
            if neg_mask[i] {
                M31(P - arr[i].0)
            } else {
                arr[i]
            }
        });
        Self::from_m31_array(abs_arr)
    }

    /// Performs signed division with remainder, handling multiple values in parallel
    pub fn signed_div_rem(&self, divisor: PackedBaseField) -> (Self, Self) {
        let dividend_arr = self.to_m31_array();
        let divisor_arr = divisor.to_array();

        let (quotients, remainders): (Vec<M31>, Vec<M31>) = (0..SIMD_WIDTH)
            .into_par_iter()
            .map(|i| {
                let (q, r) = FixedM31(dividend_arr[i]).signed_div_rem(divisor_arr[i]);
                (q.0, r.0)
            })
            .unzip();

        let quotient_arr: [M31; SIMD_WIDTH] = quotients.try_into().unwrap();
        let remainder_arr: [M31; SIMD_WIDTH] = remainders.try_into().unwrap();

        (
            Self::from_m31_array(quotient_arr),
            Self::from_m31_array(remainder_arr),
        )
    }
}

// Arithmetic implementations remain the same
impl Add for PackedFixedM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for PackedFixedM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for PackedFixedM31 {
    type Output = (Self, Self); // Returns (quotient, remainder)

    fn mul(self, rhs: Self) -> (Self, Self) {
        let prod = self.0 * rhs.0;
        Self(prod).signed_div_rem(PackedBaseField::broadcast(SCALE_FACTOR))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_packed_add() {
        let values1 = [1.5, 2.5, -3.5, 4.5];
        let mut full_values1 = [0.0; SIMD_WIDTH];
        full_values1[..4].copy_from_slice(&values1);

        let values2 = [2.0, -1.0, 1.5, -2.5];
        let mut full_values2 = [0.0; SIMD_WIDTH];
        full_values2[..4].copy_from_slice(&values2);

        let expected = [3.5, 1.5, -2.0, 2.0];

        let packed1 = PackedFixedM31::from_f64_array(full_values1);
        let packed2 = PackedFixedM31::from_f64_array(full_values2);
        let result = packed1 + packed2;

        let result_arr = result.to_fixed_array();
        for i in 0..4 {
            assert_near(result_arr[i].to_f64(), expected[i]);
        }
    }

    #[test]
    fn test_packed_mul() {
        let values1 = [1.5, 2.0, -2.5, 3.0];
        let mut full_values1 = [0.0; SIMD_WIDTH];
        full_values1[..4].copy_from_slice(&values1);

        let values2 = [2.0, -1.0, 2.0, -1.5];
        let mut full_values2 = [0.0; SIMD_WIDTH];
        full_values2[..4].copy_from_slice(&values2);

        let expected = [3.0, -2.0, -5.0, -4.5];

        let packed1 = PackedFixedM31::from_f64_array(full_values1);
        let packed2 = PackedFixedM31::from_f64_array(full_values2);
        let (result, _) = packed1 * packed2;

        let result_arr = result.to_fixed_array();
        for i in 0..4 {
            assert_near(result_arr[i].to_f64(), expected[i]);
        }
    }

    fn assert_near(a: f64, b: f64) {
        const EPSILON: f64 = 1e-2;
        assert!((a - b).abs() < EPSILON, "Expected {} to be near {}", a, b);
    }
}
