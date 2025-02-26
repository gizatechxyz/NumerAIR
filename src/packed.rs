use once_cell::sync::Lazy;
use std::ops::{Add, Mul, Sub};
use std::simd::num::{SimdInt, SimdUint};
use std::simd::Mask;
use std::simd::{cmp::SimdPartialOrd, Simd};
use stwo_prover::core::backend::simd::m31::{PackedBaseField, PackedM31, N_LANES};
use stwo_prover::core::fields::m31::P;

use crate::SCALE_FACTOR_U32;
use crate::{base::FixedM31, HALF_P};

static HALF_P_SIMD: Lazy<Simd<u32, N_LANES>> = Lazy::new(|| Simd::splat(HALF_P));
static SCALE_FACTOR_SIMD: Lazy<Simd<i64, N_LANES>> =
    Lazy::new(|| Simd::splat(SCALE_FACTOR_U32 as i64));

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct FixedPackedM31(pub PackedM31);
pub type FixedPackedBaseField = FixedPackedM31;

impl FixedPackedM31 {
    /// Creates a new instance with all elements set to the given scaled value.
    pub fn broadcast_scaled(FixedM31(value): FixedM31) -> Self {
        Self(PackedM31(Simd::splat(value.0)))
    }

    /// Creates a new instance with all elements set to the given unscaled value.
    pub fn broadcast_from_f64(value: f64) -> Self {
        FixedPackedM31::broadcast_scaled(FixedM31::from_f64(value))
    }

    /// Converts a fixed-size array of f64 values to a fixed-point SIMD representation.
    pub fn from_f64_array(values: [f64; N_LANES]) -> Self {
        Self(PackedM31(Simd::from_array(
            values.map(|v| FixedM31::from_f64(v).0 .0),
        )))
    }

    /// Converts the fixed-point SIMD values to a fixed-size array of f64.
    pub fn to_f64_array(self) -> [f64; N_LANES] {
        self.0
            .to_array()
            .map(|v| FixedM31::to_f64(FixedM31::from_m31_scaled(v)))
    }

    /// Returns a SIMD mask indicating which elements are negative.
    pub fn is_negative(self) -> Mask<i32, 16> {
        let self_simd = self.0.into_simd();
        self_simd.simd_gt(*HALF_P_SIMD)
    }

    /// Performs signed division with remainder
    /// Returns (quotient, remainder) such that:
    /// - self = quotient * div + remainder
    /// - 0 <= remainder < |div|
    /// - quotient is rounded toward negative infinity
    pub fn fixed_div_rem(self, _div: Self) -> (Self, Self)
    where
        Self: Sized,
    {
        todo!()
    }
}

impl Add for FixedPackedM31 {
    type Output = Self;

    /// Adds two fixed-point numbers
    ///
    /// The addition is performed directly in the underlying field since:
    /// (a × 2^k) + (b × 2^k) = (a + b) × 2^k
    /// where k is the scaling factor
    fn add(self, rhs: Self) -> Self::Output {
        FixedPackedM31(self.0 + rhs.0)
    }
}

impl Sub for FixedPackedM31 {
    type Output = Self;

    /// Subtracts two fixed-point numbers
    ///
    /// The subtraction is performed directly in the underlying field since:
    /// (a × 2^k) - (b × 2^k) = (a - b) × 2^k
    /// where k is the scaling factor
    fn sub(self, rhs: Self) -> Self::Output {
        FixedPackedM31(self.0 - rhs.0)
    }
}

impl Mul for FixedPackedM31 {
    type Output = (Self, Self);

    /// Multiply taking into account the fixed-point scale factor
    ///
    /// Since both inputs are scaled by 2^k, the product needs to be divided by 2^k:
    /// (a × 2^k) × (b × 2^k) = (a × b) × 2^2k
    /// result = ((a × b) × 2^2k) ÷ 2^k = (a × b) × 2^k
    fn mul(self, rhs: Self) -> Self::Output {
        static P_SIMD: Lazy<Simd<i64, N_LANES>> = Lazy::new(|| Simd::splat(P as i64));
        static HALF_P_SIMD: Lazy<Simd<i64, N_LANES>> = Lazy::new(|| Simd::splat(HALF_P as i64));

        // Cast to i64 for signed arithmetic
        let self_simd: Simd<i64, N_LANES> = self.0.into_simd().cast();
        let rhs_simd: Simd<i64, N_LANES> = rhs.0.into_simd().cast();

        // Convert to signed representation
        let is_self_neg = self_simd.simd_gt(*HALF_P_SIMD);
        let is_rhs_neg = rhs_simd.simd_gt(*HALF_P_SIMD);
        let self_signed = is_self_neg.select(-(*P_SIMD - self_simd), self_simd);
        let rhs_signed = is_rhs_neg.select(-(*P_SIMD - rhs_simd), rhs_simd);

        // Compute product, quotient, and remainder
        let prod = self_signed * rhs_signed;
        let q = prod / *SCALE_FACTOR_SIMD;
        let r = prod % *SCALE_FACTOR_SIMD;

        // Map back to [0, p-1]
        let q_field = (q % *P_SIMD + *P_SIMD) % *P_SIMD;
        let r_field = (r % *P_SIMD + *P_SIMD) % *P_SIMD;

        // Cast back to u32
        unsafe {
            (
                FixedPackedM31(PackedBaseField::from_simd_unchecked(q_field.cast::<u32>())),
                FixedPackedM31(PackedBaseField::from_simd_unchecked(r_field.cast::<u32>())),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    const EPSILON: f64 = 1e-2;

    fn assert_near(a: f64, b: f64) {
        assert!((a - b).abs() < EPSILON, "Expected {} to be near {}", a, b);
    }

    #[test]
    fn test_packed_fixed_point_ops() {
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..50 {
            // Generate test values
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            // Create packed values
            let pa = FixedPackedBaseField::broadcast_from_f64(a);
            let pb = FixedPackedBaseField::broadcast_from_f64(b);

            // Test operations
            assert_near((pa + pb).to_f64_array()[0], a + b);
            assert_near((pa - pb).to_f64_array()[0], a - b);

            let (q, _) = pa * pb;
            assert_near(q.to_f64_array()[0], a * b);
        }
    }
}
