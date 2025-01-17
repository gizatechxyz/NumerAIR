use std::simd::{cmp::SimdPartialOrd, Simd};

use num_traits::Zero;
use stwo_prover::core::{
    backend::simd::{
        conversion::Unpack,
        m31::{PackedBaseField, MODULUS},
    },
    fields::m31::BaseField,
};

use crate::{FixedPoint, HALF_P, SCALE_FACTOR};

/// Trait for implementing fixed-point arithmetic for packed base field elements
impl FixedPoint for PackedBaseField {
    fn from_f64(x: f64) -> Self {
        Self::broadcast(BaseField::from_f64(x))
    }

    fn to_f64(&self) -> f64 {
        // This assumes the PackedBaseField has been broadcasted with the same value
        self.unpack()[0].to_f64()
    }

    fn is_negative(&self) -> bool {
        self.into_simd().simd_gt(Simd::splat(HALF_P)).any()
    }

    fn abs(&self) -> Self {
        // Check which lanes are negative (> HALF_P)
        let self_simd = self.into_simd();
        let is_negative = self_simd.simd_gt(Simd::splat(HALF_P));

        // For negative values: P - value
        // For positive values: value
        let neg_values = MODULUS - self_simd;
        unsafe { Self::from_simd_unchecked(is_negative.select(neg_values, self_simd)) }
    }

    fn fixed_add(&self, rhs: Self) -> Self {
        *self + rhs
    }

    fn fixed_sub(&self, rhs: Self) -> Self {
        *self - rhs
    }

    fn fixed_mul_rem(&self, rhs: Self) -> (Self, Self)
    where
        Self: Sized,
    {
        let prod = *self * rhs;

        prod.fixed_div_rem(PackedBaseField::broadcast(SCALE_FACTOR))
    }

    fn fixed_div_rem(&self, div: Self) -> (Self, Self) {
        assert!(!div.is_zero(), "Division by zero");

        let self_simd = self.into_simd();
        let div_simd = div.into_simd();
        let half_p = Simd::splat(HALF_P);

        let is_self_neg = self_simd.simd_gt(half_p);
        let is_div_neg = div_simd.simd_gt(half_p);

        let abs_value = is_self_neg.select(MODULUS - self_simd, self_simd);
        let abs_divisor = is_div_neg.select(MODULUS - div_simd, div_simd);

        let q = abs_value / abs_divisor;
        let r = abs_value - q * abs_divisor;

        let q_out = is_self_neg.select(q + Simd::splat(1), q);
        let r_out = is_self_neg.select(abs_divisor - r, r);

        let final_q = (is_self_neg ^ is_div_neg).select(MODULUS - q_out, q_out);

        unsafe {
            (
                Self::from_simd_unchecked(final_q),
                Self::from_simd_unchecked(r_out),
            )
        }
    }

    fn fixed_div(&self, rhs: Self) -> Self {
        assert!(!rhs.is_zero(), "Division by zero");

        // Scale numerator to maintain precision
        let scaled = *self * PackedBaseField::broadcast(SCALE_FACTOR);

        let abs_scaled = scaled.abs();
        let abs_rhs = rhs.abs();
        let scaled_is_neg = scaled.into_simd().simd_gt(Simd::splat(HALF_P));
        let rhs_is_neg = rhs.into_simd().simd_gt(Simd::splat(HALF_P));

        // Unsigned division on absolute values
        let abs_result = abs_scaled.fixed_div_rem(abs_rhs).0;

        // Apply sign based on input signs
        let final_result = (scaled_is_neg ^ rhs_is_neg)
            .select(MODULUS - abs_result.into_simd(), abs_result.into_simd());

        unsafe { PackedBaseField::from_simd_unchecked(final_result) }
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
            let pa = PackedBaseField::from_f64(a);
            let pb = PackedBaseField::from_f64(b);

            // Test operations
            assert_near(pa.fixed_add(pb).to_f64(), a + b);
            assert_near(pa.fixed_sub(pb).to_f64(), a - b);
            assert_near(pa.fixed_div(pb).to_f64(), a / b);

            let (q, _) = pa.fixed_mul_rem(pb);
            assert_near(q.to_f64(), a * b);
        }
    }
}
