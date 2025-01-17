use stwo_prover::core::fields::m31::{BaseField, M31, P};

use crate::{FixedPoint, DEFAULT_SCALE, HALF_P, SCALE_FACTOR, SCALE_FACTOR_U32};


impl FixedPoint for BaseField {
    fn from_f64(x: f64) -> Self {
        let scaled = (x * (1u64 << DEFAULT_SCALE) as f64).round() as i64;
        let val = if scaled >= 0 {
            (scaled as u64 & (P as u64 - 1)) as u32
        } else {
            P - ((-scaled as u64 & (P as u64 - 1)) as u32)
        };
        M31(val)
    }

    fn to_f64(&self) -> f64 {
        let val = if self.0 > HALF_P {
            -((P - self.0) as f64)
        } else {
            self.0 as f64
        };
        val / SCALE_FACTOR_U32 as f64
    }

    fn is_negative(&self) -> bool {
        self.0 > HALF_P
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
        prod.fixed_div_rem(SCALE_FACTOR)
    }

    fn fixed_div_rem(&self, div: Self) -> (Self, Self)
    where
        Self: Sized,
    {
        let value = self.0;
        let divisor = div.0;

        let is_negative = value > HALF_P;
        let abs_value = if is_negative { P - value } else { value };

        let q = abs_value / divisor;
        let r = abs_value - q * divisor;

        if r == 0 {
            (M31(if is_negative { P - q } else { q }), M31(0))
        } else if is_negative {
            (M31(P - (q + 1)), M31(divisor - r))
        } else {
            (M31(q), M31(r))
        }
    }

    fn fixed_div(&self, rhs: Self) -> Self {
        assert!(rhs.0 != 0, "Division by zero");

        // Multiply numerator by scale factor to maintain precision.
        let scaled = *self * SCALE_FACTOR;

        // Extract absolute values and signs of operands
        let (abs_scaled, scaled_is_neg) = if scaled.0 > HALF_P {
            (M31(P - scaled.0), true)
        } else {
            (scaled, false)
        };

        let (abs_rhs, rhs_is_neg) = if rhs.0 > HALF_P {
            (M31(P - rhs.0), true)
        } else {
            (rhs, false)
        };

        // Perform division on absolute values
        let abs_result = abs_scaled.fixed_div_rem(abs_rhs).0;

        // Apply sign based on input signs
        if scaled_is_neg ^ rhs_is_neg {
            M31(P - abs_result.0) // Negative result
        } else {
            abs_result // Positive result
        }
    }

    fn abs(&self) -> Self {
        todo!()
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
        let a = BaseField::from_f64(-3.5);
        let b = BaseField::from_f64(2.0);

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

            let fa = BaseField::from_f64(a);
            let fb = BaseField::from_f64(b);

            assert_near((fa.fixed_add(fb)).to_f64(), a + b);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 200.0;
            let b = (rng.gen::<f64>() - 0.5) * 200.0;

            let fa = BaseField::from_f64(a);
            let fb = BaseField::from_f64(b);

            assert_near((fa.fixed_sub(fb)).to_f64(), a - b);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fa = BaseField::from_f64(a);
            let fb = BaseField::from_f64(b);

            let (q, _) = fa.fixed_mul_rem(fb);
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

            let fa = BaseField::from_f64(a);
            let fb = BaseField::from_f64(b);

            let result = (fa.fixed_div(fb)).to_f64();
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
            let fa = BaseField::from_f64(a);
            let fb = BaseField::from_f64(b);
            let result = (fa.fixed_div(fb)).to_f64();
            assert_near(result, expected);
        }
    }
}
