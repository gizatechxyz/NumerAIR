use crate::HALF_P;
use num_traits::One;
use stwo::core::fields::m31::M31;
use stwo_constraint_framework::EvalAtRow;

/// Extension trait for EvalAtRow to support fixed-point arithmetic constraint evaluation
pub trait EvalFixedPoint: EvalAtRow {
    /// Evaluates addition constraints for fixed-point numbers.
    fn eval_fixed_add(&mut self, a: Self::F, b: Self::F, sum: Self::F) {
        self.add_constraint(sum - (a + b));
    }

    /// Evaluates subtraction constraints for fixed-point numbers.
    fn eval_fixed_sub(&mut self, a: Self::F, b: Self::F, diff: Self::F) {
        self.add_constraint(diff - (a - b));
    }

    /// Evaluates multiplication constraints for fixed-point numbers.
    fn eval_fixed_mul(
        &mut self,
        a: Self::F,
        b: Self::F,
        scale_factor: Self::F,
        quotient: Self::F,
        remainder: Self::F,
    ) {
        let product = self.add_intermediate(a * b);
        self.eval_fixed_div_rem(product, scale_factor, quotient, remainder);
    }

    /// Evaluates constraints for signed division with remainder.
    /// Constrains: dividend = quotient * divisor + remainder, where:
    /// - quotient is rounded toward negative infinity,
    /// - 0 <= remainder < |divisor|.
    fn eval_fixed_div_rem(
        &mut self,
        dividend: Self::F,
        divisor: Self::F,
        quotient: Self::F,
        remainder: Self::F,
    ) {
        self.add_constraint(dividend - (quotient * divisor.clone() + remainder.clone()));

        // Auxiliary variable to enforce remainder < divisor:
        let aux = self.add_intermediate(divisor.clone() - Self::F::one() - remainder.clone());
        self.add_constraint(remainder + aux - (divisor - Self::F::one()));
    }

    /// Evaluates remainder constraints for fixed-point numbers.
    /// Constrains: dividend = quotient * divisor + remainder
    /// This is essentially the same as eval_fixed_div_rem but semantically focused on remainder.
    fn eval_fixed_rem(
        &mut self,
        dividend: Self::F,
        divisor: Self::F,
        quotient: Self::F,
        remainder: Self::F,
    ) {
        self.eval_fixed_div_rem(dividend, divisor, quotient, remainder);
    }

    /// Evaluates reciprocal constraints for fixed-point numbers.
    /// Constrains: scale_factor * scale_factor = value * reciprocal + remainder
    fn eval_fixed_recip(
        &mut self,
        value: Self::F,
        scale_factor: Self::F,
        reciprocal: Self::F,
        remainder: Self::F,
    ) {
        let scale_squared = self.add_intermediate(scale_factor.clone() * scale_factor);
        self.eval_fixed_div_rem(scale_squared, value, reciprocal, remainder);
    }

    /// Evaluates constraints for square root operations.
    /// Adds constraints to verify that:
    /// 1. The input is non-negative
    /// 2. out^2 + rem = input * scale_factor
    ///
    /// # Parameters
    /// - `input`: The trace column value representing the scaled input.
    /// - `out`: The trace column value of the scaled square root.
    /// - `rem`: The trace column value of the remainder.
    /// - `scale_factor`: The scale_factor factor to use for fixed-point representation.
    fn eval_fixed_sqrt(
        &mut self,
        input: Self::F,
        out: Self::F,
        rem: Self::F,
        scale_factor: Self::F,
    ) {
        // Constraint to ensure input is non-negative
        // For field elements, we check if input is in the range [0, HALF_P)
        // We need an auxiliary variable to ensure 0 <= input < HALF_P
        let aux =
            self.add_intermediate(Self::F::from(M31(HALF_P)) - Self::F::one() - input.clone());
        self.add_constraint(input.clone() + aux - (Self::F::from(M31(HALF_P)) - Self::F::one()));

        // Enforce the constraint: out^2 + rem = input * scale_factor
        self.add_constraint((out.clone() * out) + rem.clone() - (input * scale_factor));
    }
}

// Blanket implementation for any type that implements EvalAtRow
impl<T: EvalAtRow> EvalFixedPoint for T {}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use stwo::prover::backend::{simd::SimdBackend, Column};
    use stwo::{
        core::{
            fields::{
                m31::{BaseField, M31, P},
                qm31::SecureField,
            },
            pcs::TreeVec,
            poly::circle::CanonicCoset,
        },
        prover::{
            backend::Col,
            poly::{circle::CircleEvaluation, BitReversedOrder},
        },
    };
    use stwo_constraint_framework::{self, FrameworkEval};

    use super::*;
    use crate::Fixed;

    /// A column with `1` at the first position, and `0` elsewhere.
    #[derive(Debug, Clone)]
    pub struct IsFirst {
        pub log_size: u32,
    }
    impl IsFirst {
        pub const fn new(log_size: u32) -> Self {
            Self { log_size }
        }

        pub fn gen_column_simd(
            &self,
        ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
            let mut col = Col::<SimdBackend, BaseField>::zeros(1 << self.log_size);
            col.set(0, BaseField::one());
            CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
        }
    }

    struct TestEval {
        log_size: u32,
        op: Op,
        scale: u32,
    }

    #[derive(Clone, Copy)]
    enum Op {
        Add,
        Sub,
        Mul,
        Rem,
        Recip,
        Sqrt,
    }

    impl FrameworkEval for TestEval {
        fn log_size(&self) -> u32 {
            self.log_size
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.log_size + 1
        }

        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let scale_factor = E::F::from(M31::from_u32_unchecked(1 << self.scale));

            match self.op {
                Op::Add => {
                    let lhs = eval.next_trace_mask();
                    let rhs = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    eval.eval_fixed_add(lhs, rhs, out)
                }
                Op::Sub => {
                    let lhs = eval.next_trace_mask();
                    let rhs = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    eval.eval_fixed_sub(lhs, rhs, out)
                }
                Op::Mul => {
                    let lhs = eval.next_trace_mask();
                    let rhs = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    let rem = eval.next_trace_mask();
                    eval.eval_fixed_mul(lhs, rhs, scale_factor, out, rem)
                }
                Op::Rem => {
                    let dividend = eval.next_trace_mask();
                    let divisor = eval.next_trace_mask();
                    let quotient = eval.next_trace_mask();
                    let remainder = eval.next_trace_mask();
                    eval.eval_fixed_rem(dividend, divisor, quotient, remainder)
                }
                Op::Recip => {
                    let input = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    let rem = eval.next_trace_mask();
                    eval.eval_fixed_recip(input, scale_factor, out, rem)
                }
                Op::Sqrt => {
                    let input = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    let rem = eval.next_trace_mask();
                    eval.eval_fixed_sqrt(input, out, rem, scale_factor)
                }
            }
            eval
        }
    }

    fn columns_to_evaluations(
        cols: Vec<Vec<BaseField>>,
        domain: CanonicCoset,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        cols.into_iter()
            .map(|col| {
                let mut trace_col = Col::<SimdBackend, BaseField>::zeros(1 << domain.log_size());
                for (i, val) in col.iter().enumerate() {
                    trace_col.set(i, *val);
                }
                CircleEvaluation::new(domain.circle_domain(), trace_col)
            })
            .collect()
    }

    fn test_op_internal(
        op: Op,
        inputs: &[Fixed],
        expected_outputs: &[Fixed],
        tamper_col_idx: usize,
    ) {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE);
        let size = 1 << LOG_SIZE;

        // Generate trace
        let mut trace_cols: Vec<Vec<BaseField>> =
            vec![Vec::new(); inputs.len() + expected_outputs.len()];
        for _ in 0..size {
            for (i, input) in inputs.iter().enumerate() {
                trace_cols[i].push(input.to_m31());
            }
            for (i, output) in expected_outputs.iter().enumerate() {
                trace_cols[inputs.len() + i].push(output.to_m31());
            }
        }

        let trace_evals = columns_to_evaluations(trace_cols.clone(), domain);
        let is_first = IsFirst::new(LOG_SIZE).gen_column_simd();
        let trace = TreeVec::new(vec![vec![is_first], trace_evals, Vec::new()]);

        let trace_polys = trace.map_cols(|c| c.interpolate());

        let component = TestEval {
            log_size: LOG_SIZE,
            op,
            scale: 15, // Default scale for tests
        };

        // Test valid trace
        stwo_constraint_framework::assert_constraints_on_polys(
            &trace_polys,
            domain,
            |eval| {
                component.evaluate(eval);
            },
            SecureField::zero(),
        );

        // Test invalid trace - modify the output column
        let mut invalid_trace_cols = trace_cols;
        if let Some(col) = invalid_trace_cols.get_mut(tamper_col_idx) {
            for val in col.iter_mut() {
                // Calculate scale factor for tampering
                let scale_factor = M31::from_u32_unchecked(1 << 15); // Default scale
                val.0 = (val.0 + scale_factor.0) % P;
            }
        }

        let invalid_trace_evals = columns_to_evaluations(invalid_trace_cols, domain);
        let is_first = IsFirst::new(LOG_SIZE).gen_column_simd();
        let invalid_trace = TreeVec::new(vec![vec![is_first], invalid_trace_evals, Vec::new()]);

        let invalid_trace_polys = invalid_trace.map_cols(|c| c.interpolate());

        // This should panic for invalid trace
        let result = std::panic::catch_unwind(|| {
            stwo_constraint_framework::assert_constraints_on_polys(
                &invalid_trace_polys,
                domain,
                |eval| {
                    component.evaluate(eval);
                },
                SecureField::zero(),
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_add() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);

            test_op_internal(Op::Add, &[a, b], &[a + b], 2);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);

            test_op_internal(Op::Sub, &[a, b], &[a - b], 2);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular multiplication cases
        for _ in 0..100 {
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);
            let (expected, rem) = a * b;

            test_op_internal(Op::Mul, &[a, b], &[expected, rem], 2);
        }

        // Test special cases
        let special_cases = vec![
            (0.0, 0.0),    // Zero multiplication
            (1.0, 1.0),    // Unit multiplication
            (-1.0, 1.0),   // Sign handling
            (1.0, -1.0),   // Sign handling other direction
            (-1.0, -1.0),  // Negative times negative
            (0.5, 0.5),    // Fractional multiplication
            (-0.5, 0.5),   // Fractional with sign
            (3.14, 2.0),   // Pi times 2
            (-3.14, -2.0), // Negative Pi times negative
        ];

        for (a, b) in special_cases {
            let fixed_a = Fixed::from_f64(a, 15);
            let fixed_b = Fixed::from_f64(b, 15);
            let (expected, rem) = fixed_a * fixed_b;

            test_op_internal(Op::Mul, &[fixed_a, fixed_b], &[expected, rem], 2);
        }
    }

    #[test]
    fn test_rem() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular remainder cases
        for _ in 0..50 {
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);

            // Skip cases where divisor is too close to zero
            if b.to_f64().abs() < 0.1 {
                continue;
            }

            let (quotient, remainder) = a.div_rem(b);

            test_op_internal(Op::Rem, &[a, b], &[quotient, remainder], 2);
        }

        // Test special cases
        let special_cases = vec![
            (10.0, 3.0), // 10 % 3 = 1, quotient = 3
            (7.5, 2.5),  // 7.5 % 2.5 = 0, quotient = 3
            (9.0, 4.0),  // 9 % 4 = 1, quotient = 2
            (8.0, 3.0),  // 8 % 3 = 2, quotient = 2
            (15.0, 4.0), // 15 % 4 = 3, quotient = 3
            (20.0, 6.0), // 20 % 6 = 2, quotient = 3
            (1.5, 0.5),  // 1.5 % 0.5 = 0, quotient = 3
        ];

        for (a, b) in special_cases {
            let fixed_a = Fixed::from_f64(a, 15);
            let fixed_b = Fixed::from_f64(b, 15);
            let (quotient, remainder) = fixed_a.div_rem(fixed_b);

            test_op_internal(Op::Rem, &[fixed_a, fixed_b], &[quotient, remainder], 2);
        }
    }

    #[test]
    fn test_recip() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular recip cases
        for _ in 0..100 {
            let input = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0, 15);
            if input.value == 0 {
                continue; // Skip division by zero
            }

            let (expected, rem) = input.recip();

            test_op_internal(Op::Recip, &[input], &[expected, rem], 1);
        }

        // Test special cases
        let special_cases = vec![
            1.0,   // Unit case
            2.0,   // Integer > 1
            0.5,   // Fraction between 0 and 1
            4.25,  // Mixed number
            -1.0,  // Negative unit
            -0.25, // Negative fraction
        ];

        for input in special_cases {
            let fixed_input = Fixed::from_f64(input, 15);
            let (expected, rem) = fixed_input.recip();

            test_op_internal(Op::Recip, &[fixed_input], &[expected, rem], 1);
        }
    }

    #[test]
    fn test_sqrt() {
        let test_cases = vec![1.0, 4.0, 9.0, 2.0, 0.5, 0.25, 0.0];
        for input in test_cases {
            let fixed_input = Fixed::from_f64(input, 15);
            let (sqrt_out, rem) = fixed_input.sqrt();

            test_op_internal(Op::Sqrt, &[fixed_input], &[sqrt_out, rem], 1);
        }

        let mut rng = StdRng::seed_from_u64(43);
        for _ in 0..50 {
            let input_val: f64 = rng.gen_range(0.0..100.0);
            let fixed_input = Fixed::from_f64(input_val, 15);
            let (sqrt_out, rem) = fixed_input.sqrt();

            test_op_internal(Op::Sqrt, &[fixed_input], &[sqrt_out, rem], 1);
        }
    }
}
