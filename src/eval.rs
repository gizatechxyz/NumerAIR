use num_traits::One;
use stwo_prover::constraint_framework::EvalAtRow;

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
        scale: Self::F,
        quotient: Self::F,
        remainder: Self::F,
    ) {
        let product = self.add_intermediate(a * b);
        self.eval_fixed_div_rem(product, scale, quotient, remainder);
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

    /// Evaluates reciprocal constraints for fixed-point numbers.
    /// Constrains: scale * scale = value * reciprocal + remainder
    fn eval_fixed_recip(
        &mut self,
        value: Self::F,
        scale: Self::F,
        reciprocal: Self::F,
        remainder: Self::F,
    ) {
        println!("value: {:?}", value);
        println!("scale: {:?}", scale);
        println!("reciprocal: {:?}", reciprocal);
        println!("remainder: {:?}", remainder);

        let scale_squared = self.add_intermediate(scale.clone() * scale);
        self.eval_fixed_div_rem(scale_squared, value, reciprocal, remainder);
    }
}

// Blanket implementation for any type that implements EvalAtRow
impl<T: EvalAtRow> EvalFixedPoint for T {}

#[cfg(test)]
mod tests {

    use num_traits::Zero;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use stwo_prover::{
        constraint_framework::{self, preprocessed_columns::IsFirst, FrameworkEval},
        core::{
            backend::{simd::SimdBackend, Col, Column},
            fields::{
                m31::{BaseField, P},
                qm31::SecureField,
            },
            pcs::TreeVec,
            poly::{
                circle::{CanonicCoset, CircleEvaluation},
                BitReversedOrder,
            },
        },
    };

    use crate::{Fixed, SCALE_FACTOR};

    use super::*;

    struct TestEval {
        log_size: u32,
        op: Op,
    }

    #[derive(Clone, Copy)]
    enum Op {
        Add,
        Sub,
        Mul,
        Recip,
    }

    impl FrameworkEval for TestEval {
        fn log_size(&self) -> u32 {
            self.log_size
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.log_size + 1
        }

        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
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
                    eval.eval_fixed_mul(lhs, rhs, SCALE_FACTOR.into(), out, rem)
                }
                Op::Recip => {
                    let input = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    let rem = eval.next_trace_mask();
                    eval.eval_fixed_recip(input, SCALE_FACTOR.into(), out, rem)
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

    fn test_op(op: Op, inputs: Vec<Fixed>, expected_outputs: Vec<Fixed>) {
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
        };

        // Test valid trace
        constraint_framework::assert_constraints(
            &trace_polys,
            domain,
            |eval| {
                component.evaluate(eval);
            },
            SecureField::zero(),
        );

        // Test invalid trace - modify the output column
        let mut invalid_trace_cols = trace_cols;
        if let Some(last_col) = invalid_trace_cols.last_mut() {
            for val in last_col.iter_mut() {
                val.0 = (val.0 + 1) % P;
            }
        }

        let invalid_trace_evals = columns_to_evaluations(invalid_trace_cols, domain);
        let is_first = IsFirst::new(LOG_SIZE).gen_column_simd();
        let invalid_trace = TreeVec::new(vec![vec![is_first], invalid_trace_evals, Vec::new()]);

        let invalid_trace_polys = invalid_trace.map_cols(|c| c.interpolate());

        // This should panic for invalid trace
        let result = std::panic::catch_unwind(|| {
            constraint_framework::assert_constraints(
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
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);

            test_op(Op::Add, vec![a, b], vec![a + b]);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            test_op(Op::Sub, vec![a, b], vec![a - b]);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular multiplication cases
        for _ in 0..100 {
            let a = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let (expected, rem) = a * b;

            test_op(Op::Mul, vec![a, b], vec![expected, rem]);
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
            let fixed_a = Fixed::from_f64(a);
            let fixed_b = Fixed::from_f64(b);
            let (expected, rem) = fixed_a * fixed_b;

            test_op(Op::Mul, vec![fixed_a, fixed_b], vec![expected, rem]);
        }
    }

    #[test]
    fn test_recip() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular multiplication cases
        for _ in 0..100 {
            let input = Fixed::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let (expected, rem) = input.recip();

            test_op(Op::Recip, vec![input], vec![expected, rem]);
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
            let fixed_input = Fixed::from_f64(input);
            let (expected, rem) = fixed_input.recip();

            test_op(Op::Recip, vec![fixed_input], vec![expected, rem]);
        }
    }
}
