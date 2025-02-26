use num_traits::One;
use stwo_prover::constraint_framework::EvalAtRow;

/// Extension trait for EvalAtRow to support fixed-point arithmetic constraint evaluation
pub trait EvalFixedPoint: EvalAtRow {
    /// Evaluates addition constraints for fixed point numbers.
    fn eval_fixed_add(&mut self, lhs: Self::F, rhs: Self::F, out: Self::F) {
        self.add_constraint(out - (lhs + rhs));
    }

    /// Evaluates subtraction constraints for fixed point numbers.
    fn eval_fixed_sub(&mut self, lhs: Self::F, rhs: Self::F, out: Self::F) {
        self.add_constraint(out - (lhs - rhs));
    }

    /// Evaluates multiplication constraints for fixed-point numbers
    fn eval_fixed_mul(
        &mut self,
        lhs: Self::F,
        rhs: Self::F,
        scale: Self::F,
        out: Self::F,
        rem: Self::F,
    ) {
        let prod = self.add_intermediate(lhs * rhs);

        // Constrain the division by scale factor
        // out = prod / scale (quotient)
        // rem = prod % scale (remainder)
        self.eval_fixed_div_rem(prod, scale, out, rem);
    }

    /// Evaluates constraints for signed division with remainder
    /// Constrains: value = q * div + r where
    /// - q is the quotient
    /// - r is the remainder
    /// - 0 <= r < |div|
    /// - q is rounded toward negative infinity
    fn eval_fixed_div_rem(&mut self, value: Self::F, div: Self::F, q: Self::F, r: Self::F) {
        // Core relationship: value = q * div + r
        self.add_constraint(value - (q * div.clone() + r.clone()));

        // Compute an auxiliary variable to constrain the inequality r < div
        let aux = self.add_intermediate(div.clone() - Self::F::one() - r.clone());

        // Constraint that the remainder is less than the divisor
        // r + aux = div - 1
        self.add_constraint(r + aux - (div - Self::F::one()));
    }
}

// Blanket implementation for any type that implements EvalAtRow
impl<T: EvalAtRow> EvalFixedPoint for T {}

#[cfg(test)]
mod tests {

    use num_traits::Zero;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use stwo_prover::{
        constraint_framework::{self, preprocessed_columns::gen_is_first, FrameworkEval},
        core::{
            backend::{simd::SimdBackend, Col, Column, CpuBackend},
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

    use crate::{base::FixedBaseField, packed::FixedPackedBaseField, SCALE_FACTOR};

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
            }
            eval
        }
    }

    fn columns_to_evaluations(
        cols: Vec<Vec<BaseField>>,
        domain: CanonicCoset,
    ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        cols.into_iter()
            .map(|col| {
                let mut trace_col = Col::<CpuBackend, BaseField>::zeros(1 << domain.log_size());
                for (i, val) in col.iter().enumerate() {
                    trace_col.set(i, *val);
                }
                CircleEvaluation::new(domain.circle_domain(), trace_col)
            })
            .collect()
    }

    fn test_op(op: Op, inputs: Vec<FixedBaseField>, expected_outputs: Vec<FixedBaseField>) {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE);
        let size = 1 << LOG_SIZE;

        // Generate trace
        let mut trace_cols = vec![Vec::new(); inputs.len() + expected_outputs.len()];
        for _ in 0..size {
            for (i, input) in inputs.iter().enumerate() {
                trace_cols[i].push(input.0);
            }
            for (i, output) in expected_outputs.iter().enumerate() {
                trace_cols[inputs.len() + i].push(output.0);
            }
        }

        let trace_evals = columns_to_evaluations(trace_cols.clone(), domain);
        let trace = TreeVec::new(vec![vec![gen_is_first(LOG_SIZE)], trace_evals, Vec::new()]);

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
            (SecureField::zero(), None),
        );

        // Test invalid trace - modify the output column
        let mut invalid_trace_cols = trace_cols;
        if let Some(last_col) = invalid_trace_cols.last_mut() {
            for val in last_col.iter_mut() {
                val.0 = (val.0 + 1) % P;
            }
        }

        let invalid_trace_evals = columns_to_evaluations(invalid_trace_cols, domain);
        let invalid_trace = TreeVec::new(vec![
            vec![gen_is_first(LOG_SIZE)],
            invalid_trace_evals,
            Vec::new(),
        ]);

        let invalid_trace_polys = invalid_trace.map_cols(|c| c.interpolate());

        // This should panic for invalid trace
        let result = std::panic::catch_unwind(|| {
            constraint_framework::assert_constraints(
                &invalid_trace_polys,
                domain,
                |eval| {
                    component.evaluate(eval);
                },
                (SecureField::zero(), None),
            );
        });
        assert!(result.is_err());
    }

    fn test_packed_op(
        op: Op,
        inputs: Vec<FixedPackedBaseField>,
        expected_outputs: Vec<FixedPackedBaseField>,
    ) {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE);
        let size = 1 << LOG_SIZE;

        // Generate trace with SimdBackend
        let mut trace_cols =
            vec![Col::<SimdBackend, BaseField>::zeros(size); inputs.len() + expected_outputs.len()];

        // Fill trace with packed values
        for i in 0..size {
            for (col_idx, input) in inputs.iter().enumerate() {
                trace_cols[col_idx].set(i, input.0.to_array()[0]);
            }
            for (col_idx, output) in expected_outputs.iter().enumerate() {
                trace_cols[inputs.len() + col_idx].set(i, output.0.to_array()[0]);
            }
        }

        // Create circle evaluations
        let trace_evals = trace_cols
            .into_iter()
            .map(|col| {
                CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    domain.circle_domain(),
                    col,
                )
            })
            .collect();

        let trace = TreeVec::new(vec![vec![gen_is_first(LOG_SIZE)], trace_evals, Vec::new()]);

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
            (SecureField::zero(), None),
        );

        // Test invalid trace
        let mut invalid_trace_cols =
            vec![Col::<SimdBackend, BaseField>::zeros(size); inputs.len() + expected_outputs.len()];
        for i in 0..size {
            for (col_idx, input) in inputs.iter().enumerate() {
                invalid_trace_cols[col_idx].set(i, input.0.to_array()[0]);
            }
            for (col_idx, output) in expected_outputs.iter().enumerate() {
                let invalid_val = (output.0.to_array()[0].0 + 1) % P;
                invalid_trace_cols[inputs.len() + col_idx]
                    .set(i, BaseField::from_u32_unchecked(invalid_val));
            }
        }

        // Create invalid circle evaluations
        let invalid_trace_evals = invalid_trace_cols
            .into_iter()
            .map(|col| {
                CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    domain.circle_domain(),
                    col,
                )
            })
            .collect();

        let invalid_trace = TreeVec::new(vec![
            vec![gen_is_first(LOG_SIZE)],
            invalid_trace_evals,
            Vec::new(),
        ]);

        // Generate invalid polys
        let invalid_trace_polys = invalid_trace.map_cols(|c| c.interpolate());

        let result = std::panic::catch_unwind(|| {
            constraint_framework::assert_constraints(
                &invalid_trace_polys,
                domain,
                |eval| {
                    component.evaluate(eval);
                },
                (SecureField::zero(), None),
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_add() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = FixedBaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = FixedBaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);

            test_op(Op::Add, vec![a, b], vec![a + b]);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = FixedBaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = FixedBaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            test_op(Op::Sub, vec![a, b], vec![a - b]);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular multiplication cases
        for _ in 0..100 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fixed_a = FixedBaseField::from_f64(a);
            let fixed_b = FixedBaseField::from_f64(b);
            let (expected, rem) = fixed_a * fixed_b;

            test_op(Op::Mul, vec![fixed_a, fixed_b], vec![expected, rem]);
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
            let fixed_a = FixedBaseField::from_f64(a);
            let fixed_b = FixedBaseField::from_f64(b);
            let (expected, rem) = fixed_a * fixed_b;

            test_op(Op::Mul, vec![fixed_a, fixed_b], vec![expected, rem]);
        }
    }

    #[test]
    fn test_packed_fixed_point_eval() {
        // Test ADD
        {
            // Create arrays of N_LANES elements for testing packed operations
            let a = FixedPackedBaseField::broadcast_from_f64(2.5);
            let b = FixedPackedBaseField::broadcast_from_f64(1.5);
            let expected = FixedPackedBaseField::broadcast_from_f64(4.0); // 2.5 + 1.5 = 4.0

            test_packed_op(Op::Add, vec![a, b], vec![expected]);
        }

        // Test SUB
        {
            let a = FixedPackedBaseField::broadcast_from_f64(2.5);
            let b = FixedPackedBaseField::broadcast_from_f64(1.5);
            let expected = FixedPackedBaseField::broadcast_from_f64(1.0); // 2.5 - 1.5 = 1.0

            test_packed_op(Op::Sub, vec![a, b], vec![expected]);
        }

        // Test MUL
        {
            let a = FixedPackedBaseField::broadcast_from_f64(2.5);
            let b = FixedPackedBaseField::broadcast_from_f64(1.5);
            let (expected, rem) = a * b;

            test_packed_op(Op::Mul, vec![a, b], vec![expected, rem]);
        }
    }
}
