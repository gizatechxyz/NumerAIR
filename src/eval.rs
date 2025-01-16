use num_traits::One;
use stwo_prover::constraint_framework::EvalAtRow;

/// Evaluates addition constraints for fixed point numbers.
pub fn eval_fixed_add<E: EvalAtRow>(eval: &mut E, lhs: E::F, rhs: E::F, out: E::F) {
    eval.add_constraint(out - (lhs + rhs));
}

/// Evaluates subtraction constraints for fixed point numbers.
pub fn eval_fixed_sub<E: EvalAtRow>(eval: &mut E, lhs: E::F, rhs: E::F, out: E::F) {
    eval.add_constraint(out - (lhs - rhs));
}

/// Evaluates constraints for fixed-point multiplication
pub fn eval_fixed_mul<E: EvalAtRow>(
    eval: &mut E,
    lhs: E::F,
    rhs: E::F,
    scale: E::F,
    out: E::F,
    rem: E::F,
) {
    let prod = eval.add_intermediate(lhs * rhs);

    // Constrain the division by scale factor
    // out = prod / scale (quotient)
    // rem = prod % scale (remainder)
    eval_fixed_div_rem(eval, prod, scale, out, rem);
}

/// Evaluates constraints for signed division with remainder
/// Constrains: value = q * div + r where
/// - q is the quotient
/// - r is the remainder
/// - 0 <= r < |div|
/// - q is rounded toward negative infinity
fn eval_fixed_div_rem<E: EvalAtRow>(eval: &mut E, value: E::F, div: E::F, q: E::F, r: E::F) {
    // Core relationship: value = q * div + r
    eval.add_constraint(value - (q * div.clone() + r.clone()));

    // Compute an auxiliary variable to constrain the inequality r < div
    let aux = eval.add_intermediate(div.clone() - E::F::one() - r.clone());

    // Constraint that the remainder is less than the divisor
    // r + aux = div - 1
    eval.add_constraint(r + aux - (div - E::F::one()));
}

#[cfg(test)]
mod tests {

    use num_traits::Zero;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use stwo_prover::{
        constraint_framework::{self, preprocessed_columns::gen_is_first, FrameworkEval},
        core::{
            backend::{Col, Column, CpuBackend},
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

    use crate::{base::BaseFixedPoint, SCALE_FACTOR};

    use super::*;

    struct TestEval {
        log_size: u32,
        op: Op,
        total_sum: SecureField,
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
                    eval_fixed_add(&mut eval, lhs, rhs, out)
                }
                Op::Sub => {
                    let lhs = eval.next_trace_mask();
                    let rhs = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    eval_fixed_sub(&mut eval, lhs, rhs, out)
                }
                Op::Mul => {
                    let lhs = eval.next_trace_mask();
                    let rhs = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    let rem = eval.next_trace_mask();
                    eval_fixed_mul(&mut eval, lhs, rhs, SCALE_FACTOR.into(), out, rem)
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

    fn test_op(op: Op, inputs: Vec<BaseField>, expected_outputs: Vec<BaseField>) {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE);
        let size = 1 << LOG_SIZE;

        // Generate trace
        let mut trace_cols = vec![Vec::new(); inputs.len() + expected_outputs.len()];
        for _ in 0..size {
            for (i, input) in inputs.iter().enumerate() {
                trace_cols[i].push(*input);
            }
            for (i, output) in expected_outputs.iter().enumerate() {
                trace_cols[inputs.len() + i].push(*output);
            }
        }

        let trace_evals = columns_to_evaluations(trace_cols.clone(), domain);
        let trace = TreeVec::new(vec![vec![gen_is_first(LOG_SIZE)], trace_evals, Vec::new()]);

        let trace_polys = trace.map_cols(|c| c.interpolate());

        let component = TestEval {
            log_size: LOG_SIZE,
            total_sum: SecureField::zero(),
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

    #[test]
    fn test_add() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = BaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = BaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);

            test_op(Op::Add, vec![a, b], vec![a.fixed_add(b)]);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = BaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            let b = BaseField::from_f64((rng.gen::<f64>() - 0.5) * 200.0);
            test_op(Op::Sub, vec![a, b], vec![a.fixed_sub(b)]);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test regular multiplication cases
        for _ in 0..100 {
            let a = (rng.gen::<f64>() - 0.5) * 10.0;
            let b = (rng.gen::<f64>() - 0.5) * 10.0;

            let fixed_a = BaseField::from_f64(a);
            let fixed_b = BaseField::from_f64(b);
            let (expected, rem) = fixed_a.fixed_mul_rem(fixed_b);

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
            let fixed_a = BaseField::from_f64(a);
            let fixed_b = BaseField::from_f64(b);
            let (expected, rem) = fixed_a.fixed_mul_rem(fixed_b);

            test_op(Op::Mul, vec![fixed_a, fixed_b], vec![expected, rem]);
        }
    }
}
