use num_traits::One;
use stwo_prover::constraint_framework::EvalAtRow;

/// Evaluates the constraint for addition of fixed point numbers.
pub fn eval_add<E: EvalAtRow>(eval: &mut E, lhs: E::F, rhs: E::F, out: E::F) {
    eval.add_constraint(out - (lhs + rhs));
}

/// Evaluates the constraint for subtraction of fixed point numbers.
pub fn eval_sub<E: EvalAtRow>(eval: &mut E, lhs: E::F, rhs: E::F, out: E::F) {
    eval.add_constraint(out - (lhs - rhs));
}

/// Evaluates the constraints for fixed point multiplication
pub fn eval_mul<E: EvalAtRow>(
    eval: &mut E,
    lhs: E::F,   // First fixed point number
    rhs: E::F,   // Second fixed point number
    scale: E::F, // The scale factor (2^DEFAULT_SCALE)
    out: E::F,   // The output fixed point number
) {
    // First compute the raw product
    let prod = eval.add_intermediate(lhs * rhs);

    // Then constrain the division by scale factor using signed_div_rem
    // out = prod / scale (quotient)
    // rem = prod % scale (remainder)
    let rem = eval.next_trace_mask();
    eval_signed_div_rem(eval, prod, scale, out, rem);
}

/// Evaluates the constraints for the signed division and remainder.
pub fn eval_signed_div_rem<E: EvalAtRow>(
    eval: &mut E,
    value: E::F, // The value being divided
    div: E::F,   // The divisor
    q: E::F,     // The quotient output
    r: E::F,     // The remainder output
) {
    // Core relationship: value = q * div + r
    eval.add_constraint(value - (q * div.clone() + r.clone()));

    // Constraint that the remainder is less than the divisor
    // We do this by adding a new slack variable s such that r + s = div - 1
    let s = eval.next_trace_mask();
    eval.add_constraint(r + s - (div - E::F::one()));
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
                m31::{M31, P},
                qm31::SecureField,
            },
            pcs::TreeVec,
            poly::{
                circle::{CanonicCoset, CircleEvaluation},
                BitReversedOrder,
            },
        },
    };

    use crate::FixedM31;

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
                Op::Add | Op::Sub => {
                    let lhs = eval.next_trace_mask();
                    let rhs = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    match self.op {
                        Op::Add => eval_add(&mut eval, lhs, rhs, out),
                        Op::Sub => eval_sub(&mut eval, lhs, rhs, out),
                        _ => unreachable!(),
                    }
                }
            }
            eval
        }
    }

    fn columns_to_evaluations(
        cols: Vec<Vec<M31>>,
        domain: CanonicCoset,
    ) -> Vec<CircleEvaluation<CpuBackend, M31, BitReversedOrder>> {
        cols.into_iter()
            .map(|col| {
                let mut trace_col = Col::<CpuBackend, M31>::zeros(1 << domain.log_size());
                for (i, val) in col.iter().enumerate() {
                    trace_col.set(i, *val);
                }
                CircleEvaluation::new(domain.circle_domain(), trace_col)
            })
            .collect()
    }

    fn test_op(op: Op, input_values: Vec<FixedM31>, expected_output: FixedM31) {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE);
        let size = 1 << LOG_SIZE;

        // Convert inputs and outputs to M31
        let inputs: Vec<M31> = input_values.iter().map(|x| x.0).collect();
        let output = expected_output.0;

        // Generate trace
        let mut trace_cols = vec![Vec::new(); inputs.len() + 1];
        for _ in 0..size {
            for (i, input) in inputs.iter().enumerate() {
                trace_cols[i].push(*input);
            }
            println!("Output: {:?}", output);
            trace_cols[inputs.len()].push(output);
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
    fn test_random_add() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = FixedM31::new((rng.gen::<f64>() - 0.5) * 200.0);
            let b = FixedM31::new((rng.gen::<f64>() - 0.5) * 200.0);

            test_op(Op::Add, vec![a, b], a + b);
        }
    }

    #[test]
    fn test_random_sub() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = FixedM31::new((rng.gen::<f64>() - 0.5) * 200.0);
            let b = FixedM31::new((rng.gen::<f64>() - 0.5) * 200.0);
            test_op(Op::Sub, vec![a, b], a - b);
        }
    }
}
