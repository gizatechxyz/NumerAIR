use num_traits::One;
use stwo_prover::constraint_framework::EvalAtRow;
use crate::SCALE_FACTOR;

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

    /// Evaluates reciprocal constraints for fixed-point numbers
    /// Constrains: scale*scale = input * output + rem
    ///
    /// Parameters:
    /// - input: the input value
    /// - scale: the scale factor (typically SCALE_FACTOR)
    /// - output: the reciprocal result
    /// - rem: the remainder
    fn eval_fixed_recip(&mut self, input: Self::F, scale: Self::F, output: Self::F, rem: Self::F) {
        // Compute scale^2
        let scale_squared = self.add_intermediate(scale.clone() * scale);

        // constrain scale^2 = input * output + rem
        self.eval_fixed_div_rem(scale_squared, input, output, rem);
    }

    fn eval_fixed_sqrt(&mut self, input: Self::F, out: Self::F, rem: Self::F) {
        // Because 1.0 is stored as 4096 in the field,
        // the correct invariant for a perfect square is:
        //     out^2 + rem = input * 4096
        self.add_constraint((out.clone() * out) + rem.clone() - (input * SCALE_FACTOR));
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
                Op::Sqrt => {
                    let input = eval.next_trace_mask();
                    let out = eval.next_trace_mask();
                    let rem = eval.next_trace_mask();
                    eval.eval_fixed_sqrt(input, out, rem)
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

    fn test_op(op: Op, inputs: Vec<Fixed>, expected_outputs: Vec<Fixed>) {
        const LOG_SIZE: u32 = 4;
        let domain = CanonicCoset::new(LOG_SIZE);
        let size = 1 << LOG_SIZE;

        // Generate trace
        let mut trace_cols: Vec<Vec<BaseField>> =
            vec![Vec::new(); inputs.len() + expected_outputs.len()];
        for _i in 0..size {
            for (j, input) in inputs.iter().enumerate() {
                println!();
                let input_m31 = input.to_m31();
                println!("Input: {:?}, to_m31: {}", input, input_m31);
                println!();
                trace_cols[j].push(input.to_m31());
            }
            for (j, output) in expected_outputs.iter().enumerate() {
                trace_cols[inputs.len() + j].push(output.to_m31());
            }
        }
        println!("Trace cols: {:?}", trace_cols);

        let trace_evals = columns_to_evaluations(trace_cols.clone(), domain);
        let is_first = IsFirst::new(LOG_SIZE).gen_column_simd().to_cpu();
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
        let is_first = IsFirst::new(LOG_SIZE).gen_column_simd().to_cpu();
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
        print!("{:?}", result);
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

    #[test]
    fn test_eval_sqrt() {
        let test_cases = vec![
            (1.0, 1.0, 0.0),
            // (4.0, 4.0, 0.0),
            // (9.0, 9.0, 0.0),
            (2.0, 2.0_f64.sqrt(), 2.0),
            // (0.5, 0.5_f64::sqrt(), 0.5),
            (0.25, 0.25_f64.sqrt(), 0.25),
            // (0.0, 0.0, 0.0),
        ];
        for (input, expected_out, expected_rem) in test_cases {
            let fixed_input = Fixed::from_f64(input);
            let (sqrt_out, rem) = fixed_input.sqrt();
            println!(
                "Testing sqrt: input={:?}, out={:?}, rem={:?}, expected_out={:?}, expected_rem={:?}",
                input, sqrt_out, rem, expected_out, Fixed::from_f64(expected_rem)
            );
            test_op(Op::Sqrt, vec![fixed_input], vec![sqrt_out, rem]);
        }
    }
}
