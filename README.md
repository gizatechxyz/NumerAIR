# NumerAir

A fixed-point arithmetic library providing constrained fixed-point operations for [Stwo](https://github.com/starkware-libs/stwo.git)-based circuits.
The library implements fixed-point arithmetic using M31 field elements with configurable decimal precision.

## Usage

### Basic Arithmetic

```rust
use numerair::FixedM31;

// Create fixed-point numbers
let a = FixedM31::new(3.14);
let b = FixedM31::new(2.0);

// Basic arithmetic operations
let sum = a + b;
let diff = a - b;
let prod = a * b;
let quot = a / b;
```

### In Circuit Constraints

To use fixed-point operations in your Stwo Prover circuit, use the constraint evaluation functions:

```rust
use numerair::eval::{eval_add, eval_mul, eval_sub};

// In your circuit component's evaluate function:
fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
    let a = eval.next_trace_mask();
    let b = eval.next_trace_mask();

    // Constrain sum.
    let sum = eval.next_trace_mask();
    eval_add(&mut eval, a, b, sum);

    // Constrain mul.
    let prod = eval.next_trace_mask();
    let rem = eval.next_trace_mask();
    eval_mul(&mut eval, a, b, SCALE_FACTOR.into(), prod, rem);

    eval
}
```

## Contributing

Contributions are welcome! Please submit pull requests with:

- New arithmetic operations
- Improved constraints
- Additional tests
- Documentation improvements
- Performance optimizations
