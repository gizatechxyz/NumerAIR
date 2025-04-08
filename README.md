# NumerAir

A fixed-point arithmetic library providing constrained fixed-point operations for [Stwo](https://github.com/starkware-libs/stwo.git)-based circuits.
The library implements fixed-point arithmetic using M31 field elements with configurable decimal precision.

## Usage

### Basic Arithmetic

```rust

// Create fixed-point numbers
let lhs = Fixed::from_f64(3.14);
let rhs = Fixed::from_f64(2.0);

// Basic arithmetic operations
let sum = lhs + rhs;
let diff = lhs - rhs;
let (prod, rem) = lhs * rhs;
```

### In Circuit Constraints

To use fixed-point operations in your Stwo Prover circuit, use the constraint evaluation functions:

```rust
use numerair::eval::{eval_add, eval_mul, eval_sub};

// In your circuit component's evaluate function:
fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
    let lhs = eval.next_trace_mask();
    let rhs = eval.next_trace_mask();
    let rem = eval.next_trace_mask();
    let res = eval.next_trace_mask();

    // Constrain mul.
    eval.eval_fixed_mul(lhs, rhs, SCALE_FACTOR.into(), res, rem)

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

## Contributors

 <!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
 <table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/raphaelDkhn"><img src="https://avatars.githubusercontent.com/u/113879115?v=4?s=100" width="100px;" alt="raphaelDkhn"/><br /><sub><b>raphaelDkhn</b></sub></a><br /><a href="https://github.com/gizatechxyz/NumerAIR/commits?author=raphaelDkhn" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/blewater"><img src="https://avatars.githubusercontent.com/u/2580304?v=4?s=100" width="100px;" alt="Mario Karagiorgas"/><br /><sub><b>Mario Karagiorgas</b></sub></a><br /><a href="https://github.com/gizatechxyz/NumerAIR/commits?author=blewater" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
