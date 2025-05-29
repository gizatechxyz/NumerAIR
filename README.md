# NumerAir

A fixed-point arithmetic library providing constrained fixed-point operations for [Stwo](https://github.com/starkware-libs/stwo.git)-based circuits.
The library implements fixed-point arithmetic using M31 field elements with configurable decimal precision.

## Features

- Type-level scale parameter using Rust's const generics
- Fixed-point operations (addition, subtraction, multiplication, reciprocal, square root)
- Circuit constraints for all operations
- Zero memory overhead for scale information

## Usage

### Basic Arithmetic

```rust
// Using 15 bits of precision
let a = Fixed::<15>::from_f64(3.14);
let b = Fixed::<15>::from_f64(2.0);

// Basic arithmetic operations
let sum = a + b;
let diff = a - b;
let (prod, rem) = a * b;

// Convert between scales
let converted = a.convert_to::<8>();
```

### In Circuit Constraints

To use fixed-point operations in your Stwo Prover circuit, use the constraint evaluation functions:

```rust
use numerair::eval::EvalFixedPoint;

// In your circuit component's evaluate function:
fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
    let lhs = eval.next_trace_mask();
    let rhs = eval.next_trace_mask();
    let rem = eval.next_trace_mask();
    let res = eval.next_trace_mask();
    
    // Get scale factor for the specific scale you're using
    let scale_factor = E::F::from(M31::from_u32_unchecked(1 << 15)); // For Fixed<15>

    // Constrain mul using EvalFixedPoint trait
    eval.eval_fixed_mul(lhs, rhs, scale_factor, res, rem);

    eval
}
```

## How It Works

The fixed-point representation uses a const generic parameter to determine the number of bits used for the fractional part:

- `SCALE`: Type-level constant that defines the number of bits for decimal precision
- `SCALE_FACTOR`: The value 2^SCALE (automatically calculated), represents 1.0 in fixed-point
- `HALF_SCALE_FACTOR`: The value 2^(SCALE-1) (automatically calculated), used for rounding

A value `x` in floating point is represented as `floor(x * 2^SCALE)` in fixed-point.

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
