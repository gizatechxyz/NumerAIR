# NumerAir

A fixed-point arithmetic library providing constrained fixed-point operations for [Stwo](https://github.com/starkware-libs/stwo.git)-based circuits.
The library implements fixed-point arithmetic using M31 field elements with configurable decimal precision.

## Features

- Parameterized scale for flexible precision
- Fixed-point operations (addition, subtraction, multiplication, reciprocal, square root)
- Circuit constraints for all operations
- Type-safe scale implementation

## Usage

### Basic Arithmetic

```rust
// Using default scale (15 bits of precision)
let a = Fixed::<DefaultScale>::from_f64(3.14);
let b = Fixed::<DefaultScale>::from_f64(2.0);

// Basic arithmetic operations
let sum = a + b;
let diff = a - b;
let (prod, rem) = a * b;

// Using custom scale
let high_precision = Fixed::<Scale24>::from_f64(0.12345678);
let low_precision = Fixed::<Scale8>::from_f64(42.5);

// Convert between scales
let converted = high_precision.convert_to::<Scale8>();
```

### Custom Scales

You can define custom precision scales by implementing the `FixedScale` trait:

```rust
// Define a custom 12-bit scale
#[derive(Copy, Clone, Debug)]
pub struct Scale12;
impl FixedScale for Scale12 {
    const SCALE: u32 = 12;
}

// Use your custom scale
let value = Fixed::<Scale12>::from_f64(123.456);
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
    
    // Get scale factor for DefaultScale
    let scale_factor = E::F::from(M31::from_u32_unchecked(1 << DefaultScale::SCALE));

    // Constrain mul using EvalFixedPoint trait
    eval.eval_fixed_mul(lhs, rhs, scale_factor, res, rem);

    eval
}
```

## How It Works

The fixed-point representation uses a scale parameter to determine the number of bits used for the fractional part:

- `SCALE`: Number of bits used for decimal precision
- `SCALE_FACTOR`: The value 2^SCALE, represents 1.0 in fixed-point
- `HALF_SCALE_FACTOR`: The value 2^(SCALE-1), used for rounding

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
