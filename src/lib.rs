#![feature(portable_simd)]

use stwo_prover::core::fields::m31::{M31, P};

pub mod base;
pub mod eval;
pub mod packed;

// Number of bits used for decimal precision.
pub const DEFAULT_SCALE: u32 = 12;
// Scale factor = 2^DEFAULT_SCALE, used for fixed-point arithmetic.
pub const SCALE_FACTOR: M31 = M31::from_u32_unchecked(1 << DEFAULT_SCALE);
pub const SCALE_FACTOR_U32: u32 = 1 << DEFAULT_SCALE;
// Half the prime modulus.
pub const HALF_P: u32 = P / 2;
