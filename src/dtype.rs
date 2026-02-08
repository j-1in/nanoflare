use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use num_traits::{One, Zero, Float};

// Ensure the arithmetic operators return the same concrete type T (Output =
// Self).
pub trait DType:
    Debug
    + Copy
    + Clone
    + PartialEq
    + Zero
    + One
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + 'static
{
}

macro_rules! dtype_trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t {
        }
    )*)
}

dtype_trait_impl!(DType for u8 u16 u32 u64 i8 i16 i32 i64 f32 f64);

pub trait FloatDType: DType + Float {
    fn exp(self) -> Self;
}

impl FloatDType for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
}

impl FloatDType for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
}