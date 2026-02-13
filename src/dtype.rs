use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use num_traits::{Float, One, Zero};

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
    + Abs
    + Sgn
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

pub trait Abs {
    fn abs(self) -> Self;
}

macro_rules! impl_abs_signed {
    ($($t:ty)*) => ($(
        impl Abs for $t {
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
        }
    )*)
}

impl_abs_signed!(i8 i16 i32 i64 f32 f64);

macro_rules! impl_abs_unsigned {
    ($($t:ty)*) => ($(
        impl Abs for $t {
            fn abs(self) -> Self {
                self
            }
        }
    )*)
}

impl_abs_unsigned!(u8 u16 u32 u64);

pub trait Sgn {
    fn sgn(self) -> Self;
}

macro_rules! impl_sgn_signed {
    ($($t:ty)*) => ($(
        impl Sgn for $t {
            fn sgn(self) -> Self {
                <$t>::signum(self)
            }
        }
    )*)
}

impl_sgn_signed!(i8 i16 i32 i64);

macro_rules! impl_sgn_float {
    ($($t:ty)*) => ($(
        impl Sgn for $t {
            fn sgn(self) -> Self {
                if self == 0.0 {
                    0.0
                } else {
                    <$t>::signum(self)
                }
            }
        }
    )*)
}

impl_sgn_float!(f32 f64);

macro_rules! impl_sgn_unsigned {
    ($($t:ty)*) => ($(
        impl Sgn for $t {
            fn sgn(self) -> Self {
                if self == 0 {
                    0
                } else {
                    1
                }
            }
        }
    )*)
}

impl_sgn_unsigned!(u8 u16 u32 u64);

pub trait FloatDType: DType + Float {
    fn exp(self) -> Self;
    fn log(self) -> Self;
}

impl FloatDType for f32 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.ln()
    }
}

impl FloatDType for f64 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::{Abs, Sgn};

    #[test]
    fn abs_signed_and_unsigned_behaves_as_expected() {
        assert_eq!((-5i32).abs(), 5);
        assert_eq!((5i32).abs(), 5);
        assert_eq!((7u32).abs(), 7);
    }

    #[test]
    fn sgn_signed_and_unsigned_behaves_as_expected() {
        assert_eq!((-9i32).sgn(), -1);
        assert_eq!((0i32).sgn(), 0);
        assert_eq!((4i32).sgn(), 1);

        assert_eq!((0u32).sgn(), 0);
        assert_eq!((9u32).sgn(), 1);

        assert_eq!((-2.5f32).sgn(), -1.0);
        assert_eq!((0.0f32).sgn(), 0.0);
        assert_eq!((3.0f32).sgn(), 1.0);
    }
}
