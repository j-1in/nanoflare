use num_traits::{One, Zero};
use std::fmt::Debug;

pub trait DType: Debug + Copy + Clone + PartialEq + Zero + One {}

macro_rules! dtype_trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t {
        }
    )*)
}
dtype_trait_impl!(DType for u8 u16 u32 u64 i8 i16 i32 i64 f32 f64);
