pub trait DType {}

macro_rules! int_trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t {
        }
    )*)
}
int_trait_impl!(DType for u8 u16 u32 u64 i8 i16 i32 i64 f32 f64);
