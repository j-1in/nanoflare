use std::fmt::Debug;

use crate::Result;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

pub mod cpu;
pub mod cuda;

pub(crate) mod private {
    use super::Backend;
    use crate::Result;
    use crate::dtype::{DType, FloatDType};
    use crate::tensor::Tensor;

    #[doc(hidden)]
    pub trait BackendOps<T: DType, B: Backend<T>> {
        // Tensor Operations
        fn neg(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>
        where
            T: std::ops::Neg<Output = T>;
        fn abs(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn sgn(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn exp(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>
        where
            T: FloatDType;
        fn log(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>
        where
            T: FloatDType;
        fn relu(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>
        where
            T: FloatDType;
        fn sigmoid(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>
        where
            T: FloatDType;
        fn tanh(&self, a: &Tensor<T, B>) -> Result<Tensor<T, B>>
        where
            T: FloatDType;
        fn add(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn sub(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn mul(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn div(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn dot(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>>;
        fn matmul(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>>;
    }
}

pub trait Backend<T: DType>: Debug + Send + Sync + Clone + private::BackendOps<T, Self> {
    type Storage: Debug + Clone + Send + Sync + TensorStorage<T>;

    // Constructors
    fn store_zeros(&self, layout: &TensorLayout) -> Self::Storage;
    fn store_ones(&self, layout: &TensorLayout) -> Self::Storage;
    fn from_vec(&self, data: Vec<T>) -> Self::Storage;

    // Casting
    fn cast<U>(&self, a: &Tensor<T, Self>) -> Result<Tensor<U, Self>>
    where
        U: DType,
        Self: Backend<U>,
        T: num_traits::ToPrimitive,
        U: num_traits::NumCast;

    /// Sum over specified dimensions of a Tensor
    ///
    /// # Arguments
    /// * `a` - The input tensor
    /// * `dim` - The dimensions to sum over
    /// * `keepdim` - Whether to keep the summed dimensions
    ///
    /// # Returns
    /// A new tensor with the specified dimensions summed
    ///
    /// # Errors
    /// Returns an error if the specified dimensions are out of bounds
    fn sum_dim(
        &self,
        a: &Tensor<T, Self>,
        dim: impl IntoIterator<Item = usize>,
        keepdim: bool,
    ) -> Result<Tensor<T, Self>>;

    /// Mean over specified dimensions of a Tensor
    fn mean_dim(
        &self,
        a: &Tensor<T, Self>,
        dim: impl IntoIterator<Item = usize>,
        keepdim: bool,
    ) -> Result<Tensor<T, Self>>;

    /// Max over specified dimensions of a Tensor
    fn max_dim(
        &self,
        a: &Tensor<T, Self>,
        dim: impl IntoIterator<Item = usize>,
        keepdim: bool,
    ) -> Result<Tensor<T, Self>>;
}
