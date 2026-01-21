use std::fmt::Debug;

use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use crate::Result;

pub mod cpu;
pub mod cuda;

pub trait Backend<T: DType>: Debug + Send + Sync + Clone {
    type Storage: Debug + Clone + Send + Sync + TensorStorage<T>;

    // Constructors
    fn store_zeros(&self, layout: &TensorLayout) -> Self::Storage;
    fn store_ones(&self, layout: &TensorLayout) -> Self::Storage;
    fn from_vec(&self, data: Vec<T>) -> Self::Storage;

    // Tensor Operations
    fn add(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>>;
    fn sub(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>>;
    fn mul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>>;
    fn div(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>>;
    fn matmul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>>;
}
