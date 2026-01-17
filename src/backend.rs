use std::fmt::Debug;

use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

pub mod cpu;
pub mod cuda;

pub trait Backend<T: DType>: Debug + Send + Sync {
    // Constructors
    fn store_zeros(&self, layout: &TensorLayout) -> TensorStorage<T>;
    fn store_ones(&self, layout: &TensorLayout) -> TensorStorage<T>;
    fn from_vec(&self, data: Vec<T>) -> TensorStorage<T>;

    // Tensor Operations
    fn add<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Tensor<T, B>;
    fn sub<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Tensor<T, B>;
    fn mul<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Tensor<T, B>;
    fn div<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Tensor<T, B>;
    fn matmul<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Tensor<T, B>;

    fn validate_same_layout<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) {
        assert!(a.layout() == b.layout(), "Layouts must be the same.");
    }
}
