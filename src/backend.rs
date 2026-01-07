use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use std::fmt::Debug;

pub mod cpu;
pub mod cuda;

pub trait Backend<T: DType>: Debug + Send + Sync {
    fn alloc_storage(&self, layout: &TensorLayout) -> TensorStorage<T>;
    fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn sub(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn div(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;
    fn matmul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>;

    fn validate_same_layout(&self, a: &Tensor<T>, b: &Tensor<T>) {
        assert!(
            a.layout() == b.layout(),
            "Layouts must be the same for addition"
        );
    }
}
