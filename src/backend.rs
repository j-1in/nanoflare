use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use crate::tensor::{DType, Tensor};
use std::fmt::Debug;

pub mod cpu;
pub mod cuda;

pub trait Backend: Debug + Send + Sync {
    fn alloc_storage(&self, layout: &TensorLayout, dtype: DType) -> TensorStorage;
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn div(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;

    fn validate_same_layout(&self, a: &Tensor, b: &Tensor) {
        assert!(
            a.layout() == b.layout(),
            "Layouts must be the same for addition"
        );
    }
}
