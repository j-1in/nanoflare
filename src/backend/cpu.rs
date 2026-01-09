use super::Backend;
use crate::layout::TensorLayout;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use crate::{dtype::DType, storage::CpuStorage};
use std::{fmt::Debug, sync::Arc};

#[derive(Debug)]
struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl<T: DType> Backend<T> for CpuBackend {
    fn store_zeros(&self, layout: &TensorLayout) -> TensorStorage<T> {
        let size = layout.shape().size();
        let storage = CpuStorage::zeros(size);

        TensorStorage::Cpu(Arc::new(storage))
    }

    fn store_ones(&self, layout: &TensorLayout) -> TensorStorage<T> {
        let size = layout.shape().size();
        let storage = CpuStorage::ones(size);

        TensorStorage::Cpu(Arc::new(storage))
    }

    fn from_vec(&self, data: Vec<T>) -> TensorStorage<T> {
        todo!()
    }

    fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn sub(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn div(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn matmul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_addition() {
        // FIXME
        let backend = Arc::new(crate::backend::cpu::CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<u32>::zeros(layout.clone(), backend.clone());
        let b = Tensor::zeros(layout.clone(), backend.clone());
        let c = &a + &b;

        assert_eq!(c.layout(), &layout);
    }
}
