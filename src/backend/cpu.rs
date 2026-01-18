use std::fmt::Debug;
use std::sync::Arc;

use super::Backend;
use crate::Result;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::{CpuStorage, TensorStorage};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct CpuBackend;

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
    fn add<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.validate_same_layout(a, b);

        let mut res = self.store_zeros(a.layout());
        for i in 0..a.layout().shape().size() {
            let _sum = a.storage()[i] + b.storage()[i];
            res[i] = _sum;
        }

        Ok(Tensor::from_parts(
            res,
            a.layout().clone(),
            a.backend().clone(),
        ))
    }

    fn sub<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn mul<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn div<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.validate_same_layout(a, b);
        todo!()
    }

    fn matmul<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_addition() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<u32, CpuBackend>::zeros(layout.clone(), backend.clone());
        let b = Tensor::zeros(layout.clone(), backend.clone());
        let c = (&a + &b).unwrap();

        assert_eq!(c.layout(), &layout);
    }
}
