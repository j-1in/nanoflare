use std::fmt::Debug;
use std::sync::Arc;

use super::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::{CpuStorage, Storage as _};
use crate::tensor::Tensor;
use crate::Result;

#[derive(Debug, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }

    fn strided_binary_op<T, F>(
        &self,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        op: F,
    ) -> Result<Tensor<T, Self>>
    where
        T: DType,
        F: Fn(T, T) -> T + Sync + Send,
    {
        let shape = a.layout().shape().as_slice();
        let rank = shape.len();
        let numel = a.layout().shape().numel();

        let mut res = Vec::with_capacity(numel);

        let a_storage = &a.storage()[..];
        let b_storage = &b.storage()[..];
        let mut a_ptr = a.layout().offset();
        let mut b_ptr = b.layout().offset();

        let mut idx = vec![0; rank];

        for _ in 0..numel {
            let val = op(a_storage[a_ptr], b_storage[b_ptr]);
            res.push(val);

            for dim in (0..rank).rev() {
                idx[dim] += 1;

                if idx[dim] < shape[dim] {
                    a_ptr += a.layout().strides()[dim];
                    b_ptr += b.layout().strides()[dim];
                    break;
                } else {
                    idx[dim] = 0;
                    a_ptr -= a.layout().strides()[dim] * (shape[dim] - 1);
                    b_ptr -= b.layout().strides()[dim] * (shape[dim] - 1);
                }
            }
        }

        let res_layout = TensorLayout::new(shape.to_vec());
        Ok(Tensor::from_parts(
            self.from_vec(res),
            res_layout,
            a.backend().clone(),
        ))
    }
}

impl<T: DType> Backend<T> for CpuBackend {
    type Storage = CpuStorage<T>;

    fn store_zeros(&self, layout: &TensorLayout) -> Self::Storage {
        let size = layout.shape().numel();
        CpuStorage::zeros(size)
    }

    fn store_ones(&self, layout: &TensorLayout) -> Self::Storage {
        let size = layout.shape().numel();
        CpuStorage::ones(size)
    }

    fn from_vec(&self, data: Vec<T>) -> Self::Storage {
        CpuStorage::new(data)
    }

    fn add(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() && b.layout().is_contiguous() {
            let data: Vec<T> = a
                .storage()
                .as_slice()
                .iter()
                .zip(b.storage().as_slice().iter())
                .map(|(&x, &y)| x + y)
                .collect();

            return Ok(Tensor::from_parts(
                self.from_vec(data),
                a.layout().clone(),
                a.backend().clone(),
            ));
        }

        self.strided_binary_op(a, b, |x, y| x + y)
    }

    fn sub(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        todo!()
    }

    fn mul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        todo!()
    }

    fn div(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        todo!()
    }

    fn matmul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
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
