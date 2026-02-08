use std::fmt::Debug;
use std::sync::Arc;

use super::Backend;
use crate::dtype::{DType, FloatDType};
use crate::layout::TensorLayout;
use crate::storage::CpuStorage;
use crate::tensor::Tensor;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct CpuBackend;

impl<T: DType> Backend<T> for CpuBackend {
    type Storage = Arc<CpuStorage<T>>;

    fn store_zeros(&self, layout: &TensorLayout) -> Self::Storage {
        let size = layout.shape().numel();
        Arc::new(CpuStorage::zeros(size))
    }

    fn store_ones(&self, layout: &TensorLayout) -> Self::Storage {
        let size = layout.shape().numel();
        Arc::new(CpuStorage::ones(size))
    }

    fn from_vec(&self, data: Vec<T>) -> Self::Storage {
        Arc::new(CpuStorage::new(data))
    }

    fn cast<U>(&self, a: &Tensor<T, Self>) -> Result<Tensor<U, Self>>
    where
        U: DType,
        Self: Backend<U>,
        T: num_traits::ToPrimitive,
        U: num_traits::NumCast,
    {
        let from = std::any::type_name::<T>();
        let to = std::any::type_name::<U>();

        if a.layout().is_contiguous() {
            let a_slice = self.contiguous_slice(a);
            let mut data = Vec::with_capacity(a_slice.len());
            for &x in a_slice {
                let casted: Option<U> = num_traits::cast(x);
                let Some(value) = casted else {
                    return Err(Error::DTypeCastFailed { from, to });
                };
                data.push(value);
            }

            let storage = <CpuBackend as Backend<U>>::from_vec(self, data);
            return Ok(Tensor::from_parts(
                storage,
                a.layout().clone(),
                a.backend().clone(),
            ));
        }

        let shape = a.layout().shape().as_slice();
        let rank = shape.len();
        let numel = a.layout().shape().numel();

        let mut res = Vec::with_capacity(numel);
        let a_storage = &a.storage()[..];
        let mut a_ptr = a.layout().offset();
        let mut idx = vec![0; rank];

        for _ in 0..numel {
            let casted: Option<U> = num_traits::cast(a_storage[a_ptr]);
            let Some(value) = casted else {
                return Err(Error::DTypeCastFailed { from, to });
            };
            res.push(value);

            for dim in (0..rank).rev() {
                idx[dim] += 1;

                if idx[dim] < shape[dim] {
                    a_ptr += a.layout().strides()[dim];
                    break;
                } else {
                    idx[dim] = 0;
                    a_ptr -= a.layout().strides()[dim] * (shape[dim] - 1);
                }
            }
        }

        let res_layout = TensorLayout::new(shape.to_vec());
        let storage = <CpuBackend as Backend<U>>::from_vec(self, res);
        Ok(Tensor::from_parts(storage, res_layout, a.backend().clone()))
    }

    fn exp(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>>
    where
        T: FloatDType,
    {
        if a.layout().is_contiguous() {
            return self.contiguous_unary_op(a, |x| FloatDType::exp(x));
        }

        self.strided_unary_op(a, |x| FloatDType::exp(x))
    }

    fn add(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() && b.layout().is_contiguous() {
            return self.contiguous_binary_op(a, b, |x, y| x + y);
        }

        self.strided_binary_op(a, b, |x, y| x + y)
    }

    fn sub(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() && b.layout().is_contiguous() {
            return self.contiguous_binary_op(a, b, |x, y| x - y);
        }

        self.strided_binary_op(a, b, |x, y| x - y)
    }

    fn mul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() && b.layout().is_contiguous() {
            return self.contiguous_binary_op(a, b, |x, y| x * y);
        }

        self.strided_binary_op(a, b, |x, y| x * y)
    }

    fn div(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() && b.layout().is_contiguous() {
            return self.contiguous_binary_op(a, b, |x, y| x / y);
        }

        self.strided_binary_op(a, b, |x, y| x / y)
    }

    fn matmul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        todo!()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }

    /// Get a contiguous slice from a tensor
    fn contiguous_slice<'a, T: DType>(&self, a: &'a Tensor<T, Self>) -> &'a [T] {
        let a_offset = a.layout().offset();
        let numel = a.layout().shape().numel();

        &a.storage().as_slice()[a_offset..a_offset + numel]
    }

    /// Perform unary operation on contiguous tensor
    fn contiguous_unary_op<T, F>(&self, a: &Tensor<T, Self>, op: F) -> Result<Tensor<T, Self>>
    where
        T: DType,
        F: Fn(T) -> T + Sync + Send,
    {
        let a_slice = self.contiguous_slice(a);

        let data: Vec<T> = a_slice.iter().map(|&x| op(x)).collect();

        Ok(Tensor::from_parts(
            self.from_vec(data),
            a.layout().clone(),
            a.backend().clone(),
        ))
    }

    fn strided_unary_op<T, F>(&self, a: &Tensor<T, Self>, op: F) -> Result<Tensor<T, Self>>
    where
        T: DType,
        F: Fn(T) -> T + Sync + Send,
    {
        let shape = a.layout().shape().as_slice();
        let rank = shape.len();
        let numel = a.layout().shape().numel();

        let mut res = Vec::with_capacity(numel);

        let a_storage = &a.storage()[..];
        let mut a_ptr = a.layout().offset();

        let mut idx = vec![0; rank];

        for _ in 0..numel {
            let val = op(a_storage[a_ptr]);
            res.push(val);

            for dim in (0..rank).rev() {
                idx[dim] += 1;

                if idx[dim] < shape[dim] {
                    a_ptr += a.layout().strides()[dim];
                    break;
                } else {
                    idx[dim] = 0;
                    a_ptr -= a.layout().strides()[dim] * (shape[dim] - 1);
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

    /// Perform binary operation on contiguous tensors
    fn contiguous_binary_op<T, F>(
        &self,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        op: F,
    ) -> Result<Tensor<T, Self>>
    where
        T: DType,
        F: Fn(T, T) -> T + Sync + Send,
    {
        let a_slice = self.contiguous_slice(a);
        let b_slice = self.contiguous_slice(b);

        let data: Vec<T> = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();

        Ok(Tensor::from_parts(
            self.from_vec(data),
            a.layout().clone(),
            a.backend().clone(),
        ))
    }

    /// Perform binary operation on strided tensors
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    #[test]
    fn tensor_addition() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<u32, CpuBackend>::zeros(layout.clone(), backend.clone());
        let b = Tensor::zeros(layout.clone(), backend.clone());
        let c = (&a + &b).unwrap();

        assert_eq!(c.layout(), &layout);
    }

    #[test]
    fn exp_basic_f32() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<f32, CpuBackend>::ones(layout, backend);
        let out = a.exp().unwrap();
        let expected = f32::consts::E;
        for &v in out.storage().as_slice() {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn cast_f32_to_i32() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![3]);
        let storage = backend.from_vec(vec![1.0f32, 2.9, -3.1]);
        let a = Tensor::from_parts(storage, layout, backend);
        let b: Tensor<i32, CpuBackend> = a.cast().unwrap();
        assert_eq!(b.storage().as_slice(), &[1, 2, -3]);
    }

    #[test]
    fn cast_out_of_range_errors() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![1]);
        let storage = backend.from_vec(vec![f32::INFINITY]);
        let a = Tensor::from_parts(storage, layout, backend);
        let err = a.cast::<i32>().unwrap_err();
        match err {
            Error::DTypeCastFailed { .. } => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn cast_requires_grad_errors() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2]);
        let tape = Arc::new(Tape::<f32, CpuBackend>::new());
        let a = Tensor::<f32, CpuBackend>::ones(layout, backend).requires_grad(tape);
        let err = a.cast::<i32>().unwrap_err();
        match err {
            Error::RequiresGradUnsupported { op } => {
                assert_eq!(op, "cast");
            }
            _ => panic!("unexpected error variant"),
        }
    }
}
