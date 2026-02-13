use std::fmt::Debug;
use std::sync::Arc;

use super::Backend;
use crate::dtype::{DType, FloatDType};
use crate::layout::{TensorLayout, TensorShape};
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
            let a_slice = self.contiguous_slice::<T>(a);
            let mut data = Vec::with_capacity(a_slice.len());
            for &x in a_slice {
                let casted: Option<U> = <U as num_traits::NumCast>::from(x);
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
            let casted: Option<U> = <U as num_traits::NumCast>::from(a_storage[a_ptr]);
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

    fn neg(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>>
    where
        T: std::ops::Neg<Output = T>,
    {
        if a.layout().is_contiguous() {
            return self.contiguous_unary_op(a, |x| -x);
        }

        self.strided_unary_op(a, |x| -x)
    }

    fn abs(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() {
            return self.contiguous_unary_op(a, |x| x.abs());
        }

        self.strided_unary_op(a, |x| x.abs())
    }

    fn sgn(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        if a.layout().is_contiguous() {
            return self.contiguous_unary_op(a, |x| x.sgn());
        }

        self.strided_unary_op(a, |x| x.sgn())
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

    fn log(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>>
    where
        T: FloatDType,
    {
        if a.layout().is_contiguous() {
            return self.contiguous_unary_op(a, |x| FloatDType::log(x));
        }

        self.strided_unary_op(a, |x| FloatDType::log(x))
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

    fn sum_dim(
        &self,
        a: &Tensor<T, Self>,
        dim: impl IntoIterator<Item = usize>,
        keepdim: bool,
    ) -> Result<Tensor<T, Self>> {
        // Check there are no duplicate dimensions and they are within bounds
        let layout = a.layout();
        let shape = layout.shape();
        let rank = shape.len();
        let mut dims = dim.into_iter().collect::<Vec<_>>();
        dims.sort_unstable();
        let mut dims_set = std::collections::HashSet::new();

        for &d in &dims {
            if d >= rank {
                return Err(Error::AxisOutOfBounds { axis: d, rank });
            }
            if !dims_set.insert(d) {
                return Err(Error::DuplicateAxis { axis: d });
            }
        }

        if dims.len() == 0 {
            return Ok(a.clone());
        }

        let mut reduce_mask = vec![false; rank];
        for &d in &dims {
            reduce_mask[d] = true;
        }

        let out_shape: Vec<_> = shape
            .as_slice()
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                if reduce_mask[i] {
                    if keepdim { Some(1) } else { None }
                } else {
                    Some(s)
                }
            })
            .collect();

        // Fast path: contiguous layout and reduction over a suffix of dimensions
        let dims_is_suffix = {
            let first = dims[0];
            first + dims.len() == rank && dims.iter().enumerate().all(|(i, &d)| d == first + i)
        };

        if layout.is_contiguous() && dims_is_suffix {
            return self.sum_dim_fast_path(&dims, shape, &out_shape, a);
        }

        // General strided reduction
        self.sum_dim_strided(rank, &reduce_mask, layout, shape, &out_shape, a, keepdim)
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

    fn sum_dim_fast_path<T: DType>(
        &self,
        dims: &[usize],
        shape: &TensorShape,
        out_shape: &[usize],
        a: &Tensor<T, Self>,
    ) -> Result<Tensor<T, Self>> {
        let out_layout = TensorLayout::new(out_shape.to_vec());
        let out_numel = out_layout.shape().numel();

        let shape_slice = shape.as_slice();
        let outer = shape_slice[..dims[0]].iter().product::<usize>();
        let inner = shape_slice[dims[0]..].iter().product::<usize>();
        let a_slice = self.contiguous_slice(a);

        let mut out = vec![T::zero(); out_numel];
        for i in 0..outer {
            let base = i * inner;
            let mut acc = T::zero();
            for j in 0..inner {
                acc = acc + a_slice[base + j];
            }
            out[i] = acc;
        }

        let storage = self.from_vec(out);
        return Ok(Tensor::from_parts(storage, out_layout, a.backend().clone()));
    }

    fn sum_dim_strided<T: DType>(
        &self,
        rank: usize,
        reduce_mask: &[bool],
        layout: &TensorLayout,
        shape: &TensorShape,
        out_shape: &[usize],
        a: &Tensor<T, Self>,
        keepdim: bool,
    ) -> Result<Tensor<T, Self>> {
        let out_layout = TensorLayout::new(out_shape.to_vec());
        let out_numel = out_layout.shape().numel();

        let mut out = vec![T::zero(); out_numel];

        let out_strides = out_layout.strides();
        let mut out_stride_for_dim = vec![0; rank];
        let mut out_axis = 0usize;
        for i in 0..rank {
            if reduce_mask[i] {
                out_stride_for_dim[i] = 0;
            } else if keepdim {
                out_stride_for_dim[i] = out_strides[i];
            } else {
                out_stride_for_dim[i] = out_strides[out_axis];
                out_axis += 1;
            }
        }

        let shape_slice = shape.as_slice();
        let numel = shape.numel();
        let a_storage = &a.storage()[..];
        let mut a_ptr = layout.offset();
        let mut out_ptr = 0usize;
        let mut idx = vec![0; rank];

        for _ in 0..numel {
            out[out_ptr] = out[out_ptr] + a_storage[a_ptr];

            for dim in (0..rank).rev() {
                idx[dim] += 1;

                if idx[dim] < shape_slice[dim] {
                    a_ptr += layout.strides()[dim];
                    out_ptr += out_stride_for_dim[dim];
                    break;
                } else {
                    idx[dim] = 0;
                    a_ptr -= layout.strides()[dim] * (shape_slice[dim] - 1);
                    out_ptr -= out_stride_for_dim[dim] * (shape_slice[dim] - 1);
                }
            }
        }

        let storage = self.from_vec(out);
        Ok(Tensor::from_parts(storage, out_layout, a.backend().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
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
    fn neg_contiguous_f32() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![4]);
        let a = Tensor::from_parts(
            backend.from_vec(vec![1.0f32, -2.0, 0.0, 3.5]),
            layout,
            backend.clone(),
        );

        let out = backend.neg(&a).unwrap();
        assert_eq!(out.storage().as_slice(), &[-1.0, 2.0, -0.0, -3.5]);
    }

    #[test]
    fn neg_strided_view_f32() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 4]);
        let a = Tensor::from_parts(
            backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            layout,
            backend.clone(),
        );
        let view = a.skip(1, 2).unwrap(); // [[1, 3], [5, 7]]

        let out = backend.neg(&view).unwrap();
        assert_eq!(out.layout().shape().as_slice(), &[2, 2]);
        assert_eq!(out.storage().as_slice(), &[-1.0, -3.0, -5.0, -7.0]);
    }

    #[test]
    fn abs_signed_and_unsigned() {
        let backend = Arc::new(CpuBackend::new());

        let a_i32 = Tensor::from_parts(
            backend.from_vec(vec![-3i32, 0, 7]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let out_i32 = backend.abs(&a_i32).unwrap();
        assert_eq!(out_i32.storage().as_slice(), &[3, 0, 7]);

        let a_u32 = Tensor::from_parts(
            backend.from_vec(vec![1u32, 0, 7]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let out_u32 = backend.abs(&a_u32).unwrap();
        assert_eq!(out_u32.storage().as_slice(), &[1, 0, 7]);
    }

    #[test]
    fn sgn_signed_and_unsigned() {
        let backend = Arc::new(CpuBackend::new());

        let a_i32 = Tensor::from_parts(
            backend.from_vec(vec![-9i32, 0, 5]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let out_i32 = backend.sgn(&a_i32).unwrap();
        assert_eq!(out_i32.storage().as_slice(), &[-1, 0, 1]);

        let a_u32 = Tensor::from_parts(
            backend.from_vec(vec![0u32, 2, 9]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let out_u32 = backend.sgn(&a_u32).unwrap();
        assert_eq!(out_u32.storage().as_slice(), &[0, 1, 1]);
    }

    #[test]
    fn sgn_strided_view_i32() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::from_parts(
            backend.from_vec(vec![-1i32, 2, 0, -3, 4, -5]),
            TensorLayout::new(vec![2, 3]),
            backend.clone(),
        );
        let view = a.skip(1, 2).unwrap(); // [[-1, 0], [-3, -5]]

        let out = backend.sgn(&view).unwrap();
        assert_eq!(out.layout().shape().as_slice(), &[2, 2]);
        assert_eq!(out.storage().as_slice(), &[-1, 0, -1, -1]);
    }

    #[test]
    fn exp_basic_f32() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<f32, CpuBackend>::ones(layout, backend);
        let out = a.exp().unwrap();
        let expected: f32 = std::f32::consts::E;
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

    #[test]
    fn sum_dim_contiguous_axis1() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 3]);
        let storage = backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = Tensor::from_parts(storage, layout, backend.clone());

        let out = backend.sum_dim(&a, vec![1], false).unwrap();
        assert_eq!(out.layout().shape().as_slice(), &[2]);
        assert_eq!(out.storage().as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn sum_dim_keepdim_axis1() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 3]);
        let storage = backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = Tensor::from_parts(storage, layout, backend.clone());

        let out = backend.sum_dim(&a, vec![1], true).unwrap();
        assert_eq!(out.layout().shape().as_slice(), &[2, 1]);
        assert_eq!(out.storage().as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn sum_dim_multiple_axes_to_scalar() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 3]);
        let storage = backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = Tensor::from_parts(storage, layout, backend.clone());

        let out = backend.sum_dim(&a, vec![0, 1], false).unwrap();
        assert!(out.layout().shape().as_slice().is_empty());
        assert_eq!(out.storage().as_slice(), &[21.0]);
    }

    #[test]
    fn sum_dim_non_contiguous_view() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![4, 4]);
        let storage = backend.from_vec((0..16).map(|v| v as f32).collect());
        let a = Tensor::from_parts(storage, layout, backend.clone());
        let view = a.skip(1, 2).unwrap(); // take columns 0 and 2

        let out = backend.sum_dim(&view, vec![1], false).unwrap();
        assert_eq!(out.layout().shape().as_slice(), &[4]);
        assert_eq!(out.storage().as_slice(), &[2.0, 10.0, 18.0, 26.0]);
    }

    #[test]
    fn sum_dim_duplicate_axis_errors() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<f32, CpuBackend>::ones(layout, backend.clone());

        let err = backend.sum_dim(&a, vec![0, 0], false).unwrap_err();
        match err {
            Error::DuplicateAxis { axis } => assert_eq!(axis, 0),
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn sum_dim_axis_out_of_bounds_errors() {
        let backend = Arc::new(CpuBackend::new());
        let layout = TensorLayout::new(vec![2, 2]);
        let a = Tensor::<f32, CpuBackend>::ones(layout, backend.clone());

        let err = backend.sum_dim(&a, vec![2], false).unwrap_err();
        match err {
            Error::AxisOutOfBounds { axis, rank } => {
                assert_eq!(axis, 2);
                assert_eq!(rank, 2);
            }
            _ => panic!("unexpected error variant"),
        }
    }
}
