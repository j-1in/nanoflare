#![allow(dead_code, unused)]

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use super::Backend;
use crate::dtype::{DType, FloatDType};
use crate::layout::TensorLayout;
use crate::storage::CudaStorage;
use crate::tensor::Tensor;
use crate::{Error, Result, TensorShape};

#[derive(Debug, Clone)]
pub struct CudaBackend;

impl<T: DType> Backend<T> for CudaBackend {
    type Storage = Arc<CudaStorage<T>>;

    fn store_zeros(&self, layout: &TensorLayout) -> Self::Storage {
        unimplemented!()
    }

    fn store_ones(&self, layout: &TensorLayout) -> Self::Storage {
        unimplemented!()
    }

    fn from_vec(&self, data: Vec<T>) -> Self::Storage {
        unimplemented!()
    }

    fn cast<U>(&self, a: &Tensor<T, Self>) -> Result<Tensor<U, Self>>
    where
        U: DType,
        Self: Backend<U>,
        T: num_traits::ToPrimitive,
        U: num_traits::NumCast,
    {
        Err(Error::UnsupportedOperation { op: "cast", backend: "cuda" })
    }

    fn neg(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        Err(Error::UnsupportedOperation { op: "neg", backend: "cuda" })
    }

    fn exp(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>>
    where
        T: FloatDType,
    {
        Err(Error::UnsupportedOperation { op: "exp", backend: "cuda" })
    }

    fn log(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>>
    where
        T: FloatDType,
    {
        Err(Error::UnsupportedOperation { op: "log", backend: "cuda" })
    }

    fn add(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        Err(Error::UnsupportedOperation { op: "add", backend: "cuda" })
    }

    fn sub(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        Err(Error::UnsupportedOperation { op: "sub", backend: "cuda" })
    }

    fn mul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        Err(Error::UnsupportedOperation { op: "mul", backend: "cuda" })
    }

    fn div(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        Err(Error::UnsupportedOperation { op: "div", backend: "cuda" })
    }

    fn matmul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        Err(Error::UnsupportedOperation { op: "matmul", backend: "cuda" })
    }

    fn sum_dim(
        &self,
        a: &Tensor<T, Self>,
        dim: impl IntoIterator<Item = usize>,
        keepdim: bool,
    ) -> Result<Tensor<T, Self>> {
        // Check there are no duplicate dimensions and they are within bounds
        let shape = a.layout().shape();
        let mut dims = dim.into_iter().collect::<Vec<_>>();
        dims.sort_unstable();
        let mut dims_set = std::collections::HashSet::new();
        for &d in &dims {
            if d >= shape.len() {
                return Err(Error::AxisOutOfBounds { axis: d, rank: shape.len() });
            }
            if !dims_set.insert(d) {
                return Err(Error::DuplicateAxis { axis: d });
            }
        }

        Err(Error::UnsupportedOperation { op: "sum_dim", backend: "cuda" })
    }
}
