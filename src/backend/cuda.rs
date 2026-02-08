#![allow(dead_code, unused)]

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use super::Backend;
use crate::dtype::{DType, FloatDType};
use crate::layout::TensorLayout;
use crate::storage::CudaStorage;
use crate::tensor::Tensor;
use crate::{Error, Result};

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
}
