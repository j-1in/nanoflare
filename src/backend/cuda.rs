#![allow(dead_code, unused)]

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use super::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::CudaStorage;
use crate::tensor::Tensor;
use crate::Result;

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

    fn exp(&self, a: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        unimplemented!()
    }

    fn add(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        unimplemented!()
    }

    fn sub(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        unimplemented!()
    }

    fn mul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        unimplemented!()
    }

    fn div(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        unimplemented!()
    }

    fn matmul(&self, a: &Tensor<T, Self>, b: &Tensor<T, Self>) -> Result<Tensor<T, Self>> {
        unimplemented!()
    }
}
