#![allow(dead_code, unused)]

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use super::Backend;
use crate::Result;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::storage::{CudaStorage, TensorStorage};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct CudaBackend;

impl<T: DType> Backend<T> for CudaBackend {
    fn store_zeros(&self, layout: &TensorLayout) -> TensorStorage<T> {
        unimplemented!()
    }

    fn store_ones(&self, layout: &TensorLayout) -> TensorStorage<T> {
        unimplemented!()
    }

    fn from_vec(&self, data: Vec<T>) -> TensorStorage<T> {
        unimplemented!()
    }

    fn add<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        unimplemented!()
    }

    fn sub<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        unimplemented!()
    }

    fn mul<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        unimplemented!()
    }

    fn div<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        unimplemented!()
    }

    fn matmul<B: Backend<T>>(&self, a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        unimplemented!()
    }
}
