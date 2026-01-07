use crate::dtype::DType;
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    sync::{Arc, RwLock},
};

#[derive(Debug, Clone)]
pub enum TensorStorage<T: DType> {
    Cpu(Arc<RwLock<CpuStorage<T>>>),
    Cuda(Arc<RwLock<CudaStorage<T>>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuStorage<T: DType>(Vec<T>);

impl<T: DType> CpuStorage<T> {
    pub fn zeros(size: usize) -> Self {
        let data: Vec<T> = vec![T::zero(); size];
        CpuStorage(data)
    }
}

impl<T: DType> Index<usize> for CpuStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: DType> IndexMut<usize> for CpuStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// UNIMPLEMENTED
#[derive(Debug, Clone, PartialEq)]
pub struct CudaStorage<T: DType> {
    data: Vec<T>, // Placeholder for compilation
}
