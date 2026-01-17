use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use crate::dtype::DType;

#[derive(Debug, Clone)]
pub enum TensorStorage<T: DType> {
    Cpu(Arc<CpuStorage<T>>),
    Cuda(Arc<CudaStorage<T>>),
}

impl<T: DType> Index<usize> for TensorStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            TensorStorage::Cpu(storage) => &storage[index],
            TensorStorage::Cuda(storage) => &storage[index],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuStorage<T: DType>(Vec<T>);

impl<T: DType> CpuStorage<T> {
    pub fn zeros(size: usize) -> Self {
        let data: Vec<T> = vec![T::zero(); size];
        CpuStorage(data)
    }

    pub fn ones(size: usize) -> Self {
        let data: Vec<T> = vec![T::one(); size];
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

impl<T: DType> Index<usize> for CudaStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        todo!("{}", index)
    }
}

impl<T: DType> IndexMut<usize> for CudaStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        todo!("{}", index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immut_cpu_storage_indexing() {
        let storage = CpuStorage::<f32>::ones(10);
        assert_eq!(storage[0], 1.0f32);
        assert_eq!(storage[5], 1.0f32);
        assert_eq!(storage[9], 1.0f32);
    }

    #[test]
    fn mut_cpu_storage_indexing() {
        let mut storage = CpuStorage::<f32>::zeros(10);
        storage[0] = 3.14f32;
        storage[5] = 2.71f32;
        assert_eq!(storage[0], 3.14f32);
        assert_eq!(storage[5], 2.71f32);
        assert_eq!(storage[9], 0.0f32);
    }

    #[test]
    fn tensor_storage_cpu_indexing() {
        let cpu_storage = Arc::new(CpuStorage::<f32>::ones(10));
        let tensor_storage = TensorStorage::Cpu(cpu_storage);
        assert_eq!(tensor_storage[0], 1.0f32);
        assert_eq!(tensor_storage[9], 1.0f32);
    }
}
