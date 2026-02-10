use std::fmt::Debug;
use std::ops::{Index, IndexMut, RangeFull};
use std::sync::Arc;

use crate::dtype::DType;

pub trait TensorStorage<T: DType>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
}

impl<T: DType, S: TensorStorage<T>> TensorStorage<T> for Arc<S> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuStorage<T: DType>(Vec<T>);

impl<T: DType> CpuStorage<T> {
    pub fn new(data: Vec<T>) -> Self {
        CpuStorage(data)
    }
}

impl<T: DType> TensorStorage<T> for CpuStorage<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: DType> CpuStorage<T> {
    pub fn zeros(size: usize) -> Self {
        let data: Vec<T> = vec![T::zero(); size];
        CpuStorage(data)
    }

    pub fn ones(size: usize) -> Self {
        let data: Vec<T> = vec![T::one(); size];
        CpuStorage(data)
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T: DType> Index<usize> for CpuStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: DType> Index<RangeFull> for CpuStorage<T> {
    type Output = [T];

    fn index(&self, _index: RangeFull) -> &Self::Output {
        &self.0
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

impl<T: DType> TensorStorage<T> for CudaStorage<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T: DType> Index<usize> for CudaStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        unimplemented!("{}", index)
    }
}

impl<T: DType> IndexMut<usize> for CudaStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unimplemented!("{}", index)
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
        let cpu_storage = CpuStorage::<f32>::ones(10);
        assert_eq!(cpu_storage[0], 1.0f32);
        assert_eq!(cpu_storage[9], 1.0f32);
    }
}
