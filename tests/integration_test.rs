use std::sync::Arc;

use nanoflare::backend::Backend;
use nanoflare::{CpuBackend, Tensor, TensorLayout};

#[test]
fn tensor_broadcast_to_preserves_values() {
    let backend = Arc::new(CpuBackend::new());
    let storage = backend.from_vec(vec![1i32, 2, 3]);
    let t = Tensor::from_parts(storage, TensorLayout::new(vec![3]), backend.clone());

    let b = t.broadcast_to([2, 3]).unwrap();
    let layout = b.layout();

    let cases = [
        (vec![0usize, 0usize], 1i32),
        (vec![0usize, 2usize], 3i32),
        (vec![1usize, 1usize], 2i32),
    ];

    for (idx, expected) in cases {
        let flat = layout.ravel_index(&idx).unwrap();
        assert_eq!(b.storage()[flat], expected);
    }
}

#[test]
fn add_broadcasts_rhs_vector() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1i32, 2, 3, 4, 5, 6]),
        TensorLayout::new(vec![2, 3]),
        backend.clone(),
    );
    let b = Tensor::from_parts(
        backend.from_vec(vec![10i32, 20, 30]),
        TensorLayout::new(vec![3]),
        backend.clone(),
    );

    let c = (&a + &b).unwrap();
    assert_eq!(c.layout().shape().as_slice(), &[2, 3]);

    let expected = [11i32, 22, 33, 14, 25, 36];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(c.storage()[i], v);
    }
}

#[test]
fn add_broadcasts_both_sides() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1i32, 2]),
        TensorLayout::new(vec![2, 1]),
        backend.clone(),
    );
    let b = Tensor::from_parts(
        backend.from_vec(vec![10i32, 20, 30]),
        TensorLayout::new(vec![1, 3]),
        backend.clone(),
    );

    let c = (&a + &b).unwrap();
    assert_eq!(c.layout().shape().as_slice(), &[2, 3]);

    let expected = [11i32, 21, 31, 12, 22, 32];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(c.storage()[i], v);
    }
}

#[test]
fn add_broadcasts_lhs_vector() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1i32, 2, 3]),
        TensorLayout::new(vec![3]),
        backend.clone(),
    );
    let b = Tensor::from_parts(
        backend.from_vec(vec![10i32, 20, 30, 40, 50, 60]),
        TensorLayout::new(vec![2, 3]),
        backend.clone(),
    );

    let c = (&a + &b).unwrap();
    assert_eq!(c.layout().shape().as_slice(), &[2, 3]);

    let expected = [11i32, 22, 33, 41, 52, 63];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(c.storage()[i], v);
    }
}
