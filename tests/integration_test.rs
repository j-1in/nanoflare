use std::sync::Arc;

use nanoflare::backend::Backend;
use nanoflare::{CpuBackend, Error, Tensor, TensorLayout};

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

#[test]
fn dot_product_returns_expected_value() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1.0f32, 2.0, 3.0]),
        TensorLayout::new(vec![3]),
        backend.clone(),
    );
    let b = Tensor::from_parts(
        backend.from_vec(vec![4.0f32, 5.0, 6.0]),
        TensorLayout::new(vec![3]),
        backend.clone(),
    );

    let out = a.dot(&b).unwrap();
    assert_eq!(out.layout().shape().as_slice(), &[1]);
    assert_eq!(out.storage().as_slice(), &[32.0]);
}

#[test]
fn dot_product_supports_strided_vectors() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1.0f32, 99.0, 2.0, 99.0, 3.0, 99.0]),
        TensorLayout::new(vec![6]),
        backend.clone(),
    )
    .skip(0, 2)
    .unwrap();
    let b = Tensor::from_parts(
        backend.from_vec(vec![4.0f32, 77.0, 5.0, 77.0, 6.0, 77.0]),
        TensorLayout::new(vec![6]),
        backend.clone(),
    )
    .skip(0, 2)
    .unwrap();

    let out = a.dot(&b).unwrap();
    assert_eq!(out.layout().shape().as_slice(), &[1]);
    assert_eq!(out.storage().as_slice(), &[32.0]);
}

#[test]
fn dot_product_dimension_mismatch_errors() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1.0f32, 2.0, 3.0]),
        TensorLayout::new(vec![3]),
        backend.clone(),
    );
    let b = Tensor::from_parts(
        backend.from_vec(vec![1.0f32, 2.0]),
        TensorLayout::new(vec![2]),
        backend.clone(),
    );

    let err = a.dot(&b).unwrap_err();
    match err {
        Error::DotDimensionMismatch { a_dim, b_dim, .. } => {
            assert_eq!(a_dim, 3);
            assert_eq!(b_dim, 2);
        }
        _ => panic!("unexpected error variant"),
    }
}

#[test]
fn dot_product_invalid_rank_errors() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::from_parts(
        backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0]),
        TensorLayout::new(vec![2, 2]),
        backend.clone(),
    );
    let b = Tensor::from_parts(
        backend.from_vec(vec![1.0f32, 2.0]),
        TensorLayout::new(vec![2]),
        backend.clone(),
    );

    let err = a.dot(&b).unwrap_err();
    match err {
        Error::DotInvalidShape { .. } => {}
        _ => panic!("unexpected error variant"),
    }
}
