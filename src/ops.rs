use std::fmt::Debug;
use std::sync::Arc;

use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::tensor::UnbroadcastMode;
use crate::{Error, Result, Tensor, TensorShape};

/// A trait representing a tensor operation.
pub trait TensorOp<T: DType, B: Backend<T>>: Debug + Send + Sync {
    /// Returns the name of the operation.
    fn name(&self) -> &str;

    /// Computes the gradient of the operation with respect to its inputs.
    ///
    /// # Arguments
    /// * `inputs` - A slice of tensors representing the original data used for
    ///   the operation
    /// * `grad` - The gradient of the output with respect to some scalar value
    /// * `backend` - The backend to use for computation
    ///
    /// # Returns
    /// A `Result` containing a vector of tensors representing the gradients
    /// with respect to each input.
    ///
    /// # Errors
    /// Returns an error if the gradient computation fails.
    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>>;

    /// Converts the operation to its corresponding `OpType` and returns it.
    fn to_optype(&self) -> OpType;
}

/// A trait representing a unary tensor operation.
pub trait UnaryOp<T: DType, B: Backend<T>>: TensorOp<T, B> {
    /// Creates a new instance of the unary operation.
    ///
    /// # Returns
    /// A new instance of the unary operation.
    fn new(a: &Tensor<T, B>) -> Result<Self>
    where
        Self: Sized;

    /// Validates the shape of the input tensor for the unary operation.
    ///
    /// # Returns
    /// `Ok(())` if the shape is valid, `Err(Error)` otherwise.
    fn validate_shape(_a: &Tensor<T, B>) -> Result<()> {
        Ok(())
    }
}

/// The negation operation datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct NegOp;

impl<T: DType, B: Backend<T>> TensorOp<T, B> for NegOp
where
    T: std::ops::Neg<Output = T>,
{
    fn name(&self) -> &str {
        "Negate"
    }

    fn backward(
        &self,
        _inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        _backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let grad_a = (-grad)?;

        Ok(vec![grad_a])
    }

    fn to_optype(&self) -> OpType {
        OpType::Neg(self.clone())
    }
}

impl<T: DType, B: Backend<T>> UnaryOp<T, B> for NegOp
where
    T: std::ops::Neg<Output = T>,
{
    fn new(_a: &Tensor<T, B>) -> Result<Self> {
        Ok(NegOp)
    }
}

/// The absolute operation datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsOp;

impl<T: DType, B: Backend<T>> TensorOp<T, B> for AbsOp {
    fn name(&self) -> &str {
        "Abs"
    }

    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        _backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let a = &inputs[0];
        let grad_a = (&a.sgn()? * grad)?;

        Ok(vec![grad_a])
    }

    fn to_optype(&self) -> OpType {
        OpType::Abs(self.clone())
    }
}

impl<T: DType, B: Backend<T>> UnaryOp<T, B> for AbsOp {
    fn new(_a: &Tensor<T, B>) -> Result<Self> {
        Ok(AbsOp)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SgnOp;

impl<T: DType, B: Backend<T>> TensorOp<T, B> for SgnOp {
    fn name(&self) -> &str {
        "Sgn"
    }

    fn backward(
        &self,
        _inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // Derivative of sgn is 0 almost everywhere (undefined at 0).
        // We use zero as the default subgradient.
        let grad_a = Tensor::zeros(grad.layout().clone(), backend.clone());
        Ok(vec![grad_a])
    }

    fn to_optype(&self) -> OpType {
        OpType::Sgn(self.clone())
    }
}

impl<T: DType, B: Backend<T>> UnaryOp<T, B> for SgnOp {
    fn new(_a: &Tensor<T, B>) -> Result<Self> {
        Ok(SgnOp)
    }
}

/// The exponential operation datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpOp;

impl<T: DType, B: Backend<T>> TensorOp<T, B> for ExpOp
where
    T: FloatDType,
{
    fn name(&self) -> &str {
        "Exp"
    }

    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        _backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let a = &inputs[0];
        let exp_a = a.exp()?;
        let grad_a = (grad * &exp_a)?;

        Ok(vec![grad_a])
    }

    fn to_optype(&self) -> OpType {
        OpType::Exp(self.clone())
    }
}

impl<T: DType, B: Backend<T>> UnaryOp<T, B> for ExpOp
where
    T: FloatDType,
{
    fn new(_a: &Tensor<T, B>) -> Result<Self> {
        Ok(ExpOp)
    }
}

/// The logarithm operation datatype, takes the natural logarithm of a tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct LogOp;

impl<T: DType, B: Backend<T>> TensorOp<T, B> for LogOp
where
    T: FloatDType,
{
    fn name(&self) -> &str {
        "Log"
    }

    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        _backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let a = &inputs[0];
        let grad_a = (grad / a)?;

        Ok(vec![grad_a])
    }

    fn to_optype(&self) -> OpType {
        OpType::Log(self.clone())
    }
}

impl<T: DType, B: Backend<T>> UnaryOp<T, B> for LogOp
where
    T: FloatDType,
{
    fn new(_a: &Tensor<T, B>) -> Result<Self> {
        Ok(LogOp)
    }
}

/// Trait representing a binary tensor operation.
pub trait BinaryOp<T: DType, B: Backend<T>>: TensorOp<T, B> {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self>
    where
        Self: Sized;

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct AddOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for AddOp {
    fn name(&self) -> &str {
        "Add"
    }

    fn backward(
        &self,
        _inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        _backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let grad_a = grad.clone();
        let grad_b = grad.clone();
        unbroadcast_binary_grads(grad_a, grad_b, &self.lhs_orig_shape, &self.rhs_orig_shape)
    }

    fn to_optype(&self) -> OpType {
        OpType::Add(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for AddOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(AddOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        broadcasted_shapes_match(a, b)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for SubOp {
    fn name(&self) -> &str {
        "Sub"
    }

    fn backward(
        &self,
        _inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let grad_a = grad.clone();

        let zero = Tensor::zeros(grad.layout().clone(), backend.clone());
        let grad_b = (&zero - grad)?;

        unbroadcast_binary_grads(grad_a, grad_b, &self.lhs_orig_shape, &self.rhs_orig_shape)
    }

    fn to_optype(&self) -> OpType {
        OpType::Sub(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for SubOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(SubOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        broadcasted_shapes_match(a, b)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MulOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for MulOp {
    fn name(&self) -> &str {
        "Mul"
    }

    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        _backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let a = &inputs[0];
        let b = &inputs[1];

        let grad_a = (grad * b)?;
        let grad_b = (grad * a)?;

        unbroadcast_binary_grads(grad_a, grad_b, &self.lhs_orig_shape, &self.rhs_orig_shape)
    }

    fn to_optype(&self) -> OpType {
        OpType::Mul(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for MulOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(MulOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        broadcasted_shapes_match(a, b)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DivOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for DivOp {
    fn name(&self) -> &str {
        "Div"
    }

    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        let a = &inputs[0];
        let b = &inputs[1];

        let grad_a = (grad / b)?;

        let b_sq = (b * b)?;
        let numerator = (grad * a)?;

        let term = (&numerator / &b_sq)?;

        let zero = Tensor::zeros(grad.layout().clone(), backend.clone());
        let grad_b = (&zero - &term)?;

        unbroadcast_binary_grads(grad_a, grad_b, &self.lhs_orig_shape, &self.rhs_orig_shape)
    }

    fn to_optype(&self) -> OpType {
        OpType::Div(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for DivOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(DivOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        broadcasted_shapes_match(a, b)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatMulOp {
    lhs_orig_shape: TensorShape,
    rhs_orig_shape: TensorShape,
}

impl<T: DType, B: Backend<T>> TensorOp<T, B> for MatMulOp {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn backward(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        todo!("MatMul backward not implemented yet")
    }

    fn to_optype(&self) -> OpType {
        OpType::MatMul(self.clone())
    }
}

impl<T: DType, B: Backend<T>> BinaryOp<T, B> for MatMulOp {
    fn new(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<Self> {
        Self::validate_shapes(a, b)?;
        Ok(MatMulOp {
            lhs_orig_shape: a.layout().shape().clone(),
            rhs_orig_shape: b.layout().shape().clone(),
        })
    }

    fn validate_shapes(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()> {
        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(Error::MatMulInvalidShape {
                a_shape: a_shape.clone(),
                b_shape: b_shape.clone(),
            });
        }

        let a_last = a_shape[a_shape.len() - 1];
        let b_second_last = b_shape[b_shape.len() - 2];
        if a_last != b_second_last {
            return Err(Error::MatMulDimensionMismatch {
                a_last,
                b_second_last,
                a_shape: a_shape.clone(),
                b_shape: b_shape.clone(),
            });
        }

        Ok(())
    }
}

fn unbroadcast_binary_grads<T: DType, B: Backend<T>>(
    grad_a: Tensor<T, B>,
    grad_b: Tensor<T, B>,
    lhs_shape: &TensorShape,
    rhs_shape: &TensorShape,
) -> Result<Vec<Tensor<T, B>>> {
    let grad_a = grad_a.unbroadcast_to(lhs_shape, UnbroadcastMode::Sum)?;
    let grad_b = grad_b.unbroadcast_to(rhs_shape, UnbroadcastMode::Sum)?;
    Ok(vec![grad_a, grad_b])
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    Leaf,
    Neg(NegOp),
    Abs(AbsOp),
    Sgn(SgnOp),
    Exp(ExpOp),
    Log(LogOp),
    Add(AddOp),
    Sub(SubOp),
    Mul(MulOp),
    Div(DivOp),
    MatMul(MatMulOp),
}

pub(crate) fn broadcasted_shapes_match<T, B>(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<()>
where
    T: crate::dtype::DType,
    B: crate::backend::Backend<T>,
{
    let _ = broadcasted_shape(a, b)?;
    Ok(())
}

/// Computes the broadcasted shape of two tensors, if they are compatible for
/// broadcasting.
pub(crate) fn broadcasted_shape<T, B>(a: &Tensor<T, B>, b: &Tensor<T, B>) -> Result<TensorShape>
where
    T: crate::dtype::DType,
    B: crate::backend::Backend<T>,
{
    let shape_a = a.layout().shape();
    let shape_b = b.layout().shape();

    let rank_a = shape_a.len();
    let rank_b = shape_b.len();

    let max_rank = std::cmp::max(rank_a, rank_b);

    let mut out_rev = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let dim_a = if i < rank_a {
            shape_a[rank_a - 1 - i]
        } else {
            1
        };

        let dim_b = if i < rank_b {
            shape_b[rank_b - 1 - i]
        } else {
            1
        };

        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return Err(Error::LayoutMismatch {
                a: a.layout().shape().clone(),
                b: b.layout().shape().clone(),
            });
        }

        let out_dim = if dim_a == dim_b {
            dim_a
        } else if dim_a == 1 {
            dim_b
        } else {
            dim_a
        };
        out_rev.push(out_dim);
    }

    out_rev.reverse();
    Ok(TensorShape::new(out_rev))
}

impl OpType {
    pub fn name(&self) -> &'static str {
        match self {
            OpType::Leaf => "Leaf",
            OpType::Neg(_) => "Neg",
            OpType::Abs(_) => "Abs",
            OpType::Sgn(_) => "Sgn",
            OpType::Exp(_) => "Exp",
            OpType::Log(_) => "Log",
            OpType::Add(_) => "Add",
            OpType::Sub(_) => "Sub",
            OpType::Mul(_) => "Mul",
            OpType::Div(_) => "Div",
            OpType::MatMul(_) => "MatMul",
        }
    }

    /// Delegates the backward pass to the specific operation implementation
    pub fn backward<T: FloatDType, B: Backend<T>>(
        &self,
        inputs: &[Tensor<T, B>],
        grad: &Tensor<T, B>,
        backend: &Arc<B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        match self {
            OpType::Leaf => Ok(vec![]),
            OpType::Neg(op) => op.backward(inputs, grad, backend),
            OpType::Abs(op) => op.backward(inputs, grad, backend),
            OpType::Sgn(op) => op.backward(inputs, grad, backend),
            OpType::Exp(op) => op.backward(inputs, grad, backend),
            OpType::Log(op) => op.backward(inputs, grad, backend),
            OpType::Add(op) => op.backward(inputs, grad, backend),
            OpType::Sub(op) => op.backward(inputs, grad, backend),
            OpType::Mul(op) => op.backward(inputs, grad, backend),
            OpType::Div(op) => op.backward(inputs, grad, backend),
            OpType::MatMul(op) => op.backward(inputs, grad, backend),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::TensorLayout;
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;

    #[test]
    fn broadcasted_shape_matches_expected() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::<f32, _>::zeros(TensorLayout::new(vec![2, 1, 3]), backend.clone());
        let b = Tensor::<f32, _>::zeros(TensorLayout::new(vec![1, 4, 3]), backend.clone());
        let out = broadcasted_shape(&a, &b).unwrap();
        assert_eq!(out.as_slice(), &[2, 4, 3]);
    }

    #[test]
    fn broadcasted_shape_incompatible_errors() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::<f32, _>::zeros(TensorLayout::new(vec![2, 3]), backend.clone());
        let b = Tensor::<f32, _>::zeros(TensorLayout::new(vec![4, 3]), backend.clone());
        let err = broadcasted_shape(&a, &b).unwrap_err();
        match err {
            crate::Error::LayoutMismatch { .. } => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn matmul_validate_shapes_invalid_rank_errors() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::<f32, _>::zeros(TensorLayout::new(vec![3]), backend.clone());
        let b = Tensor::<f32, _>::zeros(TensorLayout::new(vec![3, 4]), backend);

        let err = MatMulOp::new(&a, &b).unwrap_err();
        match err {
            crate::Error::MatMulInvalidShape { .. } => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn matmul_validate_shapes_inner_dim_mismatch_errors() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::<f32, _>::zeros(TensorLayout::new(vec![2, 3]), backend.clone());
        let b = Tensor::<f32, _>::zeros(TensorLayout::new(vec![4, 5]), backend);

        let err = MatMulOp::new(&a, &b).unwrap_err();
        match err {
            crate::Error::MatMulDimensionMismatch { a_last, b_second_last, .. } => {
                assert_eq!(a_last, 3);
                assert_eq!(b_second_last, 4);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn matmul_validate_shapes_accepts_compatible_inputs() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::<f32, _>::zeros(TensorLayout::new(vec![2, 3]), backend.clone());
        let b = Tensor::<f32, _>::zeros(TensorLayout::new(vec![3, 4]), backend);

        assert!(MatMulOp::new(&a, &b).is_ok());
    }

    #[test]
    fn neg_op_backward_negates_incoming_gradient() {
        let backend = Arc::new(CpuBackend::new());
        let input = Tensor::from_parts(
            backend.from_vec(vec![2.0f32, -1.0, 0.5]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let grad = Tensor::ones(TensorLayout::new(vec![3]), backend.clone());

        let op = NegOp::new(&input).unwrap();
        let out = op.backward(&[input], &grad, &backend).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].storage().as_slice(), &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn sgn_op_backward_returns_zeros() {
        let backend = Arc::new(CpuBackend::new());
        let input = Tensor::from_parts(
            backend.from_vec(vec![-3.0f32, 0.0, 5.0]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let grad = Tensor::ones(TensorLayout::new(vec![3]), backend.clone());

        let op = SgnOp::new(&input).unwrap();
        let out = op.backward(&[input], &grad, &backend).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].storage().as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn add_backward_unbroadcasts_broadcasted_rhs() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::from_parts(
            backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TensorLayout::new(vec![2, 3]),
            backend.clone(),
        );
        let b = Tensor::from_parts(
            backend.from_vec(vec![10.0f32, 20.0, 30.0]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let grad = Tensor::ones(TensorLayout::new(vec![2, 3]), backend.clone());

        let op = AddOp::new(&a, &b).unwrap();
        let out = op.backward(&[a, b], &grad, &backend).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].layout().shape().as_slice(), &[2, 3]);
        assert_eq!(out[0].storage().as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out[1].layout().shape().as_slice(), &[3]);
        assert_eq!(out[1].storage().as_slice(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn sub_backward_unbroadcasts_broadcasted_rhs() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::from_parts(
            backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TensorLayout::new(vec![2, 3]),
            backend.clone(),
        );
        let b = Tensor::from_parts(
            backend.from_vec(vec![10.0f32, 20.0, 30.0]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let grad = Tensor::ones(TensorLayout::new(vec![2, 3]), backend.clone());

        let op = SubOp::new(&a, &b).unwrap();
        let out = op.backward(&[a, b], &grad, &backend).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].layout().shape().as_slice(), &[2, 3]);
        assert_eq!(out[0].storage().as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out[1].layout().shape().as_slice(), &[3]);
        assert_eq!(out[1].storage().as_slice(), &[-2.0, -2.0, -2.0]);
    }

    #[test]
    fn mul_backward_unbroadcasts_both_inputs() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::from_parts(
            backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TensorLayout::new(vec![2, 3]),
            backend.clone(),
        );
        let b = Tensor::from_parts(
            backend.from_vec(vec![10.0f32, 20.0, 30.0]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let grad = Tensor::ones(TensorLayout::new(vec![2, 3]), backend.clone());

        let op = MulOp::new(&a, &b).unwrap();
        let out = op.backward(&[a, b], &grad, &backend).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].layout().shape().as_slice(), &[2, 3]);
        assert_eq!(
            out[0].storage().as_slice(),
            &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        );
        assert_eq!(out[1].layout().shape().as_slice(), &[3]);
        assert_eq!(out[1].storage().as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn div_backward_unbroadcasts_both_inputs() {
        let backend = Arc::new(CpuBackend::new());
        let a = Tensor::from_parts(
            backend.from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TensorLayout::new(vec![2, 3]),
            backend.clone(),
        );
        let b = Tensor::from_parts(
            backend.from_vec(vec![10.0f32, 20.0, 30.0]),
            TensorLayout::new(vec![3]),
            backend.clone(),
        );
        let grad = Tensor::ones(TensorLayout::new(vec![2, 3]), backend.clone());

        let op = DivOp::new(&a, &b).unwrap();
        let out = op.backward(&[a, b], &grad, &backend).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].layout().shape().as_slice(), &[2, 3]);
        let expected_grad_a = [0.1f32, 0.05, 1.0 / 30.0, 0.1, 0.05, 1.0 / 30.0];
        for (got, expected) in out[0]
            .storage()
            .as_slice()
            .iter()
            .zip(expected_grad_a.iter())
        {
            assert!((got - expected).abs() < 1e-6);
        }

        assert_eq!(out[1].layout().shape().as_slice(), &[3]);
        let expected_grad_b = [-0.05f32, -0.0175, -0.01];
        for (got, expected) in out[1]
            .storage()
            .as_slice()
            .iter()
            .zip(expected_grad_b.iter())
        {
            assert!((got - expected).abs() < 1e-6);
        }
    }
}
