use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, RangeInclusive, Sub};
use std::sync::Arc;

use crate::autograd::{Gradients, Node, NodeId, Tape};
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::index::TensorIndex;
use crate::layout::TensorLayout;
use crate::ops::*;
use crate::{Error, Result};

macro_rules! impl_binary_op {
    // Trait-backed ops: "trait: Add, add, AddOp::new, true;"
    (trait: $trait:ident, $method:ident, $op:expr, $broadcast:expr; $($rest:tt)*) => {
        // Owned implementation: a + b
        impl<T: DType, B: Backend<T>> $trait for Tensor<T, B> {
            type Output = Result<Self>;
            fn $method(self, rhs: Self) -> Self::Output {
                match $op(&self, &rhs) {
                    Ok(op) => self.binary_op(
                        &rhs,
                        op,
                        |backend, a, b| backend.$method(a, b),
                        $broadcast,
                    ),
                    Err(e) => Err(e),
                }
            }
        }

        // Reference implementation: &a + &b
        impl<'a, 'b, T:DType, B: Backend<T>> $trait<&'b Tensor<T, B>> for &'a Tensor<T, B> {
            type Output = Result<Tensor<T, B>>;
            fn $method(self, rhs: &'b Tensor<T, B>) -> Self::Output {
                match $op(self, rhs) {
                    Ok(op) => self.binary_op(
                        rhs,
                        op,
                        |backend, a, b| backend.$method(a, b),
                        $broadcast,
                    ),
                    Err(e) => Err(e),
                }
            }
        }

        impl_binary_op!($($rest)*);
    };

    // Plain-named ops: "fn: matmul, matmul, MatMulOp::new, false;"
    (fn: $name:ident, $backend_method:ident, $op:expr, $broadcast:expr; $($rest:tt)*) => {
        impl<T: DType, B: Backend<T>> Tensor<T, B> {
            pub fn $name(&self, rhs: &Tensor<T, B>) -> Result<Self> {
                match $op(self, rhs) {
                    Ok(op) => self.binary_op(
                        rhs,
                        op,
                        |backend, a, b| backend.$backend_method(a, b),
                        $broadcast,
                    ),
                    Err(e) => Err(e),
                }
            }
        }

        impl_binary_op!($($rest)*);
    };

    // termination
    () => {};
}

macro_rules! impl_unary_op {
    ($($trait:ident, $method:ident, $op:expr);* $(;)?) => {
        $(
            // 1. Owned implementation: a + b
            impl<T:DType, B: Backend<T>> Tensor<T, B> {
                pub fn $method(&self) -> Result<Self> {
                    match $op(&self) {
                        Ok(op) => self.unary_op(op, |backend, a| backend.$method(a)),
                        Err(e) => Err(e),
                    }
                }
            }
        )*
    };
}

#[derive(Debug, Clone)]
pub struct Tensor<T: DType, B: Backend<T>> {
    storage:       B::Storage,
    layout:        TensorLayout,
    backend:       Arc<B>,
    requires_grad: bool,
    tape:          Option<Arc<Tape<T, B>>>,
    node_id:       Option<NodeId>,
}

impl_unary_op!();

impl<T, B> Tensor<T, B>
where
    T: FloatDType,
    B: Backend<T>,
{
    pub fn exp(&self) -> Result<Self> {
        match ExpOp::new(&self) {
            Ok(op) => self.unary_op(op, |backend, a| backend.exp(a)),
            Err(e) => Err(e),
        }
    }

    pub fn log(&self) -> Result<Self> {
        match LogOp::new(&self) {
            Ok(op) => self.unary_op(op, |backend, a| backend.log(a)),
            Err(e) => Err(e),
        }
    }
}

impl_binary_op!(
    trait: Add, add, AddOp::new, true;
    trait: Sub, sub, SubOp::new, true;
    trait: Mul, mul, MulOp::new, true;
    trait: Div, div, DivOp::new, true;
    fn: matmul, matmul, MatMulOp::new, false;
);

impl<T: DType, B: Backend<T>> Tensor<T, B> {
    /// Create a new tensor filled with zeros, given a specific layout and
    /// backend.
    pub fn zeros(layout: TensorLayout, backend: Arc<B>) -> Self {
        let storage = backend.store_zeros(&layout);

        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            tape: None,
            node_id: None,
        }
    }

    /// Create a new tensor filled with ones, given a specific layout and
    /// backend.
    pub fn ones(layout: TensorLayout, backend: Arc<B>) -> Self {
        let storage = backend.store_ones(&layout);

        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            tape: None,
            node_id: None,
        }
    }

    /// Create a new tensor from the given storage, layout, and backend.
    pub fn from_parts(storage: B::Storage, layout: TensorLayout, backend: Arc<B>) -> Self {
        Tensor {
            storage,
            layout,
            backend,
            requires_grad: false,
            tape: None,
            node_id: None,
        }
    }

    pub fn cast<U>(&self) -> Result<Tensor<U, B>>
    where
        U: DType,
        B: Backend<U>,
        T: num_traits::ToPrimitive,
        U: num_traits::NumCast,
    {
        if self.requires_grad {
            return Err(Error::RequiresGradUnsupported { op: "cast" });
        }

        self.backend.cast::<U>(self)
    }

    pub fn requires_grad(mut self, tape: Arc<Tape<T, B>>) -> Self {
        let node_id = tape.add_node(Node::new(
            OpType::Leaf,
            Vec::new(),
            self.layout.clone(),
            self.storage.clone(),
        ));

        self.requires_grad = true;
        self.tape = Some(tape);
        self.node_id = Some(node_id);
        self
    }

    /// Get a reference to the tensor's layout.
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    /// Get a reference to the tensor's storage.
    pub fn storage(&self) -> &B::Storage {
        &self.storage
    }

    /// Get a reference to the tensor's backend.
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    /// Get the node ID associated with this tensor in the autograd tape.
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Get some element(s) from the tensor by specifying indices or slices.
    ///
    /// # Returns
    /// A new `Tensor` view containing the selected elements.
    ///
    /// # Example
    /// ```rust
    /// use nanoflare::backend::cpu::CpuBackend;
    /// use nanoflare::{Tensor, i};
    ///
    /// let backend = std::sync::Arc::new(CpuBackend);
    /// let layout = nanoflare::TensorLayout::new(vec![2, 3]);
    /// let tensor = Tensor::<f32, _>::ones(layout, backend);
    ///
    /// // Get a single element
    /// let element = tensor.get(i![0, 1]);
    ///
    /// // Get a slice of the tensor
    /// let slice = tensor.get(i![0..2, 1..3]);
    ///
    /// // Get a sub-tensor
    /// let sub_tensor = tensor.get(i![0..1, 0..2]);
    ///
    /// // Get a full tensor
    /// let full = tensor.get(i![.., ..]);
    /// ```
    pub fn get<I: AsRef<[TensorIndex]>>(&self, indices: I) -> Result<Self> {
        let layout = self.layout.get(indices)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }

    fn unary_op<O, F>(&self, op: O, f: F) -> Result<Self>
    where
        O: UnaryOp<T, B> + 'static,
        F: FnOnce(&B, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    {
        let mut out = f(&self.backend, self)?;

        out.requires_grad = self.requires_grad;

        if !self.requires_grad {
            return Ok(out);
        }

        let tape = match &self.tape {
            Some(tape) => tape.clone(),
            None => return Ok(out),
        };

        let input_id = self.node_id.unwrap_or_else(|| {
            tape.add_node(Node::new(
                OpType::Leaf,
                Vec::new(),
                self.layout.clone(),
                self.storage.clone(),
            ))
        });

        let out_id = tape.add_node(Node::new(
            op.to_optype(),
            vec![input_id],
            out.layout.clone(),
            out.storage.clone(),
        ));

        out.tape = Some(tape);
        out.node_id = Some(out_id);

        Ok(out)
    }

    fn binary_op<O, F>(&self, rhs: &Tensor<T, B>, op: O, f: F, broadcast: bool) -> Result<Self>
    where
        O: BinaryOp<T, B> + 'static,
        F: FnOnce(&B, &Tensor<T, B>, &Tensor<T, B>) -> Result<Tensor<T, B>>,
    {
        let mut out = if broadcast {
            let out_shape = broadcasted_shape(self, rhs)?;
            let a_view = self.broadcast_to(out_shape.as_slice())?;
            let b_view = rhs.broadcast_to(out_shape.as_slice())?;
            f(&self.backend, &a_view, &b_view)?
        } else {
            f(&self.backend, self, rhs)?
        };

        let needs_grad = self.requires_grad || rhs.requires_grad;
        out.requires_grad = needs_grad;

        if !needs_grad {
            return Ok(out);
        }

        let tape = match (&self.tape, &rhs.tape) {
            (Some(left), Some(right)) if Arc::ptr_eq(left, right) => Some(left.clone()),
            (Some(left), None) => Some(left.clone()),
            (None, Some(right)) => Some(right.clone()),
            _ => None,
        };

        if let Some(tape) = tape {
            let left_id = self.node_id.unwrap_or_else(|| {
                tape.add_node(Node::new(
                    OpType::Leaf,
                    Vec::new(),
                    self.layout.clone(),
                    self.storage.clone(),
                ))
            });

            let right_id = rhs.node_id.unwrap_or_else(|| {
                tape.add_node(Node::new(
                    OpType::Leaf,
                    Vec::new(),
                    rhs.layout.clone(),
                    rhs.storage.clone(),
                ))
            });

            let out_id = tape.add_node(Node::new(
                op.to_optype(),
                vec![left_id, right_id],
                out.layout.clone(),
                out.storage.clone(),
            ));

            out.tape = Some(tape);
            out.node_id = Some(out_id);
        }

        Ok(out)
    }

    pub fn backward(&self) -> Result<Gradients<T, B>>
    where
        T: FloatDType,
    {
        if !self.requires_grad {
            return Ok(Gradients::new());
        }

        // Get the tape and root node ID, return empty gradients if missing
        let (Some(tape), Some(root_id)) = (&self.tape, self.node_id) else {
            return Ok(Gradients::new());
        };

        //  Perform a depth-first traversal to determine the order of nodes
        let mut stack = vec![root_id];
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }

            order.push(id);
            if let Some(node) = tape.node(id) {
                for parent in node.parents() {
                    stack.push(*parent);
                }
            }
        }

        order.sort_by_key(|id| id.0);
        order.reverse();

        let mut grads: HashMap<NodeId, Tensor<T, B>> = HashMap::new();
        grads.insert(
            root_id,
            Tensor::ones(self.layout.clone(), self.backend.clone()),
        );

        let add_grad =
            |grads: &mut HashMap<NodeId, Tensor<T, B>>, id: NodeId, grad: Tensor<T, B>| {
                if let Some(existing) = grads.remove(&id) {
                    let sum = (&existing + &grad).expect("tensor addition failed in backward");
                    grads.insert(id, sum);
                } else {
                    grads.insert(id, grad);
                }
            };

        for id in order {
            let grad = match grads.remove(&id) {
                Some(g) => g,
                None => continue,
            };

            let node = match tape.node(id) {
                Some(n) => n,
                None => continue,
            };

            if let OpType::Leaf = node.op() {
                continue;
            }

            let mut inputs = Vec::new();
            for parent_id in node.parents() {
                if let Some(parent_node) = tape.node(*parent_id) {
                    inputs.push(Tensor::from_parts(
                        parent_node.value().clone(),
                        parent_node.layout().clone(),
                        self.backend.clone(),
                    ));
                }
            }

            let input_grads = node
                .op()
                .backward(inputs.as_slice(), &grad, &self.backend)?;

            if input_grads.len() != node.parents().len() {
                panic!(
                    "Op {:?} returned {} gradients but has {} parents",
                    node.op(),
                    input_grads.len(),
                    node.parents().len()
                );
            }

            for (i, parent_grad) in input_grads.into_iter().enumerate() {
                add_grad(&mut grads, node.parents()[i], parent_grad);
            }
        }

        Ok(Gradients::new_from_map(grads))
    }

    // TODO: avoid unnecessary clone
    /// Permute the dimensions of the tensor layout according to the given
    /// indices and return a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::permute`.
    /// `Tensor`
    pub fn permute(&self, permuted_indices: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(permuted_indices)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }

    /// Merge the dimensions in the specified inclusive range into a single
    /// dimension and return a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::merge`.
    pub fn merge(&self, dim_range: RangeInclusive<usize>) -> Result<Self> {
        let layout = self.layout.merge(dim_range)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }

    /// Split a given dimension in the tensor into multiple adjacent dimensions
    /// with specified sizes, returning a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::split`.
    pub fn split(&self, dim: usize, shape: impl AsRef<[usize]>) -> Result<Self> {
        let layout = self.layout.split(dim, shape)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }

    /// Reshape the tensor layout to a new shape if it is contiguous in memory
    /// according to `TensorLayout::is_contiguous`. Otherwise, this operation
    /// should create a new contiguous copy and reshape that copy, TODO.
    pub fn reshape(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        if self.layout.is_contiguous() {
            let layout = self.layout.reshape(shape)?;

            Ok(Tensor {
                layout,
                storage: self.storage.clone(),
                backend: self.backend.clone(),
                requires_grad: self.requires_grad,
                tape: self.tape.clone(),
                node_id: self.node_id,
            })
        } else {
            todo!("non-contiguous reshape not implemented yet")
        }
    }

    /// Broadcast the tensor layout to a new shape.
    ///
    /// This is a view-only operation; it does not copy data.
    /// This is a wrapper around `TensorLayout::broadcast_to`.
    pub fn broadcast_to(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        let layout = self.layout.broadcast_to(shape)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }

    /// Slice the tensor along a single dimension, producing a sub-layout and
    /// returning a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::slice`.
    pub fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Result<Self> {
        let layout = self.layout.slice(dim, range)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }

    /// Create a strided view by skipping elements along a single axis with
    /// the given step size, returning a new `Tensor` with the updated layout.
    ///
    /// This is a wrapper around `TensorLayout::skip`.
    pub fn skip(&self, dim: usize, step: usize) -> Result<Self> {
        let layout = self.layout.skip(dim, step)?;

        Ok(Tensor {
            layout,
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            tape: self.tape.clone(),
            node_id: self.node_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i;

    #[test]
    fn test_tensor_indexing() {
        let backend = Arc::new(crate::backend::cpu::CpuBackend);
        let layout = TensorLayout::new(vec![2, 3]);
        let tensor = Tensor::<f32, _>::ones(layout, backend);

        // Get a single element
        let element = tensor.get(i![0, 1]).unwrap();
        assert_eq!(element.layout().shape().as_slice(), &[1, 1]);

        // Get a slice of the tensor
        let slice = tensor.get(i![0..2, 1..3]).unwrap();
        assert_eq!(slice.layout().shape().as_slice(), &[2, 2]);

        // Get a sub-tensor
        let sub_tensor = tensor.get(i![0..1, 0..2]).unwrap();
        assert_eq!(sub_tensor.layout().shape().as_slice(), &[1, 2]);

        // Get a full tensor
        let full_tensor = tensor.get(i![.., ..]).unwrap();
        assert_eq!(full_tensor.layout().shape().as_slice(), &[2, 3]);
    }
}
