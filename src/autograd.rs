use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backend::Backend;
use crate::dtype::DType;
use crate::layout::TensorLayout;
use crate::ops::OpType;
use crate::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub struct Node<T: DType, B: Backend<T>> {
    op:      OpType,
    parents: Vec<NodeId>,
    layout:  TensorLayout,
    value:   B::Storage,
}

impl<T: DType, B: Backend<T>> Node<T, B> {
    pub fn new(op: OpType, parents: Vec<NodeId>, layout: TensorLayout, value: B::Storage) -> Self {
        Node { op, parents, layout, value }
    }

    pub fn parents(&self) -> &Vec<NodeId> {
        &self.parents
    }

    pub fn op(&self) -> OpType {
        self.op
    }

    pub fn value(&self) -> &B::Storage {
        &self.value
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }
}

#[derive(Debug, Default)]
pub struct Tape<T: DType, B: Backend<T>> {
    nodes: Mutex<Vec<Node<T, B>>>,
}

impl<T: DType, B: Backend<T>> Tape<T, B> {
    pub fn new() -> Self {
        Tape { nodes: Mutex::new(Vec::new()) }
    }

    pub fn add_node(&self, node: Node<T, B>) -> NodeId {
        let mut nodes = self.nodes.lock().expect("autograd tape mutex poisoned");
        let id = NodeId(nodes.len());
        nodes.push(node);
        id
    }

    pub fn node(&self, id: NodeId) -> Option<Node<T, B>> {
        self.nodes
            .lock()
            .ok()
            .and_then(|nodes| nodes.get(id.0).cloned())
    }

    pub fn len(&self) -> usize {
        self.nodes.lock().map(|nodes| nodes.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Gradients<T: DType, B: Backend<T>> {
    map: HashMap<NodeId, Tensor<T, B>>,
}

impl<T: DType, B: Backend<T>> Gradients<T, B> {
    pub fn new() -> Self {
        Gradients { map: HashMap::new() }
    }

    pub fn new_from_map(map: HashMap<NodeId, Tensor<T, B>>) -> Self {
        Gradients { map }
    }

    pub fn get(&self, tensor: &Tensor<T, B>) -> Option<&Tensor<T, B>> {
        tensor.node_id().and_then(|id| self.map.get(&id))
    }
}
