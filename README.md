# nanoflare

A lightweight, educational tensor library for Rust with automatic differentiation support.

## Overview

`nanoflare` is designed to provide a clear and concise implementation of tensor operations and autograd mechanics. It currently supports CPU-based operations with a planned CUDA backend, making it suitable for learning about deep learning internals or for lightweight ML experimentation in Rust.

## Features

- **N-Dimensional Tensors**: Support for arbitrary tensor shapes and layouts.
- **Automatic Differentiation**: Reverse-mode autograd (backpropagation) with a dynamic computational graph.
- **Backend Abstraction**: Distinctive separation between tensor logic and hardware backends (CPU implemented, CUDA planned).
- **Type Safety**: Leverages Rust's type system to ensure correctness for floating-point and integer operations.

## Installation

Add `nanoflare` to your `Cargo.toml`:

```toml
[dependencies]
nanoflare = { path = "." } # Adjust path or version as needed
```

## Usage

### Creating Tensors

```rust
use std::sync::Arc;
use nanoflare::{CpuBackend, Tensor, TensorLayout};

fn main() {
    let backend = Arc::new(CpuBackend::new());
    
    // Create a 2x3 tensor filled with ones
    let layout = TensorLayout::new(vec![2, 3]);
    let tensor = Tensor::<f32, CpuBackend>::ones(layout, backend);
    
    println!("{:#?}", tensor);
}
```

### Basic Operations

Standard arithmetic operations are supported.

```rust
use std::sync::Arc;
use nanoflare::{CpuBackend, Tensor, TensorLayout};

let backend = Arc::new(CpuBackend::new());
let layout = TensorLayout::new(vec![2, 2]);

let a = Tensor::<f32, CpuBackend>::ones(layout.clone(), backend.clone());
let b = Tensor::<f32, CpuBackend>::ones(layout, backend);

let c = (&a + &b).unwrap(); // Element-wise addition
```

### Automatic Differentiation

`nanoflare` can track operations on tensors to compute gradients automatically.

```rust
use std::sync::Arc;
use nanoflare::{CpuBackend, Tensor, TensorLayout, Tape};

let backend = Arc::new(CpuBackend::new());
let tape = Arc::new(Tape::new());
let layout = TensorLayout::new(vec![2, 2]);

// Create tensors and enable gradient tracking
let a = Tensor::<f32, CpuBackend>::ones(layout.clone(), backend.clone())
    .requires_grad(tape.clone());
let b = Tensor::<f32, CpuBackend>::ones(layout, backend)
    .requires_grad(tape);

// Perform operations
let c = (&a + &b).unwrap();
let d = c.mul(&a).unwrap(); // d = (a + b) * a

// Backpropagate
let gradients = d.backward().unwrap();

// Access gradients
if let Some(grad_a) = gradients.get(&a) {
    println!("Gradient for a: {:#?}", grad_a);
}
```

## Roadmap & Status

This project is currently in active development.

- **CPU Backend**: Functional for basic element-wise operations (add, sub, mul, div, exp).
- **CUDA Backend**: Structure in place, implementation planned.
- **Matrix Multiplication**: Planned.
- **Broadcasting**: Planned.

## License

MIT License - see [LICENSE](LICENSE) for details.
