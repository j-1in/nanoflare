use std::sync::Arc;

use nanoflare::{CpuBackend, Tensor, TensorLayout};

#[test]
fn example_usage() {
    let backend = Arc::new(CpuBackend::new());
    let layout = TensorLayout::new(vec![2, 2]);
    let a = Tensor::<f32, CpuBackend>::ones(layout, backend);

    println!("{:#?}", a)
}
