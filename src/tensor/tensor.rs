use std::sync::{Arc, RwLock};

#[derive(Debug)]
struct Tensor {
    data: Arc<RwLock<Vec<T>>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    device: Device,
    requires_grad: bool,
    grad: Option<Tensor<T>>,
    grad_fn: Option<GradFn>,
}

#[derive(Debug)]
enum Device {
    CPU,
    GPU,
}

type GradFn = fn(&[Tensor<f32>]) -> Tensor<f32>;

impl<T: Copy+'static> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Self {
        Tensor {
            data: Arc::new(RwLock::new(data)), 
            shape, 
            device, 
            requires_grad
        }
    }

    pub fn zeros(shape: Vec<usize>, device: Device, requires_grad: bool) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![T::default(); total_elements];
        let strides = compute_strides(&shape);
        Tensor {
            data: Arc::new(RwLock::new(data)),
            shape,
            strides,
            device,
            requires_grad,
            grad: None,
            grad_fn: None,
        }
    }

    pub fn ones(shape: Vec<usize>, device: Device, requires_grad: bool) -> Self {

    }

    pub fn rand(shape: Vec<usize>, device: Device, requires_grad: bool) -> Self {

    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;

    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }

    strides.reverse();
    strides
}
