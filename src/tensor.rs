#[derive(Debug)]
pub enum TensorData {
    F64(Vec<f64>),
    I32(Vec<i32>),
}

#[derive(Debug, Clone)]
pub enum DType {
    F64,
    I32,
}

#[derive(Debug)]
pub enum TensorError {
    TypeMismatch,
    ShapeMismatch,
    EmptyTensor,
}

pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
    pub gradient: Option<Vec<f64>>,
    pub gradient_fn: Option<BackwardFn>,
    pub requires_grad: bool,
    strides: Option<Vec<usize>>,
    dtype: DType,
}

type BackwardFn = Box<dyn FnOnce(&[f64]) -> Vec<f64>>;

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: TensorData::F64(data),
            shape,
            gradient: None,
            gradient_fn: None,
            requires_grad: false,
            strides: None,
            dtype,
        }
    }

    pub fn zeroes(shape: Vec<usize>, dtype: DType) -> Self {
        let size = shape.iter().product();
        match dtype {
            DType::F64 => {
                let data = vec![0.0; size];
                Self {
                    data: TensorData::F64(data),
                    shape,
                    gradient: None,
                    gradient_fn: None,
                    requires_grad: false,
                    strides: None,
                    dtype: DType::F64,
                }
            }
            DType::I32 => {
                let data = vec![0; size];
                Self {
                    data: TensorData::I32(data),
                    shape,
                    gradient: None,
                    gradient_fn: None,
                    requires_grad: false,
                    strides: None,
                    dtype: DType::I32,
                }
            }
        }
    }

    pub fn ones(shape: Vec<usize>, dtype: DType) -> Self {
        let size = shape.iter().product();
        match dtype {
            DType::F64 => {
                let data = vec![1.0; size];
                Self {
                    data: TensorData::F64(data),
                    shape,
                    gradient: None,
                    gradient_fn: None,
                    requires_grad: false,
                    strides: None,
                    dtype: DType::F64,
                }
            }
            DType::I32 => {
                let data = vec![1; size];
                Self {
                    data: TensorData::I32(data),
                    shape,
                    gradient: None,
                    gradient_fn: None,
                    requires_grad: false,
                    strides: None,
                    dtype: DType::I32,
                }
            }
        }
    }

    pub fn add(&self, tensor: &Self) -> Result<Self, TensorError> {
        if self.shape != tensor.shape {
            return Err(TensorError::ShapeMismatch);
        }

        let grad_fn = if self.requires_grad || tensor.requires_grad {
            Some(Box::new(move |grad_output: &[f64]| {
                grad_output.to_vec()
            }) as BackwardFn)
        } else {
            None
        };

        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x+y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    gradient: None,
                    gradient_fn: grad_fn,
                    requires_grad: self.requires_grad || tensor.requires_grad,
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x+y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
                    gradient: None,
                    gradient_fn: grad_fn,
                    requires_grad: self.requires_grad || tensor.requires_grad,
                    strides: None,
                    dtype: DType::I32
                })
            },
            _ => Err(TensorError::TypeMismatch)
        }
    }

    pub fn mul(&self, tensor: &Self) -> Result<Self, TensorError> {
        if self.shape != tensor.shape {
            return Err(TensorError::ShapeMismatch);
        }

        let grad_fn = if self.requires_grad || tensor.requires_grad {
            let a_data = match &self.data {
                TensorData::F64(v) => v.clone(),
                _ => unreachable!(),
            };
            let b_data = match &tensor.data {
                TensorData::F64(v) => v.clone(),
                _ => unreachable!(),
            };

            Some(Box::new(move |grad_output: &[f64]| {
                let mut grad_a = vec![0.0; grad_output.len()];
                for i in 0..grad_output.len() {
                    grad_a[i] = grad_output[i] * b_data[i];
                }
                grad_a
            }) as BackwardFn)
        } else {
            None
        };

        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x*y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    gradient: None,
                    gradient_fn: grad_fn,
                    requires_grad: self.requires_grad || tensor.requires_grad,
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x*y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
                    gradient: None,
                    gradient_fn: grad_fn,
                    requires_grad: self.requires_grad || tensor.requires_grad,
                    strides: None,
                    dtype: DType::I32
                })
            },
            _ => Err(TensorError::TypeMismatch)
        }
    }

    pub fn sub(&self, tensor: &Self) -> Result<Self, TensorError> {
        if self.shape != tensor.shape {
            return Err(TensorError::ShapeMismatch);
        }

        let grad_fn = if self.requires_grad || tensor.requires_grad {
            Some(Box::new(move |grad_output: &[f64]| {
                grad_output.to_vec()
            }) as BackwardFn)
        } else {
            None
        };


        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x-y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    gradient: None,
                    gradient_fn: grad_fn,
                    requires_grad: self.requires_grad || tensor.requires_grad,
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x-y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
                    gradient: None,
                    gradient_fn: grad_fn,
                    requires_grad: self.requires_grad || tensor.requires_grad,
                    strides: None,
                    dtype: DType::I32
                })
            },
            _ => Err(TensorError::TypeMismatch)
        }
    }


    /*pub fn div(&self, tensor: &Self) -> Result<Self, TensorError> {
        if self.shape != tensor.shape {
            return Err(TensorError::ShapeMismatch);
        }

        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x/y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x/y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
                    strides: None,
                    dtype: DType::I32
                })
            },
            _ => Err(TensorError::TypeMismatch)
        }
    }*/

    pub fn sum(&self) -> Result<f64, TensorError> {
        if self.shape.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        match &self.data {
            TensorData::F64(values) => Ok(values.iter().copied().sum()),
            TensorData::I32(values) => Ok(values.iter().map(|&x| x as f64).sum()),
        }
    }

    pub fn mean(&self) -> Result<f64, TensorError> {
        if self.shape.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        let sum = &self.sum().unwrap();
        let len = &self.shape.iter().map(|&x| x as f64).product();
        
        Ok(sum/len)
    }
}


