fn main() {
    //Variables
    let data = vec![1.5, 7.2, 5.0, 2.6];
    let shape = vec![2,2];
    //let strides = None;
    let dtype = DType::F64;
    let tensor = Tensor::new(data, shape.clone(), dtype.clone());
    let zeroes = Tensor::zeroes(shape.clone(), dtype.clone());
    let ones = Tensor::ones(shape.clone(), dtype.clone());

    println!("{:?}", tensor);
    println!("{:?}", zeroes);
    println!("{:?}", ones);

    let somme1 = tensor.add(&ones);
    let somme2 = tensor.add(&zeroes);
    let mult1 = tensor.mul(&ones);
    let mult2 = tensor.mul(&zeroes);

    println!("Somme 1: {:?}", somme1);
    println!("Somme 2: {:?}", somme2);
    println!("Mult 1: {:?}", mult1);
    println!("Mult 2: {:?}", mult2);

}

#[derive(Debug)]
enum TensorData {
    F64(Vec<f64>),
    I32(Vec<i32>),
}

#[derive(Debug, Clone)]
enum DType {
    F64,
    I32,
}

#[derive(Debug)]
enum TensorError {
    TypeMismatch,
    ShapeMismatch,
    EmptyTensor,
}

#[derive(Debug)]
struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
    strides: Option<Vec<usize>>,
    dtype: DType,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: TensorData::F64(data),
            shape,
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
                    strides: None,
                    dtype: DType::F64,
                }
            }
            DType::I32 => {
                let data = vec![0; size];
                Self {
                    data: TensorData::I32(data),
                    shape,
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
                    strides: None,
                    dtype: DType::F64,
                }
            }
            DType::I32 => {
                let data = vec![1; size];
                Self {
                    data: TensorData::I32(data),
                    shape,
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

        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x+y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x+y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
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

        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x*y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x*y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
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

        match (&self.data, &tensor.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let result: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x-y).collect();
                Ok(Tensor {
                    data: TensorData::F64(result),
                    shape: self.shape.clone(),
                    strides: None,
                    dtype: DType::F64
                })
            },
            (TensorData::I32(a), TensorData::I32(b)) => {
                let result: Vec<i32> = a.iter().zip(b).map(|(&x, &y)| x-y).collect();
                Ok(Tensor {
                    data: TensorData::I32(result),
                    shape: self.shape.clone(),
                    strides: None,
                    dtype: DType::I32
                })
            },
            _ => Err(TensorError::TypeMismatch)
        }
    }


    pub fn div(&self, tensor: &Self) -> Result<Self, TensorError> {
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
    }

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
