fn main() {
    let data = vec![1.5, 7.2, 5.0, 2.6];
    let shape = vec![2,2];
    //let strides = None;
    let dtype = DType::F32;
    let tensor = Tensor::new(data, shape.clone(), dtype.clone());
    let zeroes = Tensor::zeroes(shape.clone(), dtype.clone());
    let ones = Tensor::ones(shape.clone(), dtype.clone());
    println!("{:?}", tensor);
    println!("{:?}", zeroes);
    println!("{:?}", ones);

}

#[derive(Debug)]
enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

#[derive(Debug, Clone)]
enum DType {
    F32,
    I32,
}

#[derive(Debug)]
struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
    strides: Option<Vec<usize>>,
    dtype: DType,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: TensorData::F32(data),
            shape,
            strides: None,
            dtype,
        }
    }

    pub fn zeroes(shape: Vec<usize>, dtype: DType) -> Self {
        let size = shape.iter().product();
        match dtype {
            DType::F32 => {
                let data = vec![0.0; size];
                Self {
                    data: TensorData::F32(data),
                    shape,
                    strides: None,
                    dtype: DType::F32,
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
            DType::F32 => {
                let data = vec![1.0; size];
                Self {
                    data: TensorData::F32(data),
                    shape,
                    strides: None,
                    dtype: DType::F32,
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
}
