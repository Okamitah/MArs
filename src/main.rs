fn main() {
    let tensor = Tensor {
        data: TensorData::F32(vec![1.0, 1.1, 1.3]),
        shape: vec![3],
        strides: None,
        dtype: DType::F32,
    };
    println!("{:?}", tensor);

}

#[derive(Debug)]
enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

#[derive(Debug)]
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
