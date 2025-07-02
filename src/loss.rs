use crate::tensor::{Tensor, TensorData, TensorError};

// Regression loss functions

pub fn mae(pred: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = a[i] - b[i];
                error += err.abs();
            }
            Ok(error/size as f64)
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = a[i] as f64 - b[i] as f64;
                error += err.abs();
            }
            Ok(error/size as f64)
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

pub fn mse(pred: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = a[i] - b[i];
                error += err.powi(2);
            }
            Ok(error/size as f64)
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = a[i] as f64 - b[i] as f64;
                error += err.powi(2);
            }
            Ok(error/size as f64)
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

pub fn rmse(pred: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = a[i] - b[i];
                error += err.powi(2);
            }
            Ok((error/size as f64).sqrt())
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = a[i] as f64 - b[i] as f64;
                error += err.powi(2);
            }
            Ok((error/size as f64).sqrt())
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

pub fn mbe(pred: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = a[i] - b[i];
                error += err;
            }
            Ok(error/size as f64)
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = a[i] as f64 - b[i] as f64;
                error += err;
            }
            Ok(error/size as f64)
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

pub fn smae(pred: &Tensor, target: &Tensor, delta: f64) -> Result<f64, TensorError> {
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = a[i] - b[i];

                if err.abs() < delta {
                    error += 0.5 * err.powi(2);
                } else {
                    error += delta * err.abs() - 0.5 * delta.powi(2);
                }
            }
            Ok(error/size as f64)
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = a[i] as f64 - b[i] as f64;

                if err.abs() < delta {
                    error += 0.5 * err.powi(2);
                } else {
                    error += delta * err.abs() - 0.5 * delta.powi(2);
                }
            }
            Ok(error/size as f64)
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

// Classification loss functions

pub fn cross_entropy(pred: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
    
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = - (b[i] * a[i].log10() + (1.0 - b[i]) * (1.0 - a[i]).log10());
                error += err;
            }
            Ok(error/size as f64)
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = - (b[i] as f64 * (a[i] as f64).log10() + (1.0 - b[i] as f64) * (1.0 - a[i] as f64).log10());
                error += err;
            }
            Ok(error/size as f64)
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

pub fn hinge(pred: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
    
    if pred.shape != target.shape {
        return Err(TensorError::TypeMismatch);
    }

    let mut error = 0.0;
    let size = pred.shape.iter().product();

    match (&pred.data, &target.data) {
        (TensorData::F64(a), TensorData::F64(b)) => {
            for i in 0..size {
                let err = max(0.0, 1.0 - b[i] * a[i]);
                error += err;
            }
            Ok(error/size as f64)
        },
        (TensorData::I32(a), TensorData::I32(b)) => {
            for i in 0..size {
                let err = max(0.0, 1.0 - b[i] as f64 * a[i] as f64);
                error += err;
            }
            Ok(error/size as f64)
        },
        _ => return Err(TensorError::TypeMismatch),
    }
}

fn max(num1: f64, num2: f64) -> f64 {
    if num1 > num2 { return num1; }
    num2
}

