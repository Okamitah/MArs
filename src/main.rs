use MA_rs::tensor::{Tensor, DType};
use MA_rs::loss::{mae, mse, rmse, cross_entropy, hinge};

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

    println!("\nMae loss: {:?}", mae(&zeroes, &ones));
    println!("Mse loss: {:?}", mse(&zeroes, &ones));
    println!("rmse loss: {:?}", rmse(&zeroes, &ones));
    println!("cr ent loss: {:?}", cross_entropy(&ones, &ones));

}

