use burn::{tensor::Tensor, backend::Wgpu};


type Backend = Wgpu;


fn main() {
    let device = Default::default();
    // テンソル二つを作る
    let tensor1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::<Backend, 2>::ones_like(&tensor1);

    println!("{}", tensor1 + tensor2);
}
