#![recursion_limit = "512"]
mod model;
mod training;
mod data;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{backend::{Wgpu, Autodiff,},
           data::dataset::Dataset,
           optim::AdamConfig,

};


fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    // モデルの保存先
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    // ModelConfigを使ってモデルを初期化
    let model = ModelConfig::new(10, 512).init::<MyAutodiffBackend>(&device);

    // {:?} を使ってモデルの構造を表示
    println!("Model: {:?}", model);
}