#![recursion_limit = "512"]

mod data;
mod model;
mod training; // ★追加

use burn::{
    backend::{wgpu::{Wgpu, WgpuDevice}, Autodiff}, // wgpuを使う場合はこちら
    // backend::{ndarray::NdArray, Autodiff}, // CPUで試す場合はこちら
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    train::{
        metric::LossMetric,
        LearnerBuilder,
    },
    record::{CompactRecorder, Recorder},
};

use data::{CocoDetectionDataset, DetectionBatcher};
use model::{BackboneConfig, NeckConfig, HeadConfig, DetectionModelConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type MyBackend = Wgpu; // ★wgpu に変更
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    
    // データセットの作成
    let dataset_train = CocoDetectionDataset::new(
        "datasets/coco128/images/train2017",      // ★自分の画像フォルダパス
        "datasets/coco128/annotations/instances_train2017.json", // ★自分のアノテーションファイルパス
        (416, 416),
    )?;
    let dataset_valid = CocoDetectionDataset::new(
        "datasets/coco128/images/train2017",      // ★検証用データも同じものを流用
        "datasets/coco128/annotations/instances_train2017.json",
        (416, 416),
    )?;
    
    // バッチャーとデータローダー
    let batcher_train = DetectionBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = DetectionBatcher::<MyBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(4)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);
    
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(4)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_valid);
    
    // モデルのConfig
    let model_config = DetectionModelConfig::new(
        BackboneConfig::new([64, 128, 256]),
        NeckConfig::new([64, 128, 256], 128),
        HeadConfig::new(5, 128), // 5クラス, Neckからのチャンネル数128
    );

    // 学習器の設定
    let learner = LearnerBuilder::new("./detection_artifacts")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(10)
        .summary()
        .build(
            model_config.init(&device),
            AdamConfig::new().init(),
            1e-4,
        );
    
    // 学習実行
    let _model_trained = learner.fit(dataloader_train, dataloader_valid);
    
    println!("Detection training completed!");
    
    Ok(())
}