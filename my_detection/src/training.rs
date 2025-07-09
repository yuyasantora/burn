use burn::{
    prelude::*,
    tensor::backend::{AutodiffBackend, Backend},
    train::{TrainOutput, TrainStep, ValidStep},
};

use crate::{
    data::DetectionBatch,
    model::{DetectionLoss, DetectionModel},
};

const NUM_CLASSES: usize = 5;

// 学習ステップの実装
impl<B: AutodiffBackend> TrainStep<DetectionBatch<B>, DetectionLoss<B>> for DetectionModel<B> {
    fn step(&self, batch: DetectionBatch<B>) -> TrainOutput<DetectionLoss<B>> {
        let item = self.forward_detection(batch.images, &batch.targets, NUM_CLASSES);
        // 損失から勾配を計算
        let grads = item.total_loss.backward();
        // モデルを更新して結果を返す
        TrainOutput::new(self, grads, item)
    
    }
}

// 検証ステップの実装
impl<B: Backend> ValidStep<DetectionBatch<B>, DetectionLoss<B>> for DetectionModel<B> {
    fn step(&self, batch: DetectionBatch<B>) -> ValidOutput<DetectionLoss<B>> {
        self.forward_detection(batch.images, &batch.targets, NUM_CLASSES)
    }
}

