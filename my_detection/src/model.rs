use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig,
        Relu,
    },
    prelude::*,
};
// 必要なopsとTensorTraitをインポート
use burn::tensor::{module::interpolate, ops::{InterpolateMode, InterpolateOptions}};
use crate::data::BoundingBox;
use burn::nn::loss::{CrossEntropyLossConfig, L1LossConfig};

// --- モデル全体を定義する ---
#[derive(Config, Debug)]
pub struct ModelConfig {
    pub backbone: BackboneConfig,
    pub neck: NeckConfig,
    pub head: HeadConfig
}

#[derive(Module, Debug)]
pub struct DetectionModel<B: Backend> {
    pub backbone: Backbone<B>,
    pub neck: Neck<B>,
    // P3, P4, P5　それぞれの特徴マップについて対応するヘッド
    pub head3: Head<B>,
    pub head4: Head<B>,
    pub head5: Head<B>,
}

impl DetectionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DetectionModel<B> {
        DetectionModel {
            backbone: self.backbone.init(device),
            neck: self.neck.init(device),
            head3: self.head.init(device),
            head4: self.head.init(device),
            head5: self.head.init(device),
        }
    }
}

pub struct DetectionOutput<B: Backend> {
    pub class_preds: Vec<Tensor<B, 4>>,
    pub bbox_preds: Vec<Tensor<B, 4>>,
}

impl<B: Backend> DetectionModel<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> DetectionOutput<B> {
        // Backboneで特徴抽出
        let features = self.backbone.forward(x);

        // NeckでFPN/PANを適用
        let fpn_output = self.neck.forward(features);

        // 各特徴マップに対して予測を行う
        let (class_pred3, bbox_pred3) = self.head3.forward(fpn_output.p3);
        let (class_pred4, bbox_pred4) = self.head4.forward(fpn_output.p4);
        let (class_pred5, bbox_pred5) = self.head5.forward(fpn_output.p5);

        DetectionOutput {
            class_preds: vec![class_pred3, class_pred4, class_pred5],
            bbox_preds: vec![bbox_pred3, bbox_pred4, bbox_pred5],
        }
    }

    // ★追加: 損失を計算するメソッド
    pub fn forward_detection(
        &self,
        images: Tensor<B, 4>,
        targets: &[Vec<BoundingBox>],
        num_classes: usize,
    ) -> DetectionLoss<B> {
        // まずはモデルの順伝播を実行して予測結果を得る
        let output = self.forward(images);
        let device = &output.class_preds[0].device();

        let mut total_class_loss = Tensor::<B, 1>::zeros([1], device);
        let mut total_bbox_loss = Tensor::<B, 1>::zeros([1], device);
        let mut num_positives = 0.0f32;

        // 簡単のため、最初の特徴マップレベル(P3)の予測のみ使用
        let class_pred_p3 = output.class_preds[0].clone();
        let bbox_pred_p3 = output.bbox_preds[0].clone();
        let [_batch_size, _num_classes, height, width] = class_pred_p3.dims();

        // バッチ内の各画像についてループ
        for (i, target_boxes) in targets.iter().enumerate() {
            // 正解ボックスがなければスキップ
            if let Some(gt_box) = target_boxes.first() {
                num_positives += 1.0;

                // --- ポジティブサンプルの特定 ---
                // 正解ボックスの中心座標をグリッドのインデックスに変換
                let gx = (gt_box.x * width as f32).floor() as usize;
                let gy = (gt_box.y * height as f32).floor() as usize;

                if gx < width && gy < height {
                    // --- 分類損失の計算 ---
                    // ポジティブな位置のクラス予測を取得
                    let pred_class = class_pred_p3.clone().slice([i..i + 1, 0..num_classes, gy..gy + 1, gx..gx + 1]).reshape([-1, num_classes]);
                    // 正解クラスのテンソルを作成
                    let target_class = Tensor::<B, 1, Int>::from_data(
                        TensorData::from([(gt_box.class_id as i32)]), device
                    );
                    // CrossEntropyLossを計算
                    let class_loss = CrossEntropyLossConfig::new().init(device).forward(pred_class, target_class);
                    total_class_loss = total_class_loss + class_loss;
                    
                    // --- BBox損失の計算 ---
                    // ポジティブな位置のBBox予測を取得
                    let pred_bbox = bbox_pred_p3.clone().slice([i..i + 1, 0..4, gy..gy + 1, gx..gx + 1]).reshape([-1, 4]);
                    // 正解BBoxのテンソルを作成
                    let target_bbox_vec = vec![gt_box.x, gt_box.y, gt_box.width, gt_box.height];
                    let target_bbox = Tensor::<B, 2>::from_data(
                        TensorData::from(target_bbox_vec).reshape([1, 4]), device
                    );
                    // L1Lossを計算
                    let bbox_loss = L1LossConfig::new().init(device).forward(pred_bbox, target_bbox);
                    total_bbox_loss = total_bbox_loss + bbox_loss;
                }
            }
        }
        
        // 損失をポジティブサンプルの数で正規化
        let class_loss = if num_positives > 0.0 { total_class_loss / num_positives } else { total_class_loss };
        let bbox_loss = if num_positives > 0.0 { total_bbox_loss / num_positives } else { total_bbox_loss };

        // 総合損失
        let total_loss = class_loss.clone() + bbox_loss.clone();

        DetectionLoss {
            total_loss,
            class_loss,
            bbox_loss,
        }
    }
}

// ★追加: 損失をまとめるための構造体
#[derive(Clone)]
pub struct DetectionLoss<B: Backend> {
    pub total_loss: Tensor<B, 1>,
    pub class_loss: Tensor<B, 1>,
    pub bbox_loss: Tensor<B, 1>,
}

// --- モデルの構成を定義するための構造体 ---
#[derive(Config, Debug)]
pub struct BackboneConfig {
    pub channels: [usize; 4],
    // デバイスを受け取り、モデルを初期化する。
    #[config(default = 3)]
    pub kernel_size: usize,
}

#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    // Block 1
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    pool1: MaxPool2d,

    // Block 2
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    pool2: MaxPool2d,

    // Block 3
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    pool3: MaxPool2d,

    relu: Relu,
}

// Config構造体に初期化ロジックを実装します
impl BackboneConfig {
    /// デバイスを受け取り、モデルを初期化する
    pub fn init<B: Backend>(&self, device: &B::Device) -> Backbone<B> {
        let k = self.kernel_size;
        // c[0]からc[2]までを使う
        let c = &self.channels; 

        let conv1 = Conv2dConfig::new([3, c[0]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn1 = BatchNormConfig::new(c[0]).init(device);
        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        
        let conv2 = Conv2dConfig::new([c[0], c[1]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn2 = BatchNormConfig::new(c[1]).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv3 = Conv2dConfig::new([c[1], c[2]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn3 = BatchNormConfig::new(c[2]).init(device);
        let pool3 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        Backbone {
            conv1, bn1, pool1,
            conv2, bn2, pool2,
            conv3, bn3, pool3,
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Backbone<B> {
    // forwardを修正し、3つの特徴マップを返す
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut outputs = Vec::new();

        let x = self.relu.forward(self.bn1.forward(self.conv1.forward(x)));
        let c3 = self.pool1.forward(x); // 112x112
        
        let x = self.relu.forward(self.bn2.forward(self.conv2.forward(c3.clone())));
        let c4 = self.pool2.forward(x); // 56x56

        let x = self.relu.forward(self.bn3.forward(self.conv3.forward(c4.clone())));
        let c5 = self.pool3.forward(x); // 28x28

        outputs.push(c3);
        outputs.push(c4);
        outputs.push(c5);
        
        outputs
    }
}


// <--- Neckモジュールの追加 ---
// Neckモジュール設定
#[derive(Config, Debug)]
pub struct NeckConfig {
    // FPNの各層で使われる共通のチャンネル数
    pub out_channels: usize,
    // バックボーンから受け取る特徴マップのチャンネル数
    // 例: [64, 128, 256] のように3つの特徴マップを想定
    pub in_channels: [usize; 3],
}

// Neckモジュール本体
#[derive(Module, Debug)]
pub struct Neck<B: Backend> {
    // --- トップダウン経路の層 ---
    lateral_conv0: Conv2d<B>,     // c5用
    upsample_conv0: Conv2d<B>,    // p4用平滑化
    
    lateral_conv1: Conv2d<B>,     // c4用
    upsample_conv1: Conv2d<B>,    // p3用平滑化
    
    lateral_conv2: Conv2d<B>,     // c3用 ★追加

    // --- ボトムアップ経路の層 ---
    downsample_conv0: Conv2d<B>,
    path_aug_conv0: Conv2d<B>,

    downsample_conv1: Conv2d<B>,
    path_aug_conv1: Conv2d<B>,
}

impl NeckConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Neck<B> {
        let c = &self.in_channels;
        let out = self.out_channels;

        // --- トップダウン経路の層を初期化 ---
        let lateral_conv0 = Conv2dConfig::new([c[2], out], [1, 1]).init(device);
        let lateral_conv1 = Conv2dConfig::new([c[1], out], [1, 1]).init(device);
        let lateral_conv2 = Conv2dConfig::new([c[0], out], [1, 1]).init(device); // c3用に追加

        // トップダウン後の平滑化用畳み込み
        let top_down_conv0 = Conv2dConfig::new([out, out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let top_down_conv1 = Conv2dConfig::new([out, out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        // --- ボトムアップ経路の層を初期化 ---
        let downsample_conv0 = Conv2dConfig::new([out, out], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)) // ★修正: SameからExplicit(1,1)に変更
            .init(device);
        let path_aug_conv0 = Conv2dConfig::new([out, out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        let downsample_conv1 = Conv2dConfig::new([out, out], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)) // ★修正: SameからExplicit(1,1)に変更
            .init(device);
        let path_aug_conv1 = Conv2dConfig::new([out, out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        Neck {
            lateral_conv0,
            upsample_conv0: top_down_conv0,
            lateral_conv1,
            upsample_conv1: top_down_conv1,
            lateral_conv2, // ★追加
            downsample_conv0,
            path_aug_conv0,
            downsample_conv1,
            path_aug_conv1,
        }
    }
}

// FPN/PANの出力を保持する構造体
pub struct FpnOutput<B: Backend> {
    pub p3: Tensor<B, 4>, // 浅い層 (高解像度)
    pub p4: Tensor<B, 4>, // 中間層
    pub p5: Tensor<B, 4>, // 深い層 (低解像度)
}

impl<B: Backend> Neck<B> {
    pub fn forward(&self, features: Vec<Tensor<B, 4>>) -> FpnOutput<B> {
        // デバッグ用のprintlnは一旦削除してOKです
        let c3 = &features[0];
        let c4 = &features[1];
        let c5 = &features[2];

        // --- トップダウン経路 (深い層から浅い層へ) ---

        let p5_td = self.lateral_conv0.forward(c5.clone());

        let [_b, _c, h4, w4] = c4.dims();
        let p5_upsampled = interpolate(p5_td.clone(), [h4, w4], InterpolateOptions::new(InterpolateMode::Nearest));
        let p4_lateral = self.lateral_conv1.forward(c4.clone());
        let p4_sum = p5_upsampled + p4_lateral;
        let p4_td = self.upsample_conv0.forward(p4_sum);

        let [_b, _c, h3, w3] = c3.dims();
        let p4_upsampled = interpolate(p4_td.clone(), [h3, w3], InterpolateOptions::new(InterpolateMode::Nearest));
        let p3_lateral = self.lateral_conv2.forward(c3.clone());
        let p3_sum = p4_upsampled + p3_lateral;
        let p3_td = self.upsample_conv1.forward(p3_sum);

        // --- ボトムアップ経路 (浅い層から深い層へ) ---

        let n3_out = p3_td;

        let n3_downsampled = self.downsample_conv0.forward(n3_out.clone());
        let n4_sum = n3_downsampled + p4_td.clone();
        let n4_out = self.path_aug_conv0.forward(n4_sum);
        
        let n4_downsampled = self.downsample_conv1.forward(n4_out.clone());
        let n5_sum = n4_downsampled + p5_td.clone();
        let n5_out = self.path_aug_conv1.forward(n5_sum);

        FpnOutput { p3: n3_out, p4: n4_out, p5: n5_out }
    }
}


// --- Headモジュールの追加 ---
#[derive(Config, Debug)]
pub struct HeadConfig {
    pub num_classes: usize,
    pub in_channels: usize,
}

#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    class_conv: Conv2d<B>,
    // バウンディングボックス予測用の畳み込み層
    bbox_conv: Conv2d<B>,
}

impl HeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Head<B> {
        // クラス予想層: [in_channels] -> [num_classes]
        let class_conv = Conv2dConfig::new([self.in_channels, self.num_classes], [1,1]).init(device);
        // BBox予想層: [in_channels] -> [4][x,y, w, h]
        let bbox_conv = Conv2dConfig::new([self.in_channels, 4], [1,1]).init(device);

        // Headに初期化した畳み込み層を渡す
        Head {class_conv, bbox_conv}
    }
}

// Headモジュールに機能を追加
impl<B: Backend> Head<B> {
    // 一つの特徴マップに対して予測を行う
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // クラス予想層
        let class_pred = self.class_conv.forward(x);
        // BBox予想層
        let bbox_pred = self.bbox_conv.forward(x);
        // タプルで返す
        (class_pred, bbox_pred)
    }
}
        