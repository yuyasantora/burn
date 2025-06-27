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
    // 深い層 -> 浅い層
    lateral_conv0: Conv2d<B>,     // 3番目の特徴マップ用
    upsample_conv0: Conv2d<B>,    // 2番目の特徴マップと結合用
    
    lateral_conv1: Conv2d<B>,     // 2番目の特徴マップ用
    upsample_conv1: Conv2d<B>,    // 1番目の特徴マップと結合用

    // --- ボトムアップ経路の層 ---
    // 浅い層 -> 深い層
    downsample_conv0: Conv2d<B>,  // 1番目の特徴マップ用
    path_aug_conv0: Conv2d<B>,    // 2番目の特徴マップと結合用

    downsample_conv1: Conv2d<B>,  // 2番目の特徴マップ用
    path_aug_conv1: Conv2d<B>,    // 3番目の特徴マップと結合用
}

impl NeckConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Neck<B> {
        let c = &self.in_channels;
        let out = self.out_channels;

        // --- トップダウン経路の層を初期化 ---
        let lateral_conv0 = Conv2dConfig::new([c[2], out], [1, 1]).init(device);
        let upsample_conv0 = Conv2dConfig::new([out + c[1], out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        let lateral_conv1 = Conv2dConfig::new([out, out], [1, 1]).init(device);
        let upsample_conv1 = Conv2dConfig::new([out + c[0], out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        // --- ボトムアップ経路の層を初期化 ---
        let downsample_conv0 = Conv2dConfig::new([out, out], [3, 3]).with_stride([2, 2]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let path_aug_conv0 = Conv2dConfig::new([out + out, out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        let downsample_conv1 = Conv2dConfig::new([out, out], [3, 3]).with_stride([2, 2]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let path_aug_conv1 = Conv2dConfig::new([out + out, out], [3, 3]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);

        Neck {
            lateral_conv0, upsample_conv0,
            lateral_conv1, upsample_conv1,
            downsample_conv0, path_aug_conv0,
            downsample_conv1, path_aug_conv1,
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
        // バックボーンからの特徴マップを分割 (c3, c4, c5)
        // features[0]が最も浅く、[2]が最も深いと仮定
        let c3 = &features[0]; // 浅い
        let c4 = &features[1];
        let c5 = &features[2]; // 深い

        // --- トップダウン経路 (深い層から浅い層へ) ---

        // 1. c5からP5を作成
        let p5_latent = self.lateral_conv0.forward(c5.clone());

        // 2. P5をアップサンプルし、c4と結合してP4を作成
        let [_b, _c, h4, w4] = c4.dims();
        let p5_upsampled = interpolate(p5_latent.clone(), [h4, w4], InterpolateOptions::new(InterpolateMode::Nearest));
        let p4_latent = self.upsample_conv0.forward(Tensor::cat(vec![p5_upsampled, c4.clone()], 1));
        
        // 3. P4をアップサンプルし、c3と結合してP3を作成
        let [_b, _c, h3, w3] = c3.dims();
        let p4_upsampled = interpolate(p4_latent.clone(), [h3, w3], InterpolateOptions::new(InterpolateMode::Nearest));
        let p3_out = self.upsample_conv1.forward(Tensor::cat(vec![p4_upsampled, c3.clone()], 1));

        // --- ボトムアップ経路 (浅い層から深い層へ) ---

        // 4. P3からダウンサンプルし、P4と結合してN4を作成
        let n4_latent = self.downsample_conv0.forward(p3_out.clone());
        let p4_out = self.path_aug_conv0.forward(Tensor::cat(vec![n4_latent, p4_latent], 1));
        
        // 5. N4からダウンサンプルし、P5と結合してN5を作成
        let n5_latent = self.downsample_conv1.forward(p4_out.clone());
        let p5_out = self.path_aug_conv1.forward(Tensor::cat(vec![n5_latent, p5_latent], 1));

        FpnOutput { p3: p3_out, p4: p4_out, p5: p5_out }
    }
}



        