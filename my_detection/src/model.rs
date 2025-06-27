use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig,
        Relu,
    },
    prelude::*,
};

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
    
    // Block 2
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    pool1: MaxPool2d,

    // Block 3
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,

    // Block 4
    conv4: Conv2d<B>,
    bn4: BatchNorm<B, 2>,
    pool2: MaxPool2d,

    // 活性化関数はパラメータを持たないので、1つで使い回せる
    relu: Relu,
}

// Config構造体に初期化ロジックを実装します
impl BackboneConfig {
    /// デバイスを受け取り、モデルを初期化する
    pub fn init<B: Backend>(&self, device: &B::Device) -> Backbone<B> {
        let k = self.kernel_size;
        // c[0], c[1], c[2], c[3] にそれぞれチャンネル数が格納される
        let c = &self.channels;

        // 各層を個別に初期化
        let conv1 = Conv2dConfig::new([3, c[0]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn1 = BatchNormConfig::new(c[0]).init(device);
        
        let conv2 = Conv2dConfig::new([c[0], c[1]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn2 = BatchNormConfig::new(c[1]).init(device);
        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv3 = Conv2dConfig::new([c[1], c[2]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn3 = BatchNormConfig::new(c[2]).init(device);

        let conv4 = Conv2dConfig::new([c[2], c[3]], [k, k]).with_padding(burn::nn::PaddingConfig2d::Same).init(device);
        let bn4 = BatchNormConfig::new(c[3]).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        Backbone {
            conv1, bn1,
            conv2, bn2, pool1,
            conv3, bn3,
            conv4, bn4, pool2,
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Backbone<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // 各層を順番に手動で適用
        // Block 1
        let x = self.relu.forward(self.bn1.forward(self.conv1.forward(x)));

        // Block 2
        let x = self.relu.forward(self.bn2.forward(self.conv2.forward(x)));
        let x = self.pool1.forward(x);

        // Block 3
        let x = self.relu.forward(self.bn3.forward(self.conv3.forward(x)));

        // Block 4
        let x = self.relu.forward(self.bn4.forward(self.conv4.forward(x)));
        let x = self.pool2.forward(x);

        x
    }
}


        