use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
    tensor::Tensor,
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub num_classes: usize,
    pub hidden_size: usize,
    #[config(default = 0.5)]
    pub dropout_rate: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            relu: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    relu: Relu,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> Model<B> {
    /// # Shapes
        // - Images [batch_size, height, width]
        // - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.pool.forward(x);

        let x = x.reshape([batch_size, 16 * 8 * 8]);

        let x = self.linear1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);

        let x = self.linear2.forward(x); // [batch_size, num_classes]

        x
    }
}