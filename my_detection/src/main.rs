mod data;
mod model;

use tempfile::Builder;
use crate::data::CocoDetectionDataset;
use crate::model::{BackboneConfig};
use burn::{
    backend::{wgpu::{Wgpu, WgpuDevice}, autodiff::Autodiff},
    tensor::Tensor,
};

// // データセットの読み込みが出来ているかを確かめる処理
// fn dataset_test() -> Result<(), Box<dyn std::error::Error>> {
//     // 1. データの読み込み
//     let image_dir = "quadrant-enumeration-disease/xrays";
//     let annotation_file = "quadrant-enumeration-disease/train_corrected.json";

//     // パスが有効かチェック
//     if !std::path::Path::new(image_dir).exists() {
//         eprintln!("エラー: 画像ディレクトリが見つかりません: {}", image_dir);
//         return Ok(());
//     }
//     if !std::path::Path::new(annotation_file).exists() {
//         eprintln!("エラー: アノテーションファイルが見つかりません: {}", annotation_file);
//         return Ok(());
//     }

//     // 2. データセットの読み込み
//     println!("\n--- データセットを読み込みます ---");
//     // モデルに合わせて検討の余地あり
//     let target_size = (224, 224);
//     let dataset = CocoDetectionDataset::new(
//         image_dir,
//         annotation_file,
//         target_size
//     )?;

//     println!("データセットの読み込み成功！");
//     println!("データセットのアイテム数: {}", dataset.len());
    

//     // 3. 最初のアイテムを取得して内容を検証
//     println!("\n--- 最初の5件のアイテムを取得します ---");
//     let num_samples = 5;
//     for i in 0..num_samples {
//         if i >= dataset.len() {
//             println!("データセットの終わりに達しました");
//             break;
//         }
//         println!("アイテム{}", i);
//         if let Some(item) = dataset.get(i) {
//             println!("  ✅ アイテム {} の取得に成功", i);
//             println!("    - リサイズ後の画像サイズ: {}x{}", item.width, item.height);
//             println!("    - バウンディングボックスの数: {}", item.bboxes.len());
            
//             // 最初のバウンディングボックスだけ表示してみる
//             if let Some(bbox) = item.bboxes.first() {
//                 println!("    - 最初のBBox: {:?}", bbox);
//             }
//         } else {
//             eprintln!("  ❌ エラー: アイテム {} の取得に失敗しました。", i);
//         }
//     }
    
//     Ok(())
// }

// モデルのバックボーンのテスト
type MyBackend = Autodiff<Wgpu>;

fn backbone_test() {
    let device = WgpuDevice::default();

    let config = BackboneConfig {
        channels: [32, 64, 128, 256],
        kernel_size: 3,
    };

    let backbone = config.init::<MyBackend>(&device);
    println!("モデルの初期化に成功しました。");
    println!("モデル構造: {:?}", backbone);

    let dummy_data: Tensor<MyBackend, 4> = Tensor::zeros([1, 3, 224, 224], &device);
    println!("\nダミー入力データの形状: {:?}", dummy_data.shape());

    let output = backbone.forward(dummy_data);
    println!("✅ 順伝播の実行に成功！");
    println!("出力テンソルの形状: {:?}", output.shape());
}

fn main() {
    backbone_test();
}