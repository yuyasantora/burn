use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::{Tensor, TensorData}
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::{Path, PathBuf}};


// COCOアノテーション形式に対応する
#[derive(Debug, Deserialize)]
pub struct CocoAnnotation {
    pub images: Vec<CocoImage>,
    pub annotations: Vec<CocoAnnotationItem>,
    pub categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize)]
pub struct CocoImage {
    pub id: u64,
    pub file_name: String,
    pub width: u64,
    pub height: u64,

}

#[derive(Debug, Deserialize)]
pub struct CocoAnnotationItem {
    pub id: u64,
    pub image_id: u64,
    pub category_id: u32,
    pub bbox: [f32; 4],
    pub area: f64
}

#[derive(Debug, Deserialize)]
pub struct CocoCategory {
    pub id: u64, 
    pub name: String,

}

// 検出用のデータ構造
#[derive(Clone, Debug)]
pub struct DetectionItem {
    pub image: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub bboxes: Vec<BoundingBox>,
}

#[derive(Clone, Debug)]
pub struct BoundingBox {
    // 全部正規化してる状態
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32, 
    pub class_id: usize
}

#[derive(Clone)]
pub struct DetectionBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Vec<Vec<BoundingBox>>,
}

// データセット実装
pub struct CocoDetectionDataset {
    image_dir: PathBuf,
    images: Vec<CocoImage>,
    annotation_map: HashMap<u64, Vec<CocoAnnotationItem>>,
    target_size: (u32, u32),
}

impl CocoDetectionDataset {
    pub fn new<P: AsRef<Path>>(
        image_dir: P,
        annotation_file: P,
        target_size: (u32, u32),

    ) -> Result<Self, Box<dyn std::error::Error>> {
        let annotation_context = std::fs::read_to_string(annotation_file)?;
        let coco_data: CocoAnnotation = serde_json::from_str(&annotation_context)?; // アノテーションデータをパース

        // アノテーションをimage_idでグループ化
        let mut annotation_map = HashMap::new();
        for ann in coco_data.annotations {
            annotation_map.entry(ann.image_id).or_insert(Vec::new()).push(ann);
        }

        Ok(Self {
            image_dir: image_dir.as_ref().to_path_buf(),
            images: coco_data.images,
            annotation_map,
            target_size
            
        })
    }

    fn load_image(&self, image_info: &CocoImage) -> Option<Vec<f32>> {
        let image_path = self.image_dir.join(&image_info.file_name);

        // image crateを使って画像を読み込む
        use image::imageops::FilterType;

        let img = image::open(image_path).ok()?;

        let resized_img = img.resize_exact(
            self.target_size.0,
            self.target_size.1,
            FilterType::Lanczos3,
        );
        let rgb_img = resized_img.to_rgb8();

        let mut image_vec =
            Vec::with_capacity((self.target_size.0 * self.target_size.1 * 3) as usize);
        for pixel in rgb_img.pixels() {
            image_vec.push(pixel[0] as f32 / 255.0); // R
            image_vec.push(pixel[1] as f32 / 255.0); // G
            image_vec.push(pixel[2] as f32 / 255.0); // B
        }

        Some(image_vec)
    }
}

impl Dataset<DetectionItem> for CocoDetectionDataset {
    fn get(&self, index: usize) -> Option<DetectionItem> {
        let image_info = &self.images[index];

        let image = self.load_image(image_info)?;

        let bboxes = self.annotation_map
            .get(&image_info.id)
            .map(|anns| {
                anns.iter()
                    .map(|ann| BoundingBox {
                        x: ann.bbox[0] / image_info.width as f32,
                        y: ann.bbox[1] / image_info.height as f32,
                        width: ann.bbox[2] / image_info.width as f32,
                        height: ann.bbox[3] / image_info.height as f32,
                        class_id: ann.category_id as usize,
                    })
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        Some(DetectionItem {
            image,
            width: self.target_size.0,
            height: self.target_size.1,
            bboxes,
        })
    }

    fn len(&self) -> usize {
        self.images.len()
    }
}


