mod data;

use tempfile::Builder;
use crate::data::CocoDetectionDataset;
use burn::data::dataset::Dataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. テスト用の環境をセットアップ
    // 一時的なディレクトリを作成し、テスト用のファイルをここに格納する
    let temp_dir = Builder::new().prefix("coco-test").tempdir()?;
    let root_path = temp_dir.path();
    let image_dir = root_path.join("images");
    std::fs::create_dir(&image_dir)?;
    println!("テスト用の一時ディレクトリ: {:?}", root_path);

    // 2. ダミーアノテーションファイルを作成
    let annotation_path = root_path.join("annotations.json");
    let annotation_json = r#"
    {
        "images": [
            {
                "id": 1,
                "file_name": "test_image.png",
                "width": 100,
                "height": 80
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "area": 1200.0
            }
        ],
        "categories": [
            {"id": 5, "name": "test_cat"}
        ]
    }"#;
    std::fs::write(&annotation_path, annotation_json)?;
    println!("ダミーのアノテーションファイルを作成: {:?}", annotation_path);

    // 3. ダミー画像ファイルをサック性
    let image_path = image_dir.join("test_image.png");
    image::RgbImage::new(100, 80).save(&image_path)?;
    println!("ダミーの画像ファイルを作成: {:?}", image_path);

    // 4. データセットを作成
    println!("\n--- データセットを読み込みます ---");
    let target_size = (224, 224);
    let dataset = CocoDetectionDataset::new(
        &image_dir,
        &annotation_path,
        target_size,
    )?;

    println!("データセットの読み込み成功！");
    println!("データセットのアイテム数: {}", dataset.len());
    assert_eq!(dataset.len(), 1, "アイテム数が1であるべき");

    // 5. --- 最初のアイテムを取得して内容を検証 ---
    println!("\n--- 最初のアイテムを取得します ---");
    if let Some(item) = dataset.get(0) {
        println!("アイテム0の取得に成功しました。");
        println!("  リサイズ後の画像サイズ: {}x{}", item.width, item.height);
        println!("  画像データ(Vec<f32>)の長さ: {}", item.image.len());
        println!("  バウンディングボックスの数: {}", item.bboxes.len());
        
        assert_eq!(item.bboxes.len(), 1, "BBoxの数が1であるべき");

        if let Some(bbox) = item.bboxes.first() {
            println!("  最初のバウンディングボックス: {:?}", bbox);

            // 正しく正規化されているかチェック
            // 元BBox: [x,y,w,h] = [10, 20, 30, 40]
            // 元画像サイズ: 100x80
            // 正規化後: [10/100, 20/80, 30/100, 40/80]
            assert_eq!(bbox.x, 10.0 / 100.0);
            assert_eq!(bbox.y, 20.0 / 80.0);
            assert_eq!(bbox.width, 30.0 / 100.0);
            assert_eq!(bbox.height, 40.0 / 80.0);
            assert_eq!(bbox.class_id, 5);
            println!("\n✅ バウンディングボックスの正規化は正常です！");
        }
    } else {
        eprintln!("エラー: アイテム0の取得に失敗しました。");
    }
    
    // main関数の終わりで`temp_dir`が破棄されると、
    // 中のファイルも自動的に削除されます。
    Ok(())
}
