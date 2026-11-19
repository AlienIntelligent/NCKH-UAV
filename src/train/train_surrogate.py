"""
Train Surrogate Model (YOLOv8) trên dữ liệu đã chuẩn hóa.
Dùng cho cả dummy data (test luồng) và real data (DOTA processed).
"""
import os
import sys
import yaml
import argparse

# Thêm project root vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ultralytics import YOLO


def create_data_yaml(dataset_dir, output_path, class_names):
    """Tạo file data.yaml cho Ultralytics từ thư mục dataset đã xử lý."""
    data_cfg = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images',
        'val': 'images',  # Tạm dùng chung, sau chia train/val riêng
        'names': class_names
    }
    with open(output_path, 'w') as f:
        yaml.dump(data_cfg, f)
    print(f"Data YAML đã tạo: {output_path}")
    return output_path


def train_surrogate(config_path="configs/base_config.yaml", data_dir=None):
    """
    Train surrogate model theo config.
    - data_dir: override đường dẫn dataset (nếu None thì đọc từ config)
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = cfg['models']['surrogate']['name']
    epochs = cfg['models']['surrogate']['epochs']
    img_size = cfg['data']['img_size']
    batch_size = cfg['data']['batch_size']
    class_names = cfg['data']['classes']

    # Xác định dataset directory
    if data_dir is None:
        data_dir = cfg['data']['surrogate_dataset']

    # Tạo hoặc tìm data.yaml
    data_yaml = os.path.join(data_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        data_yaml = create_data_yaml(data_dir, data_yaml, class_names)

    # Khởi tạo và train
    print(f"{'='*60}")
    print(f"TRAIN SURROGATE MODEL")
    print(f"  Model  : {model_name}")
    print(f"  Dataset: {data_dir}")
    print(f"  Epochs : {epochs}")
    print(f"  ImgSize: {img_size}")
    print(f"  Batch  : {batch_size}")
    print(f"{'='*60}")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project="runs/detect",
        name="surrogate_train",
        exist_ok=True,
        verbose=True,
        device='0' if cfg['environment']['device'] == 'cuda' else 'cpu'
    )

    # Lưu best weights vào models/
    best_path = os.path.join("runs/detect/surrogate_train/weights/best.pt")
    if os.path.exists(best_path):
        os.makedirs("models", exist_ok=True)
        import shutil
        dst = "models/surrogate_best.pt"
        shutil.copy(best_path, dst)
        print(f"Best weights đã copy về: {dst}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Surrogate Model")
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--data", default=None, help="Override dataset directory")
    parser.add_argument("--dummy", action="store_true", help="Train trên dummy data (test luồng)")
    args = parser.parse_args()

    if args.dummy:
        args.data = "./dummy_data/DOTA_dummy"

    train_surrogate(config_path=args.config, data_dir=args.data)
