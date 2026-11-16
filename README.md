# UAV Adversarial Camouflage Evaluation (NCKH-UAV)

> Đề tài đánh giá độ bền vững của mô hình phát hiện đối tượng trên UAV trước tấn công ngụy trang đa mảnh (DE Multi-Patch) trong điều kiện Black-box.

## 📁 Cấu trúc dự án

- `configs/`: File cấu hình YAML.
- `src/data_prep/`: Pipeline xử lý DOTA/DIOR (Slicing & Unified Mapping).
- `src/attack/`: Thuật toán Tiến hóa vi phân (DE) và Evaluator.

## 🚀 Hướng dẫn tái lập (Reproducibility)

1. **Tải Dataset:** [Link Kaggle/HuggingFace chứa DOTA/DIOR đã chuẩn hóa]
2. **Tải Pretrained Weights:** [Link Drive/HF chứa weights YOLOv8, RT-DETR]
3. **Cài đặt môi trường:** `pip install -r requirements.txt`
4. **Chạy tấn công:** `python main.py --config configs/base_config.yaml`
