"""
NCKH-UAV: Adversarial Camouflage Evaluation
============================================
Entry point chính — chạy toàn bộ pipeline hoặc từng bước riêng lẻ.

Sử dụng:
  # Chạy toàn bộ pipeline trên dummy data (test luồng)
  python main.py --mode all --dummy

  # Chỉ sinh dummy data
  python main.py --mode generate_dummy

  # Chỉ train surrogate
  python main.py --mode train --dummy

  # Chỉ chạy tấn công DE (trên mock model)
  python main.py --mode attack

  # Chạy pipeline thật (sau khi đã có data + trained model)
  python main.py --mode all --config configs/base_config.yaml
"""
import os
import sys
import argparse
import yaml

# Đảm bảo import được từ src/
sys.path.insert(0, os.path.dirname(__file__))


def step_generate_dummy():
    """Bước 0: Sinh dummy data để test luồng."""
    print("\n" + "=" * 60)
    print("BƯỚC 0: SINH DUMMY DATA")
    print("=" * 60)
    from src.data.generate_dummy import generate_dummy_dataset
    os.makedirs("dummy_data", exist_ok=True)
    generate_dummy_dataset("./dummy_data", "DOTA_dummy", num_images=50)
    generate_dummy_dataset("./dummy_data", "DIOR_dummy", num_images=50)
    print("Dummy data đã sẵn sàng!")


def step_extract_raw():
    """Bước 0.5: Giải nén data thô từ zip files."""
    print("\n" + "=" * 60)
    print("BƯỚC 0.5: GIẢI NÉN DỮ LIỆU THÔ")
    print("=" * 60)
    from src.data_prep.extract_raw import main as extract_raw
    extract_raw()


def step_process_data(config):
    """Bước 1: Giải nén + Xử lý data thật (DOTA slicing + DIOR mapping)."""
    print("\n" + "=" * 60)
    print("BƯỚC 1: XỬ LÝ DỮ LIỆU THÔ")
    print("=" * 60)

    # Giải nén trước
    step_extract_raw()

    # Paths sau khi giải nén
    dota_raw_img = "data/DOTA/raw/images"
    dota_raw_lbl = "data/DOTA/raw/labelTxt"
    dior_raw_img = "data/DIOR/raw/JPEGImages"
    dior_raw_xml = "data/DIOR/raw/Annotations"

    # DOTA: Sliding Window + OBB→HBB
    if os.path.isdir(dota_raw_img) and len(os.listdir(dota_raw_img)) > 0:
        print("\nĐang xử lý DOTA (Sliding Window + Unified Mapping)...")
        from src.data_prep.process_dota import main as process_dota
        process_dota(
            input_img_dir=dota_raw_img,
            input_lbl_dir=dota_raw_lbl,
            out_img_dir="data/processed/DOTA/images",
            out_lbl_dir="data/processed/DOTA/labels"
        )
    else:
        print(f"⚠️  Chưa thấy ảnh DOTA tại {dota_raw_img} — bỏ qua.")

    # DIOR: VOC XML → YOLO format
    if os.path.isdir(dior_raw_img) and len(os.listdir(dior_raw_img)) > 0:
        print("\nĐang xử lý DIOR (VOC → YOLO format + Unified Mapping)...")
        from src.data_prep.process_dior import main as process_dior
        process_dior(
            input_img_dir=dior_raw_img,
            input_xml_dir=dior_raw_xml,
            out_img_dir="data/processed/DIOR/images",
            out_lbl_dir="data/processed/DIOR/labels"
        )
    else:
        print(f"⚠️  Chưa thấy ảnh DIOR tại {dior_raw_img} — bỏ qua.")

    print("\n✅ Xử lý dữ liệu hoàn tất!")


def step_train_surrogate(config_path, use_dummy=False):
    """Bước 2: Train Surrogate Model."""
    print("\n" + "=" * 60)
    print("BƯỚC 2: TRAIN SURROGATE MODEL")
    print("=" * 60)
    from src.train.train_surrogate import train_surrogate
    data_dir = "./dummy_data/DOTA_dummy" if use_dummy else None
    train_surrogate(config_path=config_path, data_dir=data_dir)


def step_attack(config_path):
    """Bước 3: Chạy tấn công DE Multi-Patch."""
    print("\n" + "=" * 60)
    print("BƯỚC 3: CHẠY TẤN CÔNG DE MULTI-PATCH")
    print("=" * 60)
    from src.attack.de_optimizer import run_de_attack

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    patch_size = 32  # Sẽ đọc từ config sau
    best_patch = run_de_attack(patch_size=patch_size)
    print(f"Patch tối ưu: shape={best_patch.shape}, "
          f"mean={best_patch.mean():.1f}, std={best_patch.std():.1f}")
    return best_patch


def step_evaluate():
    """Bước 4: Đánh giá kết quả (ASR, AORR, FP)."""
    print("\n" + "=" * 60)
    print("BƯỚC 4: ĐÁNH GIÁ KẾT QUẢ")
    print("=" * 60)
    from src.attack.evaluator import RobustnessEvaluator
    evaluator = RobustnessEvaluator()

    # Demo trên dữ liệu giả lập (sẽ thay bằng inference thật)
    target_bbox = [100, 100, 300, 300]
    clean_preds = [
        [105, 105, 295, 295, 0.95, 0],  # Aircraft
        [500, 500, 600, 600, 0.80, 1]   # Vehicle (nền)
    ]
    patched_preds = [
        [500, 500, 600, 600, 0.80, 1]   # Chỉ còn vehicle
    ]

    results = evaluator.evaluate(clean_preds, patched_preds, target_bbox)
    print(f"Kết quả đánh giá:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="NCKH-UAV: Adversarial Camouflage Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", default="all",
                        choices=["all", "generate_dummy", "process", "train", "attack", "evaluate"],
                        help="Chạy bước nào (default: all)")
    parser.add_argument("--config", default="configs/base_config.yaml",
                        help="Path đến file config")
    parser.add_argument("--dummy", action="store_true",
                        help="Dùng dummy data thay vì data thật")
    args = parser.parse_args()

    print("=" * 60)
    print("  NCKH-UAV: ADVERSARIAL CAMOUFLAGE EVALUATION")
    print("  Config: " + args.config)
    print("  Mode  : " + args.mode)
    print("  Dummy : " + str(args.dummy))
    print("=" * 60)

    if args.mode in ("all", "generate_dummy"):
        if args.dummy or args.mode == "generate_dummy":
            step_generate_dummy()

    if args.mode in ("all", "process"):
        if not args.dummy:
            step_process_data(args.config)
        else:
            print("\n⏭️  Bỏ qua xử lý data (đang dùng dummy).")

    if args.mode in ("all", "train"):
        step_train_surrogate(args.config, use_dummy=args.dummy)

    if args.mode in ("all", "attack"):
        step_attack(args.config)

    if args.mode in ("all", "evaluate"):
        step_evaluate()

    print("\n" + "=" * 60)
    print("  PIPELINE HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
