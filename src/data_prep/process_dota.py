"""
Xử lý DOTA dataset: Sliding Window + OBB→HBB + Unified Mapping.
Cắt ảnh lớn thành tiles 1024x1024 với 20% overlap.

Sử dụng:
  python -m src.data_prep.process_dota
"""
import cv2
import os
import glob
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm

# --- CẤU HÌNH ---
SLICE_SIZE = 1024
OVERLAP_RATIO = 0.2
STRIDE = int(SLICE_SIZE * (1 - OVERLAP_RATIO))

# Unified Mapping: Chỉ lấy Máy bay và Xe
CLASS_MAPPING = {
    'plane': 0,
    'small-vehicle': 1,
    'large-vehicle': 1,
}


def parse_dota_label(label_path):
    """Đọc file nhãn DOTA (OBB format) và lọc theo CLASS_MAPPING."""
    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            if difficulty == 2:
                continue

            cls_name = parts[8]
            if cls_name not in CLASS_MAPPING:
                continue

            pts = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
            objs.append({'cls': CLASS_MAPPING[cls_name], 'poly': pts})
    return objs


def main(input_img_dir="data/DOTA/raw/images",
         input_lbl_dir="data/DOTA/raw/labelTxt",
         out_img_dir="data/DOTA/processed/images",
         out_lbl_dir="data/DOTA/processed/labels"):

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(input_img_dir, "*.png"))
    print(f"Tìm thấy {len(img_paths)} ảnh DOTA. Bắt đầu cắt (slice={SLICE_SIZE}, overlap={OVERLAP_RATIO})...")

    total_slices = 0
    for img_path in tqdm(img_paths, desc="DOTA Slicing"):
        img_name = os.path.basename(img_path).split('.')[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = os.path.join(input_lbl_dir, img_name + ".txt")
        if not os.path.exists(lbl_path):
            continue
        objs = parse_dota_label(lbl_path)

        # Sliding Window
        for x in range(0, w, STRIDE):
            for y in range(0, h, STRIDE):
                x2, y2 = min(x + SLICE_SIZE, w), min(y + SLICE_SIZE, h)
                window_box = box(x, y, x2, y2)

                slice_labels = []
                for obj in objs:
                    poly = Polygon(obj['poly'])
                    if not poly.is_valid:
                        poly = poly.buffer(0)

                    inter = poly.intersection(window_box)
                    if inter.is_empty:
                        continue

                    minx, miny, maxx, maxy = inter.bounds
                    if (maxx - minx) * (maxy - miny) < 100:
                        continue

                    cx = ((minx + maxx) / 2 - x) / SLICE_SIZE
                    cy = ((miny + maxy) / 2 - y) / SLICE_SIZE
                    bw = (maxx - minx) / SLICE_SIZE
                    bh = (maxy - miny) / SLICE_SIZE

                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    bw = max(0, min(1, bw))
                    bh = max(0, min(1, bh))

                    slice_labels.append(f"{obj['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                out_img_name = f"{img_name}_{x}_{y}.jpg"
                out_lbl_name = f"{img_name}_{x}_{y}.txt"

                crop_img = img[y:y2, x:x2]
                if crop_img.shape[0] != SLICE_SIZE or crop_img.shape[1] != SLICE_SIZE:
                    padded = np.zeros((SLICE_SIZE, SLICE_SIZE, 3), dtype=np.uint8)
                    padded[:crop_img.shape[0], :crop_img.shape[1]] = crop_img
                    crop_img = padded

                cv2.imwrite(os.path.join(out_img_dir, out_img_name), crop_img)
                with open(os.path.join(out_lbl_dir, out_lbl_name), 'w') as f:
                    f.write("\n".join(slice_labels))
                total_slices += 1

    print(f"Hoàn tất xử lý DOTA! Tổng số slices: {total_slices}")
    return total_slices


if __name__ == "__main__":
    main()
