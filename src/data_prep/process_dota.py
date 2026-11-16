import cv2
import os
import glob
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm
import json

# --- CẤU HÌNH ---
INPUT_IMG_DIR = "data/DOTA/raw/images"
INPUT_LBL_DIR = "data/DOTA/raw/labelTxt"
OUT_IMG_DIR = "data/DOTA/processed/images"
OUT_LBL_DIR = "data/DOTA/processed/labels"

SLICE_SIZE = 1024
OVERLAP_RATIO = 0.2  # 20% chồng lấn để tránh cắt đôi vật thể
STRIDE = int(SLICE_SIZE * (1 - OVERLAP_RATIO))

#Unified Mapping: Chỉ lấy Máy bay và Xe
# DOTA Classes: plane(0), small-vehicle(1), large-vehicle(2) -> Map về 0 và 1
CLASS_MAPPING = {
    'plane': 0,
    'small-vehicle': 1,
    'large-vehicle': 1,
    # 'helicopter': 0, # Có thể bật nếu muốn
    # 'ship': 2,       # Bỏ qua theo phạm vi chốt
}

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

def parse_dota_label(label_path):
    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            # Bỏ qua difficulty = 2 (quá khó/nhỏ) để làm sạch dữ liệu baseline
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            if difficulty == 2: continue
            
            cls_name = parts[8]
            if cls_name not in CLASS_MAPPING: continue
            
            # Tọa độ OBB (x1,y1 ... x4,y4)
            pts = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
            objs.append({'cls': CLASS_MAPPING[cls_name], 'poly': pts})
    return objs

img_paths = glob.glob(os.path.join(INPUT_IMG_DIR, "*.png"))
print(f"Tìm thấy {len(img_paths)} ảnh DOTA. Bắt đầu cắt...")

for img_path in tqdm(img_paths):
    img_name = os.path.basename(img_path).split('.')[0]
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    
    lbl_path = os.path.join(INPUT_LBL_DIR, img_name + ".txt")
    if not os.path.exists(lbl_path): continue
    objs = parse_dota_label(lbl_path)
    
    # Sliding Window
    for x in range(0, w, STRIDE):
        for y in range(0, h, STRIDE):
            # Hộp cửa sổ cắt
            x2, y2 = min(x + SLICE_SIZE, w), min(y + SLICE_SIZE, h)
            window_box = box(x, y, x2, y2)
            
            slice_labels = []
            for obj in objs:
                poly = Polygon(obj['poly'])
                if not poly.is_valid: poly = poly.buffer(0) # Fix lỗi self-intersection
                
                # Cắt đa giác theo cửa sổ
                inter = poly.intersection(window_box)
                if inter.is_empty: continue
                
                # Lấy khung bao ngang (HBB) của phần giao nhau
                minx, miny, maxx, maxy = inter.bounds
                
                # Lọc bỏ mảnh vỡ quá nhỏ ở mép ảnh (diện tích < 100px)
                if (maxx - minx) * (maxy - miny) < 100: continue
                
                # Chuẩn hóa về tọa độ YOLO (0-1) so với KÍCH THƯỚC SLICE (1024)
                # Lưu ý: x, y đang là tọa độ tuyệt đối trên ảnh gốc
                cx = ((minx + maxx) / 2 - x) / SLICE_SIZE
                cy = ((miny + maxy) / 2 - y) / SLICE_SIZE
                bw = (maxx - minx) / SLICE_SIZE
                bh = (maxy - miny) / SLICE_SIZE
                
                # Clamp về 0-1
                cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
                
                slice_labels.append(f"{obj['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            # Chỉ lưu ảnh nếu có chứa ít nhất 1 đối tượng quan tâm (hoặc lưu hết nếu muốn background)
            # Ở đây ta lưu hết để training có cả ảnh nền (negative samples)
            out_img_name = f"{img_name}_{x}_{y}.jpg"
            out_lbl_name = f"{img_name}_{x}_{y}.txt"
            
            # Cắt và lưu ảnh
            crop_img = img[y:y2, x:x2]
            if crop_img.shape[0] != SLICE_SIZE or crop_img.shape[1] != SLICE_SIZE:
                # Pad nếu ảnh ở mép không đủ 1024
                padded = np.zeros((SLICE_SIZE, SLICE_SIZE, 3), dtype=np.uint8)
                padded[:crop_img.shape[0], :crop_img.shape[1]] = crop_img
                crop_img = padded
                
            cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), crop_img)
            
            # Lưu nhãn
            with open(os.path.join(OUT_LBL_DIR, out_lbl_name), 'w') as f:
                f.write("\n".join(slice_labels))

print("Hoàn tất xử lý DOTA!")