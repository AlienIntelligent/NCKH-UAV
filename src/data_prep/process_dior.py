import xml.etree.ElementTree as ET
import os
import glob
import cv2
from tqdm import tqdm
import shutil

INPUT_IMG_DIR = "data/DIOR/raw/JPEGImages"
INPUT_XML_DIR = "data/DIOR/raw/Annotations"
OUT_IMG_DIR = "data/DIOR/processed/images"
OUT_LBL_DIR = "data/DIOR/processed/labels"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Unified Mapping cho DIOR
CLASS_MAPPING = {
    'airplane': 0,
    'vehicle': 1,
    # 'ship': 2 # Bỏ qua
}

xml_files = glob.glob(os.path.join(INPUT_XML_DIR, "*.xml"))
print(f"Tìm thấy {len(xml_files)} file nhãn DIOR.")

for xml_path in tqdm(xml_files):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    img_path = os.path.join(INPUT_IMG_DIR, filename)
    if not os.path.exists(img_path): continue
    
    # Copy ảnh sang thư mục processed (giữ nguyên kích thước gốc, Ultralytics sẽ tự resize khi train)
    shutil.copy(img_path, os.path.join(OUT_IMG_DIR, filename))
    
    # Xử lý nhãn
    img_h = int(root.find('size/height').text)
    img_w = int(root.find('size/width').text)
    
    lbl_name = os.path.splitext(filename)[0] + ".txt"
    lbl_lines = []
    
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_MAPPING: continue
        
        cls_id = CLASS_MAPPING[cls_name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        
        lbl_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
    with open(os.path.join(OUT_LBL_DIR, lbl_name), 'w') as f:
        f.write("\n".join(lbl_lines))

print("Hoàn tất xử lý DIOR!")