"""
Xử lý DIOR dataset: VOC XML → YOLO format + Unified Mapping.
Chỉ giữ lại các class: airplane → 0, vehicle → 1.

Sử dụng:
  python -m src.data_prep.process_dior
"""
import xml.etree.ElementTree as ET
import os
import glob
import shutil
from tqdm import tqdm

# Unified Mapping cho DIOR
CLASS_MAPPING = {
    'airplane': 0,
    'vehicle': 1,
}


def main(input_img_dir="data/DIOR/raw/JPEGImages",
         input_xml_dir="data/DIOR/raw/Annotations",
         out_img_dir="data/DIOR/processed/images",
         out_lbl_dir="data/DIOR/processed/labels",
         use_symlink=False):
    """
    Convert DIOR VOC XML → YOLO format.

    Parameters
    ----------
    use_symlink : bool
        If True, symlink images instead of copying (saves disk on Kaggle).
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(input_xml_dir, "*.xml"))
    print(f"Tìm thấy {len(xml_files)} file nhãn DIOR.")

    converted = 0
    skipped_no_img = 0
    skipped_no_obj = 0

    for xml_path in tqdm(xml_files, desc="DIOR"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        # Some DIOR XMLs may not include extension
        if not os.path.splitext(filename)[1]:
            filename = filename + '.jpg'

        img_path = os.path.join(input_img_dir, filename)
        if not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        # Xử lý nhãn
        size_el = root.find('size')
        img_w = int(size_el.find('width').text)
        img_h = int(size_el.find('height').text)

        lbl_lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in CLASS_MAPPING:
                continue

            cls_id = CLASS_MAPPING[cls_name]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format (normalized)
            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            lbl_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if len(lbl_lines) == 0:
            skipped_no_obj += 1
            continue

        # Copy or symlink image
        dst_img = os.path.join(out_img_dir, filename)
        if not os.path.exists(dst_img):
            if use_symlink:
                os.symlink(os.path.abspath(img_path), dst_img)
            else:
                shutil.copy(img_path, dst_img)

        # Write YOLO label
        lbl_name = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(out_lbl_dir, lbl_name), 'w') as f:
            f.write("\n".join(lbl_lines) + "\n")
        converted += 1

    print(f"DIOR: {converted} converted, {skipped_no_img} no image, "
          f"{skipped_no_obj} no target object (airplane/vehicle)")
    return converted


if __name__ == "__main__":
    main()
