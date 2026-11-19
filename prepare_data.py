"""
prepare_data.py — Giải nén & tổ chức lại data thô cho pipeline.
Chạy 1 LẦN sau khi tải xong tất cả data. CHỈ DÙNG STDLIB (không cần pip install).

Cấu trúc THỰC TẾ sau khi tải:
  data/DIOR/raw/
    Annotations.zip          → Annotations/Horizontal Bounding Boxes/*.xml
    ImageSets.zip            → Main/train.txt, val.txt, test.txt
    JPEGImages-trainval.zip  → JPEGImages-trainval/*.jpg
    JPEGImages-test.zip      → JPEGImages-test/*.jpg

  data/DOTA/raw/Compressed/
    img/part1-002.zip, part2-001.zip, ...  → ảnh .png
    train/labelTxt-v1.5/DOTA-v1.5_train/   → *.txt (đã giải nén)
    val/part1-001.zip, ...                  → ảnh val

Cấu trúc CẦN cho pipeline:
  data/DIOR/raw/JPEGImages/        (tất cả ảnh trainval + test)
  data/DIOR/raw/Annotations/       (tất cả XML — flat, không nested)
  data/DIOR/raw/ImageSets/Main/    (train.txt, val.txt, test.txt)

  data/DOTA/raw/images/            (tất cả ảnh train .png)
  data/DOTA/raw/labelTxt/          (tất cả label .txt)

Sử dụng:
  python prepare_data.py
"""
import os
import sys
import zipfile
import shutil
import glob
import time


def print_header(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def extract_zip_with_progress(zip_path, dest_dir):
    """Giải nén zip với progress đơn giản."""
    name = os.path.basename(zip_path)
    size_mb = os.path.getsize(zip_path) / 1024 / 1024

    print(f"  Giải nén: {name} ({size_mb:.0f} MB)...")
    start = time.time()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total = len(zf.namelist())
            zf.extractall(dest_dir)
            elapsed = time.time() - start
            print(f"    OK — {total} files, {elapsed:.1f}s")
            return True
    except Exception as e:
        print(f"    LOI: {e}")
        return False


def prepare_dior():
    """Giải nén và tổ chức lại DIOR dataset."""
    print_header("DIOR DATASET")

    raw_dir = os.path.join("data", "DIOR", "raw")
    if not os.path.isdir(raw_dir):
        print("  Khong tim thay data/DIOR/raw/ — bo qua.")
        return False

    # 1. Giải nén tất cả zip
    zips = glob.glob(os.path.join(raw_dir, "*.zip"))
    if not zips:
        print("  Khong co file .zip nao — co the da giai nen roi.")
    else:
        print(f"\n  Tim thay {len(zips)} file zip:")
        for zp in zips:
            extract_zip_with_progress(zp, raw_dir)

    # 2. Fix Annotations: Annotations/Horizontal Bounding Boxes/*.xml → Annotations/*.xml
    hbb_dir = os.path.join(raw_dir, "Annotations", "Horizontal Bounding Boxes")
    ann_dir = os.path.join(raw_dir, "Annotations")

    if os.path.isdir(hbb_dir):
        print("\n  Dang di chuyen Annotations/Horizontal Bounding Boxes/ → Annotations/ ...")
        xml_files = glob.glob(os.path.join(hbb_dir, "*.xml"))
        moved = 0
        for xf in xml_files:
            dest = os.path.join(ann_dir, os.path.basename(xf))
            if not os.path.exists(dest):
                shutil.move(xf, dest)
                moved += 1
        print(f"    Di chuyen {moved} XML files.")

        # Xóa thư mục rỗng
        obb_dir = os.path.join(raw_dir, "Annotations", "Oriented Bounding Boxes")
        if os.path.isdir(obb_dir):
            try:
                shutil.rmtree(obb_dir)
            except:
                pass
        if os.path.isdir(hbb_dir) and len(os.listdir(hbb_dir)) == 0:
            try:
                os.rmdir(hbb_dir)
            except:
                pass
    elif os.path.isdir(ann_dir):
        xml_count = len(glob.glob(os.path.join(ann_dir, "*.xml")))
        print(f"\n  Annotations/ da co {xml_count} XML files — OK.")
    else:
        print("\n  CANH BAO: Khong tim thay Annotations/")

    # 3. Fix JPEGImages: merge JPEGImages-trainval + JPEGImages-test → JPEGImages
    jpeg_dir = os.path.join(raw_dir, "JPEGImages")
    trainval_dir = os.path.join(raw_dir, "JPEGImages-trainval")
    test_dir = os.path.join(raw_dir, "JPEGImages-test")

    if os.path.isdir(jpeg_dir) and len(os.listdir(jpeg_dir)) > 100:
        print(f"\n  JPEGImages/ da co {len(os.listdir(jpeg_dir))} files — OK.")
    elif os.path.isdir(trainval_dir) or os.path.isdir(test_dir):
        os.makedirs(jpeg_dir, exist_ok=True)
        total_moved = 0

        for src_dir, label in [(trainval_dir, "trainval"), (test_dir, "test")]:
            if not os.path.isdir(src_dir):
                continue
            imgs = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"\n  Di chuyen {len(imgs)} anh tu {label} → JPEGImages/...")
            for img in imgs:
                src = os.path.join(src_dir, img)
                dst = os.path.join(jpeg_dir, img)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    total_moved += 1

        print(f"  Tong: {total_moved} anh da di chuyen vao JPEGImages/")
    else:
        print("\n  CANH BAO: Khong tim thay JPEGImages-trainval hoac JPEGImages-test")

    # 4. Fix ImageSets: Main/ ở top-level → ImageSets/Main/
    main_top = os.path.join(raw_dir, "Main")
    imagesets_dir = os.path.join(raw_dir, "ImageSets", "Main")

    if os.path.isdir(main_top) and not os.path.isdir(imagesets_dir):
        print("\n  Di chuyen Main/ → ImageSets/Main/...")
        os.makedirs(os.path.join(raw_dir, "ImageSets"), exist_ok=True)
        shutil.move(main_top, imagesets_dir)
        print("    OK")
    elif os.path.isdir(imagesets_dir):
        print(f"\n  ImageSets/Main/ da co — OK.")

    # Verification
    print("\n  --- Kiem tra DIOR ---")
    for name, min_count in [("Annotations", 1000), ("JPEGImages", 1000), ("ImageSets/Main", 2)]:
        d = os.path.join(raw_dir, name)
        if os.path.isdir(d):
            count = len(os.listdir(d))
            status = "OK" if count >= min_count else "IT QUA"
            print(f"    {name}/: {count} files [{status}]")
        else:
            print(f"    {name}/: KHONG TON TAI [LOI]")

    return True


def prepare_dota():
    """Giải nén và tổ chức lại DOTA dataset."""
    print_header("DOTA DATASET")

    raw_dir = os.path.join("data", "DOTA", "raw")
    compressed_dir = os.path.join(raw_dir, "Compressed")

    if not os.path.isdir(raw_dir):
        print("  Khong tim thay data/DOTA/raw/ — bo qua.")
        return False

    # Target directories
    images_dir = os.path.join(raw_dir, "images")
    labels_dir = os.path.join(raw_dir, "labelTxt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 1. Labels: Compressed/train/labelTxt-v1.5/DOTA-v1.5_train/*.txt → labelTxt/
    label_sources = [
        os.path.join(compressed_dir, "train", "labelTxt-v1.5", "DOTA-v1.5_train"),
        os.path.join(compressed_dir, "train", "labelTxt-v1.5"),
        os.path.join(raw_dir, "train", "labelTxt-v1.5", "DOTA-v1.5_train"),
        os.path.join(raw_dir, "train", "labelTxt-v1.5"),
    ]

    label_found = False
    for src in label_sources:
        if os.path.isdir(src):
            txts = glob.glob(os.path.join(src, "*.txt"))
            if txts:
                print(f"\n  Copy {len(txts)} labels tu {os.path.relpath(src)}...")
                for t in txts:
                    dst = os.path.join(labels_dir, os.path.basename(t))
                    if not os.path.exists(dst):
                        shutil.copy2(t, dst)
                label_found = True
                break

    # Also check for direct zip in raw/train/
    old_label_zip = os.path.join(raw_dir, "train", "labelTxt-v1.5", "DOTA-v1.5_train.zip")
    if not label_found and os.path.exists(old_label_zip):
        print(f"\n  Giai nen labels tu zip cu...")
        extract_zip_with_progress(old_label_zip, labels_dir)
        # May extract into subfolder
        sub = os.path.join(labels_dir, "DOTA-v1.5_train")
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                shutil.move(os.path.join(sub, f), os.path.join(labels_dir, f))
            os.rmdir(sub)
        label_found = True

    if not label_found:
        existing = glob.glob(os.path.join(labels_dir, "*.txt"))
        if existing:
            print(f"\n  labelTxt/ da co {len(existing)} files — OK.")
        else:
            print("\n  CANH BAO: Khong tim thay label files!")

    # 2. Images: Extract zips from Compressed/img/ → images/
    img_zip_dir = os.path.join(compressed_dir, "img")

    if os.path.isdir(img_zip_dir):
        img_zips = glob.glob(os.path.join(img_zip_dir, "*.zip"))
        if img_zips:
            print(f"\n  Tim thay {len(img_zips)} image zip files:")

            # Tạo temp dir để extract, rồi move png vào images/
            temp_dir = os.path.join(raw_dir, "_temp_extract")
            os.makedirs(temp_dir, exist_ok=True)

            for zp in sorted(img_zips):
                extract_zip_with_progress(zp, temp_dir)

            # Move all .png files from temp to images/
            png_count = 0
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f.lower().endswith('.png'):
                        src = os.path.join(root, f)
                        dst = os.path.join(images_dir, f)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                            png_count += 1

            print(f"  Tong: {png_count} anh .png da move vao images/")

            # Cleanup temp
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    # Check if images/ already has files
    existing_imgs = glob.glob(os.path.join(images_dir, "*.png"))
    if not existing_imgs:
        print("\n  CANH BAO: images/ van trong! Kiem tra lai cau truc zip.")
        # Try direct extract any zip in raw/
        raw_zips = glob.glob(os.path.join(raw_dir, "*.zip"))
        for zp in raw_zips:
            print(f"  Thu giai nen: {os.path.basename(zp)}...")
            extract_zip_with_progress(zp, raw_dir)

    # Verification
    print("\n  --- Kiem tra DOTA ---")
    img_count = len(glob.glob(os.path.join(images_dir, "*.png")))
    lbl_count = len(glob.glob(os.path.join(labels_dir, "*.txt")))
    print(f"    images/: {img_count} PNG files [{'OK' if img_count > 100 else 'KIEM TRA LAI'}]")
    print(f"    labelTxt/: {lbl_count} TXT files [{'OK' if lbl_count > 100 else 'KIEM TRA LAI'}]")

    return True


def main():
    print_header("CHUAN BI DU LIEU CHO PIPELINE NCKH-UAV")
    print("  Script nay chi dung stdlib — khong can pip install.")
    print("  Thoi gian: ~5-15 phut tuy toc do o cung.")

    # Kiểm tra đang ở đúng thư mục
    if not os.path.isdir("data"):
        print("\n  LOI: Khong tim thay thu muc 'data/'.")
        print("  Hay chay tu thu muc goc project: cd D:\\ADVER-ATTAC\\NCKH-UAV")
        sys.exit(1)

    prepare_dior()
    prepare_dota()

    print_header("TONG KET")
    print("  Kiem tra cac dong [OK] o tren.")
    print("  Neu tat ca OK, chay: python main.py --mode process")
    print("  De test nhanh pipeline: python main.py --mode all --dummy")


if __name__ == "__main__":
    main()
