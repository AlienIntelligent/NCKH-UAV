"""
extract_and_cleanup.py — Giải nén data + XÓA ZIP ngay sau khi extract xong.
Giải quyết vấn đề hết dung lượng ổ đĩa.

CHỈ DÙNG STDLIB — chạy được ngay không cần pip install.

Chạy: python extract_and_cleanup.py
"""
import os
import sys
import zipfile
import shutil
import glob
import time


def disk_free_gb(path="."):
    """Kiểm tra dung lượng trống (GB)."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free / (1024**3)
    except:
        return -1


def extract_zip_to_flat(zip_path, dest_dir, extensions=None, delete_after=True):
    """
    Giải nén zip, đưa tất cả file vào dest_dir (flat, bỏ subfolder structure).
    Xóa zip ngay sau khi extract xong để tiết kiệm dung lượng.
    """
    name = os.path.basename(zip_path)
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    free_gb = disk_free_gb(os.path.dirname(zip_path))

    print(f"\n  [{name}] ({size_mb:.0f} MB) — Dung luong trong: {free_gb:.1f} GB")

    os.makedirs(dest_dir, exist_ok=True)
    start = time.time()

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            if extensions:
                members = [m for m in members if any(m.lower().endswith(e) for e in extensions)]

            total = len(members)
            print(f"    Dang giai nen {total} files...")

            extracted = 0
            for i, m in enumerate(members):
                fname = os.path.basename(m)
                if not fname:
                    continue
                dst = os.path.join(dest_dir, fname)
                if not os.path.exists(dst):
                    data = z.read(m)
                    with open(dst, 'wb') as f:
                        f.write(data)
                    extracted += 1

                if (i + 1) % 2000 == 0 or (i + 1) == total:
                    print(f"    ... {i+1}/{total} ({extracted} new)")

        elapsed = time.time() - start
        print(f"    OK — {extracted} files moi, {elapsed:.0f}s")

        if delete_after:
            os.remove(zip_path)
            freed = size_mb / 1024
            print(f"    Xoa {name} — Giai phong ~{freed:.1f} GB")
            free_after = disk_free_gb(dest_dir)
            print(f"    Dung luong trong: {free_after:.1f} GB")

        return True

    except Exception as e:
        print(f"    LOI: {e}")
        return False


def step_dior():
    """Xử lý DIOR: extract zips → merge vào đúng folder → xóa zip."""
    print("\n" + "=" * 60)
    print("  DIOR DATASET")
    print("=" * 60)

    raw = os.path.join("data", "DIOR", "raw")
    jpeg_dir = os.path.join(raw, "JPEGImages")
    ann_dir = os.path.join(raw, "Annotations")

    # 1. JPEGImages-trainval.zip → JPEGImages/
    trainval_zip = os.path.join(raw, "JPEGImages-trainval.zip")
    if os.path.exists(trainval_zip):
        extract_zip_to_flat(trainval_zip, jpeg_dir,
                            extensions=['.jpg', '.jpeg', '.png'],
                            delete_after=True)
    else:
        print(f"\n  JPEGImages-trainval.zip khong ton tai (da xoa hoac da extract)")

    # 2. JPEGImages-test.zip → JPEGImages/
    test_zip = os.path.join(raw, "JPEGImages-test.zip")
    if os.path.exists(test_zip):
        extract_zip_to_flat(test_zip, jpeg_dir,
                            extensions=['.jpg', '.jpeg', '.png'],
                            delete_after=True)
    else:
        print(f"\n  JPEGImages-test.zip khong ton tai (da xoa hoac da extract)")

    # 3. Annotations.zip → Annotations/ (flat)
    ann_zip = os.path.join(raw, "Annotations.zip")
    if os.path.exists(ann_zip):
        extract_zip_to_flat(ann_zip, ann_dir,
                            extensions=['.xml'],
                            delete_after=True)
    else:
        print(f"\n  Annotations.zip khong ton tai (da extract)")

    # 4. ImageSets.zip
    is_zip = os.path.join(raw, "ImageSets.zip")
    if os.path.exists(is_zip):
        is_dir = os.path.join(raw, "ImageSets")
        os.makedirs(is_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(is_zip, 'r') as z:
                z.extractall(is_dir)
            os.remove(is_zip)
            print(f"\n  ImageSets.zip extracted & deleted")
        except Exception as e:
            print(f"\n  ImageSets.zip loi: {e}")

    # Fix: Move Main/ nếu ở top-level
    main_top = os.path.join(raw, "Main")
    is_main = os.path.join(raw, "ImageSets", "Main")
    if os.path.isdir(main_top) and not os.path.isdir(is_main):
        os.makedirs(os.path.join(raw, "ImageSets"), exist_ok=True)
        shutil.move(main_top, is_main)

    # Cleanup leftover empty dirs
    for d in ["JPEGImages-trainval", "JPEGImages-test", "Segmentation", "Layout"]:
        p = os.path.join(raw, d)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

    # Also remove OBB from Annotations if exists
    obb = os.path.join(ann_dir, "Oriented Bounding Boxes")
    if os.path.isdir(obb):
        shutil.rmtree(obb, ignore_errors=True)
    hbb = os.path.join(ann_dir, "Horizontal Bounding Boxes")
    if os.path.isdir(hbb):
        # Move remaining XMLs up
        for f in glob.glob(os.path.join(hbb, "*.xml")):
            dst = os.path.join(ann_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.move(f, dst)
        shutil.rmtree(hbb, ignore_errors=True)

    # Verify
    print("\n  --- DIOR kiem tra ---")
    jpeg_count = len(glob.glob(os.path.join(jpeg_dir, "*.jpg")))
    ann_count = len(glob.glob(os.path.join(ann_dir, "*.xml")))
    print(f"    JPEGImages/: {jpeg_count} files [{'OK' if jpeg_count > 20000 else 'THIEU — can ~23463'}]")
    print(f"    Annotations/: {ann_count} files [{'OK' if ann_count > 20000 else 'THIEU — can ~23463'}]")


def step_dota():
    """Xử lý DOTA: extract zips → images/ & labelTxt/ → xóa zip."""
    print("\n" + "=" * 60)
    print("  DOTA DATASET")
    print("=" * 60)

    raw = os.path.join("data", "DOTA", "raw")
    compressed = os.path.join(raw, "Compressed")
    images_dir = os.path.join(raw, "images")
    labels_dir = os.path.join(raw, "labelTxt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 1. Labels
    label_src = os.path.join(compressed, "train", "labelTxt-v1.5", "DOTA-v1.5_train")
    if os.path.isdir(label_src):
        txts = glob.glob(os.path.join(label_src, "*.txt"))
        print(f"\n  Copy {len(txts)} labels...")
        for t in txts:
            dst = os.path.join(labels_dir, os.path.basename(t))
            if not os.path.exists(dst):
                shutil.copy2(t, dst)
        print("    OK")

    # Also check old location
    old_label_src = os.path.join(raw, "train", "labelTxt-v1.5", "DOTA-v1.5_train")
    if os.path.isdir(old_label_src) and not os.path.isdir(label_src):
        txts = glob.glob(os.path.join(old_label_src, "*.txt"))
        print(f"\n  Copy {len(txts)} labels (old path)...")
        for t in txts:
            dst = os.path.join(labels_dir, os.path.basename(t))
            if not os.path.exists(dst):
                shutil.copy2(t, dst)

    # 2. Train images — extract each zip then delete
    img_dir = os.path.join(compressed, "img")
    if os.path.isdir(img_dir):
        zips = sorted(glob.glob(os.path.join(img_dir, "*.zip")))
        print(f"\n  Tim thay {len(zips)} image zip files:")
        for zp in zips:
            extract_zip_to_flat(zp, images_dir,
                                extensions=['.png', '.jpg', '.jpeg', '.tif', '.bmp'],
                                delete_after=True)

    # 3. Val images (optional — xóa nếu cần dung lượng)
    val_dir = os.path.join(compressed, "val")
    if os.path.isdir(val_dir):
        val_zips = sorted(glob.glob(os.path.join(val_dir, "*.zip")))
        if val_zips:
            free = disk_free_gb(".")
            total_val_mb = sum(os.path.getsize(z) for z in val_zips) / 1024 / 1024
            print(f"\n  Val data: {len(val_zips)} zips ({total_val_mb:.0f} MB)")
            print(f"  Dung luong trong: {free:.1f} GB")

            if free < total_val_mb / 1024 * 2:
                print("  ⚠️ Khong du dung luong cho val — bo qua (chi can train cho NCKH)")
            else:
                val_img_dir = os.path.join(raw, "images_val")
                os.makedirs(val_img_dir, exist_ok=True)
                for zp in val_zips:
                    extract_zip_to_flat(zp, val_img_dir,
                                        extensions=['.png', '.jpg'],
                                        delete_after=True)

    # Cleanup compressed dir if empty
    if os.path.isdir(compressed):
        # Check if anything left
        remaining = []
        for root, dirs, files in os.walk(compressed):
            remaining.extend(files)
        if not remaining:
            shutil.rmtree(compressed, ignore_errors=True)
            print("\n  Xoa Compressed/ (rong)")

    # Verify
    print("\n  --- DOTA kiem tra ---")
    img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg'))])
    lbl_count = len(glob.glob(os.path.join(labels_dir, "*.txt")))
    print(f"    images/: {img_count} files [{'OK' if img_count > 500 else 'KIEM TRA LAI'}]")
    print(f"    labelTxt/: {lbl_count} files [{'OK' if lbl_count > 500 else 'KIEM TRA LAI'}]")


def main():
    print("=" * 60)
    print("  EXTRACT & CLEANUP — Giai nen + Xoa zip tiet kiem dung luong")
    print("=" * 60)
    print(f"  Dung luong trong: {disk_free_gb('.'):.1f} GB")
    print("  Chi dung stdlib — khong can pip install.")

    if not os.path.isdir("data"):
        print("\n  LOI: Khong tim thay 'data/'. Chay tu: cd D:\\ADVER-ATTAC\\NCKH-UAV")
        sys.exit(1)

    step_dior()
    step_dota()

    print("\n" + "=" * 60)
    print("  TONG KET")
    print("=" * 60)
    print(f"  Dung luong trong: {disk_free_gb('.'):.1f} GB")
    print("  Tiep theo: .\\fix_env.bat (cai packages)")
    print("  Roi: python main.py --mode all --dummy (test pipeline)")


if __name__ == "__main__":
    main()
