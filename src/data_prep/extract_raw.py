"""
Giải nén toàn bộ data thô từ data/DIOR/raw/ và data/DOTA/raw/.
Chạy 1 lần sau khi tải xong tất cả zip/tar files.

Sử dụng:
  python -m src.data_prep.extract_raw
  # hoặc
  python src/data_prep/extract_raw.py
"""
import os
import zipfile
import tarfile
import glob
from tqdm import tqdm


def extract_all_in_dir(raw_dir):
    """Giải nén tất cả .zip và .tar.gz trong một thư mục."""
    archives = glob.glob(os.path.join(raw_dir, "*.zip")) + \
               glob.glob(os.path.join(raw_dir, "*.tar.gz")) + \
               glob.glob(os.path.join(raw_dir, "*.tar"))

    if not archives:
        print(f"  Không tìm thấy archive nào trong {raw_dir}")
        return

    for arc_path in archives:
        arc_name = os.path.basename(arc_path)
        print(f"  Đang giải nén: {arc_name} ...")

        try:
            if arc_path.endswith('.zip'):
                with zipfile.ZipFile(arc_path, 'r') as zf:
                    zf.extractall(raw_dir)
            elif arc_path.endswith('.tar.gz') or arc_path.endswith('.tar'):
                with tarfile.open(arc_path, 'r:*') as tf:
                    tf.extractall(raw_dir)
            print(f"    ✅ Hoàn tất: {arc_name}")
        except Exception as e:
            print(f"    ❌ Lỗi giải nén {arc_name}: {e}")


def main():
    print("=" * 60)
    print("GIẢI NÉN DỮ LIỆU THÔ")
    print("=" * 60)

    # Kiểm tra có file .crdownload (đang tải) không
    for dataset in ["DIOR", "DOTA"]:
        raw_dir = os.path.join("data", dataset, "raw")
        crdownloads = glob.glob(os.path.join(raw_dir, "*.crdownload"))
        if crdownloads:
            print(f"⚠️  {dataset}: Phát hiện file đang tải dở:")
            for f in crdownloads:
                size_mb = os.path.getsize(f) / 1024 / 1024
                print(f"     {os.path.basename(f)} ({size_mb:.1f} MB)")
            print(f"     → Bỏ qua {dataset}, chờ tải xong rồi chạy lại.")
            continue

        if not os.path.isdir(raw_dir):
            print(f"⚠️  Thư mục {raw_dir} chưa tồn tại — bỏ qua.")
            continue

        print(f"\n📦 {dataset}:")
        extract_all_in_dir(raw_dir)

    # Liệt kê kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ SAU GIẢI NÉN:")
    for dataset in ["DIOR", "DOTA"]:
        raw_dir = os.path.join("data", dataset, "raw")
        if os.path.isdir(raw_dir):
            subdirs = [d for d in os.listdir(raw_dir)
                       if os.path.isdir(os.path.join(raw_dir, d))]
            files = [f for f in os.listdir(raw_dir)
                     if os.path.isfile(os.path.join(raw_dir, f))]
            print(f"  {dataset}/raw/: {len(subdirs)} thư mục, {len(files)} files")
            for sd in sorted(subdirs):
                count = len(os.listdir(os.path.join(raw_dir, sd)))
                print(f"    └── {sd}/ ({count} items)")


if __name__ == "__main__":
    main()
