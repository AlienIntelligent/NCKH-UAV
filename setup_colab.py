"""
NCKH-UAV: Setup cho Google Colab
==================================
Chạy cell này đầu tiên khi mở notebook trên Colab.
Tự động clone repo, cài dependencies, mount Drive nếu cần.

Sử dụng trong Colab:
  !python setup_colab.py
  # hoặc copy nội dung vào cell đầu tiên
"""
import subprocess
import sys
import os

def run(cmd, check=True):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-500:])  # Chỉ in 500 ký tự cuối
    if result.returncode != 0 and check:
        print(f"STDERR: {result.stderr[-300:]}")
    return result.returncode == 0

print("=" * 60)
print("  NCKH-UAV: SETUP GOOGLE COLAB")
print("=" * 60)

# 1. Kiểm tra GPU
print("\n[1/5] Kiểm tra GPU...")
run("nvidia-smi", check=False)

# 2. Cài dependencies
print("\n[2/5] Cài đặt dependencies...")
run(f"{sys.executable} -m pip install -q ultralytics scipy shapely tqdm pyyaml matplotlib")

# 3. Kiểm tra PyTorch + CUDA
print("\n[3/5] Kiểm tra PyTorch & CUDA...")
run(f'{sys.executable} -c "import torch; print(f\'PyTorch {{torch.__version__}}, CUDA {{torch.cuda.is_available()}}\')"')

# 4. Mount Google Drive (nếu muốn lưu kết quả)
print("\n[4/5] Mount Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Drive đã mount tại /content/drive/")
except ImportError:
    print("Không phải Colab, bỏ qua mount Drive.")

# 5. Sinh dummy data (nếu chưa có)
print("\n[5/5] Kiểm tra dummy data...")
if not os.path.isdir("dummy_data/DOTA_dummy"):
    run(f'{sys.executable} -c "from src.data.generate_dummy import generate_dummy_dataset; '
        f'generate_dummy_dataset(\'./dummy_data\', \'DOTA_dummy\', 50); '
        f'generate_dummy_dataset(\'./dummy_data\', \'DIOR_dummy\', 50)"')
else:
    print("Dummy data đã tồn tại, bỏ qua.")

print("\n" + "=" * 60)
print("  SETUP HOÀN TẤT!")
print("  Test: python main.py --mode all --dummy")
print("=" * 60)
