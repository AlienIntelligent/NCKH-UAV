#!/usr/bin/env bash
# =====================================================================
#  run_all.sh - chay toan bo pipeline NCKH-UAV tren Linux (GPU local)
#  Dat trong thu muc local/ cua du an.  Chay:  bash local/run_all.sh
# =====================================================================
set -e

# ve thu muc goc du an (cha cua local/)
cd "$(dirname "$0")/.."
echo "[run_all] Project root: $PWD"

# kich hoat venv neu co
if [ -f "venv/bin/activate" ]; then
    echo "[run_all] Activating venv..."
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "[run_all] WARNING: khong thay venv/ - dung Python he thong."
fi

# kiem tra torch CUDA
if ! python -c "import torch;print('[run_all] torch',torch.__version__,'CUDA',torch.cuda.is_available())"; then
    echo "[run_all] Chua co PyTorch. Cai torch ban CUDA truoc, vi du:"
    echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

echo "[run_all] Ensuring ultralytics / opencv / pyyaml / tqdm ..."
pip install -q ultralytics opencv-python pyyaml tqdm

echo "[run_all] Starting pipeline..."
python local/run_local.py "$@"

echo ""
echo "[run_all] DONE. Ket qua tan cong o: attack_results/"
