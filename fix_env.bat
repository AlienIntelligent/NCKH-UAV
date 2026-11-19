@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo  FIX MOI TRUONG (chay sau khi setup_local bi loi)
echo ============================================================
echo.

REM Tang timeout pip len rat cao (5 phut moi request)
set PIP_DEFAULT_TIMEOUT=300
set PIP_RETRIES=10

REM Kich hoat venv
call venv\Scripts\activate.bat

REM --- Fix 1: Go torch CUDA (ban nang, sai) va cai lai CPU ---
echo [1/4] Fix PyTorch...
echo     Go ban torch CUDA cu (nếu co)...
pip uninstall torch torchvision torchaudio -y >nul 2>&1

echo     Cai torch + torchvision CPU (dung --extra-index-url)...
pip install --default-timeout=300 torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo     [!] Loi lan 1. Thu lai lan 2...
    pip install --default-timeout=600 --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
    echo     [!] KHONG CAI DUOC TORCH. Mang qua cham.
    echo     Cach thu cong:
    echo       1. Vao https://download.pytorch.org/whl/cpu/torch/
    echo       2. Tai file: torch-2.5.1+cpu-cp310-cp310-win_amd64.whl
    echo       3. Vao https://download.pytorch.org/whl/cpu/torchvision/
    echo       4. Tai file: torchvision-0.20.1+cpu-cp310-cp310-win_amd64.whl
    echo       5. Chay: pip install ten_file_torch.whl ten_file_torchvision.whl
    echo.
)

REM --- Fix 2: Cai tung nhom package nho (tranh 1 cai loi la mat het) ---
echo [2/4] Cai dependencies (tung nhom)...

echo     numpy + scipy...
pip install --default-timeout=300 numpy scipy
echo     ultralytics...
pip install --default-timeout=300 ultralytics
echo     opencv-python...
pip install --default-timeout=300 opencv-python
echo     Pillow + shapely + tqdm + PyYAML + matplotlib...
pip install --default-timeout=300 Pillow shapely tqdm PyYAML matplotlib

REM --- Fix 3: Sinh dummy data ---
echo [3/4] Kiem tra dummy data...
if not exist "dummy_data\DOTA_dummy\data.yaml" (
    python -c "from src.data.generate_dummy import generate_dummy_dataset; generate_dummy_dataset('./dummy_data', 'DOTA_dummy', 50); generate_dummy_dataset('./dummy_data', 'DIOR_dummy', 50)"
) else (
    echo     Da co, bo qua.
)

REM --- Fix 4: Kiem tra tat ca ---
echo [4/4] Kiem tra moi truong...
echo.
echo ============================================================
echo  KET QUA:
echo ============================================================
python -c "import torch; print(f'  [OK] PyTorch   {torch.__version__}  CUDA={torch.cuda.is_available()}')" 2>nul || echo   [FAIL] torch
python -c "import torchvision; print(f'  [OK] TorchVis  {torchvision.__version__}')" 2>nul || echo   [FAIL] torchvision
python -c "import ultralytics; print(f'  [OK] YOLO      {ultralytics.__version__}')" 2>nul || echo   [FAIL] ultralytics
python -c "import scipy; print(f'  [OK] SciPy     {scipy.__version__}')" 2>nul || echo   [FAIL] scipy
python -c "import cv2; print(f'  [OK] OpenCV    {cv2.__version__}')" 2>nul || echo   [FAIL] opencv-python
python -c "import shapely; print(f'  [OK] Shapely   {shapely.__version__}')" 2>nul || echo   [FAIL] shapely
python -c "import numpy; print(f'  [OK] NumPy     {numpy.__version__}')" 2>nul || echo   [FAIL] numpy
python -c "import tqdm; print(f'  [OK] tqdm      {tqdm.__version__}')" 2>nul || echo   [FAIL] tqdm
python -c "import yaml; print(f'  [OK] PyYAML    {yaml.__version__}')" 2>nul || echo   [FAIL] PyYAML
python -c "import matplotlib; print(f'  [OK] Matplotlib {matplotlib.__version__}')" 2>nul || echo   [FAIL] matplotlib
echo.

REM Quick pipeline test
echo ============================================================
echo  TEST NHANH PIPELINE (khong can GPU):
echo ============================================================
python -c "from src.attack.evaluator import RobustnessEvaluator; e=RobustnessEvaluator(); r=e.evaluate([[105,105,295,295,0.95,0],[500,500,600,600,0.80,1]],[[500,500,600,600,0.80,1]],[100,100,300,300]); print(f'  Evaluator: ASR={r[\"ASR\"]}, AORR={r[\"AORR\"]}')" 2>nul && echo   [OK] Evaluator test passed || echo   [FAIL] Evaluator

echo.
echo ============================================================
echo  XONG! Neu tat ca [OK]:
echo    python main.py --mode all --dummy
echo  Neu co [FAIL], chay lai: pip install ten_package
echo ============================================================
pause
