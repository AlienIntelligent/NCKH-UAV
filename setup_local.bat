@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo  NCKH-UAV: Cai dat moi truong Local (Windows, CPU)
echo ============================================================
echo.

REM Cau hinh pip chong timeout
set PIP_DEFAULT_TIMEOUT=300
set PIP_RETRIES=10

REM --- Buoc 1: Tao venv ---
echo [1/5] Tao virtual environment...
if not exist "venv\Scripts\activate.bat" (
    python -m venv venv
    echo     Tao moi venv thanh cong.
) else (
    echo     venv da ton tai, bo qua.
)
call venv\Scripts\activate.bat

REM --- Buoc 2: Nang cap pip ---
echo [2/5] Nang cap pip...
python -m pip install --upgrade pip

REM --- Buoc 3: Cai PyTorch CPU ---
echo [3/5] Cai PyTorch + TorchVision (CPU only)...
echo     (File lon ~140MB, co the mat vai phut)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo     [!] Loi tai torch. Thu lai voi timeout cao hon...
    pip install --default-timeout=600 torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
    echo     [!] Van loi. Hay thu: pip install torch torchvision
    echo     Hoac tai .whl thu cong tu https://download.pytorch.org/whl/cpu
    echo     Roi: pip install ten_file.whl
    echo.
)

REM --- Buoc 4: Cai cac package con lai (tung nhom nho) ---
echo [4/5] Cai dependencies...

echo     [4a] numpy, scipy...
pip install numpy scipy
if errorlevel 1 echo     [!] Loi cai numpy/scipy, thu lai: pip install numpy scipy

echo     [4b] ultralytics...
pip install ultralytics
if errorlevel 1 echo     [!] Loi cai ultralytics, thu lai: pip install ultralytics

echo     [4c] opencv, Pillow, shapely...
pip install opencv-python Pillow shapely
if errorlevel 1 echo     [!] Loi cai opencv/Pillow/shapely, thu lai tung cai

echo     [4d] tqdm, PyYAML, matplotlib...
pip install tqdm PyYAML matplotlib
if errorlevel 1 echo     [!] Loi cai tqdm/PyYAML/matplotlib

REM --- Buoc 5: Sinh dummy data ---
echo [5/5] Sinh dummy data de test...
if not exist "dummy_data\DOTA_dummy\data.yaml" (
    python -c "from src.data.generate_dummy import generate_dummy_dataset; generate_dummy_dataset('./dummy_data', 'DOTA_dummy', 50); generate_dummy_dataset('./dummy_data', 'DIOR_dummy', 50)"
    echo     Dummy data da san sang!
) else (
    echo     Dummy data da ton tai, bo qua.
)

REM --- Kiem tra ---
echo.
echo ============================================================
echo  KIEM TRA MOI TRUONG:
echo ============================================================
python -c "import torch; print(f'  PyTorch  : {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import ultralytics; print(f'  YOLO     : {ultralytics.__version__}')"
python -c "import scipy; print(f'  SciPy    : {scipy.__version__}')"
python -c "import cv2; print(f'  OpenCV   : {cv2.__version__}')"
python -c "import shapely; print(f'  Shapely  : {shapely.__version__}')"
python -c "import numpy; print(f'  NumPy    : {numpy.__version__}')"
echo.
echo ============================================================
echo  XONG! Test luong: python main.py --mode all --dummy
echo ============================================================
pause
