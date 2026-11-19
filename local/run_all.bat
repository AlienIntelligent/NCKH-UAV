@echo off
REM ====================================================================
REM  run_all.bat - chay toan bo pipeline NCKH-UAV tren Windows (GPU local)
REM  Dat file nay trong thu muc  local\  cua du an. Double-click hoac chay tu CMD.
REM ====================================================================
setlocal

REM --- ve thu muc goc du an (cha cua local\) ---
cd /d "%~dp0\.."
echo [run_all] Project root: %CD%

REM --- kich hoat venv neu co ---
if exist "venv\Scripts\activate.bat" (
    echo [run_all] Activating venv...
    call "venv\Scripts\activate.bat"
) else (
    echo [run_all] WARNING: khong thay venv\ - dung Python he thong.
)

REM --- kiem tra torch co CUDA khong ---
python -c "import torch;print('[run_all] torch',torch.__version__,'CUDA',torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [run_all] Chua co PyTorch. Cai torch ban CUDA truoc, vi du:
    echo     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)

REM --- cai goi phu (khong dung toi torch) ---
echo [run_all] Ensuring ultralytics / opencv / pyyaml / tqdm ...
pip install -q ultralytics opencv-python pyyaml tqdm

REM --- chay pipeline ---
echo [run_all] Starting pipeline...
python local\run_local.py %*
set RC=%errorlevel%

echo.
if "%RC%"=="0" (
    echo [run_all] DONE. Ket qua tan cong o: attack_results\
) else (
    echo [run_all] FAILED (exit %RC%). Xem log phia tren.
)
pause
endlocal
