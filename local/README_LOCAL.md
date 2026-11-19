# Chạy NCKH-UAV trên máy LOCAL (GPU)

Chạy pipeline tấn công patch đối kháng **theo đúng thiết kế hộp đen** của đề tài, không cần Kaggle và không giới hạn 12h.

**Thiết kế đúng (quan trọng):**
- **Surrogate** = YOLOv8n train trên **DOTA-only** → nhóm tối ưu (ta kiểm soát).
- **Target** = RT-DETR-l train trên **DIOR-only** → nhóm giữ kín (**khác kiến trúc + khác dữ liệu**).
- Patch tối ưu bằng DE trên surrogate/DOTA, rồi đo **transfer** trên target/DIOR (model chưa từng thấy).
- Luôn có **đối chứng patch ngẫu nhiên** cùng diện tích (bắt buộc — bài 2 cho thấy nhiễu ngẫu nhiên đã đạt ~44% ASR).

File trong `local/`:

| File | Vai trò |
|------|---------|
| `config_local.yaml` | Đường dẫn, tham số train/attack, cờ reuse |
| `run_local.py` | Orchestrator: `prepare → surrogate → target → attack` |
| `run_all.bat` / `run_all.sh` | Chạy 1-click (Windows / Linux) |
| `README_LOCAL.md` | File này |

---

## 1. Yêu cầu
- GPU NVIDIA + CUDA. VRAM: YOLOv8n@1024 ~6–8GB; RT-DETR-l@640 ~8–12GB (muốn @1024 nên ≥16GB).
- Python 3.9–3.11; **PyTorch bản CUDA** (bản CPU rất chậm).

Kiểm tra: `python -c "import torch;print(torch.cuda.is_available())"` phải in `True`.
Nếu `False`: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (đổi `cu121` theo CUDA máy).

---

## 2. Dữ liệu (đã có sẵn trên máy)
RAW đọc theo `config_local.yaml` (mặc định đã đúng):
```
data/DOTA/raw/dota_raw/{images/*.png, labelTxt/*.txt}
data/DIOR/raw/dior_raw/{JPEGImages/*.jpg, Annotations/*.xml}
```
Stage `prepare` sinh **HAI dataset tách riêng** (đây là điểm then chốt):
```
data/yolo_dota/{train,val}/{images,labels} + data.yaml   # chỉ DOTA → surrogate
data/yolo_dior/{train,val}/{images,labels} + data.yaml   # chỉ DIOR → target
```
Dùng hard-link (không tốn thêm disk, không cần admin).

---

## 3. Cài đặt
```bat
python -m venv venv & venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python pyyaml tqdm
```

---

## 4. Chạy
- **Windows:** double-click `local\run_all.bat`
- **Linux:** `bash local/run_all.sh`
- Từng stage: `python local\run_local.py --stages prepare` (hoặc `surrogate` / `target` / `attack`).

---

## 5. Thời gian ước lượng (GPU tầm trung)
| Stage | Làm gì | Thời gian |
|-------|--------|-----------|
| prepare | DOTA slice + DIOR convert + tách 2 dataset | ~20–40 phút (1 lần) |
| surrogate | YOLOv8n trên **DOTA** 50ep | ~1–2.5h |
| target | RT-DETR-l trên **DIOR** 50ep | ~3–5h |
| attack | 4 cấu hình (gồm đối chứng) + transfer | ~30–90 phút |

Local không giới hạn 12h → chạy thẳng cả 4 stage.

---

## 6. Tái sử dụng model (bỏ qua train)
```yaml
surrogate: {reuse: true, weights: "D:/.../runs/detect/surrogate_yolov8n_dota/weights/best.pt"}
target:    {reuse: true, weights: "D:/.../runs/detect/target_rtdetr_dior/weights/best.pt"}
run: {stages: ["attack"]}
```
> ⚠️ Chỉ reuse weights **đúng thiết kế**: surrogate phải là bản train DOTA-only, target là bản DIOR-only. **Không** dùng model train trên dữ liệu gộp (v5/v6) — sẽ phá điều kiện "khác dữ liệu".

---

## 7. Các cấu hình tấn công & đọc kết quả
Sửa trong `config_local.yaml` → `attack.configs`:

| Cấu hình | Ý nghĩa |
|----------|---------|
| `random_single` | **Đối chứng**: patch ngẫu nhiên 32×32, không tối ưu |
| `baseline_center` | DE, 1 patch 32×32, cố định tâm |
| `opt_position` | DE, 1 patch 32×32, **tối ưu vị trí** |
| `multi_patch_opt` | DE, 4 patch 16×16, **tối ưu vị trí** |

> **Cùng diện tích để so công bằng:** mọi cấu hình đều = **1024 px²** (1×32² = 4×16²). Nhờ vậy `random_single` là đối chứng diện-tích-khớp cho tất cả.

Mỗi cấu hình đánh giá 2 nơi: **surrogate (DOTA held-out)** và **transfer → target (DIOR held-out)**.

Output:
```
attack_results/
├─ summary.csv      # config, surr_ASR, surr_AORR, transfer_ASR, transfer_AORR, transfer_pct, is_random
├─ results.json
└─ <config>/ best_patch.npy, patch.png, positions.npy, convergence.png,
             examples_surrogate_dota/*.jpg, examples_target_dior/*.jpg
```

**Đọc bảng — tiêu chí thành công của đề tài:**
- Chỉ số CHÍNH = **transfer_ASR** (ASR trên model giữ kín DIOR).
- **Phải vượt rõ `random_single`** mới có ý nghĩa (đối chứng bắt buộc).
- So `baseline_center` ↔ `opt_position` ↔ `multi_patch_opt` → đóng góp của tối ưu vị trí / nhiều patch.
- Lưu ý thực tế: transfer thường **thấp hơn nhiều** so với đánh chính surrogate (bài 2: YOLOv3→Faster R-CNN chỉ 25.49%). Đây đúng là phần khó mà đề tài cần chứng minh.

---

## 8. Xử lý sự cố
| Lỗi | Khắc phục |
|-----|-----------|
| CUDA out of memory | Giảm `batch` (8→4→2) hoặc `imgsz` |
| torch `CUDA False` | Cài lại torch bản CUDA |
| Train chậm | Kiểm tra đang dùng GPU; bật `cache: true` |
| `weights không tồn tại` | Đặt `reuse: false` để train, hoặc trỏ đúng `weights` |
| Hết dung lượng khi prepare | Dọn `data/*/processed` cũ hoặc đổi `work_dir` |
| Muốn thử nhanh | Giảm `max_gen`, `n_attack`, bớt `configs` |

---

## 9. Khác gì so với v5/v6/v7?
- **v5/v6/v7 train cả 2 model trên DOTA+DIOR gộp** → chỉ khác kiến trúc, **mất** điều kiện "khác dữ liệu" → phép đo transfer không hợp lệ.
- Bản local này (và notebook v8) **tách DOTA↔DIOR đúng thiết kế** và thêm **đối chứng ngẫu nhiên** → kết quả mới dùng được cho báo cáo.
