# NCKH-UAV — Đánh giá độ bền vững của bộ phát hiện đối tượng trên ảnh UAV trước tấn công đối kháng dạng ngụy trang (DE, hộp đen)

> **Tên đề tài (đã thống nhất với giảng viên):** *Nghiên cứu các phương pháp tấn công đối kháng dạng ngụy trang đối với các mô hình xử lý ảnh trên UAV.*
> Tên chính thức còn mở, sẽ chốt khi rõ hướng cuối cùng.

Repo này chứa mã nguồn, notebook thực nghiệm và tài liệu của đề tài NCKH. Đề tài đặt câu hỏi: **nếu đối phương dùng UAV với một bộ phát hiện đối tượng mà ta hoàn toàn không biết (kiến trúc, dữ liệu huấn luyện, không truy vấn được), thì một vài mảng ngụy trang nhỏ đặt lên máy bay/xe có đủ làm chúng "biến mất" khỏi bộ phát hiện không?**

---

## Mục lục

1. [Bài toán và phạm vi](#1-bài-toán-và-phạm-vi)
2. [Phương pháp](#2-phương-pháp)
3. [Tiến độ đã hoàn thành](#3-tiến-độ-đã-hoàn-thành)
4. [Kết quả thực nghiệm hiện có](#4-kết-quả-thực-nghiệm-hiện-có)
5. [Dataset (link tải)](#5-dataset-link-tải)
6. [Cấu trúc repo](#6-cấu-trúc-repo)
7. [Cài đặt và chạy lại](#7-cài-đặt-và-chạy-lại)
8. [Hạn chế hiện tại](#8-hạn-chế-hiện-tại)
9. [Hướng tiếp theo](#9-hướng-tiếp-theo)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Bài toán và phạm vi

### Mô hình mối đe dọa (threat model)

| Khía cạnh | Giả định |
|---|---|
| Kiến trúc, trọng số model đối phương | Không biết |
| Dataset huấn luyện của đối phương | Không biết |
| Truy vấn model đối phương | Không được |
| Kiến thức chung | Biết đó là model phát hiện đối tượng trên ảnh quang học chụp từ trên cao |
| Khả năng của ta | Thiết kế và đặt patch trên/quanh đối tượng; không can thiệp vào ảnh hay hệ thống của đối phương |

Hệ quả: đề tài đi theo hướng **tấn công chuyển giao (transfer attack)**. Patch được tối ưu trên model thay thế (*surrogate*) do ta tự huấn luyện, rồi kiểm chứng trên model "đối phương giả định" (*target*) **khác cả kiến trúc lẫn dữ liệu huấn luyện**, không dùng khi tối ưu. Differential Evolution (DE) là bộ tối ưu **không cần gradient**.

### Phạm vi đã chốt

- **Đối tượng cốt lõi:** máy bay (đỗ trên mặt đất) và xe (xe nhỏ + xe lớn gộp một lớp). Người (bộ binh) và mô phỏng 3D Blender là hướng mở rộng.
- **Dạng tấn công:** ngụy trang đối kháng làm đối tượng **biến mất** (vanishing); không xét phân loại sai hay lệch khung.
- **Miền thực nghiệm:** thực nghiệm số trên ảnh; không bay UAV thật.
- **Hai lớp thống nhất (Unified Mapping):** `0 = aircraft`, `1 = vehicle`.

### Tiêu chí thành công ở giai đoạn hiện tại

Dựng được baseline và chứng minh phương án kết hợp "nhiều patch nhỏ + DE" **khả thi** trên nhóm model giữ kín; ngưỡng định lượng cụ thể đặt ở bước sau. Các baseline cần vượt:

1. DE một patch (Tang et al. 2024)
2. Gradient + patch lớn (Du et al. 2022)
3. Một patch lớn so với nhiều patch nhỏ (cùng tổng diện tích)
4. Patch ngẫu nhiên cùng diện tích (đối chứng bắt buộc)

Chỉ số **chính** là **ASR trên nhóm giữ kín (transfer)**; "vượt" nghĩa là tốt hơn rõ ở chỉ số chính và không kém hơn rõ rệt ở các chỉ số còn lại.

---

## 2. Phương pháp

```
   DOTA (raw) ─► chuẩn hoá ─► yolo_dota ─► Huấn luyện SURROGATE  (YOLOv8n, imgsz 1024)
                                                   │
                                                   ▼
                              Tối ưu patch bằng DE (hộp đen với surrogate,
                              không gradient; tối ưu NỘI DUNG + VỊ TRÍ patch)
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                                                      ▼
         Đánh giá trên SURROGATE (DOTA, held-out)         Đánh giá CHUYỂN GIAO trên TARGET
                        │                                   (RT-DETR-l huấn luyện trên DIOR —
                        │                                    khác kiến trúc + khác dữ liệu)
                        ▼                                                      ▲
            ASR · AORR · New FP          DIOR (raw) ─► chuẩn hoá ─► yolo_dior ─┘

   Đối chứng bắt buộc: patch NGẪU NHIÊN cùng diện tích (không tối ưu).
```

### 2.1 Chuẩn hoá dữ liệu

- **DOTA** (`src/data_prep/process_dota.py`): cắt ảnh lớn thành tile **1024×1024, overlap 20%**; đổi khung xoay (OBB) thành khung ngang (HBB); chỉ giữ `plane → 0`, `small-vehicle`/`large-vehicle → 1`; bỏ nhãn `difficult = 2`.
- **DIOR** (`src/data_prep/process_dior.py`): chuyển VOC XML → YOLO; chỉ giữ `airplane → 0`, `vehicle → 1`; hỗ trợ `use_symlink=True` (Kaggle) / copy (Windows local).
- **Tách HAI dataset riêng** (điểm then chốt của thiết kế hộp đen), mỗi dataset chia train/val **80/20**, `seed = 42`:
  - `yolo_dota/` — **chỉ DOTA** → huấn luyện surrogate + tập tối ưu/đánh giá surrogate.
  - `yolo_dior/` — **chỉ DIOR** → huấn luyện target + tập đánh giá chuyển giao.

  > Nhờ tách như vậy, target khác surrogate **cả kiến trúc lẫn dữ liệu** — đúng điều kiện "không biết dữ liệu huấn luyện của đối phương". (Các notebook v5–v7 trước đây gộp DOTA+DIOR nên chỉ khác kiến trúc; v8 sửa lại.)

### 2.2 Model thay thế (surrogate) và model đích (target)

| Vai trò | Model | Dữ liệu | Ghi chú |
|---|---|---|---|
| Surrogate | YOLOv8n (Ultralytics) | **DOTA-only** | `imgsz=1024`; nhóm tối ưu |
| Target (giả định đối phương) | RT-DETR-l | **DIOR-only** | transformer, khác họ kiến trúc; `imgsz=640` vì nặng; nhóm giữ kín |

### 2.3 Tấn công DE (`src/attack/de_attack_real.py`)

Lớp `DEAttackReal` cài đặt DE với **quần thể cố định** (không dùng `scipy.optimize.differential_evolution` vì quần thể của scipy tăng theo số chiều, quá nặng với patch 32×32×3 = 3.072 chiều). Chiến lược DE/best/1/bin.

- **Biến tối ưu:**
  - *Nội dung patch* — các điểm ảnh, giá trị [0, 255]; và
  - *(tuỳ chọn) Vị trí patch* — khi `optimize_positions=True`, DE tối ưu thêm toạ độ `(ox, oy)` của mỗi mini-patch bên trong bbox (toạ độ dạng tỉ lệ 0–1 so với bbox, nên **không phụ thuộc model** → dùng lại được cho đánh giá chuyển giao).
- Tham số mặc định: patch 32×32, `pop = 20`, `max_gen = 50`, `F = 0.8`, `CR = 0.7`.
- Hàm fitness tổng hợp:

  `L = w_obj·L_obj + w_fp·L_fp + w_tv·L_tv + w_nps·L_nps` với `w = (1.0, 0.3, 0.01, 0.001)`

  - `L_obj`: độ tin cậy còn lại của đối tượng mục tiêu — càng thấp càng tốt;
  - `L_fp`: phạt **báo động giả sinh thêm** (box dự đoán không khớp đối tượng sạch nào);
  - `L_tv`: Total Variation — ép patch mượt;
  - `L_nps`: bản đơn giản của Non-Printability Score (phạt phương sai màu).
- Đặt patch (`apply_patch`):
  - *Khi tối ưu vị trí*: đặt `num_patches` bản sao của patch tại các vị trí **DE học được** trong bbox.
  - *Khi không* (`positions=None`): `num_patches = 1` → một patch ở tâm bbox; `num_patches > 1` → nhiều mini-patch cố định ở tâm và các góc bbox (hành vi cũ, giữ để so sánh).
- Đánh giá: `evaluate_patch`, có tham số `exclude` để **tập held-out không trùng ảnh** với tập tối ưu.

### 2.4 Chỉ số đánh giá

| Chỉ số | Ý nghĩa |
|---|---|
| **ASR** (Attack Success Rate) | Tỉ lệ đối tượng bị làm biến mất (không còn box khớp `IoU ≥ match_iou` với đối tượng sạch). Chỉ số chính |
| **AORR** | Mức giảm confidence trung bình của đối tượng (theo Du et al. 2022) |
| **New FP** | Số box dự đoán **không khớp** đối tượng sạch nào (báo động giả thật sự, không phải hiệu số đếm) |

Ngưỡng dùng chung: `conf = 0.25`; **`match_iou = 0.5`** để khớp dự đoán với đối tượng (dùng **nhất quán** cho cả fitness lẫn evaluate), tách khỏi IoU của NMS trong detector.

### 2.5 Các cấu hình thí nghiệm (cùng tổng diện tích để so công bằng)

| Cấu hình | Patch | Tối ưu vị trí | Ghi chú |
|---|---|---|---|
| `random_single` | 1 × 32×32 | — | **Đối chứng** ngẫu nhiên, không tối ưu |
| `baseline_center` | 1 × 32×32 | Không | Patch đơn, cố định tâm |
| `opt_position` | 1 × 32×32 | Có | Patch đơn, tối ưu vị trí |
| `multi_patch_opt` | 4 × 16×16 | Có | Nhiều mini-patch, tối ưu vị trí |

Mọi cấu hình đều **= 1024 px²** (1×32² = 4×16²) → `random_single` là đối chứng diện-tích-khớp cho tất cả. Mỗi cấu hình đánh giá trên **surrogate (DOTA held-out)** và **transfer (DIOR held-out)**.

---

## 3. Tiến độ đã hoàn thành

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0 | Chốt bài toán: threat model hộp đen, phạm vi, tiêu chí | ✅ Xong |
| 1 | Tổng quan tài liệu 3 bài báo + công trình liên quan (ColorFD, SDMPA, AP-PA, FCA…) | ✅ Xong |
| 2 | Môi trường, chuẩn hoá DOTA/DIOR, **tách nhóm tối ưu (DOTA) / giữ kín (DIOR)** | ✅ Xong (code v8/local) |
| 3 | Chạy dữ liệu thật trên Kaggle, huấn luyện surrogate, DE mock attack | ✅ Xong (v3) |
| 4 | DE tấn công thật trên YOLOv8n; thêm **tối ưu vị trí patch** + **đối chứng ngẫu nhiên** | ✅ Code xong (v8), 🟡 chờ chạy đủ cấu hình |
| 5 | Thực nghiệm số + **đánh giá chuyển giao sang nhóm giữ kín** | 🟡 Code sẵn (v8/local), **chưa chạy ra số transfer** |
| 6 | Mở rộng (Blender, lớp người) | ⬜ Chưa (chỉ nếu còn thời gian) |
| 7 | Viết báo cáo NCKH, slide, demo | ⬜ Chưa làm |

Tài liệu kế hoạch tổng hợp: [`docs/NCKH_TongHop_Full.docx`](docs/NCKH_TongHop_Full.docx).

### Lịch sử các phiên bản notebook (Kaggle)

| Notebook | Nội dung chính |
|---|---|
| `NCKH_UAV_Kaggle_v4.ipynb` | DOTA-only (DIOR lỗi đường dẫn); DE tấn công thật lần đầu |
| `NCKH_UAV_Kaggle_v5.ipynb` | Sửa DIOR (XML→YOLO); huấn luyện trên DOTA + DIOR gộp |
| `NCKH_UAV_Kaggle_v6.ipynb` | Thêm RT-DETR-l làm target; đánh giá chuyển giao (dữ liệu vẫn gộp) |
| `NCKH_UAV_Kaggle_v7.ipynb` | Tái sử dụng weights (tránh timeout) + **tối ưu vị trí patch** |
| `NCKH_UAV_Kaggle_v8.ipynb` | **Tách DOTA (surrogate) / DIOR (target)** đúng thiết kế + **đối chứng ngẫu nhiên**; cấu hình cùng diện tích |

Ngoài notebook Kaggle, repo có **pipeline chạy local trên GPU** (`local/`) với cùng logic — xem mục 7.

---

## 4. Kết quả thực nghiệm hiện có

> Số liệu lấy từ các lần chạy trên Kaggle (2× T4). Output notebook đã xoá khi lưu repo cho nhẹ; đối chiếu lại log gốc khi viết báo cáo.
>
> ⚠️ Các số dưới đây (v4/v5) là **đánh giá tự thân surrogate** theo thiết kế **cũ (gộp dữ liệu, 1 patch cố định, metric cũ)**. Chúng **không** phải kết quả chuyển giao và **không so trực tiếp** được với số của v8 (tách DOTA/DIOR, có tối ưu vị trí, `match_iou`/FP đã sửa). Giữ lại để tham khảo lịch sử.

### 4.1 Surrogate YOLOv8n (sạch, chưa tấn công)

| Lần chạy | Dữ liệu | Kết quả |
|---|---|---|
| v2 | DIOR-only | 50 epoch ≈ 1,87 h, mAP50 = 0,902 |
| v3 | DOTA + DIOR (13.931 cặp; train 11.144 / val 2.787) | 50 epoch ≈ 3,6 h; **mAP50 = 0,9105**, mAP50-95 = 0,614; aircraft 0,945, vehicle 0,875 |

### 4.2 Tấn công DE trên surrogate (thiết kế cũ — tham khảo)

Cấu hình: patch 32×32, `pop = 20`, `gen = 50`, một patch, metric cũ.

| Phiên bản | Dữ liệu | ASR in-sample | ASR out-of-sample | AORR | FP mới | Thời gian DE |
|---|---|---|---|---|---|---|
| v4 | DOTA-only (6.625 cặp) | 93,05 % | 95,96 % (50 ảnh) | ≈ 0,59 | gần 0 | ≈ 9,4 phút |
| v5 | DOTA + DIOR gộp (13.940 cặp) | 93,79 % | 83,64 % | 0,5598 | 0 | — |

### 4.3 Chuyển giao sang model đích (RT-DETR-l / DIOR)

**Chưa có kết quả.** Notebook **v8** (và pipeline local) đã cài đúng thiết kế: surrogate YOLOv8n/DOTA → tối ưu patch (gồm tối ưu vị trí) → đo transfer trên RT-DETR/DIOR, kèm đối chứng patch ngẫu nhiên. Sau khi chạy sẽ xuất `attack_results/summary.csv` với các cột `transfer_ASR`, `transfer_AORR`, `transfer_pct` cho từng cấu hình. Chỉ số chính cần theo dõi: **transfer_ASR phải vượt rõ `random_single`**.

---

## 5. Dataset (link tải)

Toàn bộ dữ liệu thô **không** nằm trong repo (nặng vài GB, loại bằng `.gitignore`). Hai bộ dữ liệu thô đã tải lên Kaggle:

| Dataset | Nội dung | Link Kaggle |
|---|---|---|
| **DIOR (raw)** | 23.463 ảnh + nhãn VOC XML (`JPEGImages/`, `Annotations/`) | https://www.kaggle.com/datasets/thuyen001/dior-raw |
| **DOTA v1.0 (raw)** | 1.411 ảnh train + nhãn OBB (`images/`, `labelTxt/`) — bản `v2` khớp 1.411/1.411 cặp | https://www.kaggle.com/datasets/thuyen001/dota-raw |

Nguồn gốc:
- DIOR: [Google Drive của tác giả](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
- DOTA: https://captain-whu.github.io/DOTA/

### Cách tải về

**Kaggle CLI** (cần `pip install kaggle` và `kaggle.json` trong `~/.kaggle/`):

```bash
mkdir -p data/DIOR/raw data/DOTA/raw
kaggle datasets download -d thuyen001/dior-raw -p data/DIOR/raw --unzip
kaggle datasets download -d thuyen001/dota-raw -p data/DOTA/raw --unzip
```

**Chạy trực tiếp trên Kaggle:** thêm hai dataset vào notebook bằng *Add Data*; các notebook đọc từ

```
/kaggle/input/datasets/thuyen001/dota-raw
/kaggle/input/datasets/thuyen001/dior-raw/dior_raw
```

Cấu trúc kỳ vọng sau khi tải:

```
data/
├── DIOR/raw/dior_raw/{JPEGImages, Annotations}
├── DOTA/raw/dota_raw/{images, labelTxt}
├── yolo_dota/            # sinh tự động: surrogate (DOTA-only)
└── yolo_dior/            # sinh tự động: target (DIOR-only)
```

### Ánh xạ lớp thống nhất

| Lớp | DOTA | DIOR |
|---|---|---|
| `0 – aircraft` | `plane` | `airplane` |
| `1 – vehicle` | `small-vehicle`, `large-vehicle` | `vehicle` |

---

## 6. Cấu trúc repo

```
NCKH-UAV/
├── configs/
│   └── base_config.yaml          # cấu hình chung
├── docs/
│   └── NCKH_TongHop_Full.docx    # kế hoạch tổng thể, đặc tả, tổng quan tài liệu
├── src/
│   ├── attack/
│   │   ├── de_attack_real.py     # ★ DE thật (DEAttackReal): nội dung + VỊ TRÍ patch, match_iou, FP đã sửa
│   │   ├── de_optimizer.py       # bản DE mô phỏng ban đầu (test luồng)
│   │   └── evaluator.py          # ASR, AORR, New FP
│   ├── data/generate_dummy.py
│   ├── data_prep/
│   │   ├── extract_raw.py
│   │   ├── process_dota.py       # DOTA: tile 1024, OBB→HBB, Unified Mapping
│   │   └── process_dior.py       # DIOR: VOC XML → YOLO
│   ├── models/yolo_wrapper.py
│   └── train/train_surrogate.py
├── local/                        # ★ pipeline chạy LOCAL trên GPU
│   ├── run_local.py              # orchestrator: prepare → surrogate → target → attack
│   ├── config_local.yaml         # đường dẫn + tham số + cờ reuse + cấu hình tấn công
│   ├── run_all.bat / run_all.sh  # chạy 1-click (Windows / Linux)
│   └── README_LOCAL.md           # hướng dẫn chạy local
├── NCKH_UAV_Kaggle_v4.ipynb      # DOTA-only + DE thật
├── NCKH_UAV_Kaggle_v5.ipynb      # DOTA + DIOR gộp
├── NCKH_UAV_Kaggle_v6.ipynb      # + RT-DETR target (dữ liệu gộp)
├── NCKH_UAV_Kaggle_v7.ipynb      # reuse weights + tối ưu vị trí
├── NCKH_UAV_Kaggle_v8.ipynb      # ★ tách DOTA/DIOR + đối chứng ngẫu nhiên (bản mới nhất)
├── main.py                       # điểm vào pipeline (hiện chạy luồng dummy)
├── prepare_data.py
├── extract_and_cleanup.py
├── setup_local.bat / fix_env.bat
├── requirements.txt
└── LICENSE
```

Không đưa lên GitHub: `data/`, `dummy_data/`, `venv/`, trọng số (`*.pt`, `*.pth`, `*.onnx`), `runs/`, `attack_results/`, bản đóng gói `nckh-uav-code/` và `nckh-uav-code.zip` (chỉ để tải mã lên Kaggle).

---

## 7. Cài đặt và chạy lại

### A. Chạy LOCAL trên GPU (khuyến nghị khi có máy GPU — không giới hạn 12h)

Xem chi tiết trong [`local/README_LOCAL.md`](local/README_LOCAL.md). Tóm tắt:

```bat
python -m venv venv & venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python pyyaml tqdm
```

Chỉnh đường dẫn/tham số trong `local/config_local.yaml` rồi chạy 1-click:

```bat
local\run_all.bat            REM Windows  (hoặc: bash local/run_all.sh trên Linux)
```

Hoặc từng stage: `python local\run_local.py --stages prepare|surrogate|target|attack`. Có cờ `reuse` để nạp lại model đã train (bỏ qua huấn luyện).

### B. Chạy trên Kaggle (2× T4)

Dùng `NCKH_UAV_Kaggle_v8.ipynb` (bản mới nhất). Thêm vào *Add Data*: `dota-raw`, `dior-raw` và bộ mã `nckh-uav-code`. Chạy lần lượt các ô: môi trường → cài gói → symlink dữ liệu → xử lý DOTA/DIOR → **tách yolo_dota / yolo_dior** → huấn luyện YOLOv8n/DOTA → huấn luyện RT-DETR/DIOR → DE tấn công (4 cấu hình) → bảng transfer.

> ⚠️ Mỗi lần đổi `src/attack/de_attack_real.py` phải **up lại dataset `nckh-uav-code`** lên Kaggle. Nếu dùng cờ `REUSE_*`, weights phải là bản **DOTA-only / DIOR-only** — không dùng weights v5/v6 (gộp dữ liệu).
>
> Mẹo tránh timeout 12h: train RT-DETR ở một phiên riêng (`STOP_AFTER_TRAINING=True`), phiên sau `REUSE_*=True` chỉ chạy tấn công.

### C. Test luồng bằng dữ liệu giả lập

```bash
python main.py --mode all --dummy
```

> `main.py` hiện chỉ kiểm tra luồng (DE mô phỏng + evaluator demo). Tấn công thật chạy qua notebook / `local/` / `DEAttackReal`.

---

## 8. Hạn chế hiện tại

- **Chưa có số chuyển giao thật** — code v8/local đã đúng thiết kế (tách DOTA/DIOR, tối ưu vị trí, đối chứng ngẫu nhiên) nhưng **chưa chạy ra kết quả transfer**. Đây là chỉ số chính của đề tài.
- **Chưa tái lập baseline gốc:** DE một patch (Tang et al.) và AORR (Du et al.) chưa chạy để so trực tiếp số trong bài; "baseline" hiện tại là DE tự viết, không phải code gốc của bài 2/3.
- **Chưa dùng ensemble surrogate** — mới một surrogate; SDMPA cho thấy nhiều patch nhỏ chuyển giao tốt hơn khi tối ưu trên ensemble.
- Tập đánh giá nhỏ (30–50 ảnh), chạy **một seed** → chưa có khoảng tin cậy; cần nhiều seed + ≥ 200 ảnh.
- `L_nps` đang là bản đơn giản (phương sai màu), chưa dùng bảng màu in được; **chưa có augmentation EOT** (xoay, đổi tỉ lệ, sáng).
- Patch cỡ cố định; với **xe nhỏ**, patch có thể phủ gần hết đối tượng — cần nêu rõ trong phần hạn chế của báo cáo.
- Dữ liệu: 2 ảnh PNG DOTA bị hỏng (bỏ qua khi cắt tile).

---

## 9. Hướng tiếp theo

Thứ tự ưu tiên: phần cốt lõi (giai đoạn 5 + báo cáo) trước; mở rộng chỉ nếu còn thời gian.

### Đã làm trong code (chờ chạy/ghi số)

- ✅ Tách nhóm tối ưu (DOTA) / giữ kín (DIOR) — khác kiến trúc + dữ liệu.
- ✅ Tối ưu vị trí patch (DE tối ưu cả toạ độ, kiểu DeMPAA/Du).
- ✅ Đối chứng patch ngẫu nhiên cùng diện tích; cấu hình cùng tổng diện tích để so 1-vs-nhiều patch.
- ✅ Sửa metric: `match_iou` nhất quán, FP đếm box không khớp, held-out không trùng ảnh tối ưu.

### Ngắn hạn (ưu tiên cao)

1. **Chạy v8 / pipeline local** để lấy bảng transfer (surrogate DOTA vs RT-DETR/DIOR) + đối chứng ngẫu nhiên; lưu `summary.csv`, patch, ảnh ví dụ vào repo (nhẹ).
2. **Ablation số patch & kích thước** cùng tổng diện tích (1, 2, 4, …) để kiểm giả thuyết "nhiều patch nhỏ tốt hơn một patch lớn".
3. Chạy **nhiều seed**, tăng tập đánh giá (≥ 200 ảnh), báo cáo trung bình ± độ lệch chuẩn.

### Trung hạn

4. **Ensemble surrogate** (YOLOv5/YOLOv8, Faster R-CNN trên DOTA) để tăng khả năng chuyển giao; thêm model giữ kín (RetinaNet/FCOS trên DIOR).
5. **Tái lập baseline:** DE một patch (Tang et al.) trên DOTA và AORR theo Du et al.; ghi độ lệch so với số trong bài.
6. **Ablation DE vs tìm kiếm ngẫu nhiên**, trọng số `w_fp / w_tv / w_nps`.
7. **Tính khả thi vật lý:** NPS theo bảng màu in được; augmentation EOT; đặt patch theo độ cao bay.
8. Phân tích theo lớp (aircraft vs vehicle) và theo kích thước đối tượng.

### Mở rộng (nếu còn thời gian)

9. Lớp **người** (VisDrone), dữ liệu thêm (MAR20).
10. Mô phỏng **Blender** (độ cao, góc nhìn, ánh sáng); chỉ số OSR, NDR cho video.

### Hoàn thiện sản phẩm

11. Gom logic notebook vào `src/`, nối `main.py` với pipeline thật; thêm `tests/`.
12. Ghim phiên bản trong `requirements.txt`; hướng dẫn tái lập theo seed.
13. Viết báo cáo NCKH (bảng baseline, đề xuất, ablation, chuyển giao; hạn chế), slide, demo.
14. Xác nhận lại với giảng viên: sản phẩm dự kiến ("tái định danh xuyên góc nhìn") **không khớp** nội dung đề tài; sản phẩm khớp là *chương trình tạo và đánh giá lớp ngụy trang đối kháng trên các model phát hiện ảnh UAV*.

---

## 10. Tài liệu tham khảo

1. **DeMPAA** — tấn công đối kháng nhiều patch nhỏ trên ảnh viễn thám (IEEE TGRS 2024).
2. **Tang et al.** — tấn công hộp đen bằng DE vào bộ phát hiện ảnh hàng không (EAAI 2024).
3. **Du et al.** — tấn công vật lý vào bộ phát hiện trên ảnh UAV (WACV 2022).
4. Dataset: **DOTA** (Xia et al., CVPR 2018) và **DIOR** (Li et al., ISPRS 2020).
5. Công cụ: Ultralytics (YOLOv8, RT-DETR).

> Ghi chú: danh mục trên chỉ nêu tên bài/hội nghị theo hồ sơ kế hoạch; bổ sung đầy đủ tác giả, DOI khi viết báo cáo.

---

## Giấy phép

Xem [`LICENSE`](LICENSE).
