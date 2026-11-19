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

Hệ quả: đề tài đi theo hướng **tấn công chuyển giao (transfer attack)**. Patch được tối ưu trên model thay thế (*surrogate*) do ta tự huấn luyện, rồi kiểm chứng trên model "đối phương giả định" (*target*) khác kiến trúc, không dùng khi tối ưu. Differential Evolution (DE) là bộ tối ưu **không cần gradient**.

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

---

## 2. Phương pháp

```
                ┌──────────────────────────────┐
 DOTA + DIOR ──►│ Chuẩn hoá dữ liệu (YOLO fmt) │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ Huấn luyện SURROGATE         │  YOLOv8n, imgsz 1024
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ Tối ưu patch bằng DE         │  hộp đen với surrogate,
                │ (fitness = obj + FP + TV+NPS)│  không dùng gradient
                └──────────────┬───────────────┘
                               ▼
        ┌──────────────────────┴───────────────────────┐
        ▼                                              ▼
 Đánh giá trên SURROGATE                    Đánh giá chuyển giao trên TARGET
 (in-sample / out-of-sample)                (RT-DETR-l — notebook v6)
        ASR · AORR · False Positives
```

### 2.1 Chuẩn hoá dữ liệu

- **DOTA** (`src/data_prep/process_dota.py`): cắt ảnh lớn thành tile **1024×1024, overlap 20%**; đổi khung xoay (OBB) thành khung ngang (HBB); chỉ giữ `plane → 0`, `small-vehicle`/`large-vehicle → 1`; bỏ nhãn `difficult = 2`.
- **DIOR** (`src/data_prep/process_dior.py`): chuyển VOC XML → YOLO; chỉ giữ `airplane → 0`, `vehicle → 1`; hỗ trợ `use_symlink=True` để tiết kiệm đĩa trên Kaggle.
- Chia train/val theo tỉ lệ **80/20**, `seed = 42` (gộp DOTA slices + DIOR).

### 2.2 Model thay thế (surrogate) và model đích (target)

| Vai trò | Model | Ghi chú |
|---|---|---|
| Surrogate | YOLOv8n (Ultralytics) | 50 epoch, `imgsz=1024`, batch 4, 2× T4 |
| Target (giả định đối phương) | RT-DETR-l | transformer, khác họ kiến trúc; `imgsz=640` vì nặng; **notebook v6, chưa chạy** |

### 2.3 Tấn công DE (`src/attack/de_attack_real.py`)

Lớp `DEAttackReal` cài đặt DE với **quần thể cố định** (không dùng `scipy.optimize.differential_evolution` vì quần thể của scipy tăng theo số chiều, quá nặng với patch 32×32×3 = 3.072 chiều).

- Biến tối ưu: các điểm ảnh của patch, giá trị trong [0, 255].
- Tham số đã dùng: patch 32×32, `pop = 20`, `max_gen = 50`, `F = 0.8`, `CR = 0.7`.
- Hàm fitness tổng hợp:

  `L = w_obj·L_obj + w_fp·L_fp + w_tv·L_tv + w_nps·L_nps` với `w = (1.0, 0.3, 0.01, 0.001)`

  - `L_obj`: độ tin cậy (objectness/confidence) còn lại của đối tượng mục tiêu — càng thấp càng tốt;
  - `L_fp`: phạt các **báo động giả sinh thêm** ở nền do patch gây ra;
  - `L_tv`: Total Variation — ép patch mượt, tránh nhiễu tần số cao;
  - `L_nps`: bản đơn giản của Non-Printability Score (phạt phương sai màu).
- Đặt patch (`apply_patch`): `num_patches = 1` → một patch ở tâm bbox; `num_patches > 1` → nhiều mini-patch (kích thước bằng một nửa) ở tâm và các góc bbox.
- Đánh giá: `evaluate_patch` (in-sample / out-of-sample theo seed khác).

### 2.4 Chỉ số đánh giá (`src/attack/evaluator.py`, `evaluate_patch`)

| Chỉ số | Ý nghĩa |
|---|---|
| **ASR** (Attack Success Rate) | Tỉ lệ đối tượng bị làm biến mất (không còn box khớp IoU ≥ 0.5 với đối tượng sạch). Chỉ số chính |
| **AORR** | Mức giảm confidence trung bình của đối tượng (theo Du et al. 2022) |
| **New FP** | Số báo động giả **sinh thêm** ở nền so với ảnh sạch |

Ngưỡng dùng chung: `conf = 0.25`, `IoU = 0.5`.

---

## 3. Tiến độ đã hoàn thành

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0 | Chốt bài toán: threat model hộp đen, phạm vi, tiêu chí | ✅ Xong |
| 1 | Tổng quan tài liệu 3 bài báo + công trình liên quan (ColorFD, SDMPA, AP-PA, FCA…) | ✅ Xong |
| 2 | Môi trường cục bộ (Python 3.10, PyTorch CPU, Ultralytics), chuẩn hoá DOTA/DIOR, chạy được pipeline trên dữ liệu giả lập | ✅ Xong |
| 3 | Chạy dữ liệu thật trên Kaggle, huấn luyện surrogate, DE mock attack | ✅ Xong (notebook v3) |
| 4 | Sửa lỗi dữ liệu DOTA (nhầm ảnh), cài DE tấn công thật trên YOLOv8n | ✅ Xong (v4) |
| 5 | Sửa DIOR (symlink/XML→YOLO), DE tấn công trên tập kết hợp DOTA + DIOR | ✅ Xong (v5) |
| 6 | Tấn công chuyển giao sang RT-DETR-l | 🟡 Notebook v6 đã viết, **chưa chạy** |
| 7 | Viết báo cáo NCKH, slide, demo | ⬜ Chưa làm |

Tài liệu kế hoạch tổng hợp (kế hoạch 8 giai đoạn, đặc tả bài toán, tổng quan tài liệu): [`docs/NCKH_TongHop_Full.docx`](docs/NCKH_TongHop_Full.docx).

### Lịch sử các phiên bản notebook (Kaggle)

| Notebook | Nội dung chính |
|---|---|
| `NCKH_UAV_Kaggle_v4.ipynb` | Dữ liệu DOTA-only (DIOR lỗi đường dẫn, 0 ảnh); DE tấn công thật lần đầu |
| `NCKH_UAV_Kaggle_v5.ipynb` | Sửa DIOR (XML→YOLO với `use_symlink=True`); huấn luyện trên DOTA + DIOR |
| `NCKH_UAV_Kaggle_v6.ipynb` | Thêm huấn luyện RT-DETR-l làm target; đánh giá chuyển giao YOLOv8n → RT-DETR |

---

## 4. Kết quả thực nghiệm hiện có

> Số liệu lấy từ các lần chạy trên Kaggle (2× T4). Output của notebook đã được xoá khi lưu vào repo cho nhẹ; hãy đối chiếu lại với log gốc trên Kaggle khi viết báo cáo.

### 4.1 Surrogate YOLOv8n (sạch, chưa tấn công)

| Lần chạy | Dữ liệu | Kết quả |
|---|---|---|
| v2 | DIOR-only | 50 epoch ≈ 1,87 h, mAP50 = 0,902 |
| v3 | DOTA + DIOR (13.931 cặp ảnh–nhãn; train 11.144 / val 2.787) | 50 epoch ≈ 3,6 h; **mAP50 = 0,9105**, mAP50-95 = 0,614; aircraft mAP50 = 0,945, vehicle mAP50 = 0,875 |

### 4.2 Tấn công DE trên surrogate

Cấu hình chung: patch 32×32, `pop = 20`, `gen = 50`, một patch (`num_patches = 1`).

| Phiên bản | Dữ liệu huấn luyện | ASR in-sample (30 ảnh) | ASR out-of-sample | AORR | FP mới | Thời gian DE |
|---|---|---|---|---|---|---|
| v4 | DOTA-only: 6.625 cặp (5.300 train / 1.325 val) | 93,05 % | 95,96 % (50 ảnh) | ≈ 0,59 | gần 0 | ≈ 9,4 phút |
| v5 | DOTA 7.849 slice + DIOR 7.315 ảnh = 13.940 cặp (train 11.152 / val 2.788) | 93,79 % | 83,64 % | 0,5598 | 0 | — |

Nhận xét:
- Patch tối ưu bằng DE làm biến mất phần lớn đối tượng mà **không sinh thêm báo động giả**.
- ASR out-of-sample của v5 thấp hơn v4 (83,64 % so với 95,96 %) vì model mạnh hơn và dữ liệu đa dạng hơn — đây là kết quả đáng tin hơn để báo cáo.
- Biểu đồ hội tụ DE của lần chạy mô phỏng đầu tiên: [`de_convergence.png`](de_convergence.png).

### 4.3 Chuyển giao sang model đích (RT-DETR-l)

**Chưa có kết quả.** Notebook v6 đã sẵn sàng; sau khi chạy sẽ in bảng so sánh YOLOv8n (surrogate) và RT-DETR (target) kèm *transfer rate* = ASR_target / ASR_surrogate.

---

## 5. Dataset (link tải)

Toàn bộ dữ liệu thô **không** nằm trong repo (nặng vài GB, đã loại bằng `.gitignore`). Hai bộ dữ liệu thô đã được chuẩn bị và tải lên Kaggle:

| Dataset | Nội dung | Link Kaggle |
|---|---|---|
| **DIOR (raw)** | 23.463 ảnh + nhãn VOC XML (`JPEGImages/`, `Annotations/`) | https://www.kaggle.com/datasets/thuyen001/dior-raw |
| **DOTA v1.0 (raw)** | 1.411 ảnh train + nhãn OBB (`images/`, `labelTxt/`) — bản `v2` đã sửa lỗi nhầm ảnh, khớp 1.411/1.411 cặp | https://www.kaggle.com/datasets/thuyen001/dota-raw |

Nguồn gốc (bản chính thức):
- DIOR: [Google Drive của tác giả](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
- DOTA: https://captain-whu.github.io/DOTA/

### Cách tải về

**Cách 1 — Kaggle CLI** (cần `pip install kaggle` và file `kaggle.json` trong `~/.kaggle/`):

```bash
mkdir -p data/DIOR/raw data/DOTA/raw

kaggle datasets download -d thuyen001/dior-raw -p data/DIOR/raw --unzip
kaggle datasets download -d thuyen001/dota-raw -p data/DOTA/raw --unzip
```

**Cách 2 — Tải thủ công:** mở hai link Kaggle ở trên, bấm *Download*, giải nén vào `data/DIOR/raw/` và `data/DOTA/raw/`.

**Cách 3 — Chạy trực tiếp trên Kaggle:** thêm hai dataset vào notebook bằng *Add Data*; các notebook v4–v6 đọc từ

```
/kaggle/input/datasets/thuyen001/dota-raw
/kaggle/input/datasets/thuyen001/dior-raw/dior_raw
```

Cấu trúc kỳ vọng sau khi tải (script `src/data_prep/*` đọc theo đúng cấu trúc này):

```
data/
├── DIOR/raw/{JPEGImages, Annotations}
├── DOTA/raw/{images, labelTxt}
└── processed/            # sinh tự động sau bước xử lý (YOLO format)
```

Sau đó xử lý dữ liệu:

```bash
python -m src.data_prep.extract_raw     # nếu dữ liệu ở dạng zip
python -m src.data_prep.process_dota    # cắt tile 1024×1024, OBB→HBB
python -m src.data_prep.process_dior    # VOC XML → YOLO
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
│   └── base_config.yaml          # cấu hình chung (dataset, model, tham số tấn công)
├── docs/
│   └── NCKH_TongHop_Full.docx    # kế hoạch tổng thể, đặc tả bài toán, tổng quan tài liệu
├── src/
│   ├── attack/
│   │   ├── de_attack_real.py     # ★ DE tấn công thật trên YOLOv8 (DEAttackReal)
│   │   ├── de_optimizer.py       # bản DE mô phỏng ban đầu (mock model), dùng test luồng
│   │   └── evaluator.py          # ASR, AORR, New False Positives
│   ├── data/generate_dummy.py    # sinh dữ liệu giả lập để test luồng
│   ├── data_prep/
│   │   ├── extract_raw.py        # giải nén dữ liệu thô
│   │   ├── process_dota.py       # DOTA: tile 1024, OBB→HBB, Unified Mapping
│   │   └── process_dior.py       # DIOR: VOC XML → YOLO, Unified Mapping
│   ├── models/yolo_wrapper.py    # wrapper YOLOv8
│   └── train/train_surrogate.py  # huấn luyện surrogate
├── NCKH_UAV_Kaggle_v4.ipynb      # DOTA-only + DE thật
├── NCKH_UAV_Kaggle_v5.ipynb      # DOTA + DIOR
├── NCKH_UAV_Kaggle_v6.ipynb      # + RT-DETR target, đánh giá chuyển giao
├── main.py                       # điểm vào pipeline (hiện chạy luồng dummy)
├── prepare_data.py               # chuẩn bị dữ liệu
├── extract_and_cleanup.py        # giải nén & dọn dữ liệu thô
├── setup_local.bat / fix_env.bat # cài đặt môi trường trên Windows
├── setup_colab.py                # cài đặt trên Colab
├── de_convergence.png            # đường hội tụ DE (lần chạy mô phỏng)
├── requirements.txt
└── LICENSE
```

Các thư mục/tệp **không** đưa lên GitHub: `data/`, `dummy_data/`, `venv/`, trọng số (`*.pt`, `*.pth`, `*.onnx`), `runs/`, bản đóng gói `nckh-uav-code/` và `nckh-uav-code.zip` (chỉ dùng để tải mã lên Kaggle).

---

## 7. Cài đặt và chạy lại

### Máy cá nhân (Windows, CPU — để phát triển và test luồng)

```bat
setup_local.bat
```

hoặc thủ công:

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Kaggle / Colab (GPU — chạy thật)

Chạy theo thứ tự các ô trong `NCKH_UAV_Kaggle_v6.ipynb` (bản mới nhất): môi trường & GPU → cài gói → tạo symlink dữ liệu → xử lý DOTA/DIOR → chia train/val → huấn luyện YOLOv8n → huấn luyện RT-DETR-l → DE tấn công → đánh giá chuyển giao. Notebook cần thêm vào *Add Data*: `dota-raw`, `dior-raw` và bộ mã `nckh-uav-code` (hoặc clone repo này thay cho bộ mã).

Trên Colab: `!python setup_colab.py` ở ô đầu tiên.

### Test luồng bằng dữ liệu giả lập

```bash
python main.py --mode all --dummy          # toàn bộ pipeline trên dummy
python main.py --mode generate_dummy
python main.py --mode train --dummy
python main.py --mode attack
python main.py --mode evaluate
```

> `main.py` hiện chỉ kiểm tra luồng (DE mô phỏng + evaluator demo). Tấn công thật chạy qua notebook v4–v6 / `DEAttackReal`.

---

## 8. Hạn chế hiện tại

- **Chưa có kết quả chuyển giao sang model khác kiến trúc** — đây là chỉ số chính của đề tài (ASR trên nhóm giữ kín). Hiện "out-of-sample" chỉ là ảnh khác trong cùng phân phối, cùng model surrogate.
- Mọi thí nghiệm đến nay dùng **một patch** (`num_patches = 1`). Cơ chế nhiều mini-patch có trong `apply_patch` nhưng chưa đánh giá.
- Surrogate hiện được huấn luyện trên **DOTA + DIOR gộp**; kế hoạch ban đầu tách DOTA (nhóm tối ưu) và DIOR (nhóm giữ kín) để mô phỏng "dữ liệu đối phương khác hẳn".
- Patch hiện đặt tại các vị trí cố định theo bbox (tâm/góc) và được tối ưu chung cho tập ảnh; chưa tối ưu vị trí.
- `L_nps` đang là phiên bản đơn giản (phương sai màu), chưa dùng bảng màu in được thật; chưa có augmentation (xoay, đổi tỉ lệ, sáng) kiểu EOT.
- Tập đánh giá nhỏ (30–50 ảnh), chạy một seed → chưa có khoảng tin cậy.
- Chưa có đối chứng patch ngẫu nhiên (Tang et al. báo nhiễu ngẫu nhiên đã đạt ASR ≈ 44 %).
- Dữ liệu: 2 ảnh PNG DOTA bị hỏng (bỏ qua khi cắt tile).

---

## 9. Hướng tiếp theo

Thứ tự ưu tiên theo kế hoạch: phần cốt lõi (giai đoạn 3–5 và báo cáo) phải xong trước; mở rộng chỉ làm nếu còn thời gian.

### Ngắn hạn (ưu tiên cao)

1. **Chạy notebook v6** trên Kaggle: huấn luyện RT-DETR-l, đo mAP sạch, tính ASR/AORR/FP trên target và *transfer rate*. Lưu `best_patch.npy`, `de_history.npy` và ảnh ví dụ vào repo (nhẹ).
2. **Đối chứng bắt buộc:** patch ngẫu nhiên cùng diện tích; so sánh với một patch lớn cùng tổng diện tích.
3. **Đánh giá nhiều mini-patch** (`num_patches = 1, 2, 3, 5`) với **cùng tổng diện tích**, để kiểm chứng giả thuyết "nhiều patch nhỏ tốt hơn một patch lớn".
4. Chạy nhiều seed, tăng tập đánh giá (≥ 200 ảnh) và báo cáo trung bình ± độ lệch chuẩn.

### Trung hạn

5. **Tách nhóm model giữ kín đúng kế hoạch:** nhóm tối ưu (YOLOv5/YOLOv8, Faster R-CNN) huấn luyện trên DOTA; nhóm giữ kín (RT-DETR, RetinaNet/FCOS) huấn luyện trên DIOR. Đo mAP sạch từng model và ghi vào báo cáo trước khi xem kết quả tấn công.
6. **Tối ưu trên nhiều surrogate cùng lúc** (ensemble) để tăng khả năng chuyển giao.
7. **Baseline tái lập:** DE một patch (Tang et al.) trên DOTA và AORR theo Du et al.; ghi lại độ lệch so với số trong bài.
8. **Ablation:** số patch, kích thước patch, số model trong nhóm tối ưu, DE so với tìm kiếm ngẫu nhiên, trọng số `w_fp / w_tv / w_nps`.
9. **Tăng tính khả thi vật lý:** NPS theo bảng màu in được; augmentation xoay/tỉ lệ/sáng (EOT); đặt patch theo độ cao bay (kích thước tương đối theo kích thước đối tượng).
10. **Tối ưu vị trí patch** (bản đồ vị trí khả thi kiểu DeMPAA) thay cho vị trí cố định tâm/góc.
11. Phân tích theo lớp (aircraft so với vehicle) và theo kích thước đối tượng.

### Mở rộng (nếu còn thời gian)

12. Lớp **người** (VisDrone) và dữ liệu thêm (MAR20 cho máy bay quân sự).
13. Mô phỏng **Blender**: đổi độ cao, góc nhìn, ánh sáng có kiểm soát; chỉ số OSR, NDR cho video/cảnh mô phỏng.

### Hoàn thiện sản phẩm

14. Gom logic từ notebook vào `src/` và nối `main.py` với pipeline thật (hiện `main.py` chỉ chạy luồng dummy); thêm `tests/` cơ bản.
15. Ghim phiên bản trong `requirements.txt`; thêm hướng dẫn tái lập kết quả với seed.
16. Viết báo cáo NCKH (bảng baseline, đề xuất, ablation, chuyển giao; phần hạn chế), slide và demo.
17. Xác nhận lại với giảng viên: sản phẩm dự kiến ("chương trình tái định danh xuyên góc nhìn") **không khớp** với nội dung đề tài (tấn công ngụy trang vào model phát hiện); sản phẩm khớp là *chương trình tạo và đánh giá lớp ngụy trang đối kháng trên các model phát hiện ảnh UAV*.

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
