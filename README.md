# VisionQC: Robust PCB Anomaly Detection with EfficientAD

[Phạm Ngọc Thắng](https://github.com/pnthang04)

![Task](https://img.shields.io/badge/Task-Anomaly%20Detection-c0392b)
![Domain](https://img.shields.io/badge/Domain-PCB%20Quality%20Control-d35400)
![Framework](https://img.shields.io/badge/Framework-Anomalib-0078D4)
![Model](https://img.shields.io/badge/Model-EfficientAD--medium-5A9E1C)
![GPUs](https://img.shields.io/badge/Training-2×GPU-76B900?logo=nvidia&logoColor=white)

**Quick Links:** [📦 Dataset](#dataset) | [⚙️ Cấu hình](#configuration) |
[🚀 Training](#training) | [📊 Kết quả](#results) |
[🤗 Model](https://huggingface.co/thangkt/visionqc-efficientad-medium-pcb1)

## 🔎 Giới thiệu

VisionQC phát hiện và định vị lỗi bề mặt PCB bằng **EfficientAD-medium** trên tập **VisA/pcb1**, sử dụng
**Anomalib 2.6.0**. Pipeline hỗ trợ huấn luyện phân tán trên hai GPU, mixed precision, validation độc lập,
early stopping và tự động lưu checkpoint tốt nhất.

Mục tiêu chính:

- phát hiện PCB bất thường ở mức ảnh;
- định vị vùng lỗi bằng anomaly map;
- huấn luyện nhanh với batch size `32` trên mỗi GPU;
- đánh giá độc lập bằng image/pixel AUROC, F1 và AUPRO.

## 🤗 Checkpoint

Checkpoint tốt nhất được công bố công khai tại
[`thangkt/visionqc-efficientad-medium-pcb1`](https://huggingface.co/thangkt/visionqc-efficientad-medium-pcb1).

Tải checkpoint bằng Hugging Face CLI:

```bash
hf download thangkt/visionqc-efficientad-medium-pcb1 model-best.ckpt \
  --local-dir visionqc/weights/efficientad-medium-pcb1
```

Evaluate checkpoint đã tải:

```bash
bash visionqc/evaluate.sh \
  --checkpoint visionqc/weights/efficientad-medium-pcb1/model-best.ckpt
```

<a id="configuration"></a>

## ⚙️ Cấu hình thí nghiệm

| Thành phần       | Giá trị                                                  |
| ---------------- | -------------------------------------------------------- |
| Model            | EfficientAD-medium                                       |
| Dataset          | VisA                                                     |
| Category         | `pcb1`                                                   |
| Seed             | `42`                                                     |
| Train batch size | `32` mỗi GPU (`64` effective trên 2 GPU)                 |
| Eval batch size  | `8`                                                      |
| Validation       | Tách 50% từ test gốc, seed `42`                          |
| Early stopping   | Validation image AUROC, patience `20`, min delta `0.001` |
| Accelerator      | 2 GPU, DDP với hỗ trợ unused teacher parameters          |
| Precision        | FP16 mixed precision                                     |
| Giới hạn train   | 1.000 epoch hoặc 1.100 optimizer step (~70.400 sample)   |
| Metrics          | Image/pixel AUROC, image/pixel F1, pixel AUPRO           |

Cấu hình nằm tại `visionqc/configs/efficientad_pcb1.yaml`. CLI được cài với tên `visionqc`.

Upstream EfficientAD giới hạn batch train bằng `1`. Baseline này dùng subclass cục bộ `BatchedEfficientAd` để
cho phép batch `32` trên mỗi GPU mà không sửa mã nguồn lõi Anomalib. Đây là cấu hình thử nghiệm khác baseline
chính thức của EfficientAD; metric không nên được so sánh trực tiếp với kết quả batch `1`.
Giới hạn `1.100` optimizer step được scale từ baseline `70.000` step batch-1 theo effective batch `64`.

<a id="results"></a>

## 📊 Kết quả

Kết quả test của checkpoint EfficientAD-medium đã công bố:

| Metric         |    Giá trị |
| -------------- | ---------: |
| Image AUROC    | **0.9364** |
| Image F1 Score | **0.8785** |
| Pixel AUROC    | **0.9883** |
| Pixel F1 Score | **0.6095** |
| Pixel AUPRO    | **0.8686** |

Tập test độc lập với tập validation dùng cho early stopping và lựa chọn checkpoint.

## 🖥️ Triển khai trên server Linux

Yêu cầu:

- Linux x86_64;
- Git, `curl` và Python 3.10 trở lên;
- đủ dung lượng cho source, VisA, ImageNette, weights và checkpoints;
- NVIDIA driver hoạt động nếu train bằng GPU;
- Internet để tải dependencies và dữ liệu ở lần chạy đầu.

Clone và checkout đúng branch:

```bash
git clone https://github.com/pnthang04/VisionQC.git
cd VisionQC
git checkout main
git pull --ff-only origin main
```

Kiểm tra server trước khi setup:

```bash
python3 --version
df -h .
nvidia-smi
```

Setup:

```bash
bash visionqc/setup.sh
```

`setup.sh` xử lý ba trường hợp:

1. PyTorch CUDA đã hoạt động (như Kaggle): tái sử dụng môi trường GPU hiện có.
2. Có `nvidia-smi` nhưng chưa có PyTorch GPU: cài PyTorch CUDA 12.6 từ extra `cu126` của repository.
3. Không có NVIDIA GPU: cài backend CPU.

Nếu server không dùng `uv`, có thể cài bằng `requirements.txt` từ thư mục gốc:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` mặc định dành cho NVIDIA/CUDA 12.6. `setup.sh` vẫn là cách khuyến nghị vì có thể tự chọn GPU
hoặc CPU.

Xác minh môi trường:

```bash
.venv/bin/python -c "import anomalib, torch; print('Anomalib:', anomalib.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.venv/bin/python -m unittest discover visionqc/tests
```

Không bắt đầu full training nếu unit test lỗi hoặc `GPU: False` trong khi server được cấp GPU.

## ☁️ Chạy trên Kaggle

Trong Kaggle Notebook:

1. Chọn **Settings → Accelerator → GPU**.
2. Bật **Internet** để tải VisA, ImageNette và pretrained teacher weights.
3. Mở terminal tại thư mục gốc repository và chạy:

```bash
bash visionqc/setup.sh
```

Script tạo `.venv`, tái sử dụng PyTorch CUDA có sẵn của Kaggle, cài repository ở editable mode và cài hỗ trợ
OpenVINO export. Kiểm tra GPU:

```bash
nvidia-smi
.venv/bin/python -c "import anomalib, torch; print('anomalib:', anomalib.__version__); print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

Kết quả cuối phải là `GPU: True`. Nếu là `False`, bật GPU rồi restart Kaggle session.

CLI có bốn command:

```bash
.venv/bin/visionqc --help
.venv/bin/visionqc prepare
.venv/bin/visionqc smoke
.venv/bin/visionqc train
.venv/bin/visionqc evaluate
```

<a id="dataset"></a>

## 📦 Dữ liệu

Dataset được tải bằng datamodule `anomalib.data.Visa`, không sử dụng downloader riêng:

```bash
.venv/bin/visionqc prepare
```

Lệnh này tải, giải nén, convert VisA và in:

- đường dẫn dataset;
- số ảnh train normal;
- số ảnh validation;
- số ảnh test normal/anomalous;
- số mask;
- kích thước batch đầu tiên.

Dataset nằm tại `datasets/visa`; ImageNette phụ trợ của EfficientAD nằm tại `datasets/imagenette`. Cả hai đều được
Git ignore.

Validation được cấu hình bằng `val_split_mode: from_test` và `val_split_ratio: 0.5`: tập test gốc được chia theo
nhãn thành hai nửa validation/test tách biệt với seed `42`. `image_AUROC` trên validation được dùng để lưu
checkpoint tốt nhất và dừng sớm nếu không cải thiện ít nhất `0.001` trong `20` epoch liên tiếp. Smoke test không
bật early stopping.

## 🧪 Smoke test

```bash
bash visionqc/smoke_test.sh
```

Smoke test chạy tối đa hai train/validation/test batch và phải tạo checkpoint. Lần đầu vẫn cần tải pretrained
weights, ImageNette và tính teacher statistics nên có thể mất vài phút. Kết quả smoke test không phải kết quả
baseline chính thức.

Chạy unit test offline sau khi setup:

```bash
.venv/bin/python -m unittest discover visionqc/tests
```

<a id="training"></a>

## 🚀 Train chính thức

```bash
bash visionqc/train.sh
```

Resume từ checkpoint:

```bash
bash visionqc/train.sh \
  --checkpoint visionqc/results/efficientad_pcb1/BatchedEfficientAd/Visa/pcb1/v0/lightning_logs/version_0/checkpoints/model-best.ckpt
```

Anomalib có thể tạo `v1`, `v2`, ... cho các lần chạy tiếp theo. Dùng đúng checkpoint được ghi trong
`visionqc/results/efficientad_pcb1/best_checkpoint.txt`.

## 📈 Evaluate và sinh heatmap

Evaluate checkpoint mới nhất:

```bash
bash visionqc/evaluate.sh
```

Hoặc chỉ định checkpoint:

```bash
bash visionqc/evaluate.sh \
  --checkpoint visionqc/results/efficientad_pcb1/BatchedEfficientAd/Visa/pcb1/v0/lightning_logs/version_0/checkpoints/model-best.ckpt
```

Kết quả nằm trong `visionqc/results/efficientad_pcb1/`:

| Output                        | Nội dung                                            |
| ----------------------------- | --------------------------------------------------- |
| `metrics.json`, `metrics.csv` | Metrics evaluate thực tế                            |
| `anomaly_scores.csv`          | Score, ground truth, prediction và outcome từng ảnh |
| `heatmaps/`                   | Heatmap/anomaly map                                 |
| `examples/tn.png`             | Ảnh normal dự đoán đúng                             |
| `examples/tp.png`             | Ảnh anomalous dự đoán đúng                          |
| `examples/fp.png`             | False positive, nếu có                              |
| `examples/fn.png`             | False negative, nếu có                              |
| `best_checkpoint.txt`         | Checkpoint được dùng sau train                      |

Không có `fp.png` hoặc `fn.png` nghĩa là outcome tương ứng không xuất hiện trong lần evaluate.

## 🗂️ Cấu trúc dự án

```text
.
├── README.md
├── requirements.txt
└── visionqc/
    ├── configs/efficientad_pcb1.yaml
    ├── src/visionqc/
    │   ├── __init__.py
    │   ├── __main__.py
    │   └── pipeline.py
    ├── tests/test_pipeline.py
    ├── results/
    ├── logs/
    ├── pyproject.toml
    ├── setup.sh
    ├── smoke_test.sh
    ├── train.sh
    └── evaluate.sh
```

## 🛠️ Lỗi thường gặp

- **`GPU: False`**: bật GPU trong Kaggle Settings và restart session.
- **CUDA out of memory khi train**: batch `32` cần FP16 và hai GPU trống; kiểm tra tiến trình GPU cũ trước khi
  giảm `dataset.train_batch_size`.
- **CUDA out of memory khi evaluate**: giảm `dataset.eval_batch_size` từ `8` xuống `4` hoặc `1`.
- **Download/hash failure**: kiểm tra Internet của Kaggle rồi chạy lại đúng lệnh; không xóa dataset đã tải dở nếu
  chưa cần.
- **No checkpoint found**: chạy train trước hoặc truyền đường dẫn chính xác bằng `--checkpoint`.
- **No space left on device**: xóa output/checkpoint cũ và cache không còn sử dụng.
- **Windows không chạy được `.sh`**: pipeline này nhắm đến Kaggle/Linux; dùng WSL hoặc Git Bash nếu cần chạy local.

## ✅ Checklist triển khai

Có thể giao nguyên yêu cầu sau cho agent:

```text
Đứng tại thư mục gốc repository VisionQC.

1. Đọc AGENTS.md, README.md, pyproject.toml và
   visionqc/configs/efficientad_pcb1.yaml.
2. Không sửa mã nguồn lõi Anomalib.
3. Kiểm tra git status, commit hiện tại, Python, dung lượng ổ đĩa, nvidia-smi và NVIDIA driver.
4. Chạy bash visionqc/setup.sh.
5. Xác nhận Anomalib 2.6.0, PyTorch, CUDA, torch.cuda.is_available() và tên GPU.
6. Chạy unit test:
   .venv/bin/python -m unittest discover visionqc/tests
7. Chạy .venv/bin/visionqc prepare và báo cáo chính xác số ảnh/mask/batch shape.
8. Chạy bash visionqc/smoke_test.sh. Không chạy full train nếu smoke test fail.
9. Nếu smoke pass, chạy bash visionqc/train.sh.
10. Sau train, chạy bash visionqc/evaluate.sh.
11. Báo cáo checkpoint, metrics.json, anomaly_scores.csv, heatmaps và lỗi còn lại.
12. Không công bố metric giả; chỉ báo cáo kết quả được sinh từ lần evaluate thực tế.
```
