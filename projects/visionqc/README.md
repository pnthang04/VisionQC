# VisionQC – Robust PCB Anomaly Detection

Baseline phát hiện và định vị lỗi PCB bằng **EfficientAD-small** trên **VisA/pcb1**, sử dụng Anomalib 2.6.0.
Baseline hiện chưa có distribution alignment hoặc thay đổi mã nguồn lõi của Anomalib.

## Cấu hình baseline

| Thành phần | Giá trị |
|---|---|
| Model | EfficientAD-small |
| Dataset | VisA |
| Category | `pcb1` |
| Seed | `42` |
| Train batch size | `1` |
| Eval batch size | `8` |
| Accelerator | Tự động chọn GPU/CPU |
| Giới hạn train | 1.000 epoch hoặc 70.000 step |
| Metrics | Image/pixel AUROC, image/pixel F1, pixel AUPRO |

Cấu hình nằm tại `projects/visionqc/configs/efficientad_pcb1.yaml`. CLI được cài với tên `visionqc`.

## Triển khai trên server Linux

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
bash projects/visionqc/setup.sh
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
.venv/bin/python -m unittest discover projects/visionqc/tests
```

Không bắt đầu full training nếu unit test lỗi hoặc `GPU: False` trong khi server được cấp GPU.

## Chạy trên Kaggle

Trong Kaggle Notebook:

1. Chọn **Settings → Accelerator → GPU**.
2. Bật **Internet** để tải VisA, ImageNette và pretrained teacher weights.
3. Mở terminal tại thư mục gốc repository và chạy:

```bash
bash projects/visionqc/setup.sh
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

## Chuẩn bị và kiểm tra VisA

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

## Smoke test

```bash
bash projects/visionqc/smoke_test.sh
```

Smoke test chạy tối đa hai train/validation/test batch và phải tạo checkpoint. Lần đầu vẫn cần tải pretrained
weights, ImageNette và tính teacher statistics nên có thể mất vài phút. Kết quả smoke test không phải kết quả
baseline chính thức.

Chạy unit test offline sau khi setup:

```bash
.venv/bin/python -m unittest discover projects/visionqc/tests
```

## Train chính thức

```bash
bash projects/visionqc/train.sh
```

Resume từ checkpoint:

```bash
bash projects/visionqc/train.sh \
  --checkpoint projects/visionqc/results/efficientad_pcb1/EfficientAd/Visa/pcb1/v0/weights/lightning/model.ckpt
```

Anomalib có thể tạo `v1`, `v2`, ... cho các lần chạy tiếp theo. Dùng đúng checkpoint được ghi trong
`projects/visionqc/results/efficientad_pcb1/best_checkpoint.txt`.

## Evaluate và sinh heatmap

Evaluate checkpoint mới nhất:

```bash
bash projects/visionqc/evaluate.sh
```

Hoặc chỉ định checkpoint:

```bash
bash projects/visionqc/evaluate.sh \
  --checkpoint projects/visionqc/results/efficientad_pcb1/EfficientAd/Visa/pcb1/v0/weights/lightning/model.ckpt
```

Kết quả nằm trong `projects/visionqc/results/efficientad_pcb1/`:

| Output | Nội dung |
|---|---|
| `metrics.json`, `metrics.csv` | Metrics evaluate thực tế |
| `anomaly_scores.csv` | Score, ground truth, prediction và outcome từng ảnh |
| `heatmaps/` | Heatmap/anomaly map |
| `examples/tn.png` | Ảnh normal dự đoán đúng |
| `examples/tp.png` | Ảnh anomalous dự đoán đúng |
| `examples/fp.png` | False positive, nếu có |
| `examples/fn.png` | False negative, nếu có |
| `best_checkpoint.txt` | Checkpoint được dùng sau train |

Không có `fp.png` hoặc `fn.png` nghĩa là outcome tương ứng không xuất hiện trong lần evaluate.

## Cấu trúc

```text
projects/visionqc/
├── configs/efficientad_pcb1.yaml
├── src/visionqc/
│   ├── __init__.py
│   ├── __main__.py
│   └── pipeline.py
├── tests/test_pipeline.py
├── results/
├── logs/
├── pyproject.toml
├── README.md
├── setup.sh
├── smoke_test.sh
├── train.sh
└── evaluate.sh
```

## Lỗi thường gặp

- **`GPU: False`**: bật GPU trong Kaggle Settings và restart session.
- **CUDA out of memory**: giảm `dataset.eval_batch_size` từ `8` xuống `4` hoặc `1`.
- **Download/hash failure**: kiểm tra Internet của Kaggle rồi chạy lại đúng lệnh; không xóa dataset đã tải dở nếu
  chưa cần.
- **No checkpoint found**: chạy train trước hoặc truyền đường dẫn chính xác bằng `--checkpoint`.
- **No space left on device**: xóa output/checkpoint cũ và cache không còn sử dụng.
- **Windows không chạy được `.sh`**: pipeline này nhắm đến Kaggle/Linux; dùng WSL hoặc Git Bash nếu cần chạy local.

## Checklist giao cho agent trên server

Có thể giao nguyên yêu cầu sau cho agent:

```text
Đứng tại thư mục gốc repository VisionQC.

1. Đọc AGENTS.md, projects/visionqc/README.md, pyproject.toml và
   projects/visionqc/configs/efficientad_pcb1.yaml.
2. Không sửa mã nguồn lõi Anomalib.
3. Kiểm tra git status, commit hiện tại, Python, dung lượng ổ đĩa, nvidia-smi và NVIDIA driver.
4. Chạy bash projects/visionqc/setup.sh.
5. Xác nhận Anomalib 2.6.0, PyTorch, CUDA, torch.cuda.is_available() và tên GPU.
6. Chạy unit test:
   .venv/bin/python -m unittest discover projects/visionqc/tests
7. Chạy .venv/bin/visionqc prepare và báo cáo chính xác số ảnh/mask/batch shape.
8. Chạy bash projects/visionqc/smoke_test.sh. Không chạy full train nếu smoke test fail.
9. Nếu smoke pass, chạy bash projects/visionqc/train.sh.
10. Sau train, chạy bash projects/visionqc/evaluate.sh.
11. Báo cáo checkpoint, metrics.json, anomaly_scores.csv, heatmaps và lỗi còn lại.
12. Không công bố metric giả; chỉ báo cáo kết quả được sinh từ lần evaluate thực tế.
```
