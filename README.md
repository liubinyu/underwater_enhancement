# Physical-Guided Lightweight Underwater Image Enhancement

增强算法、公式、当前UIEB小样本结果及后续路线详见：
[AquaAlign-VLM 水下图像增强算法实现与后续方向](docs/underwater_enhancement_methods.md)。

项目目录职责、输入输出和PowerShell完整运行方法详见：
[AquaAlign-VLM 项目框架与运行指南](docs/project_framework_and_usage.md)。

这是一个面向个人水下图像数据的 PyTorch 工程第一版，重点是“可运行、可训练、可推理、可解释”。项目不封装第三方增强模型，而是实现了两个可训练网络：

- `baseline`: 纯深度学习轻量 encoder-decoder，用作对照组。
- `physics_guided`: 估计背景光 `A` 和传输图 `t(x)`，先按物理模型粗恢复，再用轻量网络 refinement。

当前版本默认按无配对真实水下图像训练，也保留了配对监督训练接口。

## 方法思路

水下退化可以近似写成：

```text
I(x) = J(x) * t(x) + A * (1 - t(x))
```

主模型先学习：

- `A_estimator`: 估计全局/低频背景光 `A`
- `transmission_estimator`: 估计传输图 `t(x)`
- `coarse_restore`: `J_coarse = (I - A) / max(t, eps) + A`
- `refinement_net`: 输入原图、粗恢复图和传输图，输出最终增强图

总损失为：

```text
L = w_color L_color
  + w_exp L_exposure
  + w_struct L_gradient
  + w_edge L_edge
  + w_tv L_tv
  + w_phy L_physics
  + w_sup L_supervised
```

其中监督项只有在 `data.target_dir` 提供配对图像，并在 YAML 中打开对应权重时生效。

## 目录结构

```text
underwater_enhancement/
├─ configs/
│  ├─ baseline.yaml
│  └─ physics_guided.yaml
├─ data/
│  └─ raw/images/
├─ datasets/
│  └─ underwater_dataset.py
├─ losses/
│  ├─ color_loss.py
│  ├─ structure_loss.py
│  ├─ physics_loss.py
│  └─ total_loss.py
├─ models/
│  ├─ baseline_net.py
│  ├─ physics_guided_net.py
│  └─ modules.py
├─ scripts/
│  ├─ scan_dataset.py
│  ├─ plot_rgb_hist.py
│  ├─ train.py
│  ├─ infer.py
│  └─ compare_methods.py
├─ utils/
│  ├─ metrics.py
│  ├─ image_ops.py
│  ├─ logger.py
│  └─ vis.py
├─ results/
├─ checkpoints/
├─ README.md
├─ requirements.txt
└─ main.py
```

旧版 `src/` 和 `run_baselines.py` 保留，用于传统算法对照，不影响新 PyTorch 工程。

## 安装依赖

项目统一使用名为 `aqua_align` 的 Conda 环境。本地图像增强阶段不需要安装
ms-swift、bitsandbytes 或下载视觉语言模型：

```bash
conda create -n aqua_align python=3.11 -y  # 仅首次创建时执行
conda activate aqua_align
python -m pip install -r requirements-local.txt
```

A40 服务器同样使用 `aqua_align` 作为环境名。先根据服务器驱动从 PyTorch 官方安装
选择器安装 CUDA 版 PyTorch，再安装训练依赖，避免误装 CPU 版 PyTorch：

```bash
conda create -n aqua_align python=3.11 -y
conda activate aqua_align
# 在这里执行 https://pytorch.org/get-started/locally/ 生成的 CUDA PyTorch 安装命令
python -m pip install -r requirements-server.txt
python scripts/check_environment.py
```

`requirements.txt` 默认等价于本地安装入口。`flash-attn` 是可选加速项，应在基础 smoke
test 成功后再根据服务器的 PyTorch/CUDA 组合单独安装。

## AquaAlign-VLM 传统候选数据流水线

整理 UIEB 或目录结构等价的数据集：

```bash
python scripts/prepare_uieb.py \
  --input-dir /path/to/UIEB \
  --output-dir data/processed/uieb \
  --seed 42
```

生成传统增强候选、质量指标和非交互式总览图：

```bash
python scripts/generate_candidates.py \
  --metadata data/processed/uieb/metadata.csv \
  --config configs/data.yaml \
  --output-dir data/candidates

python scripts/compute_quality_metrics.py \
  --candidates data/processed/candidates.csv \
  --output data/processed/quality_metrics.csv

python scripts/create_comparison_figures.py \
  --candidates data/processed/candidates.csv \
  --metrics data/processed/quality_metrics.csv \
  --output-dir reports/figures
```

所有 CSV 图像路径均相对于项目根目录。候选生成默认不覆盖已有结果；调试时可使用
`--limit 10 --dry-run`，确认无误后再添加 `--overwrite` 执行重建。

## 数据准备

默认配置读取：

```text
data/raw/images
```

可以直接把 JPG/PNG/JPEG 图像放进去，例如：

```text
data/raw/images/G0273464.JPG
```

如果你的数据在其他目录，修改：

```yaml
data:
  input_dir: D:/your/path/images
```

若后续有配对清晰图，把目标图放入单独目录，并保证文件名一致，然后设置：

```yaml
data:
  target_dir: D:/your/path/targets
loss:
  weights:
    supervised_l1: 1.0
    supervised_charbonnier: 0.5
    supervised_ssim: 0.2
```

## 扫描数据集

```bash
python scripts/scan_dataset.py --input_dir data/raw/images
```

或：

```bash
python main.py scan --input_dir data/raw/images
```

输出图像数量、文件名示例和尺寸分布，适合先检查高分辨率数据是否被正确读取。

## 绘制 RGB 直方图

```bash
python scripts/plot_rgb_hist.py --input_dir data/raw/images -n 30
```

结果保存到：

```text
results/histograms/rgb_histogram.png
```

可用于观察数据是否整体偏蓝、偏绿或红通道衰减明显。

## 训练 baseline

```bash
python scripts/train.py --config configs/baseline.yaml
```

输出：

- `checkpoints/baseline/latest.pth`
- `checkpoints/baseline/best.pth`
- `results/baseline/loss_curve.png`
- `results/baseline/samples/epoch_xxx_comparison.jpg`

断点续训：

```bash
python scripts/train.py --config configs/baseline.yaml --resume checkpoints/baseline/latest.pth
```

## 训练 physics-guided 模型

```bash
python scripts/train.py --config configs/physics_guided.yaml
```

输出除增强样例外，还会保存：

- `epoch_xxx_coarse.jpg`
- `epoch_xxx_transmission.png`
- `epoch_xxx_A.png`

这些图可以展示物理先验分支的可解释性。

## 推理单张图

VS Code 中直接运行 `scripts/infer.py` 时，会使用默认配置：

```text
config: configs/physics_guided.yaml
checkpoint: checkpoints/physics_guided/best.pth，若不存在则尝试 latest.pth
input: data/raw/images
output: results/inference_physics_guided
```

因此训练完成后，最简单的推理方式是：

```bash
python scripts/infer.py
```

如果需要覆盖默认路径，再传命令行参数：

```bash
python scripts/infer.py ^
  --config configs/physics_guided.yaml ^
  --checkpoint checkpoints/physics_guided/best.pth ^
  --input data/raw/images/G0273464.JPG ^
  --output_dir results/inference_physics ^
  --tile_size 1024 ^
  --overlap 96 ^
  --save_metrics
```

输出：

- `*_enhanced.jpg`
- `*_compare.jpg`
- `*_coarse.jpg`
- `*_transmission.png`
- `*_A.png`
- `metrics.csv`，若开启 `--save_metrics`

`tile_size` 用于大图分块推理，可避免一次性把 5568×4872 图像放进显存。

## 推理整个文件夹

```bash
python scripts/infer.py ^
  --config configs/physics_guided.yaml ^
  --checkpoint checkpoints/physics_guided/best.pth ^
  --input data/raw/images ^
  --output_dir results/inference_physics
```

baseline 推理只需换配置和权重：

```bash
python scripts/infer.py ^
  --config configs/baseline.yaml ^
  --checkpoint checkpoints/baseline/best.pth ^
  --input data/raw/images ^
  --output_dir results/inference_baseline
```

## 对比 baseline 与 physics-guided

```bash
python scripts/compare_methods.py ^
  --input_dir data/raw/images ^
  --baseline_dir results/inference_baseline ^
  --physics_dir results/inference_physics ^
  --output_dir results/comparisons
```

## 当前版本局限性

- 无配对训练的目标是稳妥增强，不保证达到公开数据集 SOTA。
- `UIQM` 当前提供轻量 proxy，`UCIQE`、信息熵、RMS 对比度、颜色均衡误差已可用。
- 背景光 `A` 目前是全局 RGB 估计，后续可扩展为空间低频图。
- 感知损失接口预留在 supervised loss 位置，第一版没有强依赖 VGG，以保持安装和训练简单。

## 后续扩展方向

- 加入全量 UIQM 和更多无参考水下质量指标。
- 引入更稳健的红通道补偿先验，约束 refinement 不过度增红。
- 加入 paired/unpaired 混合训练策略。
- 支持 ONNX 导出和简历展示用 Web demo。
- 引入 EMA、学习率调度和更完整的实验记录。

