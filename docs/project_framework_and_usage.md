# AquaAlign-VLM 项目框架、代码职责与运行指南

## 1. 项目定位

AquaAlign-VLM 当前由三层能力组成：

1. **传统水下图像增强数据流水线**：不需要训练，通过多种传统算法为每张水下原图生成候选增强图。
2. **轻量学习型水下增强网络**：包括纯学习残差网络和物理先验引导网络。
3. **后续视觉语言模型接口**：计划基于增强候选、质量指标和人工偏好构建VLM SFT与DPO数据；当前尚未进入训练阶段。

目前完整验证的是传统增强流水线。视觉语言模型、SFT和DPO还没有开始下载或训练。

更详细的算法公式、实测数据和后续研究路线参见：

- [水下图像增强算法实现与后续方向](underwater_enhancement_methods.md)

## 2. 总体数据流

```text
UIEB原图和参考图
        │
        ▼
数据整理 prepare_uieb.py
        │
        ├── metadata.csv
        └── train/val/test原图级划分
        │
        ▼
传统增强 generate_candidates.py
        │
        ├── Gray World
        ├── White Patch
        ├── CLAHE
        ├── Gamma × 3
        ├── Simplest Color Balance
        ├── Multi-Scale Retinex
        └── White Balance + CLAHE
        │
        ▼
候选图 + candidates.csv
        │
        ├──────────────┐
        ▼              ▼
质量指标            可视化总览
PSNR/SSIM等         原图/参考图/候选图
        │              │
        ▼              ▼
quality_metrics.csv  reports/figures/
        │
        ▼
后续SFT诊断数据和DPO偏好数据
（当前尚未进入训练阶段）
```

## 3. 当前数据状态

UIEB目录结构：

```text
data/raw/uieb/
├── raw-890/          # 890张原始水下图
├── reference-890/    # 890张参考增强图
└── challenging-60/   # 60张无参考困难图
```

经过整理后的统计：

```text
总样本：950
有参考图：890
无参考图：60
train：665
val：143
test：142
```

当前真实UIEB smoke test只对前10张有参考图生成了候选：

```text
原图：10
每图候选：9
候选总数：90
PSNR有效值：90
SSIM有效值：90
指标错误：0
```

这些结果用于验证工程流程，不代表完整UIEB benchmark。

## 4. 核心目录与文件

### 4.1 `aqua_align/`：AquaAlign核心模块

#### `aqua_align/config.py`

负责配置与路径管理：

- 读取UTF-8 YAML；
- 检查YAML根节点是否为字典；
- 返回项目根目录；
- 将配置中的相对路径解析成绝对路径；
- 将输出路径保存为项目相对路径；
- 拒绝把项目外路径写成数据集相对路径。

典型调用：

```python
from aqua_align.config import load_config

config = load_config("configs/data.yaml")
```

输出示例：

```python
{
    "candidate_generation": {...},
    "quality_metrics": {...},
}
```

#### `aqua_align/dataset.py`

负责UIEB数据发现、校验和整理。

主要功能：

- 递归发现常见图像格式；
- 自动识别`raw-890`、`reference-890`、`challenging-60`；
- 按文件名stem匹配原图和参考图；
- 检查重复文件名；
- 使用Pillow完整解码，检查损坏图像；
- 890张配对图标记为`has_reference=true`；
- 60张困难图标记为`has_reference=false`；
- 有参考图样本优先编号，无参考样本随后编号；
- 按原图级别划分train/val/test；
- 输出统一的`metadata.csv`。

它保证同一原图及其所有派生数据始终属于同一个split，避免数据泄漏。

#### `aqua_align/enhancement.py`

传统增强算法的核心实现。

统一数据约定：

```text
输入：RGB numpy.ndarray
输出：RGB uint8
像素范围：0～255
空间尺寸：与输入一致
```

支持：

- 灰度图；
- H×W×1单通道图；
- RGB图；
- RGBA图；
- `[0,1]`浮点图；
- `[0,255]`浮点或整数图。

会拒绝：

- 空图；
- 非数值数组；
- NaN和Infinity；
- 非法通道数；
- 越界数值；
- 非法Gamma和算法参数。

所有中间乘除在浮点类型中完成，避免`uint8`直接运算溢出，并且不会原地修改输入图像。

#### `aqua_align/candidates.py`

负责批量候选生成：

- 读取`metadata.csv`；
- 根据`configs/data.yaml`展开算法和参数；
- 为每张图生成多个候选；
- 支持多线程；
- 保存原图、参考图和候选图；
- 将参数保存为合法JSON；
- 输出`candidates.csv`；
- 将原图split传播到全部派生候选。

支持的控制参数：

```text
--limit
--workers
--seed
--dry-run
--overwrite
```

默认不覆盖已有结果。

#### `aqua_align/image_quality.py`

负责有参考和无参考质量指标。

有参考图时：

- PSNR；
- SSIM。

所有图像均计算：

- mean brightness；
- grayscale standard deviation；
- Laplacian variance；
- dark pixel ratio；
- saturated pixel ratio；
- R/G/B channel mean；
- channel imbalance；
- colorfulness。

特殊处理：

- 完全相同图像的PSNR保存为100 dB，避免Infinity；
- SSIM要求至少3×3；
- 无参考图的PSNR和SSIM留空；
- 失败指标保留错误信息；
- 禁止输出NaN和Infinity。

#### `aqua_align/utils.py`

提供：

- Python、NumPy和PyTorch随机种子设置；
- 控制台日志；
- UTF-8文件日志；
- 幂等的logger handler初始化。

### 4.2 `scripts/`：命令行入口

#### `scripts/prepare_uieb.py`

输入：

```text
data/raw/uieb/
```

输出：

```text
data/processed/uieb/
├── raw/
├── reference/
├── metadata.csv
└── prepare_uieb.log
```

`metadata.csv`主要字段：

```text
sample_id
raw_image
reference_image
has_reference
split
source_dataset
source_filename
source_stem
```

#### `scripts/generate_candidates.py`

输入：

- `metadata.csv`；
- `configs/data.yaml`；
- 输出目录和运行参数。

输出示例：

```text
data/candidates/uieb/sample_0001/
├── raw.jpg
├── reference.jpg
├── gray_world.jpg
├── white_patch.jpg
├── clahe.jpg
├── gamma_0.6.jpg
├── gamma_0.8.jpg
├── gamma_1.2.jpg
├── color_balance.jpg
├── retinex.jpg
└── wb_clahe.jpg
```

同时生成：

```text
data/processed/candidates.csv
```

主要字段：

```text
sample_id
split
method
parameters
raw_image
candidate_image
reference_image
has_reference
```

#### `scripts/compute_quality_metrics.py`

读取：

```text
data/processed/candidates.csv
```

输出：

```text
data/processed/quality_metrics.csv
data/processed/quality_metrics.log
```

每一行对应一个候选图。如果参考图存在，则额外计算PSNR和SSIM。

#### `scripts/create_comparison_figures.py`

将以下内容排入同一张图：

- 原图；
- 参考图；
- 所有候选图；
- 方法名称；
- 可用时显示PSNR和SSIM。

输出示例：

```text
reports/figures/uieb/sample_0001_overview.jpg
```

脚本使用matplotlib的`Agg`非交互式后端，不会调用`plt.show()`，并在每张图保存后关闭figure。

#### `scripts/check_environment.py`

输出：

- Python版本；
- PyTorch版本；
- CUDA是否可用；
- CUDA版本；
- GPU名称和显存；
- transformers、ms-swift、peft、trl版本；
- bitsandbytes是否可用；
- bf16是否支持；
- 缺失依赖和建议安装命令。

### 4.3 `configs/`：配置文件

#### `configs/data.yaml`

控制：

- 数据集划分比例；
- JPEG质量；
- 各算法是否启用；
- Gray World最大增益；
- White Patch百分位；
- CLAHE参数；
- Gamma候选值；
- Color Balance裁剪比例；
- Retinex高斯尺度；
- 暗像素和饱和像素阈值；
- 完全相同图像的PSNR上限。

#### `configs/baseline.yaml`

控制纯学习baseline网络的模型、数据、batch size、训练轮数、学习率和损失权重。

#### `configs/physics_guided.yaml`

控制物理引导网络的背景光分支、传输图分支、细化网络、训练参数和损失权重。

#### `configs/project.yaml`

记录项目名称、随机种子、默认VLM名称和公共输出目录。默认模型名称只作为配置保存，当前阶段不会自动下载。

## 5. 当前传统增强方法

### 5.1 Gray World

统计RGB三个通道的全局均值，通过通道增益使它们趋于一致。

主要用途：

- 修正蓝绿色偏色；
- 补偿红通道整体衰减。

主要风险：

- 场景不满足灰世界假设时可能错误校色；
- 可能放大红通道噪声；
- 无法恢复已经物理丢失的红通道细节。

### 5.2 White Patch

使用各通道99.5%百分位估计白点，再将该值缩放到255。

主要用途：

- 白平衡；
- 提升亮度；
- 减少单个噪点对最大值的影响。

主要风险：

- 图中没有真实白色区域时可能误判；
- 强高光可能导致整体缩放不自然；
- 部分结果偏亮或偏青。

### 5.3 CLAHE

OpenCV可用时执行：

```text
RGB → LAB → 增强L亮度通道 → RGB
```

主要用途：

- 提升局部对比度；
- 增强目标边缘和暗部纹理。

主要风险：

- 不能独立消除蓝绿色偏色；
- 参数过强会放大噪声和后向散射；
- tile设置不合理时可能出现局部不自然。

OpenCV不可用时，项目使用SciPy后备实现，执行分块限幅直方图均衡和LUT双线性混合。

### 5.4 Gamma Correction

公式：

```text
output = 255 × (input / 255) ^ gamma
```

当前候选：

```text
gamma_0.6
gamma_0.8
gamma_1.2
```

- `gamma < 1`提亮；
- `gamma > 1`压暗；
- Gamma主要改变亮度，不能独立恢复颜色或去雾。

### 5.5 Simplest Color Balance

分别裁剪各通道两端少量像素，再线性拉伸到0～255。

当前候选名：

```text
color_balance
```

它在当前10张UIEB小样本中的平均PSNR和SSIM最高，但不能据此认定为完整数据集的绝对最优方法。

### 5.6 Multi-Scale Retinex

使用三个高斯尺度：

```yaml
sigmas: [15.0, 80.0, 250.0]
```

基本计算：

```text
log(原图) - log(高斯模糊照明)
```

主要用途：

- 估计和消除不均匀照明；
- 提升局部对比度；
- 增强暗部。

当前问题：

- 容易出现红色或紫色过补偿；
- 可能过亮；
- 可能产生光晕和噪声放大；
- 是当前传统候选中耗时最大的算法。

### 5.7 White Balance + CLAHE

组合流程：

```text
Gray World白平衡 → CLAHE局部对比度增强
```

它同时处理偏色和低对比度，但也可能叠加白平衡误差、噪声和局部过增强。

## 6. 传统算法配置方法

配置文件：`configs/data.yaml`。

```yaml
candidate_generation:
  jpeg_quality: 95
  methods:
    gray_world:
      enabled: true
      max_gain: 4.0
    white_patch:
      enabled: true
      percentile: 99.5
      max_gain: 4.0
    clahe:
      enabled: true
      clip_limit: 2.0
      tile_grid_size: 8
    gamma:
      enabled: true
      values: [0.6, 0.8, 1.2]
    simplest_color_balance:
      enabled: true
      percent: 1.0
    retinex:
      enabled: true
      sigmas: [15.0, 80.0, 250.0]
      low_percentile: 1.0
      high_percentile: 99.0
    white_balance_clahe:
      enabled: true
      max_gain: 4.0
      clip_limit: 2.0
      tile_grid_size: 8
```

关闭某种方法：

```yaml
retinex:
  enabled: false
```

增加Gamma候选：

```yaml
gamma:
  enabled: true
  values: [0.5, 0.6, 0.8, 1.0, 1.2]
```

## 7. 可训练增强网络

传统候选之外，仓库还保留两套PyTorch增强网络。

### 7.1 `BaselineEnhancementNet`

文件：`models/baseline_net.py`。

结构：

```text
输入图
  ↓
轻量Encoder-Decoder
  ↓
预测受限残差
  ↓
原图 + 残差
  ↓
最终增强图
```

计算：

```text
residual = residual_scale × tanh(network(image))
enhanced = clip(image + residual, 0, 1)
```

默认`residual_scale=0.2`。该网络没有显式水下成像先验，主要作为纯学习对照组。

### 7.2 `PhysicsGuidedEnhancementNet`

文件：`models/physics_guided_net.py`。

使用简化水下/雾化成像模型：

```text
I(x) = J(x)t(x) + A(1-t(x))
```

其中：

- `I`：观测水下图；
- `J`：待恢复图；
- `t(x)`：传输图；
- `A`：全局背景光。

网络流程：

```text
原图
 ├── BackgroundLightEstimator → A
 ├── TransmissionEstimator → t
 └── 物理粗恢复 → coarse
                  │
原图 + coarse + t │
        ↓
轻量Encoder-Decoder
        ↓
细化残差
        ↓
最终增强图
```

粗恢复：

```text
coarse = clip((I - A) / max(t, eps) + A, 0, 1)
```

输出：

```python
{
    "estimated_A": ...,
    "estimated_A_map": ...,
    "estimated_t": ...,
    "coarse_restored": ...,
    "final_enhanced": ...,
}
```

### 7.3 公共网络模块

文件：`models/modules.py`。

包含：

- Depthwise separable convolution；
- Residual depthwise-separable block；
- Channel attention；
- 两级下采样；
- 双线性上采样；
- Encoder skip connection；
- 轻量Encoder-Decoder。

## 8. 训练损失

组合损失位于`losses/total_loss.py`：

```text
总损失 =
颜色恒常损失
+ 曝光损失
+ 梯度一致性损失
+ 边缘损失
+ TV平滑损失
+ 物理约束损失
+ 可选配对监督损失
```

对应文件：

- `losses/color_loss.py`：颜色均衡和曝光；
- `losses/structure_loss.py`：梯度、Sobel边缘、TV和SSIM；
- `losses/physics_loss.py`：传输图范围、平滑度和背景光约束；
- `losses/total_loss.py`：按配置组合全部损失。

当前默认配对L1、Charbonnier、SSIM权重为0。只有配置目标图并打开相应权重后才会参与训练。

## 9. 环境安装

项目本地和服务器统一使用名为`aqua_align`的Conda环境。

### 9.1 本地图像增强环境

```powershell
conda create -n aqua_align python=3.11 -y
conda activate aqua_align
python -m pip install -r requirements-local.txt
```

如果环境已经创建，只需执行：

```powershell
conda activate aqua_align
```

### 9.2 A40服务器训练环境

先安装与服务器驱动匹配的CUDA版PyTorch，再安装：

```bash
python -m pip install -r requirements-server.txt
python scripts/check_environment.py
```

不能使用带`+cpu`的PyTorch进行QLoRA、SFT或DPO。

## 10. PowerShell完整运行方式

PowerShell多行命令使用反引号 `` ` ``，不能使用Linux Bash的反斜杠 `\`。

### 10.1 检查环境

```powershell
conda activate aqua_align
python scripts/check_environment.py
```

### 10.2 整理完整UIEB

```powershell
python scripts/prepare_uieb.py `
  --input-dir data/raw/uieb `
  --output-dir data/processed/uieb `
  --seed 42 `
  --source-dataset UIEB `
  --overwrite
```

单行形式：

```powershell
python scripts/prepare_uieb.py --input-dir data/raw/uieb --output-dir data/processed/uieb --seed 42 --source-dataset UIEB --overwrite
```

### 10.3 候选生成dry-run

```powershell
python scripts/generate_candidates.py `
  --metadata data/processed/uieb/metadata.csv `
  --config configs/data.yaml `
  --output-dir data/candidates/uieb `
  --output-csv data/processed/candidates.csv `
  --limit 10 `
  --workers 2 `
  --seed 42 `
  --dry-run
```

`--dry-run`只打印计划，不写候选目录和CSV。

### 10.4 生成前10张候选

```powershell
python scripts/generate_candidates.py `
  --metadata data/processed/uieb/metadata.csv `
  --config configs/data.yaml `
  --output-dir data/candidates/uieb `
  --output-csv data/processed/candidates.csv `
  --limit 10 `
  --workers 2 `
  --seed 42 `
  --overwrite
```

### 10.5 计算质量指标

```powershell
python scripts/compute_quality_metrics.py `
  --candidates data/processed/candidates.csv `
  --output data/processed/quality_metrics.csv `
  --config configs/data.yaml
```

### 10.6 生成对比图

```powershell
python scripts/create_comparison_figures.py `
  --candidates data/processed/candidates.csv `
  --metrics data/processed/quality_metrics.csv `
  --output-dir reports/figures/uieb `
  --limit 10
```

### 10.7 运行测试

```powershell
python -m pytest -q
```

当前测试结果：

```text
27 passed
```

测试不需要GPU，也不会下载大型模型。

## 11. 学习型增强网络运行方式

### 11.1 训练physics-guided模型

```powershell
python scripts/train.py `
  --config configs/physics_guided.yaml
```

输出：

```text
checkpoints/physics_guided/
├── latest.pth
└── best.pth

results/physics_guided/
├── train.log
├── config_used.json
├── loss_history.json
├── loss_curve.png
└── samples/
```

这一步训练的是水下增强网络，不是VLM SFT。

### 11.2 physics-guided推理

```powershell
python scripts/infer.py `
  --config configs/physics_guided.yaml `
  --checkpoint checkpoints/physics_guided/best.pth `
  --input data/raw/images_gopro `
  --output_dir results/inference_physics_guided `
  --save_metrics
```

输出：

```text
*_enhanced.jpg
*_compare.jpg
*_coarse.jpg
*_transmission.png
*_A.png
metrics.csv
```

## 12. 当前产物位置

### 数据整理

```text
data/processed/uieb/metadata.csv
```

### 候选索引

```text
data/processed/candidates.csv
```

### 质量指标

```text
data/processed/quality_metrics.csv
```

### 候选图

```text
data/candidates/uieb/sample_xxxx/
```

### 总览图

```text
reports/figures/uieb/sample_xxxx_overview.jpg
```

### 学习型模型权重

```text
checkpoints/physics_guided/best.pth
checkpoints/physics_guided/latest.pth
```

### 学习型推理结果

```text
results/inference_physics_guided/
```

## 13. 当前完成情况

已经完成：

- UIEB 890对配对数据识别；
- challenging-60无参考数据处理；
- train/val/test原图级划分；
- 7类传统算法和9个候选；
- PSNR和SSIM；
- 无参考质量辅助指标；
- candidates.csv和quality_metrics.csv；
- matplotlib总览图；
- 纯CPU单元测试；
- 真实UIEB前10张smoke test；
- 轻量增强模型代码和已有checkpoint。

尚未完成：

- 对完整890对图像生成所有候选；
- 完整UIEB benchmark；
- 无损PNG指标评测；
- SFT结构化诊断数据；
- DPO chosen/rejected偏好数据；
- Qwen3-VL训练；
- Base/SFT/DPO正式比较；
- VLM Gradio诊断应用。

## 14. 推荐执行顺序

```text
1. 整理完整UIEB metadata
2. 用--limit 10完成候选smoke test
3. 检查总览图和异常结果
4. 改用无损PNG进行正式指标评测
5. 对完整890对运行全部候选
6. 分析每种方法的指标分布和失败案例
7. 加入人工审核和伪影检测
8. 构建SFT结构化诊断数据
9. 构建高置信度DPO偏好数据
10. 最后进行VLM SFT和DPO
```

当前传统增强模块的目标不是提前认定某种算法绝对最好，而是产生具有不同特征的候选。后续必须综合参考指标、无参考统计、伪影检测和人工偏好，才能形成可信的SFT与DPO训练数据。
