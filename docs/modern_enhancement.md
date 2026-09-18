# FGDPA 水下图像增强

本项目新增可直接运行的近期预训练增强方法，原有训练、传统算法和 VLM 入口继续保留。

## 方法与选择理由

采用作者发布的 [FGDPA 官方实现](https://github.com/LethyZhang/FGDPA)，作者标注为
ICME 2026 论文 Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention。
在当前环境中实际加载的 slim 模型为 **4,234 个参数**。它在低分辨率特征上计算 FFT 幅值，
生成频率引导的通道注意力，与空间注意力融合，改善颜色与细节。
部署网络使用训练后重参数化的卷积，不需要在本机重新训练。

选择它的原因是代码、可直接下载的小体积权重和许可齐全，并且本机无 CUDA 也能执行。
它不是对所有水下场景都最优的保证。另查阅了作者发布的
[MARINE-Net](https://github.com/swamynathanvp/MARINE-Net)（2026，物理引导 CNN-INR）和
[UIE_CLIP](https://github.com/Ave001025/UIE_CLIP)（2025，CLIP 引导），本次实现集中在 FGDPA。

源码固定到 `530c692feb99ecf6205bb929118a9167e2809d4d`；源码许可和出处见
`third_party/fgdpa/PROVENANCE.md` 与 `LICENSE`。作者内部类名仍为 `FGDRAUIENetS`。

## 运行

在项目根目录执行，使用已有 `aqua_align` 环境：

```powershell
conda activate aqua_align
python -m pip install -r requirements-local.txt  # 环境缺少依赖时执行
python scripts/download_fgdpa.py

# 单张图片：将路径换成自己的图片
python scripts/enhance_modern.py --input data/processed/uieb/raw/sample_0001.jpg --output-dir outputs/fgdpa_single --device cpu

# 递归处理目录全部图片
python scripts/enhance_modern.py --input data/raw/images --output-dir outputs/fgdpa_batch --device auto

# 可选配对参考图评价：相对路径和文件名须一致
python scripts/enhance_modern.py --input data/processed/uieb/raw --reference-dir data/processed/uieb/reference --output-dir outputs/fgdpa_eval --limit 10 --device cpu
```

当前机器也可直接调用 `C:/Users/LBY/.conda/envs/aqua_align/python.exe`，避免误用没有 torch 的 base 环境。
Python API 是 `aqua_align.modern_enhancement.FGDPAEnhancer`，接收 PIL 图像，返回 RGB PIL 图像。
可编辑安装时使用 `python -m pip install -e '.[enhance]'`；命令脚本从源码目录运行。

权重不纳入 Git，下载脚本固定来源并校验 SHA256；加载器也校验哈希，使用
`weights_only=True` 和严格参数匹配。`--checkpoint` 仅用于指定同一官方权重的其他存放位置。

## 输出及边界

- `enhanced/`：原分辨率 PNG；保留相对子目录及原始扩展名，避免同名 JPG/PNG 互相覆盖。
- `comparisons/`：原图、增强图以及可选参考图的缩略对比。
- `run_summary.json`：源码版本、权重哈希、参数数目、环境、耗时、每张图的路径及可选指标。
- 输出目录必须为空，且不能放在输入目录内部。再次执行需选择新目录。
- 默认 `--limit 0` 表示全部；`--threads 4` 控制 CPU 线程。
- `auto` 自动选择可用设备；显式指定不可用的 CUDA 会报错。
- 输入处理与官方保持 RGB / 255；推理使用 FP32，输出裁剪到 [0,1] 并四舍五入保存。
- 保留 EXIF 方向后的图像尺寸；不保留原图 EXIF。整图推理，不自动缩放或分块；超高分辨率图片会增加内存占用。
  由于模型包含全局统计和频域注意力，分块会改变算法结果。
- 单图失败时停止并将运行记录标记为 `failed`，已经生成的结果仍然保留。

## 本机实际演示（2026-09-10）

输入为本地 `data/processed/uieb/raw/` 按文件名排序的前 10 张，配对参考图来自
`data/processed/uieb/reference/`。CPU，PyTorch `2.13.0+cpu`，4 线程，无缩放。

| 指标 | 输入 | FGDPA |
|---|---:|---:|
| 平均 PSNR（dB） | 16.2729 | 26.1832 |
| 平均 SSIM | 0.8096 | 0.9442 |

10/10 张 PSNR 上升。图像增强调用平均 0.1249 秒，包含张量转换及输出量化，
不包含加载模型、读取文件、指标计算或保存文件；不是标准硬件速度基准。
指标由项目已有 `aqua_align.image_quality` 计算，包含 PNG 保存前的 uint8 量化。

**这不是独立测试集评价。** 官方权重在 UIEB 上训练，当前演示没有排除训练重叠，
本地 split 与作者 split 也不保证一致。若用于论文实验，须核对作者 `uieb_test.txt`
及本地 metadata 的源文件名，或者按同一划分重新训练，并在未见过的真实场景上评估。
不将本文的演示结果与其他论文的 PSNR/SSIM 排名直接比较。

完整逐图记录：`outputs/fgdpa_demo/run_summary.json`。
示例：`outputs/fgdpa_demo/comparisons/sample_0001.jpg.jpg`。

## 验证

新增测试检查真实权重与网络兼容性、非整倍数和 1×1 输入、灰度转换、
直接官方前向输出的一致性、缺失权重及损坏权重的错误行为。

```powershell
python -m pytest -q
```

未下载权重时，真实推理测试明确跳过，不伪造通过结果。
