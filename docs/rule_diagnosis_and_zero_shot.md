# AquaAlign-VLM：规则诊断与 Base VLM 零样本评测

## 阶段边界与当前状态

本阶段建立三类基线：图像统计规则诊断、规则到增强策略的映射、未经微调的 Base VLM 零样本评测。不会执行 SFT 或 DPO，也不会把规则标签或 Base VLM 输出称为人工真值。

上一阶段数据已检查：`metadata.csv` 共 950 张原图，`candidates.csv` 与 `quality_metrics.csv` 各 90 行；已检查的候选路径不存在缺失，train/val/test 未发现相同 `sample_id` 跨集合泄漏。本次规则 smoke test 使用 10 张真实 UIEB 原图。当前 `aqua_align` 环境的 PyTorch 是 CPU 构建，且未安装 Transformers，因此 Qwen3-VL 的真实推理数为 0。

## 代码框架

| 文件 | 作用 | 主要输出 |
|---|---|---|
| `aqua_align/degradation_features.py` | 从 RGB uint8 图像提取可审计统计特征 | 普通 Python 数值字典 |
| `aqua_align/rule_diagnosis.py` | 按 YAML 中的多指标阈值计算 0～3 严重度和 0～1 置信度 | 单图规则诊断 |
| `aqua_align/strategy_mapping.py` | 将活动退化映射为统一词表中的有序操作并消解冲突 | `recommended_pipeline` |
| `aqua_align/prompts.py` | 集中维护五类零样本 Prompt、JSON Schema 和图像数契约 | Prompt v0.1 |
| `aqua_align/parsing.py` | 提取 JSON、处理代码块/前后缀和尾逗号；不补造字段 | 解析状态、原始错误和修复类型 |
| `aqua_align/inference.py` | 延迟加载本地 Qwen3-VL，执行单图/多图推理 | 模型原始文本 |
| `scripts/calibrate_rules.py` | 仅用 train 集输出分布和建议分位数，不改正式规则 | `reports/rule_calibration.json` |
| `scripts/run_rule_diagnosis.py` | 批量提取特征、诊断并映射策略 | `data/processed/rule_diagnosis.jsonl` |
| `scripts/build_review_sheet.py` | 生成单图诊断与候选偏好人工审核表 | `data/annotations/rule_review.csv` |
| `scripts/run_base_vlm.py` | preflight、真实零样本推理、原文/解析/错误/环境留档 | `outputs/base_vlm/` |
| `evaluation/*.py` | 诊断、策略、排序和幻觉筛查 | JSON 评测报告 |
| `scripts/compare_rule_and_vlm.py` | 汇总规则、VLM 与人工审核证据 | `reports/baseline_comparison/` |

## 图像特征

颜色特征包括 RGB 均值和中位数、红通道衰减、蓝绿占优、红色过量、通道不平衡、Lab a/b、HSV 饱和度、高饱和比例和 colorfulness。亮度特征包括灰度均值/中位数、Lab L、暗/极暗/亮像素比例和 P95-P5 动态范围。对比度特征包括灰度标准差、分位数差、RMS contrast、局部对比度和熵。清晰度/噪声特征包括 Laplacian variance、Sobel 梯度、高频能量和噪声代理。`haze_proxy` 综合暗通道、局部对比度和边缘密度，仅作为辅助量，不能解释为真实后向散射测量。

所有输入先验证并转换为独立 RGB uint8 数组，中间运算使用浮点数；输出在写 JSON 前检查有限性。清晰度低仅表示可能模糊或细节不足，不能证明相机失焦。

## 规则和阈值来源

全部阈值位于 `configs/diagnosis_rules.yaml`，Python 中不保存诊断阈值。当前版本 `0.1` 的来源标记为 `initial_heuristics_pending_train_split_review`：它们是用于流程验证的工程初值，不是物理常数，也没有经过人工标签验证。

每类至少需要两个指标同时触发。例如低照度使用亮度均值（115/85/55）、暗像素比例（0.20/0.40/0.65）和极暗比例；低对比度使用灰度标准差（52/38/24）、分位数范围和 RMS contrast；模糊使用 Laplacian variance（180/90/40）和 Sobel 梯度。完整方向、权重和阈值以 YAML 为准。

`calibrate_rules.py` 仅允许 `--split train`。报告中的 P10/P25/P40/P60/P75/P90 是人工复核的参考，不会自动写回正式规则。阈值定稿仍需要查看典型图像和完成人工审核。

## 策略映射与冲突

统一操作词表为：`preserve_original`、`denoise`、`red_channel_compensation`、`white_balance`、`saturation_reduction`、`gamma_correction`、`retinex`、`clahe`、`mild_sharpening`。常规顺序是先去噪和色彩校正，再调整照度/对比度，最后才考虑轻度锐化。

- 有噪声时强制先去噪，移除锐化，并把 CLAHE clip limit 限制到 1.8。
- 过饱和时先降低饱和度并移除 Retinex，避免进一步放大色彩。
- 红偏色与蓝绿偏色同时触发时，禁止红通道补偿，只保留白平衡。
- 没有退化达到阈值时保留原图。

策略仍是待审核候选，不代表最佳增强参数。

## PowerShell 运行方法

PowerShell 多行续行符是反引号 `` ` ``，不是 Linux shell 的反斜杠 `\`。也可以将命令写成单行。

```powershell
conda activate aqua_align

python scripts/calibrate_rules.py `
  --metadata data/processed/uieb/metadata.csv `
  --split train `
  --output reports/rule_calibration.json

python scripts/run_rule_diagnosis.py `
  --metadata data/processed/uieb/metadata.csv `
  --rules configs/diagnosis_rules.yaml `
  --output data/processed/rule_diagnosis.jsonl

python scripts/build_review_sheet.py `
  --diagnosis data/processed/rule_diagnosis.jsonl `
  --candidates data/processed/candidates.csv `
  --output data/annotations/rule_review.csv
```

服务器安装 CUDA 版 PyTorch 后，再安装 `requirements-server.txt`。Qwen3-VL 的官方 Transformers 用法要求较新的 Transformers；本项目配置要求本地已有权重，默认 `local_files_only: true`，所以不会在运行时静默下载。

```powershell
python scripts/run_base_vlm.py `
  --config configs/base_vlm.yaml `
  --metadata data/processed/uieb/metadata.csv `
  --candidates data/processed/candidates.csv `
  --split test `
  --tasks diagnose strategy compare rank `
  --limit 10 `
  --output-dir outputs/base_vlm

python scripts/compare_rule_and_vlm.py `
  --rule-results data/processed/rule_diagnosis.jsonl `
  --vlm-results outputs/base_vlm/diagnose_parsed.jsonl `
  --human-review data/annotations/rule_review.csv `
  --output reports/baseline_comparison
```

排序任务严格按“图1原图、图2候选 A、图3候选 B”传入。推理目录保留环境、实际命令、配置、四类原始/解析 JSONL、错误和汇总。JSON 解析失败仍保留模型原文，不会生成缺失字段。

## 评测解释

有已完成的人工审核时，才计算诊断 Precision/Recall/F1、Macro-F1、severity MAE、策略 top-k agreement 和 human preference agreement。只有规则标签时使用 `rule_agreement`，不能称为准确率。没有人工标签时只输出覆盖率、格式成功率、字段完整率、频率和 `rule_statistics` 性质的描述。

幻觉脚本是透明的关键词筛查，不是语义真值检测；它报告无依据的水深、海域、相机、指标数值、鲜艳度唯一理由、可疑物体断言和图像顺序混淆。实际发现需人工复核。

## 下一步前置条件

先在审核表中完成至少 50 张单图和 100 个候选对的人工审核，再由用户审查 train-only 校准报告并决定是否调整阈值。随后在 A40 服务器准备 CUDA PyTorch、Transformers 与 Qwen3-VL-4B-Instruct 权重，先跑 5～10 个 smoke 样本并核对多图顺序、显存、格式和失败留档；通过后固定 Prompt 和规则，仅运行 test。完成这些证据前不应开始 SFT 数据固化。
