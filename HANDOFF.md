# AquaAlign-VLM 项目交接文档

更新时间：2026-07-25  
项目根目录：`D:\study\02_science_product\underwater_enhancement`  
本地 Conda 环境：`aqua_align`

## 1. 当前任务目标

项目总体目标是构建 AquaAlign-VLM 水下图像增强与多模态对齐系统。

当前已推进到以下阶段：

1. 整理 UIEB 原图和参考图；
2. 使用多种传统水下图像增强算法生成候选图；
3. 计算有参考和无参考图像质量指标；
4. 提取图像统计退化特征；
5. 建立规则化退化诊断基线；
6. 建立退化到增强策略的映射；
7. 生成人工诊断和候选偏好审核表；
8. 实现 Base VLM 零样本单图、多图推理入口；
9. 实现 JSON 解析、错误留档、诊断/策略/排序/幻觉评测和基线对比报告。

当前阶段明确禁止：

- 开始 SFT；
- 开始 DPO；
- 使用 test 集调整规则阈值或 Prompt；
- 把规则标签称为人工真值；
- 把 Base VLM 输出直接作为训练真值；
- 在没有真实模型推理时伪造输出或实验结论。

## 2. 数据与环境

### 2.1 UIEB

原始 UIEB 路径：

```text
D:\study\02_science_product\underwater_enhancement\data\raw\uieb\raw-890
```

UIEB 当前包含 reference。上一阶段已经生成：

```text
data/processed/uieb/metadata.csv
data/processed/candidates.csv
data/processed/quality_metrics.csv
data/candidates/
reports/figures/
```

已检查的状态：

- `metadata.csv`：950 行；
- `candidates.csv`：90 行；
- `quality_metrics.csv`：90 行；
- 已检查候选路径缺失数：0；
- 已检查 `sample_id` 跨 train/val/test 泄漏数：0；
- 当前候选图只覆盖少量样本，并不是完整 950 张 UIEB 的全部候选。

### 2.2 本地环境

本地环境名称已经改为：

```powershell
conda activate aqua_align
```

最近一次检查结果：

- Python 3.11；
- PyTorch 是 CPU 构建；
- `torch.cuda.is_available()` 为 `false`；
- 本地未安装可用于 Qwen3-VL 的 Transformers 依赖；
- 本地没有 Qwen3-VL 模型权重。

因此本地只适合传统增强、规则诊断、数据整理和单元测试，不适合真实 Base VLM 推理。

## 3. 已完成内容

### 3.1 传统增强和质量指标

已有实现：

- `gray_world`
- `white_patch`
- `clahe`
- `gamma_correction`
- `simplest_color_balance`
- `retinex`
- `white_balance_clahe`

相关文件：

```text
aqua_align/enhancement.py
aqua_align/image_quality.py
aqua_align/candidates.py
scripts/prepare_uieb.py
scripts/generate_candidates.py
scripts/compute_quality_metrics.py
scripts/create_comparison_figures.py
configs/data.yaml
```

方法、公式和当前运行方式已经写入：

```text
docs/underwater_enhancement_methods.md
docs/project_framework_and_usage.md
```

### 3.2 规则化退化特征

文件：

```text
aqua_align/degradation_features.py
```

实现的特征包括：

- RGB 均值和中位数；
- 红通道衰减；
- 蓝绿通道占优；
- 红色过量；
- RGB 通道不平衡；
- Lab a/b 均值；
- HSV 饱和度均值；
- 高饱和像素比例；
- colorfulness；
- 灰度均值和中位数；
- Lab L 均值；
- 暗、极暗和亮像素比例；
- 动态范围；
- 灰度标准差；
- P95-P5 分位数差；
- RMS contrast；
- 局部对比度；
- 熵；
- Laplacian variance；
- Sobel gradient mean；
- 高频能量；
- noise sigma proxy；
- 暗通道；
- 边缘密度；
- `haze_proxy`。

输入会进行检查，输出为有限的普通 Python 数值，可直接写入 JSON。低清晰度只代表“可能模糊或细节不足”，代码和文档均没有将其解释为确定的相机失焦。

### 3.3 规则诊断

相关文件：

```text
configs/diagnosis_rules.yaml
aqua_align/rule_diagnosis.py
scripts/calibrate_rules.py
scripts/run_rule_diagnosis.py
```

支持的退化：

- `blue_green_color_cast`
- `red_color_cast`
- `low_light`
- `low_contrast`
- `blur`
- `possible_haze`
- `over_saturation`
- `under_saturation`
- `possible_noise`
- `possible_detail_loss`

所有正式诊断阈值都在 YAML 中，没有散落在 Python 文件中。

当前规则版本：

```text
version: 0.1
threshold_source: initial_heuristics_pending_train_split_review
```

规则使用多个指标联合触发；严重度只取 0、1、2、3，置信度限制在 0～1。输出保留实际特征值、阈值、触发方向、判定理由和局限性，并标记：

```text
diagnosis_source=rule_based
annotation_source=rule_based
needs_review=true
```

`calibrate_rules.py` 强制只允许 `--split train`，只输出分布和建议分位数，不自动修改正式规则。

已经用真实 UIEB 原图运行：

```text
data/processed/rule_diagnosis.jsonl
```

结果：

- 处理 50 张；
- 成功 50 张；
- 失败 0 张。

### 3.4 增强策略映射

相关文件：

```text
configs/strategy_mapping.yaml
aqua_align/strategy_mapping.py
```

统一操作词表：

- `preserve_original`
- `denoise`
- `red_channel_compensation`
- `white_balance`
- `saturation_reduction`
- `gamma_correction`
- `retinex`
- `clahe`
- `mild_sharpening`

已经实现冲突处理：

- 噪声存在时先去噪；
- 噪声存在时移除锐化；
- 噪声存在时限制 CLAHE 强度；
- 过饱和时移除 Retinex；
- 红偏色存在时禁止继续做红通道补偿；
- 没有活动退化时保留原图。

输出包含步骤编号、统一操作名、参数、触发原因、冲突删除记录、警告和 `strategy_source=rule_mapping`。

### 3.5 人工审核表

相关文件：

```text
scripts/build_review_sheet.py
data/annotations/rule_review.csv
data/annotations/rule_review.summary.json
```

当前审核表：

- 单图退化审核：50 行；
- 候选图偏好对审核：100 行；
- 总计：150 行；
- 人工字段全部为空；
- 已完成人工审核数量：0；
- 明确标记规则结果不是 ground truth。

审核表包含原图路径，审核人员不会只看到规则结论。

### 3.6 Base VLM 零样本入口

相关文件：

```text
configs/base_vlm.yaml
aqua_align/prompts.py
aqua_align/inference.py
aqua_align/parsing.py
scripts/run_base_vlm.py
```

配置模型：

```text
Qwen/Qwen3-VL-4B-Instruct
```

配置：

```text
device=cuda
torch_dtype=bfloat16
max_new_tokens=768
temperature=0.0
do_sample=false
prompt_version=v0.1
local_files_only=true
```

支持：

- 单图退化诊断；
- 单图增强策略推荐；
- 原图与一个增强图比较；
- 原图、候选 A、候选 B 三图排序。

三图排序的顺序必须保持：

```text
图1：原图
图2：候选 A
图3：候选 B
```

所有长 Prompt 集中在 `aqua_align/prompts.py`，没有在脚本中复制，也没有放 few-shot 示例。

解析器支持：

- 直接 JSON；
- Markdown JSON 代码块；
- JSON 前后无关文本；
- 最小的尾逗号修复；
- 失败后保留原文和错误；
- 不补造模型没有输出的字段。

### 3.7 评测与基线报告

相关文件：

```text
evaluation/common.py
evaluation/evaluate_diagnosis.py
evaluation/evaluate_strategy.py
evaluation/evaluate_ranking.py
evaluation/evaluate_hallucination.py
scripts/compare_rule_and_vlm.py
reports/baseline_comparison/
```

对比报告要求的文件已生成：

```text
reports/baseline_comparison/
├── metrics.csv
├── diagnosis_comparison.csv
├── strategy_comparison.csv
├── ranking_comparison.csv
├── failure_cases.csv
├── baseline_report.md
└── figures/
```

没有人工标签时，评测不会输出真实准确率。只有规则标签时使用 `rule_agreement`，并明确说明其不是 accuracy。

幻觉评测目前是透明的关键词筛查，不是语义真值检测。它检查水深、海域、相机型号、编造指标、仅以鲜艳度判断质量、可疑物体断言和图像顺序混淆。

### 3.8 测试

最后一次完整运行：

```powershell
python -m pytest -q
```

结果：

```text
58 passed in 5.82s
```

同时执行过：

```powershell
python -m compileall -q aqua_align scripts evaluation
```

编译检查通过。

## 4. 当前卡在哪里

### 4.1 Base VLM 真实推理被环境阻塞

已经实际执行过 Base VLM smoke 命令，但 preflight 返回：

```text
status=blocked
successful_inferences=0
failed_inferences=0
blocker=config requires CUDA but torch.cuda.is_available() is false
fake_outputs_created=false
```

完整记录：

```text
outputs/base_vlm/environment.json
outputs/base_vlm/config.yaml
outputs/base_vlm/command.txt
outputs/base_vlm/run_summary.json
outputs/base_vlm/errors.jsonl
```

这不是模型推理失败，而是模型尚未加载前的环境阻塞。当前没有真实模型回答，因此：

- 推理样本数为 0；
- JSON 解析成功率为“不适用”；
- 不能总结 Qwen3-VL 的诊断能力；
- 不能总结其幻觉率或排序准确率；
- 不能声称 Base VLM smoke test 成功。

### 4.2 规则阈值尚未完成可靠校准

当前只生成了 10 张 train 图像的校准 smoke 报告：

```text
reports/rule_calibration_smoke.json
```

它仅用于检查流程，不是完整 train 校准。

50 张规则诊断中：

- `possible_haze`：50/50 触发；
- `low_contrast`：45/50 触发；
- `blue_green_color_cast`：44/50 触发。

这表明当前初始规则偏敏感，尤其是 `possible_haze` 很可能过度触发。上述数字只是规则输出频率，不是 UIEB 真实退化分布。

### 4.3 人工审核尚未开始

虽然已经准备 150 行审核数据，但：

```text
已完成人工审核数量 = 0
```

因此当前不能计算：

- 真实诊断 Precision/Recall/F1；
- severity MAE；
- 策略 top-k 人工一致率；
- 人工偏好排序准确率；
- 可靠的规则阈值修订方案。

### 4.4 候选图覆盖范围较小

当前 `candidates.csv` 只有 90 行，候选图主要覆盖约 10 个样本。偏好审核表虽然生成了 100 对，但来自有限样本和方法组合，不能代表完整 UIEB 分布。

## 5. 下一步

必须按照以下顺序继续。

### 第一步：完整 train-only 校准

在本地 `aqua_align` 环境运行：

```powershell
conda activate aqua_align

python scripts/calibrate_rules.py `
  --metadata data/processed/uieb/metadata.csv `
  --split train `
  --output reports/rule_calibration.json
```

如果文件已经存在且确认需要重算，再加：

```powershell
--overwrite
```

不要自动把建议阈值写回 `diagnosis_rules.yaml`。先检查分布，再人工查看典型图像。

### 第二步：完成人工审核

审核文件：

```text
data/annotations/rule_review.csv
```

需要至少完成：

- 50 张单图退化审核；
- 100 个候选偏好对审核。

严重度统一填写 0～3。审核人员必须同时查看原图/候选图，不能只看规则结论。

### 第三步：审查并更新规则

结合：

- 完整 train-only 分布；
- 典型图片；
- 人工审核标签；
- 多个指标联合判断。

优先处理：

1. `possible_haze` 过度触发；
2. `low_contrast` 过度触发；
3. 蓝绿偏色阈值是否对 UIEB 过敏；
4. blur、detail loss、noise 之间的混淆。

规则更新后：

- 增加规则版本号；
- 更新 `threshold_source`；
- 保存修改依据；
- 重新运行 train/val 检查；
- 不使用 test 集继续调规则。

### 第四步：准备 A40 服务器

A40 可以运行 Qwen3-VL-4B-Instruct 的零样本推理，也适合后续 QLoRA，但需要正确安装 CUDA 版 PyTorch。

建议顺序：

1. 创建或迁移 `aqua_align` 环境；
2. 根据服务器驱动安装匹配的 CUDA PyTorch；
3. 验证 `torch.cuda.is_available()`；
4. 再安装 `requirements-server.txt`；
5. 准备 Qwen3-VL-4B-Instruct 本地权重；
6. 保持 `local_files_only: true`，或明确决定何时下载权重。

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 第五步：服务器 Base VLM smoke test

先只运行 5～10 条：

```bash
python scripts/run_base_vlm.py \
  --config configs/base_vlm.yaml \
  --metadata data/processed/uieb/metadata.csv \
  --candidates data/processed/candidates.csv \
  --split test \
  --tasks diagnose strategy compare rank \
  --limit 10 \
  --output-dir outputs/base_vlm
```

检查：

- 模型实际读取了图像；
- compare 的图1/图2顺序正确；
- rank 严格是原图/A/B 三图；
- JSON 原文与解析结果对应；
- 失败样本保留；
- 显存没有逐样本持续增长；
- 没有无依据水深、海域、相机和指标数值；
- `run_summary.json` 中真实成功数大于 0。

Smoke 通过后，固定 Prompt v0.1 和规则版本，再运行完整 test。不要看到 test 结果后继续调 Prompt。

### 第六步：扩充候选图

当前候选覆盖较少。规则和 Base VLM 基线稳定后，可以为更大的 train/val/test 子集生成候选图和质量指标，但必须保持同一原图及其派生数据不跨 split。

### 第七步：再决定是否进入 SFT

只有满足以下条件后再开始下一阶段：

- 人工审核完成；
- 规则过敏问题得到修订；
- 真实 Base VLM smoke 成功；
- test Prompt 和规则冻结；
- 已区分人工标签、规则标签、VLM 标签；
- 明确哪些样本可以进入 SFT，哪些必须剔除或复核。

## 6. 不要踩的坑

### 6.1 PowerShell 续行

PowerShell 不能使用 Linux 的反斜杠 `\` 续行。应使用反引号：

```powershell
python scripts/run_rule_diagnosis.py `
  --metadata data/processed/uieb/metadata.csv `
  --limit 10
```

如果直接从 `--output-dir` 开始执行，PowerShell 会把 `--` 当作一元运算符并报：

```text
一元运算符“--”后面缺少表达式
```

### 6.2 不要把规则输出当真值

以下字段必须保留：

```text
annotation_source=rule_based
needs_review=true
```

不能把 `rule_diagnosis.jsonl` 直接当作高质量 SFT 标签。

### 6.3 不要使用 test 集调阈值

`calibrate_rules.py` 已强制只允许 train。不要为了改善 test 结果绕过此限制，也不要根据 test 失败样本继续改 Prompt。

### 6.4 不要误解 haze_proxy

`haze_proxy` 只是统计代理，不是水下后向散射、真实水深或水体参数的物理测量。当前它在 50/50 样本触发，是最需要人工复核的规则。

### 6.5 不要把 VLM 阻塞写成推理失败率

当前真实推理数为 0，所以：

- JSON 成功率不是 0%，而是 N/A；
- 幻觉率不是 0%；
- 排序准确率不是 0%；
- 不能说模型表现差或好。

### 6.6 不要伪造缺失模型输出

JSON 解析失败时只允许最小格式修复。不得：

- 补造 confidence；
- 补造 degradation；
- 根据规则结果替换 VLM 输出；
- 人工删除失败样本；
- 将空输出计为成功。

### 6.7 多图顺序不要混

任务契约：

```text
diagnose：1 张原图
strategy：1 张原图
compare：原图 + 1 张增强图
rank：原图 + 候选 A + 候选 B
```

尤其不要把 rank 改成只有 A/B 两图，否则模型没有原图作为比较基准。

### 6.8 CUDA PyTorch 要先装

不要只执行：

```bash
pip install -r requirements-server.txt
```

然后假设 PyTorch 自动匹配服务器 CUDA。应先按照服务器驱动安装正确的 CUDA PyTorch，并验证 CUDA 可用，再安装其余依赖。

### 6.9 `local_files_only` 的行为

`configs/base_vlm.yaml` 默认：

```text
local_files_only: true
```

服务器没有权重时会明确失败，不会联网下载。这是为了可复现和避免无意下载。如果要下载模型，应作为单独、明确的服务器准备步骤处理。

### 6.10 不要盲目使用 `--overwrite`

大部分脚本默认不覆盖结果。只有确认旧结果不再需要时才加 `--overwrite`。覆盖前先检查：

- 使用的规则版本；
- Prompt 版本；
- split；
- limit；
- 模型和生成配置；
- 原输出是否需要归档。

### 6.11 当前工作区尚未提交

本阶段新增和修改内容仍在工作区，`git status` 显示大量未跟踪文件和一个已修改的 `README.md`。继续工作前不要执行：

```text
git reset --hard
git clean -fd
git checkout -- .
```

这些命令会丢失当前实现和生成报告。应先检查 diff、决定哪些生成物应纳入版本控制，再分阶段提交。

部分 `data/processed/` 和 `outputs/` 文件可能被 `.gitignore` 忽略，即使 `git status` 没显示，也不代表文件不存在。

### 6.12 不要直接删除 smoke 产物

仓库中同时保留了正式产物和 smoke 产物，例如：

```text
reports/rule_calibration_smoke.json
reports/baseline_comparison_smoke/
reports/evaluation_smoke/
data/annotations/rule_review_smoke.csv
```

它们可以在确认正式产物完整后清理，但清理前要区分：

- 正式 50/100 审核表；
- 10 图 smoke 结果；
- Base VLM blocked 环境记录。

## 7. 常用文件入口

总说明：

```text
README.md
docs/project_framework_and_usage.md
docs/underwater_enhancement_methods.md
docs/rule_diagnosis_and_zero_shot.md
```

当前阶段配置：

```text
configs/diagnosis_rules.yaml
configs/strategy_mapping.yaml
configs/base_vlm.yaml
```

当前阶段正式产物：

```text
data/processed/rule_diagnosis.jsonl
data/annotations/rule_review.csv
data/annotations/rule_review.summary.json
outputs/base_vlm/run_summary.json
outputs/base_vlm/environment.json
reports/baseline_comparison/baseline_report.md
```

服务器依赖：

```text
requirements-server.txt
```

## 8. 交接结论

传统增强、候选图、质量指标、规则特征、规则诊断、冲突感知策略、审核表、Base VLM 推理入口、解析器和评测框架已经实现并通过 58 项测试。

当前真正的阻塞不是代码单元测试，而是实验条件：

1. 规则尚未经过完整 train 分布与人工标签校准；
2. 人工审核完成数为 0；
3. 本地无 CUDA、Transformers 和 Qwen3-VL 权重；
4. 因此 Base VLM 真实推理数为 0；
5. 候选图覆盖范围仍较小。

下一位接手者应先做完整 train-only 校准和人工审核，再到 A40 服务器完成真实 Base VLM smoke。不要提前进入 SFT/DPO。
