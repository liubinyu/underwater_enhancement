# AquaAlign-VLM 水下图像增强算法实现与后续方向

## 1. 文档范围与当前状态

当前仓库同时保留两条水下图像增强路线：

1. **传统图像处理候选生成路线**：不需要训练，用于从每张水下原图生成具有不同颜色、亮度和对比度特征的候选图，为后续质量评估、人工审核、VLM SFT 和偏好对齐准备数据。
2. **轻量可训练增强网络路线**：包括纯学习残差增强网络和物理先验引导网络，可用于独立的增强模型训练与推理。目前它们不参与 `AquaAlign-VLM` 的传统候选生成脚本。

目前经过真实 UIEB 流程验证的是第一条路线：

- UIEB 原图与参考图：890 对；
- `challenging-60`：60 张无参考困难图；
- 完整 metadata：950 条；
- 当前真实候选 smoke test：前 10 张有参考图，每张 9 个候选，共 90 个候选；
- PSNR、SSIM 和无参考辅助指标均已计算；
- 这 10 张只用于工程与视觉初步验证，**不能视为完整 UIEB benchmark 结论**。

主要代码位置：

| 功能 | 文件 |
|---|---|
| 传统增强算法 | `aqua_align/enhancement.py` |
| 批量候选生成 | `aqua_align/candidates.py` |
| 质量指标 | `aqua_align/image_quality.py` |
| UIEB 数据整理 | `aqua_align/dataset.py` |
| 算法参数 | `configs/data.yaml` |
| 纯学习增强网络 | `models/baseline_net.py` |
| 物理引导增强网络 | `models/physics_guided_net.py` |
| 组合损失 | `losses/total_loss.py` |

## 2. 当前传统增强数据流

```text
UIEB raw/reference
        │
        ▼
prepare_uieb.py
        │  metadata.csv；原图级 train/val/test 划分
        ▼
generate_candidates.py
        │  颜色校正、亮度调整、对比度增强、Retinex
        ▼
candidates.csv + 候选图
        │
        ├── compute_quality_metrics.py → quality_metrics.csv
        │
        └── create_comparison_figures.py → reports/figures/
```

当前默认候选名称为：

```text
gray_world
white_patch
clahe
gamma_0.6
gamma_0.8
gamma_1.2
color_balance
retinex
wb_clahe
```

所有方法都遵循统一接口：

- 输入：RGB `numpy.ndarray`；
- 支持灰度、单通道、RGB 和 RGBA；
- 整数输入按 `[0, 255]` 解释；
- 浮点输入允许 `[0, 1]` 或 `[0, 255]`；
- 输出：与输入空间尺寸一致的 RGB `uint8`；
- 中间计算使用 `float32` 或 `float64`，避免 `uint8` 乘除溢出；
- 拒绝空图、非法通道数、NaN、Infinity 和越界数值；
- 不原地修改输入图像。

## 3. 传统增强算法

### 3.1 Gray World 灰世界白平衡

#### 基本假设

灰世界假设认为，在具有足够多颜色的自然场景中，RGB 三个通道的全局平均值应接近同一灰度。水下图像经常表现为红通道衰减、蓝绿通道占优，因此可以通过通道增益使三通道均值趋于一致。

设三个通道均值为：

$$
\mu_R,\ \mu_G,\ \mu_B
$$

目标均值为：

$$
\mu = \frac{\mu_R + \mu_G + \mu_B}{3}
$$

每个通道的增益为：

$$
g_c = \frac{\mu}{\mu_c + \epsilon}
$$

输出为：

$$
I'_c(x) = \operatorname{clip}(g_c I_c(x), 0, 255)
$$

#### 项目实现

- 统计整张图的 RGB 均值；
- 对近零通道避免直接除零；
- 使用 `max_gain=4.0` 限制增益范围为 `[1/4, 4]`；
- 所有乘法在 `float32` 中完成；
- 最终裁剪并转换为 `uint8`。

#### 优点

- 速度快；
- 能明显减轻蓝绿色偏色；
- 不依赖参考图和训练数据；
- 很适合作为候选池中的“强颜色校正”基线。

#### 局限

- 场景本身不满足灰世界假设时会产生错误校色；
- 红通道信息已经物理性丢失时，放大红通道不能恢复真实细节；
- 可能放大红通道噪声；
- 全局增益无法处理空间不均匀照明。

### 3.2 White Patch 白点校正

#### 基本思想

经典白点法假设图像中最亮区域接近白色。直接使用最大值容易受噪点或镜面高光影响，因此项目使用通道高百分位数：

$$
w_c = P_p(I_c)
$$

其中 $P_p$ 表示第 $p$ 百分位，默认 `p=99.5`。通道增益为：

$$
g_c = \frac{255}{w_c + \epsilon}
$$

#### 项目实现

- 默认 `percentile=99.5`；
- 默认 `max_gain=4.0`；
- 各通道独立缩放；
- 对全黑或近零通道进行保护。

#### 特点

- 对含有可靠亮区域的图像通常有效；
- 比直接取通道最大值更稳健；
- 在没有真实白色物体、存在灯光高光或严重雾化时容易误判；
- 当前真实样例中部分结果偏亮或偏青，应保留过曝惩罚，不能只根据亮度上升判断质量更好。

### 3.3 CLAHE 局部对比度增强

CLAHE 是 Contrast Limited Adaptive Histogram Equalization，即限幅自适应直方图均衡。

#### OpenCV 路径

安装 OpenCV 时，项目执行：

1. RGB 转 CIE LAB；
2. 只对亮度通道 $L$ 应用 CLAHE；
3. 保持 $a/b$ 色度通道不变；
4. LAB 转回 RGB。

默认参数：

```yaml
clip_limit: 2.0
tile_grid_size: 8
```

#### SciPy 后备路径

当 OpenCV 不可用时：

1. 按 RGB 加权得到亮度；
2. 将亮度划分为局部 tile；
3. 对每个 tile 的直方图限幅；
4. 重新分配被裁剪的直方图计数；
5. 生成累计分布 LUT；
6. 对相邻 tile 的 LUT 结果进行双线性混合；
7. 按新旧亮度比缩放 RGB。

#### 优点与局限

- 能提升局部纹理、目标边缘和远景可见度；
- 限幅可以抑制普通 AHE 的极端噪声放大；
- 它主要解决对比度，不能独立校正蓝绿色偏色；
- `clip_limit` 过大可能放大噪声、颗粒和水体后向散射；
- tile 太小可能出现局部不自然或分块感。

### 3.4 Gamma Correction 幂律亮度调整

项目实现的 Gamma 变换为：

$$
I'(x) = 255\left(\frac{I(x)}{255}\right)^\gamma
$$

- $\gamma < 1$：提亮中暗区域；
- $\gamma = 1$：保持不变；
- $\gamma > 1$：压暗图像。

当前生成三个候选：

```yaml
values: [0.6, 0.8, 1.2]
```

Gamma 的优势是可控、快速且不会直接改变通道比例，但它不能恢复丢失的红光，也不能消除雾化。当前项目把多个 Gamma 值作为不同候选，而不是预先认定某个值始终最优。

### 3.5 Simplest Color Balance 百分位颜色拉伸

该方法对每个通道分别裁剪两端少量像素，再将剩余范围线性映射到 `[0, 255]`。

默认总裁剪比例为 `percent=1.0`，即每端约裁剪 `0.5%`：

$$
l_c = P_{0.5}(I_c),\qquad h_c = P_{99.5}(I_c)
$$

$$
I'_c(x)=255\frac{I_c(x)-l_c}{h_c-l_c}
$$

如果 $h_c-l_c$ 过小，则保留原通道，避免除零。

#### 特点

- 同时扩展动态范围并改变通道平衡；
- 对 UIEB 当前 10 张小样本的 PSNR/SSIM 均值最高；
- 可能剪掉高光和暗部细节；
- 分通道拉伸可能导致非自然颜色；
- 小样本排名不能直接外推到完整 UIEB。

### 3.6 Multi-Scale Retinex 多尺度 Retinex

Retinex 将图像近似分解为反射分量和照明分量：

$$
I(x)=R(x)L(x)
$$

在对数域中：

$$
\log R(x)=\log I(x)-\log L(x)
$$

项目使用不同高斯尺度估计照明：

$$
R_c(x)=\frac{1}{N}\sum_{i=1}^{N}\left[\log(I_c(x)+1)-\log(G_{\sigma_i}*I_c(x)+1)\right]
$$

默认尺度：

```yaml
sigmas: [15.0, 80.0, 250.0]
```

随后对每个通道的响应使用 `1%` 和 `99%` 百分位进行稳健归一化。

#### 优点

- 能同时改善不均匀照明和局部对比度；
- 多尺度组合兼顾局部细节和全局光照；
- 适合作为候选池中的强增强结果。

#### 当前问题

- 各通道独立归一化容易造成明显红色或紫色过补偿；
- 可能产生光晕、噪声放大和过锐化；
- 大尺度高斯卷积是当前传统候选中主要耗时项；
- 当前 UIEB 视觉抽查中 Retinex 存在明显偏红，PSNR/SSIM均值也较低，因此不能直接作为默认输出。

### 3.7 White Balance + CLAHE 组合

组合流程为：

```text
原图 → Gray World 白平衡 → LAB 亮度 CLAHE → RGB 输出
```

该方法先解决偏色，再提升局部对比度。相较单独 Gray World，它通常能显露更多纹理；相较单独 CLAHE，它能减少蓝绿色主色调。

局限是两个步骤的误差会叠加：白平衡过强后再增强局部对比度，可能同时放大色噪声、颗粒和光晕。

## 4. 当前 YAML 参数

配置位于 `configs/data.yaml`：

```yaml
candidate_generation:
  jpeg_quality: 95
  methods:
    gray_world:
      max_gain: 4.0
    white_patch:
      percentile: 99.5
      max_gain: 4.0
    clahe:
      clip_limit: 2.0
      tile_grid_size: 8
    gamma:
      values: [0.6, 0.8, 1.2]
    simplest_color_balance:
      percent: 1.0
    retinex:
      sigmas: [15.0, 80.0, 250.0]
      low_percentile: 1.0
      high_percentile: 99.0
    white_balance_clahe:
      max_gain: 4.0
      clip_limit: 2.0
      tile_grid_size: 8
```

批量生成支持：

- `--limit`：小样本实验；
- `--workers`：多线程生成；
- `--seed`：固定随机种子；
- `--dry-run`：只检查计划，不写目录或 CSV；
- `--overwrite`：显式覆盖，默认不覆盖；
- 参数以合法 JSON 字符串写入 `candidates.csv`；
- 原图、参考图和全部派生候选继承同一个原图级 split，避免数据泄漏。

## 5. 质量指标实现

### 5.1 有参考指标

#### PSNR

$$
\operatorname{MSE}=\frac{1}{3HW}\sum_x\|I(x)-R(x)\|_2^2
$$

$$
\operatorname{PSNR}=10\log_{10}\frac{255^2}{\operatorname{MSE}}
$$

完全相同时数学结果为正无穷。为保证 CSV 不出现 Infinity，项目使用有限上限 `100 dB`，该约定在配置中显式记录。

#### SSIM

项目实现 RGB 三通道窗口化 SSIM，最大窗口为 `11×11`，根据小图尺寸自动选择奇数窗口。小于 `3×3` 时明确抛出异常。最终对所有空间位置和通道求均值。

PSNR/SSIM的限制：

- 高分不一定代表人眼认为颜色更自然；
- 配准误差会明显降低分数；
- UIEB参考图本身也是人工选择或处理结果，不是唯一真实解；
- 不能用单一 PSNR 或 SSIM 自动生成绝对偏好。

### 5.2 无参考辅助指标

当前实现：

| 指标 | 含义 | 范围/解释 |
|---|---|---|
| `mean_brightness` | 灰度均值 | `[0,255]`；过低或过高都可能不好 |
| `grayscale_std` | 灰度标准差 | 辅助描述全局对比度 |
| `laplacian_variance` | Laplacian响应方差 | 辅助描述清晰度，也可能被噪声抬高 |
| `dark_pixel_ratio` | 灰度不高于15的像素比例 | `[0,1]` |
| `saturated_pixel_ratio` | 任一通道不低于250的像素比例 | `[0,1]` |
| `red/green/blue_mean` | 各通道均值 | `[0,255]` |
| `channel_imbalance` | 三通道均值最大值减最小值 | 越小不一定绝对越自然 |
| `colorfulness` | 红绿、黄蓝对手色统计量 | 颜色更强不等于质量更高 |

这些指标只用于诊断和辅助排序。尤其是清晰度、颜色丰富度和通道均衡不能单独作为偏好标签。

## 6. 10张真实 UIEB 小样本结果

以下数据来自当前 `quality_metrics.csv` 的10张有参考图样本，每种方法10个结果：

| 方法 | 平均PSNR | 平均SSIM | 平均亮度 | 平均通道不平衡 | 饱和像素比例 |
|---|---:|---:|---:|---:|---:|
| color_balance | 25.549 | 0.9052 | 117.65 | 71.19 | 0.0347 |
| white_patch | 21.781 | 0.8700 | 131.28 | 84.21 | 0.0386 |
| gamma_0.8 | 17.400 | 0.8220 | 120.68 | 98.41 | 0.0150 |
| clahe | 16.627 | 0.8209 | 112.12 | 94.49 | 0.0232 |
| gamma_0.6 | 16.509 | 0.7948 | 140.51 | 95.36 | 0.0194 |
| wb_clahe | 16.355 | 0.8222 | 109.55 | 11.29 | 0.0505 |
| gray_world | 15.691 | 0.8126 | 99.64 | 11.40 | 0.0369 |
| gamma_1.2 | 14.854 | 0.7672 | 92.17 | 95.43 | 0.0119 |
| retinex | 12.946 | 0.7165 | 165.46 | 29.42 | 0.0258 |

解读注意事项：

- `color_balance` 在这10张中的参考指标最好，但不能据此宣布它是完整 UIEB 最优方法；
- Gray World 和 WB+CLAHE 的通道不平衡最低，但颜色均衡不等于参考图一致性最高；
- Retinex 平均亮度最高且视觉上存在偏红，说明“更亮、更鲜艳”不能作为更优的充分条件；
- 应在完整890对数据上计算分布、置信区间和分场景结果，而不只看均值。

## 7. 可训练增强网络

### 7.1 BaselineEnhancementNet

纯学习 baseline 使用轻量编码器—解码器预测受限残差：

$$
\Delta = s\tanh(f_\theta(I))
$$

$$
J=\operatorname{clip}(I+\Delta,0,1)
$$

默认 `residual_scale=0.2`，限制网络对原图的最大修改幅度。

编码器—解码器包含：

- Depthwise separable convolution；
- 两级下采样；
- Residual depthwise-separable block；
- Bottleneck channel attention；
- 双线性上采样；
- 编码器跳跃连接。

该网络参数量较小，适合作为学习型对照组，但没有显式水下成像先验。

### 7.2 PhysicsGuidedEnhancementNet

物理引导模型采用简化水下/雾化成像模型：

$$
I(x)=J(x)t(x)+A(1-t(x))
$$

其中：

- $I$：观测水下图像；
- $J$：待恢复图像；
- $t(x)$：空间传输图；
- $A$：全局背景光。

网络包含三部分：

1. `BackgroundLightEstimator`：输出每张图的全局 RGB 背景光 $A$；
2. `TransmissionEstimator`：输出单通道传输图 $t(x)$，通过 `min_t=0.08` 避免除数过小；
3. `LightEncoderDecoder`：输入原图、粗恢复图和传输图，共 `3+3+1=7` 通道，预测细化残差。

粗恢复公式为：

$$
J_{coarse}(x)=\operatorname{clip}\left(\frac{I(x)-A}{\max(t(x),\epsilon)}+A,0,1\right)
$$

最终结果为：

$$
J_{final}=\operatorname{clip}(J_{coarse}+s\tanh(f_\theta(I,J_{coarse},t)),0,1)
$$

默认细化残差尺度 `s=0.15`。

### 7.3 当前训练损失

组合损失为：

$$
\mathcal L =
w_c\mathcal L_{color}+
w_e\mathcal L_{exposure}+
w_g\mathcal L_{gradient}+
w_{edge}\mathcal L_{edge}+
w_{tv}\mathcal L_{TV}+
w_p\mathcal L_{physics}+
\mathcal L_{paired}
$$

其中：

- `ColorConstancyLoss`：约束RGB通道均值接近；
- `ExposureLoss`：将局部平均亮度推向默认目标 `0.55`；
- `GradientConsistencyLoss`：保持增强前后的水平/垂直梯度；
- `EdgeDetailLoss`：保持Sobel边缘；
- `TotalVariationLoss`：抑制空间振荡和噪声；
- `PhysicsConsistencyLoss`：约束传输图平滑、范围合理，并弱约束背景光通道；
- 有配对目标时可启用 L1、Charbonnier 和 SSIM 损失。

仓库中存在 physics-guided checkpoint 和推理产物，但本阶段没有在完整 UIEB 测试集上重新训练和评测，因此不能将其与上述传统方法做正式数值比较。

## 8. 当前实现的关键限制

### 8.1 JPEG重编码影响指标

预处理图、参考图和候选图当前默认保存为 `JPEG quality=95`。因此 PSNR/SSIM 同时包含算法误差和JPEG重编码误差。正式 benchmark 应优先保存无损 PNG，或者直接用内存结果对原始参考图计算指标。

### 8.2 全局颜色假设过强

Gray World、White Patch 和 Color Balance 都是全局方法，无法处理：

- 局部人工光源；
- 空间变化的水体颜色；
- 前景与远景不同程度的衰减；
- 红通道完全丢失区域。

### 8.3 简化物理模型

当前神经物理模型使用单通道传输图和全局RGB背景光。真实水下环境具有波长相关衰减，更合理的形式应允许 $t_R,t_G,t_B$ 不同，并允许背景光随空间缓慢变化。

### 8.4 指标覆盖不足

PSNR/SSIM更偏重像素一致性；现有无参考指标是辅助统计，尚未形成经过验证的人类偏好模型。UIQM/UCIQE也尚未作为正式可靠指标纳入当前流水线。

### 8.5 参数仍是固定配置

所有图共享同一组增强参数。严重蓝偏、低照度、雾化和近自然图像实际上不应使用同一强度。

## 9. 后续方向

### P0：先完成可信基准

1. **对完整890对UIEB运行全部候选**，而不是只统计10张。
2. **改为PNG无损候选和参考图**，消除JPEG对PSNR/SSIM的干扰。
3. 使用 train/val 调参，只在 test split 报告最终指标，避免测试集调参。
4. 输出每种方法的均值、中位数、标准差、置信区间和失败案例，而非只有均值。
5. 按退化类型分组：蓝绿偏色、低照度、低对比度、雾化、模糊、噪声。

### P1：改进传统候选池

1. 增加水下红通道补偿，但限制最大补偿和噪声放大。
2. 将当前 Retinex 升级为带颜色恢复约束的 MSRCR，增加红/紫偏色惩罚。
3. 增加基于暗通道或水下暗通道的去雾候选，同时显式检查光晕。
4. 增加多尺度融合方法：融合白平衡、Gamma和CLAHE结果，而不是只串联操作。
5. 对 Gamma、CLAHE、白平衡强度做参数网格，形成更丰富但受控的候选池。
6. 增加降噪后增强和增强后轻度降噪两类候选，避免直接放大水体颗粒。

### P2：自适应策略选择

先根据图像统计估计退化，再选择候选参数：

```text
蓝绿偏色严重 → 受限红补偿 + 白平衡
低照度严重   → Gamma<1 + 温和局部对比度
雾化严重     → 物理去雾/Retinex + 光晕检查
噪声严重     → 先降噪，再做低强度增强
接近自然     → 弱增强或保持原图
```

第一版可用规则系统，后续再让 VLM 输出结构化策略。

### P3：改进学习型增强网络

1. 使用 UIEB 890 对进行严格配对训练和独立测试。
2. 将固定曝光目标 `0.55` 改为内容自适应曝光约束。
3. 使用通道相关传输图 $t_R,t_G,t_B$。
4. 将全局背景光扩展为空间低频背景光图。
5. 添加参考监督的感知损失和颜色空间损失，但避免只追求鲜艳度。
6. 加入 EMA、学习率调度、早停、混合精度和完整实验追踪。
7. 比较传统、baseline、physics-guided以及后续更强模型的参数量、速度和质量。

### P4：构建可靠偏好数据

1. 候选来源应同时包括传统方法、不同参数版本和训练模型输出。
2. 有参考样本综合 PSNR、SSIM、颜色自然度、饱和惩罚和伪影检测。
3. 无参考样本只生成 `metric_assisted` 待审核偏好，不能伪装为人工标签。
4. 使用 Pareto筛选，避免单一指标最大值决定 chosen。
5. 人工评分覆盖颜色自然度、可见性、细节保持、噪声、光晕和过增强。
6. 低置信度样本进入 review CSV，不直接进入 DPO。

### P5：评测与工程优化

1. 增加 LPIPS/DISTS 等感知距离，并明确它们同样不是绝对人类偏好。
2. 在确认公式和实现后，以 `experimental` 形式加入 UIQM/UCIQE。
3. 增加推理耗时、内存峰值、候选文件大小统计。
4. 对 Retinex 做尺度复用、下采样照明估计或GPU加速。
5. 为每次候选生成保存配置快照、代码commit和输入文件hash。
6. 建立典型成功/失败案例库，支持后续 VLM 解释训练。

## 10. 推荐执行顺序

```text
1. 输出无损PNG候选
2. 完整运行890对UIEB
3. 统计各方法分布与失败案例
4. 人工审核并调整传统算法参数
5. 加入红通道补偿、MSRCR和融合候选
6. 训练并评测paired physics-guided模型
7. 构建SFT结构化诊断数据
8. 构建高置信度chosen/rejected偏好数据
9. 最后进行VLM SFT与DPO
```

在第1至第6步没有形成可信增强候选和评测基线前，不建议直接把自动指标排名当成大模型偏好真值。
