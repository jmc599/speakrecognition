# Thesis Figure Insertion Guide - 2026-04-25

This file summarizes the thesis figures that are currently available in the repository, where to insert them in the DOCX draft, and which figures still need to be generated or captured manually.

## 1. Current DOCX

Draft checked:

- `F:\downloadf\嵌入式声纹识别系统_成熟完整论文稿_v7.docx`

The draft already contains the target anchors:

- Chapter 3.1 `系统总体架构`
- Chapter 3.3 `验证评估模块设计`
- Chapter 4.1 `数据列表构建与预处理实现`
- Chapter 4.4 / current draft section `导出与部署实现`
- Chapter 5.3 `主要实验结果`
- Chapter 5.4 `结果分析`

No figure captions starting with `图` were found in the checked draft, so the figures below should be treated as not yet inserted.

## 2. Existing Image Files

Use these relative paths from the repository root.

| Figure purpose | Recommended file | Status | Recommended placement | Suggested caption |
|---|---|---|---|---|
| Vox pretraining score distribution | `artifacts/thesis_assets/archive_figures/vox_pretrain_epoch15_vox1_o_cleaned_distribution.png` | Available | After Chapter 5.3, near the Vox pretraining paragraph | `图5-x Vox 预训练阶段 Vox1-O-cleaned 分数分布` |
| CN-Celeb.E score distribution | `artifacts/thesis_assets/archive_figures/vox2ft_s1_latest_cn200_distribution.png` | Available | After Chapter 5.3 or early Chapter 5.4 | `图5-x CN-Celeb.E 分数分布` |
| ONNX/RKNN score parity | `artifacts/thesis_assets/archive_figures/onnx_rknn_score_parity_clean.png` | Available, recommended for main text | After Chapter 5.4 deployment-consistency analysis | `图5-x ONNX 与 RKNN 分数一致性对比` |
| ONNX/RKNN score parity old version | `artifacts/results/onnx_rknn_score_parity_20samples.png` | Available, backup only | Do not use unless the clean figure is unavailable | `图5-x ONNX 与 RKNN 分数一致性对比` |
| ONNX/RKNN embedding/parity detail | `artifacts/results/onnx_rknn_parity_20samples.png` | Available, too technical for main text | Backup only | `图5-x ONNX 与 RKNN 推理一致性细节` |

Training-curve images are intentionally excluded from the main-text figure list. The archived training records are incomplete and not every epoch was logged, so these images should not be used to support a "curve keeps decreasing" statement in the thesis body.

The two distribution images requested under `artifacts/results/` were not found there:

- `artifacts/results/resnet_v7_vox2_aac12_pt_epoch15_vox1_o_cleaned_distribution.png`
- `artifacts/results/resnet_v7_vox2ft_s1_latest_cn200_distribution.png`

Use the archived copies under `artifacts/thesis_assets/archive_figures/` instead.

## 3. Existing Markdown References

These images are already referenced in:

- `artifacts/thesis_assets/README.md`
- `docs/THESIS_EVIDENCE_PACKAGE_2026-04-24.md`

This file is the concentrated insertion guide for the paper.

## 4. Figures Still To Generate

### 4.1 System Architecture Figure

Recommended placement:

- After Chapter 3.1 `系统总体架构`

Suggested caption:

- `图3-1 系统总体架构图`

Prompt for image2:

```text
请生成一张本科毕业论文用的中文工程流程图，白底，黑灰线条，横向布局，类似 draw.io/Visio，简洁清晰，不要插画、不要渐变、不要装饰图标。

图题含义：嵌入式声纹识别系统总体架构。

从左到右画 6 个模块：
音频输入 → Log-Mel 特征 → ResNet34_SE + ASP → speaker embedding → cosine / threshold → 本地或板端验证。

每个模块只保留短标签：
音频输入：录音/导入音频
Log-Mel 特征：16 kHz、64维
ResNet34_SE + ASP：声学编码与池化
speaker embedding：256维声纹向量
cosine / threshold：余弦打分与阈值判决
本地或板端验证：PC / RKNN

底部加一句说明：
训练阶段使用 AAM-Softmax，部署阶段通过 ONNX / RKNN 完成板端验证。

整体要求：16:9 PNG，文字清楚，适合插入 Word 论文正文。
```

### 4.2 Open-Set Split And Evaluation Protocol Figure

Recommended placement:

- Best: after Chapter 4.1 `数据列表构建与预处理实现`
- Alternative: after Chapter 3.3 `验证评估模块设计`

Suggested caption:

- `图4-1 开放集数据划分与评测协议`

Prompt for image2:

```text
请生成一张本科毕业论文用的中文数据划分流程图，白底，黑灰线条，横向布局，类似 draw.io/Visio，简洁清晰，不要插画、不要渐变、不要装饰图标。

图题含义：开放集数据划分与评测协议。

从左到右画一个主数据源和 4 个用途模块：
CN-Celeb 开放集数据 → train_main / dev-monitor / dev-cal，右侧单独放 CN-Celeb.E。

每个模块只保留短标签：
train_main：模型训练
dev-monitor：训练监控
dev-cal：阈值校准
CN-Celeb.E：最终评测

关系要求：
- train_main、dev-monitor、dev-cal 来自开放集训练数据。
- CN-Celeb.E 与前三者用虚线隔开，标注“只用于最终评测”。
- 在数据源旁加一句“排除评测身份，避免测试集泄漏”。
- 不要在图里放过多数字，具体样本数留在正文表格说明。

整体要求：16:9 PNG，文字清楚，适合插入 Word 论文正文。
```

### 4.3 remote_rknn Board Verification Chain

Recommended placement:

- After Chapter 4.4 / current draft section `导出与部署实现`
- Alternative: after Chapter 5.4 if it is treated as validation evidence

Suggested caption:

- `图4-x remote_rknn 板端验证链路`

Prompt for image2:

```text
请生成一张本科毕业论文用的中文工程流程图，白底，黑灰线条，横向布局，类似 draw.io/Visio，简洁清晰，不要插画、不要渐变、不要装饰图标。

图题含义：remote_rknn 板端验证链路。

从左到右画 5 个模块：
PC GUI → 音频规范化与 profile 管理 → 网络传输 → ATK-DLRK3588 板端推理 → JSON 验证结果。

每个模块只保留短标签：
PC GUI：录音/导入音频、发起注册/验证
音频规范化与 profile 管理：16 kHz WAV、active profile
网络传输：HTTP 或 SSH/SFTP
板端推理：Log-Mel、RKNN、cosine score、threshold
JSON 验证结果：score、threshold、decision

底部加一句说明：
PC 保存 profile 元数据和历史记录，开发板保存 active_profile.npy 并完成推理判决。

整体要求：16:9 PNG，文字清楚，适合插入 Word 论文正文。
```

### 4.4 Optional Passphrase Recognition Flow

Current guide status:

- The existing system architecture and `remote_rknn` figures mainly describe the voiceprint verification chain.
- They do not fully explain the new text-recognition/passphrase branch.
- Add one separate figure if the thesis includes the new feature in Chapter 4.

Recommended placement:

- Best: after the GUI verification implementation section in Chapter 4.
- Alternative: after the remote RKNN deployment section, as an optional security-extension subsection.

Suggested caption:

- `图x-x 声纹与口令双因子验证流程`

Prompt for image2:

```text
请生成一张本科毕业论文用的中文工程流程图，白底，黑灰线条，横向布局，类似 draw.io/Visio，简洁清晰，不要插画、不要渐变、不要装饰图标。

图题含义：声纹与口令双因子验证流程。

整体从左到右画 6 个模块：
录音/导入音频 → 音频规范化 → 声纹验证 → 文字识别 → 口令匹配 → 最终判决。

每个模块只保留短标签：
录音/导入音频：用户语音输入
音频规范化：16 kHz WAV
声纹验证：speaker score、threshold、accept/reject
文字识别：SenseVoiceSmall、recognized text
口令匹配：expected passphrase、text match
最终判决：声纹通过 AND 口令匹配

在声纹验证模块下方标注：
本地 PyTorch/ONNX 或 remote_rknn

在文字识别模块下方标注：
PC 侧 ASR，不部署到开发板

底部加一句说明：
启用口令开关后，系统只有在声纹验证通过且识别文本与预设口令匹配时才接受。

整体要求：16:9 PNG，文字清楚，适合插入 Word 论文正文。
```

Implementation points for thesis text:

- ASR backend: `SenseVoiceSmall` through `funasr-onnx`.
- Cache path: `F:/speakerreg_artifacts/funasr_cache`.
- Main files:
  - `gui/services/asr_service.py`
  - `gui/services/passphrase_match.py`
  - `gui/services/config_store.py`
  - `gui/pages/setup_page_pcfirst.py`
  - `gui/pages/verify_page.py`
  - `gui/main_window.py`
- Final decision rule:
  - `speaker_accept AND passphrase_match`
- Scope boundary:
  - this is a PC-side optional extension.
  - do not write it as board-side ASR or RKNN ASR deployment.

## 5. Screenshots Still Missing

No GUI screenshots or physical board photos were found in the repository scan.

Recommended screenshots to capture manually:

| Screenshot | Suggested placement | Notes |
|---|---|---|
| GUI settings page with `远端 RKNN` selected | Chapter 4.4 | Shows backend and board configuration |
| GUI enrollment/register page after a profile is registered | Chapter 4.4 or appendix | Shows registration workflow |
| GUI verification page with accept result | Chapter 5.4 | Shows functional verification |
| GUI verification page with reject result | Chapter 5.4 | Shows thresholded rejection |
| GUI setup page with passphrase verification enabled | Chapter 4 optional extension section | Shows expected passphrase, ASR backend, language, and match mode |
| GUI verification page with recognized text and passphrase match | Chapter 4 or Chapter 5.4 | Shows speaker result, recognized text, text match, and final result |
| GUI verification page with speaker pass but wrong passphrase | Chapter 5.4 optional functional test | Shows why final decision uses `speaker_accept AND passphrase_match` |
| Board terminal or JSON result screenshot | Chapter 4.4 or 5.4 | Use structured JSON result |
| Physical board connection photo | Appendix or Chapter 4.4 | Only if board, power, network cable, and label are visible |

Suggested save directory after capture:

- `artifacts/thesis_assets/screenshots/`

Existing JSON evidence that can be opened or screenshot:

- Accept result: `artifacts/results/remote_rknn_threshold_review_20260419_144135/raw_results/same_cjm_01.json`
- Reject result: `artifacts/results/remote_rknn_threshold_review_20260419_144135/raw_results/diff_01.json`

Commands to display them in PowerShell:

```powershell
Get-Content artifacts\results\remote_rknn_threshold_review_20260419_144135\raw_results\same_cjm_01.json
Get-Content artifacts\results\remote_rknn_threshold_review_20260419_144135\raw_results\diff_01.json
```

## 6. How To Open Existing Images

Use PowerShell from the repository root:

```powershell
ii artifacts\thesis_assets\archive_figures\vox_pretrain_epoch15_vox1_o_cleaned_distribution.png
ii artifacts\thesis_assets\archive_figures\vox2ft_s1_latest_cn200_distribution.png
ii artifacts\thesis_assets\archive_figures\onnx_rknn_score_parity_clean.png
ii artifacts\results\onnx_rknn_score_parity_20samples.png
ii artifacts\results\onnx_rknn_parity_20samples.png
```

If a Markdown-rendered absolute `F:` link is blocked by the IDE, open the relative paths above from the file explorer or terminal.
