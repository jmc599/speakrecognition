# 项目阶段日志（2026-04-20）

## 1. 本阶段目标与当前结论

本阶段主要完成了三条线：

1. `remote_rknn` 从半自动远端推理升级为 GUI 可直接调用的自动闭环。
2. 远端执行层从 `SSH/SFTP + 单次 runner` 升级为板端常驻 HTTP 服务。
3. 回头核查 `VoxCeleb2 aac12` 预训练线的历史结果，补齐 `margin=0.20` 的 `epoch5/10/15` 在 `Vox1-O-cleaned` 上的评测。

当前项目状态：

- GUI 与板端联调已基本打通。
- `remote_rknn` 已能在 GUI 中通过 HTTP 调用板端常驻服务完成注册/验证。
- 板端固定 IP 和服务部署已经跑通。
- 训练侧已明确：
  - `aac12` 是轻量预训练线，不是 full Vox2 的完整 recipe。
  - `margin=0.25` 这条线不是从 0 训练，更合理地应理解为：以 `0.20@15` 为权重初始化基座、改大 margin 后继续训练出来的分支。

---

## 2. 当前技术路线

### 2.1 训练主线

- 开放集数据组织
- Stage 1：`VoxCeleb2 aac1+aac2` 预训练
- Stage 2：`CN-Celeb` 微调
- 主干：
  - `ResNet34_SE + Attentive Statistics Pooling`
  - embedding dim = `256`
  - loss = `AAM-Softmax`
- 输入：
  - `16k mono`
  - `64-dim log-mel`

### 2.2 部署与验证主线

- `PyTorch -> ONNX -> RKNN`
- GUI 保留在 PC
- 板端负责 `remote_rknn` 推理
- 当前 v2 已升级为：
  - GUI -> HTTP -> 板端常驻服务 -> RKNN runtime

### 2.3 远端职责边界

PC 保留：

- PyQt5 GUI
- 配置保存/加载
- profile 元数据与 history 真源
- 多样本 embedding 合并
- 本地 ONNX / PyTorch / WeSpeaker 路径

Board 承担：

- 常驻推理服务
- active profile 副本
- 从音频到 `score / decision` 的完整推理

---

## 3. GUI / remote_rknn / 板端部署进展

### 3.1 remote_rknn v1

第一版做成了：

- PC 侧把输入统一成临时 `16k mono wav`
- 板端负责：
  - inference-only preprocessing
  - log-mel
  - RKNN embedding
  - cosine score
  - threshold decision

active profile 语义：

- PC 仍是唯一真源
- Board 只保存 `${remote_workdir}/active_profile.npy`
- 本地与远端同步采用 fail-closed 语义

### 3.2 remote_rknn v2

第二版把远端执行层从：

- `SSH/SFTP + 单次 runner`

升级为：

- 板端常驻 HTTP 服务

当前板端 HTTP 服务情况：

- 服务脚本：
  - `/root/models/rknn_remote_http_server.py`
- 模型：
  - `/root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm_v232.rknn`
- 健康检查：
  - `http://192.168.50.2:8765`
- 运行信息：
  - `runner_version = remote_rknn_http_v2`
  - `service_framework = stdlib`
  - `runtime_ready = true`

### 3.3 板端网络

- PC 有线 IP：
  - `192.168.50.1`
- Board 有线 IP：
  - `192.168.50.2`
- 平台：
  - `ATK-DLRK3588 / RK3588`
- 系统：
  - `Buildroot Linux`
- RKNN Runtime：
  - `2.3.2`

### 3.4 当前 smoke test 记录

GUI / 板端自动链路已完成基础 smoke test：

- 同人：
  - `score = 0.5443655848503113`
  - `decision = accept`
- 异人：
  - `score = 0.289140522480011`
  - `decision = reject`
- 当时使用阈值：
  - `0.342024`

### 3.5 小规模手机导入阈值复核

已做 `24` 对手机导入样本复核：

- `12 same + 12 diff`
- 当前阈值：
  - `0.342024`
- 结果：
  - `accuracy = 91.67%`
  - `FAR = 8.33%`
  - `FRR = 8.33%`
- 小规模复核推荐的 `far_1` 阈值：
  - `0.425192`

说明：

- 当前阈值 `0.342024` 是早期板端验证阈值
- 手机导入条件下，小规模复核显示阈值存在上移趋势

---

## 4. Vox 预训练历史线核查

### 4.1 `margin=0.20` 主线：`resnet_v7_vox2_aac12_pt`

已确认这条线至少训练到 `15 epoch`。

训练配置摘要：

- 数据：
  - `lists/train_list_vox2_aac12.txt`
- optimizer：
  - `adamw`
- batch：
  - `32`
- lr：
  - `1e-3`
- min_lr：
  - `1e-6`
- warmup：
  - `3`
- margin：
  - `0.20`
- max_frames：
  - `300`
- sampler：
  - `balanced_speaker`
- repeat cap：
  - `5`
- crop：
  - `random`
- `preprocess_for_inference = True`
- `val_ratio = 0.05`

增强情况：

- 已开：
  - 随机增益
  - Gaussian noise
  - SpecAugment
  - inference-style preprocess
- 未开：
  - speed perturb
  - active crop
  - MUSAN / RIR / reverb 这类重增强
  - 官方 Vox monitor

重要说明：

- 这条线的 `best` 是按 `val_loss` 选的
- 不是按 `Vox1-O-cleaned EER` 选的

### 4.2 `margin=0.25` 分支：`resnet_v7_vox2_aac12_pt_m025_ft`

已确认：

- `resnet_v7_vox2_aac12_pt_m025_ft_latest.pth` 是 **10 轮**
- `m025` 这条线不是 `--resume` 方式硬接在 `0.20` 后面
- 但根据更早的 Word 记录，它也**不是从 0 随机初始化**
- 更合理的历史还原是：
  - 先有 `0.20` 主线的 `resnet_v7_vox2_aac12_pt_epoch_15.pth`
  - 再用 `--init-model checkpoints/resnet_v7_vox2_aac12_pt_epoch_15.pth`
  - 把 `aam-margin` 从 `0.20` 改到 `0.25`
  - 以此起一条新的 `m025_ft` 分支，再单独训练 `10` 轮

旧证据来源：

- `C:\Users\jmc\Desktop\生成这个文件.docx`
- 其中明确出现了：
  - `aam-margin: 0.20 -> 0.25`
  - `不要用 --resume`
  - `--init-model checkpoints/resnet_v7_vox2_aac12_pt_epoch_15.pth`
  - `Initialized model from: checkpoints/resnet_v7_vox2_aac12_pt_epoch_15.pth`
  - `先从 resnet_v7_vox2_aac12_pt_epoch_15.pth 起一条 m025_ft 分支`

因此本日志后续统一按下面的语义表述：

- `m025_ft` = **以 `0.20@15` 为权重初始化基座的 margin fine-tune 分支**
- 不是：
  - `--resume` 续训
  - 也不是从 0 的 scratch 训练

配置摘要：

- optimizer：
  - `adamw`
- batch：
  - `32`
- lr：
  - `1e-4`
- min_lr：
  - `1e-6`
- warmup：
  - `0`
- margin：
  - `0.25`
- 其余增强基本与 `0.20` 主线相同

### 4.3 关于“别人 0.x%，我们 3.x%”的解释

当前 `3.3%` 不应直接理解为“模型坏了”。

更合理的解释是：

- 我们当前只用 `aac12`
- 不是 full Vox2
- recipe 较轻
- 训练时不按官方 Vox benchmark 选 best
- 没开更重的增强与 score norm

因此当前结果更像：

- **轻量预训练线的有效结果**

而不是：

- **full Vox2 + 完整 recipe 下的最终极限**

---

## 5. 本次补跑：`0.20` 线 `epoch5 / epoch10 / epoch15` 的 Vox1-O-cleaned

### 5.1 补跑条件

本次补跑使用：

- Vox1 根目录：
  - `F:\dataset\vox1_test_wav\wav`
- trial list：
  - 从 `pullback_core.tar.gz` 临时提取的 `lists/vox1_o_cleaned.txt`
- checkpoint：
  - `resnet_v7_vox2_aac12_pt_epoch_5.pth`
  - `resnet_v7_vox2_aac12_pt_epoch_10.pth`
  - `resnet_v7_vox2_aac12_pt_epoch_15.pth`
- 临时输出目录：
  - `C:\Users\jmc\AppData\Local\Temp\speakerreg_vox_eval\results`

### 5.2 补跑结果表

| 线 | checkpoint | EER | minDCF@0.01 | FRR@FAR<=1% | threshold_far_1 |
|---|---|---:|---:|---:|---:|
| `0.20` | `epoch_5` | `4.4508%` | `0.406208` | `12.9454%` | `0.394051` |
| `0.20` | `epoch_10` | `3.4405%` | `0.355478` | `9.5469%` | `0.369631` |
| `0.20` | `epoch_15` | `3.3714%` | `0.341536` | `9.1533%` | `0.360861` |

### 5.3 结果解读

- `0.20` 这条线在 `5 -> 10 -> 15 epoch` 上是**持续变好**的。
- 至少在当前 `aac12` recipe 下，`0.20` 没有在 `epoch10` 就明显完全到顶。
- `epoch15` 是 `0.20` 这条线当前已确认的最佳点。

---

## 6. 已有 `0.25` 线 benchmark 结果

### 6.1 已确认结果

`m025` 线已确认做过 `Vox1-O-cleaned` 的两个点：

| 线 | checkpoint | EER | minDCF@0.01 | FRR@FAR<=1% |
|---|---|---:|---:|---:|
| `0.25` | `epoch_5` | `3.3129%` | `0.347875` | `9.0575%` |
| `0.25` | `epoch_10` | `3.3501%` | `0.342336` | `9.2703%` |

### 6.2 结果解读

- 从 EER 看：
  - `0.25@5` 是已知最佳点
- 从训练记录看：
  - `m025_ft_best/latest` 仍是按 `loss` 选，不是按 `EER` 选
- 从历史来源看：
  - `m025_ft` 更合理地理解为 `0.20@15 -> init-model -> margin 0.25` 的分支微调
- 因此：
  - `loss-best` 和 `EER-best` 不是同一个 checkpoint

---

## 7. 当前 checkpoint 语义整理

### 7.1 `0.20` 线

- `resnet_v7_vox2_aac12_pt_epoch_15.pth`
  - 当前已确认的 `0.20` benchmark 最优点
- `resnet_v7_vox2_aac12_pt_best.pth`
  - `loss-best`
  - 不等于一定是 `EER-best`

### 7.2 `0.25` 线

- `resnet_v7_vox2_aac12_pt_m025_ft_latest.pth`
  - **第 10 轮**
  - 适合作为“继续训练的主线延续点”
  - 历史语义应理解为：
    - 基于 `resnet_v7_vox2_aac12_pt_epoch_15.pth` 权重初始化后的第 10 轮
- `resnet_v7_vox2_aac12_pt_m025_ft_best.pth`
  - `loss-best`
- `resnet_v7_vox2_aac12_pt_m025_ft_epoch_5.pth`
  - 当前已知的 `EER-best`

### 7.3 `balanced_aug` 线

- `resnet_v7_balanced_aug_s3_e10_*`
  - 更像一条 **from scratch** 的 CN 侧增强实验线
- 证据：
  - 旧脚本 [run_balanced_augment_scratch.sh](F:\speakerreg\server_train_resume_pack\run_balanced_augment_scratch.sh) 标题直接写了：
    - `from scratch`
  - 旧 checkpoint 元数据也显示：
    - `resume = ""`
    - `init_model = ""`
    - `init_from = None`

因此：

- `balanced_aug` 可以继续按“从 0 训练的增强线”表述
- `m025_ft` 则不能再写成从 0 训练

---

## 8. 现阶段推荐路线

当前建议继续使用：

- `resnet_v7_vox2_aac12_pt_m025_ft_latest.pth`

原因：

- 它是 `0.25` 线的完整 10 轮续训点
- 虽然从纯 EER 看 `epoch_5` 略好，但 `epoch_10` 作为继续训练基线更自然
- 便于把 `aac12` 线升级成一条正式的 benchmark-driven 预训练线

建议后续训练思路：

1. 继续只用 `aac12`
2. 以 `m025_ft_latest.pth` 为基线继续训练
3. 把总 epoch 拉到 `30~40`
4. 补上：
   - `Vox1-O-cleaned` monitor
   - `speed perturb`
5. best checkpoint 按：
   - `EER / minDCF`
   - 不再按 `val_loss`

---

## 9. 关键路径、文件与产物

### 9.1 重要 `.pth`

- `pullback_core.tar.gz -> checkpoints/resnet_v7_vox2_aac12_pt_epoch_15.pth`
- `pullback_core.tar.gz -> checkpoints/resnet_v7_vox2_aac12_pt_best.pth`
- `m025_ft_bundle.tar.gz -> checkpoints/resnet_v7_vox2_aac12_pt_m025_ft_epoch_5.pth`
- `m025_ft_bundle.tar.gz -> checkpoints/resnet_v7_vox2_aac12_pt_m025_ft_epoch_10.pth`
- `m025_ft_bundle.tar.gz -> checkpoints/resnet_v7_vox2_aac12_pt_m025_ft_latest.pth`
- 当前 Stage 2 主线：
  - `checkpoints/resnet_v7_vox2ft_s1_latest.pth`

### 9.2 重要评测 CSV

- 当前补跑输出：
  - `C:\Users\jmc\AppData\Local\Temp\speakerreg_vox_eval\results\resnet_v7_vox2_aac12_pt_epoch_5_vox1_o_cleaned_scores.csv`
  - `C:\Users\jmc\AppData\Local\Temp\speakerreg_vox_eval\results\resnet_v7_vox2_aac12_pt_epoch_10_vox1_o_cleaned_scores.csv`
  - `C:\Users\jmc\AppData\Local\Temp\speakerreg_vox_eval\results\resnet_v7_vox2_aac12_pt_epoch_15_vox1_o_cleaned_scores.csv`
- 历史归档中的已知结果：
  - `artifacts/results/resnet_v7_vox2_aac12_pt_epoch_15_vox1_o_cleaned_scores.csv`
  - `artifacts/results/resnet_v7_vox2_aac12_pt_m025_ft_epoch_5_vox1_o_cleaned_scores.csv`
  - `artifacts/results/resnet_v7_vox2_aac12_pt_m025_ft_epoch_10_vox1_o_cleaned_scores.csv`

### 9.3 已知图像资产

归档里已知存在：

- `today_2026-04-16.tar.gz -> artifacts/results/resnet_v7_vox2_aac12_pt_epoch15_vox1_o_cleaned_distribution.png`

说明：

- 当前 `0.20@15` 的分布图在归档里有历史版本
- 如果后续要做 5/10/15 的统一图，可以基于本次补跑的 score CSV 再生成

---

## 10. 当前对外可复述的简版结论

- 项目主体已基本完成，GUI、板端推理、RKNN 部署和 HTTP 化都已打通。
- `VoxCeleb2 aac12` 预训练线不是 full Vox2 重配方，而是一条轻量预训练线。
- 在这条轻量预训练线上：
  - `0.20` 的 `epoch15` 结果为 `EER 3.3714%`
  - `0.25` 的 `epoch5` 结果为当前已知最佳 `EER 3.3129%`
  - `0.25 epoch10/latest` 是当前更适合作为后续继续训练的主线 checkpoint
- 历史语义上：
  - `m025_ft` 不是从 0 训练
  - 更像是从 `0.20@15` 通过 `--init-model` 起的 margin fine-tune 分支
  - `balanced_aug` 则基本可以按 from-scratch 增强实验线表述
- 下一步不是推翻这条线，而是把它从“probe 线”升级成“benchmark-driven 的正式预训练线”。

---

## 11. 2026-04-23 PC-side Passphrase Verification

### Decision
- Keep speaker verification exactly as-is.
- Add passphrase verification only on the PC side.
- Do not change the board protocol, board daemon, or `remote_rknn` runner.
- Final verification result becomes:
  - `speaker_accept AND passphrase_match`

### Current implementation status
- Added GUI config fields:
  - `passphrase_verification_enabled`
  - `expected_passphrase`
  - `asr_backend`
  - `asr_model_name`
  - `asr_language`
  - `passphrase_match_mode`
- Added services:
  - `gui/services/asr_service.py`
  - `gui/services/passphrase_match.py`
- Added Setup page controls for:
  - enable/disable passphrase verification
  - expected passphrase
  - match mode
  - ASR model and language
- Added Verify page result display for:
  - expected passphrase
  - recognized text
  - passphrase match
  - speaker result
  - final result
- Integrated `main_window.py` so that:
  - verification still uses normalized WAV
  - ASR runs on the same normalized WAV on the PC side
  - history notes record passphrase details

### Model choice
- Current PC-side ASR direction:
  - `SenseVoiceSmall`
- Current runtime direction:
  - `funasr-onnx`
- Rationale:
  - better fit for Chinese short passphrases than generic long-form ASR
  - does not require any board-side deployment change for v1

### Verification
- Static compile check passed for:
  - `gui/services/config_store.py`
  - `gui/services/asr_service.py`
  - `gui/services/passphrase_match.py`
  - `gui/pages/setup_page_pcfirst.py`
  - `gui/pages/verify_page.py`
  - `gui/main_window.py`
- Offscreen window initialization succeeded:
  - `SpeakerIdentityTerminalWindow()` can be constructed without runtime import errors

### Remaining runtime dependency
- PC environment still needs:
  - `funasr`
  - `funasr-onnx`
- Added to:
  - `requirements_gui.txt`

### Scope boundary
- This is intentionally a PC-side v1.
- Future board-side ASR, ONNXRuntime-on-board, or RKNN/NPU exploration is out of scope for the current implementation step.

---

## 12. 2026-04-24 Thesis Evidence Consolidation

### Decision
- Freeze the thesis route around the stable system:
  - speaker embedding evaluation
  - ONNX/RKNN deployment consistency
  - PyQt5 GUI
  - remote RKNN board-side verification
  - optional PC-side passphrase recognition
- Do not mix recent unfinished training retries into the thesis main evidence.

### New thesis evidence package
- Added:
  - `docs/THESIS_EVIDENCE_PACKAGE_2026-04-24.md`
- Purpose:
  - collect paper-ready metrics
  - list usable images
  - separate main results from optional/appendix evidence
  - document the new passphrase recognition feature for thesis writing

### Main paper metrics to keep
- G1 old CN baseline:
  - checkpoint: `resnet_v7_open_s2_best.pth`
  - EER: `12.9703%`
  - minDCF@0.01: `0.647750`
  - FRR@FAR<=1%: `37.1839%`
  - FRR@FAR<=0.1%: `55.6125%`
- G2 final selected line:
  - checkpoint: `resnet_v7_vox2ft_s1_latest.pth`
  - EER: `10.6271%`
  - minDCF@0.01: `0.533754`
  - FRR@FAR<=1%: `28.0315%`
  - FRR@FAR<=0.1%: `44.2692%`
- G3 balanced augmentation comparison:
  - checkpoint: `resnet_v7_balanced_aug_s3_e10_best.pth`
  - EER: `14.1094%`
  - minDCF@0.01: `0.672898`
  - FRR@FAR<=1%: `41.2391%`
  - FRR@FAR<=0.1%: `58.6145%`

### Deployment evidence to keep
- PyTorch/ONNX profile verification:
  - PyTorch EER: `7.3000%`
  - ONNX EER: `7.3000%`
  - PyTorch threshold: `0.207305`
  - ONNX threshold: `0.207273`
- ONNX/RKNN score consistency:
  - num_pairs: `210`
  - mean_abs_diff: `0.0003789193451493269`
  - max_abs_diff: `0.002034764736890793`
  - corr: `0.9999958759290639`
  - RKNN toolkit/lite/runtime: `2.3.2`
- Remote RKNN threshold review:
  - source: `artifacts/results/remote_rknn_threshold_review_20260419_144135/`
  - trials: `24`
  - current threshold: `0.342024`
  - current accuracy: `91.67%`
  - FAR<=1% recommended threshold: `0.425192`

### Passphrase recognition thesis note
- Current status:
  - PC-side optional feature is implemented.
  - ASR uses `SenseVoiceSmall` through `funasr-onnx`.
  - model cache is redirected to `F:/speakerreg_artifacts/funasr_cache`.
  - final decision is `speaker_accept AND passphrase_match`.
- Thesis boundary:
  - write this as a GUI-side optional security extension.
  - do not claim board-side ASR deployment.
  - do not claim ASR RKNN/NPU acceleration.

### Image guidance
- Use in main text:
  - `artifacts/results/onnx_rknn_score_parity_20samples.png`
  - `artifacts/thesis_assets/archive_figures/vox2ft_s1_latest_cn200_distribution.png`
  - `artifacts/thesis_assets/archive_figures/vox_pretrain_epoch15_vox1_o_cleaned_distribution.png`
- Optional appendix:
  - `artifacts/results/onnx_rknn_parity_20samples.png`
  - `artifacts/thesis_assets/archive_figures/vox2ft_s1_training_curves_optional.png`
  - `artifacts/thesis_assets/archive_figures/balanced_aug_s3_e10_training_curves_optional.png`
- Regenerate before formal use:
  - `plots/resnet_v7_full_eval_refresh.png`
  - reason: current labels are generic and do not show model/dataset context.
- Do not use as main thesis figure:
  - `plots/eval_sample.png`
  - reason: sample size looks like a demo, not a formal result figure.

### Archive scan added
- Checked these archives without extracting to C drive:
  - `pullback_core.tar.gz`
  - `m025_ft_bundle.tar.gz`
  - `today_2026-04-16.tar.gz`
  - `pullback_research_full.tar.gz`
- Temporary scan directory:
  - `F:/temp/speakerreg_archive_scan/`
- Useful archived figures copied to:
  - `artifacts/thesis_assets/archive_figures/`
- Important finding:
  - `today_2026-04-16.tar.gz` contains better distribution figures than the generic `plots` folder.
  - `vox_pretrain_epoch15_vox1_o_cleaned_distribution.png` shows Vox pretraining target/non-target separation.
  - `vox2ft_s1_latest_cn200_distribution.png` shows the final CN-Celeb.E/CN200 target/non-target separation.
- Recalculated archived Vox1-O-cleaned metrics:
  - `0.20@15`: EER `3.3714%`, minDCF `0.341536`, FRR@FAR<=1% `9.1586%`
  - `0.25@5`: EER `3.3129%`, minDCF `0.347875`, FRR@FAR<=1% `9.0575%`
  - `0.25@10`: EER `3.3501%`, minDCF `0.342336`, FRR@FAR<=1% `9.2703%`
- Thesis decision:
  - use archived distribution figures where appropriate.
  - keep `m025` as historical/supporting evidence only, not the main thesis route.

### Recommended thesis table structure
- Table 1: G1/G2/G3 model comparison on CN-Celeb.E.
- Table 2: PyTorch/ONNX/RKNN deployment consistency.
- Table 3: remote RKNN threshold review.
- Optional table: passphrase function tests if manual cases are collected.
