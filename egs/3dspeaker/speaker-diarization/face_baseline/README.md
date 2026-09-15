# AHC + Face-only HMM

每个 AHC 簇对应独立的二状态人脸出现序列。保留原有 AHC 阈值，不使用 audio embedding、ASD 或微调；不需要 GPU。

## 分别提交

在服务器的 `3D-Speaker/egs/3dspeaker/speaker-diarization` 目录执行：

```bash
sbatch face_baseline/run_BB.sh
sbatch face_baseline/run_IL.sh
```

默认读取 `runs/<剧名>/exp_video_ablation/embs_video/*_midframe.pkl`、同一实验的 `json/subseg_ori.json`，以及 `/data/home/scv7387/run/tv_series_plus/dataset/<剧名>/raw/wav.list` 和人脸标注。`wav.list` 仅用于确定集名，不读取音频。片段 JSON 必须与人脸缓存对应，空人脸帧仍保留在时间轴中。

若已有 AHC 结果，指定其路径（必须覆盖当前缓存中的全部人脸 key）：

```bash
AHC_LABELS="/absolute/path/to/existing_ahc.json" sbatch face_baseline/run_BB.sh
```

也可设置 `DATA_ROOT`、`SOURCE_EXP`、`VISUAL_EMBS_DIR`、`SUBSEG_JSON`、`RESULT_DIR` 和 `EVAL_MODE`。例如：

```bash
DATA_ROOT="/data02/home/scv7387/run/tv_series_plus/dataset" \
SOURCE_EXP="/absolute/path/to/existing/exp_video" \
AHC_LABELS="/absolute/path/to/existing_ahc.json" \
sbatch face_baseline/run_IL.sh
```

`EVAL_MODE` 默认 `all`，与原脚本一致；如原论文使用 `test`，应同样设置 `EVAL_MODE=test`（要求标注目录已有 `valid_part_keys_face.npy`）。从其他目录提交时另设 `RECIPE_ROOT`。

未指定 `AHC_LABELS` 时，优先复用输出目录的 `faces_mid_frame_ahc.json`，否则按该剧 `diar_video.yaml` 的视觉 AHC 配置运行一次。更换输入 embedding 或配置时请使用新的 `RESULT_DIR`。

## 输出

默认目录：`runs/<剧名>/exp_video_ablation/result/face_hmm/`。

- `faces_mid_frame_ahc.json`：初始 AHC 标签。
- `faces_mid_frame_face_hmm.json`：经现有 `correct_face_labels()` 还原的人脸标签。
- 对应的 `*_accuracy.txt`：`overall_accuracy` 和 `group_0_accuracy` 至 `group_4_accuracy`，分别对应论文 Group 1–5。空组由现有评估脚本省略；`test/valid` 模式文件名含相应后缀。
- `face_hmm_params.npz`：`alpha`、`A_F`、`B_F` 及 channel 对应的 `cluster_ids`。
- `run_info.json`：输入、参数、样本数和实际修改人脸数。

HMM 的时间步为字幕中间帧；每集单独起始，参数在同一剧的所有集上估计。保留全部 AHC 簇，另预留 Others channel。参数从对角占优的转移/混淆矩阵初始化，使用固定种子和微小伪计数；恒定 channel 保留原状态。解码后按视觉相似度提供最近两个簇及原标签作为候选，无法唯一匹配时保留原标签。不读取人工标注拟合 HMM，标注仅在最后评估使用。

这是独立视觉 HMM 加既有人脸标签还原流程的 baseline。完整 HADL 的跨模态初始化、联合候选约束及修正规则可能不同；直接比较历史 Full HADL 结果时，应核对输入和评估范围。

依赖现有 3D-Speaker 环境及 `hmmlearn`（本地验证使用 0.3.3）。若缺少该依赖，在目标环境安装 `python -m pip install hmmlearn==0.3.3`。本入口不会自动安装包。
