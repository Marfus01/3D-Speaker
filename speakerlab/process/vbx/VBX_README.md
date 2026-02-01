# VBx Integration for Speaker Diarization

本模块集成了VBx (Variational Bayes HMM over x-vectors) 算法，用于增强说话人聚类结果。

## 功能概述

VBx是一种基于变分贝叶斯的隐马尔可夫模型，用于平滑说话人标签序列。它通过以下步骤工作：

1. **LDA变换训练**: 在初始聚类结果上训练LDA变换，降维到指定维度
2. **PLDA模型训练**: 训练Two-Covariance PLDA模型
3. **VBx推理**: 使用训练好的模型进行HMM推理，平滑标签序列

## 文件说明

- `vbx_utils.py`: VBx核心工具函数，包括LDA训练、PLDA变换、VBx推理等
- `vbx_plda.py`: Two-Covariance PLDA模型实现
- `vbx_enhancer.py`: VBx封装类，提供简洁的接口

## 使用方法

### 1. 在cluster_and_postprocess.py中使用

修改运行脚本，将`cluster_enhance_mode`设置为`vbx`:

```bash
python cluster_and_postprocess.py \
    --conf config.yaml \
    --wavs wav_list.txt \
    --cluster_type audio_only \
    --audio_embs_dir /path/to/embeddings \
    --result_dir /path/to/results \
    --cluster_enhance_mode vbx
```

### 2. 直接使用VBxEnhancer类

```python
from speakerlab.process.vbx_enhancer import VBxEnhancer
import numpy as np

# 准备数据
embeddings = np.random.randn(100, 256)  # (N, D) embeddings
init_labels = np.array([...])  # 初始聚类标签

# 创建VBx增强器
vbx = VBxEnhancer(
    lda_dim=128,        # LDA降维目标维度
    Fa=1.0,             # VBx参数：充分统计量缩放
    Fb=1.0,             # VBx参数：说话人正则化
    loopP=0.9,          # VBx参数：不切换说话人的概率
    num_em_iters=5,     # PLDA训练EM迭代次数
    init_smoothing=5.0, # 初始化平滑参数
    max_iters=10        # VBx最大迭代次数
)

# 训练并预测
smoothed_labels = vbx.fit_predict(embeddings, init_labels)

# 保存模型
vbx.save_models('transform.h5', 'plda.h5')
```

### 3. 加载已训练模型进行推理

```python
from speakerlab.process.vbx_enhancer import VBxEnhancer

# 创建增强器并加载模型
vbx = VBxEnhancer()
vbx.load_models('transform.h5', 'plda.h5')

# 对新数据进行推理
smoothed_labels = vbx.predict(new_embeddings, new_init_labels)
```

## 参数说明

### VBxEnhancer参数

- `lda_dim` (int, default=128): LDA降维后的维度
- `Fa` (float, default=1.0): 充分统计量缩放因子
- `Fb` (float, default=1.0): 说话人数量正则化系数，值越大说话人数越少
- `loopP` (float, default=0.9): HMM自转移概率，值越大说话人切换越少
- `num_em_iters` (int, default=5): PLDA训练的EM迭代次数
- `init_smoothing` (float, default=5.0): 初始gamma平滑参数
- `max_iters` (int, default=10): VBx推理的最大迭代次数

## 输出文件

当使用`cluster_enhance_mode=vbx`时，会生成以下文件：

- `cluster_results_audio.json`: 初始聚类结果
- `pseudo_labels_audio_vbx.json`: VBx平滑后的标签
- `vbx_transform.h5`: 训练的LDA变换参数
- `vbx_plda.h5`: 训练的PLDA模型参数

## 技术细节

### LDA变换

1. 减去均值
2. L2归一化
3. 训练LDA矩阵（基于类间/类内协方差）
4. 应用变换并再次归一化

### PLDA训练

使用Two-Covariance PLDA模型：
- 建模类间协方差（B）和类内协方差（W）
- EM算法迭代优化参数
- 输出变换矩阵和对角协方差

### VBx推理

1. 应用LDA和PLDA变换
2. 初始化gamma（说话人后验概率）
3. 迭代更新：
   - E步：计算后验概率
   - M步：更新模型参数
4. 使用前向-后向算法解码最优说话人序列

## 参考文献

- Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization", Computer Speech & Language, 2022

## 许可

Apache License 2.0
