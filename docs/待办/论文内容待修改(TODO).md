## Section2: Related Works
### 内容
2.2 Face Detection and Recognition: 新增对于目前使用的SOTA人脸识别模型相关内容的简要介绍
2.3 Speaker Recognition: 新增对于目前使用的SOTA说话人识别模型相关内容的简要介绍，并明确speaker recognition与speaker diarization的区别
2.4 Multimodal Learning of Videos: 新增一段对于AVD领域的综述。

## Section3: Data
### 内容
3.1 The BigBang Dataset: 补充强调语音片段根据字幕切分，这与VAD中通过voice-activity-detection和handle overlap的功能类似。对于更一般的多模态视频数据集，也可以采用VAD中常用的方式进行语音片段的切分。（也可以放在letter中说）
### 图表
1. Figure 3,4 中的(e):  Auto-correlation of characters’ appearance sequence，改为指定角色出现在前一帧时，后一帧也出现的概率。对于The LoveFamily Dataset，需要额外标注一些数据，避免Yanhong的取值为0.（3️⃣）

## Section4: HMM-Assisted Deep Learning for Video Analysis
### 内容
4.2 Alignment of Multimodal Information: 换成现有的聚类、对齐方式
4.3 The Statistical Model for HMM-Assisted Deep Learning: 在Full Model中，引入Active speaker face、语音长度等信息，更新公式和参数
4.4 Parameter Estimation: 更新所有参数的估计方式
4.5, 4.6: predicted probability计算及使用可以移除，因为实际训练会使用全部样本
### 图表
1. Table 1: Selection of pre-trained deep learning models in this study. 把人脸、语音识别模型换成目前使用的SOTA模型。
2. Figure 5: 引入协变量信息X
> 问题1：具体怎么画？是只加在graphical model里，还是也要在流程图里体现？

## Section5: Real Data Analyses
### 内容
5.1 Name Recognition and Face Detection: 明确指出人名识别结果仅在整个过程结束后，将簇id映射到人名时使用，不影响簇id acc的评估。
5.2 Face and Speaker Recognition: 介绍不做后续自监督学习时的结果。具体而言，将表格重新组织为
- part1(speaker)：介绍不做后续自监督学习时，对比 HADL1 与 pretrain model, cluster ensemble, 和各种 Joint Cluster Baseline in VAD(3D speaker, 约束聚类)的结果。
- part2(face)：介绍不做后续自监督学习时，对比 HADL1 与 pretrain model, cluster ensemble的结果。
> 之所以要重新组织表格，是因为VAD的方法只会根据视觉信号更新说话人标签，人脸标签不变。
> NOTE1: speaker recognition的Baseline较多，除了原来的Cluster Ensemble之外，VAD的结果也都可以比；face recognition的Baseline似乎只有原来的Cluster Ensemble
> 问题2: Integrating Audio, Visual, and Semantic Information for Enhanced
Multimodal Speaker Diarization on Multi-party Conversation这篇文章给出的约束聚类方法效果不如3D speaker中默认的联合聚类方法。是否要把这两种方法都放进来。
> 问题3: Cluster Ensemble的具体做法：原来没有VAD信息，只能用只包含一张人脸的语音-人脸对；现在是仍这样做，还是用语音-active face对？如果用后者，Face recognition就更没有Baseline了
5.3 Improvement on Representation Learning: 介绍做了自监督学习之后，HADLI 相较 HADL1 的提升。从acc和表征学习两个角度展示效果。
相当于把原来5.2中HADL_I中的内容挪下来
> 问题 4: 是否需要利用5.2中各个Baseline产生的伪标签，都做一波自监督学习，对比它们的acc？

### 图表
1. Table 3: 更新为最新的结果
2. Figure 6: 更新为最新的结果
> EER 数值也要更新（✅）


## Supplementary Materials
### 内容
1. 在S1之前，新增对AVD方法基本流程的更详细介绍
2. S4 Enhance MS with Contrastive Learning是否需要保留？
> 问题4: 是否还要把语音模态的对比学习加进来，作为预处理？
3. S5 Alignment of Multimodal Information: 
- S5.1 换成新的对齐方法，
- S5.2 重命名为将簇id映射到人名的过程，结果不做评估，只在Recognized results中展示
4. S6 Detailed Calculations for Inferring HADL: 更新为新的模型和参数计算方式
> 问题5: 在appendix是否加入消融实验，展示HADL1在利用不同信息时的效果？
5. S7.4 Cost and Speed of HADL: 可以删掉，因为耗时方面相较对比学习没有优势
### 图表
1. Figure S5, S7: 用新结果重新绘制（✅）
2. Table S1: Information of samples for evaluation. 重新统计EER所用标注数据的信息（✅）