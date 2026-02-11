下一步待办：
1. 运行一个以对比学习模型为基础的我们的方法。拿着结果和邓老师讨论，确定要不要加这个环节。具体包括：
- 对比学习效果: eer, 我们方法与 baseline方法的acc, 与不带对比学习的对比。以及，不加对比学习的原因
> 效果不好。另外，对比学习一般在没有合适的预训练模型时使用，用在这里并不合适。
2. 更新论文文字。
3. 根据讨论结果，更新baseline的结果和消融实验的结果。

## Section1: Introduction   
1.2 介绍当前工作与speaker diarization的区别。
> We would like to first clarify that the audio-part task we consider in this work is mainly to perform speaker diarization with visual information and segmentation provided by subtitles in TV plays. Building upon this, we further map the diarization results to character names as a post-processing step after speaker diarization.


## Section2: Related Works
### 内容
2.2 Face Detection and Recognition: 新增对于目前使用的SOTA人脸识别模型相关内容的简要介绍
2.3 Speaker Recognition: 新增对于目前使用的SOTA说话人识别模型相关内容的简要介绍，并明确speaker recognition与speaker diarization的区别
2.4 Multimodal Learning of Videos: 新增一段对于AVD领域的综述。

## Section3: Data
### 内容
3.1 The BigBang Dataset: 补充强调语音片段根据字幕切分，这与VAD中通过voice-activity-detection和handle overlap的功能类似。对于更一般的多模态视频数据集，也可以采用VAD中常用的方式进行语音片段的切分。（也可以放在letter中说）
3.2 介绍ner结果在整个流程中的功能
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
2. Figure 5: 引入协变量信息X（先不改）
> 问题1：具体怎么画？是只加在graphical model里，还是也要在流程图里体现？

## Section5: Real Data Analyses
### 内容
5.1 Name Recognition and Face Detection: 明确指出人名识别结果仅在整个过程结束后，将簇id映射到人名时使用，不影响簇id acc的评估。
5.2 Face and Speaker Recognition: 介绍不做后续自监督学习时的结果。具体而言，将表格重新组织为
- part1(speaker)：介绍不做后续自监督学习时，对比 HADL1 与 pretrain model, cluster ensemble, 和各种 Joint Cluster Baseline in VAD(3D speaker, 约束聚类)的结果。
- part2(face)：介绍不做后续自监督学习时，对比 HADL1 与 pretrain model, cluster ensemble的结果。
> 之所以要重新组织表格，是因为VAD的方法只会根据视觉信号更新说话人标签，人脸标签不变。
5.3 Improvement on Representation Learning: 介绍做了自监督学习之后，HADLI 相较 HADL1 的提升。从acc和表征学习两个角度展示效果。
相当于把原来5.2中HADL_I中的内容挪下来

### 图表
1. Table 3: 更新为最新的结果
2. Figure 6: 更新为最新的结果
> EER 数值也要更新（✅）


## Supplementary Materials
### 内容
1. 在S1之前，新增对AVD方法基本流程的更详细介绍，以及在处理overlap方面的局限性。
2. S4 Enhance MS with Contrastive Learning是否需要保留？
> 问题4: 是否还要把语音模态的对比学习加进来，作为预处理？
3. S5 Alignment of Multimodal Information: 
- S5.1 换成新的对齐方法，
- S5.2 重命名为将簇id映射到人名的过程，改为基于llm的过程。结果不做评估，只在Recognized results中展示
4. S6 Detailed Calculations for Inferring HADL: 更新为新的模型和参数计算方式
> 问题5: 在appendix是否加入消融实验，展示HADL1在利用不同信息时的效果？
5. S7.4 Cost and Speed of HADL: 可以删掉，因为耗时方面相较对比学习没有优势
6. 新增消融实验的结果。在不重复训练的情况下，先后去除
- audio segment length
- active speaker recognition
- face recognition in key-frames
- without HMM component
7. S7.3 新增minDCF, AUC等评估指标的结果 
### 图表
1. Figure S5, S7: 用新结果重新绘制（✅）
2. Table S1: Information of samples for evaluation. 重新统计EER所用标注数据的信息（✅）

## Baseline 待补充 ✅
### 无微调 ✅
#### Speaker Recognition
- CAM++ & SC ✅
- CAM++ & VBx (PLDA参数根据聚类结果估计)✅
- CAM++ & SC + k-means clustering with visual centers ✅
- CAM++ & Pairwise Constrained Clustering(PCC) ✅
#### Face Recognition
- CurricularFace & AHC ✅
### 微调后 ✅
利用这些聚类结果作为伪标签，微调多轮即可。
可以考虑报告微调后模型分类标签/聚类标签的结果。理论上应该报告后者，但根据过往经验，聚类标签的准确性往往不如分类标签。
仅在当前环节加入与CAM++ & CurricularFace & joint SC的比较，因为该聚类方法只能确定部分样本（只有一张人脸）的标签。剩余样本的标签需要使用微调后的模型预测。
#### Speaker Recognition
- CAM++ & SC 
> 效果越训越差是正常的，因为没有任何信息增益
- CAM++ & VBx (PLDA参数根据聚类结果估计)
> vbx初始化既可以采用微调后预测的标签，也可以重新运行SC。估计前者效果更好
- CAM++ & SC + k-means clustering with visual centers
> k-means clustering with visual centers初始化既可以采用微调后预测的标签，也可以重新运行SC。估计前者效果更好
- CAM++ & Pairwise Constrained Clustering(PCC)
> 估计效果不会太理想。因为初始化只需要embedding，不需要标签，而重新聚类对超参数敏感
上述方法都是使用视觉信息/连续性信息单向增强语音聚类结果，因此固定住人脸模块，只微调说话人识别模型即可。在报告结果的时候，也只报告语音识别结果即可。report acc 既可以是微调后分类标签，也可以是重新聚类后的标签。

- CAM++ & CurricularFace & joint SC (具体而言，首先根据活跃说话人检测结果，定位出知道活跃说话人身份的segment。然后抽取这些segment关键帧中对应身份的人脸，利用这些片段做联合聚类，然后更新两种聚类的簇id。之所以选用中间帧人脸而非活跃说话人人脸，是为了方便在下面作为人脸验证的baseline。) 
> cluster ensemble环节必须做聚类。audio only cluster 和 visual only cluster都可以用微调后的标签代替重新聚类。
> 此外，在训练模型时，由于部分样本上的簇标签可能无法覆盖全部角色，在全部样本上根据embedding重新audio/visual only聚类，效果可能比直接使用微调后预测的标签更好。
该方法是双向增强，因此需要微调说话人识别模型和人脸识别模型。在报告结果的时候，语音和人脸都要报告。report acc 只能是微调后分类标签。

#### Face Recognition
- CurricularFace & AHC
> 顾名思义，需要在有了embedding之后，重新聚类，不应该用微调后的预测
> 这个实验可以和 CAM++ & SC 的一起跑，因为两个实验语音、人脸各做各的，没有交互
- CAM++ & CurricularFace & joint SC
> 同上

### 参数功能调整✅
现在的cluster_type有以下几个功能：
1. 控制数据预处理时，是否提取视觉部分数据及提取人脸特征
> 仍用cluster_type参数（重命名为modal_type）
2. 传给cluster文件，控制我们方法中的聚类方式和HMM矫正方式, 以及是否测评人脸识别准确性
> 对于hmm方法，用cluster_type参数；对于baseline方法，使用baseline_method参数确定
3. 传给self_supervised_finetune.py文件，进一步传给cluster文件，以及控制是否测评人脸识别准确性，是否对人脸部分进行聚类
> 对于第一个功能，只需要对我们的方法传递cluster_type参数即可；对于后两个功能，结合两个参数确定

from_preds参数的功能也要重新考虑。