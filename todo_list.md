## 阶段 0：现有项目改进
### 人脸检测替换为MTCNN✅
目前使用的是version-RFB，主要优势是速度快，但精度应该低于MTCNN。考虑到之前已经测试了后者在当前数据集的表现，且效果不错，因此替换为MTCNN。为了降低开发成本，仅修改了lvision_processer.py。在官方给的 example 上测试，DER相较version-RFB无变化。
> a. 为了与之前项目保持一致，目前采用了facenet-pytorch中的MTCNN实现。后续如果要做多卡推理，可以考虑使用onnxruntime，参考https://pypi.org/project/mtcnn-onnxruntime/#description。
> b. 输入输出数据类型、内部顺序已经与 3d speaker 的 pipeline 适配。
> c. version-RFB与version-RFB的效果比较参考：https://blog.csdn.net/qq_14845119/article/details/102729567
### 存在多张人脸时的 face tracking
face track需要改进。一帧里有多张人脸，优先选择iou最大的。如果有多个iou较大的，在此处断开track，避免错配。
> 暂时不做


## 阶段 1：搭建 pipeline
搭建电视剧数据集上的pipeline，包括输入和evaluation（同时包含视觉和语音），仅做聚类不做重新训练。
### 数据集
#### 视频文件
我爱我家：F:\data\TV_series\tv_data\I love my family_new，相较old，声音、画面质量更好，且字幕内容经过了修正。
生活大爆炸：F:\data\TV_series\tv_data\the big bang theory。
里面movie origin包含字幕、音频、视频，movie在此基础上去除了字幕。
#### 字幕文件
选择直接从
1. 超算上生活大爆炸现在用的数据集：F:\data\TV_series\tv_data\the big bang theory\triples_clean
2. 超算上我爱我家现在用的数据集：F:\data\TV_series\tv_data\I love my family\triples_clean
中的speaker_text开始，里面包含了起止时间，片段名称。片段名称和标注文件时对齐的。
> 这部分数据原始获取方式如下
> a. 读取srt文件，延长台词时间窗口、切分语音、转为txt文件，参见 tv_series\code\data_preprocess\movie_seg.py。经过check, triples保存的语音。更进一步的，语音、图像都是根据txt的时间戳切分的，而txt文件时间戳与延长后的台词时间窗口对齐。
> b. 对字幕台词筛选，参见tv_series/code/data_preprocess/testset_construct/0.create_xlsx_to_label.py
#### 我爱我家字幕文件时间戳问题
在一期项目中，我爱我家triples的获取方式是，先把新、旧视频的字幕文件内容统一，然后用旧视频的时间戳，把 audio segment切出来；再用新视频的使劲戳，把中间帧切出来。文本内容和时间戳用新视频的。这种处理方式在仅需要中间帧时可行，但在需要对整段视频做face tracking时不合适。
现在的做法是，直接用新视频的字幕文件和时间戳，切分音频、视频。尽管新视频字幕文件中部分台词结束时间戳偏早（相较音频），但这一问题似乎只出现在同一个人连续说话时，在说话人交替时基本不会发生，因此不影响说话人识别，暂时不做调整。

### 数据预处理
1. stage1：删除下载视频和标注的步骤，转为检查视频、字幕文件、标注文件是否存在。✅
2. stage2：现在的项目强制要求语音的sample rate 为16k，视频帧率为25fps（active speaker detection模块要求），需要在stage 2切分音频&视频时利用ffmpeg进行转换，结果保存到硬盘。✅
3. stage3.1：删除overlap detection，voice_activity_detection和prepare_subseg_json，改为根据字幕文件获取 subseg.json和 vad.json。需要注意的是，由于后面的视觉聚类对视觉片段连续性有要求，因此在得到 vad.json时，需要将间隔小于 2s 的片段合并。考虑到后面数据读入仍采取读入整段音频/视频再定位的方式，因此不需要对音频/视频做切分，也不需要调整起止时间为为0.04的倍数。需要注意的是，subseg.json中每一个 segment的名称、segment的总数量要和之前的数据集对齐，以便于后续 evaluation 代码的复用。✅
> a. 后面如果需要，可以去除对处理过程对字幕文件的依赖，仅利用字幕文件构建标注数据集。这样就彻底变成了 speaker diarization 任务。此时，后面的语音 embedding 提取也可以按原来 batchwise提取每小段时长为1.5s的语音片段做。
> b. 当前人脸聚类要求提取人脸的视频片段在时间上不能过于碎片化，否则很难形成同一 visual speaker id连续出现的时间段，难以实现有效的联合聚类。

### 特征提取
4. stage3.1：根据语言类型，下载合适的语音特征提取 checkpoint；在语音、视觉特征提取时，根据语言类型，选择合适的模型。
> a. CAM++有多个 checkpoint 可用，但是为了能讲清预训练模型来源，英文使用https://www.modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k，中文使用https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_3dspeaker_16k/summary。更多选择参考https://www.modelscope.cn/organization/iic?tab=model。
> b. 视觉特征提取部分，MTCNN、talknet、人脸质量评估、人脸识别模型的checkpoint不随语言类型变化，它们的训练集都涵盖了多语言/人种。
5. stage3.2：使用CAM++，将 batchwise提取每小段语音的 embedding 改为，逐句处理台词语音，提取台词语音的 embedding。保存 embeddings时，与现有方式相同，每集（对应一个视频）存成一个文件，文件名包含集数。
> b. 尽管可以参照speakerlab/bin/infer_sv_batch.py的方式，将完整的语音分割/pad到固定长度的 chunk, 做 embedding 提取，然后将源自相同语音的 chunk embedding 做平均，但考虑到台词普遍没有很长，且CAM++的模型可以处理不等长语音，因此直接逐个处理台词语音，提取 embedding。
6. stage4：根据人种，设置合适的人脸检测超参数，包括在conf中设置的min_face_size和筛选候选框时的min_size，min_prob=0.75）。

> 整体与speaker3d相同，逐个处理视频，读取原始视频中根据时间确定起止帧确定的指定段，运行人脸检测-->以2s为单位处理 shot-->face tracking-->active speaker detection-->提取人脸 embedding。保存 embeddings时，与现有方式相同，每集（对应一个视频）存成一个文件，文件名包含集数。

### 聚类
1. 现在的语音聚类中，所有 minor cluster 都被重新分配到 major cluster 中。如果以others作为单独一类，需要重新考虑其划分。
2. 需要考虑，人脸聚类时只对中间帧提取人脸做，还是对多帧做。后者准确度更高，但计算量更大。
3. 可能可以对每一集的数据先做聚类，然后再合并不同集的聚类结果
4. 联合聚类时，会对语音、视觉各自出现时间进行 merge。在处理电视剧时，这一问题不存在，可以删掉这部分代码，但要注意两者的 align。

### 评估

## 阶段 2：自监督学习
加入根据聚类结果微调模型的代码，记录每一次迭代之后产生的聚类结果，并评估。
需要重点注意以下几方面：
1. 训练过程不需要自己从头写，可以参考已有代码，修正数据读取接口即可。参考https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common。
2. 原来项目用ECAPA_TDNNN逐个处理语音，获取其 hidden_embeddings，并保存到本地。可以参考这种方式。
3. 每轮训练后，保存模型checkpoint，并用该模型做重新聚类，测试效果。
4. 联合聚类时，每次聚类都会重新调整 audio labels，如果需要微调，需要用匈牙利算法尽可能保持其与上一轮的结果一致，避免分类头反复调整。

### 中间结果保存
1. 第一次提取语音 embedding时，保存 hidden——embeddings，这样后续的训练和 infer 都能以 batchwise 方式处理，速度更快。
2. evaluate_fr需要一分为二，将frames containing only one active face, and the face quality must be good enough结果单独保存，再调用人脸特征提取模型。这样，后续也不需要重新读取视频文件，运行检测-跟踪的流程。

## 阶段 3：使用HMM进行联合聚类
将联合聚类替换为hmm
1. 在用3d-speaker时，对face track质量要求应当较高；而在hmm中，可以降低质量要求。
2. 目前仅从真实做了人脸检测，仅包含一个active speaker，且face质量较高的视频做face embedding提取。在 hmm中，需要对中间帧所有检测到的人脸做embedding提取。有两种处理方式：
    a. 筛选包含中间帧的face track，只看该帧是否仅包含一个active speaker，且face质量较高，然后作为补充信息，加入hmm（优先）；
    b. 将一段video segment中 3d-speaker得到的所有active speaker embedding 利用到hmm里

## 阶段 4：使用带约束的聚类