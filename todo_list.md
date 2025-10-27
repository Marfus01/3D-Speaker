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
3. stage3.1：删除overlap detection，voice_activity_detection和prepare_subseg_json，改为根据字幕文件获取 subseg.json和 vad.json。需要注意的是，由于后面的视觉聚类对视觉片段连续性有要求，因此在得到 vad.json时，需要将间隔小于 1s 的片段合并。考虑到后面数据读入仍采取读入整段音频/视频再定位的方式，因此不需要对音频/视频做切分，也不需要调整起止时间为为0.04的倍数。需要注意的是，subseg.json中每一个 segment的名称、segment的总数量要和之前的数据集对齐，以便于后续 evaluation 代码的复用。✅
> a. 后面如果需要，可以去除对处理过程对字幕文件的依赖，仅利用字幕文件构建标注数据集。这样就彻底变成了 speaker diarization 任务。此时，后面的语音 embedding 提取也可以按原来 batchwise提取每小段时长为1.5s的语音片段做。
> b. 当前人脸聚类要求提取人脸的视频片段在时间上不能过于碎片化，否则很难形成同一 visual speaker id连续出现的时间段，难以实现有效的联合聚类。
所得片段数量（以2s为单位时，我爱我家只能提取到740个visual segments）：
I love my family: Number of audio segments: 19225, number of visual segments: 1420
the big bang theory: Number of audio segments: 6829, number of visual segments: 1611

### 特征提取
4. stage3.1：根据语言类型，下载合适的语音特征提取 checkpoint；在语音、视觉特征提取时，根据语言类型，选择合适的模型。✅
> a. CAM++有多个 checkpoint 可用，但是为了能讲清预训练模型来源，英文使用https://www.modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k，中文使用https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_3dspeaker_16k/summary。更多选择参考https://www.modelscope.cn/organization/iic?tab=model。
> b. 视觉特征提取部分，MTCNN、talknet、人脸质量评估、人脸识别模型的checkpoint不随语言类型变化，它们的训练集都涵盖了多语言/人种。
5. stage3.2：使用CAM++，将 batchwise提取每小段语音的 embedding 改为，逐句处理台词语音，提取台词语音的 embedding。保存 embeddings时，与现有方式相同，每集（对应一个视频）存成一个文件，文件名包含集数。✅
> 尽管可以参照speakerlab/bin/infer_sv_batch.py的方式，将完整的语音分割/pad到固定长度的 chunk, 做 embedding 提取，然后将源自相同语音的 chunk embedding 做平均，但考虑到台词普遍没有很长，且CAM++的模型可以处理不等长语音，因此直接逐个处理台词语音，提取 embedding。
6. stage4：根据人种，设置合适的人脸检测超参数，包括在conf中设置的min_face_size和筛选候选框时的min_size，min_prob=0.75）。✅

> 整体与speaker3d相同，逐个处理视频，读取原始视频中根据时间确定起止帧确定的指定段，运行人脸检测-->以2s为单位处理 shot-->face tracking-->active speaker detection-->提取人脸 embedding。保存 embeddings时，与现有方式相同，每集（对应一个视频）存成一个文件，文件名包含集数。

### 语音聚类
生活大爆炸耗时约 3min，我爱我家耗时约 10min。
1. 聚类是 audio only 还是 audio-visual由 sh脚本控制，相应修改cluster_and_postprocess.py。✅
2. 尝试让 run_audio直接加载 video config。✅
3. min_num_spks设置为文本处理所得拥有＞1 个别名的说话人数量，max_num_spks=5*min_num_spks。✅
4. 现在的语音聚类中，包含样本数小于等于 min_cluster_size=1 的所有 minor cluster，都被根据最近邻原则，重新分配到 major cluster 中。由于后续希望以others作为单独一类，需要先检查余弦相似度是否较高，然后再分配。✅
5. 调节超参数：尝试不同的mer_cos取值，发现对聚类效果影响不大。如果两个簇的聚类中心cos-sim>mer_cos，则会被合并。在每一个取值下，在out中打印所用的阈值，同时将使用的mer_cos阈值添加到输出 json 文件的文件名中。（经过尝试，仍设置为0.8）✅
> 生活大爆炸：不同的mer_cos取值对聚类结果影响不大。尤其是，聚类数目和各聚类簇的大小分布都比较稳定，只是最大的几个簇的成员会有非常小的变化。
> 我爱我家：不同的mer_cos取值下，major cluster仍然相对稳定，但是大小略有变化。
6. 调节超参数：尝试不同pval。pval越小，相似度矩阵越稀疏，从而聚类结果中簇数目越多，top簇的大小越小。✅
> 生活大爆炸：采用原有的0.032，即能得到具有明显数量优势的top5簇。降至0.016/0.008后，第二大的簇大小从 2000+ 降至 1800+， 剩余top5簇变化不大。其余簇除新增了 2 个大小≈100 的簇外，变化不大。
> 我爱我家：采用原有的0.032，只能得到 4 个包含元素>1的簇。降到0.008后，效果相对较好。
7. 未来可以通过在验证集上自动搜参确定pval。

### 视觉聚类
两个数据集上的聚类均能在 1min内完成。
1. 现在的视觉聚类中，包含样本数小于等于 min_cluster_size=1 的所有 minor cluster，都被根据最近邻原则，重新分配到 major cluster 中。由于后续希望以others作为单独一类，需要先检查余弦相似度是否较高，然后再分配。✅
2. 现在的视觉聚类中，需要调整的核心超参数是fix_cos_thr，对应层次聚类的停止阈值。fix_cos_thr越大，聚类簇数目越多。经过尝试，暂时仍设置为原来的默认值0.25。✅
> 生活大爆炸：即使阈值调到0.5，仍然只有 2 个簇大小占据明显优势，且大小排名靠前的簇规模相较 0.25 时变化不大。
> 我爱我家：次大簇规模相较 0.25 时明显变大，但整体情况变化不大。
3. 额外获取视觉聚类结果，保存为 json文件，方便后续评估。✅
> 实验结果显示，约 20% 的 audio segment 有 active visual speaker cluster labels，其中 label 唯一的质量明显高于非唯一的。
> 生活大爆炸：Among 6829 audio segments, 1314 segments have active visual speaker cluster labels, 1303 segments have unique active visual speaker cluster labels. 对于有>1个active visual speaker cluster labels的音频片段，也均能通过多数投票获得其视觉聚类标签。唯一的视觉聚类标签对应的说话人识别准确率为 93.81%。作为对比，纯语音聚类的准确率为 89.95%。
> 我爱我家：Among 19225 audio segments, 4192 segments have active visual speaker cluster labels, 4147 segments have unique active visual speaker cluster labels. 对于有>1个active visual speaker cluster labels的音频片段，也均能通过多数投票获得其视觉聚类标签。唯一的视觉聚类标签对应的说话人识别准确率为 94.80%。作为对比，纯语音聚类的准确率为 74.00%。

### 联合聚类
#### 修改
1. 在对语音、视觉各自出现时间进行 merge，以及 align的过程中，要注意它们是否来自同一集。✅
> 通过对每集数据的起始时间施加偏移量实现，这可以确保每集数据的时间不重叠。
2. 固定语音、人脸聚类的seed，确保每次运行结果一致。✅
#### 现有过程梳理
重新梳理现有过程：
a. 纯语音聚类，给所有n_a 个audio_segments分配语音聚类标签alabels。alabels共包含k_a个簇，簇id的集合={0,1,...,k_a-1}。
b. 纯视觉聚类，分为4步：
  在两步筛选中，b.2直接丢弃的是出现不连贯的faces，b.3直接丢弃的是喝audio segments重叠度不高的视觉簇。
  b.1. 给所有n_f 个faces分配视觉聚类标签vlabels（共包含k_v个簇）。前期测试显示，这一步的聚类就衡量说话人身份而言质量很高。
  b.2. 根据face出现的时间戳，将n_f 个faces整理为 1个v_list，并对v_list中存在的簇id从 k_a 开始重新编号。在整理过程中，会丢弃那些仅包含一帧的visual segments。因此v_list中包含的簇数量k_v'<=k_v。
  b.3 对于 v_list中包含的每一个簇，获取与其中某个visual segment重叠时长＞1s的所有audio segments，计算这些audio segments的embedding均值向量，作为该视觉簇对应的说话人embedding。如果某个视觉簇按照这一原则没有找到对应的说话人embedding，则丢弃该视觉簇。最终，得到 k_v''<=k_v' 个视觉簇，这些视觉簇均有对应的说话人embedding。
  > 当k_v''< k_v'时，视觉簇id集合是{k_a, k_a+1,..., k_a+k_v'-1} 的子集。
  b.4 根据b.3的结果，更新 v_list，同步获取k_v''个视觉簇对应的说话人embedding、在v_list中的总出现时间
  > 需要check两步丢弃后，产生的vad视觉聚类结果数量、质量有多高
c. 结合视觉聚类信息，调整每一个 audio_segment 的说话人标签。具体而言：
  c.1. 获取该 audio_segment 对应的alabel，以及该alabel对应语音簇中所有audio_segments的起止时间
  c.2. 统计所有k_v''个视觉簇，在v_list中的visual segments与当前语音簇的audio_segments重叠的总时长，记为 overlap_vspk。从中筛选出，重叠时长>0.5或者在visual segment总时长中占比>0.5的视觉簇，记为 overlap_vspk。
  c.3. 如果 overlap_vspk非空，则将当前 audio_segment 的说话人标签调整为 overlap_vspk中对应的说话人embedding与当前 audio_segment embedding 余弦相似度最高的视觉簇对应的说话人标签。反之，则保持原有的alabel不变。
d. 对经过调整的 alabels 重新编号，确保标签为连续整数，记为 final_alabels。此时，final_alabels的标签集合为{0,1,...,k_final-1}，其中 k_final <= k_a + k_v''。
#### 关键指标分析
检查以下几个方面，以便于后续确定语音簇与视觉簇的对应关系：
1. 检查联合聚类中，每步视觉簇筛选之后，剩余faces的数量和视觉簇的数量。
2. 两步筛选之后剩余的视觉簇覆盖样本的规模、准确度。
3. 检查有多少audio segments的cluster labels被视觉信息矫正过。
> 聚类结果中各个聚类簇的大小等信息参见'/data/home/scv7387/run/tv_series_plus/out/搭建pipeline/3dspeaker联合聚类/原始/细节'中的out文件。
##### 生活大爆炸
视觉簇筛选结果：
[INFO] 2025-10-26 12:04:23 visual_vad_initial clustering results: total 17 clusters, total 4766 samples.
[INFO] 2025-10-26 12:04:23 visual_vad_step1 clustering results: total 14 clusters, total 3894 samples. 这一步每个簇的size都有所减少，上一步中size最小的3个簇被丢弃。
[INFO] 2025-10-26 12:04:29 visual_vad_step2 clustering results: total 9 clusters, total 3849 samples. 这一步中，上一步中size最小的5个簇被丢弃，其余簇size不变
筛选后视觉聚类结果的质量：
Among 6829 audio segments, after filter, 863 segments have active visual speaker cluster labels, all of them have a majority active visual speaker cluster label, and 857 segments have unique active visual speaker cluster labels.
> 作为对比，筛选之前，Among 6829 audio segments, 1314 segments have active visual speaker cluster labels, all of them have a majority active visual speaker cluster label, and 1303 segments have unique active visual speaker cluster labels. 就accuracy而言，筛选之后略有下降（在0.5%以内）。
语音簇调整结果：
[INFO] 2025-10-26 12:04:29 audio_before_joint clustering results: total 27 clusters, total 6829 samples.
[INFO] Total 6213 audio segments in 7 audio clusters are re-assigned according to visual clusters during joint clustering.
[INFO] 2025-10-26 12:04:29 audio_after_joint clustering results: total 29 clusters, total 6829 samples.
进一步分析，audio_before_joint中size>1的簇中，只有大小为382、113、74的三个簇未被调整。

标注数据集上被匹配到非others说话人的vad校正后语音簇（大小>1）：
audio_after_joint clustering ids(counts): 27(2218), 28(1756), 29(1099), 30(685), 8(382), 38(180), 37(115), 10(113), 31(111), 9(74), 34(33), 16(31), 33(16)
audio_vision_vad(arranged) clustering ids(counts): ☑️0(2218), ☑️1(1756), ☑️2(1099), ☑️3(685), ☑️4(382), 5(180), 6(115), 7(113), 8(111), 9(74), 10(33), 11(31), 12(16)
正好为最大的5个，但是其中有一个来自语音聚类

标注数据集上被匹配到非others说话人的纯语音聚类簇（大小>1）：
audio_only clustering ids(counts): ☑️0(2281), ☑️1(2029), ☑️2(1066), ☑️3(659), ☑️4(382), 5(129), 6(113), 7(74), 8(33), 9(31), 10(16)
仍为最大的5个。

##### 我爱我家
视觉簇筛选结果：
[INFO] 2025-10-26 12:11:53 visual_vad_initial clustering results: total 51 clusters, total 17684 samples.
[INFO] 2025-10-26 12:11:53 visual_vad_step1 clustering results: total 45 clusters, total 15478 samples. 这一步每个簇的size都有所减少，上一步中size最小的6个簇被丢弃（观察剩余簇大小得到）。
[INFO] 2025-10-26 12:12:48 visual_vad_step2 clustering results: total 32 clusters, total 15263 samples. 这一步中，上一步中size最小的7个簇和大小为58、42、25、9的簇、以及大小为27、33的各一个簇被丢弃，其余簇size不变
筛选后视觉聚类结果的质量：
Among 19225 audio segments, after filter, 3054 segments have active visual speaker cluster labels, all of them have a majority active visual speaker cluster label, and 3039 segments have unique active visual speaker cluster labels.
> 作为对比，筛选之前，Among 19225 audio segments, 4192 segments have active visual speaker cluster labels, 4147 segments have unique active visual speaker cluster labels. 就accuracy而言，筛选之后略有提升（约1%）。
语音簇调整结果：
[INFO] 2025-10-26 12:12:48 audio_before_joint clustering results: total 28 clusters, total 19225 samples.
[INFO] Total 19207 audio segments in 10 audio clusters are re-assigned according to visual clusters during joint clustering.
[INFO] 2025-10-26 12:12:54 audio_after_joint clustering results: total 50 clusters, total 19225 samples.
进一步分析，audio_before_joint中所有 size>1 的簇都被重新分配了。

标注数据集上被匹配到非others说话人的vad校正后语音簇（top20）：
audio_after_joint clustering ids(counts): 33(3953), 28(3817), 32(3076), 30(991), 47(829), 34(700), 31(645), 29(613), 37(583), 57(521), 39(472), 68(428), 41(377), 62(294), 64(230), 63(227), 72(222), 54(169), 65(162), 46(152)
audio_vision_vad(arranged) clustering ids(counts): ☑️0(3953), ☑️1(3817), ☑️2(3076), ☑️3(991), 4(829), ☑️5(700), ☑️6(645), 7(613), ☑️8(583), 9(521), 10(472), 11(428), 12(377), 13(294), 14(230), 15(227), 16(222), 17(169), 18(162), 19(152)
并非最大的7个

标注数据集上被匹配到非others说话人的纯语音聚类簇（大小>1）：
audio_only clustering ids(counts): ☑️0(4578), ☑️1(3317), ☑️2(3093), 3(2920), ☑️4(2691), ☑️5(868), ☑️6(859), 7(387), 8(331), ☑️9(163)
并非最大的7个


### 评估（只需要考虑语音部分）
1. 将根据聚类结果获取的output rttm改为key是 segment_id, value是聚类簇 Index 的字典，方便后续 evaluation。✅
2. 用匈牙利算法匹配聚类簇与标注说话人，获得理想情况下最佳的accuracy。✅
> 多分类 accuracy 计算公式：https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification
#### 说话人acc计算具体情况
##### 方法1（目前采用）
当前聚类（不管是纯语音聚类还是vad 人脸）所得簇数量k普遍多于标注说话人数量n。在计算 accuracy 时，目前采取的方式是：
1. 在标注数据上，统计说话人为i, 聚类簇为j的音频片段数量，得到计数矩阵cm，大小为 n x k。
2. 在cm的行之后，补充 k - n 行 min(cm)-1，使得cm变为 k x k。
3. 使用匈牙利算法，在cm上寻找最大匹配，从中筛选出 rows < n 的匹配对，将剩余的 cols都分配给others。
这等价于，从聚类结果中挑选与说话人标注匹配情况最好的 n 个簇，使用匈牙利算法将它们与标注说话人进行匹配，剩余的簇都划为 others。
##### 方法2
方法1的简化版本时，从聚类结果中选择最大的 n - 1 个簇与标注说话人进行匹配，剩余的簇都划为 others：
1. 与方法1相同，获取计数矩阵 cm，大小为 n x k。
2. 如果n不等于k，以n< k为例，挑选cm中列和最大的 n 列，组成新的矩阵 cm'，大小为 n x n。
3. 在cm'上使用匈牙利算法，寻找最大匹配（row index需要对应到cm之中）。将cm剩余的 k - n 列都划为 others。
从逻辑上讲，这也是合理的，因为较小的簇很有可能属于次要说话人，只是在标注过程中未加以区分。但风险是，标注中靠后的说话人与未标注靠前的说话人大小差异可能不大，这种方式有将对应others的簇与对应标注说话人的簇混淆的风险，从而导致 accuracy 被低估。
##### 方法3
另外一种很自然的方法是，将每一个聚类簇各自独立地匹配到counts最高的标注说话人上，但这种方式会导致多个聚类簇匹配到同一个标注说话人，从而导致 accuracy 被高估。

## 阶段 2：加入仅处理说话人序列的HMM
### 使用最基本的Category HMM✅
#### 基本想法
阶段1联合聚类获得说话人聚类簇标签之后，使用一个n_states = n_clusters的HMM，对说话人标签进行平滑。
#### 实现细节
测试发现，如果完全采纳HMM的结果，accuracy会下降。因此，考虑计算隐状态解码结果的后验概率，仅采纳后验概率最高的前prop_keep比例的解码状态，用来修正观测序列。
为了将隐状态 id 与 观测状态 id 对齐，使用了朴素的方式，即认为每个隐状态对应的观测状态是该隐状态出现时，观测状态出现频率最高的那个。
#### 测试结果
1. 在生活大爆炸上测试，prop_keep=0.01时，第一组的accuracy有所提升，其余均下降。耗时约7min。进一步提升prop_keep，accuracy下降更多。
2. 在我爱我家上测试，prop_keep=0.01时，第一组的accuracy有所提升，其余均下降。耗时约54min。进一步提升prop_keep，accuracy下降更多。
## 使用带有协变量的Category HMM
### 实现方式
#### 簇标签对齐
必须确定人脸簇、语音簇之间的映射关系，hmm_X才能成功运行。
然而，现有联合聚类相当于筛选出了一些置信度高的语音-视觉簇配对，然后直接用它们完成了矫正。但对剩余的audio segments，它们可能有对应的face，但是并没有被用到。
目前借鉴联合聚类，采用以下方式，将vison簇标签的id与矫正后说话人标签的id对齐：
1. 假如主要说话人数量为n-1，则挑选前2(n-1)个语音簇，作为潜在的主要说话人簇，得到集合A。✅
2. 对集合A中的每一个簇，如果它与某个视觉簇对应（在reassigned过程中建立），则回溯该视觉簇在filter之前包含的所有faces，据此建立协变量。✅
3. 对于剩余的语音簇，划为others。✅
> a. 就对vad信息的应用而言，上述方法丢弃了filter vlabels的过程中，step2去除的小簇。因为这些小簇难以高质量地与语音簇id建立映射关系，且数量较少，对整体影响不大。
> b. 如果A中的某个语音簇没有对应的视觉簇，则该语音簇的协变量为空。
> c. 将较小簇统一划为others，也有助于减少HMM的状态数，降低计算复杂度。
> d. 其他方式(为了方便与联合聚类对比，保证协变量质量较高，仍然采用上述方式)：
>   d.1 可以参考之前电视剧参考文献、以及draft中对齐人脸、说话人聚类的方式。
>   d.2 可以充分利用同一个audio segment内，绝大多数情况下只有一个active visual speaker cluster label的事实，将这一label直接赋予整个时间段，然后统计重叠时长，建立映射关系。
#### 其他
1. 将写好的hmm_X迁移到当前项目中，完成从簇标签到hmm输入格式的转换。✅
2. 测评acc时，将cluster中手动归出的others簇直接对应到'others'说话人上，仅对剩余簇使用匈牙利算法匹配。✅
3. 记录优化结束后，HMM的各个参数。
4. 设置gamma1=gamm2的接口
### 结果
过程中，说话人转移概率的数值优化存在Warning: Speaker transition parameters optimization did not converge.的问题，但是暂时没有很好的解决方案。


## 阶段 3：测试带约束的聚类



## 阶段 4：使用HMM进行联合聚类
将联合聚类的结果用hmm进一步修正。
### 人脸模态聚类
1. 人脸部分，active speaker face数量较少，但质量较高；来自中间帧的人脸存在区域不完整、误检、模糊等问题。两者数量基本相当，可以考虑
a. 先分别做聚类，然后将后者与前者对齐；
b. 只对前者做聚类，然后最近邻分配后者的label。
c. 直接将两者合并做聚类。
2. 由于active speaker face和中间帧人脸质量存在差异，如果分别聚类，后者层次聚类的停止阈值fix_cos_thr可能需要重新调整。
3. 可能可以对每一集的数据先做聚类，然后再合并不同集的聚类结果

1. 在用3d-speaker时，对face track质量要求应当较高；而在hmm中，可以降低质量要求。
2. 目前仅从真实做了人脸检测，仅包含一个active speaker，且face质量较高的视频做face embedding提取。在 hmm中，需要对中间帧所有检测到的人脸做embedding提取。有两种处理方式：
    a. 筛选包含中间帧的face track，只看该帧是否仅包含一个active speaker，且face质量较高，然后作为补充信息，加入hmm（优先）；
    b. 将一段video segment中 3d-speaker得到的所有active speaker embedding 利用到hmm里


## 阶段 5：自监督学习
加入根据聚类结果微调模型的代码，记录每一次迭代之后产生的聚类结果，并评估。
### 整体流程
1. 训练过程不需要自己从头写，可以参考已有代码，修正数据读取接口即可。参考https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common。
2. 原来项目用ECAPA_TDNNN逐个处理语音，获取其 hidden_embeddings，并保存到本地。可以参考这种方式。
3. 每轮训练后，保存模型checkpoint，并用该模型做重新聚类，测试效果。
4. 联合聚类时，每次聚类都会重新调整 audio labels，如果需要微调，需要用匈牙利算法尽可能保持其与上一轮的结果一致，避免分类头反复调整。
5. 中间每一步，可以尝试以下两种伪标签：
a. 根据视觉信息修正后的语音聚类标签（簇数不等于说话人）
b. 视觉聚类中，根据聚类簇覆盖各集的情况和大小，获取 others 簇-->把语音和视觉对齐-->同时考虑簇大小和包含的人名数量，将语音聚类簇与标注说话人的对齐。
### 中间结果保存
1. 第一次提取语音 embedding时，保存 hidden——embeddings，这样后续的训练和 infer 都能以 batchwise 方式处理，速度更快。
2. evaluate_fr需要一分为二，将frames containing only one active face, and the face quality must be good enough结果单独保存，再调用人脸特征提取模型。这样，后续也不需要重新读取视频文件，运行检测-跟踪的流程。✅
### 评估
#### 语音部分
1. 计算eer。
> accuracy对聚类阈值相对敏感，eer是更能反映模型本身能力的指标。
#### 人脸部分
只需要对比初始模型聚类效果和微调后模型聚类效果即可。（大修时在伪标签获取方式中，加入cai2022的方法）
##### 数据集构建
中间帧人脸原有 Index 和现在的 Index 不一样（原来是直接获取中间帧，现在是固定fps后再获取）。需要
1. 先在之前的项目文件夹下，比对带有绿框标注的中间帧和裁剪得到的人脸，将各个人脸的 bbox 与人名对应，
2. 将当前项目从每个中间帧人脸检测的结果，使用匈牙利算法，根据帧内总iou最大-->iou>0.5，获取大部分当前项目提取人脸的姓名标注；
3. 手工补齐剩余人脸的姓名标注