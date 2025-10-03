待解决：
1. 每段台词语音长度不一，需要check原来代码中的处理方式
2. 现在的项目强制要求语音的sample rate 为16k，需要在预处理阶段调整。（可以采取临时方案，新建一个py文件做这些额外处理）
3. 目前的active speaker detection模块，要求视频帧率为25fps。需要在预处理阶段调整。  
4. face track需要改进。一帧里有多张人脸，优先选择iou最大的。如果有多个iou较大的，在此处断开track，避免错配。
5. 在用3d-speaker时，对face track质量要求应当较高；而在hmm中，可以降低质量要求。
6. face track提取结果需要保存到本地，避免重复计算。
7. 目前仅从真实做了人脸检测，仅包含一个active speaker，且face质量较高的视频做face embedding提取。在 hmm中，需要对中间帧所有检测到的人脸做embedding提取。有两种处理方式：
    a. 筛选包含中间帧的face track，只看该帧是否仅包含一个active speaker，且face质量较高，然后作为补充信息，加入hmm（优先）；
    b. 将一段video segment中 3d-speaker得到的所有active speaker embedding 利用到hmm里
8. 需要考虑，人脸聚类时只对中间帧提取人脸做，还是对多帧做。后者准确度更高，但计算量更大。
9. 现在的语音聚类中，所有 minor cluster 都被重新分配到 major cluster 中。如果以others作为单独一类，需要重新考虑其划分。
10. 可能可以对每一集的数据先做聚类，然后再合并不同集的聚类结果
11. 联合聚类时，每次聚类都会重新调整 audio labels，如果需要微调，需要用匈牙利算法尽可能保持其与上一轮的结果一致，避免分类头反复调整。
12. 联合聚类时，会对语音、视觉各自出现时间进行 merge。在处理电视剧时，这一问题不存在，可以删掉这部分代码，但要注意两者的 align。