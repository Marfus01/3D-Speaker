待解决：
1. 台词本中语音长度较长，是否需要做进一步拆分？
2. 现在的项目强制要求语音的sample rate 为16k，需要在预处理阶段调整。（可以采取临时方案，新建一个py文件做这些额外处理）
3. 目前的active speaker detection模块，要求视频帧率为25fps。需要在预处理阶段调整。  
4. face track需要改进。一帧里有多张人脸，优先选择iou最大的。如果有多个iou较大的，在此处断开track，避免错配。
5. 在用3d-speaker时，对face track质量要求应当较高；而在hmm中，可以降低质量要求。
6. face track提取结果需要保存到本地，避免重复计算。
7. 目前仅从真实做了人脸检测，仅包含一个active speaker，且face质量较高的视频做face embedding提取。在 hmm中，需要对中间帧所有检测到的人脸做embedding提取。有两种处理方式：
    a. 筛选包含中间帧的face track，只看该帧是否仅包含一个active speaker，且face质量较高，然后作为补充信息，加入hmm（优先）；
    b. 将一段video segment中 3d-speaker得到的所有active speaker embedding 利用到hmm里
8. 需要考虑，人脸聚类时只对中间帧提取人脸做，还是对多帧做。后者准确度更高，但计算量更大。