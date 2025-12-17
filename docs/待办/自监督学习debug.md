## Step0 加入调试接口


## Step1
确定是分类微调-分类给出伪标签-HMM平滑，还是对比学习微调-聚类给出伪标签-HMM平滑。确定范式之后，可以进一步采用样本置信度及频率加权、随机裁剪、
渐进解冻+学习率、正则化等技巧。
### 分类范式
1. 修改数据集，将所有样本用于训练✅
2. 在自监督学习文件中，在每个解冻部分embedding参数微调的epoch中，补充利用当前模型计算分类标签及分类概率的过程，并根据分类概率定义不确定度（top2分类概率之差），并将分类标签和不确定度保存到epoch自己的子文件夹✅
3. 在cluster文件中，加入直接读取语音伪标签的接口，如果语音伪标签存在，则无需执行纯语音聚类；此外，除初始化外，人脸部分也不再重新聚类，直接加载聚类伪标签/模型预测结果✅
4. 在acc计算中，加入模式valid/test，提前检查valid行号文件是否存在，不存在则创建。在两种模式下，各自调用不同的部分样本✅
5. 每一次迭代，微调停止按照预设最大epoch/最优valid acc(有一个patience)停止✅
> 不要改成根据pseudo label的acc停止，可能会导致过拟合。

## 计算速度优化
### 构建 hidden feature的dataset
在自监督学习文件中，根据解冻层数，构建 hidden feature的dataset，避免每次都从头计算embedding✅
### 调整每次HMM平滑的迭代次数
cluster文件中，如果加载了hmm参数，将收敛阈值调大一些，比如1e-1，避免过多迭代。✅
#### 按层解冻的实验结果
注意到，当前解冻层数为2, 4, 8, 13的实验，最优valid acc、对应的test acc和出现的 round 分别为：
- 纯语音聚类：valid acc 0.9075, test acc 0.9094
- 语音-vad联合聚类（hmm观测）：valid acc 0.8950, test acc 0.9088
- 初始化（hmm解码，采纳5%）: valid acc 0.9000, test acc 0.9162
- 解冻2层（acc是根据某一轮的解码评估）：best valid acc 0.9050, test acc 0.9194, round 3
- 解冻4层（acc是根据某一轮的解码评估）：best valid acc 0.9000, test acc 0.9162, round initial
- 解冻8层（acc是根据某一轮的解码评估）：best valid acc 0.9175, test acc 0.9181, round 1
- 解冻13层（acc是根据某一轮的解码评估）：best valid acc 0.9150, test acc 0.9200, round 0
#### 解冻最后一个dense block的实验结果
- round 0(prediction): best valid acc 0.8925, test acc 0.9069
- round 0(hmm解码): best valid acc 0.8950, test acc 0.9081
- best round(at round 3, prediction): best valid acc 0.9075, test acc 0.9200
- best round(at round 3, hmm解码): best valid acc 0.9100, test acc 0.9206
> 指定文件夹test acc直接测量方式：python local/compute_acc_spk.py --result_dir "/data/home/scv7387/run/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-d
/scv7387/run/tv_series_plus/dataset/the big bang theory/annotation/text_annotated.xlsx" --mode "test"构建数据集/exp1/round3/ft_epoch_3" --ref_xlsx

### 其他
1. 增大batch size-->128 --效果不佳，仍保持64✅

## 其他
1. 分类时，不确定性的准则改为概率最大类的概率--效果不佳，仍保持为top2类预测概率之差✅
2. 在cluster文件中，获取多个unreliable pp的json，自监督文件计算所有这些结果的valid acc，选择最优的pp进行后续迭代。如果发现所有pp的valid acc相较观测都没有提升，则停止迭代。✅
3. BCE loss使用加权版本，权重与类别不平衡相关✅
4. 使用hidden feature时，是否要加入某种数据增强？--暂时不需要，一期的时候也没做
5. 分类器可以用speakerlab/models/campplus/classifier.py中定义的LinearClassifier--暂时不做