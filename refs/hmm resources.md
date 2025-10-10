目前没有找到能直接用的，但是可以参考以下资源：
1. github 上的 popular projects: https://github.com/topics/hidden-markov-model
2. [momentuHMM](https://cran.r-project.org/web/packages/momentuHMM/vignettes/momentuHMM.pdf#page=9.73) R包，支持多数据流、多状态、协变量依赖参数、层次结构；参数推断​​：基于MLE + 前向算法，Viterbi算法获取最优状态序列
3. [Deeptime](https://iopscience.iop.org/article/10.1088/2632-2153/ac3de0/pdf): 同时支持离散与连续输出模型，使用Baum-Welch 完成参数估计；使用 Viterbi 隐藏状态解码，并采用多种初始化策略避免局部最优。Core implementations are in C++ with python bindings via pybind11.
4. [hmmlearn](https://github.com/hmmlearn/hmmlearn/): Python库，支持高斯、混合高斯、离散输出的HMM，使用EM算法进行参数估计，Viterbi算法进行解码。项目整体较为简洁，底层是用 C++ 实现的，速度较快。
5. Forward-Backward Algorithm 简单示例：https://github.com/Mogeng/IOHMM/tree/master/IOHMM
6. Viterbi Algorithm 简单示例：https://github.com/AntoinePassemiers/ArchMM/tree/master

计划以 4 为基础进行二次开发