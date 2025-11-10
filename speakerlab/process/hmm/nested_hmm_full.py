import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.utils import check_random_state
from functools import partial

from .monitor import ConvergenceMonitor
import time, copy, itertools


## 内存相关
### 1. 当处理实际电视剧数据时， U, V 矩阵尺寸较大，有可能导致内存不足。如果出现此类问题，可以先将每季的UV统计量存到本地，然后再读取进行累积


class NestedHMM_full():
    """
    嵌套隐马尔可夫模型
    
    Parameters
    ----------
    n_actors : int
        演员数量
    n_iter : int, optional (default: 100)
        最大迭代次数
    tol : float, optional (default: 1e-2)
        收敛阈值
    verbose : bool, optional (default: False)
        是否打印详细信息
    params : str, optional (default: "abcdefgh")
        控制哪些参数被更新
    init_params : str, optional (default: "abcdefgh")
        控制哪些参数被初始化
    random_state : int or RandomState, optional
        随机种子
    """
    
    def __init__(self, n_actors, n_iter=100, tol=1e-2, verbose=False,
                 params="abcdefghij", init_params="abcdefghij", random_state=None):
        self.n_actors = n_actors    # 演员数量
        self.n_face_states = 2 ** n_actors  # 面部状态数量 (每个演员有2个状态)
        self.n_iter = n_iter    # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印详细信息
        self.params = params    # 控制哪些参数被更新
        self.init_params = init_params  # 控制哪些参数被初始化
        self.random_state = random_state

        # 添加缓存变量，避免反复计算
        ## 所有可能的面部配置和协变量配置
        self.face_configs_arr = np.array(self._enumerate_face_configs()) # shape (n_face_states, n_actors)。每行对应 row_index 在self.n_actors位下的二进制表示
        self.X_arr = np.vstack([np.eye(self.n_actors), np.zeros((1, self.n_actors))])  # shape (n_actors+1, n_actors), 每行表示一个one-hot编码的协变量配置或全零配置

        # 创建监控器
        self.monitor_ = ConvergenceMonitor(tol, n_iter, verbose)

    def _check_and_set_n_features(self, S_hat_onehot, F_hat, X_onehot, F_potential_states_idxs):
        """
        验证嵌套HMM数据格式，要求
        - S_hat_onehot: 说话人观测，one-hot编码，形状 (n_samples, n_actors)
        - F_hat: 面部出现，二进制数据，形状 (n_samples, n_actors)
        - X_onehot: 协变量，one-hot编码，形状 (n_samples, n_actors)
        - F_potential_states_idxs: 面部观测的候选状态集合索引列表，长度为 n_samples，每个元素是一个列表，记载该样本的所有候选状态索引
        """
        if S_hat_onehot.shape != F_hat.shape:
            raise ValueError(f"S_hat_onehot and F_hat must have the same shape, got {S_hat_onehot.shape} and {F_hat.shape}")

        if X_onehot.shape != S_hat_onehot.shape:
            raise ValueError(f"X_onehot and S_hat_onehot must have the same shape, got {X_onehot.shape} and {S_hat_onehot.shape}")

        if S_hat_onehot.shape[1] != self.n_actors:
            raise ValueError(f"Expected {self.n_actors} actors, got {S_hat_onehot.shape}")
            
        # 检查 S_hat_onehot 是one-hot编码
        if not np.allclose(S_hat_onehot.sum(axis=1), 1):
            raise ValueError("S_hat_onehot must be one-hot encoded (each row sums to 1)")
        
        # 检查F_hat是二进制数据
        if not np.all(np.isin(F_hat, [0, 1])):
            raise ValueError("F_hat must contain only binary values (0 or 1)")

        # 检查X_onehot是one-hot编码（可以全为零）
        if not np.all(np.isclose(X_onehot.sum(axis=1), 0) | np.isclose(X_onehot.sum(axis=1), 1)):
            raise ValueError("X_onehot must be one-hot encoded or all zeros (each row sums to 0 or 1)")
        
        # 检查F_potential_states_idxs的长度
        if len(F_potential_states_idxs) != S_hat_onehot.shape[0]:
            raise ValueError(f"F_potential_states_idxs length must equal number of samples, got {len(F_potential_states_idxs)} and {S_hat_onehot.shape[0]}")

    def _validate_lengths(self, X, lengths):
        """
        验证序列长度，要求lengths元素之和等于X的样本数
        """
        if lengths is None:
            return [len(X)]
        
        if np.asarray(lengths).sum() != len(X):
            raise ValueError("Sum of lengths must equal number of samples")
        
        return lengths

    def _enumerate_face_configs(self):
        """
        枚举所有可能的面部配置。返回长为 n_face_states 的列表，每个元素是一个长度为 n_actors 的二进制元组，形如 (0,1,1) 每个位置表示对应演员的人脸是否存在
        """
        face_configs = []
        for i in range(self.n_face_states):
            config = []
            for j in range(self.n_actors):
                config.append((i >> j) & 1)
            face_configs.append(tuple(config))
        return face_configs

    def _init_params(self):
        """初始化嵌套HMM的参数"""
        random_state = check_random_state(self.random_state)
        
        if 'a' in self.init_params:
            # α: 对于每个 actor，其面部出现的初始概率，不要求和为1
            self.alpha_ = random_state.uniform(0.3, 0.7, self.n_actors)
            
        if 'b' in self.init_params:
            # A_F: 面部状态转移矩阵 (n_actors, 2, 2), 每行和为1
            self.A_F_ = np.zeros((self.n_actors, 2, 2))
            for actor in range(self.n_actors):
                for s in range(2):
                    self.A_F_[actor, s] = random_state.dirichlet([2, 1] if s == 0 else [1, 2])

        if 'c' in self.init_params:
            # β: 说话人初始概率的logits,不要求和为1
            self.beta_ = random_state.normal(0, 1, self.n_actors)
            self.beta_ -= self.beta_[0]  # 固定第一个演员的logit为0，作为基准
            
        if 'd' in self.init_params:
            # γ₁: 面部对说话人初始状态的影响
            self.gamma1_ = random_state.uniform(0.5, 2.0)
            
        if 'e' in self.init_params:
            # A_S: 说话人状态转移矩阵的logits (n_actors, n_actors),不要求和为1
            diag_main = np.diag(random_state.uniform(0.3, 0.7, self.n_actors))
            self.A_S_ = diag_main + (1-diag_main) * random_state.normal(0, 1, (self.n_actors, self.n_actors))
            self.A_S_ -= np.diag(self.A_S_)[:,None]    # 固定转移到自己的logit为0，作为基准
            
        if 'f' in self.init_params:
            # γ₂: 面部对说话人转移的影响
            self.gamma2_ = random_state.uniform(0.5, 2.0)
            
        if 'g' in self.init_params:
            # B_F: 面部识别混淆矩阵 (n_actors, 2, 2), 每行和为1
            self.B_F_ = np.zeros((self.n_actors, 2, 2))
            for actor in range(self.n_actors):
                for s in range(2):
                    self.B_F_[actor, s] = random_state.dirichlet([2, 1] if s == 0 else [1, 2])

        if 'h' in self.init_params:
            # B_S: 说话人识别混淆矩阵 (n_actors, n_actors), 每行和为1
            self.B_S_ = np.zeros((self.n_actors, self.n_actors))
            for actor in range(self.n_actors):
                self.B_S_[actor] = random_state.dirichlet([2 if i == actor else 1 for i in range(self.n_actors)])

        if 'i' in self.init_params:
            # η1: 协变量X取值为1对说话人初始状态的影响
            self.eta1_ = random_state.uniform(1, 3)

        if 'j' in self.init_params:
            # η2: 协变量X取值为1对说话人转移的影响
            self.eta2_ = random_state.uniform(1, 3)

    def X2index(self, x_onehot):
        """
        将协变量的one-hot编码转换为索引
        - x_onehot: 形状 (n_actors,) 的0-1数组
        - return: 索引，范围 [0, n_actors]，其中 n_actors 表示全零配置
        """
        if np.isclose(x_onehot.sum(), 1):
            return np.argmax(x_onehot)
        elif np.isclose(x_onehot.sum(), 0):
            return self.n_actors  # 全零配置
        else:
            raise ValueError("x_onehot must be one-hot encoded or all zeros")

    def candidate_sets2idxs(self, potential_states_lists, var_type='speaker'):
        """
        获取当前segment中各个audio/face sample的候选状态集合的所有可能组合，并得到它们在binary array中的row indices
        - potential_states_lists: 长度为 n_samples 的列表，每个元素是一个列表，记载该样本的候选状态索引，e.g. [[0,1], [0,2], [1,2]]
        - var_type: 'speaker' 或 'face'
        - return: list of indices, e.g. [3, 5, 6, 7]
        """
        n_samples = len(potential_states_lists)
        if var_type == 'speaker':
            assert n_samples == 1, "For speaker type, potential_states_lists should contain only one list"
            potential_states_idxs = sorted(potential_states_lists[0])
        elif var_type == 'face':
            if n_samples == 0:
                potential_states_idxs = [0]  # 如果没有face sample，返回全零配置的索引0
            else:
                # 1. 获取所有可能的状态组合（笛卡尔积）
                if n_samples > 12:
                    # 避免笛卡尔积爆炸，限制组合数量
                    unique_potential_labels = list(set([item for sublist in potential_states_lists for item in sublist]))
                    k = min(len(unique_potential_labels), self.n_actors)
                    all_combinations = [list(comb) for size in range(1, k+1) for comb in itertools.combinations(unique_potential_labels, size)]
                    # print(f"k ={k}, total combinations to consider: {len(all_combinations)}")
                else:
                    all_combinations = itertools.product(*potential_states_lists)
                # 2. 对每个组合下涵盖的元素去重（set），排序(打乱了sample顺序)，得到可做set比较的tuple
                all_combinations_processed = [tuple(sorted(set(comb))) for comb in all_combinations]
                # 3. 在组合层面去重
                all_combinations_unique = list(set(all_combinations_processed))
                # 4. 将每个组合转换为self.face_configs_arr中的row index，并排序返回
                potential_states_idxs = sorted(list(map(lambda comb: sum(2**i for i in comb), all_combinations_unique)))

        else:
            raise ValueError("var_type must be 'speaker' or 'face'")
        return potential_states_idxs

    def process_potential_list(self, F_potential_list, var_type='speaker'):
        F_potential_states_idxs = []
        # for loop 在生活大爆炸上仅需2s，为了避免OOM，暂时无需改成list(map())
        for i in range(len(F_potential_list)):
            F_potential_states_idxs.append(self.candidate_sets2idxs(F_potential_list[i], var_type))
        return F_potential_states_idxs

    def fit(self, S_hat_onehot, F_hat, X_onehot, F_potential_list, B_S_diag_min=None, B_F_diag_min=None, lengths=None):
        """训练嵌套HMM模型"""
        S_hat_onehot = np.array(S_hat_onehot)
        F_hat = np.array(F_hat)
        X_onehot = np.array(X_onehot)
        F_potential_states_idxs = self.process_potential_list(F_potential_list, var_type='face')   # of length n_samples, each element is a list of possible face state indices

        self._check_and_set_n_features(S_hat_onehot, F_hat, X_onehot,F_potential_states_idxs)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        
        # 初始化参数
        self._init_params()
        # 重置收敛监控器
        self.monitor_._reset()
        
        # EM算法主循环
        for n_iter in range(self.n_iter):
            # E步：计算前向后向概率和期望统计量
            start_time = time.time()
            stats = self._do_estep(S_hat_onehot, F_hat, X_onehot, F_potential_states_idxs, lengths)
            estep_time = time.time() - start_time

            # 检查收敛
            curr_loglik = stats['log_likelihood'] # 计算当前对数似然
            self.monitor_.report(curr_loglik)
            if self.monitor_.converged:
                break

            # M步：更新参数
            start_time = time.time()
            self._do_mstep(stats, B_S_diag_min, B_F_diag_min, lengths)
            mstep_time = time.time() - start_time

            print(f"E步耗时: {estep_time:.4f}秒")
            print(f"M步耗时: {mstep_time:.4f}秒")

        return self

    def _do_estep(self, S_hat_onehot, F_hat, X_onehot, F_potential_states_idxs, lengths):
        """E步：使用前向-后向算法计算期望统计量，同时获取数据总体的log-likelihood"""
        
        stats = self._initialize_sufficient_statistics()
        log_likelihood = 0.0
        
        forward_time = 0.0
        backward_time = 0.0
        accumulate_time = 0.0
        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            # 获取当前序列段
            seq_S_hat_onehot = S_hat_onehot[start_idx:end_idx]
            seq_F_hat = F_hat[start_idx:end_idx]
            seq_X_onehot = X_onehot[start_idx:end_idx]
            seq_F_potential_idxs = F_potential_states_idxs[start_idx:end_idx]
            
            # 前向算法
            start_time = time.time()
            fwd_lattice = self._do_forward_pass(seq_S_hat_onehot, seq_F_hat, seq_X_onehot, seq_F_potential_idxs)
            forward_time += time.time() - start_time
            
            # 后向算法
            start_time = time.time()
            bwd_lattice = self._do_backward_pass(seq_S_hat_onehot, seq_F_hat, seq_X_onehot, seq_F_potential_idxs)
            backward_time += time.time() - start_time
            
            # 计算观测序列段的对数似然 $\bbP(\cI_i^{obs}\vert\btheta^{(s)})$
            # fwd_lattice 在dim=1上稀疏，仅计算潜在状态集合涉及的状态即可
            seq_loglik = logsumexp(fwd_lattice[-1, seq_F_potential_idxs[-1], :])
            log_likelihood += seq_loglik
            
            # 更新累积统计量，实现对 i=1,...,m 的求和
            start_time = time.time()
            stats_updated = self._accumulate_sufficient_statistics(
                stats, seq_S_hat_onehot, seq_F_hat, seq_X_onehot, seq_F_potential_idxs, fwd_lattice, bwd_lattice, seq_loglik)
            accumulate_time += time.time() - start_time
            stats = stats_updated

            start_idx = end_idx
            
        stats['log_likelihood'] = log_likelihood
        
        print(f"前向算法总时间: {forward_time:.4f}秒")
        print(f"后向算法总时间: {backward_time:.4f}秒")
        print(f"累积统计量更新总时间: {accumulate_time:.4f}秒")
        
        return stats

    def _do_forward_pass(self, S_hat_onehot, F_hat, X_onehot, seq_F_potential_idxs):
        """
        前向算法计算前向概率的对数
            - S_hat_onehot: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - F_hat: 面部出现，形状 (n_samples, n_actors)，二进制数据
            - X_onehot: 协变量，形状 (n_samples, n_features)，one-hot编码
            - seq_F_potential_idxs: 面部观测的候选状态集合索引列表，长度为 n_samples，每个元素是一个列表，记载该样本的所有候选状态索引
            - return: fwd_lattice, 形状 (n_samples, n_face_states, n_actors), (t, f_idx, s) 表示时刻t面部配置为f_idx，说话人为s的对数概率 $ \log (\bbU_{i,t}(f,\varrho))$
        """
        n_samples = len(S_hat_onehot)
        
        # fwd_lattice[t, f, s] = log P(观测到t时刻, 面部配置f, 说话人s)
        fwd_lattice = np.full((n_samples, self.n_face_states, self.n_actors), -np.inf)
        
        # 初始时刻
        ## 计算初始时刻所有面部配置发生的概率
        F_idxs_init = seq_F_potential_idxs[0]
        log_face_probs = self._compute_face_initial_probs()  # shape (n_face_states,)
        log_face_probs_filtered = log_face_probs[F_idxs_init]  # shape (n_face_states_potential,)
        log_face_probs_filtered = log_face_probs_filtered - logsumexp(log_face_probs_filtered)  # 归一化
        ## 计算初始时刻所有说话人发生的概率
        active_x = self.X2index(X_onehot[0])
        log_speaker_probs_filtered = self._compute_speaker_initial_probs(F_idxs_init, active_x)  # shape (n_face_states, n_actors)
        ## 计算初始时刻所有可能隐藏状态对应的观测概率
        log_face_emissions_filtered, log_speaker_emissions = self._compute_emission_probs(F_hat[0], S_hat_onehot[0], F_idxs_init)
        fwd_lattice[0, F_idxs_init, :] = log_face_probs_filtered[:, None] + log_speaker_probs_filtered[:, :] + log_face_emissions_filtered[:, None] + log_speaker_emissions[None, :]

        # 递推
        for t in range(1, n_samples):
            F_idxs_prev = seq_F_potential_idxs[t-1]
            F_idxs_curr = seq_F_potential_idxs[t]
            prev_fwd_lattice_filtered = fwd_lattice[t-1, F_idxs_prev, :] # shape (n_face_states_potential_prev, n_actors), corresponds to prev_face_config and prev_speaker

            ## 计算 f 所有可能的转移组合的转移概率
            log_trans_face_filtered = self._compute_face_transition_probs(F_idxs_prev, F_idxs_curr)
            log_trans_face_filtered = log_trans_face_filtered - logsumexp(log_trans_face_filtered, axis=1)[:, None]  # 归一化
            ## 计算在可能的f下，各种说话人转移情况的概率
            active_x = self.X2index(X_onehot[t])
            log_trans_speaker_filtered = self._compute_speaker_transition_probs(F_idxs_curr, active_x)  # shape (n_actors_prev, n_actors_curr, n_face_states_potential)
            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_face_emissions_filtered, log_speaker_emissions = self._compute_emission_probs(F_hat[t], S_hat_onehot[t], F_idxs_curr)

            ## 计算当前时刻所有可能的 (f, \varrho)对应的概率log_probs_arr (f_prev, s_prev, f_curr, s_curr)
            log_probs_arr_filtered = (prev_fwd_lattice_filtered[:, :, None, None] +
                                      log_trans_face_filtered[:, None, :, None] + 
                                      np.transpose(log_trans_speaker_filtered, (0, 2, 1))[None, :, :, :] + 
                                      log_face_emissions_filtered[None, None, :, None] + log_speaker_emissions[None, None, None, :])

            ## 对上一时刻的 (f', \varrho') 求和，更新前向概率
            ### 对于不在F_idxs_prev中的prev_f，prev_fwd_lattice中对应行全为 -np.inf，因此不考虑也不会影响结果
            fwd_lattice[t, F_idxs_curr, :] = logsumexp(log_probs_arr_filtered, axis=(0,1))
        
        return fwd_lattice

    def _do_backward_pass(self, S_hat_onehot, F_hat, X_onehot, seq_F_potential_idxs):
        """
        后向算法计算后向概率的对数
            - S_hat_onehot: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - F_hat: 面部出现，形状 (n_samples, n_actors)，二进制数据
            - X_onehot: 协变量，形状 (n_samples, n_features)，one-hot编码
            - seq_F_potential_idxs: 面部观测的候选状态集合索引列表，长度为 n_samples，每个元素是一个列表，记载该样本的所有候选状态索引
            - return: bwd_lattice, 形状 (n_samples, n_face_states, n_actors), (t, f_idx, s) 表示时刻t面部配置为f_idx，说话人为s的对数概率 $ \log (\bbV_{i,t}(f,\varrho))$
        """
        n_samples = len(S_hat_onehot)
        
        # bwd_lattice[t, f, s] = log P(t+1时刻之后的观测 | t时刻面部配置f, 说话人s)
        bwd_lattice = np.full((n_samples, self.n_face_states, self.n_actors), -np.inf)
        
        # 终止时刻
        bwd_lattice[-1, :, :] = 0.0
        
        # 反向递推
        for t in range(n_samples - 2, -1, -1):
            F_idxs_next = seq_F_potential_idxs[t+1]
            F_idxs_curr = seq_F_potential_idxs[t]
            next_bwd_lattice_filtered = bwd_lattice[t+1, F_idxs_next, :] # shape (n_face_states_potential_next, n_actors), corresponds to next_face_config and next_speaker

            ## 计算 f 所有可能的转移组合的转移概率
            log_trans_face_filtered = self._compute_face_transition_probs(F_idxs_curr, F_idxs_next)
            log_trans_face_filtered = log_trans_face_filtered - logsumexp(log_trans_face_filtered, axis=1)[:, None]  # 归一化
            ## 计算在可能的f下，各种说话人转移情况的概率
            active_x = self.X2index(X_onehot[t+1])
            log_trans_speaker_filtered = self._compute_speaker_transition_probs(F_idxs_next, active_x)  # shape (n_actors_curr, n_actors_next, n_face_states_potential)
            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_face_emissions_filtered, log_speaker_emissions = self._compute_emission_probs(F_hat[t+1], S_hat_onehot[t+1], F_idxs_next)

            ## 计算当前时刻所有可能的 (f, \varrho)对应的概率log_probs_arr (f_curr, s_curr, f_next, s_next)
            log_probs_arr_filtered = (next_bwd_lattice_filtered[None, None, :, :] +
                                      log_trans_face_filtered[:, None, :, None] +
                                      np.transpose(log_trans_speaker_filtered, (0, 2, 1))[None, :, :, :] +
                                      log_face_emissions_filtered[None, None, :, None] + log_speaker_emissions[None, None, None, :])
            ## 对下一时刻的 (f', \varrho') 求和，更新后向概率
            bwd_lattice[t, F_idxs_curr, :] = logsumexp(log_probs_arr_filtered, axis=(2,3))
        
        return bwd_lattice

    def _compute_face_initial_probs(self):
        """
        计算所有面部配置的初始概率 $\bbP(F_{i,1,\cdot}=f)$ 的对数。
        - return: log_probs of shape (n_face_states,)
        """
        log_probs_factors = np.log(self.alpha_)[None, :] * self.face_configs_arr + np.log(1 - self.alpha_)[None, :] * (1 - self.face_configs_arr)  # shape: (n_face_states, n_actors)
        log_probs = log_probs_factors.sum(axis=1)  # shape: (n_face_states,)
        return log_probs

    def _compute_face_transition_probs(self, F_idxs_potential_prev=None, F_idxs_potential_curr=None):
        """
        计算面部配置的转移概率 $\prod_{\varrho\in\cP} \bbP(F_{i,t,\varrho}\vert F_{i,t-1,\varrho})$ 的对数
        """
        if F_idxs_potential_prev is None:
            F_idxs_potential_prev = np.arange(self.n_face_states)
        if F_idxs_potential_curr is None:
            F_idxs_potential_curr = np.arange(self.n_face_states)
        face_configs_arr_prev = self.face_configs_arr[F_idxs_potential_prev]  # shape: (n_face_states_prev, n_actors)
        face_configs_arr_curr = self.face_configs_arr[F_idxs_potential_curr]  # shape: (n_face_states_curr, n_actors)
        
        probs_factors = self.A_F_[np.arange(self.n_actors)[:, None, None], face_configs_arr_prev.T[:, :, None], face_configs_arr_curr.T[:, None, :]]  # shape: (n_actors, n_face_states_prev, n_face_states_curr)
        log_probs = np.log(probs_factors).sum(axis=0)  # shape: (n_face_states_prev,n_face_states_curr), 每个元素是从prev_config转移到curr_config的对数概率
        return log_probs

    def _compute_speaker_initial_probs(self, F_idxs_potential, active_x):
        """
        计算在已知active speaker face时，给定某些人脸出现情况，所有说话人的初始概率 $\bbP(S_{i,1}=\cdot \vert F_{i,1,\cdot}=f, X_{i,1,\cdot}=x)$ 的对数
        """
        logits = self.beta_[None, :] + self.gamma1_ * self.face_configs_arr[F_idxs_potential] + self.eta1_ * self.X_arr[active_x][None, :]  # shape: (n_face_states_potential, n_actors)
        log_probs = logits - logsumexp(logits, axis=1)[:, None]  # of shape (n_face_states_potential, n_actors), 每个元素代表给定人脸出现状态下，说话人为s的log概率
        return log_probs

    def _compute_speaker_transition_probs(self, F_idxs_potential, active_x):
        """
        计算在已知active speaker face时，给定某些人脸出现情况，从任意说话人转移到任意说话人的概率 $\bbP(S_{i,t+1}=\cdot \vert S_{i,t}=\varrho',F_{i,t+1,\cdot}=f, X_{i,t+1,\cdot}=x)$ 的对数
        """
        logits = (self.A_S_[:, :, None] + self.gamma2_ * self.face_configs_arr[F_idxs_potential].T[None, :, :] +
                  self.eta2_ * self.X_arr[active_x][None, :, None])    # shape: (n_actors_prev, n_actors_curr, n_face_states_potential)
        log_probs = logits - logsumexp(logits, axis=1)[:, None, :]  # 每个元素代表给定人脸出现状态和协变量状态下，说话人为s的log概率
        return log_probs

    def _compute_emission_probs(self, f_hat, s_hat, F_idxs_potential):
        """
        计算所有潜在隐藏状态组合对应的对数发射概率$\bB_S(S_{i,t},\hat S_{i,t}) \prod_{\varrho\in\cP} \bB_{\varrho}(F_{i,t,\varrho},\hat F_{i,t,\varrho})$
        """
        assert s_hat.shape[0] == self.n_actors
        # 对数面部观测概率
        face_emissions_B_F_ = self.B_F_[np.arange(self.n_actors)[None, :], self.face_configs_arr[F_idxs_potential], f_hat[None, :]]  # shape (n_face_states_potential, n_actors)
        log_face_emissions = np.log(face_emissions_B_F_).sum(axis=1)  # shape (n_face_states_potential,), each element corresponds to a current face_config

        # 对数说话人观测概率 $\bB_S(S_{i,t},\hat S_{i,t})$
        speaker_obs = np.argmax(s_hat)  # one-hot to index
        log_speaker_emissions = np.log(self.B_S_[:, speaker_obs])  # shape (n_actors,), each element corresponds to a current speaker
        return log_face_emissions, log_speaker_emissions

    def _initialize_sufficient_statistics(self):
        """
        初始化充分统计量，也即 M 步用到的期望值
        """
        return {
            'face_initial_counts': {}, # key is unique F_idxs_init, value is a np.array of shape (len(F_idxs_init), )
            'face_transition_counts': {},  # key is unique F_idxs_curr, value is a list contains two elements. The first element is collected F_idxs_prev list for this F_idxs_curr, value is a np.array of shape (len(F_idxs_prev), len(F_idxs_curr))
            'speaker_initial_counts': np.zeros((self.n_face_states, self.n_actors, self.n_actors + 1)),    # [f_init, s_init, x_onehot_init]
            'speaker_transition_counts': np.zeros((self.n_face_states, self.n_actors, self.n_actors, self.n_actors + 1)),  # [f_curr, s_prev, s_curr, x_onehot_curr]
            'face_emission_counts': np.zeros((self.n_actors, 2, 2)),    # [actor, face_state, observed_state]
            'speaker_emission_counts': np.zeros((self.n_actors, self.n_actors))  # [speaker_state, observed_speaker]
        }

    def _accumulate_sufficient_statistics(self, stats, S_hat_onehot, F_hat, X_onehot, seq_F_potential_idxs, fwd_lattice, bwd_lattice, seq_loglik):
        """
        更新累积充分统计量 stats，以便于后续执行参数更新
        """
        n_samples = len(S_hat_onehot)
        stats_updated = copy.deepcopy(stats)
        
        # 计算后验概率
        for t in range(n_samples):
            F_idxs_prev = seq_F_potential_idxs[t-1] if t > 0 else None
            F_idxs_curr = seq_F_potential_idxs[t]
            F_idxs_curr_key = tuple(F_idxs_curr)    # have been sorted in candidate_sets2idxs()
            # 单时刻后验概率 gamma[t, f, s] = P(F_t=f, S_t=s | 全部观测, 全部协变量) for all f in F_idxs_curr
            log_gamma_filtered = fwd_lattice[t, F_idxs_curr] + bwd_lattice[t, F_idxs_curr] - seq_loglik
            gamma_filtered = np.exp(log_gamma_filtered)   # shape (n_face_states_potential, n_actors)
            gamma_faces_filtered = gamma_filtered.sum(axis=1)  # shape: (n_face_states_potential,)，提前对speaker求和，方便后续计算面部统计量

            # 将协变量从one-hot 转为 index
            active_x = self.X2index(X_onehot[t])        
            # 累积初始统计量
            if t == 0:
                # 计算人脸初始充分统计量 $\bbE\left[\bbN(F_{\cdot,1,\cdot}=1\vert Z_{\cdot,1,\cdot}, \btheta^{(s)})\right]$ 中属于第i个片段的部分
                if F_idxs_curr_key not in stats_updated['face_initial_counts']:
                    stats_updated['face_initial_counts'][F_idxs_curr_key] = np.zeros((len(F_idxs_curr)))  # value、F_idxs_curr_key 中每个元素与一一对应
                stats_updated['face_initial_counts'][F_idxs_curr_key] += gamma_faces_filtered
                # 计算说话人初始充分统计量 $\bbE\left[\bbN(F_{\cdot,1,\cdot}=f,S_{\cdot,1}=\varrho\vert \btheta^{(s)})\right] $
                stats_updated['speaker_initial_counts'][F_idxs_curr, :, active_x] += gamma_filtered # 未更新的行对应的条件概率都为0（因为U在行上稀疏）
            
            # 累积转移统计量
            if t > 0:
                ## 计算 f 所有可能的转移组合的转移概率
                log_trans_face_filtered = self._compute_face_transition_probs(F_idxs_prev, F_idxs_curr)
                log_trans_face_filtered = log_trans_face_filtered - logsumexp(log_trans_face_filtered, axis=1)[:, None]  # 归一化
                ## 计算在可能的f下，各种说话人转移情况的概率
                log_trans_speaker_filtered = self._compute_speaker_transition_probs(F_idxs_curr, active_x)  # shape (n_actors_prev, n_actors_curr, n_face_states_potential)
                ## 计算对数转移后验概率 xi[t-1, f_prev, s_prev, f_curr, s_curr]
                log_face_emissions_filtered, log_speaker_emissions = self._compute_emission_probs(F_hat[t], S_hat_onehot[t], F_idxs_curr)

                log_xi_arr_filtered = (fwd_lattice[t-1, F_idxs_prev, :][:, :, None, None] + log_trans_face_filtered[:, None, :, None] +
                              np.transpose(log_trans_speaker_filtered, (0, 2, 1))[None, :, :, :] +
                              log_face_emissions_filtered[None, None, :, None] + log_speaker_emissions[None, None, None, :] +
                              bwd_lattice[t, F_idxs_curr, :][None, None, :, :] - seq_loglik) 
                xi_arr_filtered = np.exp(log_xi_arr_filtered) # 求和式中的每一项

                ## 计算面部转移统计量 $\bbE\left[\bbN(F_{\cdot,\cdot-1,\varrho}=\delta,F_{\cdot,\cdot,\varrho}=\delta' \vert \btheta^{(s)})\right]$
                face_transition_weights_filtered = xi_arr_filtered.sum(axis=(1, 3))  # shape: (n_face_states_potential_prev, n_face_states_potential_curr)
                if F_idxs_curr_key not in stats_updated['face_transition_counts']:
                    stats_updated['face_transition_counts'][F_idxs_curr_key] = [F_idxs_prev, np.zeros((len(F_idxs_prev), len(F_idxs_curr)))]
                ### 合并已有的转移统计量
                F_idxs_prev_old, face_transition_weights_filtered_old = stats_updated['face_transition_counts'][F_idxs_curr_key]
                F_idxs_prev_new = list(sorted(set(F_idxs_prev_old).union(set(F_idxs_prev))))
                face_transition_weights_filtered_new = np.zeros((len(F_idxs_prev_new), len(F_idxs_curr)))
                face_transition_weights_filtered_new[[F_idxs_prev_new.index(k) for k in F_idxs_prev_old], :] += face_transition_weights_filtered_old
                face_transition_weights_filtered_new[[F_idxs_prev_new.index(k) for k in F_idxs_prev], :] += face_transition_weights_filtered
                ### 更新字典
                stats_updated['face_transition_counts'][F_idxs_curr_key] = [F_idxs_prev_new, face_transition_weights_filtered_new]

                ## 存储用于说话人转移概率优化的信息，[f_curr, s_prev, s_curr]
                stats_updated['speaker_transition_counts'][F_idxs_curr, :, :, active_x] += np.transpose(xi_arr_filtered.sum(axis=0), (1, 0, 2))  # sum over prev_f_idx。# 未更新的行对应的条件概率都为0（因为F的转移概率稀疏）

            # 累积发射统计量
            ## 说话人发射统计量
            speaker_obs = np.argmax(S_hat_onehot[t]) # 说话人期望计算式中的 $\varrho'$
            stats_updated['speaker_emission_counts'][:, speaker_obs] += gamma_filtered.sum(axis=0)

            ## 面部发射统计量
            for actor in range(self.n_actors):  # 人脸期望式中的 $\varrho$
                for face_state in [0, 1]:  # 人脸期望式中的 $\delta$
                    mask_filtered = (self.face_configs_arr[F_idxs_curr, actor] == face_state)
                    stats_updated['face_emission_counts'][actor, face_state, F_hat[t, actor]] += gamma_faces_filtered[mask_filtered].sum()

        return stats_updated

    def _do_mstep(self, stats, B_S_diag_min, B_F_diag_min, lengths):
        """M步：更新参数"""
        # 更新面部初始概率
        if 'a' in self.params:
            start_time = time.time()
            self._update_face_initial_params(stats)
            self.alpha_ = np.clip(self.alpha_, 1e-6, 1-1e-6)
            print(f"面部初始参数更新耗时: {time.time() - start_time:.4f}秒")
        
        # 更新面部转移矩阵
        if 'b' in self.params:
            start_time = time.time()
            self._update_face_transition_params(stats)
            self.A_F_ = np.clip(self.A_F_, 1e-6, 1-1e-6)
            self.A_F_ /= self.A_F_.sum(axis=2, keepdims=True)  # row normalization
            print(f"面部转移参数更新耗时: {time.time() - start_time:.4f}秒")
        
        # 更新说话人初始概率参数 (beta, gamma1, eta1)
        if 'c' in self.params or 'd' in self.params or 'i' in self.params:
            start_time = time.time()
            self._update_speaker_initial_params(stats)
            print(f"说话人初始参数更新耗时: {time.time() - start_time:.4f}秒")
        
        # 更新说话人转移概率参数 (A_S, gamma2, eta2)
        if 'e' in self.params or 'f' in self.params or 'j' in self.params:
            start_time = time.time()
            self._update_speaker_transition_params(stats)
            print(f"说话人转移参数更新耗时: {time.time() - start_time:.4f}秒")
        
        # 更新面部发射矩阵
        if 'g' in self.params:
            for actor in range(self.n_actors):  # (14)式中的 $\varrho$
                for state in range(2):  # (14)式中的 $\delta$
                    total = stats['face_emission_counts'][actor, state].sum()
                    if total > 0:
                        self.B_F_[actor, state] = stats['face_emission_counts'][actor, state] / total # row normalization
                    else:
                        self.B_F_[actor, state] = np.ones(2) / 2
                        print(f"Warning: Emission probabilities for actor {actor}, state {state} were not updated due to insufficient data. Reset to uniform distribution.")
                    if B_F_diag_min is not None and self.B_F_[actor, state, state] < B_F_diag_min:
                        self.B_F_[actor, state, state] = B_F_diag_min
                        self.B_F_[actor, state, 1 - state] = 1 - B_F_diag_min
                    self.B_F_[actor, state] = np.clip(self.B_F_[actor, state], 1e-6, 1-1e-6)
                    self.B_F_[actor, state] /= self.B_F_[actor, state].sum()
        
        # 更新说话人发射矩阵  
        if 'h' in self.params:
            for speaker in range(self.n_actors):  # (15)式中的 $\varrho$
                total = stats['speaker_emission_counts'][speaker].sum()
                if total > 0:
                    self.B_S_[speaker] = stats['speaker_emission_counts'][speaker] / total  # row normalization
                else:
                    self.B_S_[speaker] = np.ones(self.n_actors) / self.n_actors
                    print(f"Warning: Emission probabilities for speaker {speaker} were not updated due to insufficient data. Reset to uniform distribution.")
                if B_S_diag_min is not None and self.B_S_[speaker, speaker] < B_S_diag_min:
                    temp_B_S_speaker = copy.deepcopy(self.B_S_[speaker])
                    self.B_S_[speaker, speaker] = B_S_diag_min
                    for i in range(self.n_actors):
                        if i != speaker:
                            self.B_S_[speaker, i] = (1-B_S_diag_min) / (1 - temp_B_S_speaker[speaker]) * temp_B_S_speaker[i]
                self.B_S_[speaker] = np.clip(self.B_S_[speaker], 1e-6, 1-1e-6)
                self.B_S_[speaker] /= self.B_S_[speaker].sum()


    def _update_face_initial_params(self, stats):
        """使用数值优化更新面部初始参数"""
        def face_initial_probs_weighted_sum(alphas, F_idxs_curr_key, expectation_weight):
            """
            计算潜在面部配置的初始概率 $\bbP(F_{i,1,\cdot}=f)$ 的对数的加权平均值。
            - return: log_probs_weighted_sum = \sum_f \log \bbP(F_{i,1,\cdot}=f) * weight(f)
            """
            F_idxs_curr = list(F_idxs_curr_key)
            alphas = np.clip(alphas, 1e-8, 1-1e-8)
            log_probs_factors = np.log(alphas)[None, :] * self.face_configs_arr[F_idxs_curr] + np.log(1 - alphas)[None, :] * (1 - self.face_configs_arr[F_idxs_curr])  # shape: (n_face_states_potential, n_actors)
            log_probs_unnormed = log_probs_factors.sum(axis=1)  # shape: (n_face_states_potential,)
            log_probs = log_probs_unnormed - logsumexp(log_probs_unnormed)  # 归一化
            log_probs_weighted_sum = np.sum(log_probs * expectation_weight) # 内层求和
            return log_probs_weighted_sum

        def objective_face_initial(params):
            loss = - sum(face_initial_probs_weighted_sum(params, k, v) for k, v in stats['face_initial_counts'].items())
            return loss
        
        # 初始参数
        x0 = self.alpha_     
        # 优化
        result = minimize(objective_face_initial, x0, method='L-BFGS-B', bounds=[(1e-6, 1-1e-6)]*self.n_actors)
        obj_init = objective_face_initial(x0)
        obj_final = objective_face_initial(result.x)
        
        if result.success or obj_final < obj_init:
            self.alpha_ = result.x
            if not result.success:
                print("Warning: Face initial parameters optimization did not fully converge, but objective improved.")
        else:
            print("Warning: face initial parameters optimization did not converge.")
        print(f"Initial objective value for face initial params: {obj_init:.4f}")
        print(f"Final objective value for face initial params: {obj_final:.4f}")

    def _update_face_transition_params(self, stats):
        """使用数值优化更新面部转移参数"""
        def face_transition_probs_weighted_sum(A_F_diags_flatten, F_idxs_curr_key, prevf_and_weights):
            """
            计算潜在面部配置的转移概率 $\bbP(F_{i,t,\cdot}\vert F_{i,t-1,\cdot}, <context>)$ 的对数的加权平均值。
            - return: log_probs_weighted_sum = \sum_{f_prev,f_curr} \log \bbP(F_{i,t,\cdot}=f_curr \vert F_{i,t-1,\cdot}=f_prev) * weight(f_prev, f_curr)
            """
            A_F_diags = A_F_diags_flatten.reshape((self.n_actors, 2))  # shape: (n_actors, 2)
            A_F_diags = np.clip(A_F_diags, 1e-8, 1-1e-8)
            A_F_ = np.zeros((self.n_actors, 2, 2))
            A_F_[:, 0, 0] = A_F_diags[:, 0]
            A_F_[:, 0, 1] = 1 -  A_F_[:, 0, 0]
            A_F_[:, 1, 1] = A_F_diags[:, 1]
            A_F_[:, 1, 0] = 1 -  A_F_[:, 1, 1]
            
            F_idxs_curr = list(F_idxs_curr_key)
            F_idxs_prev, expectation_weight = prevf_and_weights
            face_configs_arr_prev = self.face_configs_arr[F_idxs_prev]  # shape: (n_face_states_prev, n_actors)
            face_configs_arr_curr = self.face_configs_arr[F_idxs_curr]  # shape: (n_face_states_curr, n_actors)
            
            probs_factors = A_F_[np.arange(self.n_actors)[:, None, None], face_configs_arr_prev.T[:, :, None], face_configs_arr_curr.T[:, None, :]]  # shape: (n_actors, n_face_states_prev, n_face_states_curr)
            log_probs_unnormed = np.log(probs_factors).sum(axis=0)  # shape: (n_face_states_prev,n_face_states_curr)
            log_probs = log_probs_unnormed - logsumexp(log_probs_unnormed, axis=1)[:, None]  # 归一化, 每个元素是从prev_config转移到curr_config的对数概率
            log_probs_weighted_sum = np.sum(log_probs * expectation_weight) # 内层求和
            return log_probs_weighted_sum

        def objective_face_transition(params):
            loss = - sum(face_transition_probs_weighted_sum(params, k, v) for k, v in stats['face_transition_counts'].items())
            return loss
        
        # 初始参数
        x0 =  np.diagonal(self.A_F_, axis1=1, axis2=2).flatten()  # shape: (n_actors,2). diagonal elements in A_F_\rho
        # 优化
        result = minimize(objective_face_transition, x0, method='L-BFGS-B',
                        bounds=[(1e-6, 1-1e-6)]*self.n_actors*2,
                        options={'maxiter': 10, 'ftol': 1e-3})
        obj_init = objective_face_transition(x0)
        obj_final = objective_face_transition(result.x)
        
        if result.success or obj_final < obj_init:
            self.A_F_ = np.zeros((self.n_actors, 2, 2))
            self.A_F_[:, 0, 0] = result.x.reshape((self.n_actors, 2))[:, 0]
            self.A_F_[:, 0, 1] = 1 -  self.A_F_[:, 0, 0]
            self.A_F_[:, 1, 1] = result.x.reshape((self.n_actors, 2))[:, 1]
            self.A_F_[:, 1, 0] = 1 -  self.A_F_[:, 1, 1]
            if not result.success:
                print("Warning: Face transition parameters optimization did not fully converge, but objective improved.")
        else:
            print("Warning: face transition parameters optimization did not converge.")
        print(f"Initial objective value for face transition params: {obj_init:.4f}")
        print(f"Final objective value for face transition params: {obj_final:.4f}")


    def _update_speaker_initial_params(self, stats):
        """使用数值优化更新说话人初始参数"""
        def objective_speaker_initial(params):
            beta, gamma1, eta1 = np.concatenate(([0.0], params[:-2])), params[-2], params[-1]
            weights = np.transpose(stats['speaker_initial_counts'], axes=(0, 2, 1))   # [f_init, x_onehot_init, s_init]
            masks = (weights > 0)
            logits = beta[None, None, :] + gamma1*self.face_configs_arr[:, None, :] + eta1*self.X_arr[None, :, :]
            log_probs = logits - logsumexp(logits, axis=2, keepdims=True)   # log-softmax
            loss = - np.sum(weights[masks] * log_probs[masks])
            
            return loss
        
        # 初始参数
        x0 = np.concatenate([self.beta_[1:], np.array([self.gamma1_, self.eta1_])])
        # print(f"Initial parameters for speaker initial params: {x0}")
        # print(sum(stats['speaker_initial_counts']>0))
        # print(stats['speaker_initial_counts'])

            
        # 优化
        result = minimize(objective_speaker_initial, x0, method='L-BFGS-B')
        obj_init = objective_speaker_initial(x0)
        obj_final = objective_speaker_initial(result.x)
        
        if result.success or obj_final < obj_init:
            self.beta_ = np.concatenate(([0.0], result.x[:-2]))
            self.gamma1_ = result.x[-2]
            self.eta1_ = result.x[-1]
        else:
            print("Warning: Speaker initial parameters optimization did not converge.")
        print(f"Initial objective value for speaker initial params: {obj_init:.4f}")
        print(f"Final objective value for speaker initial params: {obj_final:.4f}")

    def _update_speaker_transition_params(self, stats):
        """使用数值优化更新说话人转移参数(只优化非对角线元素和gamma2, eta2)"""
        mask_offdiag = ~np.eye(self.n_actors, dtype=bool)

        def objective_speaker_transition(params):
            # params: [A_S_offdiag, gamma2, eta2]
            A_S_mat = np.zeros((self.n_actors, self.n_actors))  # 对角线强制为0
            A_S_mat[mask_offdiag] = params[:-2]# 从flattend 参数重建A_S矩阵
            gamma2 = params[-2]
            eta2 = params[-1]
            
            weights = np.transpose(stats['speaker_transition_counts'], axes=(0, 3, 1, 2)) # [f_curr, x_onehot_curr, s_prev, s_curr]
            mask = (weights > 0)
            # [n_face_states, n_x_states, n_actors_prev, n_actors_curr(speaker/face)]
            logits = A_S_mat[None, None, :, :] + gamma2 * self.face_configs_arr[:, None, None, :] + eta2 * self.X_arr[None, :, None, :]   
            log_probs = logits - logsumexp(logits, axis=3, keepdims=True)
            loss = - np.sum(weights[mask] * log_probs[mask])
            
            return loss
        
        # 初始参数：只取A_S_非对角线元素和gamma2, eta2
        x0 = np.concatenate([self.A_S_[mask_offdiag], np.array([self.gamma2_, self.eta2_])])    # shape: (n_actors*(n_actors-1) + 2,)
        
        # 优化
        result = minimize(objective_speaker_transition, x0, method='L-BFGS-B',
                          options={'maxiter': 10, 'ftol': 1e-3})
        obj_init = objective_speaker_transition(x0)
        obj_final = objective_speaker_transition(result.x)

        if result.success or obj_final < obj_init:
            # 重建A_S_，对角线为0
            self.A_S_ = np.zeros((self.n_actors, self.n_actors))
            self.A_S_[mask_offdiag] = result.x[:-2]
            self.gamma2_ = result.x[-2]
            self.eta2_ = result.x[-1]
        else:
            print("Warning: Speaker transition parameters optimization did not converge.")
        print(f"Initial objective value for speaker transition params: {obj_init:.4f}")
        print(f"Final objective value for speaker transition params: {obj_final:.4f}")

    def score(self, S_hat_onehot, F_hat, X_onehot, F_potential_list, lengths=None):
        """计算观测序列的对数似然"""
        S_hat_onehot = np.array(S_hat_onehot)
        F_hat = np.array(F_hat)
        X_onehot = np.array(X_onehot)
        F_potential_states_idxs = self.process_potential_list(F_potential_list, var_type='face')   # of length n_samples, each element is a list of possible face state indices

        # EM算法总以M步结束，为了确保计算最新的对数似然，这里重新计算一次E步        
        return self._do_estep(S_hat_onehot, F_hat, X_onehot, F_potential_states_idxs, lengths)['log_likelihood']


    def predict_proba(self, S_hat_onehot, F_hat, X_onehot, F_potential_list, lengths=None):
        """
        计算给定观测序列时隐藏状态的联合后验概率 $\\bbP(F_{i,t,\cdot}=f,S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)})$，以及求和得到的边际后验 $\pi_{i,t,\varrho} = \\bbP(F_{i,t,\\rho}=1 \\vert \cI_i^{obs}, \\btheta^{(s)})$ , $\lambda_{i,t,\\varrho} = \\bbP(S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)})$
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        F_hat : array-like, shape (n_samples, n_actors)  
            面部出现观测，二进制数据
        X_onehot : array-like, shape (n_samples, n_actors)
            观测的X状态，one-hot编码        
        F_potential_list : list of list of int
            每个audio segment中，每个人脸，所有可能的标签
        lengths : array-like of integers, optional
            每个序列的长度
            
        Returns
        -------
        posteriors : dict
            包含各种后验概率的字典:
            - 'face_states': array, shape (n_samples, n_actors)
              每个时刻每个演员面部出现的后验概率  $ \pi_{i,t,\\varrho} $
            - 'speaker_states': array, shape (n_samples, n_actors)
              每个时刻每个演员是说话人的后验概率  $ \lambda_{i,t,\\varrho} $
            - 'joint_states': array, shape (n_samples, n_face_states, n_actors)
              每个时刻联合状态 (face_config, speaker) 的后验概率 $ \\bbP(F_{i,t,\\cdot}=f,S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)}) $
        """
        S_hat_onehot = np.array(S_hat_onehot)
        F_hat = np.array(F_hat)
        X_onehot = np.array(X_onehot)
        F_potential_states_idxs = self.process_potential_list(F_potential_list, var_type='face')   # of length n_samples, each element is a list of possible face state indices
        self._check_and_set_n_features(S_hat_onehot, F_hat, X_onehot, F_potential_states_idxs)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        n_samples = len(S_hat_onehot)

        # 初始化输出
        face_posteriors = np.zeros((n_samples, self.n_actors))
        speaker_posteriors = np.zeros((n_samples, self.n_actors))
        joint_posteriors = np.zeros((n_samples, self.n_face_states, self.n_actors))

        # 对每一集的数据        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            ## 获取当前序列段
            seq_S_hat_onehot = S_hat_onehot[start_idx:end_idx]
            seq_F_hat = F_hat[start_idx:end_idx]
            seq_X_onehot = X_onehot[start_idx:end_idx]
            seq_F_potential_idxs = F_potential_states_idxs[start_idx:end_idx]
            
            ## 计算前向和后向概率，以及序列的对数似然
            fwd_lattice = self._do_forward_pass(seq_S_hat_onehot, seq_F_hat, seq_X_onehot, seq_F_potential_idxs)
            bwd_lattice = self._do_backward_pass(seq_S_hat_onehot, seq_F_hat, seq_X_onehot, seq_F_potential_idxs)
            seq_loglik = logsumexp(fwd_lattice[-1, seq_F_potential_idxs[-1], :])
            
            ## 计算每个时刻的后验概率
            for t in range(length):
                F_idxs_curr = seq_F_potential_idxs[t]
                ### 获取并存储联合后验概率 P(F_t=f, S_t=s | 全部观测)
                log_gamma_filtered = fwd_lattice[t, F_idxs_curr] + bwd_lattice[t, F_idxs_curr] - seq_loglik
                gamma_filtered = np.exp(log_gamma_filtered)   # shape (n_face_states_potential, n_actors)
                joint_posteriors[start_idx + t, ] = gamma_filtered # of shape (n_face_states, n_actors)
                
                ### 计算面部状态的边际后验概率 P(F_{t, \\rho} =1 | 当前集全部观测)
                for actor in range(self.n_actors):
                    mask_filtered = (self.face_configs_arr[F_idxs_curr, actor] == 1)
                    face_posteriors[start_idx + t, actor] += gamma_filtered.sum(axis=1)[mask_filtered].sum()  # 先对说话人求和，再对符合要求的面部配置求和
                
                ### 计算说话人状态的边际后验概率 P(S_t=s | 当前集全部观测)
                speaker_posteriors[start_idx + t, :] = gamma_filtered.sum(axis=0)
            
            start_idx = end_idx
        
        return {
            'face_states': face_posteriors,
            'speaker_states': speaker_posteriors, 
            'joint_states': joint_posteriors
        }

    def predict(self, S_hat_onehot, F_hat, X_onehot, F_potential_list, lengths=None):
        """
        使用Viterbi算法，预测最可能的隐藏状态序列(面部状态和说话人状态)
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        F_hat : array-like, shape (n_samples, n_actors)
            面部出现观测，二进制数据
        X_onehot : array-like, shape (n_samples, n_actors)
            观测的X状态，one-hot编码
        F_potential_list : list of list of of list of int
            每个audio segment中，每个人脸，所有可能的标签
        lengths : array-like of integers, optional
            每个序列的长度，如果为None，则假设是单一序列

        Returns
        -------
        face_states : array, shape (n_samples, n_actors)
            预测的面部状态序列 (0或1)
        speaker_states : array, shape (n_samples,)
            预测的说话人状态序列 (0到n_actors-1)
        """
        S_hat_onehot = np.asarray(S_hat_onehot)
        F_hat = np.asarray(F_hat)
        X_onehot = np.asarray(X_onehot)
        F_potential_states_idxs = self.process_potential_list(F_potential_list, var_type='face')   # of length n_samples, each element is a list of possible face state indices
        self._check_and_set_n_features(S_hat_onehot, F_hat, X_onehot, F_potential_states_idxs)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        
        # 初始化输出数组
        face_states = np.zeros_like(F_hat, dtype=int)
        speaker_states = np.zeros(S_hat_onehot.shape[0], dtype=int)
        
        start_idx = 0
        for seq_len in lengths:
            end_idx = start_idx + seq_len
            
            # 提取当前序列
            seq_S_hat_onehot = S_hat_onehot[start_idx:end_idx]
            seq_F_hat = F_hat[start_idx:end_idx]
            seq_X_onehot = X_onehot[start_idx:end_idx]
            seq_F_potential_idxs = F_potential_states_idxs[start_idx:end_idx]

            # 使用维特比算法预测
            seq_face_states, seq_speaker_states = self._viterbi(seq_S_hat_onehot, seq_F_hat, seq_X_onehot, seq_F_potential_idxs)
            
            # 存储结果
            face_states[start_idx:end_idx] = seq_face_states
            speaker_states[start_idx:end_idx] = seq_speaker_states
            
            start_idx = end_idx
            
        return face_states, speaker_states

    def _viterbi(self, S_hat_onehot, F_hat, X_onehot, seq_F_potential_idxs):
        """
        对单个序列使用维特比算法进行解码
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_frames, n_actors)
            说话人观测序列
        F_hat : array-like, shape (n_frames, n_actors)
            面部出现序列
        X_onehot : array-like, shape (n_frames, n_actors)    
        观测的X状态序列
        seq_F_potential_idxs : list of list of int
            每个时间点，所有可能的面部状态组合的索引
        
        Returns
        -------
        face_states : array, shape (n_frames, n_actors)
            预测的面部状态序列
        speaker_states : array, shape (n_frames,)
            预测的说话人状态序列
        """
        n_frames = S_hat_onehot.shape[0]
        
        # 初始化维特比表格 $\delta_{t}(f,s)$ 与回溯路径 $\psi_{t}(f,s)$
        ## $\delta_{t}(f,s)$: 在时刻t，面部配置f，说话人s的最大概率的对数
        viterbi = np.full((n_frames, self.n_face_states, self.n_actors), -np.inf)
        ## $\psi_{t}(f,s)$: t 时刻面部配置f，说话人s时，从1到t的路径中，后验概率最大的路径在 $t-1$ 时刻的状态 (f', s')
        path_face = np.zeros((n_frames, self.n_face_states, self.n_actors), dtype=int)  # 每个元素是 f'在 face_configs 中的索引
        path_speaker = np.zeros((n_frames, self.n_face_states, self.n_actors), dtype=int) # 每个元素是s'的索引
        
        # 初始化: t=0时刻，已知隐状态与观测的联合概率的对数
        F_idxs_init = seq_F_potential_idxs[0]
        ## 计算对数初始面部隐藏状态概率
        log_face_probs = self._compute_face_initial_probs()  # shape (n_face_states,)
        log_face_probs_filtered = log_face_probs[F_idxs_init]  # shape (n_face_states_potential,)
        log_face_probs_filtered = log_face_probs_filtered - logsumexp(log_face_probs_filtered)  # 归一化
        ## 计算对数初始说话人隐藏状态概率
        active_x = self.X2index(X_onehot[0])    # one-hot to index
        log_speaker_probs_filtered = self._compute_speaker_initial_probs(F_idxs_init, active_x)  # shape (n_face_states, n_actors)
        ## 计算对数观测概率 P(F_hat | F)*P(S_hat | S)
        log_face_emissions_filtered, log_speaker_emissions = self._compute_emission_probs(F_hat[0], S_hat_onehot[0], F_idxs_init)
        viterbi[0, F_idxs_init, :] = log_face_probs_filtered[:, None] + log_speaker_probs_filtered[:, :] + log_face_emissions_filtered[:, None] + log_speaker_emissions[None, :]
        
        # 前向传播 t=1到n_frames-1
        for t in range(1, n_frames):
            F_idxs_prev = seq_F_potential_idxs[t-1]
            F_idxs_curr = seq_F_potential_idxs[t]
            ## 计算 f 所有可能的转移组合的转移概率
            log_trans_face_filtered = self._compute_face_transition_probs(F_idxs_prev, F_idxs_curr)
            log_trans_face_filtered = log_trans_face_filtered - logsumexp(log_trans_face_filtered, axis=1)[:, None]  # 归一化
            ## 计算在可能的f下，各种说话人转移情况的概率
            active_x = self.X2index(X_onehot[t])    # one-hot to index
            log_trans_speaker_filtered = self._compute_speaker_transition_probs(F_idxs_curr, active_x)  # shape (n_actors_prev, n_actors_curr, n_face_states_potential)
            ## 计算当前时刻每个隐藏状态对应的观测概率P(F_hat_t | F_t)*P(S_hat_t | S_t)
            log_face_emissions_filtered, log_speaker_emissions = self._compute_emission_probs(F_hat[t], S_hat_onehot[t], F_idxs_curr)

            # 计算每个当前状态对应的 $\delta_{t}(f,s)$
            for i, f_idx in enumerate(F_idxs_curr):
                for j, speaker in enumerate(range(self.n_actors)):
                    # 遍历所有可能的前一状态 (f_prev, s_prev)，确定最佳前一状态
                    total_prob_prev_no_obs = viterbi[t-1, :, :] 
                    total_prob_prev_no_obs[F_idxs_prev, :] += log_trans_face_filtered[:, i][:, None] + log_trans_speaker_filtered[:, j, i][None, :]
                    best_prev_flat = np.argmax(total_prob_prev_no_obs)
                    best_prev_f, best_prev_s = np.unravel_index(best_prev_flat, total_prob_prev_no_obs.shape)

                    # 确定 $\delta_{t}(f,s)$
                    viterbi[t, f_idx, speaker] = total_prob_prev_no_obs[best_prev_f, best_prev_s] + log_face_emissions_filtered[i] + log_speaker_emissions[speaker]
                    # 确定 $\psi_{t}(f,s)$
                    path_face[t, f_idx, speaker] = best_prev_f
                    path_speaker[t, f_idx, speaker] = best_prev_s
        
        # 找到最优路径的结束状态 $i_T^\ast$: best_end_f, best_end_s
        last_viterbi = viterbi[n_frames-1, :, :]  # shape (n_face_states, n_actors)
        best_end_flat = np.argmax(last_viterbi)
        best_end_f, best_end_s = np.unravel_index(best_end_flat, last_viterbi.shape)
        
        # 回溯最优路径
        face_states = np.zeros((n_frames, self.n_actors), dtype=int)
        speaker_states = np.zeros(n_frames, dtype=int)
        curr_f = best_end_f
        curr_s = best_end_s
        for t in range(n_frames-1, -1, -1):
            # 记录当前状态
            face_states[t, :] = self.face_configs_arr[curr_f]
            speaker_states[t] = curr_s
            
            # 回溯到前一状态
            if t > 0:
                prev_f = path_face[t, curr_f, curr_s]
                prev_s = path_speaker[t, curr_f, curr_s]
                curr_f = prev_f
                curr_s = prev_s
        
        return face_states, speaker_states
