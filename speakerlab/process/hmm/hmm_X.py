import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.utils import check_random_state
from functools import partial

from .monitor import ConvergenceMonitor
import time, copy


## 内存相关
### 1. 当处理实际电视剧数据时， U, V 矩阵尺寸较大，有可能导致内存不足。如果出现此类问题，可以先将每季的UV统计量存到本地，然后再读取进行累积


class HMM_X():
    """
    简化的隐马尔可夫模型，支持协变量 X 和/或 F_hat
    
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
    params : str, optional (default: "cehi")
        控制模型包含哪些参数
    random_state : int or RandomState, optional
        随机种子
    """
    
    def __init__(self, n_actors, n_iter=100, tol=1e-2, verbose=False,
                 params="ceh", random_state=100):
        self.n_actors = n_actors    # 演员数量
        self.n_iter = n_iter    # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印详细信息
        self.params = params    # 控制模型中包含哪些参数
        self.random_state = random_state

        # 创建监控器
        self.monitor_ = ConvergenceMonitor(tol, n_iter, verbose)
        
        # 确定协变量模式
        self.covariate_mode = self._determine_covariate_mode()

        # 添加缓存变量，避免反复计算
        if self.covariate_mode in ['X_only', 'both']:
            ## 所有可能的X配置
            self.X_arr = np.vstack([np.eye(self.n_actors), np.zeros((1, self.n_actors))])  # shape (n_actors+1, n_actors), 每行表示一个one-hot编码的协变量配置或全零配置
        if self.covariate_mode in ['F_only', 'both']:
            self.n_face_states = 2 ** self.n_actors  # 面部状态数量 (每个演员有2个状态)
            self.face_state_powers = np.array([2 ** i for i in range(self.n_actors)])# shape (n_actors,), 用于将二进制向量转换为索引
            self.face_configs_arr = np.array(self._enumerate_face_configs()) # shape (n_face_states, n_actors)
        

    def _determine_covariate_mode(self):
        """
        根据 params 确定协变量模式
        返回: 'X_only', 'F_only', 'both', 'none'
        """
        has_gamma1 = 'd' in self.params
        has_gamma2 = 'f' in self.params
        has_eta1 = 'i' in self.params
        has_eta2 = 'j' in self.params

        has_X = has_eta1 or has_eta2
        has_F = has_gamma1 or has_gamma2
        
        if has_X and has_F:
            return 'both'
        elif has_X and not has_F:
            return 'X_only'
        elif has_F and not has_X:
            return 'F_only'
        elif not has_X and not has_F:
            return 'none'
        else:
            raise ValueError(f"Invalid combination of params: {self.params}")

    def _check_and_set_n_features(self, S_hat_onehot, X_onehot, F_hat):
        """
        验证HMM数据格式，要求
        - S_hat_onehot: 说话人观测，one-hot编码，形状 (n_samples, n_actors)
        - X_onehot: 协变量，one-hot编码，形状 (n_samples, n_actors)，可为None
        - F_hat: 人脸观测，二进制数据，形状 (n_samples, n_actors)，可为None
        """
        if S_hat_onehot.shape[1] != self.n_actors:
            raise ValueError(f"Expected {self.n_actors} actors, got {S_hat_onehot.shape}")
            
        # 检查 S_hat_onehot 是one-hot编码
        if not np.allclose(S_hat_onehot.sum(axis=1), 1):
            raise ValueError("S_hat_onehot must be one-hot encoded (each row sums to 1)")

        # 根据 covariate_mode 检查输入
        if self.covariate_mode in ['X_only', 'both']:
            assert X_onehot is not None, f"X_onehot is required for {self.covariate_mode} mode"
            if X_onehot.shape != S_hat_onehot.shape:
                raise ValueError(f"X_onehot and S_hat_onehot must have the same shape, got {X_onehot.shape} and {S_hat_onehot.shape}")
            if not np.all(np.isclose(X_onehot.sum(axis=1), 0) | np.isclose(X_onehot.sum(axis=1), 1)):
                raise ValueError("X_onehot must be one-hot encoded or all zeros (each row sums to 0 or 1)")
                
        elif self.covariate_mode in ['F_only', 'both']:
            assert F_hat is not None, f"F_hat is required for {self.covariate_mode} mode"
            if F_hat.shape != S_hat_onehot.shape:
                raise ValueError(f"F_hat and S_hat_onehot must have the same shape, got {F_hat.shape} and {S_hat_onehot.shape}")
            if not np.all(np.isin(F_hat, [0, 1])):
                raise ValueError("F_hat must be binary (0 or 1)")
                
        # 'none' mode: no additional checks needed

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
        """初始化HMM的参数"""
        random_state = check_random_state(self.random_state)

        if 'c' in self.params:
            # β: 说话人初始概率的logits,不要求和为1
            self.beta_ = random_state.normal(0, 1, self.n_actors)
            self.beta_ -= self.beta_[0]  # 固定第一个演员的logit为0，作为基准

        if 'd' in self.params:
            # γ1: F_hat 对说话人初始状态的影响
            self.gamma1_ = random_state.uniform(0.5, 2.0)
            
        if 'e' in self.params:
            # A_S: 说话人状态转移矩阵的logits (n_actors, n_actors),不要求和为1
            diag_main = np.diag(random_state.uniform(0.3, 0.7, self.n_actors))
            self.A_S_ = diag_main + (1-diag_main) * random_state.normal(0, 1, (self.n_actors, self.n_actors))
            self.A_S_ -= np.diag(self.A_S_)[:,None]    # 固定转移到自己的logit为0，作为基准

        if 'f' in self.params:
            # γ2: F_hat 对说话人转移的影响
            self.gamma2_ = random_state.uniform(0.5, 2.0)

        if 'h' in self.params:
            # B_S: 说话人识别混淆矩阵 (n_actors, n_actors), 每行和为1
            self.B_S_ = np.zeros((self.n_actors, self.n_actors))
            for actor in range(self.n_actors):
                self.B_S_[actor] = random_state.dirichlet([2 if i == actor else 1 for i in range(self.n_actors)])

        if 'i' in self.params:
            # η1: 协变量X取值为1对说话人初始状态的影响
            self.eta1_ = random_state.uniform(1, 3)

        if 'j' in self.params:
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

    def F2index(self, f_onehot):
        """
        将人脸观测的one-hot编码转换为索引
        - f_onehot: 形状 (n_actors,) 的0-1数组
        - return: 索引，范围 [0, n_face_states-1]
        """
        index = int(np.dot(f_onehot, self.face_state_powers))
        return index

    def fit(self, S_hat_onehot, X_onehot=None, F_hat=None, B_S_diag_min=None, lengths=None):
        """训练HMM模型"""
        S_hat_onehot = np.array(S_hat_onehot)
        if X_onehot is not None:
            X_onehot = np.array(X_onehot)
        if F_hat is not None:
            F_hat = np.array(F_hat)
        
        self._check_and_set_n_features(S_hat_onehot, X_onehot, F_hat)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        
        # 初始化参数
        self._init_params()
        # 重置收敛监控器
        self.monitor_._reset()
        
        # EM算法主循环
        for n_iter in range(self.n_iter):
            # E步：计算前向后向概率和期望统计量
            start_time = time.time()
            stats = self._do_estep(S_hat_onehot, X_onehot, F_hat, lengths)
            estep_time = time.time() - start_time

            # 检查收敛
            curr_loglik = stats['log_likelihood'] # 计算当前对数似然
            self.monitor_.report(curr_loglik)
            if self.monitor_.converged:
                break

            # M步：更新参数
            start_time = time.time()
            self._do_mstep(stats, B_S_diag_min, lengths)
            mstep_time = time.time() - start_time

            # print(f"E步耗时: {estep_time:.4f}秒")
            # print(f"M步耗时: {mstep_time:.4f}秒")

        return self

    def _do_estep(self, S_hat_onehot, X_onehot, F_hat, lengths):
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
            seq_X_onehot = X_onehot[start_idx:end_idx] if X_onehot is not None else None
            seq_F_hat = F_hat[start_idx:end_idx] if F_hat is not None else None
            
            # 前向算法
            start_time = time.time()
            fwd_lattice = self._do_forward_pass(seq_S_hat_onehot, seq_X_onehot, seq_F_hat)
            forward_time += time.time() - start_time
            
            # 后向算法
            start_time = time.time()
            bwd_lattice = self._do_backward_pass(seq_S_hat_onehot, seq_X_onehot, seq_F_hat)
            backward_time += time.time() - start_time
            
            # 计算观测序列段的对数似然 $\bbP(\cI_i^{obs}\vert\btheta^{(s)})$
            seq_loglik = logsumexp(fwd_lattice[-1])
            log_likelihood += seq_loglik
            
            # 更新累积统计量，实现对 i=1,...,m 的求和
            start_time = time.time()
            stats_updated = self._accumulate_sufficient_statistics(
                stats, seq_S_hat_onehot, seq_X_onehot, seq_F_hat, fwd_lattice, bwd_lattice, seq_loglik)
            accumulate_time += time.time() - start_time
            stats = stats_updated

            start_idx = end_idx
            
        stats['log_likelihood'] = log_likelihood
        
        # print(f"前向算法总时间: {forward_time:.4f}秒")
        # print(f"后向算法总时间: {backward_time:.4f}秒")
        # print(f"累积统计量更新总时间: {accumulate_time:.4f}秒")
        
        return stats

    def _do_forward_pass(self, S_hat_onehot, X_onehot, F_hat):
        """
        前向算法计算前向概率的对数
            - S_hat_onehot: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_onehot: 协变量，形状 (n_samples, n_actors)，one-hot编码（可以全0），可为None
            - F_hat: 人脸观测，形状 (n_samples, n_actors)，二进制数据，可为None
            - return: fwd_lattice, 形状 (n_samples, n_actors), (t, s) 表示时刻t 说话人为s的对数概率 $ \log (\bbU_{i,t}(\varrho))$
        """
        n_samples = len(S_hat_onehot)
        
        # fwd_lattice[t, s] = log P(观测到t时刻, 说话人s)
        fwd_lattice = np.full((n_samples, self.n_actors), -np.inf)
        
        # 初始时刻
        ## 计算初始时刻所有说话人发生的概率
        x_config_0 = X_onehot[0] if X_onehot is not None else None
        f_config_0 = F_hat[0] if F_hat is not None else None
        log_speaker_probs = self._compute_speaker_initial_probs(x_config_0, f_config_0)  # shape (n_actors,)
        ## 计算初始时刻所有可能隐藏状态对应的观测概率
        log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[0])
        fwd_lattice[0, :] = log_speaker_probs + log_speaker_emissions

        # 递推
        for t in range(1, n_samples):
            ## 检查 fwd_lattice[t-1, :] 中是否有 -np.inf
            prev_fwd_lattice = fwd_lattice[t-1, :] # shape (n_actors,), corresponds to prev_speaker
            num_inf = np.sum(prev_fwd_lattice == -np.inf)
            total_elements = prev_fwd_lattice.size
            if num_inf == total_elements:
                continue
            elif num_inf > 0:
                print(f"t={t}: fwd_lattice[t-1] contains {num_inf} -np.inf out of {total_elements} elements")

            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t])

            ## 计算当前时刻所有可能的转移概率
            x_config_t = X_onehot[t] if X_onehot is not None else None
            f_config_t = F_hat[t] if F_hat is not None else None
            log_trans_probs = self._compute_speaker_transition_probs(x_config_t, f_config_t)  # shape (n_actors_prev, n_actors_curr)
            
            log_probs_arr = (prev_fwd_lattice[:, None] + log_trans_probs + log_speaker_emissions[None, :])
            ## 对上一时刻的说话人求和，更新前向概率
            fwd_lattice[t, :] = logsumexp(log_probs_arr, axis=0)
        
        return fwd_lattice

    def _do_backward_pass(self, S_hat_onehot, X_onehot, F_hat):
        """
        后向算法计算后向概率的对数
            - S_hat_onehot: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_onehot: 协变量，形状 (n_samples, n_actors)，one-hot编码（可以全0），可为None
            - F_hat: 人脸观测，形状 (n_samples, n_actors)，二进制数据，可为None
            - return: bwd_lattice, 形状 (n_samples, n_actors), (t, s) 表示时刻 t 说话人为s的对数概率 $ \log (\bbV_{i,t}(\varrho))$
        """
        n_samples = len(S_hat_onehot)
        
        # bwd_lattice[t, s] = log P(t+1时刻之后的观测 | t时刻说话人s)
        bwd_lattice = np.full((n_samples, self.n_actors), -np.inf)
        
        # 终止时刻
        bwd_lattice[-1, :] = 0.0

        # 反向递推
        for t in range(n_samples - 2, -1, -1):
            ## 检查 bwd_lattice[t+1, :] 中是否有 -np.inf
            next_bwd_lattice = bwd_lattice[t+1, :] # shape (n_actors,), corresponds to next_speaker
            num_inf = np.sum(next_bwd_lattice == -np.inf)
            total_elements = next_bwd_lattice.size
            if num_inf == total_elements:
                continue
            elif num_inf > 0:
                print(f"t={t}: bwd_lattice[t+1] contains {num_inf} -np.inf out of {total_elements} elements")

            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t+1])

            ## 计算当前时刻所有可能的转移概率
            x_config_t1 = X_onehot[t+1] if X_onehot is not None else None
            f_config_t1 = F_hat[t+1] if F_hat is not None else None
            log_trans_probs = self._compute_speaker_transition_probs(x_config_t1, f_config_t1)  # shape (n_actors_prev, n_actors_curr)
            
            log_probs_arr = (next_bwd_lattice[None, :] + log_trans_probs + log_speaker_emissions[None, :])
            ## 对下一时刻的说话人求和，更新后向概率
            bwd_lattice[t, :] = logsumexp(log_probs_arr, axis=1)
        
        return bwd_lattice

    def _compute_speaker_initial_probs(self, x_config, f_config):
        """
        计算所有说话人的初始概率 $\bbP(S_{i,1}=\cdot \vert X_{i,1,\cdot}=x, F_{i,1,\cdot}=f)$ 的对数
        - x_config: shape (n_actors,), one-hot or all-zero
        - f_config: shape (n_actors,), binary
        """
        logits = self.beta_.copy()
        
        if self.covariate_mode == 'X_only':
            logits += self.eta1_ * x_config
        elif self.covariate_mode == 'F_only':
            logits += self.gamma1_ * f_config
        elif self.covariate_mode == 'both':
            logits += self.gamma1_ * f_config + self.eta1_ * x_config
        # 'none' mode: logits = beta_ only
        
        log_probs = logits - logsumexp(logits)  # of shape (n_actors,)
        return log_probs

    def _compute_speaker_transition_probs(self, x_config, f_config):
        """
        计算从所有说话人转移到所有说话人的概率矩阵的对数
        $\bbP(S_{i,t+1}=\cdot \vert S_{i,t}=\cdot, X_{i,t+1,\cdot}=x, F_{i,t+1,\cdot}=f)$
        - x_config: shape (n_actors,), one-hot or all-zero
        - f_config: shape (n_actors,), binary
        - return: shape (n_actors_prev, n_actors_curr)
        """
        # logits shape: (n_actors_prev, n_actors_curr)
        logits = self.A_S_.copy()
        
        if self.covariate_mode == 'X_only':
            logits += self.eta2_ * x_config[None, :]
        elif self.covariate_mode == 'F_only':
            logits += self.gamma2_ * f_config[None, :]
        elif self.covariate_mode == 'both':
            logits += self.gamma2_ * f_config[None, :] + self.eta2_ * x_config[None, :]
        # 'none' mode: logits = A_S_ only
        
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)  # of shape (n_actors_prev, n_actors_curr)
        return log_probs

    def _compute_emission_probs(self, s_hat):
        """
        计算所有隐藏状态对应的对数发射概率$\bB_S(S_{i,t},\hat S_{i,t})$
        """
        assert s_hat.shape[0] == self.n_actors
        # 对数说话人观测概率 $\bB_S(S_{i,t},\hat S_{i,t})$
        speaker_obs = np.argmax(s_hat)  # one-hot to index
        log_speaker_emissions = np.log(self.B_S_[:, speaker_obs])  # shape (n_actors,)
        return log_speaker_emissions

    def _initialize_sufficient_statistics(self):
        """
        初始化充分统计量，也即 M 步用到的期望值
        根据 covariate_mode 确定统计量的形状
        """
        sufficient_stats = {}

        if self.covariate_mode == 'none':
            sufficient_stats = {
                'speaker_initial_counts': np.zeros(self.n_actors),  # [s_init]
                'speaker_transition_counts': np.zeros((self.n_actors, self.n_actors)),  # [s_prev, s_curr]
            }
        elif self.covariate_mode == 'X_only':
            sufficient_stats = {
                'speaker_initial_counts': np.zeros((self.n_actors, self.n_actors + 1)),  # [s_init, x_index]
                'speaker_transition_counts': np.zeros((self.n_actors, self.n_actors, self.n_actors + 1)),  # [s_prev, s_curr, x_index]
            }
        elif self.covariate_mode == 'F_only':
            sufficient_stats = {
                'speaker_initial_counts': np.zeros((self.n_actors, self.n_face_states)),  # [s_init, f_actor]
                'speaker_transition_counts': np.zeros((self.n_actors, self.n_actors, self.n_face_states)),  # [s_prev, s_curr, f_actor]
            }
        elif self.covariate_mode == 'both':
            sufficient_stats = {
                'speaker_initial_counts': np.zeros((self.n_actors, self.n_face_states, self.n_actors + 1)),  # [s_init, f_actor, x_index]
                'speaker_transition_counts': np.zeros((self.n_actors, self.n_actors, self.n_face_states, self.n_actors + 1)),  # [s_prev, s_curr, f_actor, x_index]
            }
        else:
            raise ValueError(f"Invalid covariate_mode: {self.covariate_mode}")
        
        sufficient_stats['speaker_emission_counts'] = np.zeros((self.n_actors, self.n_actors))  # [s, s_hat]
        return sufficient_stats

    def _accumulate_sufficient_statistics(self, stats, S_hat_onehot, X_onehot, F_hat, fwd_lattice, bwd_lattice, seq_loglik):
        """
        更新累积充分统计量 stats，以便于后续执行参数更新
        """
        n_samples = len(S_hat_onehot)
        stats_updated = copy.deepcopy(stats)
        
        # 计算后验概率
        for t in range(n_samples):
            # 单时刻后验概率 gamma[t, s] = P(S_t=s | 全部观测, 全部协变量)
            gamma = fwd_lattice[t] + bwd_lattice[t] - seq_loglik
            gamma = np.exp(gamma)   # shape (n_actors)

            # 将协变量从one-hot/binary 转为 index
            if self.covariate_mode in ['X_only', 'both']:
                active_x = self.X2index(X_onehot[t])
            if self.covariate_mode in ['F_only', 'both']:
                active_f = self.F2index(F_hat[t])
            # 累积初始统计量
            if t == 0:
                # 计算说话人初始充分统计量 $\bbE\left[\bbN(S_{\cdot,1}=\varrho\vert \btheta^{(s)})\right] $
                if self.covariate_mode == 'none':
                    stats_updated['speaker_initial_counts'] += gamma
                elif self.covariate_mode == 'X_only':
                    stats_updated['speaker_initial_counts'][:, active_x] += gamma
                elif self.covariate_mode == 'F_only':
                    stats_updated['speaker_initial_counts'][:, active_f] += gamma
                elif self.covariate_mode == 'both':
                    stats_updated['speaker_initial_counts'][:, active_f, active_x] += gamma
                else:
                    raise ValueError(f"Invalid covariate_mode: {self.covariate_mode}")
            
            # 累积转移统计量
            if t > 0:
                ## 计算对数转移后验概率 xi[t-1, s_prev, s_curr]
                log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t])
                
                x_config_t = X_onehot[t] if X_onehot is not None else None
                f_config_t = F_hat[t] if F_hat is not None else None
                log_trans_probs = self._compute_speaker_transition_probs(x_config_t, f_config_t)  # shape (n_actors_prev, n_actors_curr)

                log_xi_arr = (fwd_lattice[t-1][:, None] + log_trans_probs +
                              log_speaker_emissions[None, :] + bwd_lattice[t][None, :] - seq_loglik)
                xi_arr = np.exp(log_xi_arr)  # shape (n_actors_prev, n_actors_curr)

                ## 存储用于说话人转移概率优化的信息
                if self.covariate_mode == 'none':
                    stats_updated['speaker_transition_counts'] += xi_arr
                elif self.covariate_mode == 'X_only':
                    stats_updated['speaker_transition_counts'][:, :, active_x] += xi_arr
                elif self.covariate_mode == 'F_only':
                    stats_updated['speaker_transition_counts'][:, :, active_f] += xi_arr
                elif self.covariate_mode == 'both':
                    stats_updated['speaker_transition_counts'][:, :, active_f, active_x] += xi_arr
                else:
                    raise ValueError(f"Invalid covariate_mode: {self.covariate_mode}")

            # 累积发射统计量
            ## 说话人发射统计量
            speaker_obs = np.argmax(S_hat_onehot[t]) # 说话人期望计算式中的 $\varrho'$
            stats_updated['speaker_emission_counts'][:, speaker_obs] += gamma

        return stats_updated

    def _do_mstep(self, stats, B_S_diag_min, lengths):
        """M步：更新参数"""
        # 更新说话人初始概率参数
        if 'c' in self.params or 'd' in self.params or 'i' in self.params:
            self._update_speaker_initial_params(stats)
        
        # 更新说话人转移概率参数
        if 'e' in self.params or 'f' in self.params or 'j' in self.params:
            self._update_speaker_transition_params(stats)
        
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

    def _update_speaker_initial_params(self, stats):
        """使用数值优化更新说话人初始参数"""
        if self.covariate_mode == 'none':
            # 直接用统计量更新，转为概率
            total = stats['speaker_initial_counts'].sum()
            if total > 0:
                probs = stats['speaker_initial_counts'] / total
                # 转为 logits，固定第一个为0
                probs = np.clip(probs, 1e-10, 1-1e-10)
                self.beta_ = np.log(probs) - np.log(probs[0])
            else:
                print("Warning: No data for speaker initial counts in 'none' mode.")

            return  # 直接返回，不需要数值优化
        
        def objective_speaker_initial(params):
            if self.covariate_mode == 'X_only':
                beta, eta1 = np.concatenate(([0.0], params[:-1])), params[-1]
                weights = np.transpose(stats['speaker_initial_counts'], axes=(1, 0))  # [x_index, s_init]
                
                logits = beta[None, :] + eta1 * self.X_arr[:, :]
                log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            elif self.covariate_mode == 'F_only':
                beta, gamma1 = np.concatenate(([0.0], params[:-1])), params[-1]
                weights = np.transpose(stats['speaker_initial_counts'], axes=(1, 0))  # [f_actor, s_init]
                nonzero_f_indices = np.where(weights.sum(axis=1) > 0)[0]  # 找出非零行的索引
                weights = weights[nonzero_f_indices]
                F_configs = self.face_configs_arr[nonzero_f_indices]  # 只保留对应的面部配置

                logits = beta[None, :] + gamma1 * F_configs
                log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            elif self.covariate_mode == 'both':
                beta, gamma1, eta1 = np.concatenate(([0.0], params[:-2])), params[-2], params[-1]
                weights = np.transpose(stats['speaker_initial_counts'], axes=(1, 2, 0))  # [f_actor, x_index, s_init]
                nonzero_f_indices = np.where(weights.sum(axis=(1,2)) > 0)[0]  # 找出非零行的索引
                weights = weights[nonzero_f_indices]
                F_configs = self.face_configs_arr[nonzero_f_indices]  # 只保留对应的面部配置
                
                logits = beta[None, None, :] + gamma1 * F_configs[:, None, :] + eta1 * self.X_arr[None, :, :]
                log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
            
            masks = (weights > 0)
            loss = -np.sum(weights[masks] * log_probs[masks])
            return loss

        if self.covariate_mode == 'X_only':
            x0 = np.concatenate([self.beta_[1:], np.array([self.eta1_])])  
        elif self.covariate_mode == 'F_only':
            x0 = np.concatenate([self.beta_[1:], np.array([self.gamma1_])])
        elif self.covariate_mode == 'both':
            x0 = np.concatenate([self.beta_[1:], np.array([self.gamma1_, self.eta1_])])

        result = minimize(objective_speaker_initial, x0, method='L-BFGS-B')
        obj_init = objective_speaker_initial(x0)
        obj_final = objective_speaker_initial(result.x)
        
        if result.success or obj_final < obj_init:
            if self.covariate_mode == 'X_only':
                self.beta_ = np.concatenate(([0.0], result.x[:-1]))
                self.eta1_ = result.x[-1]
            elif self.covariate_mode == 'F_only':
                self.beta_ = np.concatenate(([0.0], result.x[:-1]))
                self.gamma1_ = result.x[-1]
            elif self.covariate_mode == 'both':
                self.beta_ = np.concatenate(([0.0], result.x[:-2]))
                self.gamma1_ = result.x[-2]
                self.eta1_ = result.x[-1]
        else:
            print("Warning: Speaker initial parameters optimization did not converge.")

        print(f"Initial objective: {obj_init:.4f}, Final objective: {obj_final:.4f}")


    def _update_speaker_transition_params(self, stats):
        """使用数值优化更新说话人转移参数"""
        mask_offdiag = ~np.eye(self.n_actors, dtype=bool)

        if self.covariate_mode == 'none':
            # 直接用统计量更新，每行归一化
            for prev_s in range(self.n_actors):
                row_sum = stats['speaker_transition_counts'][prev_s].sum()
                if row_sum > 0:
                    probs = stats['speaker_transition_counts'][prev_s] / row_sum
                    probs = np.clip(probs, 1e-10, 1-1e-10)
                    self.A_S_[prev_s] = np.log(probs) - np.log(probs[prev_s])
                else:
                    print(f"Warning: No data for speaker transition from speaker {prev_s} in 'none' mode.")
            return  # 直接返回，不需要数值优化

        def objective_speaker_transition(params):
            A_S_mat = np.zeros((self.n_actors, self.n_actors))
            if self.covariate_mode == 'X_only':
                A_S_mat[mask_offdiag] = params[:-1]
                eta2 = params[-1]
                weights = np.transpose(stats['speaker_transition_counts'], axes=(2, 0, 1))  # [x_index, s_prev, s_curr]
                
                logits = A_S_mat[None, :, :] + eta2 * self.X_arr[:, None, :]
                log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
            elif self.covariate_mode == 'F_only':
                A_S_mat[mask_offdiag] = params[:-1]
                gamma2 = params[-1]
                weights = np.transpose(stats['speaker_transition_counts'], axes=(2, 0, 1))  # [f_actor, s_prev, s_curr]
                nonzero_f_indices = np.where(weights.sum(axis=(1,2)) > 0)[0]  # 找出非零行的索引
                weights = weights[nonzero_f_indices]
                F_configs = self.face_configs_arr[nonzero_f_indices]  # 只保留对应的面部配置

                logits = A_S_mat[None, :, :] + gamma2 * F_configs[:, None, :]
                log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
            elif self.covariate_mode == 'both':
                A_S_mat[mask_offdiag] = params[:-2]
                gamma2, eta2 = params[-2], params[-1]
                weights = np.transpose(stats['speaker_transition_counts'], axes=(2, 3, 0, 1))  # [f_actor, x_index, s_prev, s_curr]
                
                # 对后两个维度求和，得到 [f_actor, x_index] 的二维矩阵
                weights_sum = weights.sum(axis=(2, 3))  # shape: (n_face_states, n_actors+1)
                # 找出所有非零元素的索引
                nonzero_indices = np.nonzero(weights_sum)  # 返回两个数组: (f_indices, x_indices)
                f_indices = nonzero_indices[0]  # shape: (n_nonzero,)
                x_indices = nonzero_indices[1]  # shape: (n_nonzero,)
                
                # 根据非零索引提取对应的 weights 子集
                weights = weights[f_indices, x_indices, :, :]  # shape: (n_nonzero, n_actors, n_actors)
                
                # 根据索引提取对应的配置
                F_configs = self.face_configs_arr[f_indices]  # shape: (n_nonzero, n_actors)
                X_configs = self.X_arr[x_indices]  # shape: (n_nonzero, n_actors)
                logits = (A_S_mat[None, :, :] + gamma2 * F_configs[:, None, :] +
                          eta2 * X_configs[:, None, :])  # shape: (n_nonzero, n_actors, n_actors)
                
                log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
            
            mask = (weights > 0)
            loss = -np.sum(weights[mask] * log_probs[mask])
            return loss

        # 优化参数
        if self.covariate_mode == 'X_only':
            x0 = np.concatenate([self.A_S_[mask_offdiag], np.array([self.eta2_])])
        elif self.covariate_mode == 'F_only':
            x0 = np.concatenate([self.A_S_[mask_offdiag], np.array([self.gamma2_])])
        elif self.covariate_mode == 'both':
            x0 = np.concatenate([self.A_S_[mask_offdiag], np.array([self.gamma2_, self.eta2_])])

        result = minimize(objective_speaker_transition, x0, method='L-BFGS-B')
        obj_init = objective_speaker_transition(x0)
        obj_final = objective_speaker_transition(result.x)

        # 参数赋值
        if result.success or obj_final < obj_init:
            if self.covariate_mode == 'X_only':
                self.A_S_ = np.zeros((self.n_actors, self.n_actors))
                self.A_S_[mask_offdiag] = result.x[:-1]
                self.eta2_ = result.x[-1]
            elif self.covariate_mode == 'F_only':
                self.A_S_ = np.zeros((self.n_actors, self.n_actors))
                self.A_S_[mask_offdiag] = result.x[:-1]
                self.gamma2_ = result.x[-1]
            elif self.covariate_mode == 'both':
                self.A_S_ = np.zeros((self.n_actors, self.n_actors))
                self.A_S_[mask_offdiag] = result.x[:-2]
                self.gamma2_ = result.x[-2]
                self.eta2_ = result.x[-1]
        else:
            print("Warning: Speaker transition parameters optimization did not converge.")

        print(f"Initial objective: {obj_init:.4f}, Final objective: {obj_final:.4f}")

    def score(self, S_hat_onehot, X_onehot=None, F_hat=None, lengths=None):
        """计算观测序列的对数似然"""
        S_hat_onehot = np.array(S_hat_onehot)
        if X_onehot is not None:
            X_onehot = np.array(X_onehot)
        if F_hat is not None:
            F_hat = np.array(F_hat)

        # EM算法总以M步结束，为了确保计算最新的对数似然，这里重新计算一次E步        
        return self._do_estep(S_hat_onehot, X_onehot, F_hat, lengths)['log_likelihood']

    def predict_proba(self, S_hat_onehot, X_onehot=None, F_hat=None, lengths=None):
        """
        计算给定观测序列时说话人隐藏状态的后验概率 $\lambda_{i,t,\\varrho} = \\bbP(S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)})$
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        X_onehot : array-like, shape (n_samples, n_actors), optional
            观测的X状态，one-hot编码(可以全0)
        F_hat : array-like, shape (n_samples, n_actors), optional
            人脸观测，二进制数据
        lengths : array-like of integers, optional
            每个序列的长度
            
        Returns
        -------
        posteriors : dict
            包含各种后验概率的字典:
            - 'speaker_states': array, shape (n_samples, n_actors)
              每个时刻每个演员是说话人的后验概率  $ \lambda_{i,t,\\varrho} $
        """
        S_hat_onehot = np.array(S_hat_onehot)
        if X_onehot is not None:
            X_onehot = np.array(X_onehot)
        if F_hat is not None:
            F_hat = np.array(F_hat)
            
        self._check_and_set_n_features(S_hat_onehot, X_onehot, F_hat)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        n_samples = len(S_hat_onehot)

        # 初始化输出
        speaker_posteriors = np.zeros((n_samples, self.n_actors))

        # 对每一集的数据        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            ## 获取当前序列段
            seq_S_hat_onehot = S_hat_onehot[start_idx:end_idx]
            seq_X_onehot = X_onehot[start_idx:end_idx] if X_onehot is not None else None
            seq_F_hat = F_hat[start_idx:end_idx] if F_hat is not None else None
            
            ## 计算前向和后向概率，以及序列的对数似然
            fwd_lattice = self._do_forward_pass(seq_S_hat_onehot, seq_X_onehot, seq_F_hat)
            bwd_lattice = self._do_backward_pass(seq_S_hat_onehot, seq_X_onehot, seq_F_hat)
            seq_loglik = logsumexp(fwd_lattice[-1])
            
            ## 计算每个时刻的后验概率
            for t in range(length):
                ### 获取并存储联合后验概率 P(S_t=s | 全部观测)
                log_gamma = fwd_lattice[t] + bwd_lattice[t] - seq_loglik
                gamma = np.exp(log_gamma)
                speaker_posteriors[start_idx + t] = gamma
            
            start_idx = end_idx
        
        return {
            'speaker_states': speaker_posteriors, 
        }

    def predict(self, S_hat_onehot, X_onehot=None, F_hat=None, lengths=None):
        """
        使用Viterbi算法，预测最可能的隐藏状态序列(说话人状态)
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        X_onehot : array-like, shape (n_samples, n_actors), optional
            观测的X状态，one-hot编码(可以全0)
        F_hat : array-like, shape (n_samples, n_actors), optional
            人脸观测，二进制数据
        lengths : array-like of integers, optional
            每个序列的长度，如果为None，则假设是单一序列

        Returns
        -------
        speaker_states : array, shape (n_samples,)
            预测的说话人状态序列 (0到n_actors-1)
        """
        S_hat_onehot = np.asarray(S_hat_onehot)
        if X_onehot is not None:
            X_onehot = np.asarray(X_onehot)
        if F_hat is not None:
            F_hat = np.asarray(F_hat)
            
        self._check_and_set_n_features(S_hat_onehot, X_onehot, F_hat)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        
        # 初始化输出数组
        speaker_states = np.zeros(S_hat_onehot.shape[0], dtype=int)
        
        start_idx = 0
        for seq_len in lengths:
            end_idx = start_idx + seq_len
            
            # 提取当前序列
            seq_S_hat_onehot = S_hat_onehot[start_idx:end_idx]
            seq_X_onehot = X_onehot[start_idx:end_idx] if X_onehot is not None else None
            seq_F_hat = F_hat[start_idx:end_idx] if F_hat is not None else None

            # 使用维特比算法预测
            seq_speaker_states = self._viterbi(seq_S_hat_onehot, seq_X_onehot, seq_F_hat)
            
            # 存储结果
            speaker_states[start_idx:end_idx] = seq_speaker_states
            
            start_idx = end_idx
            
        return speaker_states
        
    def _viterbi(self, S_hat_onehot, X_onehot, F_hat):
        """
        对单个序列使用维特比算法进行解码
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_frames, n_actors)
            说话人观测序列
        X_onehot : array-like, shape (n_frames, n_actors), optional
            观测的X状态序列
        F_hat : array-like, shape (n_frames, n_actors), optional
            人脸观测序列
        
        Returns
        -------
        speaker_states : array, shape (n_frames,)
            预测的说话人状态序列
        """
        n_frames = S_hat_onehot.shape[0]
        
        # 初始化维特比表格 $\delta_{t}(s)$ 与回溯路径 $\psi_{t}(s)$
        ## $\delta_{t}(s)$: 在时刻t，说话人s的最大概率的对数
        viterbi = np.full((n_frames, self.n_actors), -np.inf)
        ## $\psi_{t}(s)$: t 时刻说话人s时，从1到t的路径中，后验概率最大的路径在 $t-1$ 时刻的状态 (s')
        path_speaker = np.zeros((n_frames, self.n_actors), dtype=int) # 每个元素是s'的索引
        
        # 初始化: t=0时刻，已知隐状态与观测的联合概率的对数
        ## 计算对数初始说话人隐藏状态概率
        x_config_0 = X_onehot[0] if X_onehot is not None else None
        f_config_0 = F_hat[0] if F_hat is not None else None
        log_speaker_probs = self._compute_speaker_initial_probs(x_config_0, f_config_0)  # shape (n_actors,)
        ## 计算对数观测概率 P(S_hat | S)
        log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[0])
        viterbi[0, :] = log_speaker_probs + log_speaker_emissions
        
        # 前向传播 t=1到n_frames-1
        for t in range(1, n_frames):
            x_config_t = X_onehot[t] if X_onehot is not None else None
            f_config_t = F_hat[t] if F_hat is not None else None
            log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t])
            log_trans_probs = self._compute_speaker_transition_probs(x_config_t, f_config_t)  # shape (n_actors_prev, n_actors_curr)

            # 计算每个当前状态对应的 $\delta_{t}(s)$
            for speaker in range(self.n_actors):
                # 遍历所有可能的前一状态 (s_prev)，确定最佳前一状态
                total_prob_prev_no_obs = viterbi[t-1, :] + log_trans_probs[:, speaker]
                best_prev_s = np.argmax(total_prob_prev_no_obs)

                # 确定 $\delta_{t}(s)$
                viterbi[t, speaker] = total_prob_prev_no_obs[best_prev_s] + log_speaker_emissions[speaker]
                # 确定 $\psi_{t}(s)$
                path_speaker[t, speaker] = best_prev_s
        
        # 找到最优路径的结束状态 $i_T^\ast$: best_end_s
        last_viterbi = viterbi[n_frames-1, :]  # shape (n_actors)
        best_end_s = np.argmax(last_viterbi)
        
        # 回溯最优路径
        speaker_states = np.zeros(n_frames, dtype=int)
        curr_s = best_end_s
        for t in range(n_frames-1, -1, -1):
            # 记录当前状态
            speaker_states[t] = curr_s
            
            # 回溯到前一状态
            if t > 0:
                prev_s = path_speaker[t, curr_s]
                curr_s = prev_s
        
        return speaker_states
