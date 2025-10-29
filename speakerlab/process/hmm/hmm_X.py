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
    
    def __init__(self, n_actors, n_iter=100, tol=1e-2, verbose=False, equal_influence=False,
                 params="cehij", init_params="cehij", random_state=None):
        self.n_actors = n_actors    # 演员数量
        self.n_iter = n_iter    # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印详细信息
        self.params = params    # 控制哪些参数被更新
        self.init_params = init_params  # 控制哪些参数被初始化
        self.equal_influence = equal_influence  # 是否强制协变量对初始和转移概率的影响相等
        self.random_state = random_state

        # 添加缓存变量，避免反复计算
        ## 预计算的转移矩阵（在每次EM迭代后需要更新）
        self._log_trans_speaker = np.zeros((self.n_actors, self.n_actors, self.n_actors+1))  # [prev_s, s, x]
        ## 所有可能的X配置
        self.X_arr = np.vstack([np.eye(self.n_actors), np.zeros((1, self.n_actors))])  # shape (n_actors+1, n_actors), 每行表示一个one-hot编码的协变量配置或全零配置

        # 创建监控器
        self.monitor_ = ConvergenceMonitor(tol, n_iter, verbose)

    def _check_and_set_n_features(self, S_hat_onehot, X_onehot):
        """
        验证嵌套HMM数据格式，要求
        - S_hat_onehot: 说话人观测，one-hot编码，形状 (n_samples, n_actors)
        - X_onehot: 协变量，one-hot编码，形状 (n_samples, n_actors)
        """
        if X_onehot.shape != S_hat_onehot.shape:
            raise ValueError(f"X_onehot and S_hat_onehot must have the same shape, got {X_onehot.shape} and {S_hat_onehot.shape}")

        if S_hat_onehot.shape[1] != self.n_actors:
            raise ValueError(f"Expected {self.n_actors} actors, got {S_hat_onehot.shape}")
            
        # 检查 S_hat_onehot 是one-hot编码
        if not np.allclose(S_hat_onehot.sum(axis=1), 1):
            raise ValueError("S_hat_onehot must be one-hot encoded (each row sums to 1)")

        # 检查X_onehot是one-hot编码（可以全为零）
        if not np.all(np.isclose(X_onehot.sum(axis=1), 0) | np.isclose(X_onehot.sum(axis=1), 1)):
            raise ValueError("X_onehot must be one-hot encoded or all zeros (each row sums to 0 or 1)")

    def _validate_lengths(self, X, lengths):
        """
        验证序列长度，要求lengths元素之和等于X的样本数
        """
        if lengths is None:
            return [len(X)]
        
        if np.asarray(lengths).sum() != len(X):
            raise ValueError("Sum of lengths must equal number of samples")
        
        return lengths

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

    def _init_params(self):
        """初始化嵌套HMM的参数"""
        random_state = check_random_state(self.random_state)

        if 'c' in self.init_params:
            # β: 说话人初始概率的logits,不要求和为1
            self.beta_ = random_state.normal(0, 1, self.n_actors)
            self.beta_ -= self.beta_[0]  # 固定第一个演员的logit为0，作为基准
            
        if 'e' in self.init_params:
            # A_S: 说话人状态转移矩阵的logits (n_actors, n_actors),不要求和为1
            diag_main = np.diag(random_state.uniform(0.3, 0.7, self.n_actors))
            self.A_S_ = diag_main + (1-diag_main) * random_state.normal(0, 1, (self.n_actors, self.n_actors))
            self.A_S_ -= np.diag(self.A_S_)[:,None]    # 固定转移到自己的logit为0，作为基准

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
            if self.equal_influence:
                self.eta2_ = self.eta1_
            else:
                self.eta2_ = random_state.uniform(1, 3)

        self._update_log_transition_matrices()

    def _update_log_transition_matrices(self):
        """预计算用于前向/后向算法的对数转移矩阵。"""
        n_actors = self.n_actors

        # 说话人状态转移张量: _log_trans_speaker[prev_s, s, x]
        for prev_s in range(n_actors):
            for active_x in range(n_actors + 1):    # 包括没有活跃说话人的情况
                x_config = self.X_arr[active_x]
                self._log_trans_speaker[prev_s, :, active_x] = self._compute_speaker_transition_probs(prev_s, x_config)

    def fit(self, S_hat_onehot, X_onehot, lengths=None):
        """训练嵌套HMM模型"""
        S_hat_onehot = np.array(S_hat_onehot)
        X_onehot = np.array(X_onehot)

        self._check_and_set_n_features(S_hat_onehot, X_onehot)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        
        # 初始化参数
        self._init_params()
        # 重置收敛监控器
        self.monitor_._reset()
        
        # EM算法主循环
        for n_iter in range(self.n_iter):
            # E步：计算前向后向概率和期望统计量
            start_time = time.time()
            stats = self._do_estep(S_hat_onehot, X_onehot, lengths)
            estep_time = time.time() - start_time

            # 检查收敛
            curr_loglik = stats['log_likelihood'] # 计算当前对数似然
            self.monitor_.report(curr_loglik)
            if self.monitor_.converged:
                break

            # M步：更新参数
            start_time = time.time()
            self._do_mstep(stats, lengths)
            mstep_time = time.time() - start_time

            # print(f"E步耗时: {estep_time:.4f}秒")
            # print(f"M步耗时: {mstep_time:.4f}秒")

        return self

    def _do_estep(self, S_hat_onehot, X_onehot, lengths):
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
            seq_X_onehot = X_onehot[start_idx:end_idx]
            
            # 前向算法
            start_time = time.time()
            fwd_lattice = self._do_forward_pass(seq_S_hat_onehot, seq_X_onehot)
            forward_time += time.time() - start_time
            
            # 后向算法
            start_time = time.time()
            bwd_lattice = self._do_backward_pass(seq_S_hat_onehot, seq_X_onehot)
            backward_time += time.time() - start_time
            
            # 计算观测序列段的对数似然 $\bbP(\cI_i^{obs}\vert\btheta^{(s)})$
            seq_loglik = logsumexp(fwd_lattice[-1])
            log_likelihood += seq_loglik
            
            # 更新累积统计量，实现对 i=1,...,m 的求和
            start_time = time.time()
            stats_updated = self._accumulate_sufficient_statistics(
                stats, seq_S_hat_onehot, seq_X_onehot, fwd_lattice, bwd_lattice, seq_loglik)
            accumulate_time += time.time() - start_time
            stats = stats_updated

            start_idx = end_idx
            
        stats['log_likelihood'] = log_likelihood
        
        # print(f"前向算法总时间: {forward_time:.4f}秒")
        # print(f"后向算法总时间: {backward_time:.4f}秒")
        # print(f"累积统计量更新总时间: {accumulate_time:.4f}秒")
        
        return stats

    def _do_forward_pass(self, S_hat_onehot, X_onehot):
        """
        前向算法计算前向概率的对数
            - S_hat_onehot: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_onehot: 协变量，形状 (n_samples, n_features)，one-hot编码
            - return: fwd_lattice, 形状 (n_samples, n_actors), (t, s) 表示时刻t 说话人为s的对数概率 $ \log (\bbU_{i,t}(\varrho))$
        """
        n_samples = len(S_hat_onehot)
        
        # fwd_lattice[t, s] = log P(观测到t时刻, 说话人s)
        fwd_lattice = np.full((n_samples, self.n_actors), -np.inf)
        
        # 初始时刻
        ## 计算初始时刻所有说话人发生的概率
        log_speaker_probs = self._compute_speaker_initial_probs(X_onehot[0])  # shape (n_actors,), 每个元素代表说话人为s的log概率
        ## 计算初始时刻所有可能隐藏状态对应的观测概率
        log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[0])
        fwd_lattice[0, :] = log_speaker_probs + log_speaker_emissions

        # 递推
        for t in range(1, n_samples):
            ## 检查 fwd_lattice[t-1, :, :] 中是否有 -np.inf
            prev_fwd_lattice = fwd_lattice[t-1, :] # shape (n_actors,), corresponds to prev_speaker
            num_inf = np.sum(prev_fwd_lattice == -np.inf)
            total_elements = prev_fwd_lattice.size
            if num_inf == total_elements:
                continue
            elif num_inf > 0:
                print(f"t={t}: fwd_lattice[t-1] contains {num_inf} -np.inf out of {total_elements} elements")

            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t])

            ## 计算当前时刻所有可能的 (f, \varrho)对应的概率log_probs_arr (s_prev, s_curr)
            active_x = self.X2index(X_onehot[t])
            log_probs_arr = (prev_fwd_lattice[:, None] + self._log_trans_speaker[:,:,active_x] + log_speaker_emissions[None, :])
            ## 对上一时刻的 (\varrho') 求和，更新前向概率
            fwd_lattice[t, :] = logsumexp(log_probs_arr, axis=(0))
        
        return fwd_lattice

    def _do_backward_pass(self, S_hat_onehot, X_onehot):
        """
        后向算法计算后向概率的对数
            - S_hat_onehot: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_onehot: 协变量，形状 (n_samples, n_features)，one-hot编码
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

            ## 计算当前时刻所有可能的 (f, \varrho)对应的概率log_probs_arr (s_curr, s_next)
            active_x = self.X2index(X_onehot[t+1])
            log_probs_arr = (next_bwd_lattice[None, :] + self._log_trans_speaker[:,:,active_x] + log_speaker_emissions[None, :])
            ## 对下一时刻的 (\varrho') 求和，更新后向概率
            bwd_lattice[t, :] = logsumexp(log_probs_arr, axis=(1))
        
        return bwd_lattice

    def _compute_speaker_initial_probs(self, x_config):
        """
        计算所有说话人的初始概率 $\bbP(S_{i,1}=\cdot \vert X_{i,1,\cdot}=x)$ 的对数
        """
        logits = self.beta_ + self.eta1_ * x_config
        log_probs = logits - logsumexp(logits)  # of shape (n_actors,), 每个元素代表说话人为s的log概率
        return log_probs

    def _compute_speaker_transition_probs(self, prev_speaker, x_config):
        """
        计算在已知上一时刻说话人时，转移到所有说话人的概率 $\bbP(S_{i,t+1}=\cdot \vert S_{i,t}=\varrho', X_{i,t+1,\cdot}=x)$ 的对数
        """
        logits = self.A_S_[prev_speaker, :] + self.eta2_ * x_config
        log_probs = logits - logsumexp(logits)  # of shape (n_actors,), 每个元素代表说话人为s的log概率
        return log_probs

    def _compute_emission_probs(self, s_hat):
        """
        计算所有隐藏状态组合对应的对数发射概率$\bB_S(S_{i,t},\hat S_{i,t}) \prod_{\varrho\in\cP} \bB_{\varrho}(F_{i,t,\varrho},\hat F_{i,t,\varrho})$
        """
        assert s_hat.shape[0] == self.n_actors
        # 对数说话人观测概率 $\bB_S(S_{i,t},\hat S_{i,t})$
        speaker_obs = np.argmax(s_hat)  # one-hot to index
        log_speaker_emissions = np.log(self.B_S_[:, speaker_obs])  # shape (n_actors,), each element corresponds to a current speaker
        return  log_speaker_emissions

    def _initialize_sufficient_statistics(self):
        """
        初始化充分统计量，也即 M 步用到的期望值
        """
        return {
            'speaker_initial_counts': np.zeros((self.n_actors, self.n_actors + 1)),    # [s_init, x_onehot_init]
            'speaker_transition_counts': np.zeros((self.n_actors, self.n_actors, self.n_actors + 1)),  # [s_prev, s_curr, x_onehot_curr]
            'speaker_emission_counts': np.zeros((self.n_actors, self.n_actors))  # [speaker_state, observed_speaker]
        }

    def _accumulate_sufficient_statistics(self, stats, S_hat_onehot, X_onehot, fwd_lattice, bwd_lattice, seq_loglik):
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

            # 将协变量从one-hot 转为 index
            active_x = self.X2index(X_onehot[t])        
            # 累积初始统计量
            if t == 0:
                # 计算说话人初始充分统计量 $\bbE\left[\bbN(S_{\cdot,1}=\varrho\vert \btheta^{(s)})\right] $
                stats_updated['speaker_initial_counts'][:, active_x] += gamma
            
            # 累积转移统计量
            if t > 0:
                ## 计算对数转移后验概率 xi[t-1, s_prev, s_curr]
                log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t])

                log_xi_arr = (fwd_lattice[t-1, :, None]  + self._log_trans_speaker[:,:,active_x] +
                              log_speaker_emissions[None, :] + bwd_lattice[t, None, :] - seq_loglik) 
                xi_arr = np.exp(log_xi_arr) # 求和式中的每一项

                ## 存储用于说话人转移概率优化的信息，[s_prev, s_curr]
                stats_updated['speaker_transition_counts'][:, :, active_x] += xi_arr

            # 累积发射统计量
            ## 说话人发射统计量
            speaker_obs = np.argmax(S_hat_onehot[t]) # 说话人期望计算式中的 $\varrho'$
            stats_updated['speaker_emission_counts'][:, speaker_obs] += gamma

        return stats_updated

    def _do_mstep(self, stats, lengths):
        """M步：更新参数"""
        if self.equal_influence:
            if 'c' in self.params or 'i' in self.params or 'e' in self.params or 'j' in self.params:
                # 同时更新说话人初始和转移参数
                self._update_speaker_initial_and_transition_params(stats)
        else:
            # 更新说话人初始概率参数 (beta, eta1)
            if 'c' in self.params or 'i' in self.params:
                self._update_speaker_initial_params(stats)
            
            # 更新说话人转移概率参数 (A_S, gamma2, eta2)
            if 'e' in self.params or 'j' in self.params:
                self._update_speaker_transition_params(stats)
        
        # 更新说话人发射矩阵  
        if 'h' in self.params:
            for speaker in range(self.n_actors):  # (15)式中的 $\varrho$
                total = stats['speaker_emission_counts'][speaker].sum()
                if total > 0:
                    self.B_S_[speaker] = stats['speaker_emission_counts'][speaker] / total  # row normalization
                    self.B_S_[speaker] = np.clip(self.B_S_[speaker], 1e-6, 1-1e-6)
                    self.B_S_[speaker] /= self.B_S_[speaker].sum()
                else:
                    self.B_S_[speaker] = np.ones(self.n_actors) / self.n_actors
                    print(f"Warning: Emission probabilities for speaker {speaker} were not updated due to insufficient data. Reset to uniform distribution.")

        # 更新预计算的对数转移矩阵
        self._update_log_transition_matrices()

    def _update_speaker_initial_params(self, stats):
        """使用数值优化更新说话人初始参数"""
        def objective_speaker_initial(params):
            beta, eta1 = np.concatenate(([0.0], params[:-1])), params[-1]
            weights = np.transpose(stats['speaker_initial_counts'], axes=(1, 0))   # [x_onehot_init, s_init]
            masks = (weights > 0)
            logits = beta[None, :] + eta1*self.X_arr[:, :]
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)   # log-softmax
            loss = - np.sum(weights[masks] * log_probs[masks])
            
            return loss
        
        # 初始参数
        x0 = np.concatenate([self.beta_[1:], np.array([self.eta1_])])
        # print(f"Initial parameters for speaker initial params: {x0}")
        # print(sum(stats['speaker_initial_counts']>0))
        # print(stats['speaker_initial_counts'])

            
        # 优化
        result = minimize(objective_speaker_initial, x0, method='L-BFGS-B')
        obj_init = objective_speaker_initial(x0)
        obj_final = objective_speaker_initial(result.x)
        
        if result.success or obj_final < obj_init:
            self.beta_ = np.concatenate(([0.0], result.x[:-1]))
            self.eta1_ = result.x[-1]
        else:
            print("Warning: Speaker initial parameters optimization did not converge.")
            print(f"Initial objective value for speaker initial params: {obj_init:.4f}")
            print(f"Final objective value for speaker initial params: {obj_final:.4f}")

    def _update_speaker_transition_params(self, stats):
        """使用数值优化更新说话人转移参数(只优化非对角线元素和eta2)"""
        mask_offdiag = ~np.eye(self.n_actors, dtype=bool)

        def objective_speaker_transition(params):
            # params: [A_S_offdiag, eta2]
            A_S_mat = np.zeros((self.n_actors, self.n_actors))  # 对角线强制为0
            A_S_mat[mask_offdiag] = params[:-1]# 从flattend 参数重建A_S矩阵
            eta2 = params[-1]
            
            weights = np.transpose(stats['speaker_transition_counts'], axes=(2, 0, 1)) # [x_onehot_curr, s_prev, s_curr]
            mask = (weights > 0)
            # [n_x_states, n_actors_prev, n_actors_curr(speaker)]
            logits = A_S_mat[None, :, :] + eta2 * self.X_arr[:, None, :]   
            log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
            loss = - np.sum(weights[mask] * log_probs[mask])
            
            return loss
        
        # 初始参数：只取A_S_非对角线元素和 eta2
        x0 = np.concatenate([self.A_S_[mask_offdiag], np.array([self.eta2_])])    # shape: (n_actors*(n_actors-1) + 1,)
        
        # 优化
        result = minimize(objective_speaker_transition, x0, method='L-BFGS-B')
        obj_init = objective_speaker_transition(x0)
        obj_final = objective_speaker_transition(result.x)
        
        if result.success or obj_final < obj_init:
            # 重建A_S_，对角线为0
            self.A_S_ = np.zeros((self.n_actors, self.n_actors))
            self.A_S_[mask_offdiag] = result.x[:-1]
            self.eta2_ = result.x[-1]
        else:
            print("Warning: Speaker transition parameters optimization did not converge.")
            print(f"Initial objective value for speaker transition params: {obj_init:.4f}")
            print(f"Final objective value for speaker transition params: {obj_final:.4f}")

    def _update_speaker_initial_and_transition_params(self, stats):
        """
        同时优化说话人初始参数(beta, eta1)和转移参数(A_S_offdiag, eta2)，
        以两个目标函数之和为损失。要求 eta1 = eta2。
        """
        mask_offdiag = ~np.eye(self.n_actors, dtype=bool)
        n_beta = self.n_actors - 1
        n_A_S = self.n_actors * (self.n_actors - 1)

        # 直接复用原有的目标函数
        def objective_speaker_initial(params):
            beta, eta1 = np.concatenate(([0.0], params[:-1])), params[-1]
            weights = np.transpose(stats['speaker_initial_counts'], axes=(1, 0))   # [x_onehot_init, s_init]
            masks = (weights > 0)
            logits = beta[None, :] + eta1*self.X_arr[:, :]
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)   # log-softmax
            loss = - np.sum(weights[masks] * log_probs[masks])
            return loss

        def objective_speaker_transition(params):
            A_S_mat = np.zeros((self.n_actors, self.n_actors))  # 对角线强制为0
            A_S_mat[mask_offdiag] = params[:-1]
            eta2 = params[-1]
            weights = np.transpose(stats['speaker_transition_counts'], axes=(2, 0, 1)) # [x_onehot_curr, s_prev, s_curr]
            mask = (weights > 0)
            logits = A_S_mat[None, :, :] + eta2 * self.X_arr[:, None, :]
            log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
            loss = - np.sum(weights[mask] * log_probs[mask])
            return loss

        # 定义联合目标函数
        def joint_objective(params):
            # 拆分参数
            beta_params = params[:n_beta]
            eta_param = params[n_beta]
            A_S_params = params[n_beta+1:n_beta+1+n_A_S]
            # 组装各自的参数
            initial_params = np.concatenate([beta_params, [eta_param]])
            transition_params = np.concatenate([A_S_params, [eta_param]])
            # 目标函数之和
            return objective_speaker_initial(initial_params) + objective_speaker_transition(transition_params)
        
        # 拼接所有参数
        assert self.eta1_ == self.eta2_, "When using joint optimization, eta1 and eta2 must be equal."
        x0 = np.concatenate([
            self.beta_[1:],                # beta (去掉第一个)
            np.array([self.eta1_]),        # eta
            self.A_S_[mask_offdiag],       # A_S 非对角线
        ])

        # 优化
        result = minimize(joint_objective, x0, method='L-BFGS-B')
        obj_init = joint_objective(x0)
        obj_final = joint_objective(result.x)

        if result.success or obj_final < obj_init:
            self.beta_ = np.concatenate(([0.0], result.x[:n_beta]))
            self.eta1_ = result.x[n_beta]
            self.A_S_ = np.zeros((self.n_actors, self.n_actors))
            self.A_S_[mask_offdiag] = result.x[n_beta+1:n_beta+1+n_A_S]
            self.eta2_ = result.x[n_beta]
        else:
            print("Warning: Joint optimization did not converge.")
            print(f"Initial joint objective value: {obj_init:.4f}")
            print(f"Final joint objective value: {obj_final:.4f}")

    def score(self, S_hat_onehot, X_onehot, lengths=None):
        """计算观测序列的对数似然"""
        S_hat_onehot = np.array(S_hat_onehot)
        X_onehot = np.array(X_onehot)

        # EM算法总以M步结束，为了确保计算最新的对数似然，这里重新计算一次E步        
        return self._do_estep(S_hat_onehot, X_onehot, lengths)['log_likelihood']


    def predict_proba(self, S_hat_onehot, X_onehot, lengths=None):
        """
        计算给定观测序列时说话人隐藏状态的后验概率 $\lambda_{i,t,\\varrho} = \\bbP(S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)})$
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        X_onehot : array-like, shape (n_samples, n_actors)
            观测的X状态，one-hot编码        
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
        X_onehot = np.array(X_onehot)
        self._check_and_set_n_features(S_hat_onehot, X_onehot)
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
            seq_X_onehot = X_onehot[start_idx:end_idx]
            
            ## 计算前向和后向概率，以及序列的对数似然
            fwd_lattice = self._do_forward_pass(seq_S_hat_onehot, seq_X_onehot)
            bwd_lattice = self._do_backward_pass(seq_S_hat_onehot, seq_X_onehot)
            seq_loglik = logsumexp(fwd_lattice[-1])
            
            ## 计算每个时刻的后验概率
            for t in range(length):
                ### 获取并存储联合后验概率 P(S_t=s | 全部观测)
                log_gamma = fwd_lattice[t] + bwd_lattice[t] - seq_loglik
                gamma = np.exp(log_gamma)
                speaker_posteriors[start_idx + t] = gamma # of shape (n_actors)
            
            start_idx = end_idx
        
        return {
            'speaker_states': speaker_posteriors, 
        }

    def predict(self, S_hat_onehot, X_onehot, lengths=None):
        """
        使用Viterbi算法，预测最可能的隐藏状态序列(说话人状态)
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        X_onehot : array-like, shape (n_samples, n_actors)
            观测的X状态，one-hot编码
        lengths : array-like of integers, optional
            每个序列的长度，如果为None，则假设是单一序列

        Returns
        -------
        speaker_states : array, shape (n_samples,)
            预测的说话人状态序列 (0到n_actors-1)
        """
        S_hat_onehot = np.asarray(S_hat_onehot)
        X_onehot = np.asarray(X_onehot)
        self._check_and_set_n_features(S_hat_onehot, X_onehot)
        lengths = self._validate_lengths(S_hat_onehot, lengths)
        
        # 初始化输出数组
        speaker_states = np.zeros(S_hat_onehot.shape[0], dtype=int)
        
        start_idx = 0
        for seq_len in lengths:
            end_idx = start_idx + seq_len
            
            # 提取当前序列
            seq_S_hat_onehot = S_hat_onehot[start_idx:end_idx]
            seq_X_onehot = X_onehot[start_idx:end_idx]

            # 使用维特比算法预测
            seq_speaker_states = self._viterbi(seq_S_hat_onehot, seq_X_onehot)
            
            # 存储结果
            speaker_states[start_idx:end_idx] = seq_speaker_states
            
            start_idx = end_idx
            
        return speaker_states
        
    def _viterbi(self, S_hat_onehot, X_onehot):
        """
        对单个序列使用维特比算法进行解码
        
        Parameters
        ----------
        S_hat_onehot : array-like, shape (n_frames, n_actors)
            说话人观测序列
        X_onehot : array-like, shape (n_frames, n_actors)    
        观测的X状态序列
        
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
        log_speaker_probs = self._compute_speaker_initial_probs(X_onehot[0])  # shape (n_actors,)
        ## 计算对数观测概率 P(S_hat | S)
        log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[0])
        viterbi[0, :] = log_speaker_probs + log_speaker_emissions
        
        # 前向传播 t=1到n_frames-1
        for t in range(1, n_frames):
            active_x = self.X2index(X_onehot[t])    # one-hot to index
            ## 计算当前时刻每个隐藏状态对应的观测概率 P(S_hat_t | S_t)
            log_speaker_emissions = self._compute_emission_probs(S_hat_onehot[t])

            # 计算每个当前状态对应的 $\delta_{t}(s)$
            for speaker in range(self.n_actors):
                # 遍历所有可能的前一状态 (s_prev)，确定最佳前一状态
                total_prob_prev_no_obs = viterbi[t-1, :] + self._log_trans_speaker[:, speaker, active_x]
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
