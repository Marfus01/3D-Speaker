# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import scipy
import sklearn
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity

import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

try:
    import umap, hdbscan
except ImportError:
    raise ImportError(
        "Package \"umap\" or \"hdbscan\" not found. \
        Please install them first by \"pip install umap-learn hdbscan\"."
        )


class SpectralCluster:
    """
    A spectral clustering method using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    Reference:
    - https://www.cnblogs.com/pinard/p/6221564.html

    Attributes:
      min_num_spks (int): Minimum number of clusters (speakers). Default is 1.
      max_num_spks (int): Maximum number of clusters (speakers). Default is 10.
      pval (float): Pruning parameter to control the sparsity of the similarity matrix. Keep max(N*pval, min_pnum) elements in each row and set others to zero. Default is 0.02.
      min_pnum (int): Minimum number of non-zero elements to retain per row during pruning. Default is 6.
      oracle_num (int or None): If provided, specifies the exact number of clusters (speakers) when getting Spectral Embeddings from the Laplacian matrix.
                    Default is None, which enables automatic estimation of the number of clusters.

    Methods:
      __call__(X, **kwargs):
        Perform spectral clustering on the input embeddings.

        Args:
          X (ndarray): Input embedding matrix of shape [N, D], where N is the number of embeddings
                 and D is the embedding dimension.
          kwargs (dict): Optional parameters:
            - pval (float): Overrides the instance's pval parameter.
            - speaker_num (int): Overrides the instance's oracle_num parameter.

        Returns:
          labels (ndarray): Cluster labels for each embedding, of shape [N].
    """

    def __init__(self, min_num_spks=1, max_num_spks=10, pval=0.02, min_pnum=6, oracle_num=None):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num

    def __call__(self, X, **kwargs):
        pval = kwargs.get('pval', None)
        oracle_num = kwargs.get('speaker_num', None)

        # Cosine similarity matrix(N, N) computation
        sim_mat = self.get_sim_mat(X)

        # Get a sparse version of similarity matrix
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings by RatioCut， and estimate the number of speakers according to gap between eigen-values if oracle_num is None
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num) # emb (N, num_of_spk)

        # Perform k-means clustering based on the Spectral Embeddings 
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval=None):
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0]-self.min_pnum)

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        if k_oracle is None:
            k_oracle = self.k

        # Eigen-decomposition of Laplacian matrix: get the smallest k eigenvalues and corresponding eigenvectors, from small to large
        lambdas, eig_vecs = scipy.sparse.linalg.eigsh(L, k=min(self.max_num_spks+1, L.shape[0]), which='SM')

        # Get/estimate the number of speakers
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            ## 获取最大特征值间隙对应的index，作为估计的说话人数
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks + 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        """
         Perform k-means clustering on the Spectral Embeddings.       
        """
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        """
            Compute the gaps between successive eigenvalues
        """
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    """
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With 
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(self, n_neighbors=20, n_components=60, min_samples=20, min_cluster_size=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X, **kwargs):
        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0]-2),
            metric=self.metric,
        ).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_samples=self.min_samples, min_cluster_size=self.min_cluster_size).fit_predict(umap_X)
        return labels

class AHCluster:
    """
    Agglomerative Hierarchical Clustering, a bottom-up approach which iteratively merges 
    the closest clusters until a termination condition is reached.
    This implementation is adapted from https://github.com/BUTSpeechFIT/VBx.

    Attributes:
      fix_cos_thr (float): Fixed cosine similarity threshold used to determine the stopping criterion 
                 for merging clusters. Default is 0.4.

    Methods:
      __call__(X, **kwargs):
      Perform agglomerative hierarchical clustering on the input embeddings.

      Args:
        X (ndarray): Input feature matrix of shape [N, p].
        kwargs (dict): Optional parameters for customization.

      Returns:
        labels (ndarray): Cluster labels for each embedding, of shape [N].    
    """

    def __init__(self, fix_cos_thr=0.4):
        self.fix_cos_thr = fix_cos_thr

    def __call__(self, X, **kwargs):
        # treat negative cosine similarity as distance
        scr_mx = cosine_similarity(X)
        scr_mx = squareform(-scr_mx, checks=False)  # [N*(N-1)/2, ]
        # 执行层次聚类：使用average linkage计算聚类间的距离。
        ## lin_mat: [N-1, 4], 每一行表示一次聚类操作，包含被聚类的两个簇的索引、它们之间的距离、以及新簇的大小
        lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
        # 调整层次聚类的距离值，使其非负
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        # 根据提前设定的距离阈值，确定层次聚类的停止位置，从而得到最终的聚类结果
        labels = fcluster(lin_mat, -self.fix_cos_thr + adjust, criterion='distance') - 1
        return labels


class CommonClustering:
    """
    Perform clustering for input embeddings and output the labels.
    Attributes:
      cluster_type (str): Clustering method. Options: 'spectral', 'umap_hdbscan', 'AHC'.
      cluster_line (int): The threshold to switch clustering method. If the number of input embeddings < cluster_line, use a simpler method.
      mer_cos (float or None): If provided, merge similar clusters based on cosine similarity threshold.
      min_cluster_size (int): Minimum size of a cluster to be considered a major cluster. Default is 4.
    """

    def __init__(self, cluster_type, cluster_line=40, mer_cos=None, min_cluster_size=4, minor_cluster_cos_thr=0.8, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        self.minor_cluster_cos_thr = minor_cluster_cos_thr
        if self.cluster_type == 'spectral':
            self.cluster = SpectralCluster(**kwargs)
        elif self.cluster_type == 'umap_hdbscan':
            kwargs['min_cluster_size'] = min_cluster_size
            self.cluster = UmapHdbscan(**kwargs)
        elif self.cluster_type == 'AHC':
            self.cluster = AHCluster(**kwargs)
        else:
            raise ValueError(
                '%s is not currently supported.' % self.cluster_type
            )
        if self.cluster_type != 'AHC':
            self.cluster_for_short = AHCluster()
        else:
            self.cluster_for_short = self.cluster

    def __call__(self, X, **kwargs):
        # clustering and return the labels
        assert len(X.shape) == 2, 'Shape of input should be [N, C]'
        if X.shape[0] <= 1:
            return np.zeros(X.shape[0], dtype=int)
        if X.shape[0] < self.cluster_line:
            labels = self.cluster_for_short(X)
        else:
            labels = self.cluster(X, **kwargs)

        # re-assign all samples in extremely minor cluster to the nearest major cluster
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size, self.minor_cluster_cos_thr)
        # merge similar clusters by cosine similarity of their centroids
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)

        return labels

    def filter_minor_cluster(self, labels, x, min_cluster_size, minor_cluster_cos_thr):
        """
        Filters out minor clusters in the given labels and reassigns their points 
        to the nearest major cluster based on cosine similarity.

        Args:
          labels (numpy.ndarray): cluster labels for each data point.
          x (numpy.ndarray): The embedding matrix where each row corresponds to a data point.
          min_cluster_size (int): The minimum size for a cluster to be considered a major cluster.
          minor_cluster_cos_thr (float): The cosine similarity threshold for reassigning points from minor clusters to major clusters.

        Returns:
          labels (numpy.ndarray): The updated array of cluster labels after re-assigning minor clusters.

        Processing Flow:
          1. Identify set of unique cluster labels and calculate the size of each cluster.
          2. Determine minor clusters (clusters with size <= min_cluster_size).
          3. If there are no minor clusters, return the original labels.
          4. If all clusters are minor, return an array of zeros (indicating no valid clusters).
          5. Compute the centroids of the major clusters.
          6. For each data point in a minor cluster, calculate its cosine similarity to the centroids 
             of the major clusters and reassign it to the nearest major cluster.
          7. Return the updated labels.

        Notes:
          - This function assumes that `x` is a 2D array where rows represent data points 
            and columns represent features.
        """
        # get index of minor clusters
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize <= min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels

        minor_cset = cset[minor_idx]
        major_idx = np.where(csize > min_cluster_size)[0]
        if len(major_idx) == 0:
            return np.zeros_like(labels)
        
        # get center of major clusters
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) \
            for i in major_cset])
        
        # re-assign minor cluster points to the nearest major cluster
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                if cos_sim.max() > minor_cluster_cos_thr:
                    labels[i] = major_cset[cos_sim.argmax()]

        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        """
        Iteratively merge similar clusters based on cosine similarity.

        Args:
          labels (np.ndarray): cluster labels for each data point.
          x (np.ndarray): feature matrix of shape [N, p].
          cos_thr (float): The cosine similarity threshold for merging clusters. 
                   Must be in the range (0, 1].

        Returns:
          np.ndarray: Updated cluster labels after merging similar clusters.

        Process:
          1. Ensure the cosine similarity threshold is within the valid range.
          2. Repeat the following steps until no more clusters can be merged:
             a. Identify unique cluster labels with more than one member.
             b. Compute the cluster centers for each unique label.
             c. Calculate the cosine similarity matrix between cluster centers.
             d. Find the pair of clusters with the highest cosine similarity.
             e. If the highest similarity is below the threshold, stop merging.
             f. Otherwise, merge the two clusters by updating their labels.
          3. Return the updated cluster labels.
        """
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            cset = np.unique(labels)
            csize = np.array([(labels == i).sum() for i in cset])
            major_cset = cset[csize > 1]
            if len(major_cset) == 1:
                break
            # compute cosine similarity between cluster centers
            centers = np.stack([x[labels == i].mean(0) \
                for i in major_cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1) # set diagonal and lower triangle to 0
            # find the most similar cluster pair
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = major_cset[np.array(idx)]
            labels[labels==c2]=c1
        return labels


class JointClustering:
    """
    Perform joint clustering for input audio and visual embeddings and output the labels.
    """

    def __init__(self, audio_cluster, vision_cluster):
        self.audio_cluster = audio_cluster
        self.vision_cluster = vision_cluster

    def __call__(self, audioX, visionX, audioT, visionT, conf):
        # audio-only and video-only clustering
        alabels = self.audio_cluster(audioX)
        vlabels = self.vision_cluster(visionX)

        # adjust audio labels to continuous integers starting from 0
        alabels = self.arrange_labels(alabels)
        # convert vision cluster labels to visual segments, each element is [st, ed, visual_spk_id], valid visual speaker durations and their averaged audio embeddings
        vlist, vspk_embs, vspk_dur = self.get_vlist_embs(audioX, alabels, vlabels, audioT, visionT, conf)

        # modify alabels according to vlabels
        aspk_num = alabels.max()+1  # number of audio clusters
        for i in range(aspk_num):
            ## get audio segment indices and embeddings for current audio speaker ID
            aspki_index = np.where(alabels==i)[0]
            aspki_embs = audioX[alabels==i]

            ## get time stamps of audio segments for current audio speaker ID, and then merge overlapping segments
            aspkiT_part = np.array(audioT)[alabels==i]  # [n, 2]
            ## Get non-overlapping audio time intervals of length m(m<=n) first, then identify visual speaker IDs that highly overlap with current audio speaker ID.
            overlap_vspk = self.overlap_spks(self.cast_overlap(aspkiT_part), vlist, vspk_dur) # list of len k'
            
            ## allocate audio segments to the most similar visual speaker, and modify alabels to corresponding visual speaker IDs(beginning from aspk_num)
            if len(overlap_vspk) > 1:
                ### look up meaned audio embeddings of these overlapping visual speakers
                centers = np.stack([vspk_embs[s] for s in overlap_vspk])  # [k', p]
                distribute_labels = self.distribute_embs(aspki_embs, centers)
                for j in range(distribute_labels.max()+1):
                    for loc in aspki_index[distribute_labels==j]:
                        alabels[loc] = overlap_vspk[j]
            elif len(overlap_vspk) == 1:
                for loc in aspki_index:
                    alabels[loc] = overlap_vspk[0]

        # adjust audio labels to continuous integers starting from 0
        alabels = self.arrange_labels(alabels)
        return alabels

    def overlap_spks(self, times, vlist, vspk_dur=None):
        """
        Identify overlapping speakers based on time intervals.

        Args:
          times (list): Elements are non-overlapping audio speaker time intervals [start_time, end_time].
          vlist (list): Elements are time intervals with visual speaker IDs, like [start_time, end_time,  visual_spk_id].
          vspk_dur (dict, optional): Mapping some visual speaker IDs to their total visual active speaking duration. If provided, it is used to set a dynamic overlap threshold.

        Returns:
          vspk_list (list): Elements are visual speaker IDs that overlapping with the given time intervals > 50% of their total visual active speaking duration or > 0.5s.
        """
        # get the vspk that highly overlaps with audio.
        ## 遍历所有 audio-visual segment pairs, 筛选两者有重叠的 segment pairs, 并统计每个 visual speaker 的重叠时长overlap_dur
        overlap_dur = {}
        for [a_st, a_ed] in times:
            for [v_st, v_ed, v_id] in vlist:
                if a_ed > v_st and v_ed > a_st:
                    if v_id not in overlap_dur:
                        overlap_dur[v_id]=0
                    overlap_dur[v_id] += min(a_ed, v_ed) - max(a_st, v_st)
        vspk_list = []
        # 对overlap_dur的keys做筛选。在提供了vspk_dur的情况下，筛选条件是重叠时长>0.5或占visual segment总时长的比例>0.5
        for v_id, dur in overlap_dur.items():
            # set the criteria for confirming overlap.
            if (vspk_dur is None and dur > 0.5) or (vspk_dur is not None and dur > min(vspk_dur[v_id]*0.5, 0.5)): # min 改成 max 效果会变差
                vspk_list.append(v_id)
        return vspk_list

    def distribute_embs(self, embs, centers):
        """
        Distributes audio embeddings to the closest center based on cosine similarity.

        Args:
          embs (np.ndarray): A 2D array of shape [n, D], where n is the number of audio embeddings 
                     and D is the dimensionality of each embedding.
          centers (np.ndarray): A 2D array of shape [k, D], where k is the number of center vectors 
                      and D is their dimensionality.

        Returns:
          np.ndarray: A 1D array of shape [n], where each element is the index of the center 
                that the corresponding embedding is most similar to.

        Process:
          1. Compute the cosine similarity between each embedding and each center.
          2. Assign each embedding to the center with the highest similarity.
        """
        ## 计算每个audio embedding与各个中心向量的余弦相似度
        norm_centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        similarity = np.matmul(norm_embs, norm_centers.T) # [n, k]
        ## 将每个audio embedding分配给与之最相似的中心向量
        argsort = np.argsort(similarity, axis=-1)
        return argsort[:, -1]

    def get_vlist_embs(self, audioX, alabels, vlabels, audioT, visionT, conf):
        """
        Processes and aligns audio and visual data to generate speaker embeddings, 
        adjusted vision labels, and speaker durations.

        Args:
          audioX (list or np.ndarray): Audio embeddings corresponding to audio segments.
          alabels (np.ndarray): audio clustering labels.
          vlabels (list or np.ndarray): Vision clustering labels.
          audioT (list of [float, float]): Start and end times of audio segments.
          visionT (list of float): Timestamps of visual frames.

        Returns:
          tuple: A tuple containing:
            - vlist_new (list): Elements are visual segments with adjusted labels, like [st, ed, visual_spk_id]. st, ed are start and end times of a continuous segment with the same visual spk_id(continuous integers starting from max audio label + 1).
            - vspk_embs (dict): Map visual speaker IDs to their averaged audio embeddings(from those audio segments with overlapping duration > 1s).
            - vspk_dur (dict): Mapping visual speaker IDs to their total visual active speaking duration.
        """
        assert len(vlabels) == len(visionT)
        # 根据视频帧的时间戳以及该帧active speaker face(visual speaker)的聚类标签，生成一个vlist，记录高质量的同一visual speaker连续出现时间段.
        vlist = []
        for i, ti in enumerate(visionT):
            ## len(vlist)==0: 初始化
            ## vlabels[i] != vlist[-1][2]: 当前帧的聚类标签与上一个连续段的标签不同，说明是一个新的连续段
            ## ti - visionT[i-1] > conf.face_det_stride*0.04 + 1e-4: 当前时间戳与前一个时间戳时间间隔过大，说明中间有至少一个detected frame没有提取出active speaker face的embedding，视为新的连续段
            if len(vlist)==0 or vlabels[i] != vlist[-1][2] or ti - visionT[i-1] > conf.face_det_stride*0.04 + 1e-4:
                ## 如果前一个连续时间段只包含一帧，认为置信度较低，将其从vlist中移除
                if len(vlist) > 0 and vlist[-1][1] - vlist[-1][0] < 1e-4:
                    # remove too short intervals. 
                    vlist.pop()
                vlist.append([ti, ti, vlabels[i]])
            else:
                vlist[-1][1] = ti

        # adjust vision labels in vlist to continuous integers starting from max audio label + 1
        vlabels_arrange = self.arrange_labels([i[2] for i in vlist], a_st=alabels.max()+1)
        vlist = [[i[0], i[1], j] for i, j in zip(vlist, vlabels_arrange)]

        # get audio spk embs aligning with 'vlist'
        vspk_embs = {}
        ## 遍历所有active speaker face segments，定位与face segments重叠时长>1s的所有audio segments。随后，将这些audio segments的embedding的均值向量归类到对应的face segments所属的视觉说话人类别中。
        for [v_st, v_ed, v_id] in vlist:
            for i, [a_st, a_ed] in enumerate(audioT):
                if a_ed >= v_st and v_ed >= a_st:
                    if min(a_ed, v_ed) - max(a_st, v_st) > 1:
                        if v_id not in vspk_embs:
                            vspk_embs[v_id] = []
                        vspk_embs[v_id].append(audioX[i])
        for k in vspk_embs:
            vspk_embs[k] = np.stack(vspk_embs[k]).mean(0)

        # 从 vlist 中，移除那些vision labels在 vspk_embs 中没有对应说话人embedding的时间段
        vlist_new = []
        for i in vlist:
            if i[2] in vspk_embs:
                vlist_new.append(i)
        # get duration of v_spk: 统计vlist_new中存在的vision labels的总出现时长
        vspk_dur = {}
        for i in vlist_new:
            if i[2] not in vspk_dur:
                vspk_dur[i[2]]=0
            vspk_dur[i[2]] += i[1]-i[0]

        return vlist_new, vspk_embs, vspk_dur

    def cast_overlap(self, input_time):
        """
        Merge overlapping time intervals.

        Args:
          input_time (np.ndarray or list of list of float): Time intervals with shape [n, 2], 
            where each sublist represents a time interval [start_time, end_time].

        Returns:
          output_time (list of list of float): A list of non-overlapping time intervals, of shape [m, 2],
            where m <= n.
        """
        if len(input_time)==0:
            return input_time
        output_time = []
        for i in range(0, len(input_time)-1):
            if i == 0 or output_time[-1][1] < input_time[i][0]:
                output_time.append(input_time[i])
            else:
                output_time[-1][1] = input_time[i][1]
        return output_time

    def arrange_labels(self, labels, a_st=0):
        """
        将聚类标签重新映射为从 a_st 开始的连续编号。
        """
        # arrange labels in order from 0.
        new_labels = []
        labels_dict = {}
        idx = a_st
        for i in labels:
            if i not in labels_dict:
                labels_dict[i] = idx
                idx += 1
            new_labels.append(labels_dict[i])
        return np.array(new_labels)
