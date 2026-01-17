import os, sys
import numpy as np
# Add parent directory to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger import get_logger


logger = get_logger()


def is_square_matrix(mat: np.ndarray):
    return mat.shape[0] == mat.shape[1]


def is_dimensions_same(mat1: np.ndarray, mat2: np.ndarray):
    return mat1.shape == mat2.shape


def affinity_matrix_refinement(affinity_mat, propagated_mat):
    """
        Get $\hat{\mathcal{A}}$, the refined affinity matrix by incorporating the propagated constraints matrix $\hat{\mathcal{Z}}$. (described in Eq.6 of the original paper)
    """
    assert is_square_matrix(affinity_mat)
    assert is_square_matrix(propagated_mat)
    assert is_dimensions_same(affinity_mat, propagated_mat)

    refined_matrix = np.zeros_like(affinity_mat)
    n = affinity_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if propagated_mat[i, j] >= 0:
                refined_matrix[i, j] = 1 - (1 - propagated_mat[i, j]) * (1 - affinity_mat[i, j])
            else:
                refined_matrix[i, j] = (1 + propagated_mat[i, j]) * affinity_mat[i, j]
    return refined_matrix


class BasicPropagation(object):

    def propagate(self, affinity_mat, constraints_mat) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, affinity_mat, constraints_mat):
        propagated_mat = self.propagate(affinity_mat, constraints_mat)
        result_mat = affinity_matrix_refinement(affinity_mat, propagated_mat)
        return result_mat


class E2CPPropagation(BasicPropagation):
    """
        E2CP Propagation and affinity matrix refinement.
        Calculate propagated constraints $\hat{Z}$ from binarized intergrated constraint score(Z'), then refine the affinity matrix $\matchcal{A}$ to get $\hat{\matchcal{A}}$.
        Args:
            alpha: float, the propagation parameter(lambda in original paper)
            knn_k: int, the k value for knn graph construction. If -1, use full graph. Used in laplacian computation for affinity_mat $\matchcal{A}$.
            temperature: float, the temperature parameter for affinity computation. Used in laplacian computation for affinity_mat $\matchcal{A}$.
    """

    def __init__(self, alpha, knn_k=0, temperature=1.0):
        self.alpha = alpha
        self.knn_k = knn_k
        self.temperature = temperature

    def do_knn(self, affinity_mat):
        """
        Perform k-Nearest Neighbors (k-NN) processing on the given affinity matrix.

        This function modifies the input affinity matrix by retaining only the k 
        smallest distances (representing the nearest neighbors) for each row, 
        converting them into RBF (Radial Basis Function) values, and symmetrizing 
        the matrix. If `knn_k` is -1, the original affinity matrix is returned 
        without modification. If `knn_k` is 0, the number of neighbors is 
        determined as `floor(log2(n)) + 1`, where `n` is the number of rows in 
        the matrix.

        Note:
          - The input `affinity_mat` is expected to represent distances, where 
            smaller values indicate closer proximity. This is the opposite of 
            typical cosine similarity matrices, where larger values indicate 
            greater similarity.
          - The function assumes that the diagonal elements of `affinity_mat` 
            are zero or irrelevant, as they represent self-distances.

        Args:
          affinity_mat (np.ndarray): A square matrix representing pairwise 
            distances between data points.

        Returns:
          np.ndarray: The processed affinity matrix with k-NN applied.
        """
        if self.knn_k == -1:
            return affinity_mat
        n = affinity_mat.shape[0]
        if self.knn_k == 0:
            # default use this
            k = int(np.floor(np.log2(n)) + 1)
        else:
            k = self.knn_k
        # get the k smallest values and their indices in each row
        knn_distances = np.sort(affinity_mat, axis=1)[:, 1:k + 1]
        knn_indices = np.argsort(affinity_mat, axis=1)[:, 1:k + 1]
        sigma = np.mean(knn_distances, axis=1)

        # set original k smallest values to RBF values, others to zero
        result_affinity_mat = np.zeros_like(affinity_mat)
        for i in range(n):
            for j in range(k):
                idx = knn_indices[i, j]
                distance = affinity_mat[i, idx]
                result_affinity_mat[i, idx] = np.exp(- (distance ** 2) / (self.temperature * sigma[i] * sigma[idx]))

        # symmetrization
        affinity_mat = (result_affinity_mat + result_affinity_mat.T) / 2.0
        return affinity_mat

    def compute_laplacian_with_knn(self, affinity_mat):
        affinity_mat = self.do_knn(affinity_mat)
        # Compute symmetric normalized Laplacian matrix
        degree = np.diag(np.sum(affinity_mat, axis=1))
        degree_norm = np.diag(1.0 / (np.sqrt(np.diag(degree)) + 1e-10))
        laplacian_matrix = degree_norm.dot(affinity_mat).dot(degree_norm)

        return laplacian_matrix

    def propagate(self, affinity_mat, constraints_matrix):
        """
            Args:
                affinity_mat: np.ndarray
                constraints_matrix: np.ndarray
            Return:
                propagated_constraints_matrix: np.ndarray
        """
        embedding_num = affinity_mat.shape[0]

        laplacian_matrix = self.compute_laplacian_with_knn(affinity_mat)

        coefficient_matrix = np.linalg.inv(np.eye(embedding_num) - self.alpha * laplacian_matrix)
        propagated_matrix = coefficient_matrix.dot(constraints_matrix).dot(coefficient_matrix)
        propagated_matrix = (1 - self.alpha) ** 2 * propagated_matrix

        return propagated_matrix
