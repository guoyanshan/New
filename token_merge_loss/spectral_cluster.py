import numpy as np
import torch
import torch.nn.functional as F
from kmeans import MultiKMeans

def spectral_cluster(attn_maps, patch_tokens, K=10, neighbor_mask=None, pre_labels=None):
    """
      Parameters
        attn_maps: Tensor (*,n_samples,n_samples)
            Attention map from Transfomrer as similarity matrix
        
        K: int
            Number of clusters, default: 10
        
        neighbor_mask: Tensor (n_samples,n_samples)
            Mask to reserve neighbors only

        pre_labels: Tensor (*,n_samples_pre)
            Label(Index of cluster) of data points of last module

      Returns
        labels:
            ['normal'] - Tensor (*,n_samples)
            ['debug'] - Tensor (len(K_trials),*,n_samples)
            Label(Index of cluster) of data points
    """

    batched = False
    if attn_maps.ndim == 3:  # Batched data
        B, N, _ = attn_maps.shape
        batched = True
    else:
        B = 2
        N = attn_maps.shape[0]
    K_1 = K

    # 1. Generate similarity matrix
    if neighbor_mask is None:
        neighbor_mask = Cal_Matrix(attn_maps, patch_tokens)

    sim_mat = (attn_maps + attn_maps.transpose(-2, -1)) * neighbor_mask
    sim_mat = torch.softmax(sim_mat, dim=-1)

    # 2. Compute degree matrix

    # 3. Laplacian Matrix and Normalized Laplacian Matrix
    normalized_laplacian_mat, diag_term = graph_laplacian(sim_mat)  # (*,N,N), (*,N)

    # 4. Top K_1 eigen vector with respect to eigen values
    eig_values, eig_vectors = torch.linalg.eigh(
        normalized_laplacian_mat)  # Eigen value decomposition of of a complex Hermitian or real symmetric matrix.
    # eigenvalues will always be real-valued, even when A is complex. It will also be ordered in ascending order.
    if batched:
        feat_mat = eig_vectors[:, :, :K_1]  # (B,N,K_1)
    else:
        feat_mat = eig_vectors[:, :K_1]  # (N,K_1)

    # 5. KMeans Cluster
    if batched:
        kmeans = MultiKMeans(n_clusters=K, n_kmeans=B, max_iter=100)
        labels = kmeans.fit_predict(feat_mat)  # (B,N)
        return labels  # (B,N)


def graph_laplacian(affinity: torch.Tensor, normed=True):
    batched = True
    if affinity.ndim == 3:  # Batched data
        B, N, _ = affinity.shape
        batched = True
    else:
        B = 2
        N = affinity.shape[0]

    if batched:
        torch.diagonal(affinity, dim1=-2, dim2=-1)[...] = 0  # (B,N)
        diag = affinity.sum(dim=-2)  # (B,N)
        if normed:
            mask = (diag == 0)  # mask of isolated node (B,N)
            diag = torch.where(mask, 1., torch.sqrt(diag).to(torch.double)).to(diag.dtype)  # (B,N)

            affinity /= diag.unsqueeze(-2)  # Row
            affinity /= diag.unsqueeze(-1)  # Col

            affinity *= -1
            # torch.diagonal(affinity,dim1=-2,dim2=-1)[...] = 1 - mask.float()
            torch.diagonal(affinity, dim1=-2, dim2=-1)[...] = 1  # (B,N)
        else:
            affinity *= -1
            torch.diagonal(affinity, dim1=-2, dim2=-1)[...] = diag
    else:
        # Non-batched
        affinity.fill_diagonal_(0)  # (N,N) symmetric matrix
        diag = affinity.sum(dim=-2)  # (N,)
        if normed:
            mask = (diag == 0)  # mask of isolated node
            diag = torch.where(mask, 1., torch.sqrt(diag).to(torch.double)).to(diag.dtype)
            affinity /= diag
            affinity /= diag[:, None]

            affinity *= -1
            # affinity.flatten()[::len(mask)+1] = 1 - mask.float()
            affinity.flatten()[::len(mask) + 1] = 1
        else:
            affinity *= -1
            affinity.flatten()[::len(diag) + 1] = diag

    return affinity, diag


# def get_neighbor_mask(N):
#     """
#         neighbor: 8
#     """
#     P = int(N ** (0.5))
#     A = torch.zeros((N, N))
#     ind = torch.arange(N)  # 0——N-1
#     row = torch.div(ind, P, rounding_mode='floor')
#
#     # Same row
#     # ind + 1
#     neigbor_ind = ind + 1
#     neighbor_row = torch.div(neigbor_ind, P, rounding_mode='floor')
#     mask = (neigbor_ind < N) & (row == neighbor_row)
#     A[ind[mask], neigbor_ind[mask]] = 1
#     # ind - 1
#     neigbor_ind = ind - 1
#     neighbor_row = torch.div(neigbor_ind, P, rounding_mode='floor')
#     mask = (neigbor_ind >= 0) & (row == neighbor_row)
#     A[ind[mask], neigbor_ind[mask]] = 1
#     # exit()
#
#     # stride = [-(P+1),-P,-(P-1),-1]
#     strides = [P - 1, P, P + 1]
#
#     for s in strides:
#         # ind + s
#         neigbor_ind = ind + s
#         neigbor_row = torch.div(neigbor_ind, P, rounding_mode='floor') - 1
#         mask = (neigbor_ind < N) & (row == neigbor_row)
#         A[ind[mask], neigbor_ind[mask]] = 1
#         # ind - s
#         neigbor_ind = ind - s
#         neigbor_row = torch.div(neigbor_ind, P, rounding_mode='floor') + 1
#         mask = (neigbor_ind >= 0) & (row == neigbor_row)
#         A[ind[mask], neigbor_ind[mask]] = 1
#
#     return A

# 计算权重
def calculate_w_ij(a, b, sigma=1):
    w_ab = torch.exp(-torch.sum((a - b) ** 2) / (2 * sigma ** 2))
    return w_ab


# 计算邻接矩阵
def Cal_Matrix(attn_maps, patch_tokens, use_gpu=True):
    batched = False
    if attn_maps.ndim == 3:  # Batched data
        B, N, _ = attn_maps.shape
        batched = True
    else:
        B = 2
        N = attn_maps.shape[2]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i != j):
                W[i][j] = calculate_w_ij(patch_tokens[i], patch_tokens[j])
            else:
                W[i][j] = 0
    if use_gpu:
        W = torch.from_numpy(W)
        W = W.cuda()
    return W


def cluster_reduce(feats, K, use_gpu=True):
    B, N, D = feats.shape  # feats: (B,N,D)

    M = torch.zeros(B, K, N)
    # B_ind = torch.arange(B).view(-1, 1).expand(-1, N)  # (B,N)
    # N_ind = torch.arange(N).view(1, -1).expand(B, -1)  # (B,N)

    M[:, :, :] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=-1)
    if use_gpu:
        M = M.cuda()
    result = torch.bmm(M, feats)

    return result

# def neighbor_mask_reduce(neighbor_mask, labels, K):
#     B, N = labels.shape
#     if neighbor_mask.ndim == 2:
#         neighbor_mask = neighbor_mask.contiguous().view(1, N, N).expand(B, -1, -1)
#
#     M = torch.zeros(B, K, N)
#     B_ind = torch.arange(B).view(-1, 1).expand(-1, N)  # (B,N)
#     N_ind = torch.arange(N).view(1, -1).expand(B, -1)  # (B,N)
#
#     M[B_ind, labels, N_ind] = 1
#     neighbor_mask = torch.bmm(M, neighbor_mask)  # (B,K,N)
#     neighbor_mask = torch.bmm(neighbor_mask, M.transpose(-2, -1))  # (B,K,K)
#     #  Clear Diagonal
#     neighbor_mask.flatten(1)[:, ::K + 1] = 0
#     return (neighbor_mask > 0).float()
