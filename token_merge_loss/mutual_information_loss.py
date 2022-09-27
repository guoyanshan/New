import torch.nn as nn
import torch
import torch.nn.functional as F


class Mutual_Information_Loss(nn.Module):
    def __init__(self, dim):
        super(Mutual_Information_Loss, self).__init__()
        pan_cls_token_width, embed_dim = dim, 64
        ms_cls_token_width, embed_dim = dim, 64
        self.pan_cls_token_proj = nn.Linear(pan_cls_token_width, embed_dim)
        self.ms_cls_token_proj = nn.Linear(ms_cls_token_width, embed_dim)

    def forward(self, pan_patch_embedding, ms_pacth_embedding):
        pan_cls_feat = F.normalize(self.pan_cls_token_proj(pan_patch_embedding[:, 0, :]))
        ms_cls_feat = F.normalize(self.ms_cls_token_proj(ms_pacth_embedding[:, 0, :]))
        pan_cls_entropy = Entropy(pan_cls_feat).cuda()
        ms_cls_entropy = Entropy(ms_cls_feat).cuda()
        multi_model_joint_entropy = Joint_entropy(pan_cls_entropy, ms_cls_entropy).cuda()
        mutual_information_loss =F.smooth_l1_loss(torch.add(pan_cls_entropy, ms_cls_entropy), multi_model_joint_entropy, reduction='mean')
        return mutual_information_loss
        # mutual_information = torch.sub(torch.add(pan_cls_entropy, ms_cls_entropy), multi_model_joint_entropy)
        # mutual_information_loss = mutual_information
        # return mutual_information_loss


def Norm(x):
    max_val_t = torch.max(x, 2)[0]
    max_val = torch.max(max_val_t, 2)[0]

    min_val_t = torch.min(x, 2)[0]
    min_val = torch.min(min_val_t, 2)[0]

    delta_t1 = torch.sub(max_val, min_val)
    delta_t2 = torch.unsqueeze(delta_t1, 2)
    delta = torch.unsqueeze(delta_t2, 3)

    min_val_t1 = torch.unsqueeze(min_val, 2)
    min_val = torch.unsqueeze(min_val_t1, 3)

    rel_t1 = torch.sub(x, min_val)
    rel_t2 = torch.div(rel_t1, delta)
    rel = torch.mul(rel_t2, 255).int()
    return rel


def Entropy(x):
    B, C, (W, H) = 1, 1, x.size()
    size = W * H
    histic = torch.zeros(size=(B, C, 256))
    for i in range(256):
        eq_i = torch.eq(x, i)
        sum_t1 = torch.sum(eq_i, dim=1)
        sum = torch.sum(sum_t1, dim=0)
        histic[:, :, i] = sum
    p_ij = torch.div(histic, size)
    h_ij_t1 = torch.add(p_ij, 1e-8)
    h_ij_t2 = p_ij * torch.log(h_ij_t1)
    h_ij = -torch.sum(h_ij_t2, dim=1)
    return torch.unsqueeze(torch.unsqueeze(h_ij, 2), 3)


def Joint_entropy(x_p, x_ms):
    # B, C, H, W = 1, x_ms.size()
    B = 1
    C = x_ms.shape[0]
    H = x_ms.shape[1]
    W = x_ms.shape[2]
    temp = torch.randint(low=4, high=5, size=(B, C, W, H)).cuda()
    histic_ms_p = torch.zeros(size=(B, C, 256, 256))
    for i in range(256):
        for j in range(256):
            eq_i_t1 = torch.eq(x_ms, i).long()
            eq_i = torch.add(eq_i_t1, 1)

            eq_j_t1 = torch.eq(x_p, j).long()
            eq_j = torch.add(eq_j_t1, 1)

            eq_ms = torch.where(eq_i == 2, eq_i, temp)
            eq_p = torch.where(eq_j == 2, eq_j, temp)
            eq_ij = torch.eq(eq_ms, eq_p)

            sum_t1 = torch.sum(eq_ij, dim=1)
            sum = torch.sum(sum_t1, dim=2)

            histic_ms_p[:, :, i, j] = sum

    p_ms_p = torch.div(histic_ms_p, 256 * 256)
    h_ms_p_t1 = torch.add(p_ms_p, 1e-8)
    h_ms_p_t2 = p_ms_p * torch.log(h_ms_p_t1)
    h_ms_p_t3 = torch.sum(h_ms_p_t2, dim=1)
    h_ms_p = -torch.sum(h_ms_p_t3, dim=1)
    return torch.unsqueeze(torch.unsqueeze(h_ms_p, 2), 3)

