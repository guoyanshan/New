from spectral_cluster import *
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from mutual_information_loss import Mutual_Information_Loss

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 将输入映射为Q、K、V三个向量
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
        attn_wo_soft = dots

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn_wo_soft


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 构建depth个Transformer模块
            # 每个模块包含的内容
            self.layers.append(nn.ModuleList([
                # Norm+Attention
                PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                # Norm+MLP
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, return_attention=True):
        for attn, ff in self.layers:
            y, attn_wo_soft = attn(x)
            x = x + y
            y = ff(x)
            x = x + y
            if return_attention:
                return x, attn_wo_soft
        return x


class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


def token_merging(attn_maps, patch_tokens, N, labels=None):
    labels = spectral_cluster(attn_maps, patch_tokens, K=int(N / 2), pre_labels=labels)
    patch_tokens = cluster_reduce(patch_tokens, int(N / 2))
    return patch_tokens, labels


class ViT(nn.Module):
    def __init__(self, *, ms_patch_size, pan_patch_size, num_patches, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', ms_channels, pan_channels, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.to_patch_embedding1 = SPT(dim=dim, patch_size=ms_patch_size, channels=ms_channels)
        self.to_patch_embedding2 = SPT(dim=dim, patch_size=pan_patch_size, channels=pan_channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.model_diff = Mutual_Information_Loss(dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img1, img2, phase='train'):
        x1 = self.to_patch_embedding1(img1)
        b, n, _ = x1.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x1 += self.pos_embedding[:, :(n + 1)]
        x1 = self.dropout(x1)

        x1, attn_wo_soft1 = self.transformer(x1)
        tokens, labels = token_merging(attn_wo_soft1, x1[:, 1:], x1.shape[1] - 1)
        x2 = torch.cat((x1[:, 0:1], tokens), dim=-2)
        x2, attn_wo_soft1 = self.transformer(x2)

        y1 = self.to_patch_embedding2(img2)
        b, n, _ = y1.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        y1 = torch.cat((cls_tokens, y1), dim=1)
        y1 += self.pos_embedding[:, :(n + 1)]
        y1 = self.dropout(y1)

        y1, attn_wo_soft2 = self.transformer(y1)
        tokens, labels = token_merging(attn_wo_soft2, y1[:, 1:], y1.shape[1] - 1)
        y2 = torch.cat((y1[:, 0:1], tokens), dim=-2)
        y2, attn_wo_soft2 = self.transformer(y2)

        out = []
        if phase == 'train':
            # loss
            diff_loss = self.model_diff(y, x)
            out.append(diff_loss)

        output = torch.cat([x, y], 1)  # 通道拼接
        output = output.mean(dim=0) if self.pool == 'mean' else output[:, 0]

        output = self.to_latent(output)
        output = self.mlp_head(output)
        out.append(output)
        return out


def VIT():
    return ViT(ms_patch_size=16, pan_patch_size=64, num_patches=2652, num_classes=8, dim=64, depth=6, heads=8,
               mlp_dim=2048, ms_channels=4, pan_channels=1)

# def test():
#     net = VIT()
#     s = net(torch.randn(20, 4, 16, 16), torch.randn(20, 1, 64, 64))
#     print(s.size())
# test()
