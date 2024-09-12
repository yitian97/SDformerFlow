import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from torch import nn
from models.STSwinNet_SNN.Spiking_modules import *
from models.STSwinNet.swin_transformer3D_v2 import Swin_BasicLayer,get_window_size,window_reverse,window_partition,PatchMerging

from timm.models.layers import DropPath, trunc_normal_
from typing import Tuple
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from spikingjelly.activation_based import layer as sj_layer
from spikingjelly.activation_based import surrogate,neuron


class VanillaAttention(nn.Module):
    def __init__(self, scale=0.125):
        super(VanillaAttention, self).__init__()
        self.scale = scale

    def forward(self, q, k, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        q = q * self.scale
        score = q @ k.transpose(-2, -1)

        return score

class SDAttention(nn.Module):
    def __init__(self):
        super(SDAttention, self).__init__()


    def forward(self, k, v):
        kv = k.mul(v)
        score = kv.sum(dim=-2, keepdim=True)

        return score

class QKAttention(nn.Module):
    def __init__(self):
        super(QKAttention, self).__init__()


    def forward(self, q, k):
        # input is 4 dimension tensor
        # B_, numHead, N, C
        #token attention
        att_token = q.sum(dim=-1, keepdim=True) #264,3,162,32
        score = k.mul(att_token)#264,3,162,1
        #channel attention
        # att_channel = q.sum(dim=-2, keepdim=True)
        # score = k.mul(att_channel)


        return score

class Spiking_QKAttention(nn.Module):
    def __init__(self):
        super(Spiking_QKAttention, self).__init__()
        att_token_sn = Spiking_neuron

    def forward(self, q, k):
        # input is 4 dimension tensor
        # B_, numHead, N, C
        #token attention
        att_token = q.sum(dim=-1, keepdim=True)
        score = k.mul(att_token)
        #channel attention
        # att_channel = q.sum(dim=-2, keepdim=True)
        # score = k.mul(att_channel)


        return score

class HammingDistanceAttention(nn.Module):
    def __init__(self):
        super(HammingDistanceAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        _, _, _, d = k.size()

        # 1. dot product Query with Key^T to compute similarity
        score = 0.5 * (1 + torch.matmul(2 * q - 1, 2 * k.transpose(2, 3) - 1) / d)

        # 2. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        #
        # # 3. multiply with Value
        # v = torch.matmul(score, v)

        return score


def window_partition_v2(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (window_size[0], B*num_windows, window_size[1], window_size[2], C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    # windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(window_size[0],-1, window_size[1],window_size[2],C)
    return windows

class Spiking_Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, norm_layer="BN", act_layer=nn.GELU, drop=0.,
                 **spiking_kwargs):
        super().__init__()

        out_features = out_features or in_features

        hidden_features = hidden_features or in_features
        self.norm_layer = norm_layer
        self.fc1 =  sj_layer.Linear(in_features, hidden_features,bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn1 = SpikingNormLayer(hidden_features, spiking_kwargs["num_steps"], spiking_kwargs["spike_norm"],
                                        v_th=spiking_kwargs["v_th"])
        self.sn1 = Spiking_neuron(**spiking_kwargs)
        self.fc2 = sj_layer.Linear(hidden_features, out_features,bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn2 = SpikingNormLayer(out_features, spiking_kwargs["num_steps"], spiking_kwargs["spike_norm"],
                                        v_th=spiking_kwargs["v_th"])
        self.sn2 = Spiking_neuron(**spiking_kwargs)

        if self.norm_layer in ["LN" , "GN"]:
            self.norm =  SpikingNormLayer(out_features, spiking_kwargs["num_steps"], norm_layer,
                                        v_th=spiking_kwargs["v_th"])

        self.drop1 = sj_layer.Dropout(drop)
        self.drop2 = sj_layer.Dropout(drop)




    def forward(self, x):
        if self.norm_layer in ["LN", "GN"]:
            x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.drop1(x)
        x = self.fc1(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x= self.bn1(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.sn1(x)

        x = self.drop2(x)
        x = self.fc2(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.bn2(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.sn2(x)

        return x

class MS_Spiking_Mlp(Spiking_Mlp):
    def forward(self, x):
        if self.norm_layer in ["LN", "GN"]:
            x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.sn1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x= self.bn1(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)

        x = self.sn2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.bn2(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)


        return x


class Spiking_BN_WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, pretrained_window_size, num_heads, version = "swinv1", qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,norm= None,**spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.version = version
        self.norm_layer= norm
        spiking_kwargs['num_steps'] = self.window_size[0]

        if self.version == 'swinv2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(3, 512, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, num_heads, bias=False))
            # self.cpb_mlp = Spiking_Cpb_Mlp(3, 512, num_heads,  drop=0.0, **spiking_kwargs)

            # get relative_coords_table

            relative_coords_d = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32) #2*Wd-1
            relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_d,relative_coords_h,
                                relative_coords_w])).permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wd-1,2*Wh-1, 2*Ww-1, 3
            if pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
                relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
            else:
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
                relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_coords_table", relative_coords_table)
        if self.version == 'swinv1':
            if spiking_kwargs["neuron_type"] in ["psn", "glif"]:
                self.scale = 1
            else:
                self.scale = qk_scale or head_dim ** -0.5
            self.vanilla_attn = VanillaAttention(self.scale)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                            num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        #
        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # self.qkv_sn = Spiking_neuron(**spiking_kwargs)
        self.linear_q = sj_layer.Linear(dim, dim , bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn_q = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.sn_q = Spiking_neuron(**spiking_kwargs)
        #
        self.linear_k = sj_layer.Linear(dim, dim, bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn_k = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.sn_k =  Spiking_neuron(**spiking_kwargs)
        #
        self.linear_v = sj_layer.Linear(dim, dim, bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn_v = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.sn_v = Spiking_neuron(**spiking_kwargs)
        #
        # self.linear_out = sj_layer.Linear(dim * 3, dim )
        # self.bn_out = SpikingNormLayer(dim ,norm,spiking_kwargs['v_th'])
        # self.sn_out = Spiking_neuron(**spiking_kwargs)

        # self.Ham_attn = HammingDistanceAttention()

        self.attn_drop = sj_layer.Dropout(attn_drop)
        self.attn_sn = Spiking_neuron(**spiking_kwargs)
        self.proj = sj_layer.Linear(dim, dim)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.proj_bn = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.proj_sn = Spiking_neuron(**spiking_kwargs)

        self.proj_drop = sj_layer.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        T ,B_, H,W,C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = self.linear_q(x)  # T ,B_, H,W,C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        q = self.sn_q(q)
        k = self.linear_k(x)  # B_, numHead, N, C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        k = self.sn_k(k)
        v = self.linear_v(x)  # B_, numHead, N, C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            v = self.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        v = self.sn_v(v)
        q,k,v = q.reshape(B_, self.num_heads, -1, C//self.num_heads), k.reshape(B_, self.num_heads, -1, C//self.num_heads), v.reshape(B_, self.num_heads, -1, C//self.num_heads)
        N = q.shape[2]
        if self.version == "swinv1":
            attn = self.vanilla_attn(q, k)
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(
                N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

            # HammingDistance attention
        # attn, store = self.Ham_attn(q, k, v)  # 192 3 200 32

        if self.version == "swinv2":
            # cosine attention
            # attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            #HammingDistance attention
            attn = self.Ham_attn(q,k)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

            # relative_position_bias_table = self.cpb_mlp(self.relative_coords_table.permute(1,0,2,3,4)).view(-1, self.num_heads)
            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)



        attn = attn + relative_position_bias.unsqueeze(0) #B_,n_head,N,N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # print('attn: ', attn.shape, ', v: ', v.shape, ', x: ', x.shape)
        x = (attn @ v).reshape(B_, self.num_heads, T, H, W, C//self.num_heads)
        x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C)
        # x = self.attn_sn(x)
        x = self.proj(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.proj_sn(x).reshape(B_,N,C)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'


    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim  * self.dim * 3
        #bn
        flops += N * self.dim * 3
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        #bn
        flops += N * self.dim
        return flops

    def record_flops(self, nW, N):
        # calculate flops for 1 window with token length of N
        flops_record = {}
        # qkv = self.qkv(x)
        flops_record["q"] = nW * N * self.dim  * self.dim
        flops_record["k"] = nW * N * self.dim * self.dim
        flops_record["v"] = nW * N * self.dim * self.dim
        #bn
        # flops += N * self.dim * 3
        # attn = (q @ k.transpose(-2, -1))
        flops_record["attn"] = nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops_record["attn"] += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops_record["proj"] =  nW * N * self.dim * self.dim
        #bn
        # flops += N * self.dim
        return flops_record

class SDSA_WindowAttention3D(Spiking_BN_WindowAttention3D):
    #proj_sn, qkv(linear,bn,sn),proj,proj_bn,

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        T, B_, H, W, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        x = self.proj_sn(x)
        q = self.linear_q(x)  # T ,B_, H,W,C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        q = self.sn_q(q)
        k = self.linear_k(x)  # B_, numHead, N, C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        k = self.sn_k(k)
        v = self.linear_v(x)  # B_, numHead, N, C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            v = self.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        v = self.sn_v(v)
        q, k, v = q.reshape(B_, self.num_heads, -1, C // self.num_heads), k.reshape(B_, self.num_heads, -1,
                                                                                    C // self.num_heads), v.reshape(B_,
                                                                                                                    self.num_heads,
                                                                                                                    -1,
                                                                                                                    C // self.num_heads)
        N = q.shape[2]
        if self.version == "swinv1":
            # q = q * self.scale
            # attn = q @ k.transpose(-2, -1)
            attn = self.vanilla_attn(q, k)
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(
                N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

            # HammingDistance attention
        # attn, store = self.Ham_attn(q, k, v)  # 192 3 200 32

        if self.version == "swinv2":
            # cosine attention
            # attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            # HammingDistance attention
            attn = self.Ham_attn(q, k)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

            # relative_position_bias_table = self.cpb_mlp(self.relative_coords_table.permute(1,0,2,3,4)).view(-1, self.num_heads)
            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        attn = attn + relative_position_bias.unsqueeze(0)  # B_,n_head,N,N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # print('attn: ', attn.shape, ', v: ', v.shape, ', x: ', x.shape)
        x = (attn @ v).reshape(B_, self.num_heads, T, H, W, C // self.num_heads)
        x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C)
        #x = self.attn_sn(x)
        x = self.proj(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = x.reshape(B_, N, C)
        # x = self.proj_drop(x)
        return x, attn

# class SDSA_QK_WindowAttention3D(nn.Module):
#     #proj_sn, qkv(linear,bn,sn),proj,proj_bn,
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__(*args, **kwargs)
#     #     self.attn_score = SDAttention()
#     def __init__(self, dim, window_size, pretrained_window_size, num_heads, version = "swinv1", qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,norm= None,**spiking_kwargs):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wd, Wh, Ww
#         self.pretrained_window_size = pretrained_window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.version = version
#         self.norm_layer= norm
#         spiking_kwargs['num_steps'] = self.window_size[0]
#
#         if spiking_kwargs["neuron_type"] in ["psn", "glif"]:
#             self.scale = 1
#         else:
#             self.scale = qk_scale or head_dim ** -0.5
#         # self.attn_score = SDAttention()
#
#         self.attn_score = QKAttention()
#         # define a parameter table of relative position bias
#         self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_heads, window_size[0] * window_size[1] * window_size[2], head_dim)))
#         # self.qkv_sn = Spiking_neuron(**spiking_kwargs)
#         self.linear_q = sj_layer.Linear(dim, dim , bias=False)
#         if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#             self.bn_q = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
#         self.sn_q = Spiking_neuron(**spiking_kwargs)
#         #
#         self.linear_k = sj_layer.Linear(dim, dim, bias=False)
#         if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#             self.bn_k = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
#         self.sn_k =  Spiking_neuron(**spiking_kwargs)
#         #
#         # self.linear_v = sj_layer.Linear(dim, dim, bias=False)
#         # if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#         #     self.bn_v = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
#         # self.sn_v = Spiking_neuron(**spiking_kwargs)
#         #
#         # self.linear_out = sj_layer.Linear(dim * 3, dim )
#         # self.bn_out = SpikingNormLayer(dim ,norm,spiking_kwargs['v_th'])
#         # self.sn_out = Spiking_neuron(**spiking_kwargs)
#
#         # self.Ham_attn = HammingDistanceAttention()
#         self.attn_sn = Spiking_neuron(**spiking_kwargs)
#         self.attn_drop = sj_layer.Dropout(attn_drop)
#         self.proj = sj_layer.Linear(dim, dim)
#         if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#             self.proj_bn = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
#         self.proj_sn = Spiking_neuron(**spiking_kwargs)
#
#         self.proj_drop = sj_layer.Dropout(proj_drop)
#
#
#
#
#
#     def forward(self, x, mask=None):
#         """ Forward function.
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, N, N) or None
#         """
#         T, B_, H, W, C = x.shape
#         # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#
#         x = self.proj_sn(x)
#         q = self.linear_q(x)  # T ,B_, H,W,C
#         if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#             q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
#         q = self.sn_q(q)
#         k = self.linear_k(x)  # B_, numHead, N, C
#         if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#             k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
#         k = self.sn_k(k)
#         # v = self.linear_v(x)  # B_, numHead, N, C if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]: v =
#         # self.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2) v = self.sn_v(v) q, k, v = q.reshape(B_,
#         # self.num_heads, -1, C // self.num_heads), k.reshape(B_, self.num_heads, -1, C // self.num_heads),
#         # v.reshape(B_, self.num_heads, -1, C // self.num_heads)
#         q, k = q.reshape(B_, self.num_heads, -1, C // self.num_heads), k.reshape(B_, self.num_heads, -1,
#                                                                                     C // self.num_heads)
#         N = q.shape[2]
#         k = k + self.positional_encoding
#         attn = self.attn_score(q, k)
#
#             # HammingDistance attention
#         # attn, store = self.Ham_attn(q, k, v)  # 192 3 200 32
#
#         # attn = attn + self.positional_encoding.unsqueeze(0)  # B_,n_head,N,N
#
#         # if mask is not None:
#         #     nW = mask.shape[0]
#         #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#         #     attn = attn.view(-1, self.num_heads, N, N)
#
#         attn = self.attn_drop(attn)
#
#         # x = self.proj_sn(x)
#         # print('attn: ', attn.shape, ', v: ', v.shape, ', x: ', x.shape)
#         x = (attn).reshape(B_, self.num_heads, T, H, W, C // self.num_heads)
#         x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C)
#         attn = self.attn_sn(x)
#         x = self.proj(x)
#         if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
#             x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
#         x = x.reshape(B_, N, C)
#         # x = self.proj_drop(x)
#         return x, attn

class Spiking_QK_WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, pretrained_window_size, num_heads, version = "swinv1", qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,norm= None,**spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.version = version
        self.norm_layer= norm
        spiking_kwargs['num_steps'] = self.window_size[0]

        if spiking_kwargs["neuron_type"] in ["psn", "glif"]:
            self.scale = 1
        else:
            self.scale = qk_scale or head_dim ** -0.5
        # self.attn_score = SDAttention()


        # define a parameter table of relative position bias
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_heads, window_size[0] * window_size[1] * window_size[2], head_dim)))
        # self.qkv_sn = Spiking_neuron(**spiking_kwargs)
        self.linear_q = sj_layer.Linear(dim, dim , bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn_q = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.sn_q = Spiking_neuron(**spiking_kwargs)
        #
        self.linear_k = sj_layer.Linear(dim, dim, bias=False)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.bn_k = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.sn_k =  Spiking_neuron(**spiking_kwargs)
        self.sn2_q = Spiking_neuron(**spiking_kwargs)
        #
        # self.linear_v = sj_layer.Linear(dim, dim, bias=False)
        # if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        #     self.bn_v = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        # self.sn_v = Spiking_neuron(**spiking_kwargs)
        #
        # self.linear_out = sj_layer.Linear(dim * 3, dim )
        # self.bn_out = SpikingNormLayer(dim ,norm,spiking_kwargs['v_th'])
        # self.sn_out = Spiking_neuron(**spiking_kwargs)

        # self.Ham_attn = HammingDistanceAttention()
        self.attn_sn = Spiking_neuron(**spiking_kwargs)
        self.attn_drop = sj_layer.Dropout(attn_drop)
        self.proj = sj_layer.Linear(dim, dim)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            self.proj_bn = SpikingNormLayer(dim, self.window_size[0], norm, spiking_kwargs['v_th'])
        self.proj_sn = Spiking_neuron(**spiking_kwargs)

        self.proj_drop = sj_layer.Dropout(proj_drop)





    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        T, B_, H, W, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        x = self.proj_sn(x.float())
        q = self.linear_q(x)  # T ,B_, H,W,C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        q = self.sn_q(q)
        k = self.linear_k(x).float()  # B_, numHead, N, C
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        positional_encoding = self.positional_encoding.reshape(T, 1, H, W, C )
        k = k + positional_encoding
        k = self.sn_k(k)
        # v = self.linear_v(x)  # B_, numHead, N, C if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]: v =
        # self.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2) v = self.sn_v(v) q, k, v = q.reshape(B_,
        # self.num_heads, -1, C // self.num_heads), k.reshape(B_, self.num_heads, -1, C // self.num_heads),
        # v.reshape(B_, self.num_heads, -1, C // self.num_heads)
        # q, k = q.reshape(B_, self.num_heads, -1, C // self.num_heads), k.reshape(B_, self.num_heads, -1,
        #                                                                C // self.num_heads)
        q, k = q.reshape(T, B_, self.num_heads, -1, C // self.num_heads), k.reshape(B_, self.num_heads, -1,
                                                                       C // self.num_heads)
        N = k.shape[2]

        # attn = self.attn_score(q, k)
        att_token = q.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        attn = k.mul(att_token.reshape(B_,self.num_heads,-1,1))
            # HammingDistance attention
        # attn, store = self.Ham_attn(q, k, v)  # 192 3 200 32

        # attn = attn + self.positional_encoding.unsqueeze(0)  # B_,n_head,N,N

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn)

        # x = self.proj_sn(x)
        # print('attn: ', attn.shape, ', v: ', v.shape, ', x: ', x.shape)
        x = (attn).reshape(B_, self.num_heads, T, H, W, C // self.num_heads)
        x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C).float()
        attn = self.attn_sn(x)
        x = self.proj(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = x.reshape(B_, N, C)
        # x = self.proj_drop(x)
        return x, attn


class Spiking_SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (tuple[int]): Window size in pre-training.

    """
    attn_module = Spiking_BN_WindowAttention3D
    mlp_module = Spiking_Mlp
    # window_part_module = window_partition_v2() # window_partition_v2 for Spiking_BN_WindowAttention3D
    def __init__(self, dim, input_resolution, num_heads, window_size=(2, 7, 7), pretrained_window_size =(0, 0, 0), shift_size=(0, 0, 0),
                 mlp_ratio=4., version= "swinv1", qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer="LN", use_checkpoint=False, **spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm_layer = norm_layer
        if self.norm_layer in ["LN", "GN"]:
            self.norm1 = SpikingNormLayer(dim,spiking_kwargs["num_steps"],norm_layer,v_th=spiking_kwargs["v_th"])
            # self.norm2 = SpikingNormLayer(dim, spiking_kwargs["num_steps"], norm_layer,v_th=spiking_kwargs["v_th"])
        self.attn = self.attn_module(
                dim, window_size=self.window_size, pretrained_window_size=pretrained_window_size,
                num_heads=num_heads,
                version=version, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                norm=norm_layer, **spiking_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = self.mlp_module(in_features=dim, hidden_features=mlp_hidden_dim, norm_layer=norm_layer,act_layer=act_layer, drop=drop, **spiking_kwargs)
        self.cnf ="ADD"

    @staticmethod
    def sew_function(x: torch.Tensor, y: torch.Tensor, cnf: str):
        if cnf == "ADD":
            return x + y
        elif cnf == "AND":
            return x * y
        elif cnf == "IAND":
            return x * (1.0 - y)
        else:
            raise NotImplementedError
    def SSA(self, x, mask_matrix, return_attention):
        B, D, H, W, C = x.shape

        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # print('window_size: ', window_size, ', shift_size: ', shift_size)
        if self.norm_layer in ["LN", "GN"]:
            x = self.norm1(x.permute(1, 0, 4, 2, 3)).permute(1, 0, 3, 4, 2) #DBCHW
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition_v2(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        attn_windows, attn_score = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        if return_attention:
            return attn_score
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        # print('attn_windows: ', attn_windows.shape)
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x


    def forward(self, x, mask_matrix, return_attention = False):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.SSA, x, mask_matrix)
        else:
            x = self.SSA(x, mask_matrix,return_attention)
        if return_attention:
            return x


        x = self.sew_function(self.drop_path(x), shortcut, self.cnf)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.mlp(x), x)
        else:

            x = self.sew_function(self.mlp(x.permute(1,0,2,3,4)).permute(1,0,2,3,4), x , self.cnf) #  B D H W C

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        # flops += self.dim * H * W
        # 3DW-MSA/SW-MSA
        nW = H * W // self.window_size[1] // self.window_size[2]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1]*self.window_size[2])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        #bn
        flops += H * W * self.dim * self.mlp_ratio
        flops += H * W * self.dim
        # norm2
        # flops += self.dim * H * W
        return flops

    def record_flops(self):
        flops_record = {}
        H, W = self.input_resolution
        # norm1
        # flops += self.dim * H * W
        # 3DW-MSA/SW-MSA
        nW = H * W // self.window_size[1] // self.window_size[2]
        flops_record["attn"] = self.attn.record_flops(nW ,self.window_size[0] * self.window_size[1]*self.window_size[2])
        # mlp
        flops_record["mlp0"] = H * W * self.dim * self.dim * self.mlp_ratio
        flops_record["mlp1"] = H * W * self.dim * self.dim * self.mlp_ratio
        #bn
        # flops += H * W * self.dim * self.mlp_ratio
        # flops += H * W * self.dim
        # norm2
        # flops += self.dim * H * W
        return flops_record

class MS_Spiking_SwinTransformerBlock3D(Spiking_SwinTransformerBlock3D):
    # attn_module = SDSA_QK_WindowAttention3D
    # attn_module = SDSA_linear_WindowAttention3D
    # attn_module = SDSA_WindowAttention3D
    attn_module = Spiking_QK_WindowAttention3D
    mlp_module = MS_Spiking_Mlp
    # window_part_module = window_partition_v2  # window_partition_v2



class SpikingPatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,  input_resolution, dim, norm_layer="BN", **spiking_kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction =  sj_layer.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = SpikingNormLayer(2 * dim, spiking_kwargs["num_steps"], norm_layer, spiking_kwargs["v_th"])
        self.sn = Spiking_neuron(**spiking_kwargs)


    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H  % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.reduction(x.permute(1, 0, 2, 3, 4))
        x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.sn(x).permute(1, 0, 2, 3, 4)
        return x

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        #bn
        flops += (H // 2) * (W // 2) * self.dim // 2
        return flops

    def record_flops(self):
        flops_record = 0
        H, W = self.input_resolution
        flops_record = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        #bn
        # flops += (H // 2) * (W // 2) * self.dim // 2
        return flops_record

class MS_SpikingPatchMerging(SpikingPatchMerging):
    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C
        x = self.sn(x.permute(1, 0, 2, 3, 4))
        x = self.reduction(x)
        x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(1, 0, 3, 4, 2)

        return x




# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class Spiking_Swin_BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    swin_block_type = Spiking_SwinTransformerBlock3D
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 pretrained_window_size=(1,7,7),
                 mlp_ratio=4.,
                 version="swinv1",
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer='LN',
                 downsample=None,
                 use_checkpoint=False,
                **spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.swin_blocks = nn.ModuleList([
            self.swin_block_type(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                version=version,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                ** spiking_kwargs
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,**spiking_kwargs)


    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.swin_blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x_out = self.downsample(x)
        else:
            x_out = x
        x_out = rearrange(x_out, 'b d h w c -> b c d h w')
        return x_out, x #x before patch merging

    def get_lst_block_attention_scores(self, x):
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for i, blk in enumerate(self.swin_blocks):
            if i < len(self.swin_blocks) - 1:
                x = blk(x, attn_mask)
            else:
                # return attention of the last block
                return blk(x, attn_mask, return_attention=True)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.swin_blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def record_flops(self):
        flops_record = {}
        for i, blk in enumerate(self.swin_blocks):
            flops_record["block"+str(i)] = blk.record_flops()

        if self.downsample is not None:
            flops_record["downsample"] = self.downsample.record_flops()

        return flops_record

class MS_Spiking_Swin_BasicLayer(Spiking_Swin_BasicLayer):
    swin_block_type = MS_Spiking_SwinTransformerBlock3D


class Spiking_SwinTransformer3D_v2(nn.Module):
    swin_layer_type = Spiking_Swin_BasicLayer
    downsample_layer_type = SpikingPatchMerging
    def __init__(self,
                 pretrained=None,
                 pretrained2d=False,
                 arc_type="swinv1",
                 embed_type="PatchEmbedLocal",
                 img_size = (320,480),
                 patch_size=(4, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 pretrained_window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=0.125,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer="BN", # spikeformer norm
                 patch_norm=False,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 norm = None, #norm for res
                 **spiking_kwargs
                 ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.norm_layer = norm_layer
        # split image into non-overlapping patches

        self.patch_embed = eval(embed_type)(img_size=img_size,
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            patch_norm=norm_layer if self.patch_norm else None, norm = norm, spiking_proj = True, **spiking_kwargs)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = sj_layer.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = self.swin_layer_type(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                mlp_ratio=mlp_ratio,
                version=arc_type,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=self.downsample_layer_type if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                **spiking_kwargs)
            self.layers.append(layer)

        # self.num_features = int(embed_dim * 2**(self.num_layers-1))
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features


        if self.norm_layer in ["LN", "GN"]:

            for i_layer in self.out_indices:
                layer = SpikingNormLayer(self.num_features[i_layer], spiking_kwargs["num_steps"], self.norm_layer,v_th=spiking_kwargs["v_th"])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)


    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x) #T
        # B, C, H/2, W/2
        # print(torch.max(x))
        # print(torch.min(x))
        x = self.pos_drop(x)
        x = rearrange(x, 't b c h w -> b c t h w')
        outs = []
        for i, layer in enumerate(self.layers):
            x, out_x = layer(x.contiguous())

            #print('---- ', x.shape)
            #print('---- ', out_x.shape)
            if i in self.out_indices:
                if self.norm_layer in ["LN", "GN"]:
                    norm_layer = getattr(self, f'norm{i}')
                    out_x = norm_layer(out_x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
                _, Ti, Hi, Wi, Ci = out_x.shape
                #print(out_x.shape)
                out = rearrange(out_x, 'n d h w c -> n c d h w')
                outs.append(out)

        return tuple(outs)

    def get_layer_attention_scores(self, x):
        """Forward function."""
        x = self.patch_embed(x) #B, C, 4, H/2, W/2
        x = self.pos_drop(x)

        attns = []
        for i, layer in enumerate(self.layers):

            attn = layer.get_lst_block_attention_scores(x.contiguous())
            if i < len(self.layers) - 1:
                x, out_x = layer(x.contiguous())

            attns.append(attn)



        return attns

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
            #layernorm
            #flops += self.num_features[i] * (self.patches_resolution[0]// (2 ** i) )* (self.patches_resolution[1] // (2 ** i))

        return flops

    def record_flops(self):
        flops_record = {}
        flops_record["patch_embed"] =  self.patch_embed.record_flops()
        for i, layer in enumerate(self.layers):
            flops_record["layer" + str(i)] = layer.record_flops()
            #layernorm
            #flops += self.num_features[i] * (self.patches_resolution[0]// (2 ** i) )* (self.patches_resolution[1] // (2 ** i))

        return flops_record


class MS_Spiking_SwinTransformer3D_v2(Spiking_SwinTransformer3D_v2):
    """
    Spiking Swin Transformer 3D with MS shortcut
    """
    swin_layer_type = MS_Spiking_Swin_BasicLayer
    downsample_layer_type = MS_SpikingPatchMerging




if __name__ == "__main__":
    _seed_ = 16146
    random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)

    chunk = torch.rand([1, 10, 2, 240, 320])  # ( B, bin, C, H, W)
    spiking_kwargs = {"num_steps": 5, "v_reset": None, "v_th": 0.5, "neuron_type": "lif",
                      "surrogate_fun": 'surrogate.ATan()', "tau": 2., "detach_reset": True, "spike_norm": "BN"}
    model = MS_Spiking_SwinTransformer3D_v2(
                arc_type="swinv1",
                embed_type="MS_Spiking_PatchEmbed_Conv_sfn",
                 img_size = (240,320),
                 patch_size=(1, 1, 2, 2),
                 in_chans=10,
                 embed_dim=96,
                 depths=[2, 2, 6],
                 num_heads=[3, 6, 12],
                 window_size=(2, 10, 10),
                 pretrained_window_size=(0, 0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=0.125,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer="BN", # spiking_norm
                 patch_norm=False,
                 out_indices=(0, 1, 2),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 norm = "BN", #norm for res
                 **spiking_kwargs)

    functional.reset_net(model)
    functional.set_step_mode(model, "m")
    print('--- Test Model ---')
    print(model)
    outps = model(chunk)
    # flops = model.flops()