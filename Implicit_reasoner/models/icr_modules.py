import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Attention_via_features(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.norm = nn.LayerNorm(dim)
        if self.with_qkv:
            self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        for p in self.q.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.kv.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.proj.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def forward(self, q, kv):
        B, N, C = q.shape
        _, M, _ = kv.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        x = self.norm(x)
        return x


class Attention_local(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        for p in self.qkv.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.proj.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Causal_intent_RelationHead(nn.Module):
    def __init__(self, config):
        super(Causal_intent_RelationHead, self).__init__()
        self.config = config
        self.decoder = nn.Linear(4096 * 2, 2)
        self.attention_vf = Attention_via_features(4096)
        self.attention_lo = Attention_local(4096)

    def forward(self, refined_pair, candidate_list):
        output_list = []
        for i, candidate_pair in enumerate(candidate_list):
            att_feature_vf = self.attention_vf(candidate_pair, refined_pair[i])
            att_feature_lo = self.attention_lo(candidate_pair)
            output_logits = self.decoder(torch.cat((att_feature_vf, att_feature_lo), dim=2))
            output_list.append(output_logits)
        return output_list


class Vision_clue_enhancement(nn.Module):
    def __init__(self, config):
        super(Vision_clue_enhancement, self).__init__()
        self.config = config
        self.decoder = nn.Linear(4096 * 2, 2)
        self.attention_vf = Attention_via_features(4096)
        self.attention_lo = Attention_local(4096)

    def forward(self, candidate_list, qformer_output):
        output_list = []
        qformer_output = qformer_output.unsqueeze(0)
        for i, candidate_pair in enumerate(candidate_list):
            att_feature_vf = self.attention_vf(candidate_pair, qformer_output)
            output_list.append(att_feature_vf)
        return output_list


class Vision_action_enhancement(nn.Module):
    def __init__(self, config):
        super(Vision_action_enhancement, self).__init__()
        self.config = config
        self.attention_vf = Attention_via_features(4096)

    def forward(self, action_qformer_output, original_qformer_output):

        refined_qformer_output = 0.2 * self.attention_vf(action_qformer_output, original_qformer_output) + original_qformer_output

        return refined_qformer_output
