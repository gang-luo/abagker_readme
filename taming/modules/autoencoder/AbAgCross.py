import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_in, kv_in, kv_pad_mask=None):
        """
        q_in: [B, Tq, C]  (e.g., antibody CDR pooled vectors or full antibody tokens)
        kv_in: [B, Tk, C] (e.g., antigen tokens)
        kv_pad_mask: [B, Tk] 1=pad, 0=valid
        """
        B, Tq, C = q_in.shape
        Tk = kv_in.size(1)
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(q_in).view(B, Tq, H, D).transpose(1, 2)   # [B,H,Tq,D]
        k = self.k_proj(kv_in).view(B, Tk, H, D).transpose(1, 2)  # [B,H,Tk,D]
        v = self.v_proj(kv_in).view(B, Tk, H, D).transpose(1, 2)  # [B,H,Tk,D]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Tq,Tk]
        if kv_pad_mask is not None:
            scores = scores.masked_fill(kv_pad_mask[:, None, None, :] == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)                                # [B,H,Tq,D]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, C)      # [B,Tq,C]
        out = self.proj_drop(self.out(ctx))
        return out, attn

class FFN(nn.Module):
    def __init__(self, dim, hidden, drop=0.1, act_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act_drop = nn.Dropout(act_drop)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.act_drop(x)
        x = self.fc2(x)
        return self.drop(x)

class CoAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, drop=0.1, attn_drop=0.1, act_drop=0.1):
        super().__init__()
        # 统一使用pre-norm
        self.ln_a1 = nn.LayerNorm(dim)
        self.ln_a2 = nn.LayerNorm(dim)
        self.ln_a3 = nn.LayerNorm(dim)
        self.ca_a_from_g = CrossAttention(dim, num_heads, attn_drop, drop)  # A<-G
        self.ca_g_from_a = CrossAttention(dim, num_heads, attn_drop, drop)  # G<-A

        self.ln_a21 = nn.LayerNorm(dim)
        self.ln_a22 = nn.LayerNorm(dim)
        self.ffn_a = FFN(dim, ffn_dim, drop, act_drop)
        self.ffn_g = FFN(dim, ffn_dim, drop, act_drop)
        
        self.ln_ab_out = nn.LayerNorm(dim)
        self.ln_ag_out = nn.LayerNorm(dim)

    def forward(self, x_ab, x_ag, ag_all=None, pad_b=None, pad_g=None):
        """
        x_ab: [B, Ta, C] 抗体 token 表示（可已过 CDR 约束的 encoder）
        x_ag: [B, Tg, C] 抗原 token 表示
        pad_b/pad_g: [B, T*] 1=valid, 0=pad
        """
        # Pre-norm: 先对输入进行归一化
        xb_norm = self.ln_a1(x_ab)
        xg_norm = self.ln_a2(x_ag)
        if ag_all is not None:
            xg_all_norm = self.ln_a3(ag_all)
        else:
            xg_all_norm = xg_norm

        # 对称交叉注意力：使用归一化后的状态进行交叉更新
        # 抗体从抗原获取信息
        delta_ab, _ = self.ca_a_from_g(xb_norm, xg_all_norm, kv_pad_mask=pad_g)
        # 抗原从抗体获取信息  
        delta_ag, _ = self.ca_g_from_a(xg_norm, xb_norm, kv_pad_mask=pad_b)

        # 残差连接
        x_ab_new = x_ab + delta_ab
        x_ag_new = x_ag + delta_ag

        # FFN层：对更新后的状态进行归一化
        xb_ffn = self.ln_a21(x_ab_new)
        xg_ffn = self.ln_a22(x_ag_new)

        x_ab_final = x_ab_new + self.ffn_a(xb_ffn)
        x_ag_final = x_ag_new + self.ffn_g(xg_ffn)

        # x_ab_final = self.ln_ab_out(x_ab_final)
        # x_ag_final = self.ln_ag_out(x_ag_final)

        return x_ab_final, x_ag_final
