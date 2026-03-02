import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int,
                 dropout: float = 0.1, activation_dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.act_drop = nn.Dropout(activation_dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)                # 比 ReLU 更常用在 Transformer
        x = self.act_drop(x)         # <<< activation dropout
        x = self.linear2(x)
        return self.out_drop(x)      # <<< output dropout

class CDRsAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, cdr_heads: int = 0,
                 attention_dropout: float = 0.1, dropout: float = 0.1,
                 return_attention_weights: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.cdr_heads = cdr_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.return_attention_weights = return_attention_weights

        # 分别投影QKV
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attention_dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, cdrs_score=None):
        """
        x: [B, T, C]
        mask: [B, T] padding mask (1 表示有效，0 表示需要 mask)
        cdrs_score: [B, T] 1表示CDR区域，0表示非CDR
        """
        B, T, C = x.size()
        H, D = self.num_heads, self.head_dim

        # 基础QKV投影
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # ---------- 标准注意力计算 ----------
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,T)

        # ---------- Padding mask ----------
        if mask is not None:
            padding_mask = mask[:, None, None, :] == 0  # True表示需要mask的位置
            scores = scores.masked_fill(padding_mask, float("-inf"))

        # CDR heads的特殊处理 - 只mask非CDR的key
        if cdrs_score is not None and self.cdr_heads > 0:
            # 直接mask非CDR的key
            cdr_key_mask = (cdrs_score[:, None, None, :] == 0)  # [B, 1, 1, T]

            # 如果 mask 全为 True，则将其置为 False (即退化为标准 Self-Attention)
            is_all_masked = cdr_key_mask.all(dim=-1, keepdim=True) # [B, 1, 1, 1]
            cdr_key_mask = cdr_key_mask.masked_fill(is_all_masked, False)
            
            # 应用到CDR heads
            scores[:, :self.cdr_heads, :, :] = scores[:, :self.cdr_heads, :, :].masked_fill(
                cdr_key_mask.expand(-1, self.cdr_heads, T, -1), 
                float("-inf")
            )

        # Softmax和后续处理
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        
        output = self.out_drop(self.out_proj(ctx))
        
        # ---------- 返回注意力权重和mask信息 ----------
        if self.return_attention_weights:
            return output, {
                'attention_weights': attn,
                'cdr_mask_info': cdrs_score,
                'padding_mask': mask
            }
        
        return output
        
class AbEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation_dropout: float = 0.1):
        super().__init__()
        self.self_attn = CDRsAttention(
            dim, num_heads,
            cdr_heads= num_heads//2,
            attention_dropout=attention_dropout,
            dropout=dropout,
            return_attention_weights=True,
        )
        self.ffn = FeedForward(
            dim, ff_dim,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, cdrs_score=None):
        # Self-Attention block
        att_output, attention_info = self.self_attn(self.norm1(x), mask=mask, cdrs_score=cdrs_score)
        x = x + att_output
        # FFN block
        output = x + self.ffn(self.norm2(x))
        return output , attention_info

        
class AbEncoderLayer_noCDRhead(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation_dropout: float = 0.1):
        super().__init__()
        self.self_attn = CDRsAttention(
            dim, num_heads,
            cdr_heads= 0,
            attention_dropout=attention_dropout,
            dropout=dropout,
            return_attention_weights=True,
        )
        self.ffn = FeedForward(
            dim, ff_dim,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, cdrs_score=None):
        # Self-Attention block
        att_output, attention_info = self.self_attn(self.norm1(x), mask=mask, cdrs_score=cdrs_score)
        x = x + att_output
        # FFN block
        output = x + self.ffn(self.norm2(x))
        return output , attention_info


class AbPooler(nn.Module):
    def __init__(self, pooling_method='mean', topk=64):
        super().__init__()
        self.pooling_method = pooling_method
        self.topk = topk
    
    def forward(self, x, attention_info):
        """
        x: [B, T, C] 输入特征
        attention_info: 包含注意力权重和mask信息的字典
        """
        attn_weights = attention_info['attention_weights']  # [B, H, T, T]
        valid_mask = attention_info['padding_mask']         # [B, T] - 注意：现在是padding mask
        
        B, T, C = x.shape
        H = attn_weights.shape[1]
        
        # 计算每个token的重要性分数
        # 方法：结合自注意力和跨注意力
        self_attention = torch.diagonal(attn_weights, dim1=-2, dim2=-1).sum(dim=1) / H  # [B, T]
        cross_attention = attn_weights.sum(dim=1).sum(dim=1) / H  # [B, T] - 先对head求和，再对query求和
        attention_importance = self_attention + cross_attention  # [B, T]
        
        if valid_mask is not None:
            attention_importance = attention_importance.masked_fill(valid_mask == 0, float("-inf"))
        
        # # 如果有CDR信息，给CDR区域额外权重
        # if 'cdr_mask_info' in attention_info and attention_info['cdr_mask_info'] is not None:
        #     cdr_positions = attention_info['cdr_mask_info']  # [B, T] - 1表示CDR，0表示非CDR

        #     # 给CDR区域的token增加重要性权重
        #     cdr_boost = 1.5 * cdr_positions.float()  # CDR区域重要性提升50%
        #     attention_importance = attention_importance + cdr_boost
        
        return self.pool_tokens(x, attention_importance, valid_mask)
    
    def pool_tokens(self, x, importance_scores, valid_mask):
        """
        根据重要性分数池化token
        """
        if self.pooling_method == 'topk':
            # 选择最重要的前K个token
            k = min(self.topk, x.shape[1])
            
            # 注意：如果importance_scores中有-inf，topk仍能正确处理
            _, topk_indices = torch.topk(importance_scores, k, dim=1)  # [B, k]
            
            # 处理可能的-inf情况，确保索引有效
            batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, k)
            pooled_x = x[batch_indices, topk_indices]  # [B, k, C]
            return pooled_x
            
        elif self.pooling_method == 'weighted':
            # 加权平均池化
            # 首先处理-inf值，将其转换为0权重
            valid_scores = torch.where(
                torch.isfinite(importance_scores),
                importance_scores,
                torch.tensor(0.0, device=importance_scores.device)
            )
            
            # 确保权重非负
            valid_scores = F.relu(valid_scores)
            
            # 归一化权重
            weights = F.softmax(valid_scores, dim=1)  # [B, T]
            
            # 应用padding mask：将padding位置的权重设为0
            if valid_mask is not None:
                weights = weights * valid_mask.float()
                # 重新归一化
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            pooled_x = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, C]
            return pooled_x.unsqueeze(1)  # [B, 1, C]
            
        else:  # mean pooling
            if valid_mask is not None:
                # 使用mask的平均池化
                mask_expanded = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
                pooled_x = torch.sum(x * mask_expanded, dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                pooled_x = x.mean(dim=1)
            return pooled_x.unsqueeze(1)  # [B, 1, C]
