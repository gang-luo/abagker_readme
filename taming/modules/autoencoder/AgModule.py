import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AgMixPooler(nn.Module):
    """
    基于
      - l_full_embs: [B, T, E] (抗原预训练模型的残基特征)
      - ssf_x      : [B, T, 5] (序列级别特征)
    计算每个 token 的权重，并将 l_full_embs 池化到指定长度 L_out。

    两路权重：
      1) conv 权重：在 token 维上滑动窗口，卷积核覆盖 [win_T, E]，输出 [B, T, 1]
      2) ssf 权重：对 ssf_x 的每维特征使用独立可学习权重，输出 [B, T, 1]
    融合：通过可学习的门控 alpha ∈ (0,1) 对两路权重做凸组合；随后对 token 维做 softmax 正规化。
    
    池化策略：
      - segment：把序列均匀切成 L_out 段，对每段内按权重做加权平均，输出 [B, L_out, E]
      - topk    ：选取权重最高的 K=L_out 个 token，按权重重标定后拼接为 [B, L_out, E]
                  （保持原 token 顺序，便于对齐下游序列任务）

    参数:
      target_len (int)         : 目标长度 L_out
      window_T (int)           : conv 的 token 窗口大小（感受野）
      pooling ('segment'|'topk'): 池化策略
      bias (bool)              : 卷积与线性层是否使用偏置
    """

    def __init__(self, target_len: int, window_T: int = 7,
                 pooling: str = 'segment', bias: bool = True):
        super().__init__()
        assert target_len > 0
        assert pooling in ('segment', 'topk')
        self.target_len = target_len
        self.window_T = window_T
        self.pooling = pooling

        # 注意：我们把 [B, T, E] 视为单通道图像 [B, 1, T, E]，
        # 使用 Conv2d(kernel_size=(win_T, E)) 生成 [B, 1, T, 1]
        self.conv_attn: nn.Conv2d | None = None  # 延后在 forward 用到 E 时构建

        # ssf 权重：每维特征使用独立可学习权重
        self.ssf_weight = nn.Parameter(torch.randn(7))  # [5]
        if bias:
            self.ssf_bias = nn.Parameter(torch.zeros(1))
        else:
            self.ssf_bias = None

        # 可学习门控，用 sigmoid 映射到 (0,1)，控制两路权重的融合比例
        self.gate_logit = nn.Parameter(torch.zeros(1))  # 初始 0 → sigmoid=0.5

        self.bias = bias

    def _build_conv_if_needed(self, E: int, device, dtype):
        """根据 E 动态构建覆盖 [window_T, E] 的 2D 卷积核。"""
        if self.conv_attn is None:
            # padding 仅在 T 维做 same padding；E 维覆盖全宽，无需 padding
            pad_T = self.window_T // 2
            self.conv_attn = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(self.window_T, E),
                stride=(1, 1),
                padding=(pad_T, 0),
                bias=self.bias,
            ).to(device=device, dtype=dtype)

    def _conv_weights(self, l_full_embs: torch.Tensor) -> torch.Tensor:
        """
        输入: l_full_embs [B, T, E]
        输出: conv 权重 [B, T, 1]
        """
        B, T, E = l_full_embs.shape
        self._build_conv_if_needed(E, l_full_embs.device, l_full_embs.dtype)
        x = l_full_embs.unsqueeze(1)  # [B, 1, T, E]
        y = self.conv_attn(x)         # [B, 1, T, 1]
        return y.squeeze(1)           # [B, T, 1]

    def _ssf_weights(self, ssf_x: torch.Tensor) -> torch.Tensor:
        """
        输入: ssf_x [B, T, 5]
        输出: ssf 权重 [B, T, 1]
        改进：使用 per-feature 的可学习权重
        """
        # ssf_x: [B, T, 5]
        # self.ssf_weight: [5]
        weights = torch.einsum('bti,i->bt', ssf_x, self.ssf_weight)  # [B, T]
        if self.ssf_bias is not None:
            weights = weights + self.ssf_bias
        return weights.unsqueeze(-1)  # [B, T, 1]

    def _fuse_and_normalize(self, w1: torch.Tensor, w2: torch.Tensor, 
                          padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        融合两路标量权重，并在 token 维上 softmax 归一化。
        输入: w1, w2 均为 [B, T, 1]
        输出: a_norm [B, T, 1], 且 sum_T a_norm = 1
        """
        alpha = torch.sigmoid(self.gate_logit)  # 标量 ∈ (0,1)
        # 数值稳定：先压缩到合适范围，再融合
        w = alpha * w1 + (1 - alpha) * w2
        a = torch.tanh(w)  # 压缩，减小极端值的影响
        a = a.squeeze(-1)  # [B, T]
        
        # 应用 padding mask
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float('-inf'))
        
        a_norm = F.softmax(a, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return a_norm

    def _segment_pool(self, x: torch.Tensor, attn: torch.Tensor, 
                     padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        分段加权池化：把 T 均匀划分为 L 段，每段内按 attn 做加权平均。
        x:     [B, T, E]
        attn:  [B, T, 1]，token 维已 softmax 过（全局和为 1）
        padding_mask: [B, T] 可选
        返回:  [B, L, E]
        """
        B, T, E = x.shape
        L = self.target_len

        # 计算每个 token 的段索引（整除划分，最后一段包含尾部剩余）
        device = x.device
        idx = torch.arange(T, device=device)
        # 按比例映射到 [0, L-1]
        seg = torch.clamp((idx * L) // max(T, 1), 0, L - 1)  # [T]

        # one-hot 聚合到段： [T] -> [T, L] -> [B, T, L]
        seg_onehot = F.one_hot(seg, num_classes=L).to(dtype=x.dtype)  # [T, L]
        seg_onehot = seg_onehot.unsqueeze(0).expand(B, -1, -1)        # [B, T, L]

        # 如果有padding_mask，将padding位置的注意力权重设为0
        if padding_mask is not None:
            attn = attn * padding_mask.unsqueeze(-1).float()

        # 计算每段的权重总和与加权表示
        # weights per segment: [B, L, 1]
        seg_w = torch.bmm(attn.transpose(1, 2), seg_onehot).transpose(1, 2)  # [B, L, 1]
        # features per segment (加权和): [B, L, E]
        weighted_x = x * attn  # [B, T, E]
        seg_feat = torch.einsum('bte, btl -> ble', weighted_x, seg_onehot)    # [B, L, E]

        # 避免除零：如果某段没有权重，退化为均值
        eps = 1e-8
        seg_feat = seg_feat / (seg_w.clamp_min(eps))  # 广播 [B, L, E]/[B, L, 1]
        return seg_feat

    def _topk_pool(self, x: torch.Tensor, attn: torch.Tensor,
                  padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        选取权重最高的 K=L_out 个 token，并保持原始顺序。
        x:    [B, T, E]
        attn: [B, T, 1]，token 维已 softmax 过（全局和为 1）
        padding_mask: [B, T] 可选
        返回: [B, L, E]
        """
        B, T, E = x.shape
        K = min(self.target_len, T)
        a = attn.squeeze(-1)  # [B, T]
        
        # 如果有 padding_mask，将 padding 位置的注意力设为极小值
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float('-inf'))
        
        vals, idx = torch.topk(a, k=K, dim=-1, largest=True, sorted=True)  # [B, K]
        # 保持原序（从小到大）
        idx_sorted, _ = torch.sort(idx, dim=-1)  # [B, K]
        # Batch gather
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(B, K)
        pooled = x[batch_idx, idx_sorted, :]  # [B, K, E]
        return pooled

    def forward(self, l_full_embs: torch.Tensor, ssf_x: torch.Tensor, 
                padding_mask: Optional[torch.BoolTensor] = None):
        """
        参数:
          l_full_embs: [B, T, E]
          ssf_x: [B, T, 5]
          padding_mask: [B, T] 布尔张量，True 表示有效位置，False 表示 padding
        返回:
          pooled: [B, L_out, E]  池化后的特征
          attn  : [B, T, 1]      融合后的全局注意力权重（可用于可视化/对齐）
        """
        assert l_full_embs.dim() == 3 and ssf_x.dim() == 3, \
            "l_full_embs 应为 [B,T,E]，ssf_x 应为 [B,T,5]"
        B, T, E = l_full_embs.shape
        assert ssf_x.shape[:2] == (B, T) and ssf_x.shape[2] == 7, \
            "ssf_x 的形状应为 [B,T,5]，且 B、T 要与 l_full_embs 一致"

        w_conv = self._conv_weights(l_full_embs)   # [B, T, 1]
        w_ssf  = self._ssf_weights(ssf_x)          # [B, T, 1]
        attn = self._fuse_and_normalize(w_conv, w_ssf, padding_mask)  # [B, T, 1]

        if self.pooling == 'segment':
            pooled = self._segment_pool(l_full_embs, attn, padding_mask)  # [B, L, E]
        else:
            pooled = self._topk_pool(l_full_embs, attn, padding_mask)     # [B, L, E]

        return pooled, attn


class AgMixPooler_noSSF(nn.Module):

    def __init__(self, target_len: int, window_T: int = 7,
                 pooling: str = 'segment', bias: bool = True):
        super().__init__()
        assert target_len > 0
        assert pooling in ('segment', 'topk')
        self.target_len = target_len
        self.window_T = window_T
        self.pooling = pooling

        self.token_reduce_layer = nn.Sequential(
            nn.Linear(896, 256),
            nn.GELU(),
            nn.Linear(256, target_len),
            nn.GELU(),
        )
        
        self.ln1 = nn.LayerNorm(512)

    def forward(self, l_full_embs: torch.Tensor, ssf_x: torch.Tensor, 
                padding_mask: Optional[torch.BoolTensor] = None):

        l_full_embs = l_full_embs.transpose(1,2)
        pooled = self.token_reduce_layer(l_full_embs).transpose(1,2)
        pooled = self.ln1(pooled)

        return pooled, None



class AgMixPooler_1206(nn.Module):

    def __init__(self, target_len: int, window_T: int = 7,
                 pooling: str = 'topk', bias: bool = True):
        super().__init__()
        assert target_len > 0
        assert pooling in ('segment', 'topk')
        self.target_len = target_len
        self.window_T = window_T
        self.pooling = pooling

        # conv define
        self.dim = 512
        self.ab_token_reduce_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim//8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.dim//8),
        )
        self.conv_attn = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.window_T, self.dim//8),
            stride=(1, 1),
            padding=(self.window_T // 2, 0),
            bias=True,
        )

        # ssf 权重：每维特征使用独立可学习权重
        self.ssf_weight = nn.Parameter(torch.randn(7))  # [5]
        if bias:
            self.ssf_bias = nn.Parameter(torch.zeros(1))
        else:
            self.ssf_bias = None

        self.gate_logit = nn.Parameter(torch.zeros(1)) # 可学习门控，用 sigmoid 映射到 (0,1)，控制两路权重的融合比例
        self.bias = bias


    def _conv_weights(self, l_full_embs: torch.Tensor) -> torch.Tensor:
        """
        输入: l_full_embs [B, T, E]
        输出: conv 权重 [B, T, 1]
        """
        B, T, E = l_full_embs.shape
        x = l_full_embs.unsqueeze(1)  # [B, 1, T, E//8]
        y = self.conv_attn(x)         # [B, 1, T, 1]
        return y.squeeze(1)           # [B, T, 1]

    def _ssf_weights(self, ssf_x: torch.Tensor) -> torch.Tensor:
        """
        输入: ssf_x [B, T, 5]
        输出: ssf 权重 [B, T, 1]
        改进：使用 per-feature 的可学习权重
        """
        # ssf_x: [B, T, 5]
        # self.ssf_weight: [5]
        weights = torch.einsum('bti,i->bt', ssf_x, self.ssf_weight)  # [B, T]
        if self.ssf_bias is not None:
            weights = weights + self.ssf_bias
        return weights.unsqueeze(-1)  # [B, T, 1]

    def _fuse_and_normalize(self, w1: torch.Tensor, w2: torch.Tensor, 
                          padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        融合两路标量权重，并在 token 维上 softmax 归一化。
        输入: w1, w2 均为 [B, T, 1]
        输出: a_norm [B, T, 1], 且 sum_T a_norm = 1
        """
        alpha = torch.sigmoid(self.gate_logit)  # 标量 ∈ (0,1)
        w = alpha * w1 + (1 - alpha) * w2
        a = torch.tanh(w)  # 压缩，减小极端值的影响
        a = a.squeeze(-1)  # [B, T]
        
        # 应用 padding mask
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float('-inf'))
        
        a_norm = F.softmax(a, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return a_norm

    def _topk_pool(self, x: torch.Tensor, attn: torch.Tensor,
                  padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        选取权重最高的 K=L_out 个 token，并保持原始顺序。
        x:    [B, T, E]
        attn: [B, T, 1]，token 维已 softmax 过（全局和为 1）
        padding_mask: [B, T] 可选
        返回: [B, L, E]
        """
        B, T, E = x.shape
        K = min(self.target_len, T)
        a = attn.squeeze(-1)  # [B, T]
        
        # 如果有 padding_mask，将 padding 位置的注意力设为极小值
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float('-inf'))
        
        vals, idx = torch.topk(a, k=K, dim=-1, largest=True, sorted=True)  # [B, K]
        # 保持原序（从小到大）
        idx_sorted, _ = torch.sort(idx, dim=-1)  # [B, K]
        # Batch gather
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(B, K)
        pooled = x[batch_idx, idx_sorted, :]  # [B, K, E]
        return pooled

    def forward(self, l_full_embs: torch.Tensor, ssf_x: torch.Tensor, 
                padding_mask: Optional[torch.BoolTensor] = None):

        l_full_pooled = self.ab_token_reduce_layer(l_full_embs)
        B, T, E = l_full_pooled.shape

        w_conv = self._conv_weights(l_full_pooled)   # [B, T, 1]
        w_ssf  = self._ssf_weights(ssf_x)          # [B, T, 1]
        attn = self._fuse_and_normalize(w_conv, w_ssf, padding_mask)  # [B, T, 1]
        pooled = self._topk_pool(l_full_embs, attn, padding_mask)     # [B, L, E]

        return pooled, attn


    
# # =========================
# # 使用示例
# # =========================
# if __name__ == "__main__":
#     B, T, E = 2, 128, 256
#     L_out = 32
#     l_full_embs = torch.randn(B, T, E)
#     ssf_x = torch.randn(B, T, 5)
    
#     # 创建 padding mask (假设前 100 个位置是有效的)
#     padding_mask = torch.ones(B, T, dtype=torch.bool)
#     padding_mask[:, 100:] = False  # 后28个位置是padding

#     model = ResiduePooler(target_len=L_out, window_T=9, pooling='segment')
#     pooled, attn = model(l_full_embs, ssf_x, padding_mask)
#     print(f"pooled shape: {pooled.shape}")  # -> torch.Size([2, 32, 256])
#     print(f"attn shape: {attn.shape}")      # -> torch.Size([2, 128, 1])
    
#     # 测试 topk 模式
#     model_topk = ResiduePooler(target_len=L_out, window_T=9, pooling='topk')
#     pooled_topk, attn_topk = model_topk(l_full_embs, ssf_x, padding_mask)
#     print(f"topk pooled shape: {pooled_topk.shape}")  # -> torch.Size([2, 32, 256])

