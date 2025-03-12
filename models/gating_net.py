import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(scale)

class DualExpertGate(nn.Module):
    """双专家动态门控网络"""

    def __init__(self, in_channels=256, feat_channels=64):

        super().__init__()
        # 权重预测头
        self.conv = nn.Sequential(
            nn.Conv2d(512, feat_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, 2, 1)
        )
        # 特征压缩层
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )

        # 空间-通道联合门控
        self.spatial_attn = SpatialAttention()
        self.ds_conv = DepthwiseSeparableConv(feat_channels, feat_channels)
        self.weight_head = nn.Conv2d(feat_channels, 2, kernel_size=1)
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 最后卷积层初始化为零，增强训练稳定性
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, 0.5)

        # 空间注意力层特殊初始化
        nn.init.xavier_uniform_(self.spatial_attn.conv.weight)
        nn.init.constant_(self.spatial_attn.conv.bias, 0.0)

    def forward(self, expert1, expert2):
        # 拼接特征
        combined = torch.cat([expert1, expert2], dim=1)

        # 生成动态权重
        weights = F.softmax(self.conv(combined), dim=1)  # [B,2,H,W]
        weighted_p = weights[:, 1:2] * expert2
        weighted_d = weights[:, 0:1] * expert1
        weighted = weights[:, 0:1] * expert1 + weights[:, 1:2] * expert2
        # 特征压缩
        x = self.compress(weighted)  # [B, 64, H, W]

        # 空间注意力引导
        spatial_attn = self.spatial_attn(x)
        x = x * spatial_attn  # 空间增强

        # 深度可分离卷积
        x = self.ds_conv(x)  # [B, 64, H, W]

        # 生成原始权重
        raw_weights = self.weight_head(x)  # [B, 2, H, W]

        # 空间维度softmax归一化
        final_weights = F.softmax(raw_weights, dim=1)  # 各位置两个权重和为1

        # return weights, spatial_attn
        return weighted_p, weighted_d, final_weights  # 返回融合特征和最终权重

# 测试代码
if __name__ == "__main__":
    # 初始化双专家门控网络
    gate_net = DualExpertGate(256, 64)

    # 模拟专家输出（相同空间尺寸）
    expert1 = torch.randn(4, 256, 16, 16)  # 专家1输出
    expert2 = torch.randn(4, 256, 16, 16)  # 专家2输出

    # 前向传播
    fused_feat, weights = gate_net(expert1, expert2)

    print(f"融合特征尺寸: {fused_feat.shape}")  # [4,256,64,64]
    print(f"权重图尺寸: {weights.shape}")  # [4,2,64,64]

    # 验证权重归一化
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))