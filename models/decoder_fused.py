import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossLevelAttention(nn.Module):
    """ 改进后的跨层注意力模块，加入反卷积层 """

    def __init__(self, low_dim, high_dim, d_model=128, window_size=7,
                 padding_mode='reflect', deconv_scale=2):
        super().__init__()
        self.window_size = window_size
        self.padding_mode = padding_mode
        # 修改反卷积参数
        self.deconv = nn.ConvTranspose2d(
            in_channels=high_dim,
            out_channels=high_dim,
            kernel_size=deconv_scale,  # 关键修改点
            stride=deconv_scale,  # 保持与上采样倍率一致
            padding=0,  # 根据实际需求调整
            output_padding=0
        )
        self.query = nn.Conv2d(low_dim, d_model, 1)
        self.key = nn.Conv2d(high_dim, d_model, 1)
        self.value = nn.Conv2d(high_dim, high_dim, 1)
        self.scale = nn.Parameter(torch.tensor(1.0 / (d_model ** 0.5)))
        self.out_conv = nn.Sequential(
            nn.Conv2d(high_dim, low_dim, 1),
            nn.BatchNorm2d(low_dim),
            nn.ReLU()
        )

    def forward(self, low_feat, high_feat):
        # print(f"[Debug] high_feat shape before deconv: {high_feat.shape}")
        high_up = self.deconv(high_feat)
        # print(f"[Debug] high_up shape after deconv: {high_up.shape}")

        Q = self.query(low_feat)
        K = self.key(high_up)
        V = self.value(high_up)

        # 动态计算填充量
        B, d, H, W = Q.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        # 对称填充（反射模式减少边界影响）
        Q = F.pad(Q, (0, pad_w, 0, pad_h), mode=self.padding_mode)
        K = F.pad(K, (0, pad_w, 0, pad_h), mode=self.padding_mode)
        V = F.pad(V, (0, pad_w, 0, pad_h), mode=self.padding_mode)

        # 窗口划分
        H_pad, W_pad = H + pad_h, W + pad_w
        Q_win = rearrange(Q, 'b c (h nh) (w nw) -> b (h w) (nh nw) c',
                          nh=self.window_size, nw=self.window_size)
        K_win = rearrange(K, 'b c (h nh) (w nw) -> b (h w) (nh nw) c',
                          nh=self.window_size, nw=self.window_size)
        V_win = rearrange(V, 'b c (h nh) (w nw) -> b (h w) (nh nw) c',
                          nh=self.window_size, nw=self.window_size)

        # 注意力计算
        attn = torch.matmul(Q_win, K_win.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        aligned = torch.matmul(attn, V_win)  # [B, num_win, win_size^2, C]

        # 恢复特征图并裁剪填充区域
        aligned = rearrange(aligned, 'b (h w) (nh nw) c -> b c (h nh) (w nw)',
                            h=H_pad // self.window_size, nh=self.window_size,
                            w=W_pad // self.window_size, nw=self.window_size)
        aligned = aligned[:, :, :H, :W]  # 移除填充部分

        return self.out_conv(aligned)


class DynamicWeightGenerator(nn.Module):
    """ 修改输入为对齐后的特征 """

    def __init__(self, feat_dims=[256, 256, 256], hidden_dim=128):  # 输入维度改为对齐后的通道数
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(sum(feat_dims), hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, 1)
        )

    def forward(self, aligned_features):  # 输入改为对齐后的特征列表
        aligned_3, aligned_4, aligned_5 = aligned_features
        feat_concat = torch.cat([aligned_3, aligned_4, aligned_5], dim=1)
        weight_maps = F.softmax(self.conv(feat_concat), dim=1)
        return weight_maps


class LDPFusion(nn.Module):
    def __init__(self, in_dims=[256, 512, 512], out_dim=256):  # 输出通道压缩
        super().__init__()
        # 通道压缩投影层
        self.proj_3 = nn.Sequential(
            nn.Conv2d(in_dims[0], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.proj_4 = nn.Sequential(
            nn.Conv2d(in_dims[1], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.proj_5 = nn.Sequential(
            nn.Conv2d(in_dims[2], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        self.align_3to4 = CrossLevelAttention(in_dims[0], in_dims[1])
        self.align_4to5 = CrossLevelAttention(in_dims[1], in_dims[2])
        self.weight_gen = DynamicWeightGenerator(feat_dims=[out_dim] * 3)  # 输入为压缩后的维度

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, features):
        c3, c4, c5 = features

        # 对齐处理并压缩通道
        aligned_3 = self.proj_3(self.align_3to4(c3, c4))  # [B,256,64,64]
        aligned_4 = self.proj_4(self.align_4to5(c4, c5))  # [B,256,32,32]
        aligned_4 = F.interpolate(aligned_4, aligned_3.shape[2:], mode='bilinear')  # 统一尺寸
        aligned_5 = self.proj_5(F.interpolate(c5, aligned_3.shape[2:], mode='bilinear'))  # [B,256,64,64]

        # 生成权重图（输入改为对齐后的特征）
        weight_maps = self.weight_gen([aligned_3, aligned_4, aligned_5])

        # 加权融合的改进实现
        weighted = (
                weight_maps[:, 0:1] * aligned_3 +
                weight_maps[:, 1:2] * aligned_4 +
                weight_maps[:, 2:3] * aligned_5
        )

        # return aligned_3, aligned_4
        return self.fusion_conv(weighted)

if __name__ == '__main__':
    # 测试用例
    batch_size = 4
    c3 = torch.randn(batch_size, 256, 64, 64)
    c4 = torch.randn(batch_size, 512, 32, 32)
    c5 = torch.randn(batch_size, 512, 16, 16)

    fusion_module = LDPFusion(in_dims=[256, 512, 512], out_dim=512)
    fused_feat = fusion_module([c3, c4, c5])

    print(f"输入尺寸: c3:{c3.shape} c4:{c4.shape} c5:{c5.shape}")
    print(f"融合输出尺寸: {fused_feat.shape}")
    # 预期输出: [4, 512, 64, 64]