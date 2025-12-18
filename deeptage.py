import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

EPS = 1e-6

# 时间对齐的替代梯度函数（Temporal-Aligned Surrogate Gradient）
class TemporalSurrogateSpike(Function):
    @staticmethod
    def forward(ctx, input, v_th, delta):
        """
        前向传播：
        input: 膜电位
        v_th: 放电阈值
        delta: 时间步缩放因子 δ(σ_t) （必须是 tensor）
        返回：是否放电（0或1）
        """
        # 保存 input, delta 用于 backward
        ctx.save_for_backward(input, delta)
        ctx.v_th = float(v_th)
        out = (input >= v_th).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用平滑的替代梯度函数进行梯度近似
        """
        input, delta = ctx.saved_tensors
        v_th = ctx.v_th

        # 计算替代梯度 sgt(x) = 1 / (1 + pi^2 * ((x-v_th)/delta)^2)
        x = (input - v_th) / (delta + EPS)
        denom = 1.0 + (math.pi ** 2) * (x ** 2)
        sgt = 1.0 / denom

        return grad_output * sgt, None, None


def surrogate_spike(input, v_th=1.0, delta=1.0):
    """
    delta 可以是 Python float 或 torch.Tensor。
    若是 tensor：使用 detach().clone().to(device).type_as(input) 保证安全拷贝且 dtype/device 匹配。
    若是 float：创建新的 tensor。
    """
    if isinstance(delta, torch.Tensor):
        # detach clone 并转换到 input 相同设备与 dtype
        delta_tensor = delta.detach().clone().to(device=input.device).type_as(input)
    else:
        delta_tensor = torch.tensor(delta, device=input.device, dtype=input.dtype)
    return TemporalSurrogateSpike.apply(input, v_th, delta_tensor)


# LIF 神经元模块（带时间步 δ 调整）
class LIFNode(nn.Module):
    def __init__(self, v_th=1.0, decay=2.0, device='cpu'):
        super().__init__()
        self.v_th = float(v_th)
        self.decay = float(decay)
        # 将 decay 转为 alpha（简单映射）
        self.alpha = 1.0 - (1.0 / max(self.decay, 1.0))
        self.device = device

    def forward(self, mem, input_current, delta_t):
        """
        mem: 前一时刻膜电位（tensor）
        input_current: 本时刻输入（tensor）
        delta_t: 当前时间步 δ（可以是 scalar tensor 或 float）
        """
        mem = mem * self.alpha + input_current

        # 确保 delta_t 是与 mem 兼容的 tensor（由 surrogate_spike 内部处理）
        spike = surrogate_spike(mem, v_th=self.v_th, delta=delta_t)

        # 重置膜电位
        mem = mem * (1.0 - spike)
        return mem, spike


# 辅助分类器（用于 STDS）
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Spiking Stage
class SpikingStage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, v_th=1.0, decay=2.0, w=0.25):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIFNode(v_th=v_th, decay=decay)
        self.w = float(w)
        # 初始 sigma1（buffer 仅用于观察，不作为可训练参数）
        self.register_buffer('_sigma1', torch.tensor(0.0), persistent=False)

    def forward_sequence(self, inputs):
        """
        支持两种输入格式：
          - Tensor: (B, T, C, H, W)
          - list:  [tensor(B,C,H,W), ...] 长度 T
        返回：
          outputs: list 长度 T，每项 B x C x H x W（膜电位特征）
          spikes:  list 长度 T，每项 B x C x H x W（脉冲）
        """
        # 兼容处理
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 5:
                B, T, C, H, W = inputs.shape
                time_seq = [inputs[:, t] for t in range(T)]
            else:
                raise ValueError("输入 Tensor 必须是 5 维 (B,T,C,H,W)")
        elif isinstance(inputs, list):
            time_seq = inputs
            T = len(time_seq)
        else:
            raise TypeError(f"不支持的输入类型: {type(inputs)}")

        device = time_seq[0].device
        # 初始化膜电位为 0
        mem = torch.zeros(time_seq[0].size(0), self.conv.out_channels,
                          time_seq[0].size(2), time_seq[0].size(3), device=device)

        outputs = []
        spikes = []
        sigma1 = None

        for t, frame in enumerate(time_seq):
            # conv + bn
            x = self.conv(frame)
            x = self.bn(x)

            # 计算 candidate membrane (用于计算 sigma_t)
            tmp_mem = mem * self.lif.alpha + x

            # 计算每样本的 L2 norm: σ_t per-sample, 然后取 batch mean
            sigma_t = torch.norm((tmp_mem - self.lif.v_th).view(tmp_mem.size(0), -1), p=2, dim=1)
            sigma_t_mean = sigma_t.mean()

            if t == 0:
                # 保持 sigma1 为常数
                sigma1 = sigma_t_mean.detach() + EPS
                # 存 buffer
                self._sigma1 = sigma1.detach()

            # 计算 δ(σ_t) = w * (σ_t/σ1) + (1-w)
            delta_t = self.w * (sigma_t_mean / (sigma1 + EPS)) + (1.0 - self.w)

            if isinstance(delta_t, torch.Tensor):
                # 安全拷贝并转换
                delta_t = delta_t.detach().clone().to(device=tmp_mem.device).type_as(tmp_mem)
            else:
                delta_t = torch.tensor(delta_t, device=tmp_mem.device, dtype=tmp_mem.dtype)

            # 更新
            mem, spike = self.lif(mem, x, delta_t)

            outputs.append(mem)
            spikes.append(spike)

        return outputs, spikes


# 简单网络结构
class SimpleSpikingNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, channels=[32, 64, 128],
                 T=4, v_th=1.0, decay=2.0, w=0.25):
        super().__init__()
        self.T = T
        self.stages = nn.ModuleList()
        self.aux_classifiers = nn.ModuleList()

        prev = in_channels
        for i, ch in enumerate(channels):
            stage = SpikingStage(prev, ch, v_th=v_th, decay=decay, w=w)
            self.stages.append(stage)
            if i < len(channels) - 1:
                self.aux_classifiers.append(AuxiliaryClassifier(ch, num_classes))
            prev = ch

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):
        """
        输入：x (B, T, C, H, W)
        输出：
          p_t: list 长度 T，每项 B x num_classes（主分类器在各 time-step 的输出）
          aux_preds: list 长度 K-1（stage 数减一），每项为 list 长度 T 的 B x num_classes（辅助头）
        """
        B, T, C, H, W = x.shape
        # 将 tensor 拆为帧序列 list，以便传入每个 stage（每个 stage 返回 list）
        current_seq = [x[:, t] for t in range(T)]
        stage_feats = []

        for k, stage in enumerate(self.stages):
            feats_k, _ = stage.forward_sequence(current_seq)
            stage_feats.append(feats_k)
            # 下一 stage 的输入为本 stage 在每个 time-step 的输出（mem）
            current_seq = feats_k

        # 主分类器在每个 time-step 的输出
        p_t = []
        for t in range(T):
            f = stage_feats[-1][t]
            gap = self.global_pool(f).view(B, -1)
            p = self.fc(gap)
            p_t.append(p)

        # 各辅助分类器输出
        aux_preds = []
        for k in range(len(self.aux_classifiers)):
            g = self.aux_classifiers[k]
            feats_k = stage_feats[k]
            preds_k = [g(feats_k[t]) for t in range(T)]
            aux_preds.append(preds_k)

        return p_t, aux_preds


# 损失函数（STDS）
def compute_stds_loss(p_t, aux_preds, target):
    T = len(p_t)
    K_minus_1 = len(aux_preds)
    K = K_minus_1 + 1
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    # 辅助分类器损失
    for k_preds in aux_preds:
        for t in range(T):
            total_loss = total_loss + ce(k_preds[t], target)

    # 主分类器损失
    for t in range(T):
        total_loss = total_loss + ce(p_t[t], target)

    total_loss = total_loss / (K * T)
    return total_loss


# 随机输入测试
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleSpikingNet(in_channels=3, num_classes=10, channels=[32, 64, 128],
                             T=4, v_th=1.0, decay=2.0, w=0.25).to(device)

    B = 8
    T = model.T
    dummy_x = torch.randn(B, T, 3, 32, 32).to(device)
    dummy_y = torch.randint(0, 10, (B,), dtype=torch.long).to(device)

    p_t, aux_preds = model(dummy_x)
    loss = compute_stds_loss(p_t, aux_preds, dummy_y)
    print("loss:", loss.item())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("训练步完成")
