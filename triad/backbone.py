import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Gate


class GatedBlock(nn.Module):
    """
    标准的 e3nn 积木块：Linear -> Gate (Activation)
    自动计算需要的 Gate 标量数量。
    """
    def __init__(self, irreps_in, irreps_out, scalar_activation=torch.nn.functional.silu, gate_activation=torch.sigmoid):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        # 1. 分析输出 Irreps，将其拆分为“标量”和“非标量”
        # 标量 (0e) 可以直接应用激活函数
        # 非标量 (1e, 2e...) 需要被“门”控制
        scalars = []
        gated_tensors = []
        
        for mul, ir in irreps_out:
            if ir.l == 0 and ir.p == 1: # 0e (Scalars)
                scalars.append((mul, ir))
            else: # Higher order (Vectors/Tensors) or pseudo-scalars
                gated_tensors.append((mul, ir))
        
        irreps_scalars = o3.Irreps(scalars)
        irreps_gated = o3.Irreps(gated_tensors)

        # 2. 计算需要的 Gate 数量
        # 每个 gated channel 需要一个对应的 scalar gate
        num_gates = irreps_gated.num_irreps
        irreps_gates = o3.Irreps(f"{num_gates}x0e")

        # 3. 定义 Gate 模块
        # act_scalars: 用于普通标量的激活 (如 SiLU)
        # act_gates: 用于门控标量的激活 (必须是 0-1 之间，如 Sigmoid)
        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[scalar_activation] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[gate_activation] * len(irreps_gates),
            irreps_gated=irreps_gated
        )

        # 4. 定义 Linear 层
        # Linear 的输出必须匹配 Gate 的输入要求：
        # [Visible Scalars] + [Gate Scalars] + [Gated Tensors]
        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=self.gate.irreps_in
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.gate(x)
        return x

