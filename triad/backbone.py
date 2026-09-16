import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Gate


class GatedBlock(nn.Module):
    """Apply an equivariant linear map and gated activation with one scalar gate per tensor channel."""
    def __init__(self, irreps_in, irreps_out, scalar_activation=torch.nn.functional.silu, gate_activation=torch.sigmoid):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        
        
        
        scalars = []
        gated_tensors = []
        
        for mul, ir in irreps_out:
            if ir.l == 0 and ir.p == 1: # 0e (Scalars)
                scalars.append((mul, ir))
            else: # Higher order (Vectors/Tensors) or pseudo-scalars
                gated_tensors.append((mul, ir))
        
        irreps_scalars = o3.Irreps(scalars)
        irreps_gated = o3.Irreps(gated_tensors)

        
        
        num_gates = irreps_gated.num_irreps
        irreps_gates = o3.Irreps(f"{num_gates}x0e")

        
        
        
        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[scalar_activation] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[gate_activation] * len(irreps_gates),
            irreps_gated=irreps_gated
        )

        
        
        # [Visible Scalars] + [Gate Scalars] + [Gated Tensors]
        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=self.gate.irreps_in
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.gate(x)
        return x

