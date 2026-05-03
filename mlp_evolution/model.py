import torch
from torch import nn
import torch.nn.functional as F

from mlp_evolution.config import SearchConfig
from mlp_evolution.genotype import Genotype
from mlp_evolution.quantization import quantize_tensor


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quantization: str | int, quantize_activation: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.quantization = quantization
        self.quantize_activation = quantize_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = quantize_tensor(self.linear.weight, self.quantization)
        bias = self.linear.bias
        x = quantize_tensor(x, self.quantization) if self.quantize_activation else x
        return F.linear(x, weight, bias)


class QuantizedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        genotype: Genotype,
        config: SearchConfig,
        output_quantization: str | int = "fp32",
        hidden_quantization_override: list[str | int] | None = None,
    ) -> None:
        super().__init__()
        self.genotype = genotype
        self.hidden_layers = nn.ModuleList()
        self.output_quantization = output_quantization

        dims = [input_dim, *genotype.active_hidden_sizes()]
        bits = hidden_quantization_override or genotype.active_quantization_bits()

        for in_dim, out_dim, bit in zip(dims[:-1], dims[1:], bits, strict=True):
            self.hidden_layers.append(
                QuantizedLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    quantization=bit,
                    quantize_activation=config.activation_quantization,
                )
            )

        last_dim = dims[-1]
        self.output_layer = QuantizedLinear(
            in_features=last_dim,
            out_features=num_classes,
            quantization=output_quantization,
            quantize_activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)
