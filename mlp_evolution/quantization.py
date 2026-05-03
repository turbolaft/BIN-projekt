import torch


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def fake_quantize_uniform(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return x
    if bits <= 0:
        raise ValueError("bits must be positive")

    q_levels = (2**bits) - 1
    x_min = x.detach().min()
    x_max = x.detach().max()

    if torch.isclose(x_min, x_max):
        return x

    scale = (x_max - x_min) / q_levels
    normalized = (x - x_min) / scale
    quantized = ste_round(normalized).clamp(0, q_levels)
    return quantized * scale + x_min


def fake_quantize_binary(x: torch.Tensor) -> torch.Tensor:
    clipped = torch.tanh(x)
    signed = clipped.sign()
    signed = torch.where(signed == 0, torch.ones_like(signed), signed)
    return clipped + (signed - clipped).detach()


def fake_quantize_ternary(x: torch.Tensor) -> torch.Tensor:
    clipped = torch.tanh(x)
    threshold = 0.5 * clipped.detach().abs().mean()
    ternary = torch.where(
        clipped > threshold,
        torch.ones_like(clipped),
        torch.where(clipped < -threshold, -torch.ones_like(clipped), torch.zeros_like(clipped)),
    )
    return clipped + (ternary - clipped).detach()


def quantize_tensor(x: torch.Tensor, scheme: str | int) -> torch.Tensor:
    if isinstance(scheme, int):
        return fake_quantize_uniform(x, scheme)
    if scheme == "binary":
        return fake_quantize_binary(x)
    if scheme == "ternary":
        return fake_quantize_ternary(x)
    if scheme == "fp32":
        return x
    raise ValueError(f"Unsupported quantization scheme: {scheme}")
