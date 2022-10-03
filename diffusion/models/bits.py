import torch
from einops import rearrange, reduce

BITS = 1


def decimal_to_bits(x, bits=BITS):
    """expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1"""
    device = x.device

    x = (x * 1).int().clamp(0, 1)  # x = (x * 255).int().clamp(0, 255)

    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
    mask = rearrange(mask, "d -> d 1 1")
    x = rearrange(x, "b c h w -> b c 1 h w")

    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, "b c d h w -> b (c d) h w")
    bits = bits * 2 - 1
    return bits


def bits_to_decimal(x, bits=BITS):
    """expects bits from -1 to 1, outputs image tensor from 0 to 1"""
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int32)

    mask = rearrange(mask, "d -> d 1 1")
    # x = rearrange(x, 'b (c d) h w -> b c d h w', d = 8)
    x = rearrange(x, "b (c d) h w -> b c d h w", d=1)  # lucas

    dec = reduce(x * mask, "b c d h w -> b c h w", "sum")
    return (dec / 1).clamp(0.0, 1.0)  # changed(dec / 255).clamp(0., 1.)
