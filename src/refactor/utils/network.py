import math
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from utils.misc import default, exists


def l2norm(t):
    return F.normalize(t, dim=-1)


class Residual(nn.Module):
    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: Optional[int] = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: Optional[int] = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable) -> None:
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super().__init__()
        """
        generic one layer FC NN for embedding things
        """
        self.input_dim = input_dim
        layers = [nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Building blocks of UNET, convolution + group norm blocks


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# Building blocks of UNET, residual blocks


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim=None, groups: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb=None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


# Additional code to the https://github.com/lucidrains/bit-diffusion/blob/main/bit_diffusion/bit_diffusion.py


class ResnetBlockClassConditioned(ResnetBlock):
    def __init__(
        self, dim: int, dim_out: int, *, num_classes: int, class_embed_dim: int, time_emb_dim=None, groups: int = 8
    ) -> None:
        super().__init__(
            dim=dim + class_embed_dim,
            dim_out=dim_out,
            time_emb_dim=time_emb_dim,
            groups=groups,
        )
        self.class_mlp = EmbedFC(num_classes, class_embed_dim)

    def forward(self, x: torch.Tensor, time_emb=None, c=None) -> torch.Tensor:
        emb_c = self.class_mlp(c)
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, emb_c], axis=1)

        return super().forward(x, time_emb)


# Building blocks of UNET, attention modules


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, scale: int = 10) -> None:
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
    


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_positional_embedding(
    x: torch.Tensor,    # ["batch", "heads", "pos", "dim"]
    sin: torch.Tensor,  # ["1", "1", "pos", "dim"]
    cos: torch.Tensor,  # ["1", "1", "pos", "dim"]
):
    return (x * cos) + (rotate_half(x) * sin)


class AttentionWithRotaryPosEmb(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        dim_rotary_embed = dim_head // 2
        self.rotary_pos_embed = RotaryPositionalEmbedding(dim_rotary_embed)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        
        q = rearrange(q, "b h d s -> b h s d")
        k = rearrange(k, "b h d s -> b h s d")

        # print(q.shape, k.shape, v.shape)
        # torch.Size([16, 8, 16, 200]) torch.Size([16, 8, 16, 200]) torch.Size([16, 8, 16, 200])
        sin, cos = self.rotary_pos_embed(k)
        rot_dim = sin.shape[-1]

        q_left, q_right = q[..., :rot_dim], q[..., rot_dim:]
        k_left, k_right = k[..., :rot_dim], k[..., rot_dim:]

        q_left = apply_rotary_positional_embedding(q_left, sin, cos)
        k_left = apply_rotary_positional_embedding(k_left, sin, cos)

        k = torch.cat([k_left, k_right], dim=-1)
        q = torch.cat([q_left, q_right], dim=-1)

        q = rearrange(q, "b h s d -> b h d s")
        k = rearrange(k, "b h s d -> b h d s")

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w) 
        return self.to_out(out)


class RotaryPositionalEmbedding(nn.Module):
    """GPT-NeoX style rotary positional embedding."""

    def __init__(
        self,
        dim: int,
        max_period: int = 10_000,
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim: int = -2):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
            embeds = torch.cat((freqs, freqs), dim=-1).to(x.device)

            # Shapes: ["batch", "heads", "pos", "dim"]
            self.sin_cached = embeds.sin()[None, None, :, :]
            self.cos_cached = embeds.cos()[None, None, :, :]
        return self.sin_cached, self.cos_cached


class CDCDEmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, input_dim)
        )

    def forward(self, x):
        x = rearrange(x, "b d w h -> b h w d")
        x = self.net(x)
        x = rearrange(x, "b h w d -> b d w h")
        return x


class CDCDTransformerBlock(nn.Module):
    def __init__(self, dim_in: int, hidden_dim: int, mlp_hidden_dim: int = 4096, num_attention_heads: int = 8):
        super().__init__()

        dim_attention_head = hidden_dim // num_attention_heads
        self.residual_norm_attention = Residual(
            PreNorm(
                dim_in,
                AttentionWithRotaryPosEmb(dim_in, heads=num_attention_heads, dim_head=dim_attention_head)
            )
        )

        mlp = CDCDEmbedFC(input_dim=hidden_dim, emb_dim=mlp_hidden_dim)

        self.residual_norm_mlp = Residual(
            PreNorm(
                dim_in,
                mlp
            )
        )

    def forward(self, inputs):
        x, t_emb = inputs
        x = self.residual_norm_attention(x, t_emb)
        x = self.residual_norm_mlp(x, t_emb)
        return x, t_emb


class CDCDTransformerEncoder(nn.Module):
    def __init__(
        self, 
        init_dim: int, 
        time_dim: int, 
        num_layers: int = 8, 
        num_attention_heads: int = 8,
        hidden_dim: int = 1024,
        mlp_hidden_dim: int = 4096, 
        resnet_block_groups: int = 8,
        embed_dim: int = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.embed_dim = init_dim if embed_dim is None else embed_dim

        self.x_lin_in = nn.Linear(init_dim, hidden_dim)
        self.time_lin_in = nn.Linear(time_dim, hidden_dim)

        attention_layers = []
        for _ in range(num_layers):
            block = CDCDTransformerBlock(
                dim_in=hidden_dim,
                hidden_dim=hidden_dim,
                mlp_hidden_dim=mlp_hidden_dim
            )
            attention_layers.append(
                block
            )

        self.net = nn.Sequential(*attention_layers)
        self.lin_out = nn.Linear(hidden_dim, init_dim)
        self.final_res_block = ResnetBlock(init_dim * 2, embed_dim, groups=resnet_block_groups, time_emb_dim=time_dim)


    def forward(self, x, t_emb=None):
        x = rearrange(x, "b w d h -> b w h d")
        r = x.clone()  # b w h d
        t_end = t_emb.clone()

        x = self.x_lin_in(x)
        t_emb = self.time_lin_in(t_emb)
        
        t_emb = rearrange(t_emb, "b d -> b d 1 1")
        x = rearrange(x, "b h w d -> b d w h")
        x, *_ = self.net((x, t_emb))
        
        x = rearrange(x, "b d h w -> b w h d")
        x = self.lin_out(x)

        x = rearrange(x, "b w h d -> b d h w")
        r = rearrange(r, "b w h d -> b d h w")

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t_end)
        
        x = rearrange(x, "b d h w -> b w h d")
        return x
    