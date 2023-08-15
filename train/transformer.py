import math
import json
import re
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import xformers.ops as xops
from xformers.components.positional_embedding import RotaryEmbedding

# from openclip
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


# args and default params follow llama (except with LayerNorm instead of RmsNorm)
@dataclass
class Params:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    norm_eps: float = 1e-5
    seq_len: int = 2048
    post_embed_norm: bool = False
    norm_type: nn.Module = nn.LayerNorm
    apply_qk_norm: bool = False


class RotaryWithCast(RotaryEmbedding):
    def forward(self, q, k, v):
        q, k = super().forward(q, k)
        return q.to(v.dtype), k.to(v.dtype), v


def xformers_attn(queries, keys, values, attn_mask):
    # TODO: maybe this is too slow
    # not sure how to make it so attn_maks doesn't change stride during forward
    if attn_mask is not None and attn_mask.shape[0] % 8 != 0:
        seq_len = attn_mask.shape[0]
        closest_8k = math.ceil(seq_len/8) * 8
        big_attn_mask = torch.zeros((closest_8k, closest_8k), device=attn_mask.device, dtype=attn_mask.dtype)
        big_attn_mask[:seq_len, :seq_len] = attn_mask
        attn_mask = big_attn_mask[:seq_len, :seq_len]
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=attn_mask)


class CustomAttn(nn.Module):
    def __init__(self, layer_id, args: Params, attn_mask=None):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(args.dim, 3 * args.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # self.pos_embed = RotaryWithCast(self.head_dim, args.seq_len)
        self.attn_fn = xformers_attn
        self.apply_qk_norm = args.apply_qk_norm

        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(
            self.out_proj.weight, std=std, a=-3 * std, b=3 * std
        )
        '''
        proj_std = (args.dim ** -0.5) * ((2 * args.n_layers) ** -0.5)
        attn_std = args.dim ** -0.5

        torch.nn.init.normal_(self.in_proj.weight, std=attn_std)
        torch.nn.init.normal_(self.out_proj.weight, std=proj_std)
        '''

        # set attn mask
        if attn_mask is not None:
            self.register_buffer('attn_mask', attn_mask, persistent=False)
        else:
            self.attn_mask = None

        # initialize norm layers for queries and keys if needed
        self.q_norm = (
            args.norm_type(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            args.norm_type(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        batchsize, seqlen, _ = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, seqlen, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, seqlen, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, seqlen, self.n_heads, self.head_dim)

        # queries, keys, vals = self.pos_embed(queries, keys, vals)

        output = self.attn_fn(queries, keys, vals, attn_mask=self.attn_mask)

        output = output.view(batchsize, seqlen, -1)

        return self.out_proj(output)


class Block(nn.Module):
    def __init__(self, layer_id, args: Params, attn_mask=None):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = CustomAttn(layer_id, args, attn_mask)

        # this follows llama / lit llama -- go to multiple of 256
        # hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)

        # self.feed_forward = xops.SwiGLU(args.dim, hidden_dim, args.dim, bias=False)
        self.layer_id = layer_id
        self.attention_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.attention.seq_len = args.seq_len

        '''
        # initialize weights trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(
            self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std
        )
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(hidden_dim)
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(
            self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std
        )
        '''
        mlp_width = int(args.dim * 4)
        self.feed_forward = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(args.dim, mlp_width)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(mlp_width, args.dim))
        ]))

        fc_std = (2 * args.dim) ** -0.5
        attn_std = args.dim ** -0.5
        proj_std = (args.dim ** -0.5) * ((2 * args.n_layers) ** -0.5)

        torch.nn.init.normal_(self.feed_forward.c_fc.weight, std=fc_std)
        torch.nn.init.normal_(self.feed_forward.c_proj.weight, std=proj_std)


    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params, attn_mask=None):
        super().__init__()
        # for convenience we often share param names with llama
        self.params = params
        self.n_layers = params.n_layers
        self.seq_len = params.seq_len
        self.post_embed_norm = (
            params.norm_type(
                params.dim,
                eps=params.norm_eps,
            )
            if params.post_embed_norm
            else nn.Identity()
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(Block(layer_id, params, attn_mask))

        self.grad_checkpointing = False

        # get class for normalization layers
        '''
        self.norm = params.norm_type(
            params.dim,
            eps=params.norm_eps,
        )
        '''

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x):
        # x = self.tok_embeddings(input)
        x = self.post_embed_norm(x)

        for layer in self.layers:
            if self.grad_checkpointing:
                x = checkpoint(layer, x)
            else:
                x = layer(x)

        # x = self.norm(x)
        return x
