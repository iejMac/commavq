import torch
import math
import numpy as np

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

from transformer import Transformer


N_FRAME_TOKENS = 128

@dataclass
class EncoderConfig:
    width: int = 256
    layers: int = 8
    heads: int = 8
    n_input_tokens: int = 2*N_FRAME_TOKENS + 2
    n_dynamics_tokens: int = 64
    output_dim: int = -1

class Encoder(nn.Module):
    def __init__(self, width, layers, heads, n_input_tokens, n_dynamics_tokens, output_dim):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.n_input_tokens = n_input_tokens
        self.n_dynamics_tokens = n_dynamics_tokens
        self.output_dim = output_dim

        self.transformer = Transformer(width, layers, heads)

        scale = width ** -0.5
        self.frame_delim = nn.Parameter(torch.randn(width) * scale)

        self.pos_emb = nn.Embedding(n_input_tokens, width)
        self.output_dim = width if output_dim == -1 else output_dim

        hidden_size = (self.width + self.output_dim) // 2
        self.proj = nn.Sequential(
            nn.Linear(self.width, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, self.output_dim, bias=False),
        )
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.pos_emb.weight, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, embs):
        embs = torch.cat((embs,
            self.frame_delim + torch.zeros(embs.shape[0], embs.shape[1], 1, embs.shape[-1], dtype=embs.dtype, device=embs.device)
        ), dim=-2)
        embs = embs.reshape(embs.shape[0], -1, embs.shape[-1])

        pos = torch.arange(0, embs.shape[1], dtype=torch.long, device=embs.device).unsqueeze(0) 
        p_embs = self.pos_emb(pos)

        t_embs = embs + p_embs

        t_embs = t_embs.permute(1, 0, 2)  # NLD -> LND
        c_embs = self.transformer(t_embs)
        c_embs = c_embs.permute(1, 0, 2)  # LND -> NLD

        c_embs = self.proj(c_embs)

        # TODO: very weakly matters but changes loss curve so I'll keep this
        f = c_embs[:, :self.n_dynamics_tokens]  # transformation is bottlenecked
        # f = c_embs[:, -self.n_dynamics_tokens:]  # transformation is bottlenecked
        return f


@dataclass
class DecoderConfig:
    width: int = 256
    layers: int = 8
    heads: int = 8
    n_input_tokens: int = N_FRAME_TOKENS + 64 + 2
    n_dynamics_tokens: int = 64
    weight_tying: bool = False
    spatial_embeddings: torch.Tensor = None

class Decoder(nn.Module):
    def __init__(self, width, layers, heads, n_input_tokens, n_dynamics_tokens, weight_tying, spatial_embeddings=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.n_input_tokens = n_input_tokens
        self.n_dynamics_tokens = n_dynamics_tokens
        self.weight_tying = weight_tying

        self.transformer = Transformer(width, layers, heads)
        self.pred_head = nn.Linear(width, 1024, bias=False)
        self.weight_tying = weight_tying
        if self.weight_tying:
            self.pred_head.weight = nn.Parameter(spatial_embeddings)
            self.pred_head.requires_grad = False

        scale = width ** -0.5
        self.frame_delim = nn.Parameter(torch.randn(width) * scale)
        self.pos_emb = nn.Embedding(n_input_tokens, width)

        # full_attn_mask = self.build_attention_mask(2 * N_FRAME_TOKENS + n_dynamics_tokens)
        # self.register_buffer('attn_mask', full_attn_mask, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.pos_emb.weight, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        torch.nn.init.normal_(self.pred_head.weight, mean=0.0, std=0.02/math.sqrt(2*self.transformer.layers))

    def build_attention_mask(self, ctx_len):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf

        # TODO: might need to build attn mask every time for dynamic stuff
        # TODO: try out blocked attention masks (frame is atomic element)
        mask = torch.empty(ctx_len, ctx_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def forward(self, x, f):
        fx = torch.cat([
            x,
            self.frame_delim.to(x.dtype)  + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            f,
            self.frame_delim.to(x.dtype)  + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        ], dim=1)  # concat space code with transformation code

        pos = torch.arange(0, fx.shape[1], dtype=torch.long, device=x.device).unsqueeze(0) 
        p_embs = self.pos_emb(pos)

        fx = fx + p_embs

        fx = fx.permute(1, 0, 2)  # NLD -> LND
        # y = self.transformer(fx, attn_mask=self.attn_mask)
        y = self.transformer(fx)
        y = y.permute(1, 0, 2)  # LND -> NLD

        logits = self.pred_head(y)
        return logits


@dataclass
class QuantizerConfig:
    n_embeddings: int = 1024
    embedding_dim: int = 256
    commitment_cost: float = 0.25

class Quantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/n_embeddings, 1/self.n_embeddings)

    def forward(self, f_emb):
        flat_input = f_emb.reshape(-1, self.embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=f_emb.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(f_emb.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), f_emb)
        q_latent_loss = F.mse_loss(quantized, f_emb.detach())
        latent_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = f_emb + (quantized - f_emb).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, latent_loss, perplexity, encoding_indices


class VQVideo(nn.Module):
    def __init__(self, encoder_config, decoder_config, quantizer_config, spatial_embeddings):
        super().__init__()
        self.width = decoder_config.width
        self.n_dynamics_tokens = encoder_config.n_dynamics_tokens

        self.register_buffer('spatial_embeddings', spatial_embeddings, persistent=False)

        hidden_size = (256 + self.width) // 2
        self.frame_proj = nn.Sequential(
            nn.Linear(256, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, self.width, bias=False),
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(quantizer_config.embedding_dim, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, self.width, bias=False),
        )

        self.encoder = Encoder(
            width=encoder_config.width,
            layers=encoder_config.layers,
            heads=encoder_config.heads,
            n_input_tokens=encoder_config.n_input_tokens,
            n_dynamics_tokens=encoder_config.n_dynamics_tokens,
            output_dim=encoder_config.output_dim,
        )
        self.decoder = Decoder(
            width=decoder_config.width,
            layers=decoder_config.layers,
            heads=decoder_config.heads,
            n_input_tokens=decoder_config.n_input_tokens,
            n_dynamics_tokens=decoder_config.n_dynamics_tokens,
            weight_tying=decoder_config.weight_tying,
            spatial_embeddings = spatial_embeddings,
        )
        self.quantizer = Quantizer(
            n_embeddings=quantizer_config.n_embeddings,
            embedding_dim=quantizer_config.embedding_dim,
            commitment_cost=quantizer_config.commitment_cost,
        )

    def encode_diff(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1) # flatten representation
        embs = self.spatial_embeddings[x]
        embs = self.frame_proj(embs)
        return self.encoder(embs)

    def decode(self, x, f):
        x = self.spatial_embeddings[x]
        x = self.frame_proj(x)
        logits = self.decoder(x, f)
        # TODO: very weakly matters but changes loss curve so I'll keep this
        # used to just not work for the second one, now it works fine
        true_logits = logits[:, :N_FRAME_TOKENS]  # for now only one frame
        # true_logits = logits[:, -N_FRAME_TOKENS:]
        return true_logits

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1) # flatten representation
        embs = self.spatial_embeddings[x]
        embs = self.frame_proj(embs)

        f_emb = self.encoder(embs)

        f, latent_loss, ppl, encodings = self.quantizer(f_emb)
        latent_info = {'latent_loss': latent_loss, 'perplexity': ppl, 'encodings': encodings}
        f = self.diff_proj(f)

        x0 = x[:, 0].reshape(x.shape[0], -1).long()
        logits = self.decode(x0, f)

        return logits, latent_info
