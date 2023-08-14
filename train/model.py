import torch
import math
import numpy as np
import torch.distributed as dist

from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

from distributed import is_master
from transformer import Transformer, Params


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings


N_FRAME_TOKENS = 128

@dataclass
class EncoderConfig:
    width: int = 256
    layers: int = 8
    heads: int = 8
    n_dynamics_tokens: int = 64
    n_frames: int = 2
    output_dim: int = -1

class Encoder(nn.Module):
    def __init__(self, width, layers, heads, n_dynamics_tokens, n_frames, output_dim):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.n_dynamics_tokens = n_dynamics_tokens
        self.n_frames = n_frames
        self.output_dim = output_dim


        n_input_tokens = n_frames * (N_FRAME_TOKENS + 1)
        transformer_params = Params(
            dim=width,
            n_layers=layers,
            n_heads=heads,
            norm_eps=1e-5,
            seq_len=n_input_tokens,
            post_embed_norm=False,
        )
        self.transformer = Transformer(transformer_params)

        # self.transformer = Transformer(width, layers, heads)

        scale = width ** -0.5
        self.frame_delim = nn.Parameter(torch.randn(width) * scale)

        # self.pos_emb = nn.Embedding(n_frames*N_FRAME_TOKENS + n_frames, width)
        self.pos_emb = sinusoidal_positional_embedding(n_input_tokens, width)
        self.pos_emb.requires_grad=False
        self.output_dim = width if output_dim == -1 else output_dim

        start_indices = torch.arange(N_FRAME_TOKENS + 1, self.n_frames * (N_FRAME_TOKENS + 1), N_FRAME_TOKENS + 1)
        dynamics_inds = torch.stack([start_indices + i for i in range(self.n_dynamics_tokens)]).T.reshape(-1)
        self.register_buffer('dynamics_inds', dynamics_inds, persistent=False)

        self.proj = nn.Linear(self.width, self.output_dim, bias=False)
        self.init_parameters()

    def init_parameters(self):
        # nn.init.normal_(self.pos_emb.weight, std=0.01)

        '''
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # nn.init.normal_(self.proj.weight, mean=0.0, std=proj_std)
        '''

    def forward(self, embs):
        embs = torch.cat((
            embs,
            self.frame_delim + torch.zeros(embs.shape[0], embs.shape[1], 1, embs.shape[-1], dtype=embs.dtype, device=embs.device),
        ), dim=-2)
        embs = embs.reshape(embs.shape[0], -1, embs.shape[-1])

        pos = torch.arange(0, embs.shape[1], dtype=torch.long, device=embs.device).unsqueeze(0).to(embs.device)
        # p_embs = self.pos_emb(pos)
        self.pos_emb = self.pos_emb.to(embs.device)
        p_embs = self.pos_emb[pos]

        t_embs = embs + p_embs
        # t_embs = embs

        c_embs = self.transformer(t_embs)

        # c_embs = self.ln_final(c_embs)
        c_embs = self.proj(c_embs)

        f = c_embs[:, self.dynamics_inds]
        return f


@dataclass
class DecoderConfig:
    width: int = 256
    layers: int = 8
    heads: int = 8
    n_dynamics_tokens: int = 64
    n_frames: int = 2
    weight_tying: bool = False
    spatial_embeddings: torch.Tensor = None

class Decoder(nn.Module):
    def __init__(self, width, layers, heads, n_dynamics_tokens, n_frames, weight_tying, spatial_embeddings=None, precision="amp"):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.n_dynamics_tokens = n_dynamics_tokens
        self.n_frames = n_frames
        self.weight_tying = weight_tying


        n_input_tokens = (n_frames - 1) * (N_FRAME_TOKENS + 1 + n_dynamics_tokens + 1)
        attn_mask = self.build_attention_mask(N_FRAME_TOKENS, n_dynamics_tokens, n_frames-1)
        
        # TODO: only when amp
        attn_mask = attn_mask.to(dtype=torch.bfloat16) if precision == "amp" else attn_mask
        transformer_params = Params(
            dim=width,
            n_layers=layers,
            n_heads=heads,
            norm_eps=1e-5,
            seq_len=n_input_tokens,
            post_embed_norm=False,
        )
        self.transformer = Transformer(transformer_params, attn_mask=attn_mask)

        self.final_proj = None
        self.pred_head = nn.Linear(width, 1024, bias=False)
        self.weight_tying = weight_tying
        if self.weight_tying:
            print("Tying prediction head to spatial embedding table...")
            if self.width != 256:
                self.final_proj = nn.Linear(self.width, 256, bias=False)
            self.pred_head.weight = nn.Parameter(spatial_embeddings)
            self.pred_head.requires_grad = False

        scale = width ** -0.5
        self.frame_delim = nn.Parameter(torch.randn(width) * scale)
        # self.pos_emb = nn.Embedding(n_input_tokens, width)
        self.pos_emb = sinusoidal_positional_embedding(n_input_tokens, width)
        self.pos_emb.requires_grad=False


        start_indices = torch.arange(0, (n_frames - 1) * (N_FRAME_TOKENS + n_dynamics_tokens + 2), (N_FRAME_TOKENS + n_dynamics_tokens + 2))
        logit_inds = torch.stack([start_indices + i for i in range(N_FRAME_TOKENS)]).T.reshape(-1)
        self.register_buffer('logit_inds', logit_inds, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        # TODO: try init here like karpathy (this is decoder)
        # nn.init.normal_(self.pos_emb.weight, std=0.01)
        '''

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        '''
        # torch.nn.init.normal_(self.pred_head.weight, mean=0.0, std=0.02/math.sqrt(2*self.transformer.layers))
        torch.nn.init.normal_(self.pred_head.weight, mean=0.0, std=0.02/math.sqrt(2*self.layers))
        # torch.nn.init.normal_(self.pred_head.weight, mean=0.0, std=proj_std)

    def build_attention_mask(self, f, t, n):
        '''
        # block level
        mask = torch.full((seq_len, seq_len), float('-inf'))
        # Block diagonal mask should have ones (i.e., zero out after fill with '-inf')
        for i in range(0, seq_len, block_size):
            mask[i:i+block_size, :i+block_size] = 0  # unmask block diagonal
        '''
        seq_len = n * (f + t + 2)
        mask = torch.full((seq_len, seq_len), float('-inf'))

        # Allow each token to attend to itself by default
        for idx in range(seq_len):
            mask[idx, idx] = 0

        for i in range(n):
            # start index for each F sequence
            f_start = i * (f + t + 2)
            
            # unmask positions in the F sequence itself
            mask[f_start:f_start+f, f_start:f_start+f] = 0
            
            # unmask positions in the previous D (if it exists)
            if f_start - 1 >= 0:
                mask[f_start:f_start+f, f_start-1] = 0
            
            # unmask positions in all previous D, T sequences
            for j in range(i):
                d_start = j * (f + t + 2) + f
                mask[f_start:f_start+f, d_start:d_start+t+1] = 0

            # unmask positions in the following D, T sequence
            next_d_start = f_start + f
            mask[f_start:f_start+f, next_d_start:next_d_start+t+1] = 0

        return mask
        
    def forward(self, x, f):
        f = f.reshape(f.shape[0], -1, self.n_dynamics_tokens, self.width)

        fx = torch.cat((
            x,
            self.frame_delim + torch.zeros(f.shape[0], f.shape[1], 1, f.shape[-1], dtype=f.dtype, device=f.device),
            f,
            self.frame_delim + torch.zeros(f.shape[0], f.shape[1], 1, f.shape[-1], dtype=f.dtype, device=f.device),
        ), dim=-2).reshape(x.shape[0], -1, x.shape[-1])

        pos = torch.arange(0, fx.shape[1], dtype=torch.long, device=x.device).unsqueeze(0).to(x.device)
        self.pos_emb = self.pos_emb.to(x.device)
        # p_embs = self.pos_emb(pos)
        p_embs = self.pos_emb[pos]

        fx = fx + p_embs

        y = self.transformer(fx)

        if self.final_proj is not None:
            y = self.final_proj(y)

        logits = self.pred_head(y)
        true_logits = logits[:, self.logit_inds]
        return true_logits


@dataclass
class QuantizerConfig:
    n_embeddings: int = 1024
    embedding_dim: int = 256
    commitment_cost: float = 0.25
    usage_threshold: float = 0.0

class Quantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost, usage_threshold=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.commitment_cost = commitment_cost
        self.usage_threshold = usage_threshold

        self.norm = lambda x: F.normalize(x, dim=-1)
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)

        # Counter variable which contains the number of times each codebook is used
        self.register_buffer('codebook_used', torch.zeros(self.n_embeddings), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        # self.embedding.weight.data.uniform_(-1/self.n_embeddings, 1/self.n_embeddings)
        self.embedding.weight.data.normal_()

    def reinit_unused_codebook(self, dist_args=None):
        # TODO: cleanup dist code
        reinit = torch.empty_like(self.embedding.weight, device=dist_args.device) 
        used = torch.ones(self.n_embeddings, dtype=torch.bool, device=dist_args.device)

        if dist_args.distributed:
            dist.reduce(self.codebook_used, dst=0)

        with torch.no_grad():
            if not dist_args.distributed or is_master(dist_args):
                if self.codebook_used.sum() != 0.0:
                     avg_probs = self.codebook_used / self.codebook_used.sum()

                     used = (avg_probs > self.usage_threshold)
                     if used.sum() != self.n_embeddings:
                         used_vecs = self.embedding.weight[used]
                         samples = torch.randint(high=used.sum(), size=(reinit.shape[0],)).to(reinit.device)
                         reinit = used_vecs[samples]
                         reinit += torch.normal(mean=0.0, std=1e-12, size=reinit.shape).to(reinit.device)
                         print(f"Reinitialized {(~used).sum()} unused embeddings")

            if dist_args.distributed:
                dist.broadcast(reinit, src=0)
                dist.broadcast(used, src=0)

            if (~used).sum() > 0.0:
                self.embedding.weight[~used] = reinit[~used]
            # TODO: maybe gradients for those need to be 0'd out for safety?
            # TODO: optimizer states?
        # Reset counter
        self.codebook_used *= 0.0

    def forward(self, f_emb):
        assert f_emb.shape[-1] == self.embedding_dim
        flat_input = f_emb.reshape(-1, self.embedding_dim)

        flat_input_norm = self.norm(flat_input)
        embedding_norm = self.norm(self.embedding.weight)

        # Calculate distances
        distances = (torch.sum(flat_input_norm**2, dim=1, keepdim=True) 
                    + torch.sum(embedding_norm**2, dim=1)
                    - 2 * torch.matmul(flat_input_norm, embedding_norm.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=f_emb.device)
        encodings.scatter_(1, encoding_indices, 1)
        if self.training:
            with torch.no_grad():
                used = torch.bincount(encoding_indices.flatten(), minlength=self.n_embeddings)
                self.codebook_used += used

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(f_emb.shape)
        quantized, f_emb = self.norm(quantized), self.norm(f_emb)

        e_latent_loss = F.mse_loss(quantized.detach(), f_emb)
        q_latent_loss = F.mse_loss(quantized, f_emb.detach())
        latent_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = f_emb + (quantized - f_emb).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, latent_loss, perplexity, encodings


class VQVideo(nn.Module):
    def __init__(self, encoder_config, decoder_config, quantizer_config, spatial_embeddings, precision="amp"):
        super().__init__()
        self.width = decoder_config.width
        self.n_dynamics_tokens = encoder_config.n_dynamics_tokens

        self.register_buffer('spatial_embeddings', spatial_embeddings, persistent=False)

        self.frame_proj = nn.Linear(256, self.width, bias=False)
        self.diff_proj = nn.Linear(quantizer_config.embedding_dim, self.width, bias=False)

        self.encoder = Encoder(
            width=encoder_config.width,
            layers=encoder_config.layers,
            heads=encoder_config.heads,
            n_dynamics_tokens=encoder_config.n_dynamics_tokens,
            n_frames=encoder_config.n_frames,
            output_dim=quantizer_config.embedding_dim,
        )
        self.decoder = Decoder(
            width=decoder_config.width,
            layers=decoder_config.layers,
            heads=decoder_config.heads,
            n_dynamics_tokens=decoder_config.n_dynamics_tokens,
            n_frames=decoder_config.n_frames,
            weight_tying=decoder_config.weight_tying,
            spatial_embeddings=spatial_embeddings,
            precision=precision,
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
        true_logits = self.decoder(x, f)
        return true_logits

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1) # flatten representation
        embs = self.spatial_embeddings[x]
        embs = self.frame_proj(embs)

        f_emb = self.encoder(embs)

        f, latent_loss, ppl, encodings = self.quantizer(f_emb)
        latent_info = {'latent_loss': latent_loss, 'perplexity': ppl, 'encodings': encodings}
        f = self.diff_proj(f)

        xs = x[:, :-1]
        logits = self.decode(xs, f)

        return logits, latent_info
