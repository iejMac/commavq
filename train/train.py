import torch
import time
import numpy as np
import wandb

from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from torch import optim

from dataloader import TokenLoader
from distributed import is_master, init_distributed_device
from evaluate import compute_acc_metrics, compute_usage_loss, evaluate_model
from model import VQVideo, EncoderConfig, DecoderConfig, QuantizerConfig, N_FRAME_TOKENS


class AttrDict(dict):
    """
    Lets us access dict keys with <dict>.key
    """

    # pylint: disable=super-with-arguments
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if  __name__ == "__main__":
    args = AttrDict(
        dist_backend="nccl",
        dist_url="env://",
        no_set_device_rank=False,
    )

    device = init_distributed_device(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Logging
    enable_wandb = True and is_master(args)
    log_every_n_steps = 1
    soft_eval_every_n_steps = 100
    eval_every_n_steps, validation_steps = 5000, 100
    save_checkpoint_n_steps = -1

    if enable_wandb:
        wandb.init(
            project="diff-encoding",
        )

    # Data Prep
    batch_size = 128
    n_frames = 2
    # train_dataloader = TokenLoader('datasets/commavq-mini.npy', batch_size, n_frames=n_frames)
    train_dataloader = TokenLoader('datasets/commavq-train.npy', batch_size, n_frames=n_frames)
    val_dataloader = TokenLoader('datasets/commavq-val.npy', batch_size, n_frames=n_frames)

    # Model Prep
    common_width = 256

    n_dynamics_tokens = 64
    quantized_width = 256

    spatial_embeddings = torch.load("embedding.pt")
    spatial_embeddings.requires_grad = False

    encoder_config = EncoderConfig(
        width=common_width,
        layers=8,
        heads=8,
        n_input_tokens=2*N_FRAME_TOKENS + 2,
        n_dynamics_tokens=n_dynamics_tokens,
        output_dim=quantized_width,
    )
    decoder_config = DecoderConfig(
        width=common_width,
        layers=8,
        heads=8,
        n_input_tokens=N_FRAME_TOKENS + n_dynamics_tokens + 2,
        n_dynamics_tokens=n_dynamics_tokens,
        weight_tying=False,
    )
    quantizer_config = QuantizerConfig(
        n_embeddings=1024,
        embedding_dim=quantized_width,
        commitment_cost=0.25,
    )

    model = VQVideo(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        quantizer_config=quantizer_config,
        spatial_embeddings = spatial_embeddings,
    ).to(device)
    model.device = device

    if is_master(args):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {n_params} parameters")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # Opt Prep
    iters = 100000
    grad_clip_norm = -1
    reinit_unused_codebook_steps = 1000

    opt = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.98),  # (0.9, 0.999)
        eps=1e-6,  # 1e-8
    )

    i = 0
    t0 = time.time()

    for X in train_dataloader:
        if i >= iters:
            break
        X = X.long().to(device)
        labels = X[:, 1:].reshape(X.shape[0], -1)

        data_time = time.time() - t0

        # Forward pass
        opt.zero_grad()

        true_logits, latent_info = model(X)

        prep_logits, prep_labels = true_logits.reshape(-1, 1024), labels.reshape(-1)
        reco_loss = F.cross_entropy(prep_logits, prep_labels)
        latent_loss = latent_info['latent_loss']
        
        loss = reco_loss + latent_loss

        loss.backward()

        # Clip gradients
        if grad_clip_norm != -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
        # Compute gradient norm
        grad_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), 2.0).to(device)
                for p in model.parameters() if p.grad is not None
            ]),
            2.0,
        )

        opt.step()

        if (reinit_unused_codebook_steps != -1) and ((i+1) % reinit_unused_codebook_steps == 0):
            mod = model.module if args.distributed else model
            mod.quantizer.reinit_unused_codebook(latent_info['encodings'], args)

        batch_time = time.time() - t0

        log = {
            "perf/step": i,
            "perf/data_time": data_time,
            "perf/batch_time": batch_time,
            "perf/tokens_s_gpu": X.numel()/batch_time,
            "train/reco_loss": reco_loss.item(),
            "train/latent_loss": latent_loss.item(),
            "train/perplexity": latent_info['perplexity'].item(),
            "train/grad_norm": grad_norm.item(),
        }

        # Evals
        acc_logs = compute_acc_metrics(true_logits.argmax(dim=-1), X, "train")
        log.update(acc_logs)
        # Check if you're using f embedding
        if ((i+1) % soft_eval_every_n_steps == 0) and is_master(args):
            mod = model.module if args.distributed else model
            usage_log = compute_usage_loss(mod, X)
            log.update(usage_log)
        if (eval_every_n_steps != -1) and ((i+1) % eval_every_n_steps == 0):
            mod = model.module if args.distributed else model
            val_log = evaluate_model(mod, val_dataloader, validation_steps)
            log.update(val_log)

        # Checkpointing
        if (save_checkpoint_n_steps != -1) and ((i+1) % save_checkpoint_n_steps == 0):
            torch.save(model.state_dict(), 'latest_vq_video.pth')

        if ((i+1) % log_every_n_steps == 0) and is_master(args):
            print(f"Step {i}")
            print("--------")
            for name, val in log.items():
                print(f"{name}: {val}")
            if enable_wandb:
                wandb.log(log)

        i += 1
        t0 = time.time()
