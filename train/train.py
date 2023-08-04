import os
import sys
import json
import torch
import time
import numpy as np
import wandb
import random
import torch.distributed as dist

from datetime import datetime
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from torch import optim

from dataloader import TokenLoader
from distributed import is_master, init_distributed_device
from evaluate import compute_acc_metrics, compute_usage_loss, evaluate_model
from model import VQVideo, EncoderConfig, DecoderConfig, QuantizerConfig, N_FRAME_TOKENS
from params import parse_args


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args):
    args = parse_args(args)

    device = init_distributed_device(args)

    if args.name is None:
        model_name = args.model.split("/")[-1].split(".")[0]
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = '-'.join([
            date_str,
            f"model_{model_name}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
        ])

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Logging
    log_base_path = os.path.join(args.logs, args.name)
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        os.makedirs(log_base_path, exist_ok=True)
        os.makedirs(args.checkpoint_path, exist_ok=True)
    args.enable_wandb = args.enable_wandb and is_master(args)

    if args.enable_wandb:
        wandb.init(
            project="diff-encoding",
            name=args.name,
        )

    # Data Prep
    train_dataloader = TokenLoader(args.train_data, args.batch_size, n_frames=args.n_frames)
    val_dataloader = TokenLoader(args.val_data, args.batch_size, n_frames=args.n_frames)

    # Model Prep
    with open(args.model, "r") as f:
        model_config = json.load(f)

    quantized_width = model_config['quantizer_cfg']['embedding_dim']
    n_dynamics_tokens = model_config['n_dynamics_tokens']

    spatial_embeddings = torch.load(model_config['spatial_embedding'])
    spatial_embeddings.requires_grad = False

    encoder_config = EncoderConfig(
        n_input_tokens=args.n_frames*N_FRAME_TOKENS + args.n_frames,
        n_dynamics_tokens=n_dynamics_tokens,
        output_dim=quantized_width,
        **model_config['encoder_cfg'],
    )
    decoder_config = DecoderConfig(
        n_input_tokens=N_FRAME_TOKENS + n_dynamics_tokens + 2,
        n_dynamics_tokens=n_dynamics_tokens,
        spatial_embeddings=spatial_embeddings if model_config['decoder_cfg']['weight_tying'] else None,
        **model_config['decoder_cfg'],
    )
    quantizer_config = QuantizerConfig(
        **model_config['quantizer_cfg'],
    )

    random_seed(args.seed, 0)
    model = VQVideo(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        quantizer_config=quantizer_config,
        spatial_embeddings = spatial_embeddings,
    ).to(device)
    model.device = device

    random_seed(args.seed, args.rank)

    if is_master(args):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {n_params} parameters")

    if args.distributed:
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # Opt Prep
    
    opt = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.wd,
    )

    start_step = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'step' in checkpoint:
            start_step = checkpoint['step']
            sd = checkpoint['state_dict']
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            opt.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Resuming checkpoint {args.resume} (step {start_step})")

    i = start_step
    t0 = time.time()

    for X in train_dataloader:
        if i > args.iters:
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
        if args.grad_clip_norm != -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        # Compute gradient norm
        grad_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), 2.0).to(device)
                for p in model.parameters() if p.grad is not None
            ]),
            2.0,
        )

        opt.step()

        log = {}

        # TODO: make this work with reinit
        if (i % args.check_usage_frequency == 0):
            with torch.no_grad():
                if args.distributed:
                    mod = model.module.quantizer
                    temp_cu = mod.codebook_used.clone()
                    dist.reduce(mod.codebook_used, dst=0)

                    if is_master(args):
                        avg_probs = mod.codebook_used / mod.codebook_used.sum()
                        mod.codebook_used = temp_cu
                        log["train/perplexity"] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                else:
                    mod = model.quantizer
                    avg_probs = mod.codebook_used / mod.codebook_used.sum()
                    log["train/perplexity"] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                # TODO: doesnt work with reinit since zeros out

                if args.reinit_unused_codebook_frequency == -1:
                    mod.codebook_used *= 0.0

        if (args.reinit_unused_codebook_frequency != -1) and (i % args.reinit_unused_codebook_frequency == 0):
            mod = model.module if args.distributed else model
            mod.quantizer.reinit_unused_codebook(args)

        batch_time = time.time() - t0

        train_log = {
            "perf/step": i,
            "perf/data_time": data_time,
            "perf/batch_time": batch_time,
            "perf/tokens_s_gpu": X.numel()/batch_time,
            "train/reco_loss": reco_loss.item(),
            "train/latent_loss": latent_loss.item(),
            # "train/perplexity": latent_info['perplexity'].item(),
            "train/grad_norm": grad_norm.item(),
        }

        log.update(train_log)

        # Evals
        acc_logs = compute_acc_metrics(true_logits.argmax(dim=-1), X, "train")
        log.update(acc_logs)
        # Check if you're using f embedding and x0 together
        if (i % args.check_usage_frequency == 0) and is_master(args):
            mod = model.module if args.distributed else model
            usage_log = compute_usage_loss(mod, X)
            log.update(usage_log)
        if (args.val_frequency != -1) and (i % args.val_frequency == 0) and is_master(args):
            mod = model.module if args.distributed else model
            val_log = evaluate_model(mod, val_dataloader, args.val_steps)
            log.update(val_log)

        # Checkpointing
        if (args.save_frequency != -1) and (i % args.save_frequency == 0) and is_master(args):
            checkpoint_dict = {
                "step": i,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
            }

            checkpoint_name = os.path.join(args.checkpoint_path, f"step_{'latest' if args.save_most_recent else i}.pth")
            torch.save(checkpoint_dict, checkpoint_name)

        if (i % args.log_every_n_steps == 0) and is_master(args):
            print(f"Step {i}")
            print("--------")
            for name, val in log.items():
                print(f"{name}: {val}")
            if args.enable_wandb:
                wandb.log(log)

        i += 1
        t0 = time.time()

if __name__ == "__main__":
    main(sys.argv[1:])
