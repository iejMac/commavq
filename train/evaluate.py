import torch
import torch.distributed as dist

from torch.nn import functional as F
from model import N_FRAME_TOKENS

from distributed import is_master


def compute_perplexity(model, args):
    with torch.no_grad():
        temp_cu = model.codebook_used.clone() if args.distributed else None
        dist.reduce(model.codebook_used, dst=0) if args.distributed else None

        perplexity = 0.0
        if not args.distributed or is_master(args):
            avg_probs = model.codebook_used / model.codebook_used.sum()
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if args.distributed and is_master(args):
            model.codebook_used = temp_cu
        if args.reinit_unused_codebook_frequency == -1:
            model.codebook_used *= 0.0
        return perplexity


def compute_acc_metrics(pred, X, ns=[1], split="train"):
    acc_log = {}


    for n in ns:
        xn = X[:, n].reshape(X.shape[0], -1).long()
        xnm1 = X[:, n-1].reshape(X.shape[0], -1).long()
        pred_n = pred[:, (n-1)*N_FRAME_TOKENS:(n)*N_FRAME_TOKENS]

        pred_xn_eq_xnm1 = (pred_n == xnm1)
        pred_eq_xn = (pred_n == xn)
        xnm1_eq_xn = (xnm1 == xn)

        pred_xn_xnm1_acc = (pred_xn_eq_xnm1).sum()/xnm1.numel()
        pred_xn_acc = (pred_eq_xn).sum()/xn.numel()
        pred_xn_n_xnm1_acc = (pred_eq_xn * ~(xnm1_eq_xn)).sum()/(xn.numel() - xnm1_eq_xn.sum())
        xnm1_xn_eq = (xnm1_eq_xn).sum()/xn.numel()
        
        acc_log[f"{split}/pred_x{n}_x{n-1}_acc"] = pred_xn_xnm1_acc.item()
        acc_log[f"{split}/pred_x{n}_acc"] = pred_xn_acc.item()
        acc_log[f"{split}/pred_x{n}_n_x{n-1}_acc"] = pred_xn_n_xnm1_acc.item()
        acc_log[f"{split}/x{n-1}_x{n}_eq"] = xnm1_xn_eq.item()
    return acc_log


def compute_usage_loss(model, X, split="train"):
    usage_log = {}
    xs = X[:, :-1].long()
    xs = xs.reshape(xs.shape[0], xs.shape[1], -1)
    prep_labels = X[:, 1:].reshape(X.shape[0], -1).reshape(-1)
    
    with torch.no_grad():
        fake_encodings = torch.randint(0, model.quantizer.n_embeddings, (xs.shape[0], model.n_dynamics_tokens * (model.encoder.n_frames - 1))).long().to(X.device)
        fake_f = model.diff_proj(model.quantizer.embedding.weight[fake_encodings])
        fake_xs = torch.randint(0, 1024, xs.shape).long().to(X.device)
        f = model.diff_proj(model.encode_diff(X))
        f, _, _, _ = model.quantizer(f)
        fake_f_logits = model.decode(xs, fake_f)
        fake_xs_logits = model.decode(fake_xs, f)

        fake_f_prep_logits = fake_f_logits.reshape(-1, 1024)
        fake_xs_prep_logits = fake_xs_logits.reshape(-1, 1024)
        unused_f_loss = F.cross_entropy(fake_f_prep_logits, prep_labels)
        unused_xs_loss = F.cross_entropy(fake_xs_prep_logits, prep_labels)
        usage_log[f"{split}/unused_f_loss"] = unused_f_loss.item()
        usage_log[f"{split}/unused_xs_loss"] = unused_xs_loss.item()

    return usage_log 


def evaluate_model(model, val_dataloader, n_steps):
    val_log = {
        "val/reco_loss": 0.0,
        "val/unused_f_loss": 0.0,
        "val/unused_xs_loss": 0.0,
    }

    ns = [1]
    if model.encoder.n_frames > 2:
        ns.append(model.encoder.n_frames - 1)

    for n in ns:
        val_log[f"val/pred_x{n}_x{n-1}_acc"] = 0.0
        val_log[f"val/pred_x{n}_acc"] = 0.0
        val_log[f"val/pred_x{n}_n_x{n-1}_acc"] = 0.0
        val_log[f"val/x{n-1}_x{n}_eq"] = 0.0

    i = 0
    with torch.no_grad():
        for X in val_dataloader:
            if i >= n_steps:
                break

            step_log = {}

            X = X.long().to(model.device)
            labels = X[:, 1:].reshape(X.shape[0], -1)

            true_logits, latent_info = model(X)
            prep_logits, prep_labels = true_logits.reshape(-1, 1024), labels.reshape(-1)
            reco_loss = F.cross_entropy(prep_logits, prep_labels)
            step_log["val/reco_loss"] = reco_loss.item()

            step_log.update(compute_acc_metrics(true_logits.argmax(dim=-1), X, ns, "val"))
            step_log.update(compute_usage_loss(model, X, "val"))

            for k, v in step_log.items():
                val_log[k] += v

            i += 1

    for k, v in val_log.items():
        val_log[k] /= n_steps
    
    return val_log
