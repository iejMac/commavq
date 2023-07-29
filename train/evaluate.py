import torch

from torch.nn import functional as F
from model import N_FRAME_TOKENS


def compute_acc_metrics(pred, X, ns=[1], split="train"):
    acc_log = {}

    x0 = X[:, 0].reshape(X.shape[0], -1).long()

    for n in ns:
        xn = X[:, n].reshape(X.shape[0], -1).long()
        pred_n = pred[:, (n-1)*N_FRAME_TOKENS:(n)*N_FRAME_TOKENS]

        pred_xn_eq_x0 = (pred_n == x0)
        pred_eq_xn = (pred_n == xn)
        x0_eq_xn = (x0 == xn)

        pred_xn_x0_acc = (pred_xn_eq_x0).sum()/x0.numel()
        pred_xn_acc = (pred_eq_xn).sum()/xn.numel()
        pred_xn_n_x0_acc = (pred_eq_xn * ~(x0_eq_xn)).sum()/(xn.numel() - x0_eq_xn.sum())
        x0_xn_eq = (x0_eq_xn).sum()/xn.numel()
        
        acc_log[f"{split}/pred_x{n}_x0_acc"] = pred_xn_x0_acc.item()
        acc_log[f"{split}/pred_x{n}_acc"] = pred_xn_acc.item()
        acc_log[f"{split}/pred_x{n}_n_x0_acc"] = pred_xn_n_x0_acc.item()
        acc_log[f"{split}/x0_x{n}_eq"] = x0_xn_eq.item()
    return acc_log


def compute_usage_loss(model, X, split="train"):
    usage_log = {}
    x0 = X[:, 0].reshape(X.shape[0], -1).long()
    prep_labels = X[:, 1:].reshape(X.shape[0], -1).reshape(-1)
    
    with torch.no_grad():
        fake_encodings = torch.randint(0, model.quantizer.n_embeddings, (x0.shape[0], model.n_dynamics_tokens * (model.encoder.n_frames - 1))).long().to(X.device)
        fake_f = model.diff_proj(model.quantizer.embedding.weight[fake_encodings])
        fake_x0 = torch.randint(0, 1024, x0.shape).long().to(X.device)
        f = model.diff_proj(model.encode_diff(X))
        f, _, _, _ = model.quantizer(f)
        fake_f_logits = model.decode(x0, fake_f)
        fake_x0_logits = model.decode(fake_x0, f)

        fake_f_prep_logits = fake_f_logits.reshape(-1, 1024)
        fake_x0_prep_logits = fake_x0_logits.reshape(-1, 1024)
        unused_f_loss = F.cross_entropy(fake_f_prep_logits, prep_labels)
        unused_x0_loss = F.cross_entropy(fake_x0_prep_logits, prep_labels)
        usage_log[f"{split}/unused_f_loss"] = unused_f_loss.item()
        usage_log[f"{split}/unused_x0_loss"] = unused_x0_loss.item()

    return usage_log 


def evaluate_model(model, val_dataloader, n_steps):
    val_log = {
        "val/reco_loss": 0.0,
        "val/unused_f_loss": 0.0,
        "val/unused_x0_loss": 0.0,
    }

    ns = [1]
    if model.encoder.n_frames > 2:
        ns.append(model.encoder.n_frames - 1)

    for n in ns:
        val_log[f"val/pred_x{n}_x0_acc"] = 0.0
        val_log[f"val/pred_x{n}_acc"] = 0.0
        val_log[f"val/pred_x{n}_n_x0_acc"] = 0.0
        val_log[f"val/x0_x{n}_eq"] = 0.0

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
