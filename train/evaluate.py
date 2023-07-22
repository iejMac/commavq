import torch

from torch.nn import functional as F


def compute_acc_metrics(pred, X, split="train"):
    acc_log = {}

    x0 = X[:, 0].reshape(X.shape[0], -1).long()
    x1 = X[:, 1].reshape(X.shape[0], -1).long()

    pred_eq_x0 = (pred == x0)
    pred_eq_x1 = (pred == x1)
    x0_eq_x1 = (x0 == x1)

    pred_x0_acc = (pred_eq_x0).sum()/x0.numel()
    pred_x1_acc = (pred_eq_x1).sum()/x1.numel()
    pred_x1_n_x0_acc = (pred_eq_x1 * ~(x0_eq_x1)).sum()/(x1.numel() - x0_eq_x1.sum())
    x0_x1_eq = (x0_eq_x1).sum()/x1.numel()
    
    acc_log[f"{split}/pred_x0_acc"] = pred_x0_acc.item()
    acc_log[f"{split}/pred_x1_acc"] = pred_x1_acc.item()
    acc_log[f"{split}/pred_x1_n_x0_acc"] = pred_x1_n_x0_acc.item()
    acc_log[f"{split}/x0_x1_eq"] = x0_x1_eq.item()

    return acc_log


def compute_usage_loss(model, X, split="train"):
    usage_log = {}
    x0 = X[:, 0].reshape(X.shape[0], -1).long()
    prep_labels = X[:, 1:].reshape(X.shape[0], -1).reshape(-1)
    
    with torch.no_grad():
        fake_f = torch.randn((X.shape[0], model.n_dynamics_tokens, model.width)).to(X.device)
        fake_x0 = torch.randint(0, 1024, x0.shape).long().to(X.device)
        f = model.diff_proj(model.encode_diff(X))
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
        "val/pred_x0_acc": 0.0,
        "val/pred_x1_acc": 0.0,
        "val/pred_x1_n_x0_acc": 0.0,
        "val/x0_x1_eq": 0.0,
    }

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

            step_log.update(compute_acc_metrics(true_logits.argmax(dim=-1), X, "val"))
            step_log.update(compute_usage_loss(model, X, "val"))

            for k, v in step_log.items():
                val_log[k] += v

            i += 1

    for k, v in val_log.items():
        val_log[k] /= n_steps
    
    return val_log
