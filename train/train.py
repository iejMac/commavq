import torch
import time
import numpy as np
import wandb

from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from torch import optim

from model import VQVideo, N_FRAME_TOKS
from dataloader import TokenLoader
from distributed import is_master, init_distributed_device


class AttrDict(dict):
    """
    Lets us access dict keys with <dict>.key
    """

    # pylint: disable=super-with-arguments
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if  __name__ == "__main__":
    '''
    num_proc = 40 # CPUs go brrrr
    ds = load_dataset('commaai/commavq', num_proc=num_proc)


    tokens = np.load(ds['0'][0]['path']) # first segment from the first data shard
    '''
    # Input: [b, 2, N_FRAME_TOKS]
    # Flattened: [b, 256]

    # Bottleneck: [b, N_FRAME_TOKS + s] : s < 128 n Bottleneck[:128] = Input[:128]
    # I.e. Transformation_Code = Encoder(Input): [b, s]
    #      Output = Decoder(Input[:N_FRAME_TOKS] + Transformation_Code): [b, 128]

    # Output: [b, 2, N_FRAME_TOKS]

    '''
    For now try to get results with constant bottleneck
    i.e. try to get some encoder model that can encode the diffs in 50% less tokens

    Next try variable length codings based on difference between tokens of frames
    Need to add delimiter tokens <|X1|> abcd <|F|> 12 <|X2>| xzyw <|EOT|> pad pad pad 


    Current Objective:
        Get the model to learn the spatial embedding table by passing it through the dynamics
        bottleneck

        N_DYN_TOKS == N_SPATIAL_TOKS

    '''

    args = AttrDict(
        dist_backend="nccl",
        dist_url="env://",
        no_set_device_rank=False,
    )

    device = init_distributed_device(args)

    '''
    # TODO: compare this
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    '''

    # Logging
    enable_wandb = True and is_master(args)

    if enable_wandb:
        wandb.init(
            project="diff-encoding",
        )

    # Data Prep
    spatial_embeddings = torch.load("embedding.pt").to(device)
    spatial_embeddings.requires_grad = False

    batch_size = 16
    n_frames = 2
    n_dynamics_tokens = 64
    dataloader = TokenLoader('commavq-mini.npy', batch_size, n_frames=n_frames)
    # dataloader = TokenLoader('commavq.npy', batch_size, n_frames=n_frames)

    # Model Prep
    model = VQVideo(
        n_dynamics_toks = n_dynamics_tokens,
        n_frames = n_frames,
        spatial_embeddings = spatial_embeddings,
    ).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # Opt Prep
    iters = 10000000

    opt = optim.AdamW(model.parameters())

    i = 0
    t0 = time.time()
    for X in dataloader:
        if i >= iters:
            break
        X = X.long().to(device)
        labels = X[:, 1:].reshape(X.shape[0], -1)

        data_time = time.time() - t0

        # Forward pass
        opt.zero_grad()

        true_logits = model(X)

        prep_logits, prep_labels = true_logits.reshape(-1, 1024), labels.reshape(-1)
        reco_loss = F.cross_entropy(prep_logits, prep_labels)
        # latent_loss = model.compute_latent_loss(f_emb, f)
        
        # loss = reco_loss + latent_loss
        # loss = latent_loss
        loss = reco_loss

        loss.backward()
        opt.step()

        batch_time = time.time() - t0

        log = {
            "perf/step": i,
            "perf/data_time": data_time,
            "perf/batch_time": batch_time,
            "perf/tokens_s_gpu": X.numel()/batch_time,
        }

        # Check if you're using f embedding
        x0 = X[:, 0].reshape(X.shape[0], -1).long()
        if is_master(args):
            with torch.no_grad():
                fake_f = torch.randn((batch_size, n_dynamics_tokens, 256)).to(device)
                mod = model.module if args.distributed else model
                fake_logits = mod.decode(x0, fake_f)

                fake_prep_logits = fake_logits.reshape(-1, 1024)
                unused_f_loss = F.cross_entropy(fake_prep_logits, prep_labels)
                log['train/unused_f_loss'] = unused_f_loss.item()
    

        pred = true_logits.argmax(dim=-1)
        x0 = x0
        x1 = labels 
        pred_x0_acc = (pred == x0).sum()/x0.numel()
        pred_x1_acc = (pred == x1).sum()/x1.numel()
        x0_x1_eq = (x0 == x1).sum()/x1.numel()

        log["train/reco_loss"] = reco_loss.item()
        log["train/pred_x0_acc"] = pred_x0_acc.item()
        log["train/pred_x1_acc"] = pred_x1_acc.item()
        log["train/x0_x1_eq"] = x0_x1_eq.item()

        if is_master(args):
            print(f"Step {i}")
            print("--------")
            for name, val in log.items():
                print(f"{name}: {val}")
            if enable_wandb:
                wandb.log(log)

        i += 1
        t0 = time.time()

    last_pred = pred[0]
    last_x0 = x0[0]
    last_x1 = x1[0]

    print('====== X0')
    print(last_x0)
    print('====== X1')
    print(last_x1)
    print('====== PRED')
    print(last_pred)
    print('======')
    print("pred - x0")
    print((last_pred == last_x0).sum()/last_x0.numel())
    print("pred - x1")
    print((last_pred == last_x1).sum()/last_x1.numel())
    print("x0 - x1")
    print((last_x0 == last_x1).sum()/last_x1.numel())

