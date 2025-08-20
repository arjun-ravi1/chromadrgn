#!/usr/bin/env python
"""
Train a cryoDRGN VAE with known poses + structural conditioning from Chroma.

Usage example (after installing cryoDRGN 3.4.4 and Chroma):

  $ python train_cryoDRGN_with_Chroma.py \
        particles.mrcs \
        --poses angles.pkl \
        --ctf ctf.pkl \
        --protein myprotein.pdb \
        --outdir outs/with_chroma \
        --zdim 8 \
        --lr 1e-4 \
        --num-epochs 20 \
        --batch-size 16 \
        --beta 0.5
"""

import argparse
import os
import pickle
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cryodrgn
from cryodrgn import ctf, dataset
from cryodrgn.beta_schedule import get_beta_schedule
from cryodrgn.lattice import Lattice
from cryodrgn.models import HetOnlyVAE, unparallelize
from cryodrgn.pose import PoseTracker

from chroma.models import Chroma
from chroma import Protein

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Particle images (.mrcs, .star, .cs or .txt)",
    )
    parser.add_argument(
        "--poses",
        type=os.path.abspath,
        required=True,
        help="Pickle file with known poses (quats/Euler + translations)",
    )
    parser.add_argument(
        "--ctf",
        type=os.path.abspath,
        help="Pickle file with CTF parameters",
    )
    parser.add_argument(
        "--protein",
        type=os.path.abspath,
        required=True,
        help="Protein structure/sequence for Chroma (e.g. .pdb or .fasta)",
    )
    parser.add_argument(
        "-o", "--outdir", type=os.path.abspath, required=True, help="Output directory"
    )
    parser.add_argument(
        "--zdim",
        type=int,
        required=True,
        help="Base latent dimension (before adding Chroma embedding)",
    )
    parser.add_argument(
        "--beta",
        default=None,
        help="Beta schedule (name) or float constant",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-n", "--num-epochs", type=int, default=20, help="Number of epochs"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, help="Batch size"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision (torch.cuda.amp)",
    )
    parser.add_argument(
        "--multigpu",
        action="store_true",
        help="Use DataParallel on multiple GPUs",
    )

    # ======== decoder / encoder hyperparams (match cryoDRGN 3.4.4) ========
    group = parser.add_argument_group("Encoder Network")
    group.add_argument(
        "--enc-layers",
        dest="qlayers",
        type=int,
        default=3,
        help="Encoder hidden layers",
    )
    group.add_argument(
        "--enc-dim",
        dest="qdim",
        type=int,
        default=1024,
        help="Encoder hidden dim",
    )
    group.add_argument(
        "--encode-mode",
        choices=("conv", "resid", "mlp", "tilt"),
        default="resid",
    )
    group.add_argument("--enc-mask", type=int, help="Circular mask for encoder")

    group = parser.add_argument_group("Decoder Network")
    group.add_argument(
        "--dec-layers",
        dest="players",
        type=int,
        default=3,
        help="Decoder hidden layers",
    )
    group.add_argument(
        "--dec-dim",
        dest="pdim",
        type=int,
        default=1024,
        help="Decoder hidden dim",
    )
    group.add_argument(
        "--pe-type",
        choices=[
            "geom_ft",
            "geom_full",
            "geom_lowf",
            "geom_nohighf",
            "linear_lowf",
            "gaussian",
            "none",
        ],
        default="gaussian",
    )
    group.add_argument("--feat-sigma", type=float, default=0.5)
    group.add_argument("--pe-dim", type=int, help="Frequencies for positional encoding")
    group.add_argument(
        "--domain", choices=("hartley", "fourier"), default="fourier"
    )
    group.add_argument(
        "--activation", choices=("relu", "leaky_relu"), default="relu"
    )

    # Bravo, done!
    return parser


def train_batch(
    model: nn.Module,
    lattice: Lattice,
    y: torch.Tensor,
    rots: torch.Tensor,
    trans: torch.Tensor,
    ctf_params: Optional[torch.Tensor],
    chroma_cond: torch.Tensor,
    optim: torch.optim.Optimizer,
    beta: float,
    use_amp: bool = False,
    scaler=None,
):
    """
    Single training step:
      1) phase‐flip inputs if CTF
      2) encode -> reparameterize
      3) concat chroma_cond to z
      4) decode with known rots/trans/CTF -> y_recon, mask
      5) MSE + beta * KLD
    """
    model.train()
    optim.zero_grad()

    with torch.cuda.amp.autocast(enabled=use_amp):
        # 1) phase‐flip
        if ctf_params is not None:
            B, D, _ = y.shape
            freqs = lattice.freqs2d.unsqueeze(0).expand(B, D, D) / ctf_params[:, 0].view(
                B, 1, 1
            )
            c = ctf.compute_ctf(
                freqs, *torch.split(ctf_params[:, 1:], 1, dim=1)
            ).view(B, D, D)
            y_in = y * c.sign()
        else:
            y_in = y

        # 2) encode + reparam
        _model = unparallelize(model)
        z_mu, z_logvar = _model.encode(y_in)
        z = _model.reparameterize(z_mu, z_logvar)

        # 3) inject Chroma conditioner (broadcast if needed)
        #    chroma_cond: [1, E] or [B, E]
        if chroma_cond.dim() == 2 and chroma_cond.size(0) == 1:
            z = torch.cat([z, chroma_cond.expand(z.size(0), -1)], dim=1)
        else:
            assert (
                chroma_cond.size(0) == z.size(0)
            ), "Batch size mismatch with chroma condition"
            z = torch.cat([z, chroma_cond], dim=1)

        # 4) decode
        y_recon, mask = model.decode(z, lattice, rots, trans, ctf_params)

        # 5) losses
        # only compute error inside the mask
        mse = F.mse_loss(y_recon * mask, y * mask, reduction="sum") / mask.sum()
        kld = -0.5 * torch.sum(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp()
        ) / y.shape[0]
        loss = mse + beta * kld

    # backward
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

    return loss.item(), mse.item(), kld.item()


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.outdir, "run.log"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.info(f"cryodrgn=={cryodrgn.__version__}  Chroma conditioning")

    # device & mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 1) Load Chroma + Protein + precompute embedding
    chroma = Chroma().to(device)
    # TODO: replace the following stub with the real Chroma API call
    #       that takes your PDB/FASTA and returns a fixed‐size vector.
    #       e.g. chroma_cond = chroma.encode_protein(protein)
    if args.protein.endswith(".pdb"):
        protein = Protein.from_pdb(args.protein)
    else:
        protein = Protein.from_fasta(args.protein)
    protein = protein.to(device)

    with torch.no_grad():
        # stub: assume Chroma has a method `embed_protein` returning tensor [1, E]
        # Replace with whatever your version uses!
        chroma_cond = chroma.embed_protein(protein)  # <-- **YOUR API HERE**
        # Example fallback:
        # chroma_cond = torch.randn(1, chroma.latent_dim, device=device)
    E = chroma_cond.size(1)
    logger.info(f"Using Chroma conditioner of dim E={E}")

    # 2) Load dataset, poses, CTF
    data = dataset.ImageDataset(
        args.particles,
        device=device,
        invert_data=True,
        window=True,
    )
    loader = dataset.make_dataloader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        shuffler_size=0,
        seed=None,
        num_workers=4,
    )

    # load known poses
    pose_tracker = PoseTracker.from_file(args.poses, device=device)
    rots, trans = pose_tracker.rots, pose_tracker.trans  # rots: [N,3,3], trans: [N,2]

    # load CTF params if provided
    ctf_params = None
    if args.ctf:
        with open(args.ctf, "rb") as f:
            ctf_params = torch.from_numpy(pickle.load(f)).to(device)

    # 3) Build lattice & model (expand zdim by E)
    lattice = Lattice(data.D, extent=0.5, device=device)

    full_zdim = args.zdim + E
    model = HetOnlyVAE(
        lattice,
        zdim=full_zdim,
        qlayers=args.qlayers,
        qdim=args.qdim,
        players=args.players,
        pdim=args.pdim,
        encode_mode=args.encode_mode,
        enc_mask=args.enc_mask,
        pe_type=args.pe_type,
        feat_sigma=args.feat_sigma,
        pe_dim=args.pe_dim,
        domain=args.domain,
        activation=args.activation,
        tilt_params=dict(
            tdim=getattr(args, "tdim", None),
            tlayers=getattr(args, "tlayers", None),
            t_emb_dim=getattr(args, "t_emb_dim", None),
            ntilts=getattr(args, "ntilts", None),
        )
        if args.encode_mode == "tilt"
        else None,
    ).to(device)

    if args.multigpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Beta schedule
    beta_sched = get_beta_schedule(args.beta, args.num_epochs)
    # if it returns a list/array, we'll index; else we'll treat as constant

    logger.info("Starting training …")
    for epoch in range(args.num_epochs):
        model.train()
        beta = beta_sched[epoch] if hasattr(beta_sched, "__getitem__") else beta_sched

        for batch in loader:
            y = batch[0].to(device)            # [B, D, D]
            idx = batch[-1].long()             # indices in [0..N)
            rots_b = rots[idx]                 # [B,3,3]
            trans_b = trans[idx]               # [B,2]
            ctf_b = ctf_params[idx] if ctf_params is not None else None

            loss, mse, kld = train_batch(
                model,
                lattice,
                y,
                rots_b,
                trans_b,
                ctf_b,
                chroma_cond,  # broadcast inside train_batch
                optim,
                beta,
                use_amp,
                scaler,
            )
            logger.info(
                f"Epoch {epoch:3d} | Loss {loss:.4e} | MSE {mse:.4e} | KLD {kld:.4e}"
            )

        # checkpoint
        torch.save(
            unparallelize(model).state_dict(),
            os.path.join(args.outdir, f"weights_epoch_{epoch}.pth"),
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
Key points to check/fill in:

Chroma embedding stub
Replace


       chroma_cond = chroma.embed_protein(protein)
       ```  
    with whichever Chroma call actually returns a fixed-size `Tensor` of shape `[1, E]`.  

2.  **PoseTracker format**  
    Make sure your `args.poses` PKL contains the same format (quaternions or rotation matrices + translations) that `PoseTracker` expects.  

3.  **CTF PKL format**  
    We assume `ctf.pkl` is an `N×(1+4)` numpy array where column 0 is defocus and cols 1–4 are CTF coeffs; adjust slicing if yours differs.  

4.  **cryodrgn decode signature**  
    The call  
       
<button class="copy-icon" onclick="copyCode(this)"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><!-- Simplified clipboard icon --><rect x="4" y="4" width="16" height="16" rx="2" ry="2" fill="none"></rect><!-- Simplified code brackets --><line x1="10" y1="10" x2="8" y2="12"></line><line x1="8" y1="12" x2="10" y2="14"></line><line x1="14" y1="10" x2="16" y2="12"></line><line x1="16" y1="12" x2="14" y2="14"></line></svg></button>
```python
       y_recon, mask = model.decode(z, lattice, rots_b, trans_b, ctf_b)
       ```  
    matches cryoDRGN 3.4.4’s API; if yours differs slightly, just reorder or rename those arguments.  

Once you fill in the Chroma‐API line, this script should run “out of the box” against cryoDRGN 3.4.4 and a compatible Chroma library. Let me know if you run into any shape mismatches or missing methods!
