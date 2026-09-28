#!/usr/bin/python

#########################################################
# Manoj M Wagle (USydney; MIT CSAIL)
#########################################################


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class LinearEncoder(nn.Module):
    def __init__(self, n_genes, d_hidden=2048):
        super().__init__()
        self.d_hidden = d_hidden
        self.encode = nn.Linear(n_genes, d_hidden)

    def forward(self, bulk):
        return F.relu(self.encode(bulk))

class FiLMGeneDecoder(nn.Module):
    def __init__(self, n_ct, d_hidden, n_genes, d_film=64, dropout=0.1):
        super().__init__()
        gene_in = n_ct + 1
        self.d_film = d_film
        self.mlp1  = nn.Linear(gene_in, d_film)
        self.drop1 = nn.Dropout(dropout)
        self.mlp2  = nn.Linear(d_film, d_film)
        self.drop2 = nn.Dropout(dropout)
        self.mlp3  = nn.Linear(d_film, d_film)
        self.drop3 = nn.Dropout(dropout)
        self.mlp4  = nn.Linear(d_film, d_film)
        self.drop4 = nn.Dropout(dropout)
        self.mlp5  = nn.Linear(d_film, d_film)
        self.drop5 = nn.Dropout(dropout)
        self.out   = nn.Linear(d_film, n_ct)
        self.film1 = nn.Linear(d_hidden, 2 * d_film)
        self.film2 = nn.Linear(d_hidden, 2 * d_film)
        self.film3 = nn.Linear(d_hidden, 2 * d_film)
        self.film4 = nn.Linear(d_hidden, 2 * d_film)
        self.film5 = nn.Linear(d_hidden, 2 * d_film)
        self.gene_scale1_g = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale1_b = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale2_g = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale2_b = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale3_g = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale3_b = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale4_g = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale4_b = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale5_g = nn.Parameter(torch.zeros(n_genes, d_film))
        self.gene_scale5_b = nn.Parameter(torch.zeros(n_genes, d_film))
        for film in (self.film1, self.film2, self.film3, self.film4, self.film5):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            film.bias.data[:d_film] = 1.0

    def forward(self, h, ref_profile, bulk):
        B = bulk.shape[0]
        ref = ref_profile.t().unsqueeze(0).expand(B, -1, -1)
        gene_in = torch.cat([ref, bulk.unsqueeze(-1)], -1)

        f1 = self.film1(h)
        g1 = f1[:, :self.d_film].unsqueeze(1) * (1.0 + self.gene_scale1_g.unsqueeze(0))
        b1 = f1[:, self.d_film:].unsqueeze(1) * (1.0 + self.gene_scale1_b.unsqueeze(0))

        f2 = self.film2(h)
        g2 = f2[:, :self.d_film].unsqueeze(1) * (1.0 + self.gene_scale2_g.unsqueeze(0))
        b2 = f2[:, self.d_film:].unsqueeze(1) * (1.0 + self.gene_scale2_b.unsqueeze(0))

        f3 = self.film3(h)
        g3 = f3[:, :self.d_film].unsqueeze(1) * (1.0 + self.gene_scale3_g.unsqueeze(0))
        b3 = f3[:, self.d_film:].unsqueeze(1) * (1.0 + self.gene_scale3_b.unsqueeze(0))

        f4 = self.film4(h)
        g4 = f4[:, :self.d_film].unsqueeze(1) * (1.0 + self.gene_scale4_g.unsqueeze(0))
        b4 = f4[:, self.d_film:].unsqueeze(1) * (1.0 + self.gene_scale4_b.unsqueeze(0))

        f5 = self.film5(h)
        g5 = f5[:, :self.d_film].unsqueeze(1) * (1.0 + self.gene_scale5_g.unsqueeze(0))
        b5 = f5[:, self.d_film:].unsqueeze(1) * (1.0 + self.gene_scale5_b.unsqueeze(0))

        x = self.drop1(F.gelu(self.mlp1(gene_in)))
        x = g1 * x + b1
        res = x
        x = self.drop2(F.gelu(self.mlp2(x)))
        x = g2 * x + b2 + res
        res = x
        x = self.drop3(F.gelu(self.mlp3(x)))
        x = g3 * x + b3 + res
        res = x
        x = self.drop4(F.gelu(self.mlp4(x)))
        x = g4 * x + b4 + res
        res = x
        x = self.drop5(F.gelu(self.mlp5(x)))
        x = g5 * x + b5 + res
        return F.softmax(self.out(x), dim=-1).permute(0, 2, 1)

class DeconvModel(nn.Module):
    def __init__(
        self,
        n_genes: int,
        cell_type_names: List[str],
        ref_profile: torch.Tensor,
        d_hidden: int = 2048,
        dropout: float = 0.1,
        d_film: int = 64,
        gene_in_mode: str = "full",
    ):
        super().__init__()
        self.cell_type_names = cell_type_names
        self.n_cell_types = len(cell_type_names)
        self.gene_in_mode = gene_in_mode
        self.register_buffer("ref_profile", ref_profile.clone().float())
        self.encoder      = LinearEncoder(n_genes, d_hidden)
        self.film_decoder = FiLMGeneDecoder(self.n_cell_types, d_hidden, n_genes, d_film, dropout)

    def _film_inputs(self, bulk):
        parts     = set(self.gene_in_mode.split("+"))
        h         = self.encoder(bulk) if "h" in parts \
                    else torch.zeros(bulk.shape[0], self.encoder.d_hidden, device=bulk.device)
        film_ref  = self.ref_profile if "ref" in parts else torch.zeros_like(self.ref_profile)
        film_bulk = bulk             if "bulk" in parts else torch.zeros_like(bulk)
        return h, film_ref, film_bulk

    def _decompose(self, bulk, masks):
        bulk_linear = torch.expm1(bulk)
        outputs = {}
        for i, ct in enumerate(self.cell_type_names):
            ct_linear = (bulk_linear * masks[:, i, :]).clamp(min=0)
            ct_lib = ct_linear.sum(-1, keepdim=True).clamp(min=1.0)
            outputs[ct] = torch.log1p(ct_linear / ct_lib * 1e4)
        return outputs

    def forward(self, bulk):
        h, film_ref, film_bulk = self._film_inputs(bulk)
        masks      = self.film_decoder(h, film_ref, film_bulk)
        decomposed = self._decompose(bulk, masks)
        outputs    = {ct: {"mu": expr} for ct, expr in decomposed.items()}
        return outputs, h

    @torch.no_grad()
    def get_expression(self, bulk):
        self.eval()
        h, film_ref, film_bulk = self._film_inputs(bulk)
        return self._decompose(bulk, self.film_decoder(h, film_ref, film_bulk))

def _gene_ccc(pred: torch.Tensor, target: torch.Tensor,
              gene_weights: Optional[torch.Tensor] = None,
              eps: float = 1e-8) -> torch.Tensor:
    pred_m = pred.mean(0)
    targ_m = target.mean(0)
    pred_v = pred.var(0)
    targ_v = target.var(0)
    cov = ((pred - pred_m) * (target - targ_m)).mean(0)
    denom = pred_v + targ_v + (pred_m - targ_m).pow(2) + eps
    ccc = 2.0 * cov / denom
    ccc_loss = 1.0 - ccc
    if gene_weights is not None:
        return (ccc_loss * gene_weights).mean()
    return ccc_loss.mean()

def compute_loss(
    outputs: Dict[str, dict],
    targets: Dict[str, torch.Tensor],
    cell_type_names: List[str],
    gene_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(iter(outputs.values()))["mu"].device
    total_loss = torch.tensor(0.0, device=device)
    for ct in cell_type_names:
        total_loss = total_loss + _gene_ccc(
            outputs[ct]["mu"], targets[ct], gene_weights=gene_weights)
    total_loss = total_loss / len(cell_type_names)
    return total_loss, {"loss": total_loss.item()}