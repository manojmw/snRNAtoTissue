#!/usr/bin/python

#########################################################
# Manoj M Wagle (USydney; MIT CSAIL)
#########################################################


import os, argparse, json, logging, random, copy
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset, DataLoader

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

from model import DeconvModel, compute_loss

class PseudobulkDataset(Dataset):
    def __init__(self, adata, cell_type_col, cell_type_names, gene_names,
                 n_samples_per_donor=100, min_total_cells=100, max_total_cells=600, donor_col=None):
        self.cell_type_names = cell_type_names
        common_genes = [g for g in gene_names if g in adata.var_names]
        adata = adata[:, common_genes].copy()
        self.n_genes = len(common_genes)

        X = adata.X
        if hasattr(X, "toarray"): X = X.toarray()
        X = X.astype(np.float32)

        donor_ct_indices, donors = {}, []
        if donor_col is not None:
            all_donors = list(adata.obs[donor_col].unique())
            skipped = 0
            for donor in all_donors:
                d_mask = adata.obs[donor_col] == donor
                cell_idx, has_all = {}, True
                for ct in cell_type_names:
                    idx = np.where((d_mask & (adata.obs[cell_type_col] == ct)).values)[0]
                    cell_idx[ct] = idx
                    if len(idx) == 0: has_all = False
                if not has_all: skipped += 1; continue
                donors.append(donor); donor_ct_indices[donor] = cell_idx
            if skipped > 0:
                logger.info(f"Skipped {skipped}/{len(all_donors)} donors missing >=1 cell type.")
            assert len(donors) > 0
            n_samples = len(donors) * n_samples_per_donor
        else:
            ct_indices = {ct: np.where((adata.obs[cell_type_col] == ct).values)[0] for ct in cell_type_names}
            donors = [None]; donor_ct_indices = {None: ct_indices}
            n_samples_per_donor *= 10; n_samples = n_samples_per_donor

        donor_dirichlet_alpha = {}
        for d in donors:
            if d is None: continue
            counts = np.array([len(donor_ct_indices[d][ct]) for ct in cell_type_names], dtype=np.float32)
            props = counts / counts.sum()
            donor_dirichlet_alpha[d] = props * (1.0 / props.min())
        logger.info(f"Per-donor Dirichlet alpha computed (scale=1/min_prop per donor)")

        logger.info(f"Pre-generating {n_samples} pseudobulk samples...")
        n_ct, G = len(cell_type_names), self.n_genes
        self.bulk = torch.zeros(n_samples, G)
        self.targets = {ct: torch.zeros(n_samples, G) for ct in cell_type_names}

        sample_idx = 0
        for donor in tqdm.tqdm(donors, desc="Generating pseudobulk"):
            cell_idx = donor_ct_indices[donor]
            n_donor_samples = n_samples_per_donor if donor is not None else n_samples
            alpha = donor_dirichlet_alpha[donor] if donor is not None else np.ones(n_ct)
            for _ in range(n_donor_samples):
                n_total = np.random.randint(min_total_cells, max_total_cells + 1)
                fracs = np.random.dirichlet(alpha)
                ct_counts = np.maximum((fracs * n_total).astype(int), 1)
                ct_sums = {}
                for i, ct in enumerate(cell_type_names):
                    sampled = np.random.choice(cell_idx[ct], size=ct_counts[i], replace=True)
                    ct_sums[ct] = X[sampled].sum(axis=0).ravel().astype(np.float32)
                bulk_raw = sum(ct_sums.values()); bulk_lib = float(bulk_raw.sum())
                if bulk_lib > 0:
                    self.bulk[sample_idx] = torch.from_numpy(
                        np.log1p(bulk_raw / bulk_lib * 1e4).astype(np.float32))
                    for ct in cell_type_names:
                        ct_lib = float(ct_sums[ct].sum())
                        scale = 1e4 / ct_lib if ct_lib > 0 else 0.0
                        self.targets[ct][sample_idx] = torch.from_numpy(
                            np.log1p(ct_sums[ct] * scale).astype(np.float32))
                sample_idx += 1

        self.n_samples = n_samples
        logger.info(f"Pre-generation complete: {n_samples} samples x {G} genes")
        del X

    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        return self.bulk[idx], {ct: self.targets[ct][idx] for ct in self.cell_type_names}

class PseudobulkReconDataset(Dataset):
    def __init__(self, bulk, proportions):
        self.bulk = bulk
        self.proportions = proportions
    def __len__(self): return self.bulk.shape[0]
    def __getitem__(self, idx):
        return self.bulk[idx], self.proportions[idx]

def load_bulk_tensor(bulk_h5ad, gene_names):
    adata_bulk = sc.read_h5ad(bulk_h5ad)
    X = adata_bulk.X
    if hasattr(X, "toarray"): X = X.toarray()
    X = X.astype(np.float32)
    g2i = {g: i for i, g in enumerate(adata_bulk.var_names)}
    aligned = np.zeros((X.shape[0], len(gene_names)), dtype=np.float32)
    matched = sum(1 for g in gene_names if g in g2i)
    for j, g in enumerate(gene_names):
        if g in g2i: aligned[:, j] = X[:, g2i[g]]
    row_sums = aligned.sum(1, keepdims=True); row_sums[row_sums == 0] = 1.0
    logger.info(f"Bulk tensor: {X.shape[0]} donors, {matched}/{len(gene_names)} genes matched")
    return torch.from_numpy(np.log1p(aligned / row_sums * 1e4))

def train_epoch(model, dataloader, optimizer, device, max_grad_norm=5.0, epoch_desc="",
                gene_weights=None, bulk_expr=None, domain_weight=1.0):
    model.train(); totals = {}; n = 0
    N_bulk = bulk_expr.shape[0] if bulk_expr is not None else 0
    for bulk, targets in tqdm.tqdm(dataloader, desc=epoch_desc, leave=False):
        bulk = bulk.to(device)
        targets = {ct: t.to(device) for ct, t in targets.items()}
        outputs, h_sn = model(bulk)
        loss, metrics = compute_loss(outputs, targets, model.cell_type_names,
                                     gene_weights=gene_weights)
        if bulk_expr is not None and domain_weight > 0:
            B = bulk.shape[0]
            idx = torch.randperm(N_bulk)[:B]
            h_bulk = model.encoder(bulk_expr[idx].to(device))
            src_c = h_sn  - h_sn.mean(0, keepdim=True)
            tgt_c = h_bulk - h_bulk.mean(0, keepdim=True)
            cov_src = src_c.T @ src_c / max(B - 1, 1)
            cov_tgt = tgt_c.T @ tgt_c / max(B - 1, 1)
            domain_loss = ((cov_src - cov_tgt) ** 2).mean()
            loss = loss + domain_weight * domain_loss
            metrics["domain_loss"] = domain_loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        for k, v in metrics.items(): totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in totals.items()}

@torch.no_grad()
def validate_loss(model, dataloader, device, gene_weights=None):
    model.eval(); totals = {}; n = 0
    for bulk, targets in dataloader:
        bulk = bulk.to(device)
        targets = {ct: t.to(device) for ct, t in targets.items()}
        outputs, _h = model(bulk)
        _, metrics = compute_loss(outputs, targets, model.cell_type_names,
                                  gene_weights=gene_weights)
        for k, v in metrics.items(): totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in totals.items()}

def validate_hvg_spearman(model, dataloader, device, hvg_indices, cell_type_names):
    model.eval()
    all_preds = {ct: [] for ct in cell_type_names}
    all_targs = {ct: [] for ct in cell_type_names}
    with torch.no_grad():
        for bulk, targets in dataloader:
            bulk = bulk.to(device)
            outputs, _ = model(bulk)
            for ct in cell_type_names:
                all_preds[ct].append(outputs[ct]["mu"].cpu().numpy())
                all_targs[ct].append(targets[ct].numpy())
    spearman_vals = []
    for ct in cell_type_names:
        preds = np.concatenate(all_preds[ct], axis=0)[:, hvg_indices]
        targs = np.concatenate(all_targs[ct], axis=0)[:, hvg_indices]
        m = _compute_gene_wise_metrics(preds, targs)
        spearman_vals.append(m["gene_spearman"])
    mean_hvg_s = float(np.mean(spearman_vals))
    return {"hvg_spearman": mean_hvg_s, "loss": 1.0 - mean_hvg_s}

def _compute_gene_wise_metrics(preds, targs):
    from scipy.stats import rankdata
    p_dm = preds - preds.mean(0, keepdims=True); t_dm = targs - targs.mean(0, keepdims=True)
    p_norm = np.sqrt((p_dm**2).sum(0)); t_norm = np.sqrt((t_dm**2).sum(0))
    truth_vary = t_norm > 1e-12; pred_vary = p_norm > 1e-12; both_vary = truth_vary & pred_vary
    gene_rho = np.zeros(truth_vary.sum())
    if both_vary.any():
        bv_in_tv = pred_vary[truth_vary]
        pr = np.apply_along_axis(rankdata, 0, preds[:, both_vary])
        tr = np.apply_along_axis(rankdata, 0, targs[:, both_vary])
        pr_dm = pr - pr.mean(0, keepdims=True); tr_dm = tr - tr.mean(0, keepdims=True)
        rdenom = np.sqrt((pr_dm**2).sum(0)) * np.sqrt((tr_dm**2).sum(0))
        rvalid = rdenom > 1e-12; rho_vals = np.zeros(both_vary.sum())
        rho_vals[rvalid] = (pr_dm[:, rvalid] * tr_dm[:, rvalid]).sum(0) / rdenom[rvalid]
        gene_rho[bv_in_tv] = rho_vals
    return {"gene_spearman": float(gene_rho.mean()) if len(gene_rho) > 0 else 0,
            "n_genes_vary":  int(truth_vary.sum())}

def get_valid_donors(adata, donor_col, cell_type_col, cell_type_names):
    all_donors = adata.obs[donor_col].unique()
    valid = [d for d in all_donors
             if all(((adata.obs[donor_col] == d) & (adata.obs[cell_type_col] == ct)).any()
                    for ct in cell_type_names)]
    logger.info(f"Valid donors (all cell types): {len(valid)}/{len(all_donors)}")
    return valid

def prepare_data(adata, cell_type_col, donor_col,
                 n_samples_per_donor=100, seed=42, no_val=False):
    cell_type_names = sorted(adata.obs[cell_type_col].unique().tolist())
    gene_names = adata.var_names.tolist()
    logger.info(f"{len(cell_type_names)} cell types | {len(gene_names)} genes")

    valid_donors = get_valid_donors(adata, donor_col, cell_type_col, cell_type_names)
    rng = np.random.RandomState(seed); rng.shuffle(valid_donors)

    if no_val:
        train_donors = valid_donors
        val_donors = []
        logger.info(f"No-val mode: all {len(train_donors)} donors used for training (no early stopping)")
    else:
        n_val = max(4, int(len(valid_donors) * 0.1))
        val_donors = valid_donors[:n_val]
        train_donors = valid_donors[n_val:]
        logger.info(f"Train: {len(train_donors)} donors | Val: {len(val_donors)} donors (early stopping)")

    train_ds = PseudobulkDataset(
        adata[adata.obs[donor_col].isin(train_donors)].copy(),
        cell_type_col, cell_type_names, gene_names,
        n_samples_per_donor=n_samples_per_donor, donor_col=donor_col)
    val_ds = None
    if not no_val:
        val_ds = PseudobulkDataset(
            adata[adata.obs[donor_col].isin(val_donors)].copy(),
            cell_type_col, cell_type_names, gene_names,
            n_samples_per_donor=n_samples_per_donor, donor_col=donor_col)

    all_train_targets = torch.stack([train_ds.targets[ct] for ct in cell_type_names], dim=0)
    gene_std = all_train_targets.std(dim=1).mean(dim=0)
    n_hvg = min(5000, len(gene_names))
    _, hvg_indices = torch.topk(gene_std, n_hvg)
    hvg_indices = hvg_indices.numpy()
    logger.info(f"Precomputed {n_hvg} HVG indices from training data")

    gene_weights = gene_std / gene_std.median()
    gene_weights = gene_weights.clamp(min=0.1)
    gene_weights = gene_weights / gene_weights.mean()
    logger.info(f"Gene weights (SD-based): median={gene_weights.median():.4f}, "
                f"mean={gene_weights.mean():.4f}, max={gene_weights.max():.4f}, "
                f"min={gene_weights.min():.4f}, >2x median: {(gene_weights > 2).sum()} genes")

    logger.info("Computing reference mean profile from training targets...")
    ref_profile = torch.stack([
        train_ds.targets[ct].float().mean(dim=0)
        for ct in cell_type_names
    ])
    logger.info(f"Reference profile: {ref_profile.shape}, range [{ref_profile.min():.2f}, {ref_profile.max():.2f}]")

    X_raw = adata.X
    if hasattr(X_raw, "toarray"): X_raw = X_raw.toarray()
    X_raw = X_raw.astype(np.float32)
    per_donor_ct_means = {}
    per_donor_proportions = {}
    per_donor_pseudobulk = {}
    for donor in train_donors + val_donors:
        per_donor_ct_means[donor] = {}
        counts = np.array([
            ((adata.obs[donor_col] == donor) & (adata.obs[cell_type_col] == ct)).sum()
            for ct in cell_type_names], dtype=np.float32)
        per_donor_proportions[donor] = torch.from_numpy(counts / counts.sum())
        donor_mask = (adata.obs[donor_col] == donor).values
        donor_sum = X_raw[donor_mask].sum(axis=0).ravel()
        donor_lib = float(donor_sum.sum())
        donor_scale = 1e4 / donor_lib if donor_lib > 0 else 0.0
        per_donor_pseudobulk[donor] = torch.from_numpy(np.log1p(donor_sum * donor_scale).astype(np.float32))
        for ct in cell_type_names:
            mask = ((adata.obs[donor_col] == donor) & (adata.obs[cell_type_col] == ct)).values
            cells = X_raw[mask]
            ct_sum = cells.sum(axis=0).ravel()
            ct_lib = float(ct_sum.sum())
            scale = 1e4 / ct_lib if ct_lib > 0 else 0.0
            per_donor_ct_means[donor][ct] = torch.from_numpy(np.log1p(ct_sum * scale).astype(np.float32))
    logger.info(f"Per-donor CT means + proportions + pseudobulk computed for {len(train_donors)} train + {len(val_donors)} val donors")

    return {"cell_type_names": cell_type_names, "gene_names": gene_names,
            "train_donors": train_donors, "val_donors": val_donors,
            "train_ds": train_ds, "val_ds": val_ds,
            "gene_weights": gene_weights, "hvg_indices": hvg_indices,
            "ref_profile": ref_profile,
            "per_donor_ct_means": per_donor_ct_means,
            "per_donor_proportions": per_donor_proportions,
            "per_donor_pseudobulk": per_donor_pseudobulk}

def run_training(prebuilt, output_dir,
                 n_epochs=80, batch_size=16, lr=1e-3,
                 d_hidden=256, d_film=64, dropout=0.1,
                 device_str="auto", seed=42, no_val=False,
                 bulk_expr=None, domain_weight=1.0,
                 pruning_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else "cpu")
    logger.info("Starting to run..."); logger.info(f"Device: {device}")

    cell_type_names = prebuilt["cell_type_names"]
    gene_names      = prebuilt["gene_names"]
    train_donors    = prebuilt["train_donors"]
    val_donors      = prebuilt["val_donors"]
    train_ds        = prebuilt["train_ds"]
    val_ds          = prebuilt["val_ds"]
    gene_weights  = prebuilt["gene_weights"].to(device)
    hvg_indices   = prebuilt["hvg_indices"]
    ref_profile   = prebuilt["ref_profile"]

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = None
    if not no_val and val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    gene_in_mode = prebuilt.get("gene_in_mode", "bulk+ref+h")
    model = DeconvModel(n_genes=len(gene_names), cell_type_names=cell_type_names,
                        ref_profile=ref_profile,
                        d_hidden=d_hidden,
                        d_film=d_film, dropout=dropout,
                        gene_in_mode=gene_in_mode).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_val_loss, best_state, patience = float("inf"), None, 0

    for epoch in range(n_epochs):
        m = train_epoch(model, train_dl, optimizer, device,
                        epoch_desc=f"[Training] E{epoch+1}/{n_epochs}",
                        gene_weights=gene_weights,
                        bulk_expr=bulk_expr, domain_weight=domain_weight)
        scheduler.step()
        domain_str = f" | CORAL {m['domain_loss']:.4f}" if "domain_loss" in m else ""
        if no_val:
            logger.info(f"Ep {epoch+1:3d} | Train {m['loss']:.6e}{domain_str}")
            if pruning_callback is not None and pruning_callback(epoch, m["loss"]):
                logger.info(f"Trial pruned at epoch {epoch+1}"); break
            if m["loss"] < best_val_loss:
                best_val_loss = m["loss"]; patience = 0; best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= 10: logger.info(f"Early stop at epoch {epoch+1}"); break
        else:
            ccc_val = validate_loss(model, val_dl, device, gene_weights=gene_weights)
            val = validate_hvg_spearman(model, val_dl, device, hvg_indices, cell_type_names)
            logger.info(f"Ep {epoch+1:3d} | Train {m['loss']:.6e}{domain_str} | CCC Val {ccc_val['loss']:.6e} | HVG S {val['hvg_spearman']:.6e}")
            if pruning_callback is not None and pruning_callback(epoch, val["loss"]):
                logger.info(f"Trial pruned at epoch {epoch+1}"); break
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]; patience = 0; best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= 10: logger.info(f"Early stop at epoch {epoch+1}"); break

    torch.save(best_state, os.path.join(output_dir, "best_model.pt"))
    model.load_state_dict(best_state)

    config = {"cell_type_names": cell_type_names, "gene_names": gene_names,
              "n_genes": len(gene_names), "n_cell_types": len(cell_type_names),
              "train_donors": [str(d) for d in train_donors],
              "val_donors": [str(d) for d in val_donors],
              "seed": seed, "d_hidden": d_hidden, "d_film": d_film,
              "gene_in_mode": gene_in_mode}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Done. Saved to {output_dir}/")
    logger.info(f"Best model saved (val loss = {best_val_loss:.6e})")
    logger.info("Completed successfully!!")
    return best_val_loss

def build_pseudobulk_recon_dataset(donors, per_donor_pseudobulk, per_donor_proportions, cell_type_names):
    if not donors:
        logger.warning("No donors given -- skipping pseudobulk reconstruction dataset")
        return None
    bulk_tensor = torch.stack([per_donor_pseudobulk[d] for d in donors])
    prop_tensor = torch.stack([per_donor_proportions[d] for d in donors])
    logger.info(f"Pseudobulk recon dataset: {len(donors)} donors (Phase 2 target)")
    return PseudobulkReconDataset(bulk_tensor, prop_tensor)

def _recon_linear_mix(outputs, props, cell_type_names):
    bulk_pred_linear = sum(props[:, i:i+1] * torch.expm1(outputs[ct]["mu"])
                            for i, ct in enumerate(cell_type_names))
    bulk_pred_linear = (bulk_pred_linear
                         / bulk_pred_linear.sum(dim=1, keepdim=True).clamp_min(1e-8) * 1e4)
    return torch.log1p(bulk_pred_linear)

def finetune_reconstruction(output_dir, train_ds, val_ds, gene_names, cell_type_names,
                             ref_profile, gene_weights,
                             d_hidden=1024, d_film=128, dropout=0.3,
                             gene_in_mode="bulk+ref+h", n_epochs=50, lr=1e-5,
                             batch_size=32, device_str="auto", phase_label="Phase"):
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else "cpu")
    model = DeconvModel(n_genes=len(gene_names), cell_type_names=cell_type_names,
                        ref_profile=ref_profile,
                        d_hidden=d_hidden, d_film=d_film, dropout=dropout,
                        gene_in_mode=gene_in_mode).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device))
    logger.info(f"{phase_label} (recon): train={len(train_ds)} | val={len(val_ds)} | lr={lr} | max_epochs={n_epochs}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss, best_state, patience = float("inf"), None, 0

    for epoch in range(n_epochs):
        model.train(); train_loss = 0.0; n = 0
        for bulk, props in tqdm.tqdm(train_dl, desc=f"[Recon FT] E{epoch+1}/{n_epochs}", leave=False):
            bulk, props = bulk.to(device), props.to(device)
            outputs, _ = model(bulk)
            bulk_pred = _recon_linear_mix(outputs, props, cell_type_names)
            loss = F.mse_loss(bulk_pred, bulk)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item(); n += 1
        train_loss /= n

        model.eval(); val_loss = 0.0; n = 0
        with torch.no_grad():
            for bulk, props in val_dl:
                bulk, props = bulk.to(device), props.to(device)
                outputs, _ = model(bulk)
                bulk_pred = _recon_linear_mix(outputs, props, cell_type_names)
                val_loss += F.mse_loss(bulk_pred, bulk).item(); n += 1
        val_loss /= n
        logger.info(f"Recon FT Ep {epoch+1:3d} | Train {train_loss:.6e} | Val {val_loss:.6e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience = 0; best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 10:
                logger.info(f"{phase_label} early stop at epoch {epoch+1}"); break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(output_dir, "best_model.pt"))
    logger.info(f"{phase_label} recon complete. best_model.pt overwritten (val loss={best_val_loss:.6e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad",                default="path/to/snrna_reference.h5ad")
    parser.add_argument("--finetune_epochs",     type=int,   default=400,
                        help="Phase 2: fine-tuning epochs on pseudobulk reconstruction (0 = skip)")
    parser.add_argument("--finetune_lr",         type=float, default=1e-5,
                        help="Phase 2: fine-tuning learning rate")
    parser.add_argument("--bulk_h5ad",            default="path/to/bulk_data.h5ad",
                    help="Target bulk h5ad for CORAL encoder h domain alignment")
    parser.add_argument("--cell_type_col",       default="cell_type")
    parser.add_argument("--donor_col",           default="donor_id")
    parser.add_argument("--output_dir",          default="./output") 
    parser.add_argument("--n_epochs",            type=int,   default=400)
    parser.add_argument("--batch_size",          type=int,   default=32)
    parser.add_argument("--lr",                  type=float, default=1.53e-4)
    parser.add_argument("--n_samples_per_donor", type=int,   default=25)
    parser.add_argument("--no_val",              action="store_true", default=False,
                        help="Disable val split; early stop on train loss instead")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--d_hidden",            type=int,   default=1024,
                        help="Encoder width (number of latent features)")
    parser.add_argument("--d_film",              type=int,   default=128,
                        help="FiLM decoder hidden dim (per-gene MLP width)")
    parser.add_argument("--dropout",             type=float, default=0.3,
                        help="Dropout rate for FiLM decoder")
    parser.add_argument("--domain_weight",        type=float, default=1,
                        help="Weight for CORAL domain alignment loss")
    parser.add_argument("--gene_in",             default="bulk+ref+h",
                        help="Active inputs to FiLM: any '+'-combo of bulk, ref, h (default: bulk+ref+h)")
    parser.add_argument("--device",              default="auto")
    args = parser.parse_args()

    logger.info(f"Loading: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    logger.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    prebuilt = prepare_data(adata, args.cell_type_col, args.donor_col,
                            args.n_samples_per_donor, args.seed, args.no_val)
    prebuilt["gene_in_mode"] = args.gene_in

    bulk_expr = None
    if args.bulk_h5ad is not None:
        bulk_expr = load_bulk_tensor(args.bulk_h5ad, prebuilt["gene_names"])

    run_training(prebuilt, args.output_dir,
                 n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
                 d_hidden=args.d_hidden, d_film=args.d_film, dropout=args.dropout,
                 device_str=args.device, seed=args.seed, no_val=args.no_val,
                 bulk_expr=bulk_expr, domain_weight=args.domain_weight)

    if args.finetune_epochs > 0:
        train_recon_ds = build_pseudobulk_recon_dataset(
            prebuilt["train_donors"], prebuilt["per_donor_pseudobulk"],
            prebuilt["per_donor_proportions"], prebuilt["cell_type_names"])
        val_recon_ds = build_pseudobulk_recon_dataset(
            prebuilt["val_donors"], prebuilt["per_donor_pseudobulk"],
            prebuilt["per_donor_proportions"], prebuilt["cell_type_names"])
        if train_recon_ds is not None and val_recon_ds is not None:
            logger.info("Phase 2: pseudobulk reconstruction fine-tuning")
            finetune_reconstruction(
                args.output_dir, train_recon_ds, val_recon_ds,
                prebuilt["gene_names"], prebuilt["cell_type_names"],
                prebuilt["ref_profile"], prebuilt["gene_weights"],
                d_hidden=args.d_hidden, d_film=args.d_film, dropout=args.dropout,
                gene_in_mode=args.gene_in, n_epochs=args.finetune_epochs, lr=args.finetune_lr,
                batch_size=args.batch_size, device_str=args.device, phase_label="Phase 2")