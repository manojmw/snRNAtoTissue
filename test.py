#!/usr/bin/python

#########################################################
# Manoj M Wagle (USydney; MIT CSAIL)
#########################################################


import os, argparse, json, logging, random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

from model import DeconvModel

def load_model(model_dir: str, device: torch.device) -> tuple:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    n_ct, n_genes = len(config["cell_type_names"]), config["n_genes"]
    ref_profile = torch.zeros(n_ct, n_genes)

    model = DeconvModel(
        n_genes=n_genes,
        cell_type_names=config["cell_type_names"],
        ref_profile=ref_profile,
        d_hidden=config.get("d_hidden", 2048),
        d_film=config.get("d_film",      64),
        gene_in_mode=config.get("gene_in_mode", "bulk+ref+h"),
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded model from {model_dir}")
    return model, config

def prepare_bulk_input(adata_bulk, model_gene_names):
    X = adata_bulk.X
    if hasattr(X, "toarray"): X = X.toarray()
    X = X.astype(np.float32)
    bulk_g2i = {g: i for i, g in enumerate(adata_bulk.var_names)}
    n_donors, n_model = X.shape[0], len(model_gene_names)
    aligned = np.zeros((n_donors, n_model), dtype=np.float32)
    matched = 0
    for j, gene in enumerate(model_gene_names):
        if gene in bulk_g2i: aligned[:, j] = X[:, bulk_g2i[gene]]; matched += 1
    pct = matched / n_model * 100
    logger.info(f"Gene alignment: {matched}/{n_model} ({pct:.1f}%)")
    if pct < 50: logger.warning("<50% gene overlap — results may be unreliable")
    row_sums = aligned.sum(1, keepdims=True); row_sums[row_sums == 0] = 1.0
    return torch.from_numpy(np.log1p(aligned / row_sums * 1e4))

@torch.no_grad()
def predict(model, bulk_tensor, device, batch_size=32):
    model.eval(); N = bulk_tensor.shape[0]
    all_expr = {ct: [] for ct in model.cell_type_names}
    for start in range(0, N, batch_size):
        batch = bulk_tensor[start:start + batch_size].to(device)
        expr = model.get_expression(batch)
        for ct in model.cell_type_names: all_expr[ct].append(expr[ct].cpu().numpy())
    return {ct: np.concatenate(v) for ct, v in all_expr.items()}

def save_results(all_expr, donor_ids, gene_names, cell_type_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    adata = ad.AnnData(X=all_expr[cell_type_names[0]],
                       obs=pd.DataFrame({"donor": donor_ids}, index=donor_ids),
                       var=pd.DataFrame(index=gene_names))
    for ct in cell_type_names: adata.layers[ct] = all_expr[ct]
    adata.uns["cell_type_names"]  = cell_type_names
    out_path = os.path.join(output_dir, "REGULATE_output.h5ad")
    adata.write_h5ad(out_path)
    logger.info(f"Saved {out_path}")
    logger.info(f"  Expression: {len(donor_ids)} donors x {len(gene_names)} genes, {len(cell_type_names)} layers")

def main(bulk_h5ad, model_dir, output_dir, batch_size=32, donor_col="donor_id", device_str="auto"):
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else "cpu")
    logger.info("Starting to run...")
    model, config = load_model(model_dir, device)
    adata_bulk = sc.read_h5ad(bulk_h5ad)
    logger.info(f"Bulk data: {adata_bulk.shape[0]} donors x {adata_bulk.shape[1]} genes")
    donor_ids = (adata_bulk.obs[donor_col].tolist()
                 if donor_col and donor_col in adata_bulk.obs.columns
                 else adata_bulk.obs.index.tolist())
    bulk_tensor = prepare_bulk_input(adata_bulk, config["gene_names"])
    logger.info("Running inference...")
    all_expr = predict(model, bulk_tensor, device, batch_size)
    logger.info(f"Saving results to {output_dir}/")
    save_results(all_expr, donor_ids, config["gene_names"], config["cell_type_names"], output_dir)
    logger.info("Completed successfully!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REGULATE inference pipeline")
    parser.add_argument("--bulk_h5ad",  default="path/to/bulk_data.h5ad")
    parser.add_argument("--model_dir",  default="./output") 
    parser.add_argument("--output_dir", default="./predictions") 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--donor_col",  default="donor_id")
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()
    main(bulk_h5ad=args.bulk_h5ad, model_dir=args.model_dir, output_dir=args.output_dir,
         batch_size=args.batch_size, donor_col=args.donor_col, device_str=args.device)
