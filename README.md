# Inferring cell type-specific tissue expression from single-nucleus data with deep learning

<img width="1100" height="370" alt="Screenshot 2026-09-28 at 3 00 08 PM" src="https://github.com/user-attachments/assets/edff140a-8bd8-4959-b2e7-a8437d449971" />


## Usage

### Training

**Input:** 
- A single-nucleus RNA-seq `.h5ad` reference, with `cell_type` and `donor_id` columns in `.obs`
- The target bulk RNA-seq `.h5ad` (`--bulk_h5ad`) for which cell type-specific expression needs to be inferred

```bash
python main.py \
  --h5ad path/to/snrna_reference.h5ad \
  --bulk_h5ad path/to/bulk_data.h5ad \
  --cell_type_col cell_type \
  --donor_col donor_id \
  --output_dir ./output
```

**Output:** 
- `output/best_model.pt` (trained weights)
- `output/config.json` (cell types, genes, architecture config)

### Inference

**Input:** 
- The target bulk RNA-seq `.h5ad` (`--bulk_h5ad`)
- The trained model directory from the training step (`--model_dir`, containing `best_model.pt` and `config.json`)

```bash
python test.py \
  --bulk_h5ad path/to/bulk_data.h5ad \
  --model_dir ./output \
  --output_dir ./predictions
```

**Output:** 
- `predictions/REGULATE_output.h5ad`, an AnnData object with `.layers[cell_type]` matrix (donors x genes) per cell type

---

## Requirements

- Python >= 3.10
- PyTorch
- numpy
- pandas
- scanpy
- anndata
- scipy
- tqdm
