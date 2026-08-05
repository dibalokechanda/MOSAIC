# MOSAIC: Towards a Universal Latent Space for Computational Pathology Foundation Models

![MOSAIC](images/mosaic.jpg)

Towards a Universal Latent Space for Computational Pathology Foundation Models.
Full research plan in [PLAN.md](PLAN.md).

**Implemented so far:** Phase I — representational similarity.

## Layout

```
utils/
  preprocessing.py          coercion, pairing checks, centering, joint subsampling
  cka.py                    linear CKA, RBF-kernel CKA, HSIC, Gram helpers
  cca.py                    SVCCA, PWCCA, raw CCA decomposition
  procrustes.py             orthogonal Procrustes similarity/distance + fitted transform
  cosine.py                 cosine-RSM representational similarity analysis
  distance_correlation.py   Szekely distance correlation (biased + unbiased)
  pairwise.py               metric registry, N x N similarity-matrix driver
  visualization.py          heatmaps, dendrograms, MDS, UMAP
scripts/run_phase1.py       end-to-end driver (CSVs + figures)
tests/test_metrics.py       invariance / property tests
```

## Quick start

Smoke test on synthetic data with a known structure (no embeddings needed):

```bash
python scripts/run_phase1.py --demo --out results/phase1_demo
```

On real embeddings — one file per model (`.npy`, `.npz`, `.pt`), all **row-paired**
so that row *i* is the same patch in every file:

```bash
python scripts/run_phase1.py --emb-dir /path/to/embeddings --out results/phase1 --max-samples 5000
```

Library use:

```python
from utils import compute_all_similarity_matrices, plot_model_space, save_figure

reps = {"UNI": uni, "CONCH": conch, "Virchow": virchow}   # each (n_patches, d_model)
mats = compute_all_similarity_matrices(reps, max_samples=5000)

fig = plot_model_space(mats["linear_cka"], suptitle="Linear CKA")
save_figure(fig, "results/linear_cka.pdf")
```

Every metric takes two row-paired matrices with possibly different feature
dimensions and returns a scalar:

```python
from utils import linear_cka, kernel_cka, svcca, pwcca
from utils import orthogonal_procrustes_similarity, cosine_rsa_similarity, distance_correlation

linear_cka(X, Y)                        # rotation + isotropic scale invariant
kernel_cka(X, Y, threshold=0.4)         # RBF, sensitive to nonlinear structure
svcca(X, Y, var_threshold=0.99)         # PCA-truncated CCA
pwcca(X, Y)                             # projection-weighted CCA (symmetrised)
orthogonal_procrustes_similarity(X, Y)  # ||X^T Y||_* / (||X||_F ||Y||_F)
cosine_rsa_similarity(X, Y)             # Spearman between cosine RSMs
distance_correlation(X, Y)              # 0 iff statistically independent
```

## Metric selection notes

| Metric | Invariant to | Cost | Watch out for |
|---|---|---|---|
| `linear_cka` | rotation, isotropic scale | O(n d²) | blind to nonlinear structure |
| `kernel_cka` | rotation, isotropic scale | O(n²) | bandwidth `threshold` matters; sweep it |
| `svcca` | any invertible linear map | O(n d²) | `var_threshold` is the key knob |
| `pwcca` | any invertible linear map | O(n d²) | asymmetric by construction; symmetrised by default |
| `procrustes` | rotation, isotropic scale | O(n d²) | the only one that returns the transform |
| `cosine_rsa` | rotation, isotropic scale | O(n²) + ranking | Spearman on n(n−1)/2 pairs is the bottleneck |
| `distance_correlation` | rotation, isotropic scale | O(n² d) | use `unbiased=True` (the default) |

Two practical cautions, both visible in the `--demo` output:

- **CCA-family metrics saturate.** SVCCA scores ~0.63 for two *independent*
  random matrices in the demo, because with enough dimensions some directions
  correlate by chance. Read SVCCA/PWCCA as relative rankings, not absolute
  agreement, and always report the linear-CKA number beside them.
- **The `n_samples >> n_features` rule is not optional.** With ~2560-dim
  embeddings (Virchow), a few thousand patches is the bare minimum; the code
  warns when `n_samples <= n_features`.

## Deliverables produced

`scripts/run_phase1.py` writes, per metric:

- `matrices/<metric>.csv` — the N × N similarity matrix
- `figures/<metric>_model_space.<fmt>` — heatmap + dendrogram + MDS (+ UMAP)
- `figures/<metric>_clustered.<fmt>` — clustered heatmap with dendrograms

and across metrics:

- `matrices/pairwise_long.csv` — one row per model pair, one column per metric
- `matrices/metric_agreement.csv` — Spearman agreement between metrics over pairs
- `figures/all_metrics_panel.<fmt>` — all metrics on a shared colour scale

> **On UMAP for the model space:** with ~8 models UMAP has far too few points
> for its manifold assumptions and the layout is dominated by initialisation.
> It is implemented (per the plan) and warns when called this way, but **MDS is
> the ordination to trust here** — pass `--no-umap` for the cleaner figure.
> UMAP becomes the right tool at the patch level in later phases.

## Tests

```bash
python -m pytest tests/ -q
```

65 property tests: self-similarity, rotation/scale/translation invariance,
symmetry, behaviour on independent vs. partially-shared representations,
cross-dimension handling, and driver/output invariants.

## Requirements

numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, umap-learn
(torch only for loading `.pt` embeddings). All present in the base conda env
on `alt`.
