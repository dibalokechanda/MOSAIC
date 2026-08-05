# MOSAIC: Towards a Universal Latent Space for Computational Pathology Foundation Models

<div align="center">
  <img src="images/mosaic.jpg" alt="MOSAIC" width="400"/>
  <br/>
  <br/>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  </a>
  <a href="https://github.com/dibalokechanda/MOSAIC/releases">
    <img src="https://img.shields.io/badge/version-1.0.0-success.svg" alt="Version">
  </a>
</div>

Towards a Universal Latent Space for Computational Pathology Foundation Models.
Full research plan in [PLAN.md](PLAN.md).

**Implemented so far:** Phase I (representational similarity) and Phase V
(shared latent space).

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
  alignment/                Phase V aligners (shared encode/decode interface)
    base.py                 preprocessing, decoders, split, save/load
    gcca.py                 generalized CCA (MAX-VAR)
    mcca.py                 multi-view CCA (SUMCOR)
    gpa.py                  generalized Procrustes analysis
    joint_pca.py            joint PCA baseline
    autoencoder.py          shared latent autoencoder (torch)
    optimal_transport.py    Wasserstein Procrustes + Sinkhorn
  alignment_metrics.py      reconstruction, alignment quality, retrieval, transfer
scripts/run_phase1.py       Phase I driver (CSVs + figures)
scripts/run_phase5.py       Phase V driver (comparison table + reports + figures)
tests/                      property tests for both phases
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

---

# Phase V — shared latent space

Six aligners behind one interface. Each learns a per-model **encoder** into a
shared space and a **decoder** back out of it.

```bash
python scripts/run_phase5.py --demo --out results/phase5_demo --latent-dim 16
```

```python
from utils.alignment import build_aligner, split_views
from utils.alignment_metrics import compare_aligners

train, test, _, _ = split_views(views, test_size=0.2)      # one shared row split
aligner = build_aligner("gcca", latent_dim=64).fit(train)

Z      = aligner.transform(test)                  # {model: (n, 64)} in the shared space
back   = aligner.inverse_transform(Z["UNI"], "UNI")        # out of the shared space
uni_ish = aligner.translate(test["CONCH"], "CONCH", "UNI") # Phase VI: CONCH -> UNI
consensus = aligner.consensus(test)                        # one embedding per patch

summary, reports = compare_aligners({"gcca": aligner, ...}, test)
```

| Aligner | Answers | Notes |
|---|---|---|
| `joint_pca` | Does anything beat concatenation + PCA? | The baseline everything must beat |
| `gcca` | Is there one latent matrix all models are linear views of? | `eigenvalues_` ∈ [1, M] say how many dimensions are shared |
| `mcca` | Same, via pairwise correlations | Reduces exactly to CCA for two models (tested) |
| `procrustes` | Is a **rigid rotation** enough? | Strictest test; exact inverse, best geometry preservation |
| `autoencoder` | Is the shared structure nonlinear? | Beating GCCA here means linearity was the limit |
| `optimal_transport` | Rigid alignment via OT | Supervised works; **unsupervised is exploratory** |

## Evaluation

`evaluate_aligner` deliberately reports two opposing failure modes side by side,
because optimising either alone gives a degenerate result:

- **Under-alignment** — `alignment_error`, `paired_cosine`, `recall@k`, `mrr`
- **Collapse** — `reconstruction_r2`, `effective_rank`, `neighborhood_preservation`

Always evaluate on held-out patches via `split_views`; a map with more
parameters than samples reconstructs its training set perfectly.

## Three findings from building this

- **Free scaling breaks generalized Procrustes.** Without a constraint the
  consensus shrinks geometrically and the iteration never converges. Ten Berge's
  normalisation (Σ scaleᵢ² = M) is applied; it converges in ~7 iterations.
- **Whitening trades geometry for isotropy.** GCCA and MCCA whiten by
  construction, and their `neighborhood_preservation` drops to ~0.67 where rigid
  Procrustes holds ~1.0 — same retrieval, very different local structure. Joint
  PCA exposes this as `whiten` (default off). If a downstream MIL model cares
  about local neighbourhoods, this matters more than the retrieval numbers.
- **Unsupervised OT alignment is not reliable.** Supervised mode recovers
  planted rotations perfectly (matching accuracy 1.000). Unsupervised reached at
  best ~15–25× chance and often chance, for two separate reasons: whitening
  removes the second-order structure that would identify the rotation, and the
  transport cost barely discriminates between restarts. `supervised=True` is the
  default; treat any unsupervised number as a lower bound.

---

## Tests

```bash
python -m pytest tests/ -q
```

Property tests for both phases — Phase I: self-similarity, rotation/scale/
translation invariance, symmetry, independent vs. partially-shared
representations. Phase V: encode/decode round-trips, held-out generalisation,
recovery of a planted shared latent, refusal to align an unrelated model,
GCCA eigenvalue bounds, MCCA-equals-CCA at M=2, Procrustes orthogonality and
convergence, Sinkhorn double stochasticity, and collapse detection.

Slow tests (autoencoder training, optimal transport) are marked `slow`:

```bash
python -m pytest tests/ -q -m "not slow"
```

## Requirements

numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, umap-learn
(torch only for loading `.pt` embeddings). All present in the base conda env
on `alt`.
