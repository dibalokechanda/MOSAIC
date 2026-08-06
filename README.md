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

---

## Contents

1. [Hypothesis and research questions](#1-hypothesis-and-research-questions)
2. [Encoders](#2-encoders)
3. [Cohorts](#3-cohorts)
4. [Feature extraction (trident)](#4-feature-extraction-trident)
5. [The pairing constraint and encoder sets](#5-the-pairing-constraint-and-encoder-sets)
6. [Labels: TCGA and CPTAC](#6-labels-tcga-and-cptac)
7. [Downstream tasks](#7-downstream-tasks)
8. [Metrics](#8-metrics)
9. [Running the study](#9-running-the-study)
10. [Figures](#10-figures)
11. [Results so far](#11-results-so-far)
12. [Layout, tests, requirements](#12-layout-tests-requirements)

---

## 1. Hypothesis and research questions

> **Central hypothesis.** Independently trained pathology foundation models
> learn *different coordinate systems over a shared underlying morphological
> manifold.*

If that holds, representations should be alignable, semantic neighbourhoods
should survive alignment, features should transfer between models, and a
downstream classifier should be able to live inside one shared space.

The full statement, the five research questions, and the phase-by-phase plan are
in **[PLAN.md](PLAN.md)**. In brief:

| RQ | Question | Where it is answered |
|---|---|---|
| RQ1 | How similar are the representation spaces? | Phase I — `representation_similarity.py` |
| RQ2 | Does the pretraining objective drive latent geometry? | Phase I + the `family` field in the registry |
| RQ3 | Can several models be aligned into one shared space? | Phase V — `shared_latent_space.py` |
| RQ4 | Does the aligned space improve downstream learning? | Phase VIII — `downstream_mil.py` |
| RQ5 | Does a model trained on one encoder generalise to another? | Phase VI — `cross_model_transfer.py` |

Phase IV (layer-wise) and the magnification ablation are supporting analyses.
Phases II, III and the biological analysis are **not implemented** and have been
removed from the plan.

---

## 2. Encoders

Twelve patch encoders. Machine-verified fields (dimension, HF id, precision)
come from disk and from trident's source; descriptive fields are hand-entered
and flagged for verification in `configs/encoders.yaml`.

| Encoder | Registry key | Dim | Architecture | Objective | Family |
|---|---|---|---|---|---|
| UNI2-h | `uni_v2` | 1536 | ViT-H/14 + registers | DINOv2 | vision SSL |
| Prov-GigaPath | `gigapath` | 1536 | ViT-g/14 | DINOv2 | vision SSL |
| Virchow | `virchow` | 2560 | ViT-H/14 | DINOv2 | vision SSL |
| Virchow2 | `virchow2` | 2560 | ViT-H/14 | DINOv2 + pathology augs | vision SSL |
| H-optimus-0 | `hoptimus0` | 1536 | ViT-g/14 | DINOv2 | vision SSL |
| GPFM | `gpfm` | 1024 | ViT-L/14 | multi-teacher distillation | vision SSL |
| CTransPath | `ctranspath` | 768 | CNN + Swin-T | SRCL contrastive | vision SSL |
| CONCH | `conch_v1` | 512 | ViT-B/16 + text | CoCa (contrastive + captioning) | vision-language |
| CONCH v1.5 | `conch_v15` | 768 | ViT-L/16 @448 | vision-language (TITAN encoder) | vision-language |
| KEEP | `keep` | 768 | ViT-L/16 | knowledge-enhanced VL | vision-language |
| MUSK | `musk` | 1024 | BEiT-3 @384 | masked modelling + VL | vision-language |
| **ResNet50** | `resnet50` | 1024 | ResNet50 (truncated at layer3) | ImageNet supervised | **control** |

**ResNet50 is the control.** It has never seen a pathology slide. Every claim
about a shared *morphological* manifold has to show the pathology models agree
with each other substantially more than with it — otherwise the "shared
structure" could be generic image statistics.

`family` is the grouping that tests RQ2: if the objective dominates, vision-SSL
models should cluster apart from vision-language ones.

Virchow and Virchow2 emit **2560 = class token (1280) ∥ mean patch tokens
(1280)** — two different kinds of representation concatenated, worth remembering
when reading their scores.

---

## 3. Cohorts

| Cohort | Store key | Slides | Composition |
|---|---|---|---|
| TCGA | `master_benchmark` | 2169 | BRCA 1126, LUAD 531, LUSC 512 |
| CPTAC | `cptac_benchmark` | 2296 | LUAD 1139, BRCA 654, COAD 369, LSCC 134 |

Only these have features. The `Datasets/TCGA-{RCC,BLCA,CRC,ESCA,PRAD,TGCT,THCA}`
folders hold manifests but **no extracted features**, so they cannot enter any
analysis.

CPTAC-LSCC is only ~12% extracted (134 of 1081 slides, 28 patients), which is why
its mutation tasks are excluded and why `cptac_nsclc` is 1139:134.

---

## 4. Feature extraction (trident)

Features were produced with [trident](https://github.com/mahmoodlab/trident).
The pipeline, and why its output shape governs everything downstream:

```
WSI (.svs)
   │
   ├─ 1. tissue segmentation         → contours, thumbnails, wsi_states/*.json
   │
   ├─ 2. patch coordinate extraction → <mag>x_<patch>px_0px_overlap/patches/<slide>_patches.h5
   │        one coordinate grid per (magnification, patch_size)
   │        stores ONLY coords (n, 2) in level-0 pixels, plus attrs
   │
   └─ 3. per-encoder feature extraction
            for each encoder: read patches at those coords → forward pass
            → features_<encoder>/<slide>.h5   features (n, d) + coords (n, 2)
```

Store layout:

```
<feature_root>/<cohort>/<mag>x_<patch>px_0px_overlap/
    patches/<slide>_patches.h5      coords only
    features_<encoder>/<slide>.h5   features (n, d) + coords (n, 2)
    _config_feats_<encoder>.json    the run config
```

Reproducing an extraction (one encoder, one magnification):

```bash
python run_batch_of_slides.py \
    --task all --wsi_dir /path/to/slides \
    --job_dir /users/home/dchanda/trident_features/<cohort> \
    --patch_encoder uni_v2 --mag 10 --patch_size 256
```

Two consequences shape the whole project:

- **Only final pooled embeddings are stored.** Intermediate block activations are
  not recoverable, which is why Phase IV re-runs the models on re-cropped
  patches (§9).
- **The coordinate grid belongs to the directory, not the encoder.** Every
  encoder run against `10x_256px` inherits that grid — the basis of §5.

Inspect what exists at any time:

```bash
python scripts/scan_features.py --verify 2   # inventory + verify coordinate pairing
```

---

## 5. The pairing constraint and encoder sets

Every metric compares two matrices **row by row**: row *i* must be the same
tissue patch under both models. Because the coordinate grid belongs to the
`(magnification, patch_size)` directory:

> Encoders sharing a grid are paired **by row index**. Encoders on different
> grids are **not** — a 224px grid and a 256px grid cover different tissue.

So the unit of analysis is `(cohort, magnification, patch_size)`, a **feature
group**, and **all 12 encoders cannot be compared at once** without
re-extraction. Coordinates were verified byte-identical within every group.

| Group | Encoders | *n* | Slides |
|---|---|---|---|
| **`cptac_benchmark/10x_256px`** | conch_v1, ctranspath, gigapath, keep, resnet50, uni_v2 | **6** | 2296 |
| `cptac_benchmark/{5,20}x_256px` | ctranspath, gigapath, keep, resnet50, uni_v2 | 5 | 2296 |
| `master_benchmark/10x_256px` | conch_v1, ctranspath, gigapath, resnet50, uni_v2 | 5 | 2169 |
| `master_benchmark/20x_256px` | ctranspath, gigapath, keep, resnet50, uni_v2 | 5 | 1126 |
| `{cptac,master}/20x_224px` | gpfm, hoptimus0, virchow, virchow2 | 4 | 2296 / 2169 |
| `*/{5,10}x_224px` | gpfm, hoptimus0, virchow2 | 3 | — |
| `*/512px` | conch_v1, conch_v15 | 2 | — |
| `*/384px` | musk | 1 | — |

**`cptac_benchmark/10x_256px` is the flagship**: six encoders spanning all three
families across 2296 slides. The 224px group is a useful second experiment (four
large vision-SSL models) but has **no vision-language model**, so RQ2 must be
answered on the 256px group.

MUSK is stranded alone at 384px and CONCH v1.5 pairs only with CONCH v1 at 512px;
both need re-extraction at 256px to join the main comparison.

```python
from utils.features import FeatureStore

store = FeatureStore()
group = store.best_group()                      # cptac_benchmark/10x_256px
views = group.sample_patches(n_patches=20000)   # {encoder: (20000, d)}, row-paired
```

---

## 6. Labels: TCGA and CPTAC

No manual annotation is required — every label is derivable or downloadable.
Downloads land in `configs/clinical/`.

### 6.1 TCGA subtype — from the GDC manifests (already on disk)

GDC manifests are **per subtype**, so the manifest a slide appears in *is* its
label:

```
Datasets/TCGA-LUNG/manifest/gdc_manifest.LUAD.txt   → slides labelled LUAD
Datasets/TCGA-LUNG/manifest/gdc_manifest.LUSC.txt   → slides labelled LUSC
```

`utils.labels.build_tcga_labels()` parses these; the patient id is the first 12
characters of the barcode (`TCGA-05-4244`).

### 6.2 TCGA clinical — from the GDC API

```bash
python scripts/download_clinical.py        # → configs/clinical/tcga_clinical.csv
```

Queries `https://api.gdc.cancer.gov/cases` for open-access, case-level fields:
`primary_diagnosis`, `ajcc_pathologic_stage`, `vital_status`, `days_to_death`,
`days_to_last_follow_up`. Returns 2187 patients (BRCA 1098, LUAD 585, LUSC 504).

Derived columns:

- **`brca_subtype`** — `Infiltrating duct carcinoma, NOS` → IDC,
  `Lobular carcinoma, NOS` → ILC. Cases carrying *both* diagnoses are dropped
  rather than forced into a class.
- **`stage_group`** — `Stage I/II` → early, `Stage III/IV` → late. Substages
  carry a trailing letter (`Stage IIIA`), so the numeral is matched as a prefix;
  `Stage X` and `Stage 0` map to nothing.

Survival columns (`event`, `time`) are computed; no survival task is registered
yet.

### 6.3 CPTAC cohort — from the TCIA manifests (already on disk)

`Datasets/CPTAC/*/manifest/*_manifest.csv` carry `collection`, `patient_id` and
`cancer_type`. Feature-store slide names use any of three identifier spellings
(bare `slide_id`, `patient-slide`, or the URL basename), so all three are
indexed; all 2296 slides matched **exactly**, with zero cross-cohort ambiguity.

### 6.4 CPTAC mutations — from cBioPortal

```bash
python scripts/download_cptac_mutations.py    # → configs/clinical/cptac_mutations.csv
```

Studies chosen for maximal overlap with the extracted WSIs:

| Cohort | cBioPortal study | Genes | Sequenced patients |
|---|---|---|---|
| CPTAC-BRCA | `breast_cptac_gdc` | PIK3CA, MAP3K1, GATA3 | 126 |
| CPTAC-COAD | `coad_cptac_2019` | KRAS, PIK3CA, TP53 | 106 |
| CPTAC-LUAD | `luad_cptac_gdc` | TP53, STK11, KRAS | 229 |
| CPTAC-LSCC | `lusc_cptac_2021` | TP53, PIK3R1, KEAP1 | 108 *(unused)* |

Two correctness rules, both of which change what the labels mean:

- **The denominator is the sequenced cohort.** A patient with no mutation record
  is wild-type *only if sequenced at all*. Patients absent from the study's
  `_sequenced` sample list have unknown status and are **excluded**, never
  silently called WT — that would pad the negative class with unprofiled cases.
- **Silent mutations do not count** — only non-synonymous consequences.

Observed frequencies match the literature (LSCC TP53 94%, LUAD KRAS 30%, BRCA
PIK3CA 31%), which is the sanity check that the join is correct.

---

## 7. Downstream tasks

**14 tasks, every one confined to a single cancer type.** Tissue-of-origin tasks
(breast-vs-lung, CPTAC 4-class) are deliberately excluded: they saturate for any
pathology encoder and so cannot rank representations, which is the entire point
of Phase VIII here.

| Task | Classes | Slides | Patients | Balance |
|---|---|---|---|---|
| `tcga_nsclc` | LUAD / LUSC | 1043 | 946 | 531 / 512 |
| `tcga_brca_subtype` | IDC / ILC | 958 | 898 | 769 / 189 |
| `tcga_brca_stage` | early / late | 1012 | 946 | 767 / 245 |
| `tcga_nsclc_stage` | early / late | 831 | 752 | 661 / 170 |
| `cptac_nsclc` | LUAD / LSCC | 1273 | 275 | 1139 / 134 |
| `cptac_luad_tp53` | WT / MUT | 1058 | 227 | 49% mutated |
| `cptac_luad_kras` | WT / MUT | 1058 | 227 | 32% mutated |
| `cptac_luad_stk11` | WT / MUT | 1058 | 227 | 13% mutated |
| `cptac_brca_pik3ca` | WT / MUT | 377 | 116 | 30% mutated |
| `cptac_brca_gata3` | WT / MUT | 377 | 116 | 10% mutated |
| `cptac_brca_map3k1` | WT / MUT | 377 | 116 | 8% mutated |
| `cptac_coad_tp53` | WT / MUT | 223 | 106 | 47% mutated |
| `cptac_coad_kras` | WT / MUT | 223 | 106 | 32% mutated |
| `cptac_coad_pik3ca` | WT / MUT | 223 | 106 | 22% mutated |

```bash
python scripts/downstream_mil.py --list-tasks     # descriptions + caveats
```

**Mutation prediction from H&E is a weak-signal task** — AUCs of 0.6–0.75 are the
norm — which is exactly why these discriminate between encoders where subtyping
saturates near 0.97.

**Every split is patient-grouped.** TCGA and CPTAC both have several slides per
patient; a slide-level random split puts the same patient on both sides and
inflates every number. `utils.labels.grouped_split` is the only splitter exposed,
and it stratifies by assigning each patient their majority label.

Three input conditions per task:

1. **single** — one encoder's patches (per-encoder baseline)
2. **concat** — all encoders concatenated per patch. Strictly *more* information
   than the shared space, so this is the bar Phase V must clear.
3. **shared** — patches mapped through a fitted aligner, fitted on **training
   slides only**

MIL heads: `ABMIL` (gated attention), `TransMIL` (self-attention), `mean`
(pooling control, to show what attention actually buys).

---

## 8. Metrics

### 8.1 Representational similarity (Phase I)

Given row-paired, column-centered $X \in \mathbb{R}^{n \times d_1}$ and
$Y \in \mathbb{R}^{n \times d_2}$:

**Linear CKA** — invariant to rotation and isotropic scale, computed in feature
space at $O(nd^2)$ rather than the $O(n^2)$ Gram form:

$$\mathrm{CKA}(X,Y)=\frac{\lVert Y^\top X\rVert_F^2}{\lVert X^\top X\rVert_F\;\lVert Y^\top Y\rVert_F}$$

**Kernel CKA** — the nonlinear counterpart, via HSIC on centered Gram matrices
$K,L$ with an RBF kernel whose bandwidth is a fraction of the median pairwise
distance:

$$\mathrm{CKA}_k=\frac{\mathrm{HSIC}(K,L)}{\sqrt{\mathrm{HSIC}(K,K)\,\mathrm{HSIC}(L,L)}}$$

**SVCCA** — PCA-truncate each view to a variance threshold, then average the
canonical correlations: $\frac{1}{k}\sum_i \rho_i$.

**PWCCA** — weight each canonical correlation by how much of the original
representation its direction accounts for ($h_i$ = i-th canonical variate of $X$,
$x_j$ = its feature columns):

$$\alpha_i=\sum_j\lvert\langle h_i, x_j\rangle\rvert,\qquad \mathrm{PWCCA}=\sum_i\tilde{\alpha}_i\,\rho_i$$

**Orthogonal Procrustes** — the value the optimal rotation attains, normalised;
$\lVert\cdot\rVert_*$ is the nuclear norm:

$$s(X,Y)=\frac{\lVert X^\top Y\rVert_*}{\lVert X\rVert_F\,\lVert Y\rVert_F}$$

**Cosine RSA** — Spearman correlation between the two models' cosine
representational similarity matrices (strict upper triangles).

**Distance correlation** — zero *iff* statistically independent:

$$\mathrm{dCor}=\sqrt{\frac{\mathrm{dCov}^2(X,Y)}{\sqrt{\mathrm{dVar}^2(X)\,\mathrm{dVar}^2(Y)}}}$$

| Metric | Invariant to | Cost | Watch out for |
|---|---|---|---|
| `linear_cka` | rotation, isotropic scale | O(nd²) | blind to nonlinear structure |
| `kernel_cka` | rotation, isotropic scale | O(n²) | sweep the bandwidth |
| `svcca` | any invertible linear map | O(nd²) | `var_threshold` is the key knob |
| `pwcca` | any invertible linear map | O(nd²) | asymmetric; symmetrised by default |
| `procrustes` | rotation, isotropic scale | O(nd²) | the only one returning the transform |
| `cosine_rsa` | rotation, isotropic scale | O(n²) | Spearman over n(n−1)/2 pairs |
| `distance_correlation` | rotation, isotropic scale | O(n²d) | use `unbiased=True` (default) |

> **CCA-family metrics saturate.** SVCCA scores ~0.63 between two *independent*
> random matrices, because with enough dimensions some directions correlate by
> chance. Read SVCCA/PWCCA as rankings, never as absolute agreement, and always
> report linear CKA beside them.

Also required: **n_samples ≫ n_features**. With Virchow's 2560 dimensions a few
thousand patches is the floor; the code warns when `n ≤ d`.

### 8.2 Alignment (Phase V)

Six methods behind one encode/decode interface.

**GCCA (MAX-VAR)** — posits one latent $G$ that every model is a linear view of:

$$\min_{G,\,U_1\ldots U_M}\ \sum_m \lVert G - X_m U_m\rVert_F^2 \quad\text{s.t.}\quad G^\top G = I$$

Its eigenvalues lie in $[1, M]$; a value near $M$ means all $M$ models agree on
that direction. **The eigenvalue spectrum is the headline diagnostic** — it reads
out how many dimensions of morphology are actually shared.

**MCCA (SUMCOR)** — the same question via pairwise correlations, relaxed to the
generalised eigenproblem $Cw=\lambda Dw$ with $C$ the block covariance of the
concatenated views and $D$ its block diagonal. Reduces exactly to CCA at $M=2$.

**Generalized Procrustes** — the strictest test, *rotation only*:
$\min_{Q}\lVert X - YQ\rVert_F$ subject to $Q^\top Q=I$.

**Joint PCA** — the baseline everything must beat.
**Shared autoencoder** — the only nonlinear aligner: per-model encoders and
decoders with reconstruction plus a latent alignment term.
**Optimal transport** — Wasserstein Procrustes; supervised mode works,
unsupervised is exploratory and unreliable.

Evaluation deliberately reports **two opposing failure modes** together, because
optimising either alone gives a degenerate result:

- *under-alignment* — `alignment_error`, `paired_cosine`, `recall@1`, `mrr`
- *collapse* — `reconstruction_r2`, `effective_rank`, `neighborhood_preservation`

### 8.3 Retrieval (Phase VII) and transfer (Phase VI)

With binary relevance, $R$ relevant items and ranked hits $h_i$:

$$\mathrm{AP}=\frac{1}{R}\sum_i \frac{\sum_{j\le i} h_j}{i}\,h_i,\qquad
\mathrm{MRR}=\frac{1}{\lvert Q\rvert}\sum_q\frac{1}{\mathrm{rank}_q},\qquad
\mathrm{NDCG}=\frac{\sum_i h_i/\log_2(i+1)}{\sum_{i\le R}1/\log_2(i+1)}$$

Two relevance modes, because only one makes all four metrics distinct:

- **identity** — the correct answer is the *same patch* (one relevant item), so
  mAP collapses to MRR by definition
- **label** — all patches of the same class are relevant, making mAP and NDCG
  genuinely different measures of ranking quality

The **unaligned control** is per-model independent PCA to the same dimension, so
any gain is attributable to alignment rather than to dimensionality reduction.

Phase VI adds a **linear probe transfer**: train logistic regression on the
target's *real* embeddings, then test it on embeddings translated from the
source. The gap to the ceiling is what Phase VI actually measures.

Downstream (Phase VIII): AUC, accuracy, macro F1, balanced accuracy. **Read
balanced accuracy and AUC** — every task is imbalanced, so plain accuracy
flatters a majority-class predictor.

---

## 9. Running the study

### End to end

```bash
python scripts/run_study.py --out results/smoke --preset smoke      # ~35 min, validates the chain
python scripts/run_study.py --out results/main  --preset standard
python scripts/run_study.py --out results/paper --preset full       # all 14 tasks, 5 aligners
```

Seven stages — `inventory`, `similarity`, `magnification`, `alignment`,
`transfer`, `retrieval`, `downstream`. Each is a separate subprocess, so one
failure does not abort the rest and any stage re-runs alone:

```bash
python scripts/run_study.py --out results/main --stages retrieval downstream
python scripts/run_study.py --out results/main --preset full --dry-run
```

| Preset | patches | latent dim | MIL epochs | tasks | aligners |
|---|---|---|---|---|---|
| `smoke` | 3k | 32 | 5 | 1 | joint_pca, gcca |
| `standard` | 20k | 64 | 50 | 4 | + procrustes |
| `full` | 50k | 64 | 80 | all 14 | + mcca, autoencoder |

### Across two GPUs

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_study.py --out results/full_run/analysis \
    --preset full --stages inventory similarity magnification alignment transfer retrieval &
CUDA_VISIBLE_DEVICES=1 python scripts/run_study.py --out results/full_run/downstream \
    --preset full --stages downstream &
```

Logs land in `<out>/logs/<stage>.log`; `<out>/run_manifest.json` records exactly
what ran. `make_figures.py` handles either layout.

### Individual stages

```bash
python scripts/representation_similarity.py --group best --n-patches 20000 --out results/sim
python scripts/magnification_ablation.py    --series best --out results/mag
python scripts/shared_latent_space.py       --group best --latent-dim 64 --out results/align
python scripts/cross_model_transfer.py      --list-pairs
python scripts/cross_model_retrieval.py     --group best --out results/retr
python scripts/downstream_mil.py --task tcga_nsclc --group master_benchmark/10x_256px --out results/mil
```

### Phase IV needs images, not the store

Intermediate activations were never saved, so layer-wise analysis re-runs the
models with forward hooks on re-cropped patches:

```bash
python scripts/download_slides.py --n 4        # ~77 MB, slides already in the store
python scripts/layerwise_alignment.py --encoders uni_v2 conch_v1 ctranspath resnet50 \
    --n-patches 512 --pool cls --out results/layerwise
```

Slides are chosen to be *already in the feature store*, so patches are re-cropped
at trident's own coordinates — the same tissue as every other phase.
`--pool {cls,mean,cls_mean}` materially changes the answer; sweep it.

> GigaPath cannot be used here: trident's loader asserts `timm==0.9.16` and the
> environment has 1.0.26. Its stored features are unaffected — this affects only
> Phase IV, the sole stage that loads weights.

---

## 10. Figures

```bash
python scripts/make_figures.py --run results/full_run --out results/figures --format pdf
```

`utils/paperfigs.py` provides two archetypes:

- **`heatmap_row`** — panels each with an **independent** colourbar,
  `mask='lower'` for the staircase look, bold in-cell values that flip
  white/black against cell brightness. Independent scales are the point: metrics
  sit at different levels, and a shared scale flattens every panel but the
  widest-ranging one. Use `shared_limits=True` only when the *level shift itself*
  is the finding — the magnification row does.
- **`grouped_bars`** — light-to-dark ramp with the highlighted series darkest,
  values printed vertically inside the bars, dashed separators between groups.

All labels pass through `clean_label()`, so registry keys never reach a figure:
`uni_v2` → UNI2, `gigapath` → GigaPath, `resnet50` → ResNet50,
`cptac_luad_tp53` → CPTAC LUAD TP53. Unknown identifiers fall back to
underscores→hyphens, so a newly added encoder cannot leak an underscore.

---

## 11. Results so far

- **Cross-model retrieval works only with alignment.** GCCA reaches Recall@1
  **0.81** on held-out patches (NDCG 0.91); the unaligned control sits at 0.0015
  against a chance level of 0.0011.
- **Transfer preserves decisions, not geometry.** A probe trained on model B's
  *real* embeddings keeps **96.4%** of its accuracy on features translated from A
  (0.820 → 0.787), while reconstruction is only R² 0.59. Crossing models costs
  just 0.6 points beyond the shared space itself — so report the probe, not R².
- **A rotation is not enough.** Rigid Procrustes gets Recall@1 0.41 where
  GCCA/joint-PCA get ~0.91: the models are related by general linear maps, not
  rotations. Procrustes does preserve local neighbourhoods best (0.88 vs 0.60).
- **Agreement falls with magnification** (CKA 0.62 → 0.60 → 0.56 at 5x/10x/20x),
  and the *structure* shifts: 5x-vs-20x rank stability is only 0.73 Spearman.
  Which models look most alike depends on the magnification chosen.
- **Layer-wise, UNI2 and CONCH converge with depth** (CKA 0.836 early → 0.942
  late), contradicting the "universal early, model-specific late" expectation.
  CONCH↔ResNet50 diverges as expected. Directional only — 252 patches, CLS pooling.
- **The control behaves.** Pathology models agree with each other at linear CKA
  0.60–0.78, and with ResNet50 at only 0.39–0.59.

---

## 12. Layout, tests, requirements

```
utils/
  preprocessing.py          coercion, pairing checks, centering, subsampling
  cka.py cca.py procrustes.py cosine.py distance_correlation.py   Phase I metrics
  pairwise.py               metric registry, N x N similarity driver
  visualization.py          heatmaps, dendrograms, MDS, UMAP
  paperfigs.py              publication figures + canonical label names
  features.py               store discovery, row-paired patch sampling
  ablation.py               magnification ablation
  alignment/                6 aligners behind one encode/decode interface
  alignment_metrics.py      reconstruction, alignment quality, collapse detection
  transfer.py               Phase VI fidelity / retrieval / linear probe
  retrieval.py              Phase VII Recall@K, mAP, MRR, NDCG
  labels.py mil.py bags.py  Phase VIII tasks, MIL heads, bag construction
  layers.py                 Phase IV hooks + layer-wise similarity
configs/
  encoders.yaml             model registry (dims, HF ids, objectives)
  clinical/                 downloaded TCGA clinical + CPTAC mutation labels
scripts/
  run_study.py              end-to-end runner (7 stages)
  scan_features.py          inventory + pairing verification
  representation_similarity.py   magnification_ablation.py   shared_latent_space.py
  cross_model_transfer.py        cross_model_retrieval.py    downstream_mil.py
  layerwise_alignment.py         make_figures.py
  download_clinical.py           download_cptac_mutations.py  download_slides.py
tests/                      229 property tests
```

```bash
python -m pytest tests/ -q                 # 229 tests
python -m pytest tests/ -q -m "not slow"   # skip model training
```

Tests check mathematical invariants, not just outputs: self-similarity = 1,
invariance to rotation / scale / translation, symmetry, near-chance behaviour on
independent data, mAP and NDCG against hand-computed rankings, GCCA eigenvalue
bounds, MCCA-equals-CCA at $M=2$, Sinkhorn double stochasticity, patient-split
leakage, and that encoders on different grids never appear as a transferable
pair.

**Requirements:** numpy, scipy, scikit-learn, pandas, matplotlib, seaborn,
umap-learn, h5py, torch, pyyaml, joblib. Phase IV additionally needs openslide
and a trident checkout for the model zoo. All present in the base conda env on
`alt`.
