# Towards a Universal Latent Space for Computational Pathology Foundation Models

**Status:** Research plan (v1)
**Location:** `~/pfm_latent_space` (host: `alt`)

### Alternative titles

- Do Computational Pathology Foundation Models Learn a Common Morphological Representation?
- Learning a Shared Latent Space Across Computational Pathology Foundation Models
- Foundation Model Interoperability in Computational Pathology via Representation Alignment

---

## 1. Motivation

Recent pathology foundation models (PFMs) — UNI, CONCH, KEEP, MUSK, Virchow, Phikon, GigaPath, and others — differ along every axis that is normally assumed to matter:

- architecture
- pretraining objective
- pretraining dataset
- multimodal supervision

Despite these differences, many reach comparable downstream performance. This raises a fundamental question:

> **Do independently trained pathology foundation models converge to a common latent representation of tissue morphology?**

Existing work compares these models almost exclusively through downstream metrics (AUC, accuracy, F1). Very little is known about the relationships *between their internal representation spaces*. This project sets out to characterize, align, and exploit those spaces.

---

## 2. Central hypothesis

**Independent pathology foundation models learn different coordinate systems over a shared underlying morphological manifold.**

If the hypothesis holds, then:

- representations should be alignable across models,
- semantic neighborhoods should be preserved under alignment,
- features should transfer between models,
- downstream classifiers should be able to operate inside a single shared latent space.

---

## 3. Research questions

| # | Question |
|---|---|
| RQ1 | How similar are representation spaces across pathology foundation models? |
| RQ2 | Do different pretraining objectives produce similar latent geometry? |
| RQ3 | Can multiple PFMs be aligned into a shared latent representation? |
| RQ4 | Does the aligned representation improve downstream learning? |
| RQ5 | Can a downstream model trained on one encoder generalize to another encoder after alignment? |

---

## 4. Datasets

Use the existing benchmark suite. Candidates:

- TCGA
- PANDA
- CAMELYON
- BRACS
- RCC
- (extend as needed)

**Key requirement:** every image patch must have embeddings from *all* foundation models under study — paired embeddings are what make similarity and alignment analysis well-defined.

## 5. Models

Candidate PFMs (extend depending on availability and licensing):

- UNI
- UNI2
- CONCH
- KEEP
- MUSK
- Virchow
- Phikon
- GigaPath

---

## Phase I — Representation similarity

> **Status: implemented.** All seven metrics and the three outputs live in
> `utils/`, driven by `scripts/run_phase1.py`. See [README.md](README.md).

**Goal:** quantify how similar different PFMs are.

**Methods**

- Linear CKA
- Kernel CKA
- SVCCA
- PWCCA
- Orthogonal Procrustes
- Cosine similarity
- Distance correlation

**Outputs**

- Pairwise similarity matrix (N × N heatmap)
- Hierarchical clustering of models
- MDS / UMAP visualization of the model space

**Expected insights**

- Which models are closest to one another?
- Does the training objective dominate similarity structure?
- Or does the architecture dominate?

---

## Phase II — Local geometry analysis

Global similarity can mask local structure, so measure neighborhood-level agreement directly.

**Metrics**

- Nearest-neighbor overlap
- Trustworthiness
- Continuity
- Neighborhood preservation
- Mutual k-NN consistency

**Question:** do different models preserve the *same* morphological neighborhoods?

---

## Phase III — Organ-wise analysis

Rather than computing similarity globally, compute representational similarity **per organ** (lung, kidney, colon, …).

**Questions**

- Does representation agreement depend on tissue type?
- Which organs are universally represented across models?
- Which organs are encoder-specialized?

---

## Phase IV — Layer-wise alignment

Compare *every transformer block*, not only the final embedding.

**Questions**

- Where in the network do models begin to diverge?
- Do early layers encode universal morphology?
- Are late layers model-specific?

**Deliverables**

- Layer-by-layer similarity heatmaps
- Alignment trajectories across depth

---

## Phase V — Shared latent space  *(core methodological contribution)*

> **Status: implemented.** All six approaches live in `utils/alignment/`
> behind one encode/decode interface, with evaluation in
> `utils/alignment_metrics.py` and a driver at `scripts/run_phase5.py`.
> See [README.md](README.md). Caveat: the *unsupervised* optimal-transport
> mode is exploratory and does not currently work reliably; supervised OT does.

**Goal:** construct a common latent representation shared across multiple PFMs.

**Candidate approaches**

- Generalized Canonical Correlation Analysis (GCCA)
- Multi-view CCA
- Orthogonal Procrustes
- Joint PCA
- Shared latent autoencoder
- Optimal transport alignment

**Outputs**

- The common embedding space itself
- Per-model projection functions (into and out of the shared space)
- Reconstruction error
- Alignment quality metrics

---

## Phase VI — Cross-model transfer

**Question:** can one model's representation be converted into another's?

**Experiments:** CONCH → UNI, UNI → CONCH, KEEP → MUSK, Virchow → CONCH.

**Evaluation:** cosine similarity, retrieval, linear-probe accuracy, feature reconstruction.

---

## Phase VII — Cross-model retrieval

Build the database with **model A**, issue queries with **model B**.

- Without alignment → baseline retrieval
- With alignment → cross-model retrieval

**Metrics:** Recall@K, mAP, MRR, NDCG.

---

## Phase VIII — Downstream learning

Keep the downstream head simple (ABMIL).

**Conditions**

1. Baseline: single encoder → ABMIL
2. Concatenation of encoders → ABMIL
3. Shared latent space → ABMIL

**Metrics:** AUC, accuracy, F1, balanced accuracy.

Optionally add one more MIL baseline (CLAM or TransMIL) to show the findings are not an artifact of ABMIL.

---

## Ablation studies

- Latent dimensionality
- Alignment method
- Number of encoders included
- Organ
- Training-set size
- Retrieval metric

---

## Biological analysis

Which morphology types show high cross-model agreement?

- Tumor
- Stroma
- Lymphocytes
- Necrosis
- Adipose
- Glands

**Questions:** which morphological concepts appear universal, and which remain encoder-specific?

---

## Expected contributions

1. The first large-scale representational similarity analysis across computational pathology foundation models.
2. A framework for aligning PFM embeddings into a shared latent space.
3. A systematic study of how pretraining objectives shape morphological representations.
4. A demonstration of cross-model interoperability for retrieval and downstream learning.
5. Evidence supporting — or refuting — the existence of a common latent morphology manifold.

---

## Open items

- Confirm dataset availability and which cohorts give full model coverage.
- Confirm which PFM weights are actually obtainable (gated repos / licensing).
- Decide compute split between `alt` and `raj` (HPC) for embedding extraction.
