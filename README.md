# DFR Reproduction — Waterbirds

**Reproducing:** Kirichenko, Izmailov & Wilson, *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*, ICLR 2023.
**Group 4:** Muhammad Asad Piracha (30621), Muhammad Taha Ali (31545), Aleena Aqeel (31635)

## What this paper is about

Image classifiers can learn to rely on background instead of the actual object when the two are
correlated in training data. This paper's claim: the model still learns the real (foreground)
features perfectly well — the problem is entirely that the **final linear layer** overweights the
spurious background feature. Their fix, **Deep Feature Reweighting (DFR)**: freeze the trained
feature extractor and retrain only the last layer on a small, group-balanced dataset. This
notebook reproduces the "before" half of that story: training the biased ERM baseline and
confirming the bias is real and measurable. DFR itself (the fix) is our Week 4 milestone.

## Dataset

**Waterbirds** — CUB bird photos composited onto Places backgrounds, constructed so that in
training, 95% of waterbirds appear on water backgrounds and 95% of landbirds on land backgrounds.
Validation and test sets are balanced 50/50 across backgrounds, so they expose the shortcut.

| Split | Size |
|---|---|
| Train | 4,795 |
| Validation | 1,199 |
| Test | 5,794 |

Four groups are tracked throughout, defined by (bird type × background):

| Group | Description | Frequency in training |
|---|---|---|
| 0 | landbird / land | majority |
| 1 | landbird / water | minority |
| 2 | waterbird / land | minority |
| 3 | waterbird / water | majority |

Downloaded directly from Stanford's mirror (`nlp.stanford.edu/data/dro/...`) rather than through
the `wilds` package's default CodaLab source, which is unreliable — see the notebook for details.

## Design choices

| Choice | What we used | Why |
|---|---|---|
| Model | ResNet-50, ImageNet-pretrained (`torchvision`) | Matches the paper; transfer learning is necessary given our data/compute budget |
| Training method | Standard ERM (cross-entropy, no group info used) | This is deliberately the "before" model the paper studies |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 | Standard choice for CNN fine-tuning; matches the paper's optimizer family |
| Learning rate schedule | 1e-3 with cosine annealing | Standard fine-tuning default |
| Batch size | 32 | Fits comfortably on a free-tier Colab T4 GPU |
| Epochs | 15 | **Reduced from the paper's ~100-epoch sweep** — a deliberate compute-budget cut; 15 epochs was enough to see the spurious-correlation gap clearly (see Results) |
| Image preprocessing | Resize/crop to 224×224, ImageNet normalization; random crop + horizontal flip on train only | Required input format for a pretrained ResNet-50; augmentation reduces overfitting on a small dataset |
| Data source | Direct Stanford tarball, not `wilds`'s CodaLab downloader | CodaLab downloads were failing (0-byte / timeout); same underlying dataset either way |

## Results — ERM baseline

Validation accuracy per epoch (final training run, 15 epochs):

| Epoch | Train loss | Val overall acc. | Val **worst-group** acc. |
|---|---|---|---|
| 1 | 0.269 | 0.736 | 0.188 |
| 5 | 0.066 | 0.869 | 0.602 |
| 9 | 0.038 | 0.891 | 0.669 |
| 12 | 0.031 | 0.896 | 0.632 |
| 15 | 0.032 | 0.896 | 0.654 |

Final validation per-group accuracy (epoch 15):

| Group | Accuracy |
|---|---|
| landbird / land (majority) | 0.996 |
| landbird / water (minority) | 0.848 |
| waterbird / land (minority) | 0.654 |
| waterbird / water (majority) | 0.955 |

**Held-out test set (final numbers):**

| Metric | Value |
|---|---|
| Overall accuracy | 0.897 |
| Worst-group accuracy | 0.732 |
| landbird / land (majority) | 0.996 |
| landbird / water (minority) | 0.832 |
| waterbird / land (minority) | 0.732 |
| waterbird / water (majority) | 0.942 |

### What this shows

Overall accuracy converges to ~89-90% on both validation and test, but **worst-group accuracy
lands at 73% on the test set** (~65-70% during validation) — roughly a 16-17 point gap from
overall accuracy on test. In both validation and test, the worst-performing group is consistently
**waterbird / land**: the model, having learned "water background → waterbird" as a shortcut,
struggles precisely on the images where that shortcut fails. This is exactly the failure mode the
paper predicts for a standard ERM model, and it's the gap DFR is designed to close in Week 4.

Note: rerunning training produces slightly different numbers each time (data shuffling and
weight initialization involve randomness) — the exact figures above are from our final run, not
necessarily reproducible bit-for-bit, though the qualitative pattern (overall accuracy high,
worst-group accuracy noticeably lower, waterbird/land as the hardest group) is stable across runs.

## Limitations (Week 3 scope)

- **15 epochs**, vs. the paper's more extensive training schedule and hyperparameter sweep — a
  deliberate reduction for a free-tier GPU budget, not a bug.
- No hyperparameter tuning (learning rate, weight decay) beyond standard fine-tuning defaults —
  the paper tunes these per dataset; we did not replicate that sweep.
- This baseline is the ERM half only. DFR (last-layer retraining) and the reweighting-size
  experiment are Week 4 work, not included in this repo yet.

## Repository contents

- `DFR_Waterbirds_Reproduction_Week3.ipynb` — full notebook: dataset download, sanity checks, ERM
  training, evaluation.
- `PROVENANCE.md` — tracks which code is ours vs. reused libraries, per academic integrity
  guidelines.
- `requirements.txt` — pinned dependency versions.

Checkpoints (`erm_final.pt`) and raw result logs are stored in Google Drive rather than this repo,
since they exceed GitHub's practical file-size limits — see `.gitignore`.

## How to reproduce

1. Open the notebook in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Run all cells top to bottom. Dataset download + extraction: a few minutes. Full ERM training:
   ~30-60 minutes on a free T4.
