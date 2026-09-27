# DFR Reproduction — Waterbirds

**Reproducing:** Kirichenko, Izmailov & Wilson, *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*, ICLR 2023.
**Group 4:** Muhammad Asad Piracha (30621), Muhammad Taha Ali (31545), Aleena Aqeel (31635)

**Milestone scope:** Stage 2 (paper understanding) + Stage 3 (dataset prep, implementation, ERM baseline training). Deep Feature Reweighting (DFR) itself and the reweighting-size experiment are Week 4 work and are not part of this milestone.

---

## 1. Literature Review

### What is the problem, and why does it matter?
Image classifiers trained with standard empirical risk minimization (ERM) can learn to rely on
features that are correlated with the label in the training data but are not actually causal —
"spurious correlations." A well-known example: if most photos of waterbirds happen to be taken
near water, a model can get high training accuracy by learning "water background → waterbird"
instead of learning what a waterbird actually looks like. This matters beyond birds: the same
failure mode shows up in medical imaging (models keying on hospital-specific scanner artifacts
instead of pathology) and hiring/lending models keying on demographic proxies instead of
job-relevant signals. A model that looks accurate on average can be badly wrong on the specific
subgroup where the shortcut breaks.

### What did people do before this paper, and what was missing?
Prior work mostly falls into two camps:
- **Group-aware training** (e.g., Group DRO, Sagawa et al., 2020) explicitly reweights the
  training loss to prioritize the worst-performing group. This works, but it needs group labels
  (e.g., "this photo's background is water") for the **entire training set**, which is expensive
  or impossible to collect at scale.
- **Group-label-free heuristics** (e.g., JTT — Just Train Twice, Liu et al., 2021; Learning from
  Failure) try to *infer* which examples are likely minority-group members (typically, examples
  the first-pass model gets wrong) and upweight them in a second training pass. These avoid
  needing group labels during training, but still retrain the entire network and rely on the
  heuristic correctly identifying minority examples.

What was missing: nobody had cleanly shown *how much* of the network actually needs to change to
fix the problem — both camps retrain the whole network.

### What is the proposed solution, concretely?
The paper's central claim: a standard ERM-trained network already learns good, usable features —
the failure lives almost entirely in how the **final linear classification layer** weights those
features, not in the features themselves. Their method, **Deep Feature Reweighting (DFR)**:
1. Train a network normally (ERM), no group labels needed.
2. Freeze the entire network except the final linear layer.
3. Retrain *only* that last layer — via logistic regression, not gradient descent — on a small,
   group-balanced dataset (group labels are needed here, but only for a small held-out set, not
   the whole training set).

### What's actually novel?
Not the architecture, not the loss function, not the training procedure for the base model — all
of that is standard. The novelty is the **empirical claim** that step 3 alone (last-layer-only
retraining on a *small* balanced set) is sufficient to close most of the worst-group accuracy gap,
challenging the assumption that fixing spurious correlations requires retraining deep
representations. This is a data-strategy and evaluation-protocol contribution more than an
architectural one.

### What's the full algorithm, step by step?
1. Train ResNet-50 (or similar) with standard cross-entropy loss on the biased training set — no
   group information used. This is the ERM baseline (**this milestone's deliverable**).
2. Freeze the trained network's weights except the final `fc` layer.
3. Pass a small, group-balanced set of held-out examples through the frozen network to get their
   penultimate-layer features (2048-dimensional for ResNet-50).
4. Fit a logistic regression classifier on those features (**Week 4**).
5. Evaluate: replace the original last layer with this newly fit classifier and measure worst-group
   accuracy on the test set.

### What dataset(s), and why those?
The paper evaluates on **Waterbirds** and **CelebA** (a face dataset where hair color is spuriously
correlated with gender label). We reproduce on **Waterbirds only** — a deliberate scope reduction
for our compute/time budget, since Waterbirds is smaller (~4,795 train images vs. CelebA's
~160,000) and sufficient to demonstrate the core phenomenon and the method.

### What metrics, and what do they measure (and fail to measure)?
- **Overall (average) accuracy**: fraction of all test examples correctly classified. Fails to
  measure whether the model is actually fair/robust across subgroups — a model that is right 95%
  of the time can still be consistently wrong on a specific 10% subgroup, and this metric would
  hide that.
- **Worst-group accuracy**: accuracy on the single worst-performing group (of the four
  label × background combinations). This is the paper's primary metric — it directly measures the
  failure mode being studied and can't be inflated by majority-group performance.

### What were the headline results, and what limitations did the authors flag?
The paper reports that last-layer retraining (DFR) closes most of the gap between ERM and
group-aware methods like Group DRO, on both Waterbirds and CelebA, despite needing group labels
only on a small held-out set rather than the full training set. Limitations the authors
acknowledge: DFR still needs *some* group-labeled data (just much less than Group DRO), and its
success depends on the frozen feature extractor having already learned adequate features in the
first place — if the ERM-trained backbone's representations are too entangled with the spurious
feature, no amount of last-layer retraining can recover information that was never captured.

### Cross-checking the field's reception
Prior work this paper builds on/against: Group DRO (Sagawa et al., 2020) and JTT (Liu et al., 2021)
are the main comparison points discussed above. Follow-up work citing DFR (e.g., approaches using
generative/synthetic data to build the balanced reweighting set without needing real group labels
at all) treats "freeze and retrain the last layer" as a standing building block, extending it
toward removing DFR's remaining dependency on group labels — which suggests the field received the
core claim as durable rather than a one-off result.

---

## 2. Design & Workflow

### Compute & environment
- **Google Colab, free-tier T4 GPU.**
- Dependencies pinned in `requirements.txt` per the brief's environment-hygiene guidance (paper-era
  repos often assume older library versions).
- Dataset accessed via the `wilds` package rather than a hand-rolled downloader — a standard,
  citable data-loading utility (see `PROVENANCE.md`). `wilds`'s default CodaLab download source
  proved unreliable during development, so the notebook falls back to a direct Stanford-hosted
  mirror of the same archive when needed.

### Workflow (following the brief's recommended order)
1. **Minimal forward pass first**: one batch through an untrained model, checking tensor shapes and
   that the loss is finite, before any training was launched.
2. **Tiny-subset sanity check**: trained on 200 images for 2 epochs to confirm the training loop
   itself has no bugs, before committing to a full run.
3. **Full training with per-epoch logging**: overall and per-group validation accuracy logged every
   epoch (not just at the end), so the point where the worst-group/overall gap opens up is visible
   in the training curve, not just the final number.
4. **Evaluation with the paper's own metric definitions**: worst-group accuracy computed the same
   way as the paper — accuracy on the single lowest-performing of the four label × background
   groups, not a soft or averaged variant.
5. **Honest comparison**, not just reporting a number — see Results below.

### Key design choices

| Choice | What we used | Why |
|---|---|---|
| Model | ResNet-50, ImageNet-pretrained (`torchvision`) | Matches the paper; transfer learning is necessary given our data/compute budget |
| Training method | Standard ERM (cross-entropy, no group info) | Deliberately the "before" model the paper studies |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 | Standard for CNN fine-tuning |
| LR schedule | 1e-3, cosine annealing | Standard fine-tuning default |
| Batch size | 32 | Fits a free-tier T4 |
| Epochs | 15 | **Reduced from the paper's ~100-epoch sweep** — a deliberate compute-budget cut, sufficient to see the spurious-correlation gap clearly |
| Preprocessing | 224×224, ImageNet normalization; random crop + flip on train only | Required for pretrained ResNet-50; augmentation limits overfitting on a small dataset |

---

## 3. Implementation & Reproduction — Results

### Dataset

| Split | Size |
|---|---|
| Train | 4,795 |
| Validation | 1,199 |
| Test | 5,794 |

Four groups tracked (bird type × background): landbird/land and waterbird/water are the majority
("expected") combinations; landbird/water and waterbird/land are the minority combinations where
the spurious shortcut fails.

### ERM baseline training (validation accuracy per epoch)

| Epoch | Train loss | Val overall acc. | Val **worst-group** acc. |
|---|---|---|---|
| 1 | 0.273 | 0.759 | 0.248 |
| 5 | 0.064 | 0.861 | 0.647 |
| 9 | 0.040 | 0.890 | 0.692 |
| 12 | 0.031 | 0.893 | 0.707 |
| 15 | 0.030 | 0.892 | 0.669 |

Final validation per-group accuracy (epoch 15): landbird/land 0.998, landbird/water 0.830,
waterbird/land 0.669, waterbird/water 0.962.

### Held-out test set (final numbers)

| Metric | Value |
|---|---|
| Overall accuracy | 0.897 |
| Worst-group accuracy | 0.732 |
| landbird / land (majority) | 0.996 |
| landbird / water (minority) | 0.834 |
| waterbird / land (minority) | 0.732 |
| waterbird / water (majority) | 0.941 |

### Comparison to prior literature (honest comparison, per the brief)

Standard ERM baselines reported for Waterbirds in the literature commonly fall in the **60-69%
worst-group accuracy** range (e.g., Sagawa et al., 2020; subsequent group-robustness papers using
it as a baseline). Our worst-group accuracy of **73.2%** on the test set is in a similar ballpark,
slightly higher than the most commonly cited figures. We attribute this most plausibly to:
- **Fewer training epochs (15 vs. the ~100+ often used)** — less time for the model to fully
  overfit to the majority-group shortcut, which can paradoxically leave the minority groups
  slightly better off than a longer, more thoroughly converged ERM run.
- **No hyperparameter sweep** — we used standard fine-tuning defaults rather than tuning learning
  rate/weight decay specifically for this dataset, which introduces run-to-run variance (we saw
  worst-group accuracy vary by several points across our own reruns with identical settings, purely
  from training randomness).

We are not treating this as a failure to match — the qualitative pattern the paper predicts is
clearly present and is what matters for this milestone: **overall accuracy is high (~90%) while
worst-group accuracy is meaningfully lower (~73%)**, with waterbird/land consistently the hardest
group across every run. That gap is precisely the phenomenon DFR is designed to close, and closing
it is Week 4's work.

---

## 4. Limitations (Week 3 scope)

- **15 epochs**, vs. the paper's more extensive training/hyperparameter sweep — a deliberate
  reduction for a free-tier GPU budget.
- No hyperparameter tuning beyond standard fine-tuning defaults.
- Reproduced on **Waterbirds only**, not CelebA (the paper's second dataset) — a scope reduction
  for time/compute, noted here explicitly per the brief's guidance on dataset substitutions.
- This is the ERM half only. DFR (last-layer retraining) and the reweighting-size experiment are
  Week 4 work, not included in this repository yet.

## 5. Repository contents

- `DFR_Waterbirds_Reproduction_Week3.ipynb` — full notebook: dataset download, sanity checks, ERM
  training, evaluation.
- `PROVENANCE.md` — tracks which code is ours vs. reused libraries vs. method-from-the-paper, per
  academic integrity guidelines.
- `requirements.txt` — pinned dependency versions.

Checkpoints (`erm_final.pt`) and raw result logs are kept in Google Drive rather than this repo,
since they exceed GitHub's practical file-size limits — see `.gitignore`.

## 6. How to reproduce

1. Open the notebook in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Run all cells top to bottom. Dataset download + extraction: a few minutes. Full ERM training:
   ~30-60 minutes on a free T4.

Note: rerunning training produces slightly different numbers each time (data shuffling and weight
initialization involve randomness) — across our reruns, test-set overall/worst-group accuracy
landed consistently around 0.897/0.73, so the qualitative pattern is stable even though exact
per-epoch numbers vary run to run.
