# Provenance Log — DFR Waterbirds Reproduction (Group 4)

Update this file as you go, not retroactively. Tag every non-trivial code block as one of:
`written-by-us` / `adapted-from` / `reused-as-is`.

## Data pipeline
- Waterbirds loading via `wilds` package (`get_dataset`, `get_subset`, `get_train_loader`) —
  **reused-as-is** from the WILDS benchmark library (Koh et al., 2021). Not our code; it's a
  standard, citable data-loading utility.
- Group-id computation (`group_ids_from_metadata`), visualization, dataloader wiring —
  **written-by-us**.

## Model
- ResNet-50 architecture + ImageNet weights — **reused-as-is** from `torchvision.models`.
- Replacing the final `fc` layer, freezing for feature extraction — **written-by-us**.

## Training loop
- ERM training loop (optimizer, scheduler, per-epoch eval, checkpointing) — **written-by-us**,
  following standard PyTorch training-loop conventions, not copied from the official DFR repo.

## Results (Week 3)
- ERM baseline training curves and per-group/worst-group accuracy are **obtained by our own team**.

---
*Add a "DFR method" and "Experiment (reweighting-size sweep)" section here in Week 4, once that
work starts — don't backfill it now. Fill in dates/commits as you actually do the work.*
