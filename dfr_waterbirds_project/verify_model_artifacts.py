#!/usr/bin/env python3
"""
Verify the saved Waterbirds DFR artifacts WITHOUT Streamlit, images, or WILDS.

Run this from the project root that contains:
    checkpoints/erm_final.pt
    checkpoints/dfr_head.pt
    results/dfr_features.npz
    results/dfr_results.json

It checks:
  1. DFR head predictions on the saved test features.
  2. ERM fc predictions on the same saved test features.

If this passes, the model artifacts are correct. Any remaining problem is in
the app's inference code, image preprocessing, or demo-image labels.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "results" / "dfr_features.npz"
HEAD = ROOT / "checkpoints" / "dfr_head.pt"
ERM = ROOT / "checkpoints" / "erm_final.pt"
RESULTS = ROOT / "results" / "dfr_results.json"

EXPECTED_DFR = {
    "overall": 0.9309630652399034,
    "worst_group": 0.9086474501108648,
    "per_group": [0.9574279379157428, 0.9086474501108648, 0.9143302180685359, 0.9330218068535826],
}
EXPECTED_ERM = {
    "overall": 0.8940283051432516,
    "worst_group": 0.7897196412086487,
    "per_group": [0.995121955871582, 0.8057649731636047, 0.7897196412086487, 0.9532710313796997],
}
TOL = 1e-5


def metrics(pred, y, g):
    per_group = [float((pred[g == i] == y[g == i]).mean()) for i in range(4)]
    return {
        "overall": float((pred == y).mean()),
        "worst_group": float(min(per_group)),
        "per_group": per_group,
    }


def close(a, b):
    return (
        abs(a["overall"] - b["overall"]) < TOL
        and abs(a["worst_group"] - b["worst_group"]) < TOL
        and np.allclose(a["per_group"], b["per_group"], atol=TOL)
    )


def main():
    for path in [FEATURES, HEAD, ERM]:
        if not path.exists():
            print(f"MISSING: {path}")
            return 1

    data = np.load(FEATURES)
    X = torch.from_numpy(data["test_X"]).float()
    y = data["test_y"].astype(int)
    g = data["test_g"].astype(int)

    head = torch.load(HEAD, map_location="cpu", weights_only=True)
    z = ((X - head["scaler_mean"]) / head["scaler_scale"]) @ head["coef"] + head["intercept"]
    dfr_pred = (z > 0).numpy().astype(int)
    dfr = metrics(dfr_pred, y, g)

    state = torch.load(ERM, map_location="cpu", weights_only=True)
    erm_logits = X @ state["fc.weight"].float().T + state["fc.bias"].float()
    erm_pred = erm_logits.argmax(dim=1).numpy().astype(int)
    erm = metrics(erm_pred, y, g)

    print("DFR head + saved features")
    print(json.dumps(dfr, indent=2))
    print("\nERM head + saved features")
    print(json.dumps(erm, indent=2))

    dfr_ok = close(dfr, EXPECTED_DFR)
    erm_ok = close(erm, EXPECTED_ERM)

    print("\nDFR matches expected:", dfr_ok)
    print("ERM matches expected:", erm_ok)

    if not (dfr_ok and erm_ok):
        print("\nFAIL: artifacts do not reproduce the reported metrics.")
        return 1

    print("\nPASS: artifacts are correct. If the app disagrees, the bug is in the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
