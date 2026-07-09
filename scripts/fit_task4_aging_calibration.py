#!/usr/bin/env python3
"""Fit the Task 4 online per-channel aging-AGC reference and hyperparameters."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.aging_calibration import fit_reference_channel_max, write_aging_header
from vlp_hackathon.dataset import CONF2_3X3_LED_INDICES

LEAK = 0.99993
GAIN_CLIP = (1.0, 6.0)


def main() -> None:
    clean_df = pd.read_csv(ROOT / "data" / "train_clean_3x3_1cm.csv")
    ref_max = fit_reference_channel_max(clean_df, CONF2_3X3_LED_INDICES)

    out_npz = ROOT / "models" / "task4_aging_calibration.npz"
    out_npz.parent.mkdir(exist_ok=True)
    np.savez(
        out_npz,
        led_indices=CONF2_3X3_LED_INDICES,
        ref_channel_max=ref_max,
        leak=np.float32(LEAK),
        gain_min=np.float32(GAIN_CLIP[0]),
        gain_max=np.float32(GAIN_CLIP[1]),
    )

    header_path = ROOT / "firmware" / "vlp_serial" / "aging_data.h"
    write_aging_header(ref_max, LEAK, GAIN_CLIP, header_path)

    print("Per-channel reference max (fresh installation):")
    for idx, v in zip(CONF2_3X3_LED_INDICES.tolist(), ref_max.tolist()):
        print(f"  led_{idx}: {v:.4f}")
    print(f"leak={LEAK} gain_clip={GAIN_CLIP}")
    print(f"Wrote {out_npz}")
    print(f"Wrote {header_path}")


if __name__ == "__main__":
    main()
