#!/usr/bin/env python3
"""Fit per-channel raw-to-clean RSS calibration gains and clip bounds for Task 2."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.calibration import (
    fit_clip_bounds,
    fit_raw_to_clean_scale,
    write_calibration_header,
)
from vlp_hackathon.dataset import CONF2_3X3_LED_INDICES


def main() -> None:
    raw_df = pd.read_csv(ROOT / "data" / "train_raw_3x3_1cm.csv")
    clean_df = pd.read_csv(ROOT / "data" / "train_clean_3x3_1cm.csv")

    scale = fit_raw_to_clean_scale(raw_df, clean_df, CONF2_3X3_LED_INDICES)
    clip_lo, clip_hi = fit_clip_bounds(clean_df, CONF2_3X3_LED_INDICES)

    out_npz = ROOT / "models" / "task2_raw_calibration.npz"
    out_npz.parent.mkdir(exist_ok=True)
    np.savez(
        out_npz,
        led_indices=CONF2_3X3_LED_INDICES,
        scale=scale,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )

    header_path = ROOT / "firmware" / "vlp_serial" / "calibration_data.h"
    write_calibration_header(scale, clip_lo, clip_hi, header_path)

    print("Per-channel raw->clean calibration scale and clip bounds:")
    for idx, s, lo, hi in zip(
        CONF2_3X3_LED_INDICES.tolist(), scale.tolist(), clip_lo.tolist(), clip_hi.tolist()
    ):
        print(f"  led_{idx}: scale={s:.4f} clip=[{lo:.4f}, {hi:.4f}]")
    print(f"Wrote {out_npz}")
    print(f"Wrote {header_path}")


if __name__ == "__main__":
    main()
