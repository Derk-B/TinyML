from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_raw_to_clean_scale(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    led_indices: np.ndarray,
) -> np.ndarray:
    """Per-channel multiplicative gain mapping raw RSS onto the clean RSS scale.

    Each raw location is sampled multiple times with occasional dropout-style
    outlier readings, so the per-location median is matched against the single
    clean reading at that (x, y) rather than the raw mean, which the outliers
    would skew.
    """
    feature_columns = [f"led_{i}" for i in led_indices.tolist()]
    raw_median = raw_df.groupby(["x", "y"])[feature_columns].median().reset_index()
    merged = clean_df[["x", "y", *feature_columns]].merge(
        raw_median, on=["x", "y"], suffixes=("_clean", "_raw")
    )

    scale = np.ones(len(feature_columns), dtype=np.float32)
    for i, col in enumerate(feature_columns):
        raw_vals = merged[f"{col}_raw"].to_numpy(dtype=np.float64)
        clean_vals = merged[f"{col}_clean"].to_numpy(dtype=np.float64)
        mask = raw_vals > 1e-6
        if mask.any():
            scale[i] = float(np.median(clean_vals[mask] / raw_vals[mask]))
    return scale


def fit_clip_bounds(
    clean_df: pd.DataFrame,
    led_indices: np.ndarray,
    *,
    lo_percentile: float = 1.0,
    hi_percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel [lo, hi] clip bounds from the clean training distribution.

    Applied after the multiplicative gain, this reins in the calibration's
    overcorrection on noisy/near-dropout raw readings: the gain alone shifts
    the central tendency but leaves per-sample noise untouched, and clipping
    to the range the frozen model actually saw during training bounds the
    damage from outliers without needing multiple readings per query.
    """
    feature_columns = [f"led_{i}" for i in led_indices.tolist()]
    values = clean_df[feature_columns].to_numpy(dtype=np.float64)
    lo = np.percentile(values, lo_percentile, axis=0).astype(np.float32)
    hi = np.percentile(values, hi_percentile, axis=0).astype(np.float32)
    return lo, hi


def write_calibration_header(
    scale: np.ndarray,
    clip_lo: np.ndarray,
    clip_hi: np.ndarray,
    header_path: Path,
) -> None:
    def c_float(value: float) -> str:
        text = f"{float(value):.9g}"
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text + "f"

    def c_array(values: np.ndarray) -> str:
        return ", ".join(c_float(v) for v in values)

    header_path.write_text(
        "#pragma once\n\n"
        f"constexpr int kCalibrationChannels = {len(scale)};\n"
        f"constexpr float kCalibrationScale[{len(scale)}] = {{{c_array(scale)}}};\n"
        f"constexpr float kCalibrationClipLo[{len(clip_lo)}] = {{{c_array(clip_lo)}}};\n"
        f"constexpr float kCalibrationClipHi[{len(clip_hi)}] = {{{c_array(clip_hi)}}};\n",
        encoding="utf-8",
    )
