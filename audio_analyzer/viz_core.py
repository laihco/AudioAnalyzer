# audio_analyzer/viz_core.py
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from .analysis_core import normalize_01, zscore
except ImportError:
    from analysis_core import normalize_01, zscore



PALETTE_PRESETS = {
    # warm → punchy → golden → dusk purple
    "sunset": ["#ffcdb2", "#ffb4a2", "#e5989b", "#b5838d", "#6d6875"],

    # sky → bright cyan → soft cloud → sun highlight
    "daytime": ["#ccd5ae", "#e9edc9", "#fefae0", "#faedcd", "#d4a373"],

    # deep → indigo → violet → neon accent (clubby night)
    "nighttime": ["#3a015c", "#4f0147", "#35012c", "#290025", "#11001c"],

    # dawn pink → peach → early sky → soft gold
    "sunrise": ["#cdb4db", "#ffc8dd", "#ffafcc", "#bde0fe", "#a2d2ff"],
}


def get_palette_by_name(name: str) -> List[str]:
    if not name:
        raise ValueError("palette name is required")
    key = name.strip().lower()
    if key == "night":
        key = "nighttime"
    if key not in PALETTE_PRESETS:
        raise ValueError(f"Unknown palette '{name}'. Choose from: {', '.join(PALETTE_PRESETS.keys())}")
    return PALETTE_PRESETS[key]


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Color must be #RRGGBB")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)


def palette_hex_to_rgb01(palette_hex: List[str]) -> np.ndarray:
    if len(palette_hex) != 5:
        raise ValueError("Palette must have exactly 5 colors (top -> bottom).")
    return np.stack([hex_to_rgb01(h) for h in palette_hex], axis=0).astype(np.float32)


def rgb01_to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_rgb_image(out_path: Path, img: np.ndarray) -> None:
    """Save float RGB image (H,W,3) in 0..1 to PNG."""
    plt.figure(figsize=(img.shape[1] / 160, img.shape[0] / 160), dpi=160)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def compute_palette_stop_positions(
    energy: float,
    *,
    shift_range: float = 0.18,
    min_gap: float = 0.04,
) -> np.ndarray:
    e = float(np.clip(energy, 0.0, 1.0))
    base = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    shift = (e - 0.5) * 2.0 * float(shift_range)

    pos = base.copy()
    pos[1:4] = np.clip(base[1:4] + shift, 0.0, 1.0)

    pos[0] = 0.0
    for i in range(1, 5):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    pos[4] = 1.0

    return np.clip(pos, 0.0, 1.0).astype(np.float32)


def render_palette_gradient_frame(
    width: int,
    height: int,
    palette_rgb: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    palette_rgb = np.asarray(palette_rgb, dtype=np.float32)
    stops = np.asarray(stops, dtype=np.float32)

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)

    col = np.zeros((height, 3), dtype=np.float32)

    for seg in range(4):
        y0, y1 = float(stops[seg]), float(stops[seg + 1])
        c0, c1 = palette_rgb[seg], palette_rgb[seg + 1]

        m = (y >= y0) & (y <= y1)
        if not np.any(m):
            continue

        t = (y[m] - y0) / max(1e-9, (y1 - y0))
        col[m, :] = (c0[None, :] * (1.0 - t[:, None])) + (c1[None, :] * t[:, None])

    col[y < stops[0], :] = palette_rgb[0][None, :]
    col[y > stops[-1], :] = palette_rgb[-1][None, :]

    img = np.tile(col[:, None, :], (1, width, 1))
    return np.clip(img, 0.0, 1.0)


def render_palette_gradient_timeline(
    energy: np.ndarray,
    palette_rgb: np.ndarray,
    *,
    width: int = 1070,
    height: int = 280,
    blur_time: int = 9,
    shift_range: float = 0.18,
) -> np.ndarray:

    energy = np.asarray(energy, dtype=np.float32)
    T = energy.shape[0]

    # resample energy to match width columns
    x_src = np.linspace(0, T - 1, num=T, dtype=np.float32)
    x_dst = np.linspace(0, T - 1, num=width, dtype=np.float32)
    e_w = np.interp(x_dst, x_src, energy).astype(np.float32)

    # optional temporal smoothing to reduce barcode feel
    if blur_time and blur_time > 1:
        k = int(blur_time)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=np.float32) / float(k)
        e_w = np.convolve(e_w, kernel, mode="same")

    img = np.zeros((height, width, 3), dtype=np.float32)

    # Build each column
    # (This is fast enough for typical widths; video frames are rendered separately.)
    for x in range(width):
        stops = compute_palette_stop_positions(float(e_w[x]), shift_range=shift_range)
        col = render_palette_gradient_frame(1, height, palette_rgb, stops)[:, 0, :]
        img[:, x, :] = col

    return np.clip(img, 0.0, 1.0)
