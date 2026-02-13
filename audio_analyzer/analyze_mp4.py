from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from PIL import Image

try:
    # Package mode: python -m audio_analyzer.analyze_mp4
    from .analysis_core import AnalysisParams, safe_mkdir
    from .viz_core import (
        get_palette_by_name,
        palette_hex_to_rgb01,
        render_palette_gradient_frame,
        rgb01_to_uint8,
    )
except Exception:
    # Script mode: python analyze_mp4.py
    from analysis_core import AnalysisParams, safe_mkdir  # type: ignore
    from viz_core import (  # type: ignore
        get_palette_by_name,
        palette_hex_to_rgb01,
        render_palette_gradient_frame,
        rgb01_to_uint8,
    )


# ----------------------------
# Helpers
# ----------------------------
def _smooth_over_time(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), kernel, mode="same").astype(np.float32)


def _ffmpeg_export_video(
    frames_dir: Path,
    fps: float,
    audio_input: Path,
    out_mp4: Path,
    ffmpeg_exe: str,
) -> None:
    cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-i", str(audio_input),
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def _compute_waveform_amplitude_per_frame(
    y: np.ndarray,
    sr: int,
    hop_sec: float,
    amp_percentile: float,
) -> np.ndarray:
    hop = max(1, int(round(sr * hop_sec)))
    n_frames = int(np.ceil(len(y) / float(hop)))

    amps = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        e = min(len(y), s + hop)
        seg = y[s:e]
        amps[i] = 0.0 if seg.size == 0 else float(np.percentile(np.abs(seg), amp_percentile))
    return amps


def _normalize_01_by_percentiles(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    denom = max(1e-9, hi - lo)
    return np.clip((x - lo) / denom, 0.0, 1.0).astype(np.float32)


def compute_palette_stop_positions_waveform(
    a01: float,
    shift_range: float,
    min_gap: float,
) -> np.ndarray:
    a01 = float(np.clip(a01, 0.0, 1.0))
    base = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    # S-curve: gives more motion in the middle range, less "stuck"
    x = (a01 - 0.5) * 2.0
    x = np.tanh(1.6 * x)
    shift = x * float(shift_range)

    # Weighted stops reduces the "single dividing line" look
    weights = np.array([0.0, 0.65, 1.0, 0.65, 0.0], dtype=np.float32)
    pos = np.clip(base + shift * weights, 0.0, 1.0)
    pos[0] = 0.0
    pos[4] = 1.0

    # Enforce spacing (forward + backward)
    for i in range(1, 5):
        pos[i] = max(float(pos[i]), float(pos[i - 1] + min_gap))
    for i in range(3, -1, -1):
        pos[i] = min(float(pos[i]), float(pos[i + 1] - min_gap))

    pos[0] = 0.0
    pos[4] = 1.0
    return np.clip(pos, 0.0, 1.0).astype(np.float32)


def _resolve_ffmpeg(user_path: Optional[str]) -> str:
    if user_path:
        p = Path(user_path).expanduser()
        if p.exists():
            return str(p)
        raise SystemExit(f"ffmpeg not found at: {p}")
    which = shutil.which("ffmpeg")
    if which:
        return which
    raise SystemExit(
        "ffmpeg not found. Install it and ensure 'ffmpeg' is on PATH, "
        "or pass --ffmpeg /path/to/ffmpeg."
    )


def _load_cloud(cloud_path: Path, width: int, alpha: float, y_offset: int) -> Image.Image:
    img = Image.open(str(cloud_path)).convert("RGBA")
    w, h = img.size
    new_w = int(width)
    new_h = int(round(h * (new_w / float(w))))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    r, g, b, a = img.split()
    a = a.point(lambda p: int(p * float(np.clip(alpha, 0.0, 1.0))))
    img = Image.merge("RGBA", (r, g, b, a))

    # store offset in info for convenience (optional)
    img.info["y_offset"] = int(y_offset)
    return img


# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audio-driven gradient generator (outputs an MP4).",
    )
    p.add_argument("--input", required=True, type=str, help="Path to input audio (.mp3/.wav).")
    p.add_argument("--out", default="outputs", type=str, help="Output directory.")
    p.add_argument("--name", default="final_gradient.mp4", type=str, help="Output MP4 filename.")

    p.add_argument("--palette", default="nighttime", type=str,
                   help="Palette preset: sunset | daytime | nighttime | sunrise")
    p.add_argument("--width", default=1070, type=int, help="Video width.")
    p.add_argument("--height", default=280, type=int, help="Video height.")

    # Core midterm parameters
    p.add_argument("--reactivity", default=3.0, type=float,
                   help="0.0–5.0. Higher = bigger motion to music.")
    p.add_argument("--smoothness", default=0.6, type=float,
                   help="0.0–1.0. Higher = smoother/flowier motion.")

    # Analysis / sampling
    p.add_argument("--sr", default=22050, type=int, help="Audio sample rate.")
    p.add_argument("--hop-sec", default=0.10, type=float,
                   help="Seconds per frame. Lower = more responsive but slower to render.")

    # Robust amplitude measurement & normalization
    p.add_argument("--amp-percentile", default=85.0, type=float,
                   help="Percentile(|samples|) per hop window. 80–95 is typical.")
    p.add_argument("--norm-lo", default=5.0, type=float, help="Global percentile low for normalization.")
    p.add_argument("--norm-hi", default=80.0, type=float, help="Global percentile high for normalization.")

    # Stop behavior
    p.add_argument("--stop-shift", default=0.18, type=float, help="Max stop shift range.")
    p.add_argument("--min-gap", default=0.04, type=float, help="Minimum gap between stops.")

    # Optional cloud overlay
    p.add_argument("--cloud", default="", type=str, help="Optional path to cloud PNG overlay.")
    p.add_argument("--cloud-alpha", default=0.35, type=float, help="Cloud opacity 0..1.")
    p.add_argument("--cloud-y", default=-10, type=int, help="Cloud Y offset.")
    p.add_argument("--cloud-x", default=0, type=int, help="Cloud X offset.")

    # ffmpeg
    p.add_argument("--ffmpeg", default="", type=str, help="Optional path to ffmpeg executable.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_dir = Path(args.out).expanduser().resolve()
    safe_mkdir(out_dir)

    ffmpeg_exe = _resolve_ffmpeg(args.ffmpeg)

    # Map smoothness -> temporal smoothing window.
    # smoothness=0 -> 1 frame (no smoothing); smoothness=1 -> ~11 frames
    smoothness = float(np.clip(args.smoothness, 0.0, 1.0))
    smooth_k = int(round(1 + smoothness * 10))
    if smooth_k % 2 == 0:
        smooth_k += 1

    # Map reactivity -> gain + stop shift boost
    reactivity = float(np.clip(args.reactivity, 0.0, 5.0))
    r = (reactivity / 5.0) ** 0.7
    gain = 1.0 + 5.0 * r          # 1.0..3.8
    stop_shift = float(args.stop_shift) * (0.9 + 2.6 * r)  # ~0.14..0.29

    # --- load audio ---
    y, sr = librosa.load(str(in_path), sr=int(args.sr), mono=True)

    params = AnalysisParams(sr=int(args.sr), hop_sec=float(args.hop_sec))

    amps = _compute_waveform_amplitude_per_frame(
        y=y,
        sr=sr,
        hop_sec=float(params.hop_sec),
        amp_percentile=float(args.amp_percentile),
    )

    a01 = _normalize_01_by_percentiles(amps, p_lo=float(args.norm_lo), p_hi=float(args.norm_hi))
    a01 = _smooth_over_time(a01, smooth_k)

    # centered gain (reactivity)
    a01 = np.clip((a01 - 0.5) * gain + 0.5, 0.0, 1.0).astype(np.float32)

    # --- palette ---
    palette_hex = get_palette_by_name(args.palette)
    palette_rgb = palette_hex_to_rgb01(palette_hex)

    # --- optional cloud ---
    cloud_img = None
    cloud_path = args.cloud.strip()
    if cloud_path:
        cloud_img = _load_cloud(
            cloud_path=Path(cloud_path).expanduser().resolve(),
            width=int(args.width),
            alpha=float(args.cloud_alpha),
            y_offset=int(args.cloud_y),
        )

    fps = 1.0 / max(1e-9, float(params.hop_sec))

    # --- render frames (temp) + mux to final mp4 ---
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        n_frames = int(a01.shape[0])

        for i in range(n_frames):
            stops = compute_palette_stop_positions_waveform(
                float(a01[i]),
                shift_range=stop_shift,
                min_gap=float(args.min_gap),
            )
            frame = render_palette_gradient_frame(
                int(args.width),
                int(args.height),
                palette_rgb,
                stops,
            )
            rgb_u8 = rgb01_to_uint8(frame)

            if cloud_img is not None:
                bg = Image.fromarray(rgb_u8, mode="RGB").convert("RGBA")
                bg.paste(cloud_img, (int(args.cloud_x), int(args.cloud_y)), cloud_img)
                rgb_u8 = np.array(bg.convert("RGB"), dtype=np.uint8)

            Image.fromarray(rgb_u8).save(frames_dir / f"frame_{i:06d}.png")

        out_mp4 = out_dir / args.name
        _ffmpeg_export_video(
            frames_dir=frames_dir,
            fps=float(fps),
            audio_input=in_path,
            out_mp4=out_mp4,
            ffmpeg_exe=ffmpeg_exe,
        )

    print(f"✅ Wrote: {out_mp4}")


if __name__ == "__main__":
    main()
