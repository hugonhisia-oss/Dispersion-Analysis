"""dispersion_Lfmax_La_git.py

Quantitative dispersion analysis for BNKT-ST / PVDF-HFP nanocomposites.

This script reports ONLY two metrics (vector-figure ready):
  - Lf     : paper-consistent free-space length (random-window statistics, e.g. Khare & Burris)
  - La     : deterministic maximum-inscribed-square agglomeration length

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage import io, color, filters, morphology, transform, measure
import matplotlib.pyplot as plt

__version__ = "1.1.0"

# --------------------------------------------------------------------
# Plot styling (journal-friendly, vector output)
# --------------------------------------------------------------------

def set_journal_style() -> None:
    """Matplotlib rcParams tuned for journal figures (vector export)."""
    plt.rcParams.update({
        # fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        # lines/markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        # axes
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        # vector text handling
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # figure
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _get_palette(style: str) -> Dict[str, str]:
    """Two-series palette. style='bw' (grayscale) or style='color' (colorblind-friendly)."""
    style = (style or "bw").lower()
    if style == "color":
        # Okabe-Ito-ish palette (colorblind-friendly)
        return {"Unmodified": "#0072B2", "Modified": "#D55E00"}
    return {"Unmodified": "0.15", "Modified": "0.55"}


def _plot_metric(
    df: pd.DataFrame,
    ycol: str,
    ylabel: str,
    title: str,
    outstem: str,
    yerrcol: str | None = None,
    outdir: Path | None = None,
    formats: List[str] | None = None,
    style: str = "bw",
    figsize: Tuple[float, float] = (3.35, 2.5),
) -> None:
    """Plot a single metric vs filler content for both systems and save as vector outputs."""
    systems = ["Unmodified", "Modified"]
    markers = {"Unmodified": "o", "Modified": "s"}
    linestyles = {"Unmodified": "-", "Modified": "--"}
    colors = _get_palette(style)

    if outdir is None:
        outdir = Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    if not formats:
        formats = ["pdf", "svg"]
    formats = [f.lower().lstrip('.') for f in formats]

    fig, ax = plt.subplots(figsize=figsize)  # ~single-column figure

    # --- make x positions equally spaced (categorical axis) ---
    x_labels = [0.5, 0.75, 1.0, 2.0, 3.0]
    x_pos = np.arange(len(x_labels))  # 0,1,2,3,4
    pos_map = {v: i for i, v in enumerate(x_labels)}

    for sys in systems:
        sub = df[df["system"] == sys]
        if sub.empty:
            continue
        x_raw = sub["filler_wt"].to_numpy()
        x = np.array([pos_map[float(v)] for v in x_raw])
        y = sub[ycol].to_numpy()
        yerr = None
        if yerrcol is not None and yerrcol in sub.columns:
            yerr = sub[yerrcol].to_numpy()

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=markers[sys],
            linestyle=linestyles[sys],
            color=colors[sys],
            label=sys,
            capsize=2.5,
            elinewidth=0.9,
        )

    ax.set_xlabel("Filler content (wt%)", fontname="Times New Roman", fontweight="bold")
    ax.set_ylabel(ylabel, fontname="Times New Roman", fontweight="bold")
    ax.set_title(title, fontname="Times New Roman", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["0.5", "0.75", "1", "2", "3"])
    ax.set_xlim(-0.4, len(x_labels) - 1 + 0.4)

    ax.minorticks_off()

    # keep ticks on left/bottom, remove ticks on top/right
    ax.tick_params(axis="both", which="major", direction="out",
                   bottom=True, left=True, top=False, right=False,
                   length=3.5, width=1.0)

    # (optional) keep minor ticks off (already), but if you later turn them on:
    ax.tick_params(axis="both", which="minor",
                   bottom=True, left=True, top=False, right=False,
                   length=2.0, width=0.8)

    ax.grid(False)
    leg = ax.legend(frameon=False, handlelength=2.0)
    for txt in leg.get_texts():
        txt.set_fontname("Times New Roman")
        txt.set_fontweight("bold")

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontname("Times New Roman")
        t.set_fontweight("bold")

    for fmt in formats:
        fig.savefig(outdir / f"{outstem}.{fmt}")
    plt.close(fig)

# --------------------------------------------------------------------
# File & sample information
# --------------------------------------------------------------------

DATA_DIR = Path(".")

SAMPLES: List[Dict] = [
    # Unmodified BNKT-ST
    dict(
        system="Unmodified",
        filler_wt=0.5,
        filenames=[
            "BNKT-ST/0.5wt%/1.tiff", "BNKT-ST/0.5wt%/2.tiff", "BNKT-ST/0.5wt%/3.tiff",
            "BNKT-ST/0.5wt%/4.tiff", "BNKT-ST/0.5wt%/5.tiff"
        ],
    ),
    dict(
        system="Unmodified",
        filler_wt=0.75,
        filenames=[
            "BNKT-ST/0.75wt%/1.tiff", "BNKT-ST/0.75wt%/2.tiff", "BNKT-ST/0.75wt%/3.tiff",
            "BNKT-ST/0.75wt%/4.tiff", "BNKT-ST/0.75wt%/5.tiff"
        ],
    ),
    dict(
        system="Unmodified",
        filler_wt=1.0,
        filenames=[
            "BNKT-ST/1wt%/1.tiff", "BNKT-ST/1wt%/2.tiff", "BNKT-ST/1wt%/3.tiff",
            "BNKT-ST/1wt%/4.tiff", "BNKT-ST/1wt%/5.tiff"
        ],
    ),
    dict(
        system="Unmodified",
        filler_wt=2.0,
        filenames=[
            "BNKT-ST/2wt%/1.tiff", "BNKT-ST/2wt%/2.tiff", "BNKT-ST/2wt%/3.tiff",
            "BNKT-ST/2wt%/4.tiff", "BNKT-ST/2wt%/5.tiff"
        ],
    ),
    dict(
        system="Unmodified",
        filler_wt=3.0,
        filenames=[
            "BNKT-ST/3wt%/1.tiff", "BNKT-ST/3wt%/2.tiff", "BNKT-ST/3wt%/3.tiff",
            "BNKT-ST/3wt%/4.tiff", "BNKT-ST/3wt%/5.tiff"
        ],
    ),

    # KH550-modified BNKT-ST
    dict(
        system="Modified",
        filler_wt=0.5,
        filenames=[
            "KH550-BNKT-ST/0.5wt%/1.tiff", "KH550-BNKT-ST/0.5wt%/2.tiff", "KH550-BNKT-ST/0.5wt%/3.tiff",
            "KH550-BNKT-ST/0.5wt%/4.tiff", "KH550-BNKT-ST/0.5wt%/5.tiff"
        ],
    ),
    dict(
        system="Modified",
        filler_wt=0.75,
        filenames=[
            "KH550-BNKT-ST/0.75wt%/1.tiff", "KH550-BNKT-ST/0.75wt%/2.tiff", "KH550-BNKT-ST/0.75wt%/3.tiff",
            "KH550-BNKT-ST/0.75wt%/4.tiff", "KH550-BNKT-ST/0.75wt%/5.tiff"
        ],
    ),
    dict(
        system="Modified",
        filler_wt=1.0,
        filenames=[
            "KH550-BNKT-ST/1wt%/1.tiff", "KH550-BNKT-ST/1wt%/2.tiff", "KH550-BNKT-ST/1wt%/3.tiff",
            "KH550-BNKT-ST/1wt%/4.tiff", "KH550-BNKT-ST/1wt%/5.tiff"
        ],
    ),
    dict(
        system="Modified",
        filler_wt=2.0,
        filenames=[
            "KH550-BNKT-ST/2wt%/1.tiff", "KH550-BNKT-ST/2wt%/2.tiff", "KH550-BNKT-ST/2wt%/3.tiff",
            "KH550-BNKT-ST/2wt%/4.tiff", "KH550-BNKT-ST/2wt%/5.tiff"
        ],
    ),
    dict(
        system="Modified",
        filler_wt=3.0,
        filenames=[
            "KH550-BNKT-ST/3wt%/1.tiff", "KH550-BNKT-ST/3wt%/2.tiff", "KH550-BNKT-ST/3wt%/3.tiff",
            "KH550-BNKT-ST/3wt%/4.tiff", "KH550-BNKT-ST/3wt%/5.tiff"
        ],
    ),
]

# --------------------------------------------------------------------
# Image pre-processing parameters
# --------------------------------------------------------------------

MAX_IMAGE_DIM = 2500          # keep original behavior (only downsamples if extremely large)
GAUSSIAN_SIGMA = 2.0           # smoothing after (optional) resize
CROP_BOTTOM_FRACTION = 0.15    # crop away scale bar region etc.

# Agglomerate definition parameters (paper-consistent)
# Physical meaning: if particle gap < (DCHAR_FRACTION_FOR_GAP * d_char), treat as agglomerate.
DCHAR_FRACTION_FOR_GAP = 3.0

# --- Paper-consistent random-window statistics for Lf ---
USE_RANDOM_WINDOWS = True
N_WINDOWS = 10000
WINDOW_TARGET = 0.5


# Shape filtering
MIN_PARTICLE_AREA = 30

# Unified scale (all images use 240 px for a 10 um bar)
ACTUAL_SCALE_BAR_LENGTH_NM = 10000.0
UNIFORM_SCALE_PIXELS = 240.0

# --------------------------------------------------------------------
# Core utility functions
# --------------------------------------------------------------------

def load_and_preprocess_with_uniform_scale(path: Path) -> Tuple[np.ndarray, float, str]:
    """Read TIFF, normalize, crop, optional downscale, smooth; return (gray, nm_per_pixel, 'uniform')."""
    img = io.imread(path.as_posix())

    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
            img = color.rgb2gray(img)
        elif img.shape[2] == 3:  # RGB
            img = color.rgb2gray(img)
        else:
            img = img[:, :, 0]

    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)

    # crop bottom strip (scale bar etc.)
    h, w = img.shape
    crop_h = int(h * (1.0 - CROP_BOTTOM_FRACTION))
    img = img[:crop_h, :]

    # resize to at most MAX_IMAGE_DIM
    h, w = img.shape
    scale_resize = min(1.0, MAX_IMAGE_DIM / max(h, w))
    if scale_resize < 1.0:
        img = transform.resize(
            img,
            (int(h * scale_resize), int(w * scale_resize)),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )

    img = filters.gaussian(img, sigma=GAUSSIAN_SIGMA)

    scale_nm_per_pixel_original = ACTUAL_SCALE_BAR_LENGTH_NM / UNIFORM_SCALE_PIXELS
    scale_nm_per_pixel = scale_nm_per_pixel_original / scale_resize

    print(f"\nFile: {path.name}")
    print(f"  Uniform scale: {scale_nm_per_pixel:.2f} nm/pixel")

    return img, scale_nm_per_pixel, "uniform"

# --------------------------------------------------------------------
# Image processing functions
# --------------------------------------------------------------------

def segment_particles(gray: np.ndarray) -> np.ndarray:
    # --- replacement start ---
    # Method: original image - background (large-scale Gaussian blur) ~= previous top-hat result.
    # sigma should be larger than the particle size, e.g., around 30–50.
    background = filters.gaussian(gray, sigma=30.0)
    tophat = gray - background
    # Note: subtraction may produce negative values; clamp to 0 (or renormalize). We use max(0) here.
    tophat = np.maximum(tophat, 0)
    # --- replacement end ---

    # The thresholding logic may need minor tuning because the value distribution changes,
    # but the adaptive rule mean + 2.5 * std often still works well.
    m_val = float(np.mean(tophat))
    s_val = float(np.std(tophat))
    thresh = m_val + 2.5 * s_val

    particles = tophat > thresh

    particles = morphology.remove_small_objects(particles, min_size=40)
    particles = morphology.binary_opening(particles, morphology.disk(1))
    particles = morphology.binary_dilation(particles, morphology.disk(1))

    return particles


def free_space_length(
    mask: np.ndarray,
    n_windows: int = 0,   # legacy, ignored (kept for API compatibility)
    rng_seed: int = 0,    # legacy, ignored (kept for API compatibility)
    target: float = 0.5,  # legacy, ignored (kept for API compatibility)
) -> float:
    """
    Return the edge length (pixels) of the maximum axis-aligned square fully contained in the True region.

    Implementation: pad a 1-px "wall" of False, then chessboard distance transform.
    """
    if not mask.any():
        return 0.0

    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    dist = ndimage.distance_transform_cdt(padded, metric="chessboard")
    max_radius = float(dist.max())

    # engineering convention used in the original script
    max_L = 2.0 * max_radius
    if max_L == 0.0 and mask.any():
        max_L = 1.0

    return float(max_L)


# --------------------------------------------------------------------
# Paper-consistent free-space length Lf (random-window statistics)
# --------------------------------------------------------------------
def _integral_image(mask01: np.ndarray) -> np.ndarray:
    """Integral image for fast rectangle sums."""
    return mask01.astype(np.uint8).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ii: np.ndarray, r0: int, c0: int, h: int, w: int) -> int:
    """Sum over [r0:r0+h, c0:c0+w) using integral image."""
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    total = int(ii[r1, c1])
    if r0 > 0:
        total -= int(ii[r0 - 1, c1])
    if c0 > 0:
        total -= int(ii[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ii[r0 - 1, c0 - 1])
    return int(total)


def free_space_length_paper(
    mask: np.ndarray,
    n_windows: int = N_WINDOWS,
    rng_seed: int = 0,
    target: float = WINDOW_TARGET,
) -> float:
    """
    Paper-consistent free-space length Lf (pixels) using random-window statistics.

    Interpretation:
      Lf = max L such that P(window of size LxL lies fully in allowed region) >= target.

    - mask=True indicates ALLOWED region (Lf uses ~particle_mask).
    - Uses binary search over L and an integral-image test for window validity.

    Notes:
    - Deterministic given rng_seed.
    - If n_windows<=0, falls back to deterministic maximum-inscribed-square length (distance transform).
    """
    if not mask.any():
        return 0.0

    h, w = mask.shape
    maxL = int(min(h, w))

    # fallback (keeps behavior for debugging / speed)
    if n_windows is None or int(n_windows) <= 0:
        return float(free_space_length(mask))

    rng_seed = int(rng_seed) & 0xFFFFFFFF
    n_windows = int(n_windows)
    target = float(target)

    # bad pixels = disallowed
    bad = (~mask).astype(np.uint8)
    # summed-area table with 1px pad (like dispersion.py)
    sat = np.pad(bad, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    def p_good(L: int) -> float:
        if L <= 0 or L > h or L > w:
            return 0.0
        # size-dependent RNG so different L have stable sampling patterns
        rngL = np.random.default_rng((rng_seed + 1315423911 * int(L)) & 0xFFFFFFFF)
        ys = rngL.integers(0, h - L + 1, size=n_windows, endpoint=False)
        xs = rngL.integers(0, w - L + 1, size=n_windows, endpoint=False)

        y0, x0 = ys, xs
        y1, x1 = ys + L, xs + L
        s = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
        return float(np.mean(s == 0))

    lo, hi, best = 1, maxL, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if p_good(mid) >= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return float(max(best, 1))

def characteristic_diameter_px(particle_mask: np.ndarray) -> float:
    """Median equivalent-circle diameter of connected components (pixels)."""
    lbl = measure.label(particle_mask, connectivity=2)
    props = measure.regionprops(lbl)

    diameters: List[float] = []
    for p in props:
        if p.area < 3:
            continue
        d_eq = float(np.sqrt(4.0 * p.area / np.pi))
        diameters.append(d_eq)

    return float(np.median(diameters)) if diameters else 5.0

AGGLOM_SMOOTH_RADIUS = 1   # px: smooth agglomerate boundary; set to 0 to disable
def build_agglomerate_mask(particle_mask: np.ndarray, scale_nm_per_pixel: float) -> np.ndarray:
    """
    Build agglomerate mask via morphological closing based on characteristic particle diameter.
    NOTE: scale_nm_per_pixel is currently unused (kept for compatibility with external scripts).
    """
    _ = scale_nm_per_pixel  # keep signature stable

    d_char = characteristic_diameter_px(particle_mask)
    gap_thresh = max(1.0, DCHAR_FRACTION_FOR_GAP * d_char)

    # Closing radius: keep at least 1 to avoid the degenerate disk(0) case.
    closing_radius = max(1, int(round(gap_thresh / 2.0)))

    # Key: use an isotropic disk structuring element (not a rectangle) to obtain smoother boundaries.
    selem = morphology.disk(closing_radius)
    agglom = morphology.binary_closing(particle_mask, selem)

    # light smoothing to remove burrs and sharp corners (without significantly changing agglomerate scale).
    if AGGLOM_SMOOTH_RADIUS and AGGLOM_SMOOTH_RADIUS > 0:
        r = int(AGGLOM_SMOOTH_RADIUS)
        agglom = morphology.binary_opening(agglom, morphology.disk(r))
        agglom = morphology.binary_closing(agglom, morphology.disk(r))

    agglom = morphology.remove_small_objects(agglom, min_size=MIN_PARTICLE_AREA)
    agglom = morphology.binary_dilation(agglom, morphology.disk(1))
    return agglom



def compute_Lf_La(particle_mask: np.ndarray, agglom_mask: np.ndarray, *, n_windows: int = N_WINDOWS, rng_seed: int = 0, target: float = WINDOW_TARGET) -> Tuple[float, float]:
    """Compute Lf (paper, random-window) and La (deterministic max-inscribed-square), both in pixels."""
    lf = free_space_length_paper(~particle_mask, n_windows=n_windows, rng_seed=rng_seed, target=target)  # matrix region
    la = free_space_length(agglom_mask)  # keep La definition unchanged (max square in agglomerate region)
    return float(lf), float(la)


# Backward-compatible alias (old name in v1.1.0)
def compute_Lfmax_La(particle_mask: np.ndarray, agglom_mask: np.ndarray) -> Tuple[float, float]:
    return compute_Lf_La(particle_mask, agglom_mask, n_windows=N_WINDOWS, rng_seed=0, target=WINDOW_TARGET)


def _qc_stats(
    particle_mask: np.ndarray,
    agglom_mask: np.ndarray,
    scale_nm_per_pixel: float,
    lf_px: float | None = None,
    la_px: float | None = None,
) -> Dict[str, float]:
    """
    QC outputs (kept keys for backward compatibility).
    If lf_px / la_px are provided, reuse them to avoid recomputation.
    """
    if lf_px is None:
        lf_px = free_space_length(~particle_mask)   # consistent with main definition: max square in matrix (free space)
    if la_px is None:
        la_px = free_space_length(agglom_mask)

    lf_nm = float(lf_px) * scale_nm_per_pixel
    la_nm = float(la_px) * scale_nm_per_pixel

    return {
        "Lf_nm_mean": float(lf_nm),
        "Lf_nm_std": 0.0,
        "La_nm_mean": float(la_nm),
        "La_nm_std": 0.0,
        "particle_area_fraction": float(particle_mask.mean()),
        "agglom_area_fraction": float(agglom_mask.mean()),
    }


# --------------------------------------------------------------------
# Main analysis function
# --------------------------------------------------------------------

def _load_samples(samples_json: Path | None) -> List[Dict]:
    """Load sample definitions; default uses the in-script SAMPLES."""
    if samples_json is None:
        return SAMPLES
    data = json.loads(samples_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("samples-json must be a JSON list of sample dicts")
    for s in data:
        if not all(k in s for k in ("system", "filler_wt", "filenames")):
            raise ValueError("Each sample must contain: system, filler_wt, filenames")
    return data


def _parse_formats(s: str) -> List[str]:
    fmts = []
    for x in (s or "").split(","):
        x = x.strip().lower().lstrip(".")
        if x:
            fmts.append(x)
    return fmts or ["pdf", "svg"]


def main(argv: List[str] | None = None) -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(
        description="Compute dispersion metrics Lf (paper) and La and export journal-style vector figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", default=".", help="Root directory containing the image folders")
    ap.add_argument("--outdir", default="outputs", help="Directory to write CSV and figures")
    ap.add_argument("--samples-json", default="", help="Optional JSON overriding the in-script SAMPLES")
    ap.add_argument("--styles", default="bw,color", help="Comma-separated styles to export: bw,color. Default exports bw main + color backup")
    ap.add_argument("--style", choices=["bw", "color"], default=None, help="(Deprecated) Export a single style; overrides --styles")
    ap.add_argument("--formats", default="pdf,svg", help="Comma-separated formats, e.g. pdf,svg")
    ap.add_argument("--figsize", type=float, nargs=2, default=(3.35, 2.5), metavar=("W", "H"), help="Figure size in inches")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    ap.add_argument("--version", action="store_true", help="Print version and exit")
    args = ap.parse_args(argv)

    if args.version:
        print(__version__)
        return

    data_dir = Path(args.data_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    samples_json = Path(args.samples_json).expanduser().resolve() if args.samples_json else None
    formats = _parse_formats(args.formats)

    # Parse styles (bw main + optional color backup)
    if args.style is not None:
        styles = [args.style]
    else:
        styles = [s.strip() for s in str(args.styles).split(",") if s.strip()]
    valid_styles = {"bw", "color"}
    bad = [s for s in styles if s not in valid_styles]
    if bad:
        raise SystemExit(f"Unknown style(s): {bad}. Choose from {sorted(valid_styles)}")

    samples = _load_samples(samples_json)

    print("=" * 80)
    print("QUANTITATIVE DISPERSION ANALYSIS")
    print("=" * 80)

    records: List[Dict] = []

    for s in samples:
        system = s["system"]
        wt = s["filler_wt"]
        file_list = s["filenames"]

        Lf_nm_list: List[float] = []
        La_nm_list: List[float] = []
        scale_list: List[float] = []

        for fname in file_list:
            img_path = data_dir / fname
            if not img_path.exists():
                if not args.quiet:
                    print(f"Warning: Image not found {img_path}, skipping.")
                continue

            try:
                gray, scale_nm_per_pixel, _method = load_and_preprocess_with_uniform_scale(img_path)

                particle_mask = segment_particles(gray)
                agglom_mask = build_agglomerate_mask(particle_mask, scale_nm_per_pixel)

                seed = abs(hash(img_path.as_posix())) % (2 ** 32)
                Lf_pixel, La_pixel = compute_Lf_La(particle_mask, agglom_mask, n_windows=N_WINDOWS if USE_RANDOM_WINDOWS else 0, rng_seed=seed, target=WINDOW_TARGET)

                Lf_nm = Lf_pixel * scale_nm_per_pixel
                La_nm = La_pixel * scale_nm_per_pixel

                Lf_nm_list.append(Lf_nm)
                La_nm_list.append(La_nm)
                scale_list.append(scale_nm_per_pixel)

                if not args.quiet:
                    print(f"  {system:>10s} {wt:>4} wt% | {img_path.name:>12s} | Lf={Lf_nm:8.1f} nm | La={La_nm:8.1f} nm")
            except Exception as e:
                if not args.quiet:
                    print(f"Error processing {fname}: {e}")

        if Lf_nm_list:
            rec = {
                "system": system,
                "filler_wt": wt,
                "n_images": len(Lf_nm_list),
                "avg_scale_nm_per_pixel": float(np.mean(scale_list)),
                "Lf_nm": float(np.mean(Lf_nm_list)),
                "Lf_nm_std": float(np.std(Lf_nm_list, ddof=1)) if len(Lf_nm_list) > 1 else 0.0,
                "La_nm": float(np.mean(La_nm_list)),
                "La_nm_std": float(np.std(La_nm_list, ddof=1)) if len(La_nm_list) > 1 else 0.0,
            }
            records.append(rec)

    if not records:
        print("No records processed.")
        return

    df = pd.DataFrame.from_records(records).sort_values(["system", "filler_wt"]).reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "dispersion_results.csv", index=False)

    
    set_journal_style()

    for style in styles:
        suffix = "" if style == "bw" else "_color"

        _plot_metric(
            df,
            ycol="Lf_nm",
            yerrcol="Lf_nm_std",
            ylabel=r"$L_f$ (nm)",
            title=r"Maximum free-space length $L_f$",
            outstem=f"Fig_Lf{suffix}",
            outdir=outdir,
            formats=formats,
            style=style,
            figsize=(float(args.figsize[0]), float(args.figsize[1])),
        )

        _plot_metric(
            df,
            ycol="La_nm",
            yerrcol="La_nm_std",
            ylabel=r"$L_a$ (nm)",
            title=r"Agglomeration length $L_a$",
            outstem=f"Fig_La{suffix}",
            outdir=outdir,
            formats=formats,
            style=style,
            figsize=(float(args.figsize[0]), float(args.figsize[1])),
        )
    if not args.quiet:
        print("\nSaved:")
        print(f"  - {outdir / 'dispersion_results.csv'}")
        for fmt in formats:
            print(f"  - {outdir / ('Fig_Lf.' + fmt)}")
            print(f"  - {outdir / ('Fig_La.' + fmt)}")


if __name__ == "__main__":
    main(sys.argv[1:])
