#!/usr/bin/env python3
"""
check_Lfmax_La_git.py

QC visualization for dispersion metrics (Lf paper + La unchanged):
  - Lf     : paper-consistent free-space length in matrix region (~particle_mask)
  - La     : maximum-inscribed-square length in agglomerate region (agglomerate_mask)

This script is designed to be reviewer-friendly:
- CLI (no manual editing of constants)
- Vector export (PDF/SVG)
- Works with the same sample list used by dispersion_Lfmax_La_git.py (or a user-supplied JSON)

Typical usage
-------------
# QC the first sample in the default sample list and export color:
python check_Lfmax_La_git.py --data-dir . --outdir outputs_qc --sample-index 0 --formats pdf,svg

# QC a specific system + wt%:
python check_Lfmax_La_git.py --data-dir . --outdir outputs_qc --system Unmodified --filler-wt 1.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage as ndimage


# -----------------------------
# Matplotlib style helpers
# -----------------------------
def set_journal_style() -> None:
    """Journal-friendly rcParams (vector export)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "lines.linewidth": 1.1,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })


def parse_formats(s: str) -> List[str]:
    fmts = [x.strip().lower().lstrip(".") for x in str(s).split(",") if x.strip()]
    return fmts if fmts else ["pdf", "svg"]


# -----------------------------
# Dynamic import of analysis module
# -----------------------------
def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("dispersion_module", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------
# Geometry: maximum inscribed square position (axis-aligned)
# -----------------------------
def find_max_square_position(mask: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """
    Find the top-left corner and size (pixels) of the maximum axis-aligned square fully contained in 'mask==True'.

    Uses chessboard distance transform:
      max_radius = max(dist_transform)
      square size = 2 * max_radius
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    # pad to reduce boundary artefacts
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    dist = ndimage.distance_transform_cdt(padded, metric="chessboard")

    max_radius = int(dist.max())
    if max_radius <= 0:
        return (0, 0), 0

    # pick a center point
    coords = np.argwhere(dist == max_radius)
    center_r, center_c = coords[len(coords) // 2]

    size = 2 * max_radius
    # Convert to top-left corner in original (unpadded) coordinates
    tl_r = (center_r - max_radius) - 1
    tl_c = (center_c - max_radius) - 1
    return (int(tl_r), int(tl_c)), int(size)


# -----------------------------
# For Lf (paper): find an example valid square window to overlay
# -----------------------------
def _integral_image(mask01: np.ndarray) -> np.ndarray:
    ii = mask01.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return ii

def _rect_sum(ii: np.ndarray, r0: int, c0: int, h: int, w: int) -> int:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    total = ii[r1, c1]
    if r0 > 0:
        total -= ii[r0 - 1, c1]
    if c0 > 0:
        total -= ii[r1, c0 - 1]
    if r0 > 0 and c0 > 0:
        total += ii[r0 - 1, c0 - 1]
    return int(total)

def find_example_good_square(allowed_mask: np.ndarray, size: int, n_tries: int = 30000, seed: int = 0):
    """Randomly find one size×size window fully inside allowed_mask (True=allowed)."""
    if size <= 0:
        return None, 0
    H, W = allowed_mask.shape
    if size > H or size > W:
        size = min(H, W)
    ii = _integral_image(allowed_mask)
    target = size * size
    rng = np.random.default_rng(seed)
    for _ in range(n_tries):
        r = int(rng.integers(0, H - size + 1))
        c = int(rng.integers(0, W - size + 1))
        if _rect_sum(ii, r, c, size, size) == target:
            return (r, c), size
    return None, size

def find_example_square_at_or_below(allowed_mask: np.ndarray, size: int, n_tries: int = 30000, seed: int = 0, max_shrink: int = 80):
    """Try size first; if not found, shrink gradually (up to max_shrink px)."""
    cur = int(size)
    for k in range(max_shrink + 1):
        tl, used = find_example_good_square(allowed_mask, cur, n_tries=n_tries, seed=seed + k)
        if tl is not None:
            return tl, used
        cur = max(1, cur - 1)
    return None, 0


# -----------------------------
# QC plotting
# -----------------------------
def _overlay_square(ax, tl: Tuple[int, int], size: int, edgecolor: str = "k") -> None:
    if size <= 0:
        return
    rect = patches.Rectangle((tl[1], tl[0]), size, size, fill=False, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(rect)


def qc_one_sample(
    module,
    data_dir: Path,
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Process all images in one sample and compute Lf,max and La (nm) + square positions (pixels).
    """
    results: List[Dict[str, Any]] = []
    for rel in sample["filenames"]:
        img_path = (data_dir / rel).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        gray, scale_nm_per_px, _ = module.load_and_preprocess_with_uniform_scale(img_path)
        p_mask = module.segment_particles(gray)
        agg_mask = module.build_agglomerate_mask(p_mask, scale_nm_per_px)

        # Lf (paper) on matrix region: random-window statistics
        seed = abs(hash(img_path.as_posix())) % (2 ** 32)
        lf_px = module.free_space_length_paper(~p_mask, n_windows=getattr(module, 'N_WINDOWS', 10000), rng_seed=seed, target=getattr(module, 'WINDOW_TARGET', 0.5))
        # Overlay: show one example window of that size (Lf has no unique location)
        tl, used = find_example_square_at_or_below(~p_mask, int(round(lf_px)), n_tries=50000, seed=seed)
        lf_tl = tl if tl is not None else (0, 0)
        lf_size = int(used) if used else 0
        # La on agglomerate region
        la_tl, la_size = find_max_square_position(agg_mask)

        results.append({
            "img_path": img_path,
            "gray": gray,
            "scale_nm_per_px": float(scale_nm_per_px),
            "p_mask": p_mask,
            "agg_mask": agg_mask,
            "lf_tl": lf_tl,
            "lf_size": lf_size,
            "la_tl": la_tl,
            "la_size": la_size,
            "Lf_nm": float(lf_px) * float(scale_nm_per_px),
            "Lf_shown_nm": float(lf_size) * float(scale_nm_per_px),
            "La_nm": float(la_size) * float(scale_nm_per_px),
        })
    return results


def plot_qc_grid(
    rows: List[Dict[str, Any]],
    outpath: Path,
    show_titles: bool = True,
) -> None:
    """
    Create a grid:
      col1: grayscale image
      col2: particle mask overlay + Lf,max square (matrix region)
      col3: agglomerate mask overlay + La square
    """
    set_journal_style()
    edge_lf = "#1f77b4"   # blue
    edge_la = "#ff7f0e"   # orange

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(6.8, 2.2 * n), constrained_layout=True)

    if n == 1:
        axes = np.array([axes])

    for i, r in enumerate(rows):
        gray = r["gray"]
        p_mask = r["p_mask"]
        agg_mask = r["agg_mask"]

        # 1) raw
        ax = axes[i, 0]
        ax.imshow(gray, cmap="gray")
        ax.set_axis_off()
        if show_titles:
            ax.set_title(f"Raw ({r['img_path'].name})", pad=3)

        # 2) particle overlay + Lf (paper)
        ax = axes[i, 1]
        ax.imshow(p_mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
        _overlay_square(ax, r["lf_tl"], r["lf_size"], edgecolor=edge_lf)
        ax.set_axis_off()
        if show_titles:
            ax.set_title(rf"$L_f$ = {r['Lf_nm']:.0f} nm" + ('' if (r.get('Lf_shown_nm', r['Lf_nm']) >= 0.98 * r['Lf_nm']) else rf" (shown: {r.get('Lf_shown_nm', 0):.0f} nm)"), pad=3)

        # 3) agglomerate overlay + La
        ax = axes[i, 2]
        ax.imshow(agg_mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
        _overlay_square(ax, r["la_tl"], r["la_size"], edgecolor=edge_la)
        ax.set_axis_off()
        if show_titles:
            ax.set_title(rf"$L_a$ = {r['La_nm']:.0f} nm", pad=3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def load_samples_from_json(json_path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("samples JSON must be a list of {system,filler_wt,filenames}")
    return obj


def pick_sample(samples: List[Dict[str, Any]], args) -> Dict[str, Any]:
    if args.system and args.filler_wt is not None:
        for s in samples:
            if str(s.get("system")) == str(args.system) and float(s.get("filler_wt")) == float(args.filler_wt):
                return s
        raise SystemExit(f"No sample found for system={args.system}, filler_wt={args.filler_wt}")
    # fallback to index
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise SystemExit(f"--sample-index out of range (0..{len(samples)-1})")
    return samples[int(args.sample_index)]


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="QC visualization for Lf,max and La (vector export).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", default=".", help="Root directory containing the image folders")
    ap.add_argument("--outdir", default="outputs_qc", help="Directory to write QC figures")
    ap.add_argument("--analysis-script", default="dispersion_Lf_La_git.py", help="Path to the analysis script to import functions from")
    ap.add_argument("--samples-file", default="", help="Optional JSON file defining samples (list of {system,filler_wt,filenames})")
    ap.add_argument("--sample-index", type=int, default=6, help="Sample index in the sample list")
    ap.add_argument("--system", default="", help="Sample selector: system name (optional)")
    ap.add_argument("--filler-wt", type=float, default=None, help="Sample selector: filler wt%% (optional)")
    ap.add_argument("--formats", default="pdf,svg", help="Comma-separated formats, e.g. pdf,svg")
    ap.add_argument("--no-titles", action="store_true", help="Disable per-panel titles (cleaner for supplement)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    analysis_script = Path(args.analysis_script)

    module = load_module_from_path(analysis_script)

    if args.samples_file:
        samples = load_samples_from_json(Path(args.samples_file))
    else:
        # fall back to sample list in analysis script
        samples = getattr(module, "SAMPLES", None)
        if samples is None:
            raise SystemExit("No SAMPLES found. Provide --samples-file JSON.")
    sample = pick_sample(samples, args)

    rows = qc_one_sample(module, data_dir, sample)

    formats = parse_formats(args.formats)

    # file stem (safe)
    stem = f"QC_{sample.get('system','sample')}_{sample.get('filler_wt','')}".replace(" ", "_").replace("%", "pct")
    for fmt in formats:
        outpath = outdir / f"{stem}_color.{fmt}"
        plot_qc_grid(rows, outpath, show_titles=not args.no_titles)

    print(f"Saved QC figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
