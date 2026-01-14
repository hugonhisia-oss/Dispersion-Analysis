# Dispersion metrics (Lf and La) — code for peer review

This repository computes **two dispersion metrics** from SEM images (TIFF), then exports **journal-ready vector figures**.

- **Lf** (*free-space length, paper-consistent*): computed on the **matrix / free-space region** `~particle_mask` using **random-window statistics**  
  \(L_f\) is defined as the **largest square edge length** \(L\) such that  
  \(P(\text{a random } L\times L \text{ window lies fully in free space}) \ge \texttt{WINDOW_TARGET}\).  
  Implementation: integral-image window test + binary search (deterministic given a seed).

- **La** (*agglomeration length*): computed on the **agglomerate region** `agglomerate_mask` as the **maximum inscribed axis-aligned square** (chessboard distance transform; deterministic).

The main analysis is in `dispersion_Lf_La_git.py`, and the optional QC visualizer is `check_Lf_La_git.py`.

---

## Quick start

### 1) Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2) Run the analysis (CSV + figures)
```bash
python dispersion_Lf_La_git.py --data-dir . --outdir outputs --formats pdf,svg
```

**Outputs (default):**
- `outputs/dispersion_results.csv`
- `outputs/Fig_Lf.pdf` and `outputs/Fig_Lf.svg` (main, **grayscale**)
- `outputs/Fig_La.pdf` and `outputs/Fig_La.svg` (main, **grayscale**)
- `outputs/Fig_Lf_color.pdf/svg` and `outputs/Fig_La_color.pdf/svg` (backup, **color**)

### 3) QC visualization (optional, reviewer-friendly)
```bash
python check_Lf_La_git.py --data-dir . --outdir outputs_qc --sample-index 0 --formats pdf,svg
```

This exports a multi-row grid per sample:
1. raw grayscale image  
2. particle mask + an example square of size \(L_f\) (Lf has **no unique location**, so we show a valid example window)  
3. agglomerate mask + the **maximum** inscribed square for \(L_a\)

---

## Data layout

The scripts expect relative paths listed in `SAMPLES` inside `dispersion_Lf_La_git.py`, for example:
```
KH550-BNKT-ST/1wt%/1.tiff
```

### Overriding the sample list (recommended for sharing)

Create a JSON file like:

```json
[
  {
    "system": "Unmodified",
    "filler_wt": 1.0,
    "filenames": [
      "BNKT-ST/1wt%/1.tiff",
      "BNKT-ST/1wt%/2.tiff",
      "BNKT-ST/1wt%/3.tiff",
      "BNKT-ST/1wt%/4.tiff",
      "BNKT-ST/1wt%/5.tiff"
    ]
  }
]
```

Then run:
```bash
python dispersion_Lf_La_git.py --data-dir . --outdir outputs --samples-json path/to/samples.json
```

For QC with the same JSON:
```bash
python check_Lf_La_git.py --data-dir . --outdir outputs_qc --samples-file path/to/samples.json --sample-index 0
```

---

## Figure export options

By default the analysis exports:
- **bw** figures as main submission figures (`Fig_Lf.*`, `Fig_La.*`)
- **color** figures as backups (`*_color.*`)

You can control this with:
```bash
python dispersion_Lf_La_git.py --styles bw
python dispersion_Lf_La_git.py --styles color
python dispersion_Lf_La_git.py --styles bw,color
```

Formats are controlled by:
```bash
python dispersion_Lf_La_git.py --formats pdf,svg
```

---

## Method details (matches the implementation)

### Image preprocessing (uniform scale)
- Crops the bottom strip (`CROP_BOTTOM_FRACTION`, default 0.15) to remove scale bar region.
- Optionally downsamples very large images (`MAX_IMAGE_DIM`, default 2500 px).
- Gaussian smoothing (`GAUSSIAN_SIGMA`, default 2.0).
- Uses a **uniform physical scale** assuming the scale bar is **10 µm = 240 px**  
  (`ACTUAL_SCALE_BAR_LENGTH_NM = 10000`, `UNIFORM_SCALE_PIXELS = 240`).

### Particle segmentation (binary `particle_mask`)
- Background removal using a large Gaussian blur (`sigma=30`) and subtraction (top-hat–like).
- Threshold = mean + 2.5×std on the background-removed image.
- Morphology cleanup (remove small objects, opening, dilation).

### Agglomerate mask (`agglomerate_mask`)
- Computes a characteristic particle diameter (median equivalent-circle diameter).
- Treats gaps smaller than `DCHAR_FRACTION_FOR_GAP × d_char` as connected (via morphological closing).
- Uses an isotropic disk structuring element and light boundary smoothing.

### Lf (paper-consistent random-window statistic)
- Default: `N_WINDOWS = 10000`, `WINDOW_TARGET = 0.5`.
- Per-image RNG seed is derived from the image path hash, making results **reproducible** for the same input path.
- If you want the older deterministic “max square in free space” for debugging, set `USE_RANDOM_WINDOWS = False`
  (or run with `n_windows<=0` inside `compute_Lf_La`).

### La (maximum inscribed square in agglomerate region)
- Deterministic maximum axis-aligned square using chessboard distance transform.

---

## Reproducibility notes
- With the default settings, results are reproducible **given identical input paths and parameters** (Lf uses seeded sampling).
- Changing segmentation thresholds, agglomerate construction parameters, or preprocessing constants will change results; keep them fixed across samples.

---

## Suggested “Code availability” statement (paper)
> The code used to compute the dispersion metrics \(L_f\) and \(L_a\) and to generate vector figures is available in a public repository (link), archived at a release tag (or commit hash). Dependencies and usage instructions are provided in the repository.
