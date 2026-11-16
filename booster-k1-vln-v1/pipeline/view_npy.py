"""
view_npy.py - quick look at a saved depth .npy (raw float32 meters).

It loads the array with numpy.load, prints a summary + a coarse grid of the
actual depth values, and writes an annotated PNG (the colorized depth with the
metre value printed at sample points) next to the .npy - so you can eyeball
whether the depth looks right.

Run:
    python view_npy.py                      # defaults to run-1/checkpoint-1/front_depth.npy
    python view_npy.py <path-to-a-.npy>     # any depth .npy you want to inspect
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent       # booster-k1-vln/
DEFAULT_NPY = PROJECT_ROOT / "checkpoints" / "run-1" / "checkpoint-1" / "front_depth.npy"

DEPTH_MAX_M = 6.0        # color scale clip (matches save_checkpoint)
GRID_COLS = 9            # how many sample points across / down for the numbers
GRID_ROWS = 6


def _sample_points(h, w):
    ys = np.linspace(0, h - 1, GRID_ROWS).astype(int)
    xs = np.linspace(0, w - 1, GRID_COLS).astype(int)
    return ys, xs


def print_summary(depth):
    valid = np.isfinite(depth) & (depth > 0.0)
    print(f"shape: {depth.shape}   dtype: {depth.dtype}")
    if valid.any():
        v = depth[valid]
        print(f"valid pixels: {100 * valid.mean():.1f}%   "
              f"min: {v.min():.2f} m   max: {v.max():.2f} m   mean: {v.mean():.2f} m")
    else:
        print("no valid depth pixels (all 0 / NaN / inf)")


def print_grid(depth):
    """Print a coarse GRID_ROWS x GRID_COLS table of depth values (meters)."""
    h, w = depth.shape
    ys, xs = _sample_points(h, w)
    print("\ndepth in meters, sampled on a grid ('--' = no reading):")
    print("        " + "".join(f"x={x:<5}" for x in xs))
    for y in ys:
        cells = []
        for x in xs:
            d = depth[y, x]
            cells.append(f"{d:>7.2f}" if np.isfinite(d) and d > 0 else f"{'--':>7}")
        print(f"y={y:<5}" + "".join(cells))


def annotate(depth):
    """Colorized depth with the metre value drawn at each sample point."""
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(d / DEPTH_MAX_M, 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[d <= 0.0] = (0, 0, 0)

    h, w = depth.shape
    ys, xs = _sample_points(h, w)
    for y in ys:
        for x in xs:
            val = depth[y, x]
            txt = f"{val:.1f}" if np.isfinite(val) and val > 0 else "--"
            org = (int(x) - 14, int(y) + 4)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NPY
    if not path.exists():
        print(f"File not found: {path}")
        print("Pass a path, e.g.: python view_npy.py checkpoints/run-1/checkpoint-1/front_depth.npy")
        return 1

    depth = np.load(str(path))
    print(f"loaded {path}")
    print_summary(depth)
    print_grid(depth)

    out = path.with_name(path.stem + "_annotated.png")
    cv2.imwrite(str(out), annotate(depth))
    print(f"\nannotated image saved: {out}")
    print("open it to see the metre values printed on the depth map.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
