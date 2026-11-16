"""
Combined grid image (human / debug view).

Stitches the 3 views into one wide image with a colored header per view and
white gutters between them (the v1 look). This is a DEBUG artifact saved into
each checkpoint folder; the LLM is still shown the 3 views SEPARATELY.

Separation of concerns:
  * The grid drawing and the per-view colors/labels come from the grid step
    (build_separate_grid_images). This file does NOT re-implement the grid.
  * This file only adds the colored header bands + white gutters and stitches.

Imported (in the pipeline):
    from build_combinded_grid_image.build_combinded_grid_image import combine_views
    combined = combine_views(views)      # views = {name: raw BGR image}

Standalone (independent test):
    python build_combinded_grid_image.py   # reads the sample images here, saves the combined image
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the grid step (sibling folder) for grid drawing + view colors/labels.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from build_separate_grid_images.build_separate_grid_images import draw_grid, VIEW_STYLES

# Layout (v1 look): colored header band per panel + white gutters.
HEADER_TEXT_COLOR = (255, 255, 255)   # white text on the colored header
CANVAS_BG = (255, 255, 255)           # white gutters / background
BORDER = 8                            # colored frame thickness (l/r/bottom)
HEADER_H = 60                         # header band height (top of each panel)
GUTTER = 44                           # white space between and around panels

FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_FONT_SCALE = 1.1

VIEW_ORDER = ["front", "left", "right"]

HERE = Path(__file__).parent


def frame_panel(image: np.ndarray, style: dict) -> np.ndarray:
    """Wrap a gridded view in its colored frame with a header bar."""
    color = style["color"]
    label = style["tag"]

    framed = cv2.copyMakeBorder(image, HEADER_H, BORDER, BORDER, BORDER,
                                cv2.BORDER_CONSTANT, value=color)
    (tw, th), _ = cv2.getTextSize(label, FONT, HEADER_FONT_SCALE, 2)
    tx = max((framed.shape[1] - tw) // 2, 10)
    ty = (HEADER_H + th) // 2
    cv2.putText(framed, label, (tx, ty), FONT, HEADER_FONT_SCALE,
                HEADER_TEXT_COLOR, 2, cv2.LINE_AA)
    return framed


def combine_panels(panels: list) -> np.ndarray:
    """Place framed panels on a white canvas with even gutters between them."""
    padded = [
        cv2.copyMakeBorder(p, GUTTER, GUTTER, GUTTER // 2, GUTTER // 2,
                           cv2.BORDER_CONSTANT, value=CANVAS_BG)
        for p in panels
    ]
    strip = np.hstack(padded)
    return cv2.copyMakeBorder(strip, 0, 0, GUTTER // 2, GUTTER // 2,
                              cv2.BORDER_CONSTANT, value=CANVAS_BG)


def combine_views(views: dict) -> np.ndarray:
    """Build the combined debug image from {name: raw BGR image}.

    Draws the grid on each view (reusing the grid step), frames each with its
    colored header, and stitches them side by side with gutters.
    """
    panels = [frame_panel(draw_grid(views[name]), VIEW_STYLES[name]) for name in VIEW_ORDER]
    return combine_panels(panels)


if __name__ == "__main__":
    # Independent test: read the sample views in this folder, save the combined image.
    sample_views = {}
    for view in VIEW_ORDER:
        matches = sorted(HERE.glob(f"k1_{view}_*.png"))
        if not matches:
            raise FileNotFoundError(f"No sample image for '{view}' in {HERE}")
        sample_views[view] = cv2.imread(str(matches[-1]))

    combined = combine_views(sample_views)
    out_path = HERE / "k1_combined_grid.png"
    cv2.imwrite(str(out_path), combined)
    print(f"saved {out_path}  ({combined.shape[1]}x{combined.shape[0]})")
