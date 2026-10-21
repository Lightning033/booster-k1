import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from build_separate_grid_images.build_separate_grid_images import draw_grid, VIEW_STYLES

HEADER_TEXT_COLOR = (255, 255, 255)
CANVAS_BG = (255, 255, 255)
BORDER = 8
HEADER_H = 60
GUTTER = 44

FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_FONT_SCALE = 1.1

VIEW_ORDER = ["front", "left", "right"]

HERE = Path(__file__).parent


def frame_panel(image: np.ndarray, style: dict) -> np.ndarray:
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
    padded = [
        cv2.copyMakeBorder(p, GUTTER, GUTTER, GUTTER // 2, GUTTER // 2,
                           cv2.BORDER_CONSTANT, value=CANVAS_BG)
        for p in panels
    ]
    strip = np.hstack(padded)
    return cv2.copyMakeBorder(strip, 0, 0, GUTTER // 2, GUTTER // 2,
                              cv2.BORDER_CONSTANT, value=CANVAS_BG)


def combine_views(views: dict) -> np.ndarray:
    panels = [frame_panel(draw_grid(views[name]), VIEW_STYLES[name]) for name in VIEW_ORDER]
    return combine_panels(panels)


if __name__ == "__main__":
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
