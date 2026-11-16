"""
Booster K1 - Build a grid overlay on EACH view (separate images).

Standalone component for the Booster K1 VLN pipeline. It does NOT import from
Zerui's server/client code.

Why separate images (not a stitched strip): the views are sent to the LLM as
individual images. That keeps each view at full resolution inside the model's
tile budget, avoids the ultra-wide-aspect squish of a stitched strip, and gives
a clean per-view coordinate frame. (Zerui's own server also sends the views as
separate images and only stitches for debugging.)

Input  : 3 images  -> k1_front_*, k1_left_*, k1_right_*  (1280x720)
Output : 3 images  -> grid_front.png, grid_left.png, grid_right.png
The input images are never modified; we read a copy and write new files.

No shrinking. The output keeps the input resolution (1280x720) and the grid is
drawn INSIDE that frame, so the 0-1000 coordinate system spans the whole image
exactly. That means a point the model returns maps straight back to a pixel with
    x = u / 1000 * width ,  y = v / 1000 * height
regardless of any internal resize the model does on its side (the 0-1000 grid is
resolution independent, so it stays valid for Gemini / GPT / Claude / Qwen-VL).

Coordinate system drawn on each view (0 .. COORD_MAX on BOTH axes):
    u = horizontal : 0 = left edge, 1000 = right edge
    v = vertical   : 0 = top edge,  1000 = bottom edge

Run it directly:
    python build_separate_grid_images.py
"""

from pathlib import Path
import cv2
import numpy as np

# ----------------------------- configuration -----------------------------

COORD_MAX = 1000          # LLM coordinate range on each axis (0 .. COORD_MAX)
GRID_STEP = 100           # draw a grid line every 100 coordinate units
LABEL_STEP = 200          # write a number label every 200 units (every 2nd line)
GRID_ALPHA = 0.45         # grid line opacity (0 = invisible, 1 = solid)

# Colors are BGR because we read/write with OpenCV (not RGB).
LINE_COLOR = (210, 210, 210)      # light gray grid lines
LINE_THICKNESS = 1
LABEL_TEXT_COLOR = (0, 255, 255)  # yellow coordinate labels
LABEL_BG_COLOR = (0, 0, 0)        # black box behind labels for readability
TAG_TEXT_COLOR = (255, 255, 255)  # white text on the colored view tag

# Per-view identity: a distinct, "speakable" color + a tag label.
VIEW_STYLES = {
    "front": {"color": (60, 190, 60),  "tag": "FRONT VIEW", "out": "grid_front.png"},  # green
    "left":  {"color": (40, 150, 250), "tag": "LEFT VIEW",  "out": "grid_left.png"},   # orange
    "right": {"color": (240, 130, 40), "tag": "RIGHT VIEW", "out": "grid_right.png"},  # blue
}
VIEW_ORDER = ["front", "left", "right"]

# Sizes (pixels), tuned for a ~1280-wide native view.
BORDER_THICKNESS = 8      # colored identity border, drawn inset so size is kept
TAG_PADDING = 8           # padding inside the view-name tag box

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1
TAG_FONT_SCALE = 0.9
TAG_THICKNESS = 2

# Output format. PNG keeps grid lines / labels perfectly crisp for the model.
OUTPUT_EXT = "png"        # "png" (lossless, best) or "jpg"
JPEG_QUALITY = 95         # only used when OUTPUT_EXT == "jpg"

HERE = Path(__file__).parent


# ------------------------------- drawing ----------------------------------

def draw_grid(image: np.ndarray) -> np.ndarray:
    """Draw a 0..COORD_MAX coordinate grid + labels on a copy of `image`.

    Output has the SAME size as the input. Grid lines are alpha-blended (so the
    scene stays visible); the numeric labels are drawn opaque for readability.
    """
    h, w = image.shape[:2]

    def u_to_x(u: int) -> int:
        return int(round(u / COORD_MAX * (w - 1)))

    def v_to_y(v: int) -> int:
        return int(round(v / COORD_MAX * (h - 1)))

    # 1) Grid lines on a separate layer, then blend with the scene.
    lines = image.copy()
    for u in range(0, COORD_MAX + 1, GRID_STEP):
        x = u_to_x(u)
        cv2.line(lines, (x, 0), (x, h - 1), LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    for v in range(0, COORD_MAX + 1, GRID_STEP):
        y = v_to_y(v)
        cv2.line(lines, (0, y), (w - 1, y), LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    out = cv2.addWeighted(lines, GRID_ALPHA, image, 1.0 - GRID_ALPHA, 0.0)

    # 2) Coordinate labels (opaque) at the coarse crossings.
    for u in range(0, COORD_MAX + 1, LABEL_STEP):
        for v in range(0, COORD_MAX + 1, LABEL_STEP):
            x, y = u_to_x(u), v_to_y(v)
            label = f"{u},{v}"
            (tw, th), _ = cv2.getTextSize(label, FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
            cx = min(max(x, tw // 2 + 3), w - tw // 2 - 3)   # keep inside frame
            cy = min(max(y, th + 3), h - 4)
            cv2.rectangle(out, (cx - tw // 2 - 2, cy - th - 3),
                          (cx + tw // 2 + 2, cy + 4), LABEL_BG_COLOR, -1)
            cv2.putText(out, label, (cx - tw // 2, cy),
                        FONT, LABEL_FONT_SCALE, LABEL_TEXT_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    return out


def draw_identity(image: np.ndarray, style: dict) -> None:
    """Draw the view-name tag IN PLACE (top-left corner).

    No border: the views are sent as separate images, so a frame isn't needed
    to tell them apart, and dropping it keeps the edge coordinate labels fully
    visible. The small colored tag stays as a cheap identity cue.
    """
    color = style["color"]
    tag = style["tag"]

    # View-name tag: filled colored box + white text, nudged down from the
    # top-left corner so the (0,0) coordinate label above it stays visible.
    (tw, th), _ = cv2.getTextSize(tag, FONT, TAG_FONT_SCALE, TAG_THICKNESS)
    x0, y0 = 4, 36
    cv2.rectangle(image, (x0, y0),
                  (x0 + tw + 2 * TAG_PADDING, y0 + th + 2 * TAG_PADDING),
                  color, -1)
    cv2.putText(image, tag, (x0 + TAG_PADDING, y0 + th + TAG_PADDING),
                FONT, TAG_FONT_SCALE, TAG_TEXT_COLOR, TAG_THICKNESS, cv2.LINE_AA)


def add_grid_overlay(image: np.ndarray, style: dict) -> np.ndarray:
    """Full overlay for one view: grid + labels + colored identity. Same size."""
    out = draw_grid(image)
    draw_identity(out, style)
    return out


def overlay_views(views: dict) -> dict:
    """Overlay the grid on several views at once (high-level entry for the pipeline).

    Args:
        views: {view_name: BGR image}, e.g. {"front": img, "left": img, "right": img}.
    Returns:
        {view_name: grid image (BGR)}, same sizes as the inputs.
    """
    return {name: add_grid_overlay(image, VIEW_STYLES[name]) for name, image in views.items()}


# ------------------------------- loading ----------------------------------

def find_latest_view(folder: Path, view: str) -> Path:
    """Return the newest 'k1_<view>_*.png' file (timestamped names sort by time)."""
    matches = sorted(folder.glob(f"k1_{view}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No input image found for view '{view}' in {folder}")
    return matches[-1]


def save_image(path: Path, image: np.ndarray) -> None:
    if OUTPUT_EXT == "jpg":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        cv2.imwrite(str(path), image)


def main() -> None:
    print(f"Reading views from: {HERE}")
    for view in VIEW_ORDER:
        style = VIEW_STYLES[view]
        in_path = find_latest_view(HERE, view)

        image = cv2.imread(str(in_path))          # BGR; input is never modified
        if image is None:
            raise ValueError(f"Could not read image: {in_path}")

        overlay = add_grid_overlay(image, style)

        out_name = Path(style["out"]).with_suffix(f".{OUTPUT_EXT}").name
        out_path = HERE / out_name
        save_image(out_path, overlay)

        size_kb = out_path.stat().st_size / 1024
        print(f"  {view:5s}: {in_path.name} ({image.shape[1]}x{image.shape[0]}) "
              f"-> {out_name} ({overlay.shape[1]}x{overlay.shape[0]}, {size_kb:.0f} KB)")

    print("Done. Input images left untouched.")


if __name__ == "__main__":
    main()
