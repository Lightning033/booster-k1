from pathlib import Path
import cv2
import numpy as np


COORD_MAX = 1000
GRID_STEP = 100
LABEL_STEP = 200
GRID_ALPHA = 0.45

LINE_COLOR = (210, 210, 210)
LINE_THICKNESS = 1
LABEL_TEXT_COLOR = (0, 255, 255)
LABEL_BG_COLOR = (0, 0, 0)
TAG_TEXT_COLOR = (255, 255, 255)

VIEW_STYLES = {
    "front": {"color": (60, 190, 60),  "tag": "FRONT VIEW", "out": "grid_front.png"},
    "left":  {"color": (40, 150, 250), "tag": "LEFT VIEW",  "out": "grid_left.png"},
    "right": {"color": (240, 130, 40), "tag": "RIGHT VIEW", "out": "grid_right.png"},
}
VIEW_ORDER = ["front", "left", "right"]

BORDER_THICKNESS = 8
TAG_PADDING = 8

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1
TAG_FONT_SCALE = 0.9
TAG_THICKNESS = 2

OUTPUT_EXT = "png"
JPEG_QUALITY = 95

HERE = Path(__file__).parent


def draw_grid(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    def u_to_x(u: int) -> int:
        return int(round(u / COORD_MAX * (w - 1)))

    def v_to_y(v: int) -> int:
        return int(round(v / COORD_MAX * (h - 1)))

    lines = image.copy()
    for u in range(0, COORD_MAX + 1, GRID_STEP):
        x = u_to_x(u)
        cv2.line(lines, (x, 0), (x, h - 1), LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    for v in range(0, COORD_MAX + 1, GRID_STEP):
        y = v_to_y(v)
        cv2.line(lines, (0, y), (w - 1, y), LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    out = cv2.addWeighted(lines, GRID_ALPHA, image, 1.0 - GRID_ALPHA, 0.0)

    for u in range(0, COORD_MAX + 1, LABEL_STEP):
        for v in range(0, COORD_MAX + 1, LABEL_STEP):
            x, y = u_to_x(u), v_to_y(v)
            label = f"{u},{v}"
            (tw, th), _ = cv2.getTextSize(label, FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
            cx = min(max(x, tw // 2 + 3), w - tw // 2 - 3)
            cy = min(max(y, th + 3), h - 4)
            cv2.rectangle(out, (cx - tw // 2 - 2, cy - th - 3),
                          (cx + tw // 2 + 2, cy + 4), LABEL_BG_COLOR, -1)
            cv2.putText(out, label, (cx - tw // 2, cy),
                        FONT, LABEL_FONT_SCALE, LABEL_TEXT_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    return out


def draw_identity(image: np.ndarray, style: dict) -> None:
    color = style["color"]
    tag = style["tag"]

    (tw, th), _ = cv2.getTextSize(tag, FONT, TAG_FONT_SCALE, TAG_THICKNESS)
    x0, y0 = 4, 36
    cv2.rectangle(image, (x0, y0),
                  (x0 + tw + 2 * TAG_PADDING, y0 + th + 2 * TAG_PADDING),
                  color, -1)
    cv2.putText(image, tag, (x0 + TAG_PADDING, y0 + th + TAG_PADDING),
                FONT, TAG_FONT_SCALE, TAG_TEXT_COLOR, TAG_THICKNESS, cv2.LINE_AA)


def add_grid_overlay(image: np.ndarray, style: dict) -> np.ndarray:
    out = draw_grid(image)
    draw_identity(out, style)
    return out


def overlay_views(views: dict) -> dict:
    return {name: add_grid_overlay(image, VIEW_STYLES[name]) for name, image in views.items()}


def find_latest_view(folder: Path, view: str) -> Path:
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

        image = cv2.imread(str(in_path))
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
