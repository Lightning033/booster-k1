"""
Coordinate -> pixel  (Booster K1 VLN, mapping step 1 of 3).

The LLM returns its chosen point as (u, v) on a 0..COORD_MAX grid that was drawn
over each view. This converts that back to a real pixel (px, py) in the image,
matching exactly how the grid was drawn (build_separate_grid_images maps
0..COORD_MAX across 0..width-1 and 0..height-1). Because depth is aligned to
color, the same (px, py) also indexes the depth map.

Next steps live in their own modules:
    step 2: pixel -> real-world point (uses depth + camera intrinsics)
    step 3: command the robot to walk to that point

Imported (in the pipeline):
    from coordinate_to_pixel.coordinate_to_pixel import coordinate_to_pixel
    px, py = coordinate_to_pixel(result["u"], result["v"], views[result["view"]])

Standalone (independent test):
    python coordinate_to_pixel.py
"""

import math

COORD_MAX = 1000


def coordinate_to_pixel(u, v, image, coord_max=COORD_MAX):
    """Map an LLM (u, v) in 0..coord_max to a pixel (px, py) in `image`.

    Args:
        u, v:  LLM coordinates (u = horizontal, v = vertical), 0..coord_max.
        image: the view the coordinates refer to; only its height/width are used
               (an (H, W[, C]) array). Depth is aligned to color, so the same
               (px, py) indexes the depth map too.
    Returns:
        (px, py): integer pixel within the image bounds.
    Raises:
        ValueError: coordinates are not finite numbers within the grid.
    """
    for value in (u, v):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= coord_max or not math.isfinite(value)):
            raise ValueError("Coordinates must be finite numbers within the grid")
    height, width = image.shape[:2]
    px = int(round(u / coord_max * (width - 1)))
    py = int(round(v / coord_max * (height - 1)))
    return px, py


if __name__ == "__main__":
    import numpy as np

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)   # H=720, W=1280 (K1 color res)
    print("image size: 1280x720")
    for u, v in [(0, 0), (1000, 1000), (500, 500), (492, 906)]:
        print(f"(u={u:>4}, v={v:>4}) -> pixel {coordinate_to_pixel(u, v, dummy)}")
