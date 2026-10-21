import math

COORD_MAX = 1000


def coordinate_to_pixel(u, v, image, coord_max=COORD_MAX):
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

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    print("image size: 1280x720")
    for u, v in [(0, 0), (1000, 1000), (500, 500), (492, 906)]:
        print(f"(u={u:>4}, v={v:>4}) -> pixel {coordinate_to_pixel(u, v, dummy)}")
