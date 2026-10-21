import numpy as np

CALIBRATED_IMAGE_SHAPE = (720, 1280)
FX = 643.864013671875
FY = 643.1196899414062
CX = 640.4425048828125
CY = 366.6420593261719

DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 10.0


def read_depth(depth_image, px, py):
    if depth_image is None:
        return 0.0
    h, w = depth_image.shape[:2]
    px = max(0, min(w - 1, int(px)))
    py = max(0, min(h - 1, int(py)))

    d = float(depth_image[py, px])
    if DEPTH_MIN_M < d < DEPTH_MAX_M:
        return d

    return 0.0


def pixel_to_camera_point(px, py, depth_image):
    d = read_depth(depth_image, px, py)
    if d <= 0.0:
        return None
    x = (px - CX) * d / FX
    y = (py - CY) * d / FY
    z = d
    return np.array([x, y, z], dtype=np.float32)


if __name__ == "__main__":
    depth = np.full((720, 1280), 2.0, dtype=np.float32)
    for px, py in [(640, 360), (0, 0), (1279, 719)]:
        print(f"pixel ({px:>4},{py:>3}) @2.0m -> camera XYZ {pixel_to_camera_point(px, py, depth)}")
    print("no reading ->", pixel_to_camera_point(640, 360, np.zeros((720, 1280), np.float32)))
