"""
Pixel -> real-world point (Booster K1 VLN, mapping step 2 of 3).

Back-projects a pixel (px, py) + its depth into a 3D point using the K1 color
camera intrinsics. Depth is aligned to color, so the color intrinsics apply to
the depth pixels too.

The returned point is in the CAMERA OPTICAL FRAME (camera_color_optical_frame):
    X = right, Y = down, Z = forward   (all meters)

For v1, robot_control first turns the body to the selected view's head yaw,
then uses X/Z to turn and walk. Camera mount offset/pitch and Y are ignored.

Imported (in the pipeline):
    from pixel_to_world.pixel_to_world import pixel_to_camera_point
    point_cam = pixel_to_camera_point(px, py, depth_images[view])   # (X,Y,Z) m, or None

Standalone (independent test):
    python pixel_to_world.py
"""

import numpy as np

# K1 color-camera intrinsics (from /camera/camera/color/camera_info, 1280x720).
CALIBRATED_IMAGE_SHAPE = (720, 1280)  # numpy order: height, width
FX = 643.864013671875
FY = 643.1196899414062
CX = 640.4425048828125
CY = 366.6420593261719

DEPTH_MIN_M = 0.1        # only readings strictly between MIN and MAX are usable
DEPTH_MAX_M = 10.0


def read_depth(depth_image, px, py):
    """Depth (meters) at (px, py), without borrowing nearby readings.

    Returns 0.0 when there is no usable reading (or no depth image).
    """
    if depth_image is None:
        return 0.0
    h, w = depth_image.shape[:2]
    px = max(0, min(w - 1, int(px)))
    py = max(0, min(h - 1, int(py)))

    d = float(depth_image[py, px])          # numpy is [row=py, col=px]
    if DEPTH_MIN_M < d < DEPTH_MAX_M:
        return d

    # Unknown or rejected distance must not become a movement target.
    return 0.0


def pixel_to_camera_point(px, py, depth_image):
    """Back-project a pixel + depth to a 3D point in the camera optical frame.

    Returns np.array([X, Y, Z]) in meters (X right, Y down, Z forward), or None
    if there is no usable depth at that pixel.
    """
    d = read_depth(depth_image, px, py)
    if d <= 0.0:
        return None
    x = (px - CX) * d / FX
    y = (py - CY) * d / FY
    z = d
    return np.array([x, y, z], dtype=np.float32)


if __name__ == "__main__":
    # Synthetic flat 2.0 m plane at the K1 color resolution.
    depth = np.full((720, 1280), 2.0, dtype=np.float32)
    for px, py in [(640, 360), (0, 0), (1279, 719)]:
        print(f"pixel ({px:>4},{py:>3}) @2.0m -> camera XYZ {pixel_to_camera_point(px, py, depth)}")
    print("no reading ->", pixel_to_camera_point(640, 360, np.zeros((720, 1280), np.float32)))
