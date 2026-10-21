import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture_3_views_and_3_depth_images.capture_3_views_and_3_depth_images import (
    close_camera,
    get_3_views_and_3_depth_images,
    initialize_camera,
)
from build_separate_grid_images.build_separate_grid_images import overlay_views
from build_combinded_grid_image.build_combinded_grid_image import combine_views
from save_checkpoint.save_checkpoint import (
    save_checkpoint_images,
    save_checkpoint_result,
    start_new_run,
)
from llm_client.llm_client import ask_gesture, ask_llm
from coordinate_to_pixel.coordinate_to_pixel import coordinate_to_pixel
from pixel_to_world.pixel_to_world import CALIBRATED_IMAGE_SHAPE, pixel_to_camera_point
from navigation_interrupt import begin_navigation_cleanup
from robot_control.robot_control import (
    enter_walk_mode,
    look_behind,
    move_to_target,
    perform_gesture,
    stop,
)

NAVIGATION_INSTRUCTION = "Move and stand besides the orange ladder. Ensure to keep safe distance from the ladder. Shake your hand after reaching destination."
SAFETY_STOP = 20
MAX_LOOK_BEHIND = 2
STOP_SETTLE_S = 10.0


def validate_capture_resolution(views, depth_images):
    for view in ("front", "left", "right"):
        rgb_shape = views[view].shape[:2]
        depth = depth_images[view]
        if depth is not None and depth.shape != rgb_shape:
            raise ValueError(f"RGB/depth shape mismatch for {view}: "
                             f"RGB {rgb_shape}, depth {depth.shape}")
        if rgb_shape != CALIBRATED_IMAGE_SHAPE:
            raise ValueError(f"Calibration requires 1280 x 720 RGB and depth for {view}; "
                             f"received RGB {rgb_shape} (height, width)")


def main(instruction=NAVIGATION_INSTRUCTION):
    new_planning = []
    thought_history = []
    look_behind_count = 0
    checkpoints = 0

    run_dir = start_new_run()
    camera = None
    robot_started = False
    try:
        camera = initialize_camera()
        robot_started = True
        enter_walk_mode()
        stop()
        while checkpoints < SAFETY_STOP:

            views, depth_images = get_3_views_and_3_depth_images(camera)
            validate_capture_resolution(views, depth_images)
            grid_images = overlay_views(views)
            combined_image = combine_views(views)
            checkpoint_dir = save_checkpoint_images(
                checkpoints + 1, views, grid_images, combined_image,
                depth_images=depth_images, root=run_dir)

            result = ask_llm(grid_images, instruction, new_planning, thought_history)
            save_checkpoint_result(checkpoint_dir, result)
            if result["action"] != "retry":
                new_planning = result["new_planning"]
            thought_history.append(result["thought"])

            action = result["action"]

            print(result)

            if action == "stop":
                if result.get("invalid_reply"):
                    raise RuntimeError(f"Invalid navigation reply: {result['thought']}")
                gesture = ask_gesture(instruction)
                print(gesture)
                perform_gesture(gesture["gesture"])
                break

            elif action == "retry":
                print("Rejected model coordinates; capturing again without body movement.")

            elif action == "look_behind":
                if look_behind_count >= MAX_LOOK_BEHIND:
                    raise RuntimeError(
                        f"Navigation stopped after {MAX_LOOK_BEHIND} consecutive "
                        "look-behind turns without reaching the goal.")
                look_behind()
                look_behind_count += 1

            elif action == "move":
                look_behind_count = 0
                view = result["view"]
                px, py = coordinate_to_pixel(result["u"], result["v"], views[view])
                point_cam = pixel_to_camera_point(px, py, depth_images[view])
                print(f"target: {view}, pixel=({px},{py}) camera_xyz={point_cam}")
                move_to_target(point_cam, view)

            else:
                raise ValueError(f"Unknown navigation action: {action!r}")

            checkpoints += 1
        else:
            raise RuntimeError(
                f"Navigation reached the {SAFETY_STOP}-checkpoint limit "
                "without reaching the goal.")
    finally:
        begin_navigation_cleanup()
        try:
            if robot_started:
                stop()
                time.sleep(STOP_SETTLE_S)
        finally:
            close_camera(camera)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Booster K1 VLN on the lab PC.")
    parser.add_argument("instruction", nargs="?", default=NAVIGATION_INSTRUCTION)
    main(parser.parse_args().instruction)
