"""
Checkpoint saving step.

Each run gets its own folder (run-1, run-2, ...) under CHECKPOINT_ROOT, and each
decision step within a run gets a checkpoint-1, checkpoint-2, ... folder inside
that run. We save the 3 raw views, the 3 grid-overlaid views, and the 3 depth
maps (a colorized PNG for viewing plus the raw float32-meter .npy) there, so you
can see exactly what the robot saw. After the LLM returns, its parsed result is
saved as readable JSON in llm_output.txt in the same checkpoint folder.
Runs are never overwritten, so you keep a history of attempts for human review
(the LLM never reads these back).

Imported (in the pipeline):
    from save_checkpoint.save_checkpoint import save_checkpoint_images, start_new_run
    run_dir = start_new_run()                                   # once per run
    save_checkpoint_images(step_number, views, grid_views, depth_images=depths, root=run_dir)

Standalone (independent test):
    python save_checkpoint.py     # saves a tiny synthetic checkpoint to a temp folder
"""

import json
from pathlib import Path

import cv2
import numpy as np

# checkpoints/ lives at the project root (this file is in <root>/save_checkpoint/).
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"

# Depth (meters) beyond this is clipped when colorizing the human-view PNG.
DEPTH_MAX_M = 6.0


def _save_views(folder, views, suffix=""):
    folder.mkdir(parents=True, exist_ok=True)
    for name, image in views.items():
        cv2.imwrite(str(folder / f"{name}{suffix}.png"), image)


def _depth_to_color(depth):
    """Colorize a float32 depth map (meters) into a human-viewable BGR image.

    Closer = blue, farther = red (JET); pixels with no reading (0 / NaN / inf)
    are shown black. This is a VISUALIZATION only - raw metric values are not
    preserved.
    """
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(d / DEPTH_MAX_M, 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[d <= 0.0] = (0, 0, 0)          # no-reading pixels -> black
    return color


def _save_depths(folder, depth_images):
    """Save each depth map two ways:
      - '<name>_depth.png' : colorized, for quick human review.
      - '<name>_depth.npy' : raw float32 meters (exact values), for the
        pixel->world mapping and debugging. Load with numpy.load(path).
    """
    folder.mkdir(parents=True, exist_ok=True)
    for name, depth in depth_images.items():
        if depth is None:
            continue
        depth = np.asarray(depth, dtype=np.float32)
        cv2.imwrite(str(folder / f"{name}_depth.png"), _depth_to_color(depth))
        np.save(str(folder / f"{name}_depth.npy"), depth)


def save_checkpoint_images(step_number, views, grid_views, combined=None,
                           depth_images=None, root=CHECKPOINT_ROOT):
    """Save one checkpoint folder.

    Args:
        step_number:  1-based step index -> folder 'checkpoint-<step_number>'.
        views:        {name: raw BGR image}
        grid_views:   {name: grid BGR image}
        combined:     optional single BGR image (all views stitched) -> combined_grid.png
        depth_images: optional {name: float32 depth map in meters} -> saved as
                      <name>_depth.png (colorized for review) AND <name>_depth.npy
                      (raw float32 meters). None entries are skipped.
        root:         base folder for checkpoints (default CHECKPOINT_ROOT).
    Returns:
        Path to the checkpoint folder.
    """
    folder = Path(root) / f"checkpoint-{step_number}"
    folder.mkdir(parents=True, exist_ok=True)
    _save_views(folder, views)
    _save_views(folder, grid_views, suffix="_grid")
    if depth_images is not None:
        _save_depths(folder, depth_images)
    if combined is not None:
        cv2.imwrite(str(folder / "combined_grid.png"), combined)
    return folder


def save_checkpoint_result(checkpoint_dir, result):
    """Save ask_llm()'s result beside the images, before executing its action.

    checkpoint_dir is the folder returned by save_checkpoint_images().
    The .txt file contains the complete result as indented UTF-8 JSON.
    """
    path = Path(checkpoint_dir) / "llm_output.txt"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def start_new_run(root=CHECKPOINT_ROOT):
    """Create the next run folder (run-1, run-2, ...) under `root` and return it.

    Each run keeps its own checkpoint-N folders, so previous runs are preserved
    for human review instead of being overwritten.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def run_index(name):
        suffix = name[len("run-"):]
        return int(suffix) if name.startswith("run-") and suffix.isdigit() else None

    used = [run_index(p.name) for p in root.glob("run-*") if p.is_dir()]
    next_index = max([i for i in used if i is not None], default=0) + 1

    run_dir = root / f"run-{next_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    import tempfile
    import numpy as np

    dummy = {name: np.full((60, 80, 3), 90, dtype=np.uint8) for name in ("front", "left", "right")}
    out = save_checkpoint_images(1, dummy, dummy, root=Path(tempfile.mkdtemp()))
    print("wrote", sorted(p.name for p in out.iterdir()), "to", out)
