import json
from pathlib import Path

import cv2
import numpy as np

CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"

DEPTH_MAX_M = 6.0


def _save_views(folder, views, suffix=""):
    folder.mkdir(parents=True, exist_ok=True)
    for name, image in views.items():
        cv2.imwrite(str(folder / f"{name}{suffix}.png"), image)


def _depth_to_color(depth):
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(d / DEPTH_MAX_M, 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[d <= 0.0] = (0, 0, 0)
    return color


def _save_depths(folder, depth_images):
    folder.mkdir(parents=True, exist_ok=True)
    for name, depth in depth_images.items():
        if depth is None:
            continue
        depth = np.asarray(depth, dtype=np.float32)
        cv2.imwrite(str(folder / f"{name}_depth.png"), _depth_to_color(depth))
        np.save(str(folder / f"{name}_depth.npy"), depth)


def save_checkpoint_images(step_number, views, grid_views, combined=None,
                           depth_images=None, root=CHECKPOINT_ROOT):
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
    path = Path(checkpoint_dir) / "llm_output.txt"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def start_new_run(root=CHECKPOINT_ROOT):
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
