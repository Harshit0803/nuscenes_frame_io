import os
from random import sample
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from src.nuscenes_frame_io.frame_io import get_frame

# =========================
# CONFIG
# =========================
DATAROOT = "/home/harshit/nuScenes_data"
VERSION = "v1.0-mini"
SAMPLE_INDEX = 0  # change to view different frames


def load_lidar_xyz(bin_path: str) -> np.ndarray:
    """nuScenes lidar .pcd.bin is float32 with 5 dims: x,y,z,intensity,ring_index."""
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape(-1, 5)[:, :3]
    return pts


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    sample_token = nusc.sample[SAMPLE_INDEX]["token"]
    sample = nusc.get("sample", sample_token)
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        print(ann["category_name"], ann["num_lidar_pts"])


    # print(sample_token["num_lidar_pts"])
    frame = get_frame(nusc, sample_token)

    pts = load_lidar_xyz(frame["lidar"]["path"])

    # Basic BEV scatter
    plt.figure()
    plt.scatter(pts[:, 0], pts[:, 1], s=0.2)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"LiDAR TOP BEV (sample {SAMPLE_INDEX})")
    plt.show()


if __name__ == "__main__":
    main()
