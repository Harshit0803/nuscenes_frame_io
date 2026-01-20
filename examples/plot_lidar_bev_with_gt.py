import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from nuscenes_frame_io.frame_io import get_frame

DATAROOT = "/home/harshit/nuScenes_data"
VERSION = "v1.0-mini"
SAMPLE_INDEX = 0

X_RANGE = (0.0, 60.0)
Y_RANGE = (-30.0, 30.0)
Z_RANGE = (-3.0, 3.0)


def load_lidar_xyz(bin_path: str) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:, :3]


def transform_box_global_to_lidar(box_g, T_lidar_from_global: np.ndarray):
    """
    box_g: nuscenes.utils.data_classes.Box in GLOBAL frame
    returns (center_l, quat_l, size) in LiDAR sensor frame
    """
    # center
    c = np.array(box_g.center, dtype=np.float64)  # (3,)
    c_h = np.hstack([c, 1.0])
    c_l = (T_lidar_from_global @ c_h)[:3]

    # orientation: R_l = R_T * R_g  (where R_T is rotation part of T_lidar_from_global)
    R_T = T_lidar_from_global[:3, :3]
    R_g = box_g.orientation.rotation_matrix
    R_l = R_T @ R_g
    q_l = Quaternion(matrix=R_l)

    # size is unchanged
    size = box_g.wlh  # (w, l, h)

    return c_l, q_l, size


def box_bev_corners(center, quat: Quaternion, wlh):
    """
    Returns 4 BEV corners (x,y) of the bottom face in the box local frame,
    rotated by quat and translated by center.
    """
    w, l, h = wlh
    # corners in box local frame (bottom face z = -h/2)
    # order: front-left, front-right, rear-right, rear-left (in local frame)
    x = l / 2.0
    y = w / 2.0
    local = np.array([
        [ x,  y, -h/2],
        [ x, -y, -h/2],
        [-x, -y, -h/2],
        [-x,  y, -h/2],
    ], dtype=np.float64)

    R = quat.rotation_matrix
    world = (R @ local.T).T + center.reshape(1, 3)
    return world[:, :2]  # (4,2)


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    sample_token = nusc.sample[SAMPLE_INDEX]["token"]
    frame = get_frame(nusc, sample_token)

    pts = load_lidar_xyz(frame["lidar"]["path"])
    mask = (
        (pts[:, 0] >= X_RANGE[0]) & (pts[:, 0] <= X_RANGE[1]) &
        (pts[:, 1] >= Y_RANGE[0]) & (pts[:, 1] <= Y_RANGE[1]) &
        (pts[:, 2] >= Z_RANGE[0]) & (pts[:, 2] <= Z_RANGE[1])
    )
    pts = pts[mask]

    fig, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"LiDAR BEV + GT Boxes (sample {SAMPLE_INDEX})")

    T_global_from_lidar = frame["lidar"]["T_global_from_sensor"]
    T_lidar_from_global = np.linalg.inv(T_global_from_lidar)

    drawn = 0
    for ann in frame["anns"]:
        ann_token = ann["token"]
        box_g = nusc.get_box(ann_token)  # GLOBAL frame

        center_l, quat_l, wlh = transform_box_global_to_lidar(box_g, T_lidar_from_global)

        # filter to plot window
        cx, cy, cz = center_l
        if not (X_RANGE[0] <= cx <= X_RANGE[1] and Y_RANGE[0] <= cy <= Y_RANGE[1] and Z_RANGE[0] <= cz <= Z_RANGE[1]):
            continue

        bev = box_bev_corners(center_l, quat_l, wlh)
        poly = np.vstack([bev, bev[0]])
        ax.plot(poly[:, 0], poly[:, 1], linewidth=1.0)

        drawn += 1

    print(f"Plotted {pts.shape[0]} points, drew {drawn} GT boxes.")
    plt.show()


if __name__ == "__main__":
    main()
