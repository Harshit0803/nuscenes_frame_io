import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from src.nuscenes_frame_io.frame_io import get_frame

# =========================
# CONFIG
# =========================
DATAROOT = "/home/harshit/nuScenes_data"
VERSION = "v1.0-mini"
SAMPLE_INDEX = 0
CAM_NAME = "CAM_FRONT"   # try CAM_FRONT_LEFT, CAM_FRONT_RIGHT


def load_lidar_xyz(bin_path: str) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:, :3]


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    sample_token = nusc.sample[SAMPLE_INDEX]["token"]
    frame = get_frame(nusc, sample_token)

    # --- Load data ---
    pts_l = load_lidar_xyz(frame["lidar"]["path"])  # LiDAR sensor frame
    cam = frame["cameras"][CAM_NAME]
    K = cam["K"]
    img = cv2.imread(cam["path"], cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {cam['path']}")

    H, W = img.shape[:2]

    # --- Transform LiDAR points into camera frame ---
    T_global_from_lidar = frame["lidar"]["T_global_from_sensor"]
    T_global_from_cam = cam["T_global_from_sensor"]

    T_cam_from_global = np.linalg.inv(T_global_from_cam)
    T_cam_from_lidar = T_cam_from_global @ T_global_from_lidar

    # Homogeneous transform
    pts_h = np.hstack([pts_l, np.ones((pts_l.shape[0], 1), dtype=np.float64)])
    pts_c = (T_cam_from_lidar @ pts_h.T).T[:, :3]  # camera frame

    # Keep points in front of camera
    z = pts_c[:, 2]
    valid = z > 1.0  # meters
    pts_c = pts_c[valid]
    z = z[valid]

    # --- Project to pixels ---
    # u,v = K * [x,y,z]
    proj = (K @ pts_c.T).T  # (N,3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inside]
    v = v[inside]
    z = z[inside]

    # --- Draw points (color by depth, closer = brighter) ---
    # Normalize depth to 0..1 for visualization
    z_clip = np.clip(z, 1.0, 60.0)
    t = 1.0 - (z_clip - 1.0) / (60.0 - 1.0)  # closer -> higher
    colors = (255 * t).astype(np.uint8)

    out = img.copy()
    for px, py, c in zip(u, v, colors):
        # BGR: draw bluish points with intensity by depth
        out = cv2.circle(out, (int(px), int(py)), 1, (int(c), 0, 255 - int(c)), -1)

    cv2.imshow(f"LiDAR -> {CAM_NAME} projection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
