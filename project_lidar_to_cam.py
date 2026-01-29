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
CAM_NAME = "CAM_FRONT"   


def load_lidar_xyz(bin_path: str) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:, :3]

def transform_points_h(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points. Returns Nx3."""
    pts_xyz = np.asarray(pts_xyz, dtype=np.float64)
    ones = np.ones((pts_xyz.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts_xyz, ones])  # (N,4)
    out = (T @ pts_h.T).T               # (N,4)
    return out[:, :3]


def project_cam(K: np.ndarray, pts_c: np.ndarray):
    """
    Project Nx3 camera-frame points to pixel coords.
    Returns (u,v,z) where u,v are float arrays and z is depth.
    """
    pts_c = np.asarray(pts_c, dtype=np.float64)
    z = pts_c[:, 2]
    proj = (K @ pts_c.T).T  # (N,3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    return u, v, z


def box_corners_in_cam_from_global_box(box_g, T_cam_from_global: np.ndarray) -> np.ndarray:
    """
    box_g is a nuScenes Box in GLOBAL frame.
    Returns 8 corners in CAMERA frame as (8,3).
    """
    # Box corners in GLOBAL frame: (3,8)
    corners_g = box_g.corners().T  # -> (8,3)

    # Transform corners to camera frame
    corners_c = transform_points_h(corners_g, T_cam_from_global)  # (8,3)
    return corners_c


def draw_3d_box_2d(img: np.ndarray, corners_uv: np.ndarray, color=(0, 255, 0), thickness=2):
    """
    Draw projected 3D box (8 corners) on image.
    corners_uv: (8,2) in the same order as nuScenes Box.corners() provides.
    """
    c = corners_uv.astype(int)

    # nuScenes corners ordering (standard):
    # 0-3 are one face, 4-7 are the opposite face.
    # We'll draw edges:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
        (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    for i, j in edges:
        cv2.line(img, tuple(c[i]), tuple(c[j]), color, thickness, lineType=cv2.LINE_AA)


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

    drawn = 0
    skipped_behind = 0
    skipped_off = 0

    for ann in frame["anns"]:
        ann_token = ann["token"]
        box_g = nusc.get_box(ann_token)  # GLOBAL frame Box

        # Corners in camera frame
        corners_c = box_corners_in_cam_from_global_box(box_g, T_cam_from_global)  # (8,3)

        # If most corners are behind camera, skip (z<=0)
        zc = corners_c[:, 2]
        if np.count_nonzero(zc > 0.5) < 6:
            skipped_behind += 1
            continue

        u_b, v_b, z_b = project_cam(K, corners_c)
        corners_uv = np.stack([u_b, v_b], axis=1)  # (8,2)

        # Skip if completely outside image (optional; keeps view clean)
        if (
            np.all(corners_uv[:, 0] < 0) or np.all(corners_uv[:, 0] >= W) or
            np.all(corners_uv[:, 1] < 0) or np.all(corners_uv[:, 1] >= H)
        ):
            skipped_off += 1
            continue

        draw_3d_box_2d(out, corners_uv, color=(0, 255, 0), thickness=2)
        drawn += 1

    print(f"Projected LiDAR points: {len(u)}")
    print(f"Drew GT boxes: {drawn} (skipped behind: {skipped_behind}, skipped offscreen: {skipped_off})")

    cv2.imshow(f"LiDAR -> {CAM_NAME} + GT 3D boxes", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
