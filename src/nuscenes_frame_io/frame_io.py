import numpy as np
from nuscenes.nuscenes import NuScenes
from .transforms import quat_trans_to_T

DEFAULT_CAMS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

def _get_T_ego_from_sensor(nusc: NuScenes, sample_data_token: str):
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T = quat_trans_to_T(cs["rotation"], cs["translation"])
    return T, cs

def _get_T_global_from_ego(nusc: NuScenes, sample_data_token: str):
    sd = nusc.get("sample_data", sample_data_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T = quat_trans_to_T(ep["rotation"], ep["translation"])
    return T, ep

def get_frame(nusc: NuScenes, sample_token: str, cam_names=None):
    """
    Returns a dict containing:
      - lidar path, transforms
      - camera paths, intrinsics, transforms
      - annotations (raw box params in GLOBAL)
    """
    sample = nusc.get("sample", sample_token)
    if cam_names is None:
        cam_names = DEFAULT_CAMS

    frame = {
        "sample_token": sample_token,
        "timestamp": sample["timestamp"],
        "lidar": {},
        "cameras": {},
        "anns": []
    }

    # -------- LiDAR TOP --------
    lidar_token = sample["data"]["LIDAR_TOP"]
    T_ego_from_lidar, _ = _get_T_ego_from_sensor(nusc, lidar_token)
    T_global_from_ego, _ = _get_T_global_from_ego(nusc, lidar_token)
    T_global_from_lidar = T_global_from_ego @ T_ego_from_lidar

    frame["lidar"] = {
        "token": lidar_token,
        "path": nusc.get_sample_data_path(lidar_token),
        "T_ego_from_sensor": T_ego_from_lidar,
        "T_global_from_sensor": T_global_from_lidar,
    }

    # -------- Cameras --------
    for cam in cam_names:
        if cam not in sample["data"]:
            continue
        cam_token = sample["data"][cam]
        T_ego_from_cam, cs = _get_T_ego_from_sensor(nusc, cam_token)
        T_global_from_ego, _ = _get_T_global_from_ego(nusc, cam_token)
        T_global_from_cam = T_global_from_ego @ T_ego_from_cam

        K = np.array(cs["camera_intrinsic"], dtype=np.float64) if cs.get("camera_intrinsic") else None

        frame["cameras"][cam] = {
            "token": cam_token,
            "path": nusc.get_sample_data_path(cam_token),
            "K": K,
            "T_ego_from_sensor": T_ego_from_cam,
            "T_global_from_sensor": T_global_from_cam,
        }

    # -------- Annotations (GLOBAL frame in nuScenes) --------
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        frame["anns"].append({
            "token": ann_token,
            "category": ann["category_name"],
            "translation": ann["translation"],
            "size": ann["size"],           # width, length, height (nuScenes order)
            "rotation": ann["rotation"],   # quaternion [w,x,y,z]
            "num_lidar_pts": ann.get("num_lidar_pts", None),
            "num_radar_pts": ann.get("num_radar_pts", None),
        })

    return frame
