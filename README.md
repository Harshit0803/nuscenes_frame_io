# nuscenes_frame_io


ego frame -> Origin: center of the vehicle
+x  forward
+y  left
+z  up

so all data in this format to go global: sensor -> ego -> global
T_global_from_sensor = T_global_from_ego @ T_ego_from_sensor

for data:
1 lidar

6 cams:
CAM_FRONT
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
CAM_BACK
CAM_BACK_LEFT
CAM_BACK_RIGHT

for each cam: extrinsics (T from sensor to ego) and K matrix (intrinsic)

sample = nusc.get("sample", sample_token)
lidar_token = sample["data"]["LIDAR_TOP"]
T_ego_from_lidar, _ = _get_T_ego_from_sensor(nusc, lidar_token)
T_global_from_ego, _ = _get_T_global_from_ego(nusc, lidar_token)
T_global_from_lidar = T_global_from_ego @ T_ego_from_lidar

cam_token = sample["data"][cam]
T_ego_from_cam, cs = _get_T_ego_from_sensor(nusc, cam_token)
T_global_from_ego, _ = _get_T_global_from_ego(nusc, cam_token)
T_global_from_cam = T_global_from_ego @ T_ego_from_cam

K = np.array(cs["camera_intrinsic"], dtype=np.float64) if cs.get("camera_intrinsic") else None

calibrated_sensor -> cs
ego_pose -> ep

frame["cameras"][cam] = {
            "token": cam_token,
            "path": nusc.get_sample_data_path(cam_token),
            "K": K,
            "T_ego_from_sensor": T_ego_from_cam,
            "T_global_from_sensor": T_global_from_cam,
        }

annotations:
['token', 'sample_token', 'instance_token', 'visibility_token', 'attribute_tokens', 'translation', 'size', 'rotation', 'prev', 'next', 'num_lidar_pts', 'num_radar_pts', 'category_name']

<img width="1600" height="900" alt="LiDAR -  CAM_FRONT projection_screenshot_20 01 2026" src="https://github.com/user-attachments/assets/0ddb973d-6392-48dc-bccf-cf009736db42" />

<img width="1850" height="982" alt="Figure_1" src="https://github.com/user-attachments/assets/3a94e55d-5c51-48ad-bd68-cfa6171bb8fc" />


![lidar top:](<Lidar top.png>)


