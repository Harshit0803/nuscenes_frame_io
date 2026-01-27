import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from nuscenes.nuscenes import NuScenes
from src.nuscenes_frame_io.frame_io import get_frame

DATAROOT = "/home/harshit/nuScenes_data"
VERSION = "v1.0-mini"

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
sample_token = nusc.sample[0]["token"]
print(sample_token)
frame = get_frame(nusc, sample_token)

print("sample:", frame["sample_token"])
print("lidar path:", frame["lidar"]["path"])
print("cams:", list(frame["cameras"].keys()))
print("cams K:", frame["cameras"]["CAM_FRONT"]["K"])
print("anns:", len(frame["anns"]))
