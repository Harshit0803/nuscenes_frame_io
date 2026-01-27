import numpy as np
from pyquaternion import Quaternion # type: ignore


def quat_trans_to_T(q, t): 
    """q: [w,x,y,z] or Quaternion ; t: [x,y,z] -> 4x4"""
    if not isinstance(q, Quaternion):
        q = Quaternion(q)
    R = q.rotation_matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def invert_T(T):

    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t #-np.dot(R.T, t)
    return Ti


def transform_points(points_xyz, T):
    """points_xyz: (N,3), T:(4,4)"""
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([points_xyz, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]
