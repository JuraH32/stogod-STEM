from __future__ import annotations

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import CameraIntrinsics, CameraPose


def unity_pose_to_opencv_c2w(pose: CameraPose) -> np.ndarray:
    """
    Convert a Unity-convention CameraPose to an OpenCV camera-to-world 4×4 matrix.

    Unity is left-handed (X right, Y up, Z forward). OpenCV is right-handed (X right,
    Y down, Z forward). camera_to_world_matrix() packs columns as (right, up, forward,
    position) in Unity space. Applying diag(1,-1,1,1) on both sides negates the Y axis
    of the rotation block and the Y component of the translation, converting the
    left-handed basis to a right-handed one consistent with OpenCV conventions.
    """
    M = np.array(pose.camera_to_world_matrix(), dtype=np.float32)
    M[1, :] *= -1
    M[:, 1] *= -1
    return M


def build_intrinsics_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(intrinsics.matrix, dtype=np.float32)


def build_normalised_intrinsics(intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Return the K matrix normalised for MoGe-2.
    MoGe expects fx/W, fy/H, cx/W, cy/H so that coordinates are in [0,1] pixel space.
    """
    W, H = intrinsics.image_size
    K = build_intrinsics_matrix(intrinsics)
    K_norm = K.copy()
    K_norm[0, :] /= W
    K_norm[1, :] /= H
    return K_norm


def unproject_depth_to_world(
    depth_map: np.ndarray,
    validity_mask: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    c2w_opencv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unproject valid depth pixels to 3D world space.

    Args:
        depth_map:     (H, W) float32, metric depth in metres
        validity_mask: (H, W) bool, True where depth is valid
        rgb_image:     (H, W, 3) uint8
        intrinsics:    camera intrinsics (cx, cy, fx, fy)
        c2w_opencv:    (4, 4) float32 OpenCV camera-to-world matrix

    Returns:
        points_world:  (N, 3) float32
        colors_rgb:    (N, 3) uint8
        pixel_coords:  (N, 2) int32 — (row, col) per point
    """
    rows, cols = np.where(validity_mask)
    z = depth_map[rows, cols].astype(np.float32)
    x = (cols.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    ones = np.ones(len(z), dtype=np.float32)
    pts_h = np.stack([x, y, z, ones], axis=1)     # (N, 4)
    pts_world = (c2w_opencv @ pts_h.T).T[:, :3]   # (N, 3)
    colors = rgb_image[rows, cols]                  # (N, 3) uint8
    pixel_coords = np.stack([rows, cols], axis=1).astype(np.int32)
    return pts_world.astype(np.float32), colors, pixel_coords


def project_points_to_camera(
    points_world: np.ndarray,
    c2w_opencv: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D world points into image pixel coordinates.

    Returns:
        uv:          (N, 2) float32 — (col, row) pixel coordinates
        depth_cam:   (N,) float32 — depth in camera Z axis
        valid_mask:  (N,) bool — True where point is in front of camera
    """
    w2c = np.linalg.inv(c2w_opencv)
    ones = np.ones((len(points_world), 1), dtype=np.float32)
    pts_h = np.concatenate([points_world, ones], axis=1)  # (N, 4)
    pts_cam = (w2c @ pts_h.T).T[:, :3]                    # (N, 3)

    valid = pts_cam[:, 2] > 0.0
    z = np.where(valid, pts_cam[:, 2], 1.0)
    u = pts_cam[:, 0] / z * intrinsics.fx + intrinsics.cx
    v = pts_cam[:, 1] / z * intrinsics.fy + intrinsics.cy

    W, H = intrinsics.image_size
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_mask = valid & in_bounds

    return np.stack([u, v], axis=1).astype(np.float32), pts_cam[:, 2].astype(np.float32), valid_mask
