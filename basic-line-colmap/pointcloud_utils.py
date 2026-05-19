#!/usr/bin/env python3
import re
import numpy as np

# Shared helpers for parsing camera text and ray geometry.


def parse_xyz(line):
    def find(name):
        match = re.search(rf"{name}\s*=\s*([-\d.]+)", line)
        if not match:
            raise ValueError(f"Missing {name} in line: {line}")
        return float(match.group(1))

    return np.array([find("X"), find("Y"), find("Z")], dtype=float)


def unit(vec):
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def parse_box_input(path):
    cameras = {}
    # Input format groups fields under numbered headers like "1)".
    current = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            match = re.match(r"(\d+)\)", line)
            if match:
                current = int(match.group(1))
                cameras[current] = {}
                continue
            if current is None:
                continue

            if line.startswith("CamPosition"):
                cameras[current]["position"] = parse_xyz(line)
            elif line.startswith("CamForward"):
                cameras[current]["forward"] = parse_xyz(line)
            elif line.startswith("CamRight") or "CamRight" in line:
                cameras[current]["right"] = parse_xyz(line)
            elif line.startswith("CamUp") or "CamUp" in line:
                cameras[current]["up"] = parse_xyz(line)

    return cameras


def intrinsics_from_fov(res_x, res_y, fov_deg=90.0):
    # Horizontal FOV is 90 deg. Match the ray construction used in the task statement.
    f = (res_x / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = res_x / 2.0 - 0.5
    cy = res_y / 2.0 - 0.5
    return np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def camera_basis(cam):
    # Camera basis vectors in world coordinates.
    right = unit(cam["right"])
    up = unit(cam["up"])
    forward = unit(cam["forward"])
    return right, up, forward


def camera_rotation_world_to_cam(cam):
    # World-to-camera rotation from right/up/forward basis.
    right, up, forward = camera_basis(cam)
    r_world = np.column_stack((right, up, forward))
    r_cam = r_world.T
    return r_cam, r_world


def stereo_relative_pose(cam_a, cam_b):
    # Return R, T so that X_b = R * X_a + T (camera coordinates).
    r_cw_a, r_wc_a = camera_rotation_world_to_cam(cam_a)
    r_cw_b, _ = camera_rotation_world_to_cam(cam_b)
    c_a = cam_a["position"]
    c_b = cam_b["position"]
    r = r_cw_b @ r_wc_a
    t = r_cw_b @ (c_a - c_b)
    return r, t


def ray_from_pixel(row, col, res_x, res_y, cam):
    # Pinhole projection with 90 deg FOV per task statement.
    coeff_right = 2.0 * (col - res_x / 2.0 + 0.5) / res_x
    coeff_up = -2.0 * (row - res_y / 2.0 + 0.5) / res_y
    coeff_up = coeff_up * (res_y / res_x)
    direction = cam["forward"] + coeff_right * cam["right"] + coeff_up * cam["up"]
    direction = direction / np.linalg.norm(direction)
    origin = cam["position"]
    return origin, direction


def closest_point_to_rays(origins, directions):
    # Least-squares point minimizing distance to all rays.
    a_mat = np.zeros((3, 3), dtype=float)
    b_vec = np.zeros(3, dtype=float)
    eye = np.eye(3)

    for origin, direction in zip(origins, directions):
        direction = direction / np.linalg.norm(direction)
        proj = eye - np.outer(direction, direction)
        a_mat += proj
        b_vec += proj @ origin

    point, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    return point


def line_distance(point, origin, direction):
    # Distance from a point to an infinite line (origin + direction).
    direction = direction / np.linalg.norm(direction)
    vec = point - origin
    return np.linalg.norm(vec - direction * np.dot(vec, direction))
