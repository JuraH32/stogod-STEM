#!/usr/bin/env python3
import argparse
import csv
import os
import re

import cv2
import numpy as np

from pointcloud_utils import (
    camera_rotation_world_to_cam,
    intrinsics_from_fov,
    parse_box_input,
    stereo_relative_pose,
)


def detect_series(images_dir, image_prefix, image_ext):
    # Infer prefix and indices from filenames like box1.png, entrance12.png.
    if image_prefix:
        pattern = re.compile(rf"^{re.escape(image_prefix)}(\d+)\.{re.escape(image_ext)}$")
        prefixes = {image_prefix}
    else:
        pattern = re.compile(rf"^(.*?)(\d+)\.{re.escape(image_ext)}$")
        prefixes = set()

    indices = []
    for name in os.listdir(images_dir):
        match = pattern.match(name)
        if not match:
            continue
        prefix = match.group(1)
        index = int(match.group(2))
        indices.append(index)
        prefixes.add(prefix)

    if not indices:
        raise SystemExit(f"No images found in {images_dir} with extension .{image_ext}")
    if not image_prefix:
        if len(prefixes) != 1:
            raise SystemExit(
                "Multiple image prefixes detected. Provide --image-prefix explicitly."
            )
        image_prefix = next(iter(prefixes))

    return image_prefix, sorted(indices)


def load_gray_image(path, downscale=1, clahe=None):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Failed to read image: {path}")
    if clahe is not None:
        image = clahe.apply(image)
    if downscale > 1:
        width = int(image.shape[1] / downscale)
        height = int(image.shape[0] / downscale)
        if width < 1 or height < 1:
            raise SystemExit("Downscale too large for image size")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def ensure_num_disparities(value):
    value = int(value)
    if value <= 0:
        value = 16
    return int(np.ceil(value / 16.0) * 16)


def build_sgbm(args):
    num_disp = ensure_num_disparities(args.num_disparities)
    block = args.block_size
    if block % 2 == 0:
        block += 1
    p1 = 8 * block * block
    p2 = 32 * block * block
    return cv2.StereoSGBM_create(
        minDisparity=args.min_disparity,
        numDisparities=num_disp,
        blockSize=block,
        P1=p1,
        P2=p2,
        disp12MaxDiff=args.disp12_max_diff,
        preFilterCap=args.pre_filter_cap,
        uniquenessRatio=args.uniqueness_ratio,
        speckleWindowSize=args.speckle_window_size,
        speckleRange=args.speckle_range,
        mode=args.sgbm_mode,
    )


def compute_disparity(left, right, matcher, args):
    if args.use_wls and hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        disp_left = matcher.compute(left, right)
        disp_right = right_matcher.compute(right, left)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(args.wls_lambda)
        wls.setSigmaColor(args.wls_sigma)
        disp = wls.filter(disp_left, left, None, disp_right)
    else:
        if args.use_wls:
            print("WLS requested but cv2.ximgproc not available, skipping WLS.")
        disp = matcher.compute(left, right)
    return disp.astype(np.float32) / 16.0


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0.0:
        return points, None
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[unique_idx], unique_idx


def pair_indices(count, pair_window, pair_step, max_pairs):
    pairs = []
    for i in range(0, count - 1, pair_step):
        upper = min(i + 1 + pair_window, count)
        for j in range(i + 1, upper):
            pairs.append((i, j))
            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Dense stereo AutoV2 point cloud builder.")
    parser.add_argument("--camera-input", required=True, help="Path to camera input text file")
    parser.add_argument("--images-dir", required=True, help="Folder with images")
    parser.add_argument("--images", nargs="+", type=int, default=None, help="Image indices")
    parser.add_argument("--image-prefix", default=None, help="Filename prefix, e.g. box or entrance")
    parser.add_argument("--image-ext", default="png", help="Image extension without dot")
    parser.add_argument("--res-x", type=int, default=1920, help="Image width in pixels")
    parser.add_argument("--res-y", type=int, default=1080, help="Image height in pixels")
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Horizontal field of view")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale factor for speed")
    parser.add_argument("--pair-window", type=int, default=1, help="How many forward neighbors to pair")
    parser.add_argument("--pair-step", type=int, default=1, help="Step between base images")
    parser.add_argument("--max-pairs", type=int, default=0, help="Max number of pairs (0 = no limit)")
    parser.add_argument("--output", default="auto_v2_points.csv", help="Output CSV file")

    parser.add_argument("--min-disparity", type=int, default=0, help="SGBM minDisparity")
    parser.add_argument("--num-disparities", type=int, default=160, help="SGBM numDisparities")
    parser.add_argument("--block-size", type=int, default=5, help="SGBM blockSize (odd)")
    parser.add_argument("--pre-filter-cap", type=int, default=31, help="SGBM preFilterCap")
    parser.add_argument("--uniqueness-ratio", type=int, default=10, help="SGBM uniquenessRatio")
    parser.add_argument("--speckle-window-size", type=int, default=100, help="SGBM speckleWindowSize")
    parser.add_argument("--speckle-range", type=int, default=2, help="SGBM speckleRange")
    parser.add_argument("--disp12-max-diff", type=int, default=1, help="SGBM disp12MaxDiff")
    parser.add_argument(
        "--sgbm-mode",
        type=int,
        default=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        help="SGBM mode value",
    )
    parser.add_argument("--use-wls", action="store_true", help="Enable WLS disparity filter")
    parser.add_argument("--wls-lambda", type=float, default=8000.0, help="WLS lambda")
    parser.add_argument("--wls-sigma", type=float, default=1.5, help="WLS sigmaColor")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE before stereo")

    parser.add_argument("--stride", type=int, default=2, help="Pixel stride for sampling")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Min depth in camera space")
    parser.add_argument("--max-depth", type=float, default=2000.0, help="Max depth in camera space")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel size for downsampling")
    parser.add_argument(
        "--max-points-per-pair",
        type=int,
        default=0,
        help="Cap points per pair (0 = no limit)",
    )
    parser.add_argument(
        "--max-points-total",
        type=int,
        default=0,
        help="Cap total points written (0 = no limit)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    prefix, detected_indices = detect_series(
        args.images_dir, args.image_prefix, args.image_ext
    )
    image_ids = detected_indices if args.images is None else args.images

    cameras = parse_box_input(args.camera_input)
    for image_id in image_ids:
        if image_id not in cameras:
            raise SystemExit(f"Missing camera data for image {image_id}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    image_paths = [
        os.path.join(args.images_dir, f"{prefix}{image_id}.{args.image_ext}")
        for image_id in image_ids
    ]
    images = [load_gray_image(path, args.downscale, clahe) for path in image_paths]

    height, width = images[0].shape
    if (args.res_x, args.res_y) != (width, height):
        scale_x = width / float(args.res_x)
        scale_y = height / float(args.res_y)
    else:
        scale_x = 1.0
        scale_y = 1.0

    k = intrinsics_from_fov(args.res_x, args.res_y, args.fov_deg)
    k[0, 0] *= scale_x
    k[1, 1] *= scale_y
    k[0, 2] *= scale_x
    k[1, 2] *= scale_y
    dist = np.zeros((5, 1), dtype=float)

    matcher = build_sgbm(args)
    pairs = pair_indices(len(image_ids), args.pair_window, args.pair_step, args.max_pairs)

    rng = np.random.default_rng(args.seed)
    total_written = 0
    point_id = 1

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "point_id",
                "x",
                "y",
                "z",
                "avg_error",
                "img_a",
                "img_b",
                "disparity",
            ]
        )

        for idx_a, idx_b in pairs:
            image_id_a = image_ids[idx_a]
            image_id_b = image_ids[idx_b]

            cam_a = cameras[image_id_a]
            cam_b = cameras[image_id_b]

            r, t = stereo_relative_pose(cam_a, cam_b)
            r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
                k, dist, k, dist, (width, height), r, t, flags=cv2.CALIB_ZERO_DISPARITY
            )

            map1x, map1y = cv2.initUndistortRectifyMap(
                k, dist, r1, p1, (width, height), cv2.CV_32FC1
            )
            map2x, map2y = cv2.initUndistortRectifyMap(
                k, dist, r2, p2, (width, height), cv2.CV_32FC1
            )

            left_rect = cv2.remap(images[idx_a], map1x, map1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(images[idx_b], map2x, map2y, cv2.INTER_LINEAR)

            disparity = compute_disparity(left_rect, right_rect, matcher, args)
            valid = disparity > float(args.min_disparity)

            if args.stride > 1:
                disparity = disparity[:: args.stride, :: args.stride]
                valid = valid[:: args.stride, :: args.stride]

            points = cv2.reprojectImageTo3D(disparity, q)

            points = points.reshape(-1, 3)
            disp_flat = disparity.reshape(-1)
            valid = valid.reshape(-1)

            finite = np.isfinite(points).all(axis=1)
            valid &= finite
            points = points[valid]
            disp_flat = disp_flat[valid]

            r_cw_a, r_wc_a = camera_rotation_world_to_cam(cam_a)
            c_a = cam_a["position"]
            pc1 = points @ r1.T
            depth = pc1[:, 2]
            depth_mask = (depth > args.min_depth) & (depth < args.max_depth)
            points = points[depth_mask]
            disp_flat = disp_flat[depth_mask]

            m = r_wc_a @ r1.T
            world_points = points @ m.T + c_a

            if args.voxel_size > 0.0:
                world_points, voxel_idx = voxel_downsample(world_points, args.voxel_size)
                if voxel_idx is not None:
                    disp_flat = disp_flat[voxel_idx]

            if args.max_points_per_pair > 0 and world_points.shape[0] > args.max_points_per_pair:
                pick = rng.choice(
                    world_points.shape[0],
                    size=args.max_points_per_pair,
                    replace=False,
                )
                world_points = world_points[pick]
                disp_flat = disp_flat[pick]

            remaining = args.max_points_total - total_written
            if args.max_points_total > 0 and remaining <= 0:
                break
            if args.max_points_total > 0 and world_points.shape[0] > remaining:
                world_points = world_points[:remaining]
                disp_flat = disp_flat[:remaining]

            for point, disp in zip(world_points, disp_flat):
                writer.writerow(
                    [
                        point_id,
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        0.0,
                        image_id_a,
                        image_id_b,
                        float(disp),
                    ]
                )
                point_id += 1

            total_written += world_points.shape[0]

    print(f"Pairs processed: {len(pairs)}")
    print(f"Points written: {total_written}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
