#!/usr/bin/env python3
import argparse
import csv
import os
import re
from itertools import combinations

import cv2
import numpy as np

from pointcloud_utils import parse_box_input, ray_from_pixel, closest_point_to_rays, line_distance


def sift_or_die():
    if not hasattr(cv2, "SIFT_create"):
        raise SystemExit(
            "SIFT not available. Install opencv-contrib-python and retry."
        )
    return cv2.SIFT_create()


def load_gray_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Failed to read image: {path}")
    return image


def load_color_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sample_color(image, row, col):
    height, width = image.shape[:2]
    r = int(round(row))
    c = int(round(col))
    r = max(0, min(height - 1, r))
    c = max(0, min(width - 1, c))
    return image[r, c]


def ratio_matches(matches, ratio):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


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


def main():
    parser = argparse.ArgumentParser(description="Automatic SIFT-based 3D points for a dataset.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--camera-input", help="Path to camera input text file")
    input_group.add_argument("--box-input", help="Path to camera input text file")
    parser.add_argument("--images-dir", required=True, help="Folder with images")
    parser.add_argument(
        "--images",
        nargs="+",
        type=int,
        default=None,
        help="Image indices to use (default: auto-detect)",
    )
    parser.add_argument("--image-prefix", default=None, help="Filename prefix, e.g. box or entrance")
    parser.add_argument("--image-ext", default="png", help="Image extension without dot")
    parser.add_argument("--res-x", type=int, default=1920, help="Image width in pixels")
    parser.add_argument("--res-y", type=int, default=1080, help="Image height in pixels")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio for SIFT matching")
    parser.add_argument("--max-error", type=float, default=8.0, help="Reject points above this avg error")
    parser.add_argument("--min-angle-deg", type=float, default=1.0, help="Reject near-parallel rays")
    parser.add_argument(
        "--detail",
        type=float,
        default=1.0,
        help="Density multiplier (>1 yields more points, may add noise)",
    )
    parser.add_argument(
        "--max-matches-per-pair",
        type=int,
        default=0,
        help="Limit matches per image pair (0 = no limit)",
    )
    parser.add_argument("--write-rgb", action="store_true", help="Append r,g,b columns")
    parser.add_argument(
        "--rgb-source",
        choices=["img_a", "img_b", "avg"],
        default="img_a",
        help="Which image to sample for r,g,b",
    )
    parser.add_argument("--output", default="auto_points.csv", help="Output CSV file")
    args = parser.parse_args()

    camera_input = args.camera_input or args.box_input
    if not camera_input:
        raise SystemExit("Missing --camera-input")

    prefix, detected_indices = detect_series(
        args.images_dir, args.image_prefix, args.image_ext
    )
    if args.images is None:
        image_ids = detected_indices
    else:
        image_ids = args.images

    # Adjust thresholds based on requested density.
    detail = max(0.1, args.detail)
    ratio = min(0.95, args.ratio + 0.04 * (detail - 1.0))
    max_error = args.max_error * detail
    min_angle_deg = args.min_angle_deg / detail
    max_matches_per_pair = args.max_matches_per_pair
    if max_matches_per_pair > 0:
        max_matches_per_pair = max(1, int(round(max_matches_per_pair * detail)))

    cameras = parse_box_input(camera_input)
    for image_id in image_ids:
        if image_id not in cameras:
            raise SystemExit(f"Missing camera data for image {image_id}")

    image_paths = [
        os.path.join(args.images_dir, f"{prefix}{image_id}.{args.image_ext}")
        for image_id in image_ids
    ]
    images = [load_gray_image(path) for path in image_paths]
    color_images = [load_color_image(path) for path in image_paths] if args.write_rgb else None

    sift = sift_or_die()
    keypoints = []
    descriptors = []
    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(des)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    point_id = 1
    total_pairs = 0
    total_matches = 0

    for idx_a, idx_b in combinations(range(len(image_ids)), 2):
        # Each match from an image pair yields one 3D point.
        des_a = descriptors[idx_a]
        des_b = descriptors[idx_b]
        if des_a is None or des_b is None:
            continue

        matches = matcher.knnMatch(des_a, des_b, k=2)
        good = ratio_matches(matches, ratio)
        total_pairs += 1
        total_matches += len(good)

        if max_matches_per_pair > 0:
            good = good[: max_matches_per_pair]

        image_id_a = image_ids[idx_a]
        image_id_b = image_ids[idx_b]

        for match in good:
            kp_a = keypoints[idx_a][match.queryIdx]
            kp_b = keypoints[idx_b][match.trainIdx]

            col_a, row_a = kp_a.pt
            col_b, row_b = kp_b.pt

            origin_a, direction_a = ray_from_pixel(
                row_a, col_a, args.res_x, args.res_y, cameras[image_id_a]
            )
            origin_b, direction_b = ray_from_pixel(
                row_b, col_b, args.res_x, args.res_y, cameras[image_id_b]
            )

            dot = float(np.dot(direction_a, direction_b))
            dot = max(-1.0, min(1.0, dot))
            angle_deg = float(np.degrees(np.arccos(dot)))
            if angle_deg < min_angle_deg:
                continue

            point = closest_point_to_rays([origin_a, origin_b], [direction_a, direction_b])
            errors = [
                line_distance(point, origin_a, direction_a),
                line_distance(point, origin_b, direction_b),
            ]
            avg_error = float(np.mean(errors))

            if avg_error > max_error:
                continue

            row = [
                point_id,
                point[0],
                point[1],
                point[2],
                avg_error,
                image_id_a,
                image_id_b,
                row_a,
                col_a,
                row_b,
                col_b,
                angle_deg,
            ]

            if args.write_rgb:
                color_a = sample_color(color_images[idx_a], row_a, col_a)
                color_b = sample_color(color_images[idx_b], row_b, col_b)
                if args.rgb_source == "img_b":
                    color = color_b
                elif args.rgb_source == "avg":
                    color = (color_a.astype(float) + color_b.astype(float)) / 2.0
                else:
                    color = color_a
                r, g, b = [int(max(0, min(255, round(float(v))))) for v in color]
                row.extend([r, g, b])

            rows.append(row)
            point_id += 1

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "point_id",
            "x",
            "y",
            "z",
            "avg_error",
            "img_a",
            "img_b",
            "img_a_row",
            "img_a_col",
            "img_b_row",
            "img_b_col",
            "angle_deg",
        ]
        if args.write_rgb:
            header.extend(["r", "g", "b"])
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Processed {total_pairs} image pairs")
    print(f"Initial matches: {total_matches}, kept points: {len(rows)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
