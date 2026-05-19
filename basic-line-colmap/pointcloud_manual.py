#!/usr/bin/env python3
import argparse
import csv
import numpy as np

from pointcloud_utils import parse_box_input, ray_from_pixel, closest_point_to_rays, line_distance


def prompt_point(image_ids, point_idx):
    # Collect one pixel coordinate per image for the same real-world point.
    pixels = []
    for image_id in image_ids:
        while True:
            raw = input(f"Point {point_idx} for image {image_id} (row col): ").strip()
            if not raw:
                continue
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print("Enter two numbers: row col")
                continue
            try:
                row = float(parts[0])
                col = float(parts[1])
                pixels.append((row, col))
                break
            except ValueError:
                print("Numbers only")
    return pixels


def main():
    parser = argparse.ArgumentParser(description="Manual 3D point estimation from pixel picks.")
    parser.add_argument("--box-input", required=True, help="Path to boxInput.txt")
    parser.add_argument("--images", nargs=3, type=int, default=[1, 2, 3], help="Image indices to use")
    parser.add_argument("--points", type=int, default=4, help="How many points to enter")
    parser.add_argument("--res-x", type=int, default=1920, help="Image width in pixels")
    parser.add_argument("--res-y", type=int, default=1080, help="Image height in pixels")
    parser.add_argument("--output", default="manual_points.csv", help="Output CSV file")
    args = parser.parse_args()

    cameras = parse_box_input(args.box_input)
    for image_id in args.images:
        if image_id not in cameras:
            raise SystemExit(f"Missing camera data for image {image_id}")

    rows = []
    for point_idx in range(1, args.points + 1):
        # Triangulate each user-selected point from multiple views.
        pixels = prompt_point(args.images, point_idx)
        origins = []
        directions = []
        for (row, col), image_id in zip(pixels, args.images):
            origin, direction = ray_from_pixel(
                row, col, args.res_x, args.res_y, cameras[image_id]
            )
            origins.append(origin)
            directions.append(direction)

        point = closest_point_to_rays(origins, directions)
        errors = [line_distance(point, o, d) for o, d in zip(origins, directions)]
        avg_error = float(np.mean(errors))

        rows.append([point_idx, point[0], point[1], point[2], avg_error])
        print(
            f"Point {point_idx}: X={point[0]:.3f} Y={point[1]:.3f} Z={point[2]:.3f} avg_err={avg_error:.3f}"
        )

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "x", "y", "z", "avg_error"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} points to {args.output}")


if __name__ == "__main__":
    main()
