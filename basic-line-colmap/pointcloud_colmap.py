#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path


def parse_k_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(numbers) < 9:
        raise ValueError("K.txt must contain at least 9 numbers")
    vals = [float(n) for n in numbers[:9]]
    fx = vals[0]
    fy = vals[4]
    cx = vals[2]
    cy = vals[5]
    return fx, fy, cx, cy


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, label):
    print(f"\n[{label}]")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_colmap(colmap_bin):
    if os.path.sep in colmap_bin or (os.path.altsep and os.path.altsep in colmap_bin):
        if os.path.isfile(colmap_bin) and os.access(colmap_bin, os.X_OK):
            return colmap_bin
        raise SystemExit(f"COLMAP binary not found or not executable: {colmap_bin}")

    resolved = shutil.which(colmap_bin)
    if resolved:
        return resolved
    raise SystemExit(
        "COLMAP not found on PATH. Install it or pass --colmap-bin /path/to/colmap"
    )


def export_points_to_csv(points_path, output_csv, max_error, min_track_len):
    rows = []
    with open(points_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            error = float(parts[7])
            track_len = max(0, (len(parts) - 8) // 2)
            if max_error is not None and error > max_error:
                continue
            if min_track_len is not None and track_len < min_track_len:
                continue
            rows.append([point_id, x, y, z, error, track_len])

    if not rows:
        raise SystemExit("No points left after filtering")

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "x", "y", "z", "avg_error", "track_len"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} points to {output_csv}")


def build_filtered_images(images_dir, image_exts, output_dir):
    exts = {
        ext.strip().lower().lstrip(".")
        for ext in image_exts.split(",")
        if ext.strip()
    }
    if not exts:
        raise SystemExit("No image extensions provided")

    paths = []
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower().lstrip(".")
        if suffix in exts:
            paths.append(path)

    if not paths:
        raise SystemExit("No images found for the provided extensions")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(paths, key=lambda p: p.name):
        dst = output_dir / src.name
        target = src.resolve()
        try:
            os.symlink(target, dst)
        except OSError:
            shutil.copy2(target, dst)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run a COLMAP sparse reconstruction with fixed intrinsics."
    )
    parser.add_argument("--images-dir", required=True, help="Folder with input images")
    parser.add_argument("--workspace", required=True, help="Working directory for COLMAP")
    parser.add_argument("--k-file", required=True, help="Path to K.txt intrinsics file")
    parser.add_argument(
        "--colmap-bin",
        default="colmap",
        help="COLMAP binary (default: colmap)",
    )
    parser.add_argument(
        "--image-exts",
        default="jpg,jpeg,png,tif,tiff,bmp",
        help="Comma-separated list of image extensions to include",
    )
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        help="COLMAP camera model (default: PINHOLE)",
    )
    parser.add_argument(
        "--camera-params",
        default=None,
        help="Override camera params as fx,fy,cx,cy (default: read K.txt)",
    )
    parser.add_argument(
        "--single-camera",
        type=int,
        default=1,
        help="Use one shared camera model for all images (1/0)",
    )
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="Matching strategy",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=5,
        help="Sequential matcher overlap (only for sequential)",
    )
    parser.add_argument("--use-gpu", type=int, default=0, help="Enable GPU (1) or CPU (0)")
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=0,
        help="Max SIFT features per image (0 = COLMAP default)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=0,
        help="Max image size for SIFT extraction (0 = COLMAP default)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Number of extraction threads (0 = COLMAP default)",
    )
    parser.add_argument(
        "--max-num-matches",
        type=int,
        default=32768,
        help="Max matches per image pair (must be > 0 on some COLMAP builds)",
    )
    parser.add_argument(
        "--export-type",
        choices=["TXT", "PLY"],
        default="TXT",
        help="Export sparse model type",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Optional CSV export for pointcloud_visualize.py",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=None,
        help="Max reprojection error for CSV filtering",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=None,
        help="Min observation count for CSV filtering",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    workspace = Path(args.workspace)
    db_path = workspace / "database.db"
    sparse_path = workspace / "sparse"
    export_path = workspace / "export"

    ensure_dir(workspace)
    ensure_dir(sparse_path)
    ensure_dir(export_path)

    if args.camera_params:
        camera_params = args.camera_params
    else:
        fx, fy, cx, cy = parse_k_file(args.k_file)
        camera_params = f"{fx},{fy},{cx},{cy}"

    colmap_bin = resolve_colmap(args.colmap_bin)

    filtered_images = build_filtered_images(
        images_dir, args.image_exts, workspace / "images_filtered"
    )

    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(filtered_images),
        "--ImageReader.camera_model",
        args.camera_model,
        "--ImageReader.camera_params",
        camera_params,
        "--ImageReader.single_camera",
        str(args.single_camera),
        "--SiftExtraction.use_gpu",
        str(args.use_gpu),
    ]
    if args.max_num_features > 0:
        feature_cmd.extend(["--SiftExtraction.max_num_features", str(args.max_num_features)])
    if args.max_image_size > 0:
        feature_cmd.extend(["--SiftExtraction.max_image_size", str(args.max_image_size)])
    if args.num_threads > 0:
        feature_cmd.extend(["--SiftExtraction.num_threads", str(args.num_threads)])

    if args.matcher == "sequential":
        match_cmd = [
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            str(args.use_gpu),
            "--SequentialMatching.overlap",
            str(args.sequential_overlap),
        ]
        if args.max_num_matches > 0:
            match_cmd.extend(["--SiftMatching.max_num_matches", str(args.max_num_matches)])
    else:
        match_cmd = [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            str(args.use_gpu),
        ]
        if args.max_num_matches > 0:
            match_cmd.extend(["--SiftMatching.max_num_matches", str(args.max_num_matches)])

    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(filtered_images),
        "--output_path",
        str(sparse_path),
    ]

    run_cmd(feature_cmd, "feature_extractor")
    run_cmd(match_cmd, "matcher")
    run_cmd(mapper_cmd, "mapper")

    model_dirs = [p for p in sparse_path.iterdir() if p.is_dir()]
    if not model_dirs:
        raise SystemExit("No sparse model found. Check COLMAP output.")
    model_dir = sorted(model_dirs)[0]

    converter_cmd = [
        colmap_bin,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(export_path),
        "--output_type",
        args.export_type,
    ]
    run_cmd(converter_cmd, "model_converter")

    if args.export_csv:
        points_path = export_path / "points3D.txt"
        if not points_path.exists():
            raise SystemExit("points3D.txt not found. Use --export-type TXT.")
        export_points_to_csv(points_path, args.export_csv, args.max_error, args.min_track_len)


if __name__ == "__main__":
    main()
