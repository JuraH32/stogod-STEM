#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def next_plot_path(output_dir, stem):
    # Avoid overwriting by selecting the next available plot filename.
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = output_dir / f"{stem}_plot_{index:03d}.png"
        if not candidate.exists():
            return candidate
    raise SystemExit("Could not find a free plot filename")


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP sparse reconstruction then visualize in one command."
    )
    parser.add_argument("--images-dir", required=True, help="Folder with input images")
    parser.add_argument("--workspace", required=True, help="Working directory for COLMAP")
    parser.add_argument("--k-file", required=True, help="Path to K.txt intrinsics file")
    parser.add_argument("--export-csv", default="colmap_points.csv", help="CSV output file")
    parser.add_argument("--output-dir", default="plots", help="Folder for saved plots")

    parser.add_argument("--colmap-bin", default="colmap", help="COLMAP binary")
    parser.add_argument("--camera-model", default="PINHOLE", help="COLMAP camera model")
    parser.add_argument("--camera-params", default=None, help="Override fx,fy,cx,cy")
    parser.add_argument("--single-camera", type=int, default=1, help="Use one camera model")
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="Matching strategy",
    )
    parser.add_argument("--sequential-overlap", type=int, default=5, help="Sequential overlap")
    parser.add_argument("--use-gpu", type=int, default=0, help="Enable GPU (1/0)")
    parser.add_argument("--image-exts", default="jpg,jpeg,png,tif,tiff,bmp")
    parser.add_argument("--max-num-features", type=int, default=0)
    parser.add_argument("--max-num-matches", type=int, default=32768)
    parser.add_argument("--max-image-size", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--export-type", default="TXT", choices=["TXT", "PLY"])
    parser.add_argument("--max-error", type=float, default=None)
    parser.add_argument("--min-track-len", type=int, default=None)

    parser.add_argument("--viz-max-error", type=float, default=None)
    parser.add_argument("--viz-color-by-error", action="store_true")
    parser.add_argument("--viz-color-field", default=None)
    parser.add_argument("--viz-color-rgb", action="store_true")
    parser.add_argument("--viz-color-label", default=None)
    parser.add_argument("--viz-zoom", type=float, default=1.2)
    parser.add_argument("--viz-pad", type=float, default=0.02)
    parser.add_argument("--viz-point-size", type=float, default=10.0)
    parser.add_argument("--viz-figsize", default="12,9")
    parser.add_argument("--viz-dpi", type=int, default=160)
    parser.add_argument("--viz-elev", type=float, default=None)
    parser.add_argument("--viz-azim", type=float, default=None)
    args = parser.parse_args()

    colmap_cmd = [
        sys.executable,
        "pointcloud_colmap.py",
        "--images-dir",
        args.images_dir,
        "--workspace",
        args.workspace,
        "--k-file",
        args.k_file,
        "--export-csv",
        args.export_csv,
        "--colmap-bin",
        args.colmap_bin,
        "--camera-model",
        args.camera_model,
        "--single-camera",
        str(args.single_camera),
        "--matcher",
        args.matcher,
        "--sequential-overlap",
        str(args.sequential_overlap),
        "--use-gpu",
        str(args.use_gpu),
        "--image-exts",
        args.image_exts,
        "--max-num-matches",
        str(args.max_num_matches),
        "--export-type",
        args.export_type,
    ]
    if args.camera_params:
        colmap_cmd.extend(["--camera-params", args.camera_params])
    if args.max_num_features > 0:
        colmap_cmd.extend(["--max-num-features", str(args.max_num_features)])
    if args.max_image_size > 0:
        colmap_cmd.extend(["--max-image-size", str(args.max_image_size)])
    if args.num_threads > 0:
        colmap_cmd.extend(["--num-threads", str(args.num_threads)])
    if args.max_error is not None:
        colmap_cmd.extend(["--max-error", str(args.max_error)])
    if args.min_track_len is not None:
        colmap_cmd.extend(["--min-track-len", str(args.min_track_len)])

    subprocess.run(colmap_cmd, check=True)

    plot_dir = Path(args.output_dir)
    plot_path = next_plot_path(plot_dir, Path(args.export_csv).stem)

    viz_cmd = [
        sys.executable,
        "pointcloud_visualize.py",
        "--input",
        args.export_csv,
        "--output",
        str(plot_path),
        "--show",
        "--zoom",
        str(args.viz_zoom),
        "--pad",
        str(args.viz_pad),
        "--point-size",
        str(args.viz_point_size),
        "--figsize",
        args.viz_figsize,
        "--dpi",
        str(args.viz_dpi),
    ]
    if args.viz_color_by_error:
        viz_cmd.append("--color-by-error")
    if args.viz_color_rgb:
        viz_cmd.append("--color-rgb")
    if args.viz_color_field:
        viz_cmd.extend(["--color-field", args.viz_color_field])
    if args.viz_color_label:
        viz_cmd.extend(["--color-label", args.viz_color_label])
    if args.viz_max_error is not None:
        viz_cmd.extend(["--max-error", str(args.viz_max_error)])
    if args.viz_elev is not None:
        viz_cmd.extend(["--elev", str(args.viz_elev)])
    if args.viz_azim is not None:
        viz_cmd.extend(["--azim", str(args.viz_azim)])

    subprocess.run(viz_cmd, check=True)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
