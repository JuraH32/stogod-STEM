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
        description="Run auto triangulation then visualize in one command."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "v2"],
        default="auto",
        help="Pipeline mode: sparse auto or dense v2",
    )
    parser.add_argument("--camera-input", required=True, help="Path to camera input text file")
    parser.add_argument("--images-dir", required=True, help="Folder with images")
    parser.add_argument("--output-csv", default="auto_points.csv", help="Output CSV file")
    parser.add_argument("--output-dir", default=".", help="Folder for saved plots")

    parser.add_argument("--images", nargs="+", type=int, default=None, help="Image indices")
    parser.add_argument("--image-prefix", default=None, help="Filename prefix, e.g. box or entrance")
    parser.add_argument("--image-ext", default="png", help="Image extension without dot")
    parser.add_argument("--res-x", type=int, default=1920, help="Image width in pixels")
    parser.add_argument("--res-y", type=int, default=1080, help="Image height in pixels")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio for SIFT matching")
    parser.add_argument("--max-error", type=float, default=2.0, help="Reject points above this avg error")
    parser.add_argument("--min-angle-deg", type=float, default=2.0, help="Reject near-parallel rays")
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

    parser.add_argument("--v2-fov-deg", type=float, default=90.0, help="V2 horizontal FOV")
    parser.add_argument("--v2-downscale", type=int, default=1, help="V2 downscale factor")
    parser.add_argument("--v2-pair-window", type=int, default=1, help="V2 neighbor window")
    parser.add_argument("--v2-pair-step", type=int, default=1, help="V2 pair step")
    parser.add_argument("--v2-max-pairs", type=int, default=0, help="V2 max pairs (0 = no limit)")
    parser.add_argument("--v2-min-disparity", type=int, default=0, help="V2 minDisparity")
    parser.add_argument("--v2-num-disparities", type=int, default=160, help="V2 numDisparities")
    parser.add_argument("--v2-block-size", type=int, default=5, help="V2 blockSize")
    parser.add_argument("--v2-uniqueness", type=int, default=10, help="V2 uniquenessRatio")
    parser.add_argument("--v2-speckle-window", type=int, default=100, help="V2 speckleWindowSize")
    parser.add_argument("--v2-speckle-range", type=int, default=2, help="V2 speckleRange")
    parser.add_argument("--v2-disp12-max-diff", type=int, default=1, help="V2 disp12MaxDiff")
    parser.add_argument("--v2-pre-filter-cap", type=int, default=31, help="V2 preFilterCap")
    parser.add_argument("--v2-use-wls", action="store_true", help="V2 use WLS filter")
    parser.add_argument("--v2-wls-lambda", type=float, default=8000.0, help="V2 WLS lambda")
    parser.add_argument("--v2-wls-sigma", type=float, default=1.5, help="V2 WLS sigmaColor")
    parser.add_argument("--v2-clahe", action="store_true", help="V2 apply CLAHE")
    parser.add_argument("--v2-stride", type=int, default=2, help="V2 pixel stride")
    parser.add_argument("--v2-min-depth", type=float, default=0.1, help="V2 min depth")
    parser.add_argument("--v2-max-depth", type=float, default=2000.0, help="V2 max depth")
    parser.add_argument("--v2-voxel-size", type=float, default=0.0, help="V2 voxel size")
    parser.add_argument("--v2-max-points-per-pair", type=int, default=0, help="V2 cap per pair")
    parser.add_argument("--v2-max-points-total", type=int, default=0, help="V2 cap total points")

    parser.add_argument("--viz-max-error", type=float, default=2.0, help="Filter points by avg_error")
    parser.add_argument("--viz-color-by-error", action="store_true", help="Color points by avg_error")
    parser.add_argument("--viz-color-field", default=None, help="Color points by CSV column")
    parser.add_argument("--viz-color-label", default=None, help="Colorbar label override")
    parser.add_argument("--viz-zoom", type=float, default=1.2, help="Zoom factor")
    parser.add_argument("--viz-pad", type=float, default=0.02, help="Padding fraction")
    parser.add_argument("--viz-point-size", type=float, default=10.0, help="Marker size")
    parser.add_argument("--viz-figsize", default="12,9", help="Figure size W,H")
    parser.add_argument("--viz-dpi", type=int, default=160, help="Figure/output DPI")
    args = parser.parse_args()

    # Run triangulation to produce the CSV.
    if args.mode == "auto":
        auto_cmd = [
            sys.executable,
            "pointcloud_auto.py",
            "--camera-input",
            args.camera_input,
            "--images-dir",
            args.images_dir,
            "--output",
            args.output_csv,
            "--res-x",
            str(args.res_x),
            "--res-y",
            str(args.res_y),
            "--ratio",
            str(args.ratio),
            "--max-error",
            str(args.max_error),
            "--min-angle-deg",
            str(args.min_angle_deg),
            "--detail",
            str(args.detail),
            "--max-matches-per-pair",
            str(args.max_matches_per_pair),
            "--image-ext",
            args.image_ext,
        ]

        if args.images:
            auto_cmd.append("--images")
            auto_cmd.extend([str(i) for i in args.images])
        if args.image_prefix:
            auto_cmd.extend(["--image-prefix", args.image_prefix])

        subprocess.run(auto_cmd, check=True)
    else:
        v2_cmd = [
            sys.executable,
            "pointcloud_auto_v2.py",
            "--camera-input",
            args.camera_input,
            "--images-dir",
            args.images_dir,
            "--output",
            args.output_csv,
            "--res-x",
            str(args.res_x),
            "--res-y",
            str(args.res_y),
            "--fov-deg",
            str(args.v2_fov_deg),
            "--downscale",
            str(args.v2_downscale),
            "--pair-window",
            str(args.v2_pair_window),
            "--pair-step",
            str(args.v2_pair_step),
            "--max-pairs",
            str(args.v2_max_pairs),
            "--min-disparity",
            str(args.v2_min_disparity),
            "--num-disparities",
            str(args.v2_num_disparities),
            "--block-size",
            str(args.v2_block_size),
            "--uniqueness-ratio",
            str(args.v2_uniqueness),
            "--speckle-window-size",
            str(args.v2_speckle_window),
            "--speckle-range",
            str(args.v2_speckle_range),
            "--disp12-max-diff",
            str(args.v2_disp12_max_diff),
            "--pre-filter-cap",
            str(args.v2_pre_filter_cap),
            "--wls-lambda",
            str(args.v2_wls_lambda),
            "--wls-sigma",
            str(args.v2_wls_sigma),
            "--stride",
            str(args.v2_stride),
            "--min-depth",
            str(args.v2_min_depth),
            "--max-depth",
            str(args.v2_max_depth),
            "--voxel-size",
            str(args.v2_voxel_size),
            "--max-points-per-pair",
            str(args.v2_max_points_per_pair),
            "--max-points-total",
            str(args.v2_max_points_total),
            "--image-ext",
            args.image_ext,
        ]

        if args.images:
            v2_cmd.append("--images")
            v2_cmd.extend([str(i) for i in args.images])
        if args.image_prefix:
            v2_cmd.extend(["--image-prefix", args.image_prefix])
        if args.v2_use_wls:
            v2_cmd.append("--use-wls")
        if args.v2_clahe:
            v2_cmd.append("--clahe")

        subprocess.run(v2_cmd, check=True)

    plot_dir = Path(args.output_dir)
    plot_path = next_plot_path(plot_dir, Path(args.output_csv).stem)

    # Save a plot and open the interactive window.
    viz_cmd = [
        sys.executable,
        "pointcloud_visualize.py",
        "--input",
        args.output_csv,
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
    if args.viz_color_field:
        viz_cmd.extend(["--color-field", args.viz_color_field])
    if args.viz_color_label:
        viz_cmd.extend(["--color-label", args.viz_color_label])
    if args.viz_max_error is not None:
        viz_cmd.extend(["--max-error", str(args.viz_max_error)])

    subprocess.run(viz_cmd, check=True)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
