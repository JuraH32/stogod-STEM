#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def load_points(path, color_field=None):
    xs = []
    ys = []
    zs = []
    errors = []
    colors = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            errors.append(float(row.get("avg_error", 0.0)))
            if color_field:
                colors.append(float(row.get(color_field, 0.0)))
    return (
        np.array(xs),
        np.array(ys),
        np.array(zs),
        np.array(errors),
        np.array(colors) if color_field else None,
    )


def set_equal_axes(ax, xs, ys, zs, zoom=1.0, pad=0.02):
    # Keep equal axis scale so shapes are not visually distorted.
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    if max_range == 0.0:
        max_range = 1.0
    if pad < 0.0:
        pad = 0.0
    if zoom <= 0.0:
        zoom = 1.0

    max_range *= 1.0 + 2.0 * pad

    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    half = (max_range / 2.0) / zoom
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    ax.set_box_aspect((1, 1, 1))


def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must be in W,H format")
    try:
        width = float(parts[0])
        height = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("figsize must be numbers") from exc
    if width <= 0.0 or height <= 0.0:
        raise argparse.ArgumentTypeError("figsize values must be > 0")
    return (width, height)


def main():
    parser = argparse.ArgumentParser(description="3D scatter visualization for point clouds.")
    parser.add_argument("--input", default="auto_points.csv", help="CSV file from pointcloud_auto.py")
    parser.add_argument("--max-error", type=float, default=None, help="Filter points by avg_error")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N points")
    parser.add_argument("--color-by-error", action="store_true", help="Color points by avg_error")
    parser.add_argument("--color-field", default=None, help="Column to color by")
    parser.add_argument("--color-label", default=None, help="Label for colorbar")
    parser.add_argument("--output", default=None, help="Save plot to PNG")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    parser.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(12.0, 9.0),
        help="Figure size in inches (W,H), e.g. 12,9",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.2,
        help="Zoom factor for axis limits (>1 zooms in)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.02,
        help="Padding around bounds as a fraction of range",
    )
    parser.add_argument("--point-size", type=float, default=10.0, help="Marker size")
    parser.add_argument("--dpi", type=int, default=160, help="Figure/output DPI")
    parser.add_argument("--elev", type=float, default=None, help="Initial elevation angle")
    parser.add_argument("--azim", type=float, default=None, help="Initial azimuth angle")
    args = parser.parse_args()

    color_field = args.color_field
    if args.color_by_error:
        color_field = "avg_error"
    xs, ys, zs, errors, colors = load_points(args.input, color_field)

    if xs.size == 0:
        raise SystemExit("No points found in CSV")

    mask = np.ones_like(errors, dtype=bool)
    if args.max_error is not None:
        # Filter noisy points before plotting.
        mask &= errors <= args.max_error

    xs, ys, zs, errors = xs[mask], ys[mask], zs[mask], errors[mask]
    if colors is not None:
        colors = colors[mask]

    if args.sample is not None and args.sample < xs.size:
        rng = np.random.default_rng(42)
        idx = rng.choice(xs.size, size=args.sample, replace=False)
        xs, ys, zs, errors = xs[idx], ys[idx], zs[idx], errors[idx]
        if colors is not None:
            colors = colors[idx]

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib not installed. Install with: python -m pip install matplotlib"
        ) from exc

    fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    if colors is not None:
        scatter = ax.scatter(
            xs, ys, zs, c=colors, s=args.point_size, cmap="viridis", alpha=0.9
        )
        label = args.color_label or color_field
        fig.colorbar(scatter, ax=ax, shrink=0.6, label=label)
    else:
        ax.scatter(xs, ys, zs, s=args.point_size, alpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, xs, ys, zs, zoom=args.zoom, pad=args.pad)

    if args.elev is not None or args.azim is not None:
        elev = args.elev if args.elev is not None else ax.elev
        azim = args.azim if args.azim is not None else ax.azim
        ax.view_init(elev=elev, azim=azim)
    ax.set_title(Path(args.input).name)

    if args.output:
        plt.tight_layout()
        plt.savefig(args.output, dpi=args.dpi)
        print(f"Saved plot to {args.output}")

    if args.show or not args.output:
        plt.show()


if __name__ == "__main__":
    main()
