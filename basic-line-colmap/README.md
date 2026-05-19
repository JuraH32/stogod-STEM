# STEM Games 2026 - Point Cloud Reconstruction

## Goal

Estimate 3D point clouds from multiple images of the same scene with known camera poses
(Box and Entrance). Each pixel defines a 3D ray; matching pixels across images lets us
triangulate 3D points.

## Approach

- Convert pixel coordinates into 3D rays using the provided camera pose and 90 deg FOV.
- Match visual features across images with SIFT.
- Triangulate a 3D point as the least-squares closest point to the two rays.
- Visualize the resulting point cloud with a 3D scatter plot.

## Setup (Linux Mint, fish shell)

Recommended: use a virtual environment so SIFT is available.

```bash
sudo apt install -y python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install numpy opencv-contrib-python matplotlib
```

Alternative system packages (SIFT may be missing):

```bash
sudo apt install -y python3-numpy python3-opencv python3-matplotlib
```

### COLMAP (for Fountain/Statue)

Install and verify COLMAP so the `colmap` command is available:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:colmap/colmap
sudo apt update
sudo apt install -y colmap

colmap -h
```

## Quick Start (one command)

Box:

```bash
python3 pointcloud_run.py \
  --camera-input TestImages/Box/boxInput.txt \
  --images-dir TestImages/Box \
  --output-csv auto_points.csv \
  --output-dir plots \
  --viz-color-by-error
```

Entrance with higher density:

```bash
python3 pointcloud_run.py \
  --camera-input TestImages/Entrance/entranceInput.txt \
  --images-dir TestImages/Entrance \
  --output-csv entrance_points.csv \
  --output-dir plots \
  --viz-color-by-error \
  --detail 5
```

The runner:

- generates the CSV,
- saves a unique plot image in the output directory,
- opens the interactive window (pan/zoom).

## Manual MVP (hand-picked points)

Use this when you want to verify a few points by hand.

```bash
python3 pointcloud_manual.py \
  --box-input TestImages/Box/boxInput.txt \
  --images 1 2 3 \
  --points 4 \
  --output manual_points.csv
```

## Auto Triangulation (no visualization)

This uses all detected images in the folder by default.

```bash
python3 pointcloud_auto.py \
  --camera-input TestImages/Box/boxInput.txt \
  --images-dir TestImages/Box \
  --max-error 2.0 \
  --min-angle-deg 2.0 \
  --output auto_points.csv
```

## Visualize an existing CSV

```bash
python3 pointcloud_visualize.py \
  --input auto_points.csv \
  --color-by-error \
  --max-error 2.0 \
  --output plots/auto_points_plot.png \
  --show
```

## Tuning tips

- Increase `--detail` to get more points (also more noise).
- Raise `--ratio` or `--max-error` for density, lower them for precision.
- Increase `--min-angle-deg` to reduce unstable triangulation.
- Use `--viz-max-error` in the runner to filter noisy points in the plot.

## Fountain/Statue (unknown cameras) with COLMAP

Use COLMAP to estimate camera poses from the images, using the provided intrinsics from K.txt.
This produces a sparse point cloud you can export to TXT/PLY and optionally convert to CSV
for visualization in `pointcloud_visualize.py`.

Fountain example:

```bash
python3 pointcloud_colmap_run.py \
  --images-dir TestImages/Fountain \
  --workspace colmap_fountain \
  --k-file TestImages/Fountain/K.txt \
  --matcher sequential \
  --sequential-overlap 5 \
  --use-gpu 0 \
  --viz-color-by-error \
  --viz-max-error 4.0 \
  --export-type TXT \
  --export-csv fountain_points.csv \
  --max-error 4.0 \
  --min-track-len 3
```

If feature extraction gets killed (out of memory), limit image size and threads:

```bash
python3 pointcloud_colmap_run.py \
  --images-dir TestImages/Fountain \
  --workspace colmap_fountain \
  --k-file TestImages/Fountain/K.txt \
  --matcher sequential \
  --sequential-overlap 5 \
  --use-gpu 0 \
  --max-image-size 1600 \
  --num-threads 2 \
  --max-num-features 12000 \
  --viz-color-by-error \
  --viz-max-error 4.0 \
  --export-type TXT \
  --export-csv fountain_points.csv \
  --max-error 4.0 \
  --min-track-len 3
```

If `colmap` is not on PATH, add `--colmap-bin /path/to/colmap` to the command.

Statue example:

```bash
python3 pointcloud_colmap_run.py \
  --images-dir TestImages/Statue \
  --workspace colmap_statue \
  --k-file TestImages/Statue/K.txt \
  --matcher sequential \
  --sequential-overlap 5 \
  --use-gpu 0 \
  --viz-color-by-error \
  --viz-max-error 4.0 \
  --export-type TXT \
  --export-csv statue_points.csv \
  --max-error 4.0 \
  --min-track-len 3
```

Visualize the exported CSV:

```bash
python3 pointcloud_visualize.py \
  --input fountain_points.csv \
  --max-error 4.0 \
  --output plots/fountain_colmap.png \
  --show
```

## Visualizer controls

When `--show` is set, matplotlib provides interactive controls:

- Left-drag: rotate
- Right-drag or shift+left-drag: pan
- Scroll: zoom

You can also set the initial view with `--elev` and `--azim`, plus `--zoom` and `--pad`:

```bash
python3 pointcloud_visualize.py \
  --input fountain_points.csv \
  --color-by-error \
  --elev 20 \
  --azim 35 \
  --zoom 1.4 \
  --show
```

## Outputs

- `auto_points.csv` (from `pointcloud_auto.py`)
  - Columns: point_id, x, y, z, avg_error, img_a, img_b, img_a_row, img_a_col,
    img_b_row, img_b_col, angle_deg
- `manual_points.csv` (from `pointcloud_manual.py`)
  - Columns: point_id, x, y, z, avg_error
- Plots (from `pointcloud_run.py`)
  - Saved as `plots/<csv_stem>_plot_###.png` without overwriting

## Files

- pointcloud_utils.py: parsing + ray math
- pointcloud_manual.py: manual point triangulation
- pointcloud_auto.py: automatic SIFT matching + 2-view triangulation
- pointcloud_colmap_run.py: COLMAP + visualization runner
- pointcloud_visualize.py: 3D scatter plot visualizer
- pointcloud_run.py: one-command pipeline (triangulate + plot)
