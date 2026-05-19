"""
Point cloud and depth map visualizer.

Usage (from inside StemGames2026_ProjectTask/):

  python visualize.py Box                    # scene cloud in browser
  python visualize.py Box --image 3          # per-image cloud for view #3
  python visualize.py Box --all-images       # all per-image clouds, colour-coded by view
  python visualize.py Box --depth            # depth map contact sheet (opens as image)
  python visualize.py Box --compare 2 5      # two per-image clouds side-by-side

  python visualize.py --list                 # list available scenes and outputs
"""
from __future__ import annotations

import sys
import argparse
import webbrowser
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Maximum points to show in a single Plotly trace (keeps browser responsive)
MAX_POINTS_PER_TRACE = 200_000

# Distinct colours for per-image traces (cycles if more views than colours)
VIEW_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


# ---------------------------------------------------------------------------
# PLY reader (pure numpy — no open3d required)
# ---------------------------------------------------------------------------

def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a binary little-endian XYZRGB PLY file.
    Returns (points (N,3) float32, colors (N,3) uint8).
    """
    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")

    n_vertices = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
            break

    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    body = np.frombuffer(raw[header_end:], dtype=vertex_dtype, count=n_vertices)
    points = np.stack([body["x"], body["y"], body["z"]], axis=1)
    colors = np.stack([body["red"], body["green"], body["blue"]], axis=1)
    return points, colors


def subsample(points: np.ndarray, colors: np.ndarray, max_pts: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_pts:
        return points, colors
    idx = np.random.choice(len(points), max_pts, replace=False)
    return points[idx], colors[idx]


def rgb_to_hex(colors: np.ndarray) -> list[str]:
    """Convert (N,3) uint8 RGB array to list of '#rrggbb' strings."""
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _make_scene_layout(title: str):
    import plotly.graph_objects as go
    return go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )


def show_cloud(points: np.ndarray, colors: np.ndarray, title: str = "Point Cloud", max_points: int = MAX_POINTS_PER_TRACE) -> None:
    import plotly.graph_objects as go

    pts, cols = subsample(points, colors, max_points)
    fig = go.Figure(
        data=[go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=rgb_to_hex(cols), opacity=0.85),
            name=title,
        )],
        layout=_make_scene_layout(title),
    )
    _open_figure(fig, title)


def show_multi_cloud(
    clouds: list[tuple[np.ndarray, np.ndarray]],
    names: list[str],
    title: str = "Multi-view Point Clouds",
    max_points: int = MAX_POINTS_PER_TRACE,
) -> None:
    import plotly.graph_objects as go

    per_trace = max(1, max_points // len(clouds))
    traces = []
    for (pts, cols), name, colour in zip(clouds, names, VIEW_PALETTE * (len(clouds) // len(VIEW_PALETTE) + 1)):
        pts, _ = subsample(pts, cols, per_trace)
        # Colour by view rather than by pixel colour for easy differentiation
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=colour, opacity=0.7),
            name=name,
        ))

    fig = go.Figure(data=traces, layout=_make_scene_layout(title))
    _open_figure(fig, title)


def show_side_by_side(
    clouds: list[tuple[np.ndarray, np.ndarray]],
    names: list[str],
    max_points: int = MAX_POINTS_PER_TRACE,
) -> None:
    """Show 2 clouds in side-by-side subplots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(clouds)
    specs = [[{"type": "scatter3d"}] * n]
    fig = make_subplots(rows=1, cols=n, specs=specs, subplot_titles=names)

    per_trace = max(1, max_points // n)
    for col_idx, ((pts, cols), name) in enumerate(zip(clouds, names), start=1):
        pts, cols = subsample(pts, cols, per_trace)
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=rgb_to_hex(cols), opacity=0.85),
                name=name,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(title="Side-by-side comparison", height=700)
    _open_figure(fig, "compare")


def _open_figure(fig, name: str) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix=f"moge_{name}_", delete=False)
    fig.write_html(tmp.name, include_plotlyjs="cdn")
    print(f"Opening: {tmp.name}")
    webbrowser.open(f"file://{tmp.name}")


# ---------------------------------------------------------------------------
# Depth map grid
# ---------------------------------------------------------------------------

def show_depth_grid(scene: str, output_root: Path) -> None:
    depth_dir = output_root / scene / "depth_maps"
    pngs = sorted(depth_dir.glob("*_depth.png"))
    if not pngs:
        print(f"No depth PNGs found in {depth_dir}")
        return

    cols = 4
    rows = (len(pngs) + cols - 1) // cols
    thumb_w, thumb_h = 480, 270
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 24)), (16, 18, 24))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)

    for i, png in enumerate(pngs):
        img = Image.open(png).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))
        c, r = i % cols, i // cols
        x, y = c * thumb_w, r * (thumb_h + 24)
        sheet.paste(img, (x + (thumb_w - img.width) // 2, y + (thumb_h - img.height) // 2))
        label = png.stem.replace("_depth", "")
        draw.text((x + 6, y + thumb_h + 4), label, fill=(200, 210, 220))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix=f"depth_{scene}_", delete=False)
    sheet.save(tmp.name)
    print(f"Depth grid saved: {tmp.name}")
    import subprocess
    subprocess.Popen(["open", tmp.name])  # macOS: opens in Preview


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def available_scenes(output_root: Path) -> list[str]:
    if not output_root.exists():
        return []
    return [d.name for d in sorted(output_root.iterdir()) if d.is_dir()]


def per_image_plys(output_root: Path, scene: str) -> list[Path]:
    cloud_dir = output_root / scene / "per_image_clouds"
    return sorted(cloud_dir.glob("*.ply")) if cloud_dir.exists() else []


def scene_ply(output_root: Path, scene: str) -> Path:
    return output_root / scene / "scene_cloud.ply"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MoGe-2 point clouds and depth maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("scene", nargs="?", help="Scene name (e.g. Box)")
    parser.add_argument("--list", action="store_true", help="List available scenes and outputs")
    parser.add_argument("--image", type=int, metavar="N", help="Show per-image cloud for view N")
    parser.add_argument("--all-images", action="store_true", help="Show all per-image clouds colour-coded by view")
    parser.add_argument("--depth", action="store_true", help="Show depth map contact sheet")
    parser.add_argument("--compare", type=int, nargs=2, metavar=("A", "B"),
                        help="Side-by-side comparison of two per-image clouds")
    parser.add_argument("--max-points", type=int, default=MAX_POINTS_PER_TRACE,
                        help=f"Max points per trace (default {MAX_POINTS_PER_TRACE:,})")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Output directory to visualize from (default: StemGames2026_ProjectTask/outputs)"
    )
    args = parser.parse_args()

    max_pts = args.max_points
    output_root = Path(args.output_root)

    if args.list:
        scenes = available_scenes(output_root)
        if not scenes:
            print(f"No outputs found under {output_root}. Run run_pipeline.py first.")
            return
        for s in scenes:
            plys = per_image_plys(output_root, s)
            sc = scene_ply(output_root, s)
            print(f"\n{s}:")
            print(f"  scene_cloud.ply  {'✓' if sc.exists() else '✗'}  ({sc.stat().st_size // 1024 // 1024} MB)" if sc.exists() else "  scene_cloud.ply  ✗")
            print(f"  per-image clouds: {len(plys)} files")
        return

    if not args.scene:
        parser.print_help()
        return

    scene = args.scene

    if args.depth:
        show_depth_grid(scene, output_root)
        return

    if args.image is not None:
        plys = per_image_plys(output_root, scene)
        matches = [p for p in plys if f"{args.image}" in p.stem]
        if not matches:
            # Try to match by index position
            if 1 <= args.image <= len(plys):
                matches = [plys[args.image - 1]]
            else:
                print(f"No per-image cloud found for view {args.image} in {scene}.")
                print(f"Available: {[p.name for p in plys]}")
                return
        path = matches[0]
        pts, cols = read_ply(path)
        show_cloud(pts, cols, title=f"{scene} — {path.stem}", max_points=max_pts)
        return

    if args.compare:
        a_idx, b_idx = args.compare
        plys = per_image_plys(output_root, scene)
        def _get(n):
            matches = [p for p in plys if p.stem.endswith(str(n))]
            return matches[0] if matches else (plys[n - 1] if 1 <= n <= len(plys) else None)
        pa, pb = _get(a_idx), _get(b_idx)
        if not pa or not pb:
            print(f"Could not find clouds for views {a_idx} and {b_idx}.")
            return
        clouds = [read_ply(pa), read_ply(pb)]
        show_side_by_side(clouds, [pa.stem, pb.stem], max_points=max_pts)
        return

    if args.all_images:
        plys = per_image_plys(output_root, scene)
        if not plys:
            print(f"No per-image clouds found for {scene}.")
            return
        clouds = [read_ply(p) for p in plys]
        names = [p.stem for p in plys]
        show_multi_cloud(clouds, names, title=f"{scene} — all views", max_points=max_pts)
        return

    # Default: show scene cloud
    sc = scene_ply(output_root, scene)
    if not sc.exists():
        print(f"No scene cloud found at {sc}. Run run_pipeline.py --scenes {scene} first.")
        return
    pts, cols = read_ply(sc)
    print(f"Loaded {len(pts):,} points from {sc.name}")
    show_cloud(pts, cols, title=f"{scene} — scene cloud", max_points=max_pts)


if __name__ == "__main__":
    main()
