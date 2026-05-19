"""
Preview generated scene meshes in the browser.

Usage (from inside StemGames2026_ProjectTask/):
    python visualize_mesh.py Box
    python visualize_mesh.py Box --show-cloud
    python visualize_mesh.py --list
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from StemGames2026_ProjectTask.visualize import MAX_POINTS_PER_TRACE, read_ply, rgb_to_hex, subsample

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def available_scenes(output_root: Path) -> list[str]:
    if not output_root.exists():
        return []
    return [path.name for path in sorted(output_root.iterdir()) if path.is_dir() and path.name != "colmap"]


def scene_mesh(output_root: Path, scene: str) -> Path:
    return output_root / scene / "scene_mesh.ply"


def scene_cloud(output_root: Path, scene: str) -> Path:
    return output_root / scene / "scene_cloud.ply"


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


def _open_figure(fig, name: str) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix=f"mesh_{name}_", delete=False)
    fig.write_html(tmp.name, include_plotlyjs="cdn")
    print(f"Opening: {tmp.name}")
    webbrowser.open(f"file://{tmp.name}")


def show_mesh(scene: str, output_root: Path, opacity: float, show_cloud: bool, cloud_max_points: int) -> None:
    import plotly.graph_objects as go
    import trimesh

    mesh_path = scene_mesh(output_root, scene)
    if not mesh_path.exists():
        print(f"No scene mesh found at {mesh_path}. Run run_meshing.py or run_pipeline.py first.")
        return

    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        print(f"Mesh at {mesh_path} is empty.")
        return

    mesh_kwargs = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
        flatshading=True,
        name=f"{scene} mesh",
    )
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(vertices):
        mesh_kwargs["vertexcolor"] = mesh.visual.vertex_colors[:, :3]
    else:
        mesh_kwargs["color"] = "#9A6324"

    traces = [go.Mesh3d(**mesh_kwargs)]

    if show_cloud:
        cloud_path = scene_cloud(output_root, scene)
        if cloud_path.exists():
            points, colors = read_ply(cloud_path)
            pts, cols = subsample(points, colors, cloud_max_points)
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=1.0, color=rgb_to_hex(cols), opacity=0.22),
                    name=f"{scene} cloud",
                )
            )

    fig = go.Figure(data=traces, layout=_make_scene_layout(f"{scene} — scene mesh"))
    _open_figure(fig, scene)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview generated scene meshes")
    parser.add_argument("scene", nargs="?", help="Scene name (e.g. Box)")
    parser.add_argument("--list", action="store_true", help="List available scenes and mesh outputs")
    parser.add_argument("--opacity", type=float, default=0.92, help="Mesh opacity (default: 0.92)")
    parser.add_argument("--show-cloud", action="store_true", help="Overlay the scene cloud as a translucent point trace")
    parser.add_argument(
        "--cloud-max-points", type=int, default=MAX_POINTS_PER_TRACE // 4,
        help="Maximum number of point-cloud samples when --show-cloud is used"
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Output directory to visualize from (default: StemGames2026_ProjectTask/outputs)"
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)

    if args.list:
        scenes = available_scenes(output_root)
        if not scenes:
            print(f"No outputs found under {output_root}.")
            return
        for scene in scenes:
            mesh_path = scene_mesh(output_root, scene)
            cloud_path = scene_cloud(output_root, scene)
            print(f"\n{scene}:")
            print(f"  scene_cloud.ply  {'✓' if cloud_path.exists() else '✗'}")
            print(f"  scene_mesh.ply   {'✓' if mesh_path.exists() else '✗'}")
        return

    if not args.scene:
        parser.print_help()
        return

    show_mesh(
        scene=args.scene,
        output_root=output_root,
        opacity=args.opacity,
        show_cloud=args.show_cloud,
        cloud_max_points=args.cloud_max_points,
    )


if __name__ == "__main__":
    main()