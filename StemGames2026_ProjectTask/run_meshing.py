"""
Generate scene meshes from existing fused point clouds.

Usage (from inside StemGames2026_ProjectTask/):
    python run_meshing.py                        # mesh all scenes with scene_cloud.ply
    python run_meshing.py --scenes Box Statue   # mesh specific scenes
    python run_meshing.py --max-points 100000   # cap meshing input size

Outputs are written to outputs/{scene_name}/scene_mesh.ply.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from StemGames2026_ProjectTask.pipeline.mesh import TrimeshVoxelMesher
from StemGames2026_ProjectTask.visualize import read_ply

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def available_scenes(output_root: Path) -> list[str]:
    return [path.name for path in sorted(output_root.iterdir()) if path.is_dir() and path.name != "colmap"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene meshes from scene clouds")
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene names to process (default: all with scene clouds)"
    )
    parser.add_argument(
        "--pitch", type=float, default=0.01,
        help="Minimum voxel pitch for meshing (adaptive scaling may increase it)"
    )
    parser.add_argument(
        "--min-points", type=int, default=128,
        help="Minimum number of points required to attempt meshing"
    )
    parser.add_argument(
        "--max-points", type=int, default=200_000,
        help="Maximum number of points passed into meshing after deterministic sampling"
    )
    args = parser.parse_args()

    scenes = args.scenes if args.scenes else available_scenes(OUTPUT_ROOT)
    mesher = TrimeshVoxelMesher(
        pitch=args.pitch,
        min_points=args.min_points,
        max_points=args.max_points,
    )

    failures: list[str] = []
    for scene in scenes:
        scene_dir = OUTPUT_ROOT / scene
        scene_cloud_path = scene_dir / "scene_cloud.ply"
        scene_mesh_path = scene_dir / "scene_mesh.ply"

        if not scene_cloud_path.exists():
            print(f"[{scene}] skipped: no scene cloud at {scene_cloud_path}")
            failures.append(scene)
            continue

        points, colors = read_ply(scene_cloud_path)
        print(f"[{scene}] meshing {len(points):,} points from {scene_cloud_path}")
        try:
            result = mesher.mesh(scene, points, colors, scene_mesh_path)
        except Exception as exc:
            print(f"[{scene}] mesh failed: {exc}")
            failures.append(scene)
            continue

        print(
            f"[{scene}] mesh complete: {result.vertex_count:,} vertices, "
            f"{result.face_count:,} faces → {result.mesh_path} ({result.backend})"
        )

    if failures:
        raise SystemExit(f"Meshing failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()