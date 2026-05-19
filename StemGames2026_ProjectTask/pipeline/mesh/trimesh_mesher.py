from __future__ import annotations

from pathlib import Path

import numpy as np

from StemGames2026_ProjectTask.pipeline.mesh.base import MeshResult, SceneMesher


class TrimeshVoxelMesher(SceneMesher):
    """
    Convert a point cloud to a triangle mesh by voxelizing points and running
    marching cubes through trimesh.

    The output is intentionally approximate: this is a best-effort surface model
    built from the fused point cloud, not a watertight reconstruction guarantee.
    """

    def __init__(self, pitch: float, min_points: int = 128) -> None:
        if pitch <= 0.0:
            raise ValueError("pitch must be positive")
        if min_points < 8:
            raise ValueError("min_points must be at least 8")
        self._pitch = float(pitch)
        self._min_points = int(min_points)

    def mesh(
        self,
        scene_name: str,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path,
    ) -> MeshResult:
        if len(points) < self._min_points:
            raise ValueError(
                f"Need at least {self._min_points} points to mesh {scene_name}; got {len(points)}"
            )

        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("trimesh is required for mesh generation") from exc

        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)

        mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=self._pitch)
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError(f"Marching cubes produced an empty mesh for {scene_name}")

        if len(colors) == len(points):
            from scipy.spatial import cKDTree

            nearest = cKDTree(points)
            _, idx = nearest.query(mesh.vertices, k=1)
            vertex_colors = colors[np.asarray(idx, dtype=np.int64)]
            alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
            mesh.visual.vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)

        return MeshResult(
            mesh_path=output_path,
            vertex_count=int(len(mesh.vertices)),
            face_count=int(len(mesh.faces)),
            backend="trimesh-voxel-marching-cubes",
        )