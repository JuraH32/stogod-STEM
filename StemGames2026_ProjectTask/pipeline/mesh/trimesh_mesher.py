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

    def __init__(self, pitch: float, min_points: int = 128, max_points: int = 200_000) -> None:
        if pitch <= 0.0:
            raise ValueError("pitch must be positive")
        if min_points < 8:
            raise ValueError("min_points must be at least 8")
        if max_points < min_points:
            raise ValueError("max_points must be at least min_points")
        self._pitch = float(pitch)
        self._min_points = int(min_points)
        self._max_points = int(max_points)

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
        points, colors = self._downsample_inputs(points, colors)
        pitch = self._resolve_pitch(points)

        try:
            mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=pitch)
        except ModuleNotFoundError as exc:
            if exc.name == "skimage":
                raise RuntimeError(
                    "scikit-image is required for trimesh marching-cubes meshing"
                ) from exc
            raise

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
            backend=f"trimesh-voxel-marching-cubes@pitch={pitch:.6g}",
        )

    def _resolve_pitch(self, points: np.ndarray) -> float:
        if len(points) < 2:
            return self._pitch

        from scipy.spatial import cKDTree

        if len(points) > 4096:
            sample_idx = np.linspace(0, len(points) - 1, num=4096, dtype=np.int64)
            sample = points[sample_idx]
        else:
            sample = points

        tree = cKDTree(points)
        distances, _ = tree.query(sample, k=2)
        nearest = np.asarray(distances[:, 1], dtype=np.float32)
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        if len(nearest) == 0:
            return self._pitch

        # Use a pitch tied to observed point spacing so sparse large-scale scenes
        # do not attempt marching cubes at an unrealistically fine resolution.
        adaptive_pitch = float(np.median(nearest) * 1.5)
        return max(self._pitch, adaptive_pitch)

    def _downsample_inputs(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(points) <= self._max_points:
            return points, colors

        sample_idx = np.linspace(0, len(points) - 1, num=self._max_points, dtype=np.int64)
        return points[sample_idx], colors[sample_idx]