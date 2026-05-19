from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from StemGames2026_ProjectTask.pipeline.mesh import TrimeshVoxelMesher


def _make_cube_points(side: int = 6, spacing: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, spacing * (side - 1), side, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    colors = np.clip(grid / grid.max(initial=1.0), 0.0, 1.0)
    colors = (colors * 255.0).astype(np.uint8)
    return grid, colors


class TrimeshMesherTests(unittest.TestCase):
    def test_mesher_exports_non_empty_ply(self) -> None:
        points, colors = _make_cube_points()
        mesher = TrimeshVoxelMesher(pitch=0.1, min_points=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene_mesh.ply"
            result = mesher.mesh("SyntheticCube", points, colors, output_path)

            self.assertEqual(result.mesh_path, output_path)
            self.assertGreater(result.vertex_count, 0)
            self.assertGreater(result.face_count, 0)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_mesher_rejects_tiny_point_clouds(self) -> None:
        mesher = TrimeshVoxelMesher(pitch=0.1, min_points=32)
        points = np.zeros((16, 3), dtype=np.float32)
        colors = np.zeros((16, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                mesher.mesh("TooSmall", points, colors, Path(tmpdir) / "scene_mesh.ply")


if __name__ == "__main__":
    unittest.main()