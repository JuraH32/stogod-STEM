from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class MeshResult:
    mesh_path: Path
    vertex_count: int
    face_count: int
    backend: str


class SceneMesher(ABC):
    @abstractmethod
    def mesh(
        self,
        scene_name: str,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path,
    ) -> MeshResult:
        """
        Build a triangle mesh from a scene point cloud.

        Args:
            scene_name: name of the scene being meshed
            points:     (N, 3) float32 world-space points
            colors:     (N, 3) uint8 per-point RGB colours
            output_path: destination mesh path

        Returns:
            MeshResult describing the written mesh artifact.
        """
        ...