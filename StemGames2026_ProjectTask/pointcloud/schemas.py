from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    matrix: Matrix3
    image_size: tuple[int, int]
    source: Literal["provided_k", "derived_from_fov"]
    fov_degrees: float | None = None
    fov_axis: Literal["horizontal", "vertical"] | None = None

    @property
    def fx(self) -> float:
        return self.matrix[0][0]

    @property
    def fy(self) -> float:
        return self.matrix[1][1]

    @property
    def cx(self) -> float:
        return self.matrix[0][2]

    @property
    def cy(self) -> float:
        return self.matrix[1][2]


@dataclass(frozen=True)
class CameraPose:
    position: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3

    def camera_to_world_matrix(self) -> Matrix4:
        return (
            (self.right[0], self.up[0], self.forward[0], self.position[0]),
            (self.right[1], self.up[1], self.forward[1], self.position[1]),
            (self.right[2], self.up[2], self.forward[2], self.position[2]),
            (0.0, 0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class SceneView:
    scene_name: str
    index: int
    image_path: Path
    intrinsics: CameraIntrinsics
    pose: CameraPose | None
    pose_status: Literal["known", "needs_estimation"]
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneDataset:
    scene_name: str
    root_dir: Path
    views: tuple[SceneView, ...]
    pose_source: Literal["provided", "missing"]
    metadata_files: tuple[Path, ...]
    notes: tuple[str, ...] = ()

    @property
    def requires_pose_estimation(self) -> bool:
        return self.pose_source == "missing"

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return tuple(view.image_path for view in self.views)
