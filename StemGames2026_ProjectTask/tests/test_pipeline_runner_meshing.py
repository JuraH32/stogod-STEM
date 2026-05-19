from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from StemGames2026_ProjectTask.pipeline.mesh.base import MeshResult
from StemGames2026_ProjectTask.pipeline.runner import PipelineConfig, PipelineRunner
from StemGames2026_ProjectTask.pipeline.schemas import DepthResult, EstimatedPose, SceneResult
from StemGames2026_ProjectTask.pointcloud.schemas import CameraIntrinsics, CameraPose, SceneDataset, SceneView


def _identity_pose() -> CameraPose:
    return CameraPose(
        position=(0.0, 0.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )


def _make_dataset(root: Path) -> SceneDataset:
    image_path = root / "synthetic.png"
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(image_path)

    intrinsics = CameraIntrinsics(
        matrix=((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)),
        image_size=(2, 2),
        source="provided_k",
        fov_degrees=90.0,
        fov_axis="horizontal",
    )
    view = SceneView(
        scene_name="Synthetic",
        index=1,
        image_path=image_path,
        intrinsics=intrinsics,
        pose=_identity_pose(),
        pose_status="known",
    )
    return SceneDataset(
        scene_name="Synthetic",
        root_dir=root,
        views=(view,),
        pose_source="provided",
        metadata_files=(),
    )


class _FakePoseProvider:
    def can_provide(self, dataset: SceneDataset) -> bool:
        return True

    def provide(self, dataset: SceneDataset) -> list[EstimatedPose]:
        return [
            EstimatedPose(
                view_index=1,
                pose=_identity_pose(),
                confidence=1.0,
                source="ground_truth",
            )
        ]


class _FakeDepthEstimator:
    def estimate(self, view: SceneView, pose: CameraPose, reference_points_cam=None) -> DepthResult:
        depth_map = np.ones((2, 2), dtype=np.float32)
        return DepthResult(
            view=view,
            depth_map=depth_map,
            points_cam=np.zeros((2, 2, 3), dtype=np.float32),
            validity_mask=np.ones((2, 2), dtype=bool),
            scale_factor=1.0,
            scale_source="none",
        )


class _FakeFuser:
    def fuse(self, per_view) -> SceneResult:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        return SceneResult(
            scene_name=per_view[0].view.scene_name,
            per_view=per_view,
            scene_points=points,
            scene_colors=colors,
            source_view_index=np.zeros(len(points), dtype=np.int32),
            source_pixel=np.zeros((len(points), 2), dtype=np.int32),
        )


class _IdentityPostProcessor:
    def process(self, points, colors, source_view, source_pixel):
        return points, colors, source_view, source_pixel


class _FakeMesher:
    def mesh(self, scene_name: str, points: np.ndarray, colors: np.ndarray, output_path: Path) -> MeshResult:
        output_path.write_text("ply\n", encoding="ascii")
        return MeshResult(
            mesh_path=output_path,
            vertex_count=12,
            face_count=24,
            backend="fake-mesher",
        )


class _FailingMesher:
    def mesh(self, scene_name: str, points: np.ndarray, colors: np.ndarray, output_path: Path) -> MeshResult:
        raise RuntimeError("synthetic meshing failure")


class PipelineRunnerMeshingTests(unittest.TestCase):
    def test_runner_records_mesh_artifact_when_meshing_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = _make_dataset(tmp_path)
            cfg = PipelineConfig(
                output_root=tmp_path / "outputs",
                pose_provider=_FakePoseProvider(),
                depth_estimator=_FakeDepthEstimator(),
                fuser=_FakeFuser(),
                post_processor=_IdentityPostProcessor(),
                mesher=_FakeMesher(),
            )

            with patch("StemGames2026_ProjectTask.pipeline.runner._build_scale_hints", return_value={1: None}):
                result = PipelineRunner(cfg).run_scene(dataset)

            self.assertIsNotNone(result.scene_ply_path)
            self.assertTrue(result.scene_ply_path.exists())
            self.assertEqual(result.scene_mesh_backend, "fake-mesher")
            self.assertEqual(result.scene_mesh_vertex_count, 12)
            self.assertEqual(result.scene_mesh_face_count, 24)
            self.assertIsNotNone(result.scene_mesh_path)
            self.assertTrue(result.scene_mesh_path.exists())
            self.assertIsNone(result.mesh_warning)

    def test_runner_keeps_scene_cloud_when_meshing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = _make_dataset(tmp_path)
            cfg = PipelineConfig(
                output_root=tmp_path / "outputs",
                pose_provider=_FakePoseProvider(),
                depth_estimator=_FakeDepthEstimator(),
                fuser=_FakeFuser(),
                post_processor=_IdentityPostProcessor(),
                mesher=_FailingMesher(),
            )

            with patch("StemGames2026_ProjectTask.pipeline.runner._build_scale_hints", return_value={1: None}):
                result = PipelineRunner(cfg).run_scene(dataset)

            self.assertIsNotNone(result.scene_ply_path)
            self.assertTrue(result.scene_ply_path.exists())
            self.assertIsNone(result.scene_mesh_path)
            self.assertIn("synthetic meshing failure", result.mesh_warning or "")


if __name__ == "__main__":
    unittest.main()