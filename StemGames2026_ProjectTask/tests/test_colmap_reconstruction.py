from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from StemGames2026_ProjectTask.pipeline.reconstruction.colmap import (
    ReconstructionConfig,
    build_manual_camera_reader_options,
    discover_project_paths,
    resolve_matcher_mode,
    run_scene_reconstruction,
)
from StemGames2026_ProjectTask.pointcloud import load_scene


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_ROOT = PROJECT_ROOT / "TestImages"


class _FakeFeatureExtractionOptions:
    def __init__(self) -> None:
        self.max_image_size = None
        self.use_gpu = None
        self.sift = SimpleNamespace(max_num_features=None)


class _FakeSiftExtractionOptions:
    def __init__(self) -> None:
        self.max_num_features = None


class _FakeImageReaderOptions:
    def __init__(self) -> None:
        self.camera_model = ""
        self.camera_params = ""


class _FakeFeatureMatchingOptions:
    def __init__(self) -> None:
        self.use_gpu = None


class _FakeSequentialPairingOptions:
    def __init__(self) -> None:
        self.overlap = None


class _FakeExhaustivePairingOptions:
    pass


class _FakeTwoViewGeometryOptions:
    pass


class _FakeMapperOptions:
    def __init__(self) -> None:
        self.abs_pose_refine_focal_length = True
        self.abs_pose_refine_extra_params = True


class _FakeIncrementalPipelineOptions:
    def __init__(self) -> None:
        self.multiple_models = False
        self.max_num_models = None
        self.min_model_size = None
        self.ba_refine_focal_length = True
        self.ba_refine_principal_point = True
        self.ba_refine_extra_params = True
        self.ba_use_gpu = True
        self.mapper = _FakeMapperOptions()


class _FakePatchMatchOptions:
    def __init__(self) -> None:
        self.max_image_size = None


class _FakeStereoFusionOptions:
    pass


class _FakePoissonMeshingOptions:
    def __init__(self) -> None:
        self.num_threads = None


class _FakeReconstruction:
    def __init__(self, reg_images: int = 4, points: int = 42) -> None:
        self._reg_images = reg_images
        self._points = points

    def num_reg_images(self) -> int:
        return self._reg_images

    def num_points3D(self) -> int:
        return self._points

    def write(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "cameras.bin").write_bytes(b"cameras")
        (output_path / "images.bin").write_bytes(b"images")
        (output_path / "points3D.bin").write_bytes(b"points")

    def export_PLY(self, output_path: Path) -> None:
        Path(output_path).write_text(
            "ply\nformat ascii 1.0\nend_header\n0 0 0\n",
            encoding="utf-8",
        )


class _FakePycolmap:
    class CameraMode:
        SINGLE = "SINGLE"

    FeatureExtractionOptions = _FakeFeatureExtractionOptions
    SiftExtractionOptions = _FakeSiftExtractionOptions
    ImageReaderOptions = _FakeImageReaderOptions
    FeatureMatchingOptions = _FakeFeatureMatchingOptions
    SequentialPairingOptions = _FakeSequentialPairingOptions
    ExhaustivePairingOptions = _FakeExhaustivePairingOptions
    TwoViewGeometryOptions = _FakeTwoViewGeometryOptions
    IncrementalPipelineOptions = _FakeIncrementalPipelineOptions
    PatchMatchOptions = _FakePatchMatchOptions
    StereoFusionOptions = _FakeStereoFusionOptions
    PoissonMeshingOptions = _FakePoissonMeshingOptions

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def extract_features(self, **kwargs) -> None:
        self.calls.append(("extract_features", kwargs))
        Path(kwargs["database_path"]).write_bytes(b"db")

    def match_exhaustive(self, **kwargs) -> None:
        self.calls.append(("match_exhaustive", kwargs))

    def match_sequential(self, **kwargs) -> None:
        self.calls.append(("match_sequential", kwargs))

    def incremental_mapping(self, **kwargs):
        self.calls.append(("incremental_mapping", kwargs))
        sparse_root = Path(kwargs["output_path"])
        sparse_root.mkdir(parents=True, exist_ok=True)
        (sparse_root / "0").mkdir(exist_ok=True)
        return {0: _FakeReconstruction()}

    def undistort_images(self, **kwargs) -> None:
        self.calls.append(("undistort_images", kwargs))
        Path(kwargs["output_path"]).mkdir(parents=True, exist_ok=True)

    def patch_match_stereo(self, **kwargs) -> None:
        self.calls.append(("patch_match_stereo", kwargs))
        raise RuntimeError("patch-match unavailable")

    def stereo_fusion(self, **kwargs) -> None:
        self.calls.append(("stereo_fusion", kwargs))
        Path(kwargs["output_path"]).write_text("ply\n", encoding="utf-8")

    def poisson_meshing(self, **kwargs) -> None:
        self.calls.append(("poisson_meshing", kwargs))
        Path(kwargs["output_path"]).write_text("ply\n", encoding="utf-8")


class ColmapReconstructionTests(unittest.TestCase):
    def test_discover_project_paths_from_reconstruct_script(self) -> None:
        paths = discover_project_paths(REPO_ROOT / "reconstruct.py")

        self.assertEqual(paths.repo_root, REPO_ROOT)
        self.assertEqual(paths.project_root, PROJECT_ROOT)
        self.assertEqual(paths.test_images_root, TEST_IMAGES_ROOT)
        self.assertEqual(paths.output_root, PROJECT_ROOT / "outputs" / "colmap")

    def test_reader_options_use_scene_intrinsics(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Fountain")
        reader_options = build_manual_camera_reader_options(_FakePycolmap, dataset.views[0].intrinsics)

        self.assertEqual(reader_options.camera_model, "PINHOLE")
        self.assertEqual(
            reader_options.camera_params,
            "2759.48,2764.16,1520.69,1006.81",
        )

    def test_auto_matcher_prefers_exhaustive_for_small_scenes(self) -> None:
        self.assertEqual(resolve_matcher_mode("auto", 12), "exhaustive")
        self.assertEqual(resolve_matcher_mode("auto", 40), "sequential")
        self.assertEqual(resolve_matcher_mode("sequential", 12), "sequential")

    def test_run_scene_reconstruction_falls_back_to_sparse_when_dense_fails(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Box")
        fake_backend = _FakePycolmap()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReconstructionConfig(
                output_root=Path(tmpdir),
                dense_mode="auto",
                overwrite=True,
            )
            with patch("platform.system", return_value="Linux"):
                result = run_scene_reconstruction(dataset, config, backend=fake_backend)
            self.assertTrue(result.is_valid_output)
            self.assertTrue(result.dense_attempted)
            self.assertFalse(result.dense_succeeded)
            self.assertEqual(result.final_artifact_path, result.artifacts.sparse_ply_path)
            self.assertTrue(any("Dense reconstruction failed" in warning for warning in result.warnings))
            self.assertTrue(any(name == "match_exhaustive" for name, _ in fake_backend.calls))

    def test_run_scene_reconstruction_uses_requested_sequential_matcher(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Entrance")
        fake_backend = _FakePycolmap()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReconstructionConfig(
                output_root=Path(tmpdir),
                matcher_mode="sequential",
                dense_mode="off",
                overwrite=True,
            )
            result = run_scene_reconstruction(dataset, config, backend=fake_backend)
            self.assertTrue(result.is_valid_output)
            self.assertEqual(result.matcher_mode, "sequential")
            self.assertTrue(any(name == "match_sequential" for name, _ in fake_backend.calls))
            self.assertFalse(any(name == "match_exhaustive" for name, _ in fake_backend.calls))


if __name__ == "__main__":
    unittest.main()