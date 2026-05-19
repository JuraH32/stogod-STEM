from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from StemGames2026_ProjectTask.pointcloud import load_project_scenes
from StemGames2026_ProjectTask.pointcloud.schemas import CameraIntrinsics, SceneDataset


class ReconstructionError(RuntimeError):
    pass


@dataclass(slots=True)
class ProjectPaths:
    repo_root: Path
    project_root: Path
    test_images_root: Path
    output_root: Path


@dataclass(slots=True)
class ReconstructionConfig:
    output_root: Path
    matcher_mode: str = "auto"
    dense_mode: str = "auto"
    overwrite: bool = False
    max_image_size: int = 1600
    max_num_features: int = 8192
    sequential_overlap: int = 5
    min_model_size: int = 2

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.matcher_mode = self.matcher_mode.lower()
        self.dense_mode = self.dense_mode.lower()

        if self.matcher_mode not in {"auto", "exhaustive", "sequential"}:
            raise ValueError(f"Unsupported matcher mode: {self.matcher_mode}")
        if self.dense_mode not in {"auto", "off", "required"}:
            raise ValueError(f"Unsupported dense mode: {self.dense_mode}")
        if self.max_image_size <= 0:
            raise ValueError("max_image_size must be positive")
        if self.max_num_features <= 0:
            raise ValueError("max_num_features must be positive")
        if self.sequential_overlap <= 0:
            raise ValueError("sequential_overlap must be positive")
        if self.min_model_size <= 0:
            raise ValueError("min_model_size must be positive")


@dataclass(slots=True)
class SceneArtifacts:
    scene_root: Path
    database_path: Path
    sparse_root: Path
    selected_model_path: Path
    sparse_ply_path: Path
    dense_root: Path
    fused_ply_path: Path
    mesh_path: Path
    status_path: Path


@dataclass(slots=True)
class SceneRunResult:
    scene_name: str
    image_count: int
    matcher_mode: str
    sparse_model_count: int
    registered_images: int
    sparse_points: int
    dense_attempted: bool
    dense_succeeded: bool
    mesh_succeeded: bool
    final_artifact_path: Path
    artifacts: SceneArtifacts
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid_output(self) -> bool:
        return (
            self.registered_images > 0
            and self.sparse_points > 0
            and self.final_artifact_path.exists()
            and self.final_artifact_path.stat().st_size > 0
        )


def discover_project_paths(entry_file: str | Path) -> ProjectPaths:
    repo_root = Path(entry_file).resolve().parent
    project_root = repo_root / "StemGames2026_ProjectTask"
    test_images_root = project_root / "TestImages"
    output_root = project_root / "outputs" / "colmap"

    if not project_root.is_dir():
        raise ReconstructionError(f"Project directory not found: {project_root}")
    if not test_images_root.is_dir():
        raise ReconstructionError(f"Dataset directory not found: {test_images_root}")

    return ProjectPaths(
        repo_root=repo_root,
        project_root=project_root,
        test_images_root=test_images_root,
        output_root=output_root,
    )


def load_requested_scenes(project_root: Path, scene_names: list[str] | None = None) -> list[SceneDataset]:
    datasets = load_project_scenes(project_root)
    if scene_names is None:
        return list(datasets.values())

    unknown = sorted(set(scene_names) - set(datasets))
    if unknown:
        raise ReconstructionError(
            f"Unknown scene(s): {unknown}. Available scenes: {sorted(datasets)}"
        )
    return [datasets[name] for name in scene_names]


def resolve_matcher_mode(requested_mode: str, image_count: int) -> str:
    requested_mode = requested_mode.lower()
    if requested_mode != "auto":
        return requested_mode
    return "exhaustive" if image_count <= 32 else "sequential"


def run_scene_reconstruction(
    dataset: SceneDataset,
    config: ReconstructionConfig,
    backend: Any | None = None,
) -> SceneRunResult:
    pycolmap = backend if backend is not None else _import_pycolmap()
    artifacts = _prepare_workspace(dataset, config)
    image_names = [view.image_path.name for view in dataset.views]
    matcher_mode = resolve_matcher_mode(config.matcher_mode, len(image_names))
    warnings: list[str] = []

    _extract_features(pycolmap, dataset, artifacts, config, image_names)
    _match_images(pycolmap, artifacts.database_path, matcher_mode, config)
    selected_reconstruction, sparse_model_count = _run_sparse_mapping(
        pycolmap, dataset, artifacts, config
    )

    registered_images = _read_count(selected_reconstruction, "num_reg_images")
    sparse_points = _read_count(selected_reconstruction, "num_points3D")
    if registered_images <= 0:
        raise ReconstructionError(
            f"COLMAP did not register any images for scene '{dataset.scene_name}'."
        )
    if sparse_points <= 0:
        raise ReconstructionError(
            f"COLMAP did not triangulate any points for scene '{dataset.scene_name}'."
        )

    _write_selected_model(selected_reconstruction, artifacts.selected_model_path)
    _export_sparse_point_cloud(selected_reconstruction, artifacts.sparse_ply_path)
    final_artifact_path = artifacts.sparse_ply_path

    dense_attempted = False
    dense_succeeded = False
    mesh_succeeded = False
    dense_reason = _dense_support_reason(config.dense_mode)
    if dense_reason is not None:
        if config.dense_mode == "required":
            raise ReconstructionError(dense_reason)
        warnings.append(dense_reason)
    elif config.dense_mode != "off":
        dense_attempted = True
        try:
            _run_dense_reconstruction(pycolmap, dataset, artifacts, config, image_names)
            _assert_non_empty_file(artifacts.fused_ply_path, "dense fused point cloud")
            final_artifact_path = artifacts.fused_ply_path
            dense_succeeded = True

            try:
                _run_poisson_meshing(pycolmap, artifacts, config)
                _assert_non_empty_file(artifacts.mesh_path, "Poisson mesh")
                mesh_succeeded = True
            except Exception as exc:  # pragma: no cover - exercised in real runtime only
                warnings.append(f"Poisson meshing failed: {exc}")
        except Exception as exc:
            if config.dense_mode == "required":
                raise ReconstructionError(
                    f"Dense reconstruction failed for scene '{dataset.scene_name}': {exc}"
                ) from exc
            warnings.append(f"Dense reconstruction failed: {exc}")

    result = SceneRunResult(
        scene_name=dataset.scene_name,
        image_count=len(image_names),
        matcher_mode=matcher_mode,
        sparse_model_count=sparse_model_count,
        registered_images=registered_images,
        sparse_points=sparse_points,
        dense_attempted=dense_attempted,
        dense_succeeded=dense_succeeded,
        mesh_succeeded=mesh_succeeded,
        final_artifact_path=final_artifact_path,
        artifacts=artifacts,
        warnings=warnings,
    )
    _write_status_file(result)
    return result


def run_scene_batch(
    datasets: list[SceneDataset],
    config: ReconstructionConfig,
    backend: Any | None = None,
) -> list[SceneRunResult]:
    return [run_scene_reconstruction(dataset, config, backend=backend) for dataset in datasets]


def build_manual_camera_reader_options(pycolmap: Any, intrinsics: CameraIntrinsics) -> Any:
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = ",".join(
        f"{value:.12g}"
        for value in (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    )
    return reader_options


def _import_pycolmap() -> Any:
    try:
        import pycolmap  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise ReconstructionError(
            "pycolmap is not installed in the active environment. "
            "Install it into /Users/jurahostic/Documents/STEM/stogod-STEM/.venv or run the script "
            "with an interpreter that already provides pycolmap."
        ) from exc
    return pycolmap


def _prepare_workspace(dataset: SceneDataset, config: ReconstructionConfig) -> SceneArtifacts:
    scene_root = config.output_root / dataset.scene_name
    if scene_root.exists():
        if not config.overwrite:
            raise ReconstructionError(
                f"Output workspace already exists for scene '{dataset.scene_name}': {scene_root}. "
                "Re-run with --overwrite to replace it."
            )
        shutil.rmtree(scene_root)

    sparse_root = scene_root / "sparse"
    dense_root = scene_root / "dense"
    scene_root.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    return SceneArtifacts(
        scene_root=scene_root,
        database_path=scene_root / "database.db",
        sparse_root=sparse_root,
        selected_model_path=sparse_root / "selected",
        sparse_ply_path=scene_root / "scene_sparse.ply",
        dense_root=dense_root,
        fused_ply_path=scene_root / "scene_fused.ply",
        mesh_path=scene_root / "scene_mesh.ply",
        status_path=scene_root / "summary.json",
    )


def _extract_features(
    pycolmap: Any,
    dataset: SceneDataset,
    artifacts: SceneArtifacts,
    config: ReconstructionConfig,
    image_names: list[str],
) -> None:
    extraction_options = pycolmap.FeatureExtractionOptions()
    if hasattr(extraction_options, "max_image_size"):
        extraction_options.max_image_size = config.max_image_size
    if hasattr(extraction_options, "use_gpu"):
        extraction_options.use_gpu = False
    if hasattr(extraction_options, "sift") and hasattr(extraction_options.sift, "max_num_features"):
        extraction_options.sift.max_num_features = config.max_num_features

    reader_options = build_manual_camera_reader_options(pycolmap, dataset.views[0].intrinsics)
    camera_mode = getattr(getattr(pycolmap, "CameraMode", None), "SINGLE", None)

    call_kwargs = {
        "database_path": artifacts.database_path,
        "image_path": dataset.root_dir,
        "image_names": image_names,
        "reader_options": reader_options,
        "extraction_options": extraction_options,
    }
    if camera_mode is not None:
        call_kwargs["camera_mode"] = camera_mode

    sift_options_ctor = getattr(pycolmap, "SiftExtractionOptions", None)
    if sift_options_ctor is not None:
        sift_options = sift_options_ctor()
        if hasattr(sift_options, "max_num_features"):
            sift_options.max_num_features = config.max_num_features
        try:
            pycolmap.extract_features(**call_kwargs, sift_options=sift_options)
            return
        except TypeError:
            pass

    pycolmap.extract_features(**call_kwargs)


def _match_images(
    pycolmap: Any,
    database_path: Path,
    matcher_mode: str,
    config: ReconstructionConfig,
) -> None:
    matching_options = getattr(pycolmap, "FeatureMatchingOptions", lambda: None)()
    if matching_options is not None and hasattr(matching_options, "use_gpu"):
        matching_options.use_gpu = False

    verification_options = getattr(pycolmap, "TwoViewGeometryOptions", lambda: None)()

    if matcher_mode == "sequential":
        if hasattr(pycolmap, "SequentialPairingOptions"):
            pairing_options = pycolmap.SequentialPairingOptions()
            if hasattr(pairing_options, "overlap"):
                pairing_options.overlap = config.sequential_overlap
            kwargs = {
                "database_path": database_path,
                "pairing_options": pairing_options,
            }
            if matching_options is not None:
                kwargs["matching_options"] = matching_options
            if verification_options is not None:
                kwargs["verification_options"] = verification_options
            pycolmap.match_sequential(**kwargs)
            return

        pairing_options = pycolmap.SequentialMatchingOptions()
        pairing_options.overlap = config.sequential_overlap
        pycolmap.match_sequential(database_path=database_path, matching_options=pairing_options)
        return

    kwargs = {"database_path": database_path}
    if matching_options is not None:
        kwargs["matching_options"] = matching_options
    if verification_options is not None:
        kwargs["verification_options"] = verification_options
    if hasattr(pycolmap, "ExhaustivePairingOptions"):
        kwargs["pairing_options"] = pycolmap.ExhaustivePairingOptions()
    pycolmap.match_exhaustive(**kwargs)


def _run_sparse_mapping(
    pycolmap: Any,
    dataset: SceneDataset,
    artifacts: SceneArtifacts,
    config: ReconstructionConfig,
) -> tuple[Any, int]:
    mapping_options = getattr(pycolmap, "IncrementalPipelineOptions", lambda: None)()
    if mapping_options is not None:
        for attr, value in (
            ("multiple_models", True),
            ("max_num_models", 10),
            ("min_model_size", config.min_model_size),
            ("ba_refine_focal_length", False),
            ("ba_refine_principal_point", False),
            ("ba_refine_extra_params", False),
            ("ba_use_gpu", False),
        ):
            if hasattr(mapping_options, attr):
                setattr(mapping_options, attr, value)
        mapper = getattr(mapping_options, "mapper", None)
        if mapper is not None:
            if hasattr(mapper, "abs_pose_refine_focal_length"):
                mapper.abs_pose_refine_focal_length = False
            if hasattr(mapper, "abs_pose_refine_extra_params"):
                mapper.abs_pose_refine_extra_params = False

    kwargs = {
        "database_path": artifacts.database_path,
        "image_path": dataset.root_dir,
        "output_path": artifacts.sparse_root,
    }
    if mapping_options is not None:
        kwargs["options"] = mapping_options

    raw_result = pycolmap.incremental_mapping(**kwargs)
    reconstructions = _normalise_reconstructions(raw_result)
    if not reconstructions:
        raise ReconstructionError(
            f"COLMAP did not produce a sparse model for scene '{dataset.scene_name}'."
        )

    best_index, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: (_read_count(item[1], "num_reg_images"), _read_count(item[1], "num_points3D")),
    )
    if _read_count(best_reconstruction, "num_points3D") <= 0:
        raise ReconstructionError(
            f"COLMAP returned only empty sparse models for scene '{dataset.scene_name}'."
        )

    best_dir = artifacts.sparse_root / str(best_index)
    if best_dir.exists() and not any(best_dir.iterdir()):
        shutil.rmtree(best_dir)
    return best_reconstruction, len(reconstructions)


def _write_selected_model(reconstruction: Any, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    if hasattr(reconstruction, "write"):
        reconstruction.write(output_path)
        return
    raise ReconstructionError("Selected COLMAP reconstruction cannot be written back to disk.")


def _export_sparse_point_cloud(reconstruction: Any, output_path: Path) -> None:
    export_method = getattr(reconstruction, "export_PLY", None)
    if export_method is None:
        export_method = getattr(reconstruction, "export_ply", None)
    if export_method is None:
        raise ReconstructionError("Selected COLMAP reconstruction cannot export a PLY point cloud.")
    export_method(output_path)
    _assert_non_empty_file(output_path, "sparse point cloud")


def _run_dense_reconstruction(
    pycolmap: Any,
    dataset: SceneDataset,
    artifacts: SceneArtifacts,
    config: ReconstructionConfig,
    image_names: list[str],
) -> None:
    artifacts.dense_root.mkdir(parents=True, exist_ok=True)
    pycolmap.undistort_images(
        output_path=artifacts.dense_root,
        input_path=artifacts.selected_model_path,
        image_path=dataset.root_dir,
        image_names=image_names,
    )

    patch_match_options = getattr(pycolmap, "PatchMatchOptions", lambda: None)()
    if patch_match_options is not None and hasattr(patch_match_options, "max_image_size"):
        patch_match_options.max_image_size = config.max_image_size
    patch_kwargs = {"workspace_path": artifacts.dense_root}
    if patch_match_options is not None:
        patch_kwargs["options"] = patch_match_options
    pycolmap.patch_match_stereo(**patch_kwargs)

    fusion_options = getattr(pycolmap, "StereoFusionOptions", lambda: None)()
    fusion_kwargs = {
        "output_path": artifacts.fused_ply_path,
        "workspace_path": artifacts.dense_root,
    }
    if fusion_options is not None:
        fusion_kwargs["options"] = fusion_options
    pycolmap.stereo_fusion(**fusion_kwargs)


def _run_poisson_meshing(pycolmap: Any, artifacts: SceneArtifacts, config: ReconstructionConfig) -> None:
    poisson_options = getattr(pycolmap, "PoissonMeshingOptions", lambda: None)()
    if poisson_options is not None and hasattr(poisson_options, "num_threads"):
        poisson_options.num_threads = -1
    mesh_kwargs = {
        "input_path": artifacts.fused_ply_path,
        "output_path": artifacts.mesh_path,
    }
    if poisson_options is not None:
        mesh_kwargs["options"] = poisson_options
    pycolmap.poisson_meshing(**mesh_kwargs)


def _dense_support_reason(dense_mode: str) -> str | None:
    if dense_mode == "off":
        return "Dense reconstruction disabled by configuration."
    if dense_mode == "auto" and platform.system() == "Darwin":
        return "Dense reconstruction skipped: COLMAP PatchMatch stereo requires CUDA and is not supported on macOS."
    return None


def _normalise_reconstructions(raw_result: Any) -> dict[int, Any]:
    if isinstance(raw_result, dict):
        return {int(index): reconstruction for index, reconstruction in raw_result.items()}
    if raw_result is None:
        return {}
    if isinstance(raw_result, (list, tuple)):
        return {index: reconstruction for index, reconstruction in enumerate(raw_result)}
    if hasattr(raw_result, "size") and hasattr(raw_result, "get"):
        return {index: raw_result.get(index) for index in range(raw_result.size())}
    return {0: raw_result}


def _read_count(reconstruction: Any, attribute_name: str) -> int:
    attribute = getattr(reconstruction, attribute_name)
    value = attribute() if callable(attribute) else attribute
    return int(value)


def _assert_non_empty_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ReconstructionError(f"Expected non-empty {label} at {path}")


def _write_status_file(result: SceneRunResult) -> None:
    payload = {
        "scene_name": result.scene_name,
        "image_count": result.image_count,
        "matcher_mode": result.matcher_mode,
        "sparse_model_count": result.sparse_model_count,
        "registered_images": result.registered_images,
        "sparse_points": result.sparse_points,
        "dense_attempted": result.dense_attempted,
        "dense_succeeded": result.dense_succeeded,
        "mesh_succeeded": result.mesh_succeeded,
        "final_artifact_path": str(result.final_artifact_path),
        "warnings": result.warnings,
    }
    result.artifacts.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
