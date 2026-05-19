# Repository Implementation Report

Date: 2026-05-19

## 1. Executive Summary

This repository implements a modular point-cloud reconstruction pipeline for the STEM Games 2026 project data. The codebase is centered on four scene datasets:

- `Box`
- `Entrance`
- `Fountain`
- `Statue`

The implementation is not a single monolithic script. It is organized into a loader layer, a pipeline layer, output writers, visualization utilities, and a small reporting toolchain.

At a high level, the repository currently supports two processing modes:

1. `Known-pose scenes` (`Box`, `Entrance`): camera poses are parsed directly from metadata files and used as ground truth.
2. `Intrinsics-only scenes` (`Fountain`, `Statue`): camera intrinsics come from `K.txt`, while poses are expected to be estimated by MapAnything.

The main implemented runtime is:

1. Load scene metadata and images into normalized Python dataclasses.
2. Choose a pose source based on scene type.
3. Optionally run a full-scene reconstruction step for pose-free scenes.
4. Estimate per-image depth with MoGe-2.
5. Unproject valid pixels into world-space points.
6. Save per-image artifacts.
7. Fuse all views into one scene cloud.
8. Apply statistical outlier filtering.
9. Write the final fused point cloud to disk.

The repository already contains generated outputs for `Box` and `Entrance`, plus image-analysis outputs for all four scenes.

## 2. What Exists In The Repository

### Top level

- `compute_fov.py`
  Small standalone utility that parses `K.txt` files and computes horizontal and vertical field of view from the first image in each scene directory.

- `StemGames2026_ProjectTask/`
  Main package containing the pipeline, data loaders, reports, tests, outputs, and CLI scripts.

### Main package structure

- `run_pipeline.py`
  Entry-point CLI for running the full reconstruction pipeline.

- `visualize.py`
  CLI for inspecting generated point clouds and depth maps.

- `pointcloud/`
  Dataset schemas and scene loaders.

- `pipeline/`
  Core reconstruction pipeline, split into depth, pose, reconstruction, fusion, postprocessing, coordinate transforms, and file I/O.

- `reports/`
  Analysis scripts and markdown documentation.

- `tests/`
  Unit tests for loader behavior.

- `outputs/`
  Generated per-scene pipeline artifacts.

## 3. Data Model And Scene Normalization

The loader layer is intentionally model-agnostic. Its job is to normalize different raw scene formats into one consistent schema.

### Implemented dataclasses in `pointcloud/schemas.py`

- `CameraIntrinsics`
  Stores the 3x3 calibration matrix, image size, the source of calibration (`provided_k` or `derived_from_fov`), and optional FOV metadata.

- `CameraPose`
  Stores camera `position`, `forward`, `right`, and `up` vectors. It also exposes `camera_to_world_matrix()` in the repository's Unity-style convention.

- `SceneView`
  Represents one image view. Each view has a scene name, index, image path, intrinsics, optional pose, pose status, and metadata.

- `SceneDataset`
  Represents a whole scene folder. It stores all `SceneView` objects, the scene root, metadata file references, notes, and whether pose estimation is required.

### What the loaders support

There are two concrete loader implementations in `pointcloud/loaders.py`.

#### `KnownPoseSceneLoader`

Used when a scene directory contains `*Input.txt` metadata.

Implemented behavior:

- Reads the metadata text.
- Extracts the camera field of view from text.
- Derives camera intrinsics from horizontal FOV and image size.
- Parses one pose entry per image.
- Constructs `SceneView` objects with `pose_status="known"`.

This is the path used for:

- `Box`
- `Entrance`

#### `IntrinsicsOnlySceneLoader`

Used when a scene directory contains `K.txt`.

Implemented behavior:

- Parses a numeric 3x3 intrinsic matrix from `K.txt`.
- Computes horizontal FOV implied by `fx` and image width.
- Marks each image as requiring pose estimation.
- Preserves scene-specific reported FOV hints when available.

This is the path used for:

- `Fountain`
- `Statue`

### Loader robustness that is already implemented

- Image discovery is natural-sort aware, so `image2` comes before `image10`.
- Pose metadata parsing tolerates the known formatting inconsistency where `CamRight` or `CamUp` may omit the colon.
- Intrinsics-only scenes preserve both `K.txt`-derived FOV and any externally reported scene FOV hint.

### Scene-specific metadata behavior

- `Statue`
  The hard-coded reported horizontal FOV of `90.0` matches the calibration implied by `K.txt`.

- `Fountain`
  The hard-coded reported horizontal FOV of `84.0` does not match the FOV implied by `K.txt`. The implementation preserves that value as metadata only and does not override calibration.

## 4. The Main Runtime Entry Point

The main operational CLI is `run_pipeline.py`.

### What `run_pipeline.py` does

- Adds the repository root to `sys.path` so the package can be executed from inside `StemGames2026_ProjectTask/`.
- Loads all project scenes with `load_project_scenes()`.
- Accepts CLI options for scene selection, compute device, voxel size, and post-processing parameters.
- Builds a `PipelineConfig` per selected scene.
- Runs the scene through `PipelineRunner`.

### Config selection logic

The file defines `POSED_SCENES = {"Box", "Entrance"}`.

For posed scenes it constructs:

- `GroundTruthPoseProvider`
- `MoGe2DepthEstimator`
- `DepthFuser`
- `StatisticalPostProcessor`
- `reconstructor=None`

For unposed scenes it constructs:

- `MapAnythingPoseProvider`
- `MapAnythingReconstructor`
- `MoGe2DepthEstimator`
- `DepthFuser`
- `StatisticalPostProcessor`

The pose provider and reconstructor deliberately share the same MapAnything provider instance so that a single cached reconstruction result can be reused.

## 5. Pipeline Architecture

The core orchestration lives in `pipeline/runner.py`.

### `PipelineConfig`

The runtime is built from swappable interfaces:

- `pose_provider`
- `depth_estimator`
- `fuser`
- `post_processor`
- optional `reconstructor`

This is the main architectural decision in the repository. The code is designed so that individual stages can be replaced without changing loader or runner logic.

### `PipelineRunner.run_scene()` stage-by-stage

#### Stage 0: Output directory setup

For each scene the runner creates:

- `outputs/<scene>/depth_maps/`
- `outputs/<scene>/per_image_clouds/`
- `outputs/<scene>/pixel_maps/`

#### Stage 1: Pose provision

`self._cfg.pose_provider.provide(dataset)` is called first.

This produces one `EstimatedPose` per input image.

#### Stage 2: Optional scene reconstruction

If a reconstructor is configured, the runner invokes it before per-view depth estimation.

Current use:

- only enabled for unposed scenes
- intended to provide a global point cloud and pose-consistent scale hints

#### Stage 2b: Metric scale hints for posed scenes

If there is no reconstructor, the runner computes metric reference points by triangulating adjacent-view SIFT matches.

Implemented behavior in `_build_triangulation_hints()`:

- load grayscale image pairs
- detect SIFT features
- match with brute-force matcher and Lowe ratio test
- triangulate using known poses and intrinsics
- transform triangulated points into the target camera frame
- keep only points in front of the camera

These camera-frame points are later used to scale MoGe-2 depth predictions into metric space.

#### Stage 3: Per-view depth estimation and unprojection

For each `SceneView`:

1. Select a scale reference.
2. Run the configured depth estimator.
3. Load the RGB image.
4. Convert the pose into an OpenCV-style camera-to-world matrix.
5. Unproject valid depth pixels into world-space XYZ.
6. Save per-view artifacts.
7. Append a `PerViewResult` object.

Saved per-view artifacts are:

- `{stem}_depth.npy`
- `{stem}_depth.png`
- `{stem}.ply`
- `{stem}_pixel_map.npy`

#### Stage 4: Fusion

The configured fuser merges all per-view point clouds into a single scene cloud.

#### Stage 5: Post-processing

The configured post-processor removes outliers from the fused cloud.

#### Stage 6: Final write

The runner writes `outputs/<scene>/scene_cloud.ply`.

## 6. Implemented Stage Modules

### Coordinate handling in `pipeline/coords.py`

This module is critical because the repository mixes:

- Unity-style pose storage in the dataclasses
- OpenCV-style geometry during projection and unprojection

Implemented functions:

- `unity_pose_to_opencv_c2w()`
  Converts the repository's stored pose basis into an OpenCV-compatible right-handed camera-to-world matrix by flipping the Y axis.

- `build_intrinsics_matrix()`
  Converts the stored tuple-based intrinsics into a numpy matrix.

- `build_normalised_intrinsics()`
  Produces MoGe-style normalized intrinsics.

- `unproject_depth_to_world()`
  Turns valid depth pixels plus RGB values into world-space points and associated pixel coordinates.

- `project_points_to_camera()`
  Projects world points into an image and returns pixel coordinates, camera-space depth, and a validity mask.

### Pose providers

#### `GroundTruthPoseProvider`

Implemented in `pipeline/pose/ground_truth.py`.

Behavior:

- Valid only for datasets with `pose_source="provided"`.
- Returns the already loaded `CameraPose` values as `EstimatedPose` objects.
- Sets `confidence=1.0` and `source="ground_truth"`.

#### `MapAnythingPoseProvider`

Implemented in `pipeline/pose/map_anything.py`.

Behavior:

- Valid only for datasets with `pose_source="missing"`.
- Lazy-loads the MapAnything model.
- Converts all images to tensors.
- Passes image tensors and intrinsic matrices into `model.reconstruct()`.
- Caches the resulting scene points, camera matrices, and confidence values.
- Converts MapAnything camera matrices back into the repository's `CameraPose` format.

Important implementation detail:

- the provider caches the full reconstruction result so that downstream code can reuse it without another forward pass

### Scene reconstruction

#### `MapAnythingReconstructor`

Implemented in `pipeline/reconstruction/map_anything.py`.

Behavior:

- If the shared `MapAnythingPoseProvider` already has a cache, reuse it.
- Otherwise, run MapAnything reconstruction explicitly.
- Accept optional pose hints and convert them to OpenCV camera-to-world matrices.
- Extract the fused point cloud, per-view camera matrices, and confidences from the model output.

The extraction logic is defensive. It tries several possible attribute names and dictionary keys such as:

- `points`
- `point_cloud`
- `vertices`
- `xyz_rgb`

This shows that the MapAnything integration is written to tolerate model API variation.

### Depth estimation

#### `MoGe2DepthEstimator`

Implemented in `pipeline/depth/moge2.py`.

Behavior:

- Lazy-loads the MoGe model on first use.
- Auto-selects compute device in priority order: CUDA, then MPS, then CPU.
- Loads RGB image data and converts it to a tensor.
- Derives horizontal FOV from intrinsics for model inference.
- Calls `model.infer()`.
- Cleans invalid `inf` values in returned depth and point tensors.
- Optionally aligns predicted depth scale to metric references.

Scale alignment is implemented in `_align_scale()`.

It works by:

1. Projecting reference 3D points back into image coordinates.
2. Sampling the predicted model depth at those pixels.
3. Computing metric-depth to model-depth ratios.
4. Using the median ratio as a scale factor.

This is the main bridge between affine-invariant monocular depth prediction and metric scene geometry.

### Fusion

#### `DepthFuser`

Implemented in `pipeline/fusion/depth_fuser.py`.

Behavior:

- concatenates all per-view point clouds
- filters points by depth range
- keeps track of which view and pixel each point came from
- performs voxel downsampling with deterministic first-point retention

This is a pure numpy implementation. It does not rely on Open3D.

### Post-processing

#### `StatisticalPostProcessor`

Implemented in `pipeline/postprocess/statistical.py`.

Behavior:

- builds a `scipy.spatial.KDTree`
- computes mean distance to the nearest `k` neighbors for each point
- removes points whose mean distance is greater than `mean + std_ratio * std`

This is a conventional statistical outlier filter.

## 7. Output Writers And Artifact Formats

The repository writes several file formats through small dedicated helpers.

### `pipeline/io/depth_writer.py`

Implemented outputs:

- `.npy` metric depth arrays
- false-color `.png` depth previews

The PNG writer computes display range from valid depth percentiles and colors valid depths with a JET-like palette.

### `pipeline/io/ply_writer.py`

Writes binary little-endian PLY files with vertex fields:

- `x`
- `y`
- `z`
- `red`
- `green`
- `blue`

### `pipeline/io/pixel_map.py`

Writes structured numpy arrays with fields:

- `view_idx`
- `row`
- `col`

This provides a mapping from 3D points back to source image pixels.

## 8. Visualization And Reporting Tools

### `visualize.py`

This script is a practical inspection tool for generated artifacts.

Implemented features:

- list available output scenes
- show the fused scene cloud in a browser via Plotly
- show an individual per-image cloud
- show all per-image clouds color-coded by view
- compare two per-image clouds side by side
- build and open a depth-map contact sheet

Implementation notes:

- PLY reading is implemented locally in numpy.
- Large clouds are randomly subsampled to keep browser rendering responsive.
- On macOS, depth contact sheets are opened with the `open` command.

### `reports/analyze_images.py`

This is a dataset characterization utility rather than part of the reconstruction runtime.

Implemented metrics per image:

- brightness mean
- contrast standard deviation
- Laplacian variance as a blur proxy
- edge energy
- saturation mean

Implemented outputs:

- `reports/generated/scene_analysis.json`
- one contact sheet image per scene

### `compute_fov.py`

This is a standalone calibration sanity-check script.

Implemented behavior:

- parse a `K.txt` file
- load the first image in a directory
- compute horizontal and vertical FOV
- print the results for `Statue` and `Fountain`

## 9. What Has Already Been Generated In This Workspace

At the time of inspection, the repository already contains:

### Pipeline outputs

- `outputs/Box/`
- `outputs/Entrance/`

Each of those scenes already has:

- `depth_maps/`
- `per_image_clouds/`
- `pixel_maps/`
- `scene_cloud.ply`

### Reporting outputs

`reports/generated/` currently contains:

- `scene_analysis.json`
- `box_contact_sheet.jpg`
- `entrance_contact_sheet.jpg`
- `fountain_contact_sheet.jpg`
- `statue_contact_sheet.jpg`

This means the image-analysis pipeline has been run for all scenes, while the reconstruction pipeline has only been run for `Box` and `Entrance` in the checked-in workspace state.

## 10. Test Coverage And Validation Status

The repository contains one implemented test module: `tests/test_loaders.py`.

What is covered:

- loading `Box` as a known-pose scene
- robustness to metadata formatting variations
- loading `Fountain` as an intrinsics-only scene
- preservation of the `Statue` FOV consistency hint
- discovery of all four scenes through the project loader

What is not covered by automated tests in the repository:

- `PipelineRunner`
- MoGe-2 integration
- MapAnything integration
- triangulation hint generation
- voxel fusion
- statistical post-processing
- PLY writing and reading
- depth-map writing
- visualization CLI behavior

So the implemented architecture is modular, but automated verification is currently concentrated almost entirely in the data-loading layer.

Current verification result in this workspace:

- `tests/test_loaders.py` passes when run with the repository root on `PYTHONPATH`
- `pytest` is not installed in the checked virtual environment, so the test module currently runs most directly via the standard-library unittest entry path

## 11. Inferred Runtime Dependencies

There is no `pyproject.toml`, `requirements.txt`, or other package manifest in the repository root at the time of inspection. Dependencies must therefore be inferred from imports.

### Core dependencies used directly in code

- `numpy`
- `Pillow`

### Pipeline and model dependencies

- `torch`
- `moge`
- `huggingface_hub`
- `mapanything`
- `opencv-python` or equivalent `cv2`
- `scipy`

### Visualization dependency

- `plotly`

### Standard-library utilities used heavily

- `argparse`
- `pathlib`
- `dataclasses`
- `json`
- `tempfile`
- `webbrowser`

This means the repository is currently code-complete enough to express the intended workflow, but not yet environment-complete in the sense of a reproducible dependency declaration.

## 12. Architectural Strengths

The strongest implemented qualities in the repository are:

1. Clear separation between scene loading and reconstruction.
2. Interface-based pipeline stages that can be swapped independently.
3. Support for both known-pose and pose-free scenes.
4. Explicit artifact writing at both per-view and fused-scene levels.
5. Practical inspection tooling for outputs.
6. Defensive integration code around unstable external model APIs.

## 13. Current Limitations And Caveats

The repository is functional, but several limitations are visible from the implementation.

### Environment reproducibility is incomplete

There is no declared dependency manifest, so a new user cannot recreate the runtime environment from repository metadata alone.

### Test coverage is narrow

The most expensive and highest-risk parts of the system, namely model inference, fusion, and geometry processing, are not covered by tests.

### External integrations are assumed, not vendored

The MoGe and MapAnything paths depend on third-party packages and their model APIs being available and compatible.

### Some integration code is intentionally provisional

The MapAnything extraction helpers probe multiple possible field names, which is useful, but also indicates that the integration is adapting to uncertain output schemas rather than locking to a single known API.

### Unposed-scene scaling logic remains somewhat experimental

The runner computes scale hints for unposed scenes by projecting the global MapAnything cloud and reusing the visible subset as a reference for MoGe-2. The intention is clear, but this part of the pipeline is less direct and less battle-tested than the known-pose triangulation path.

## 14. End-To-End Behavior By Scene Type

### `Box` and `Entrance`

End-to-end behavior:

1. Parse provided metadata and derive intrinsics from FOV.
2. Reuse provided camera poses.
3. Triangulate adjacent-view SIFT matches to get metric reference points.
4. Run MoGe-2 depth on each image.
5. Scale depth predictions with triangulated references.
6. Unproject to world coordinates.
7. Save per-image clouds and depth maps.
8. Fuse and filter into one scene cloud.

This is the most complete and least speculative path in the repository.

### `Fountain` and `Statue`

End-to-end behavior:

1. Parse `K.txt` intrinsics.
2. Run MapAnything to recover poses and a global cloud.
3. Reuse the cached MapAnything result in the reconstructor.
4. Select visible global points per view as metric hints.
5. Run MoGe-2 per image.
6. Fuse and post-process as above.

This path is implemented in code, but the checked-in workspace does not yet contain corresponding `outputs/Fountain/` or `outputs/Statue/` artifacts.

## 15. Bottom-Line Assessment

What is implemented today is more than a prototype script. It is a small but well-structured reconstruction framework with:

- a normalized scene ingestion layer
- a configurable multi-stage pipeline
- two pose acquisition strategies
- one dense depth backend
- one fusion backend
- one post-processing backend
- output serialization utilities
- dataset-analysis tooling
- visualization tooling

The repository is strongest in structure and workflow decomposition. It is weaker in packaging, automated validation, and stabilization of the model-backed paths for the unposed scenes.

If the question is "what actually works today," the answer is:

- data loading is implemented and tested
- the known-pose reconstruction path is implemented and has generated outputs in the workspace
- visualization and image-analysis tooling are implemented and usable
- the MapAnything-backed path is implemented in code, but it is still more integration-heavy and less verified than the posed-scene path