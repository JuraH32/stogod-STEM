# STEM Games 2026 Point-Cloud Pipeline

This repository contains a multi-stage 3D reconstruction workflow for the STEM Games datasets.

At a high level, the project can:

- load the four scenes (`Box`, `Entrance`, `Fountain`, `Statue`)
- estimate or load camera poses
- predict per-image depth with MoGe-2
- unproject each depth map into world-space point clouds
- fuse the per-image clouds into one scene cloud
- generate a best-effort triangle mesh from the fused cloud
- preview the resulting point clouds and meshes in the browser

The current verified workflow uses the checked-in virtual environment and runs the Python entry points directly. In practice, this has been more reliable than `uv run` for the heavy pipeline jobs.

## Repository Layout

- `StemGames2026_ProjectTask/run_pipeline.py`: full point-cloud pipeline, including mesh generation at the end
- `StemGames2026_ProjectTask/run_meshing.py`: generate or regenerate `scene_mesh.ply` from an existing `scene_cloud.ply`
- `StemGames2026_ProjectTask/visualize.py`: preview scene clouds, per-image clouds, and depth maps
- `StemGames2026_ProjectTask/visualize_mesh.py`: preview generated scene meshes
- `reconstruct.py`: separate PyCOLMAP reconstruction utility for COLMAP-only reconstruction outputs
- `StemGames2026_ProjectTask/outputs/`: generated artifacts for each scene
- `StemGames2026_ProjectTask/tests/`: focused standard-library test scripts

## Dataset Summary

The project works on four scenes:

- `Box`
- `Entrance`
- `Fountain`
- `Statue`

The pipeline treats them differently:

- `Box` and `Entrance` are posed scenes. They use provided ground-truth camera poses.
- `Fountain` and `Statue` are unposed scenes. They use the COLMAP-based reconstruction path to estimate poses before depth fusion.

## Environment Setup

The commands below assume you are at the repository root:

```bash
cd /Users/jurahostic/Documents/STEM/stogod-STEM
```

The known-good interpreter is:

```bash
.venv/bin/python
```

The heavy pipeline commands should be run with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE
```

For example:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py --scenes Box
```

### Required Runtime Packages

The project code relies on these Python packages in the active environment:

- `numpy`
- `scipy`
- `Pillow`
- `plotly`
- `torch`
- `moge`
- `opencv-python`
- `trimesh`
- `scikit-image`

Optional but important for the unposed reconstruction path:

- `pycolmap`

If mesh generation is skipped with a message about `skimage`, install `scikit-image` into `.venv`.

## Recommended Workflow

There are two practical ways to run the project:

1. Run the full point-cloud pipeline, which now also tries to create a scene mesh.
2. If you already have `scene_cloud.ply` files, run standalone meshing afterward with `run_meshing.py`.

The second path is useful when you want to regenerate meshes without paying the full depth-estimation cost again.

## Run The Full Pipeline

From the repository root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py
```

To run all four scenes explicitly:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py --scenes Box Entrance Fountain Statue
```

To run a single scene:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py --scenes Box
```

Useful options:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py --scenes Fountain --device cpu
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py --scenes Statue --voxel-size 0.005
```

### What `run_pipeline.py` Does

The pipeline stages are:

1. Pose provision
   - posed scenes use provided poses
   - unposed scenes use the COLMAP-based pose provider
2. Optional scene reconstruction for unposed scenes
   - used to improve scale hints and pose reuse
3. Per-view depth estimation and unprojection
   - MoGe-2 predicts depth
   - valid pixels are unprojected into world-space point clouds
4. Fusion
   - per-view clouds are merged into a single scene cloud
5. Post-processing
   - statistical filtering removes outlier points
6. Scene meshing
   - the fused scene cloud is converted into a best-effort triangle mesh

### Expected Outputs

After a successful run, each scene folder under `StemGames2026_ProjectTask/outputs/{scene}/` should contain:

- `depth_maps/`
- `per_image_clouds/`
- `pixel_maps/`
- `scene_cloud.ply`
- `scene_mesh.ply`

## Regenerate Meshes From Existing Scene Clouds

If the scene clouds already exist and you only want meshes, use the standalone meshing command.

Run all scenes:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_meshing.py --scenes Box Entrance Fountain Statue
```

Run a subset:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_meshing.py --scenes Box Statue
```

Tune meshing parameters:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_meshing.py --scenes Box --pitch 0.01 --max-points 100000
```

### Meshing Notes

The mesher uses `trimesh.voxel.ops.points_to_marching_cubes`, but it is not a fixed-resolution mesher anymore.

It now:

- adapts pitch to the observed nearest-neighbor spacing in the cloud
- deterministically downsamples very large point clouds before marching cubes

This matters because some scene clouds are extremely large and sparse in world units. A fixed pitch of `0.01` is far too fine for scenes like `Box` and `Entrance`.

## Preview Point Clouds

Use the point-cloud visualizer for scene clouds, per-image clouds, and depth maps.

List available outputs:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize.py --list
```

Show a scene cloud:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize.py Box
```

Show one per-image cloud:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize.py Box --image 3
```

Show all per-image clouds:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize.py Box --all-images
```

Show the depth-map contact sheet:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize.py Box --depth
```

Compare two per-image clouds:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize.py Box --compare 2 5
```

## Preview Meshes

Use the separate mesh viewer for generated triangle meshes.

List mesh availability:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize_mesh.py --list
```

Open a mesh preview:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize_mesh.py Box
```

Overlay the scene cloud on top of the mesh:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize_mesh.py Box --show-cloud
```

Adjust mesh opacity:

```bash
.venv/bin/python StemGames2026_ProjectTask/visualize_mesh.py Statue --opacity 0.75
```

The viewer writes a temporary HTML file and opens it in the browser.

## COLMAP-Only Reconstruction Utility

`reconstruct.py` is a separate utility for portable PyCOLMAP reconstruction. It is useful when you want to inspect the COLMAP path independently from the MoGe-based point-cloud pipeline.

Run all scenes:

```bash
.venv/bin/python reconstruct.py --scenes Box Entrance Fountain Statue --dense-mode auto --overwrite
```

Run one scene:

```bash
.venv/bin/python reconstruct.py --scenes Fountain --dense-mode auto --overwrite
```

This writes outputs under:

```text
StemGames2026_ProjectTask/outputs/colmap/
```

## Tests

The repository currently uses focused standard-library test scripts instead of `pytest`.

Known-good commands:

```bash
export PYTHONPATH=$PYTHONPATH:.
.venv/bin/python StemGames2026_ProjectTask/tests/test_loaders.py
.venv/bin/python StemGames2026_ProjectTask/tests/test_trimesh_mesher.py
.venv/bin/python StemGames2026_ProjectTask/tests/test_pipeline_runner_meshing.py
```

## Troubleshooting

### `uv run` fails or aborts

If `uv run` exits with an error or abort signal on long jobs, use the direct interpreter path shown above instead:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python StemGames2026_ProjectTask/run_pipeline.py --scenes Box
```

### The command says it cannot find `run_pipeline.py`

Make sure you include the path from the repository root:

```bash
StemGames2026_ProjectTask/run_pipeline.py
```

If you `cd` into `StemGames2026_ProjectTask/`, then the script path becomes just `run_pipeline.py`.

### Mesh generation is skipped

Common causes:

- `scikit-image` is missing
- the scene cloud is too sparse or too small after filtering
- the previous fixed pitch was too fine for the scene scale

The current mesher already handles large scene scale much better, but if needed you can rerun standalone meshing with a different `--max-points` or `--pitch`.

### Unposed scenes fail earlier than meshing

The likely problem is in the COLMAP reconstruction or pose-estimation path rather than in meshing. Inspect:

- `StemGames2026_ProjectTask/pipeline/pose/colmap_pose.py`
- `StemGames2026_ProjectTask/pipeline/reconstruction/colmap_scene.py`
- `StemGames2026_ProjectTask/pipeline/depth/moge2.py`

### The mesh preview opens but looks empty

Check that:

- `scene_mesh.ply` exists and is non-empty
- the browser finished loading the generated HTML file
- the scene scale is large enough that you may need to rotate or zoom out after opening

## Current Verified Output State

At the time this README was added, the following artifacts had been verified as present and non-empty:

- `StemGames2026_ProjectTask/outputs/Box/scene_cloud.ply`
- `StemGames2026_ProjectTask/outputs/Box/scene_mesh.ply`
- `StemGames2026_ProjectTask/outputs/Entrance/scene_cloud.ply`
- `StemGames2026_ProjectTask/outputs/Entrance/scene_mesh.ply`
- `StemGames2026_ProjectTask/outputs/Fountain/scene_cloud.ply`
- `StemGames2026_ProjectTask/outputs/Fountain/scene_mesh.ply`
- `StemGames2026_ProjectTask/outputs/Statue/scene_cloud.ply`
- `StemGames2026_ProjectTask/outputs/Statue/scene_mesh.ply`

If you rerun the pipeline or standalone meshing, these files will be refreshed.