# Point Cloud Reconstruction Approach

## What Was Added

- A model-agnostic loader layer in `StemGames2026_ProjectTask/pointcloud/`.
- A reproducible image-analysis script in `StemGames2026_ProjectTask/reports/analyze_images.py`.
- Generated summaries and contact sheets in `StemGames2026_ProjectTask/reports/generated/`.

The loader output is intentionally neutral with respect to the reconstruction model. Each dataset is normalized into the same `SceneDataset -> SceneView` schema, with intrinsics always available and pose fields either filled (`Box`, `Entrance`) or left empty for downstream estimation (`Fountain`, `Statue`).

## Loader Design

### Unified Scene Schema

Each image becomes a `SceneView` with:

- `image_path`
- `intrinsics`
- `pose` or `None`
- `pose_status`
- per-view metadata

Each folder becomes a `SceneDataset` with:

- `scene_name`
- `views`
- `pose_source`
- `metadata_files`
- scene notes

This keeps the loader reusable for multiple downstream approaches:

- direct triangulation from known poses
- classical SfM + MVS
- learned correspondence models
- learned dense reconstruction models
- hybrid pipelines that mix classical and learned stages

### Supported Metadata Types

- `Box` and `Entrance`: parsed by `KnownPoseSceneLoader`
- `Statue` and `Fountain`: parsed by `IntrinsicsOnlySceneLoader`

Important implementation detail: the known-pose parser is tolerant to small formatting inconsistencies in the provided files, including lines where `CamRight` or `CamUp` omit the trailing colon.

For the intrinsics-only scenes, the loader now keeps two FOV signals when available:

- the horizontal FOV implied by `K.txt`
- the separately reported horizontal FOV hint

For `Statue`, the reported 90 degree FOV matches `K.txt` exactly. For `Fountain`, the reported value of about 84 degrees does not match the horizontal FOV implied by `K.txt` at the provided resolution, so it is preserved only as an external capture hint and not used to override calibration.

## Quantitative Scene Summary

Values below come from `reports/generated/scene_analysis.json`.

| Scene | Images | Resolution | Poses | Median Adjacent Correlation | Median Contrast | Median Edge Energy |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Box | 12 | 1920x1080 | known | 0.781 | 0.085 | 0.0150 |
| Entrance | 12 | 1920x1080 | known | 0.463 | 0.178 | 0.0248 |
| Fountain | 11 | 3072x2048 | estimated later | 0.634 | 0.143 | 0.0224 |
| Statue | 18 | 1920x1080 | estimated later | 0.515 | 0.114 | 0.0145 |

Interpretation:

- Higher adjacent correlation usually means stronger overlap and smaller frame-to-frame change.
- Higher contrast and edge energy usually indicate more recoverable local structure for matching and dense reconstruction.
- These numbers do not replace visual inspection, but they are useful for prioritizing models and tuning thresholds.

## Scene-by-Scene Assessment

### Box

Visual characteristics:

- Small object with strong geometric edges and a nearly complete orbit around it.
- A calibration-style checker marker appears on the crate.
- Background sand and wall are smoother and less informative than the object itself.
- Lighting is stable and the viewpoint progression is very regular.

Why this scene is favorable:

- Camera poses are already known.
- Overlap is high across adjacent frames.
- The object has clear corners, planar faces, and repeated observations of the same edges.

Main risks:

- Large texture-poor regions in sand and wall can create weak or ambiguous matches.
- The crate has repeated slat patterns, so edge-only matching can drift if the background dominates.

Best approach:

- Skip pose estimation entirely.
- Use the provided pose vectors and derived intrinsics to run calibrated multi-view triangulation or dense stereo directly.
- Segment or crop around the crate before dense reconstruction so background texture does not dominate matching.
- Use the checker marker and crate corners as sanity checks for reprojection error.

Expected result:

- This should be the easiest dataset to turn into a clean object-centric point cloud.

### Entrance

Visual characteristics:

- Larger scene with foreground statue, doorway, car, steps, wall reliefs, and deep background.
- Strong shadows and strong color contrast produce many edges.
- The viewpoint arc is broader and less redundant than `Box`.
- There is meaningful occlusion change as the camera moves around the scene.

Why this scene is favorable:

- Camera poses are already known.
- Rich geometry and high edge content support correspondence search.
- Different depth layers should give a visibly rich point cloud.

Main risks:

- Occlusions change a lot between views.
- Large depth variation makes naive matching harder than in `Box`.
- Sky and extreme sunlit regions can create photometric inconsistency.

Best approach:

- Use the provided poses directly.
- Reconstruct with a multi-scale calibrated stereo pipeline rather than a pure sparse triangulation pipeline.
- Mask sky and, if the goal is the architectural entrance rather than the entire scene, consider also masking parts of the far background.
- Favor robust patch-based or learned dense matching over purely local block matching.

Expected result:

- This should produce a richer but noisier cloud than `Box`, with the best quality around the doorway, statue, and car.

### Fountain

Visual characteristics:

- Real image set with detailed masonry, facade texture, and ornamented geometry.
- The object is mostly seen from a frontal-to-oblique arc rather than a full orbit.
- There are calibration markers near the object.
- The basin and gold figure may introduce reflective or photometrically unstable regions.

Why this scene is favorable:

- Resolution is the highest of all datasets.
- Surface detail and masonry texture provide strong natural features.
- Intrinsics are already known through `K.txt`.
- The reported camera FOV can still be kept as auxiliary metadata for downstream experiments.

Main risks:

- Poses are not given and must be estimated first.
- View coverage is incomplete, so back-side geometry will remain weak or missing.
- Reflective or water-like surfaces can break dense stereo assumptions.
- The reported FOV of about 84 degrees is inconsistent with the horizontal FOV implied by `K.txt`, so any model that consumes both needs one clear authority.

Best approach:

- First run an intrinsics-aware SfM stage with bundle adjustment.
- Treat `K.txt` as authoritative calibration and the 84 degree value as a loose acquisition note unless another convention is confirmed.
- Use robust feature extraction and matching, then refine poses before any dense reconstruction.
- Exclude the most reflective basin or water regions if they destabilize matching.
- Once poses are stable, run a dense MVS or a strong learned multi-view depth model and fuse depth maps into a point cloud.

Expected result:

- This is likely the best candidate for a detailed real-world point cloud on the visible front and side surfaces, but not for full 360-degree completeness.

### Statue

Visual characteristics:

- Nearly full orbit around a standing figure.
- Strong direct sunlight creates hard shadows.
- The statue body is comparatively smooth and low-texture.
- Large portions of the scene are sand or wall, both of which are weakly distinctive.
- Calibration markers appear in the environment and can help stabilize geometry.

Why this scene is favorable:

- Coverage is much closer to a complete 360-degree object sweep.
- Intrinsics are known.
- The reported 90 degree FOV is consistent with `K.txt`, so calibration metadata is internally coherent.
- The silhouette changes significantly across the orbit, which is useful if feature matching alone is weak.

Main risks:

- Low local texture on the statue makes correspondence estimation difficult.
- Harsh shadows may move relative to shape cues and can be mistaken for geometry.
- Background surfaces are repetitive and can dominate keypoint extraction.

Best approach:

- Start with intrinsics-aware pose estimation, but expect classical local features alone to be fragile.
- Use either stronger learned correspondences or a hybrid strategy that combines feature matching with object masking.
- If the goal is the statue itself, segment the statue before dense reconstruction or fusion.
- Consider silhouette-aware constraints or visual-hull style priors if sparse matching on the statue remains unstable.

Expected result:

- This scene can produce a more complete object cloud than `Fountain`, but pose estimation will be harder and the final surface may require stronger regularization.

## Recommended Modular Pipeline

To keep model experimentation isolated from dataset parsing, split the system into five modules.

### 1. Data Loader

Responsibility:

- read images
- normalize intrinsics
- attach known poses when available
- flag scenes that require pose estimation

Current implementation:

- `StemGames2026_ProjectTask/pointcloud/loaders.py`

### 2. Pose Provider

Responsibility:

- return camera extrinsics for every `SceneView`

Swappable implementations:

- `GroundTruthPoseProvider` for `Box` and `Entrance`
- `SfMPoseProvider` for classical feature matching + bundle adjustment
- `LearnedPoseProvider` for models that jointly estimate correspondences or camera geometry

### 3. Dense Reconstruction Provider

Responsibility:

- turn calibrated images into depth maps, point predictions, or pairwise triangulated structure

Swappable implementations:

- calibrated stereo / PatchMatch-style MVS
- COLMAP/OpenMVS-style dense reconstruction
- learned multi-view depth models
- direct point prediction models

### 4. Fusion Provider

Responsibility:

- merge per-view depth or point estimates into one cloud
- remove outliers and low-confidence points

Swappable implementations:

- simple depth fusion
- TSDF-style fusion
- confidence-weighted point merge

### 5. Post-Processing Provider

Responsibility:

- crop scene bounds
- filter background
- estimate normals
- optionally mesh or colorize the final point cloud

Swappable implementations:

- object-only masking
- statistical outlier removal
- voxel downsampling
- normal estimation and meshing

## Best Overall Strategy For The Best Point Cloud

If the goal is the best possible result rather than one universal method, the strongest strategy is dataset-specific:

- `Box`: use known poses plus object-centric calibrated MVS or direct multi-view triangulation.
- `Entrance`: use known poses plus robust dense stereo with masking for sky and optionally far background.
- `Fountain`: use intrinsics-aware SfM first, then dense MVS on the validated camera graph.
- `Statue`: use intrinsics-aware pose estimation plus segmentation-aware dense reconstruction, and be prepared to try learned correspondences if classical matching is unstable.

If one modular benchmark pipeline must be reused across all four scenes, use this order:

1. Load with the current loader layer.
2. If poses are missing, estimate poses with known intrinsics.
3. Run object/background masking for the target structure.
4. Run dense multi-view depth or a learned multi-view reconstruction model.
5. Fuse depths or points with confidence thresholds.
6. Apply outlier removal and optional voxel downsampling.

## Priority Order For Model Experiments

If time is limited, try models in this order:

1. `Box` with known poses to validate the dense fusion stage.
2. `Entrance` with known poses to validate the same stage on a deeper and more cluttered scene.
3. `Fountain` to validate pose estimation on a textured real scene.
4. `Statue` to stress-test low-texture matching and segmentation.

That order isolates failure causes cleanly:

- failures on `Box` or `Entrance` point to the dense reconstruction stage
- failures on `Fountain` point mostly to pose estimation or dense fusion under real imagery
- failures on `Statue` point to texture poverty, masking, or correspondence robustness