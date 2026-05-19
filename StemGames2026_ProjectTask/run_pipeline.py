"""
Top-level entry point for the MoGe-2 + MapAnything point cloud pipeline.

Usage (from inside StemGames2026_ProjectTask/):
    python run_pipeline.py                        # process all 4 scenes
    python run_pipeline.py --scenes Box Entrance  # process specific scenes
    python run_pipeline.py --device cpu           # force CPU inference
    python run_pipeline.py --voxel-size 0.005     # finer voxel grid

Outputs are written to outputs/{scene_name}/:
    depth_maps/          {stem}_depth.npy + {stem}_depth.png per image
    per_image_clouds/    {stem}.ply per image (world-space)
    pixel_maps/          {stem}_pixel_map.npy per image
    scene_cloud.ply      merged scene point cloud
    scene_mesh.ply       best-effort triangle mesh generated from the scene cloud
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow running from inside StemGames2026_ProjectTask/ by adding the repo root
# (one level up) to sys.path so that `StemGames2026_ProjectTask` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from StemGames2026_ProjectTask.pointcloud.loaders import load_project_scenes
from StemGames2026_ProjectTask.pipeline.depth.moge2 import MoGe2DepthEstimator
from StemGames2026_ProjectTask.pipeline.fusion.depth_fuser import DepthFuser
from StemGames2026_ProjectTask.pipeline.mesh import TrimeshVoxelMesher
from StemGames2026_ProjectTask.pipeline.pose.ground_truth import GroundTruthPoseProvider
from StemGames2026_ProjectTask.pipeline.pose.colmap_pose import ColmapPoseProvider
from StemGames2026_ProjectTask.pipeline.postprocess.statistical import StatisticalPostProcessor
from StemGames2026_ProjectTask.pipeline.reconstruction.colmap_scene import ColmapReconstructor
from StemGames2026_ProjectTask.pipeline.runner import PipelineConfig, PipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent   # StemGames2026_ProjectTask/
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

POSED_SCENES = {"Box", "Entrance"}


def build_config(
    scene_name: str,
    output_root: Path,
    device: str,
    voxel_size: float,
    nb_neighbors: int,
    std_ratio: float,
) -> PipelineConfig:
    depth_estimator = MoGe2DepthEstimator(device=device)  # None → auto-detect
    # No depth ceiling — world units vary by scene (Unity may use cm rather than m).
    # Statistical post-processing removes depth outliers instead.
    fuser = DepthFuser(voxel_size=voxel_size, min_depth=0.0, max_depth=1e9)
    post_processor = StatisticalPostProcessor(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    mesher = TrimeshVoxelMesher(pitch=max(voxel_size, 1e-4))

    if scene_name in POSED_SCENES:
        return PipelineConfig(
            output_root=output_root,
            pose_provider=GroundTruthPoseProvider(),
            depth_estimator=depth_estimator,
            fuser=fuser,
            post_processor=post_processor,
            mesher=mesher,
            reconstructor=None,
        )
    else:
        # ColmapPoseProvider and ColmapReconstructor share the same instance so
        # the SfM reconstruction is computed only once.
        provider = ColmapPoseProvider(colmap_output_root=output_root / "colmap")
        return PipelineConfig(
            output_root=output_root,
            pose_provider=provider,
            depth_estimator=depth_estimator,
            fuser=fuser,
            post_processor=post_processor,
            mesher=mesher,
            reconstructor=ColmapReconstructor(pose_provider=provider),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MoGe-2 + MapAnything point cloud pipeline")
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene names to process (default: all). E.g. --scenes Box Entrance"
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device for MoGe-2 inference (default: auto-detect cuda→mps→cpu)"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.01,
        help="Voxel grid size in metres for scene cloud downsampling (default: 0.01)"
    )
    parser.add_argument(
        "--nb-neighbors", type=int, default=20,
        help="Number of neighbours for statistical outlier removal (default: 20)"
    )
    parser.add_argument(
        "--std-ratio", type=float, default=2.0,
        help="Standard deviation ratio for outlier removal threshold (default: 2.0)"
    )
    parser.add_argument(
        "--output-root", type=Path, default=OUTPUT_ROOT,
        help="Root directory for all outputs"
    )
    args = parser.parse_args()

    datasets = load_project_scenes(PROJECT_ROOT)

    scenes_to_run = args.scenes if args.scenes else list(datasets.keys())
    unknown = set(scenes_to_run) - set(datasets.keys())
    if unknown:
        parser.error(f"Unknown scene(s): {sorted(unknown)}. Available: {sorted(datasets.keys())}")

    for scene_name in scenes_to_run:
        dataset = datasets[scene_name]
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}  ({len(dataset.views)} views, pose_source={dataset.pose_source})")
        print(f"{'='*60}")

        cfg = build_config(
            scene_name=scene_name,
            output_root=args.output_root,
            device=args.device,
            voxel_size=args.voxel_size,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )
        runner = PipelineRunner(cfg)
        result = runner.run_scene(dataset)

        print(f"\n  Done: {len(result.scene_points):,} points in scene cloud")
        print(f"  Output cloud: {result.scene_ply_path}")
        if result.scene_mesh_path is not None:
            print(f"  Output mesh:  {result.scene_mesh_path}")
        elif result.mesh_warning is not None:
            print(f"  Output mesh:  skipped ({result.mesh_warning})")


if __name__ == "__main__":
    main()
