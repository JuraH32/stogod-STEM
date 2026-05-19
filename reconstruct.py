from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from StemGames2026_ProjectTask.pipeline.reconstruction.colmap import (
    ReconstructionConfig,
    ReconstructionError,
    discover_project_paths,
    load_requested_scenes,
    run_scene_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portable PyCOLMAP scene reconstruction for all STEM Games datasets."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene names to reconstruct. Defaults to all scenes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for reconstruction outputs. Defaults to StemGames2026_ProjectTask/outputs/colmap.",
    )
    parser.add_argument(
        "--matcher-mode",
        default="auto",
        choices=["auto", "exhaustive", "sequential"],
        help="Image matching strategy. 'auto' uses exhaustive matching for these small scene batches.",
    )
    parser.add_argument(
        "--dense-mode",
        default="auto",
        choices=["auto", "off", "required"],
        help="Dense reconstruction policy. 'auto' skips CUDA-only dense stereo where unsupported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing output workspace for the selected scene(s).",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Maximum image size used during feature extraction.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum number of SIFT features per image.",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=5,
        help="Neighbor overlap when sequential matching is selected.",
    )
    parser.add_argument(
        "--min-model-size",
        type=int,
        default=2,
        help="Minimum sparse model size that COLMAP should keep.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = discover_project_paths(__file__)
        config = ReconstructionConfig(
            output_root=args.output_root or paths.output_root,
            matcher_mode=args.matcher_mode,
            dense_mode=args.dense_mode,
            overwrite=args.overwrite,
            max_image_size=args.max_image_size,
            max_num_features=args.max_num_features,
            sequential_overlap=args.sequential_overlap,
            min_model_size=args.min_model_size,
        )
        datasets = load_requested_scenes(paths.project_root, args.scenes)

        print(f"Project root: {paths.project_root}")
        print(f"Dataset root: {paths.test_images_root}")
        print(f"Output root:  {config.output_root}")

        results = run_scene_batch(datasets, config)
    except ReconstructionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    failed = False
    for result in results:
        print(f"\n[{result.scene_name}]")
        print(f"  images:             {result.image_count}")
        print(f"  matcher:            {result.matcher_mode}")
        print(f"  sparse models:      {result.sparse_model_count}")
        print(f"  registered images:  {result.registered_images}")
        print(f"  sparse points:      {result.sparse_points}")
        print(f"  dense succeeded:    {result.dense_succeeded}")
        print(f"  mesh succeeded:     {result.mesh_succeeded}")
        print(f"  final artifact:     {result.final_artifact_path}")
        print(f"  summary:            {result.artifacts.status_path}")
        for warning in result.warnings:
            print(f"  warning:            {warning}")
        failed = failed or not result.is_valid_output

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
