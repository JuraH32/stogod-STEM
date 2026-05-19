from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from StemGames2026_ProjectTask.pointcloud import load_project_scenes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated"
THUMBNAIL_SIZE = (320, 180)
CONTACT_SHEET_COLUMNS = 4


@dataclass(frozen=True)
class ImageMetrics:
    brightness_mean: float
    contrast_std: float
    blur_laplacian_variance: float
    edge_energy: float
    saturation_mean: float


@dataclass(frozen=True)
class SceneSummary:
    scene_name: str
    image_count: int
    image_size: tuple[int, int]
    pose_source: str
    adjacent_view_correlation_median: float | None
    brightness_mean_median: float
    contrast_std_median: float
    blur_laplacian_variance_median: float
    edge_energy_median: float
    saturation_mean_median: float


def analyze_project() -> dict[str, SceneSummary]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_project_scenes(PROJECT_ROOT)
    summaries: dict[str, SceneSummary] = {}

    for scene_name, dataset in datasets.items():
        scene_images = [Image.open(view.image_path).convert("RGB") for view in dataset.views]
        try:
            metrics = [compute_metrics(image) for image in scene_images]
            correlation = adjacent_view_correlation(scene_images)
            write_contact_sheet(scene_name, dataset.views, scene_images)
            summaries[scene_name] = SceneSummary(
                scene_name=scene_name,
                image_count=len(dataset.views),
                image_size=dataset.views[0].intrinsics.image_size,
                pose_source=dataset.pose_source,
                adjacent_view_correlation_median=correlation,
                brightness_mean_median=float(np.median([item.brightness_mean for item in metrics])),
                contrast_std_median=float(np.median([item.contrast_std for item in metrics])),
                blur_laplacian_variance_median=float(
                    np.median([item.blur_laplacian_variance for item in metrics])
                ),
                edge_energy_median=float(np.median([item.edge_energy for item in metrics])),
                saturation_mean_median=float(np.median([item.saturation_mean for item in metrics])),
            )
        finally:
            for image in scene_images:
                image.close()

    output_path = OUTPUT_DIR / "scene_analysis.json"
    output_path.write_text(
        json.dumps({name: asdict(summary) for name, summary in summaries.items()}, indent=2),
        encoding="utf-8",
    )
    return summaries


def compute_metrics(image: Image.Image) -> ImageMetrics:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0

    north = np.roll(grayscale, -1, axis=0)
    south = np.roll(grayscale, 1, axis=0)
    east = np.roll(grayscale, -1, axis=1)
    west = np.roll(grayscale, 1, axis=1)
    laplacian = north + south + east + west - (4.0 * grayscale)
    gradient_x = east - west
    gradient_y = north - south
    edge_energy = np.mean(np.sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y)))

    return ImageMetrics(
        brightness_mean=float(np.mean(grayscale)),
        contrast_std=float(np.std(grayscale)),
        blur_laplacian_variance=float(np.var(laplacian)),
        edge_energy=float(edge_energy),
        saturation_mean=float(np.mean(hsv[..., 1])),
    )


def adjacent_view_correlation(images: list[Image.Image]) -> float | None:
    if len(images) < 2:
        return None

    correlations: list[float] = []
    previous = downsample_grayscale(images[0])
    for image in images[1:]:
        current = downsample_grayscale(image)
        correlations.append(normalized_correlation(previous, current))
        previous = current
    return float(np.median(correlations))


def downsample_grayscale(image: Image.Image) -> np.ndarray:
    reduced = image.convert("L").resize((160, 160))
    return np.asarray(reduced, dtype=np.float32)


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def write_contact_sheet(scene_name: str, views, images: list[Image.Image]) -> None:
    rows = (len(images) + CONTACT_SHEET_COLUMNS - 1) // CONTACT_SHEET_COLUMNS
    sheet_width = CONTACT_SHEET_COLUMNS * THUMBNAIL_SIZE[0]
    sheet_height = rows * (THUMBNAIL_SIZE[1] + 28)
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(16, 18, 24))
    draw = ImageDraw.Draw(sheet)

    for offset, (view, image) in enumerate(zip(views, images, strict=True)):
        column = offset % CONTACT_SHEET_COLUMNS
        row = offset // CONTACT_SHEET_COLUMNS
        x = column * THUMBNAIL_SIZE[0]
        y = row * (THUMBNAIL_SIZE[1] + 28)

        thumbnail = image.copy()
        thumbnail.thumbnail(THUMBNAIL_SIZE)
        paste_x = x + (THUMBNAIL_SIZE[0] - thumbnail.width) // 2
        paste_y = y + (THUMBNAIL_SIZE[1] - thumbnail.height) // 2
        sheet.paste(thumbnail, (paste_x, paste_y))
        draw.text((x + 8, y + THUMBNAIL_SIZE[1] + 6), f"{scene_name} #{view.index}", fill=(230, 232, 236))

    sheet.save(OUTPUT_DIR / f"{scene_name.lower()}_contact_sheet.jpg", quality=90)


if __name__ == "__main__":
    summaries = analyze_project()
    print(json.dumps({name: asdict(summary) for name, summary in summaries.items()}, indent=2))