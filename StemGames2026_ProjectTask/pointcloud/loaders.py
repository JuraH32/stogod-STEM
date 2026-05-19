from __future__ import annotations

import math
import re
import struct
from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import CameraIntrinsics, CameraPose, Matrix3, SceneDataset, SceneView, Vector3

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SCENE_REPORTED_HORIZONTAL_FOVS = {
    "Statue": 90.0,
    "Fountain": 84.0,
}
ENTRY_PATTERN = re.compile(
    r"(?P<index>\d+)\)\s*"
    r"CamPosition:\s*(?P<position>[^\n]+)\s*"
    r"CamForward:\s*(?P<forward>[^\n]+)\s*"
    r"CamRight:?\s*(?P<right>[^\n]+)\s*"
    r"CamUp:?\s*(?P<up>[^\n]+)",
    re.MULTILINE,
)
VECTOR_PATTERN = re.compile(
    r"X=(?P<x>[-+]?\d+(?:\.\d+)?)\s+Y=(?P<y>[-+]?\d+(?:\.\d+)?)\s+Z=(?P<z>[-+]?\d+(?:\.\d+)?)"
)
FLOAT_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


class SceneLoader(ABC):
    @abstractmethod
    def can_load(self, scene_dir: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self, scene_dir: Path) -> SceneDataset:
        raise NotImplementedError


class KnownPoseSceneLoader(SceneLoader):
    def can_load(self, scene_dir: Path) -> bool:
        return any(scene_dir.glob("*Input.txt"))

    def load(self, scene_dir: Path) -> SceneDataset:
        metadata_path = next(iter(sorted(scene_dir.glob("*Input.txt"))))
        metadata_text = metadata_path.read_text(encoding="utf-8")
        image_paths = _sorted_image_paths(scene_dir)
        if not image_paths:
            raise ValueError(f"No images found in {scene_dir}")

        fov_degrees = _parse_fov_degrees(metadata_text)
        intrinsics = _intrinsics_from_horizontal_fov(image_paths[0], fov_degrees)
        pose_entries = _parse_pose_entries(metadata_text)

        if len(pose_entries) != len(image_paths):
            raise ValueError(
                f"Pose/image count mismatch in {scene_dir}: "
                f"{len(pose_entries)} metadata entries for {len(image_paths)} images."
            )

        views = []
        for image_path, entry in zip(image_paths, pose_entries, strict=True):
            pose = CameraPose(
                position=entry["position"],
                forward=entry["forward"],
                right=entry["right"],
                up=entry["up"],
            )
            views.append(
                SceneView(
                    scene_name=scene_dir.name,
                    index=entry["index"],
                    image_path=image_path,
                    intrinsics=intrinsics,
                    pose=pose,
                    pose_status="known",
                    metadata={
                        "metadata_file": metadata_path.name,
                        "fov_degrees": fov_degrees,
                        "fov_axis": "horizontal",
                    },
                )
            )

        return SceneDataset(
            scene_name=scene_dir.name,
            root_dir=scene_dir,
            views=tuple(views),
            pose_source="provided",
            metadata_files=(metadata_path,),
            notes=(
                "Camera intrinsics were derived from the metadata FOV using a horizontal-FOV assumption.",
                "Pose vectors preserve the raw CamRight/CamUp/CamForward basis from the dataset.",
            ),
        )


class IntrinsicsOnlySceneLoader(SceneLoader):
    def can_load(self, scene_dir: Path) -> bool:
        return (scene_dir / "K.txt").exists()

    def load(self, scene_dir: Path) -> SceneDataset:
        intrinsic_path = scene_dir / "K.txt"
        if not intrinsic_path.exists():
            raise ValueError(f"Missing intrinsic matrix file in {scene_dir}")

        image_paths = _sorted_image_paths(scene_dir)
        if not image_paths:
            raise ValueError(f"No images found in {scene_dir}")

        intrinsics = _parse_intrinsics(intrinsic_path, image_paths[0])
        reported_fov = SCENE_REPORTED_HORIZONTAL_FOVS.get(scene_dir.name)
        fov_matches_intrinsics = _reported_fov_matches_intrinsics(reported_fov, intrinsics.fov_degrees)
        views = tuple(
            SceneView(
                scene_name=scene_dir.name,
                index=index,
                image_path=image_path,
                intrinsics=intrinsics,
                pose=None,
                pose_status="needs_estimation",
                metadata={
                    "metadata_file": intrinsic_path.name,
                    "pose_estimation_required": 1,
                    "derived_horizontal_fov_degrees": intrinsics.fov_degrees or -1.0,
                    **(
                        {
                            "reported_horizontal_fov_degrees": reported_fov,
                            "reported_fov_matches_intrinsics": fov_matches_intrinsics,
                        }
                        if reported_fov is not None
                        else {}
                    ),
                },
            )
            for index, image_path in enumerate(image_paths, start=1)
        )

        return SceneDataset(
            scene_name=scene_dir.name,
            root_dir=scene_dir,
            views=views,
            pose_source="missing",
            metadata_files=(intrinsic_path,),
            notes=_intrinsics_only_notes(scene_dir.name, intrinsics.fov_degrees, reported_fov, fov_matches_intrinsics),
        )


LOADERS: tuple[SceneLoader, ...] = (
    KnownPoseSceneLoader(),
    IntrinsicsOnlySceneLoader(),
)


def load_scene(scene_dir: str | Path) -> SceneDataset:
    scene_path = Path(scene_dir)
    if not scene_path.exists() or not scene_path.is_dir():
        raise ValueError(f"Scene directory does not exist: {scene_path}")

    for loader in LOADERS:
        if loader.can_load(scene_path):
            return loader.load(scene_path)

    raise ValueError(
        f"Could not determine loader for {scene_path}. Expected either *Input.txt or K.txt metadata."
    )


def load_project_scenes(project_root: str | Path) -> dict[str, SceneDataset]:
    root_path = Path(project_root)
    test_images_dir = root_path / "TestImages" if (root_path / "TestImages").is_dir() else root_path
    scene_dirs = sorted(path for path in test_images_dir.iterdir() if path.is_dir())
    return {scene_dir.name: load_scene(scene_dir) for scene_dir in scene_dirs}


def _sorted_image_paths(scene_dir: Path) -> list[Path]:
    image_paths = [
        path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=_natural_image_key)


def _natural_image_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.stem)
    number = int(match.group(1)) if match else -1
    return path.stem.rstrip("0123456789"), number


def _parse_fov_degrees(metadata_text: str) -> float:
    match = re.search(
        r"field of view[^\d]*(?P<fov>\d+(?:\.\d+)?)\s*degrees",
        metadata_text,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not determine FOV from metadata file.")
    return float(match.group("fov"))


def _intrinsics_from_horizontal_fov(image_path: Path, fov_degrees: float) -> CameraIntrinsics:
    width, height = _image_size(image_path)
    focal_length = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    principal_point = (width / 2.0, height / 2.0)
    matrix: Matrix3 = (
        (focal_length, 0.0, principal_point[0]),
        (0.0, focal_length, principal_point[1]),
        (0.0, 0.0, 1.0),
    )
    return CameraIntrinsics(
        matrix=matrix,
        image_size=(width, height),
        source="derived_from_fov",
        fov_degrees=fov_degrees,
        fov_axis="horizontal",
    )


def _parse_pose_entries(metadata_text: str) -> list[dict[str, int | Vector3]]:
    entries: list[dict[str, int | Vector3]] = []
    for match in ENTRY_PATTERN.finditer(metadata_text):
        entries.append(
            {
                "index": int(match.group("index")),
                "position": _parse_vector(match.group("position")),
                "forward": _parse_vector(match.group("forward")),
                "right": _parse_vector(match.group("right")),
                "up": _parse_vector(match.group("up")),
            }
        )

    if not entries:
        raise ValueError("No camera pose entries found in metadata file.")
    return entries


def _parse_vector(vector_text: str) -> Vector3:
    match = VECTOR_PATTERN.search(vector_text)
    if not match:
        raise ValueError(f"Could not parse vector from line: {vector_text!r}")
    return (
        float(match.group("x")),
        float(match.group("y")),
        float(match.group("z")),
    )


def _parse_intrinsics(intrinsic_path: Path, image_path: Path) -> CameraIntrinsics:
    values = [float(value) for value in FLOAT_PATTERN.findall(intrinsic_path.read_text(encoding="utf-8"))]
    if len(values) != 9:
        raise ValueError(f"Expected 9 numeric values in {intrinsic_path}, found {len(values)}.")

    width, height = _image_size(image_path)
    matrix: Matrix3 = (
        (values[0], values[1], values[2]),
        (values[3], values[4], values[5]),
        (values[6], values[7], values[8]),
    )
    horizontal_fov = _horizontal_fov_from_intrinsics(width, matrix[0][0])
    return CameraIntrinsics(
        matrix=matrix,
        image_size=(width, height),
        source="provided_k",
        fov_degrees=horizontal_fov,
        fov_axis="horizontal",
    )


def _horizontal_fov_from_intrinsics(image_width: int, focal_length_x: float) -> float:
    if focal_length_x <= 0.0:
        raise ValueError("Expected positive fx when deriving FOV from intrinsics.")
    return math.degrees(2.0 * math.atan(image_width / (2.0 * focal_length_x)))


def _reported_fov_matches_intrinsics(
    reported_fov: float | None,
    derived_horizontal_fov: float | None,
    tolerance_degrees: float = 3.0,
) -> int:
    if reported_fov is None or derived_horizontal_fov is None:
        return 0
    return int(abs(reported_fov - derived_horizontal_fov) <= tolerance_degrees)


def _intrinsics_only_notes(
    scene_name: str,
    derived_horizontal_fov: float | None,
    reported_fov: float | None,
    fov_matches_intrinsics: int,
) -> tuple[str, ...]:
    notes = [
        "Camera intrinsics were loaded directly from K.txt.",
        "Extrinsics are intentionally left empty so downstream pose estimation can remain model-agnostic.",
    ]
    if derived_horizontal_fov is not None:
        notes.append(
            f"K.txt implies an approximate horizontal FOV of {derived_horizontal_fov:.2f} degrees."
        )
    if reported_fov is not None and fov_matches_intrinsics:
        notes.append(
            f"The reported horizontal FOV for {scene_name} ({reported_fov:.2f} degrees) is consistent with K.txt."
        )
    elif reported_fov is not None:
        notes.append(
            f"A reported horizontal FOV of {reported_fov:.2f} degrees was preserved as a capture hint, but K.txt remains authoritative."
        )
    return tuple(notes)


def _image_size(image_path: Path) -> tuple[int, int]:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return _png_size(image_path)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_size(image_path)
    raise ValueError(f"Unsupported image format for {image_path}")


def _png_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as file_handle:
        signature = file_handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG signature in {image_path}")

        _chunk_length = file_handle.read(4)
        chunk_type = file_handle.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"Missing IHDR chunk in {image_path}")

        width, height = struct.unpack(">II", file_handle.read(8))
        return width, height


def _jpeg_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as file_handle:
        if file_handle.read(2) != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG signature in {image_path}")

        while True:
            marker_start = file_handle.read(1)
            while marker_start == b"\xff":
                marker_start = file_handle.read(1)
            if not marker_start:
                break

            marker = marker_start[0]
            if marker in {0xD8, 0xD9}:
                continue

            segment_length_bytes = file_handle.read(2)
            if len(segment_length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"Invalid JPEG segment length in {image_path}")

            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                _precision = file_handle.read(1)
                height, width = struct.unpack(">HH", file_handle.read(4))
                return width, height

            file_handle.seek(segment_length - 2, 1)

    raise ValueError(f"Could not determine JPEG dimensions for {image_path}")
