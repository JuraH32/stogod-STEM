from __future__ import annotations

import math

import numpy as np
from PIL import Image

from StemGames2026_ProjectTask.pointcloud.schemas import CameraPose, SceneView
from StemGames2026_ProjectTask.pipeline.schemas import DepthResult
from StemGames2026_ProjectTask.pipeline.depth.base import DepthEstimator


def _best_device() -> str:
    """Return the best available torch device string for this machine."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class MoGe2DepthEstimator(DepthEstimator):
    """
    Per-image depth estimator backed by MoGe-2 (microsoft/moge, v2 API).

    MoGe v2 infer() accepts horizontal FOV in degrees (fov_x) and returns:
      - points  (H, W, 3): camera-space 3D points; inf where invalid
      - depth   (H, W)   : Z-depth in metres; inf where invalid
      - mask    (H, W)   : bool, True where valid
      - intrinsics (3,3) : recovered/given K matrix (normalised)

    When reference_points_cam is provided the depth is already in camera space
    and scale-aligned via median ratio to those metric anchor points.
    """

    def __init__(
        self,
        model_id: str = "Ruicheng/moge-vitl",
        device: str | None = None,
        resolution_level: int = 9,
    ) -> None:
        self._model_id = model_id
        self._device_str = device or _best_device()
        self._resolution_level = resolution_level
        self._model = None  # lazy-loaded on first call
        self._torch = None

    def _load_model(self) -> None:
        try:
            import torch
            from moge.model import import_model_class_by_version
        except ImportError as e:
            raise ImportError(
                "MoGe-2 requires 'moge' and 'torch'. "
                "Install with: pip install git+https://github.com/microsoft/MoGe.git torch"
            ) from e

        # Auto-select model version from checkpoint to support both v1 and v2 weights.
        # The Ruicheng/moge-vitl checkpoint on HuggingFace is v1 (no 'neck' key).
        import torch as _torch
        _ckpt = _torch.load(
            self._model_id if __import__("pathlib").Path(self._model_id).exists()
            else __import__("huggingface_hub").hf_hub_download(self._model_id, "model.pt"),
            map_location="cpu", weights_only=True,
        )
        version = "v2" if "neck" in _ckpt.get("model_config", {}) else "v1"
        MoGeModel = import_model_class_by_version(version)

        self._torch = torch
        device = torch.device(self._device_str)
        self._model = MoGeModel.from_pretrained(self._model_id).to(device).eval()

    def estimate(
        self,
        view: SceneView,
        pose: CameraPose,
        reference_points_cam: np.ndarray | None = None,
    ) -> DepthResult:
        if self._model is None:
            self._load_model()

        torch = self._torch

        rgb = np.array(Image.open(view.image_path).convert("RGB"))
        image_tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

        # MoGe v2 uses horizontal FOV in degrees for scale recovery.
        # Derive it from the known intrinsics: fov_x = 2*atan(W/(2*fx)).
        W = view.intrinsics.image_size[0]
        fov_x_deg = math.degrees(2.0 * math.atan(W / (2.0 * view.intrinsics.fx)))

        # MPS does not support float16 autocast reliably; disable fp16 there.
        use_fp16 = self._device_str != "mps"

        output = self._model.infer(
            image_tensor,
            fov_x=fov_x_deg,
            resolution_level=self._resolution_level,
            use_fp16=use_fp16,
        )

        # depth and points have inf where the model masked pixels as invalid.
        depth_raw  = output["depth"].cpu().numpy().astype(np.float32)    # (H, W)
        points_raw = output["points"].cpu().numpy().astype(np.float32)   # (H, W, 3)
        mask       = output["mask"].cpu().numpy().astype(bool)            # (H, W)

        # Replace inf sentinels with 0 so downstream arithmetic is safe.
        depth_clean  = np.where(mask, depth_raw,  0.0).astype(np.float32)
        points_clean = np.where(mask[..., None], points_raw, 0.0).astype(np.float32)

        scale_factor, scale_source = self._align_scale(
            depth_clean, mask, view.intrinsics, reference_points_cam
        )

        return DepthResult(
            view=view,
            depth_map=depth_clean * scale_factor,
            points_cam=points_clean * scale_factor,
            validity_mask=mask,
            scale_factor=scale_factor,
            scale_source=scale_source,
        )

    def _align_scale(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics,
        reference_points_cam: np.ndarray | None,
    ) -> tuple[float, str]:
        """
        Compute scale_factor so that depth * scale_factor ≈ metric depth.

        Projects each reference 3D point (in camera space, metric) back to a
        pixel, reads the model's depth there, and computes the ratio
        reference_z / model_z. Returns the median ratio over all valid hits.
        """
        if reference_points_cam is None or len(reference_points_cam) == 0:
            return 1.0, "none"

        H, W = depth.shape
        ref_z = reference_points_cam[:, 2]
        ref_x = reference_points_cam[:, 0]
        ref_y = reference_points_cam[:, 1]

        front = ref_z > 0.0
        u = np.where(front, ref_x / ref_z * intrinsics.fx + intrinsics.cx, -1.0)
        v = np.where(front, ref_y / ref_z * intrinsics.fy + intrinsics.cy, -1.0)

        col = np.round(u).astype(int)
        row = np.round(v).astype(int)
        in_bounds = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        hits = front & in_bounds

        if not np.any(hits):
            return 1.0, "none"

        row_h, col_h = row[hits], col[hits]
        model_z = depth[row_h, col_h]
        ref_z_h = ref_z[hits]

        good = mask[row_h, col_h] & (model_z > 0.0)
        if not np.any(good):
            return 1.0, "none"

        ratios = ref_z_h[good] / model_z[good]
        scale = float(np.median(ratios))
        source = "triangulation" if abs(scale - 1.0) > 1e-6 else "none"
        return scale, source
