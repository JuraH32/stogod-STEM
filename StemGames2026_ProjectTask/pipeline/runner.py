from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from StemGames2026_ProjectTask.pointcloud.loaders import load_project_scenes
from StemGames2026_ProjectTask.pointcloud.schemas import SceneDataset, SceneView

from StemGames2026_ProjectTask.pipeline import coords
from StemGames2026_ProjectTask.pipeline.io import depth_writer, pixel_map as pixel_map_io, ply_writer
from StemGames2026_ProjectTask.pipeline.mesh.base import SceneMesher
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose, PerViewResult, SceneResult
from StemGames2026_ProjectTask.pipeline.depth.base import DepthEstimator
from StemGames2026_ProjectTask.pipeline.fusion.base import Fuser
from StemGames2026_ProjectTask.pipeline.pose.base import PoseProvider
from StemGames2026_ProjectTask.pipeline.postprocess.base import PostProcessor
from StemGames2026_ProjectTask.pipeline.reconstruction.base import SceneReconstructor


@dataclass
class PipelineConfig:
    output_root: Path
    pose_provider: PoseProvider
    depth_estimator: DepthEstimator
    fuser: Fuser
    post_processor: PostProcessor
    mesher: SceneMesher | None = None
    reconstructor: SceneReconstructor | None = None  # None for posed datasets


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def run_scene(self, dataset: SceneDataset) -> SceneResult:
        out_dir   = self._cfg.output_root / dataset.scene_name
        depth_dir = out_dir / "depth_maps"
        cloud_dir = out_dir / "per_image_clouds"
        pmap_dir  = out_dir / "pixel_maps"
        for d in (depth_dir, cloud_dir, pmap_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Stage 1: Pose provision
        estimated_poses: list[EstimatedPose] = self._cfg.pose_provider.provide(dataset)
        pose_by_index = {ep.view_index: ep for ep in estimated_poses}

        # Stage 2 (optional): MapAnything global reconstruction
        # For unposed scenes this provides globally-consistent scale hints and
        # a baseline scene cloud without running a second forward pass (the
        # reconstructor shares the pose provider's internal cache).
        map_anything_cloud: np.ndarray | None = None
        if self._cfg.reconstructor is not None:
            map_anything_cloud, _ = self._cfg.reconstructor.reconstruct(
                dataset, hints=estimated_poses
            )

        # Compute per-view metric scale hints via SIFT + landmark triangulation for
        # every dataset (posed and unposed alike).  For unposed scenes the COLMAP-
        # estimated poses are accurate enough for triangulation and give far more
        # reference points than the sparse COLMAP cloud projection (~10-50 pts/view).
        print(f"  [{dataset.scene_name}] Computing scale hints (landmark + SIFT) …")
        scale_hints = _build_scale_hints(dataset.views, pose_by_index)
        n_good = sum(1 for v in scale_hints.values() if v is not None)
        print(f"  [{dataset.scene_name}] Scale hints: {n_good}/{len(dataset.views)} views succeeded")

        # Stage 3: Per-view depth estimation + unprojection
        per_view_results: list[PerViewResult] = []
        for view in dataset.views:
            ep = pose_by_index[view.index]

            # Skip views that COLMAP could not register — their identity pose would
            # place all unprojected points at world-origin, polluting the merged cloud.
            if ep.confidence == 0.0:
                print(f"  [{view.scene_name} #{view.index}] Skipping — no pose registered")
                continue

            # Triangulation hints first; fall back to COLMAP cloud projection only
            # if SIFT triangulation failed for this view.
            ref_pts_cam = scale_hints.get(view.index)
            if ref_pts_cam is None and map_anything_cloud is not None:
                ref_pts_cam = _compute_scale_hint(ep, map_anything_cloud, view.intrinsics)

            depth_result = self._cfg.depth_estimator.estimate(
                view, ep.pose, reference_points_cam=ref_pts_cam
            )

            rgb = np.array(Image.open(view.image_path).convert("RGB"))
            c2w = coords.unity_pose_to_opencv_c2w(ep.pose)
            pts_world, colors, pix_coords = coords.unproject_depth_to_world(
                depth_result.depth_map,
                depth_result.validity_mask,
                rgb,
                view.intrinsics,
                c2w,
            )

            pvr = PerViewResult(
                view=view,
                pose=ep.pose,
                depth_result=depth_result,
                points_world=pts_world,
                colors_rgb=colors,
                pixel_coords=pix_coords,
            )

            stem = view.image_path.stem
            pvr.depth_npy_path = depth_dir / f"{stem}_depth.npy"
            pvr.depth_png_path = depth_dir / f"{stem}_depth.png"
            pvr.ply_path       = cloud_dir / f"{stem}.ply"
            pvr.pixel_map_path = pmap_dir  / f"{stem}_pixel_map.npy"

            depth_writer.write_depth_npy(pvr.depth_npy_path, depth_result.depth_map)
            depth_writer.write_depth_png(
                pvr.depth_png_path, depth_result.depth_map, depth_result.validity_mask
            )
            ply_writer.write_ply(pvr.ply_path, pts_world, colors)
            pixel_map_io.write_pixel_map(pvr.pixel_map_path, pix_coords, np.zeros(len(pix_coords), dtype=np.int32))

            per_view_results.append(pvr)
            print(f"  [{view.scene_name} #{view.index}] {len(pts_world):,} points  "
                  f"scale={depth_result.scale_factor:.4f} ({depth_result.scale_source})")

        # Stage 4: Fusion
        scene_result = self._cfg.fuser.fuse(per_view_results)

        # Stage 5: Post-processing
        pts, cols, sv, sp = self._cfg.post_processor.process(
            scene_result.scene_points,
            scene_result.scene_colors,
            scene_result.source_view_index,
            scene_result.source_pixel,
        )
        scene_result = SceneResult(
            scene_name=dataset.scene_name,
            per_view=per_view_results,
            scene_points=pts,
            scene_colors=cols,
            source_view_index=sv,
            source_pixel=sp,
        )

        # Write scene cloud
        scene_result.scene_ply_path = out_dir / "scene_cloud.ply"
        ply_writer.write_ply(scene_result.scene_ply_path, pts, cols)

        print(f"  [{dataset.scene_name}] scene cloud: {len(pts):,} points → {scene_result.scene_ply_path}")

        # Stage 6 (optional): best-effort surface mesh generation
        if self._cfg.mesher is not None:
            mesh_path = out_dir / "scene_mesh.ply"
            try:
                mesh_result = self._cfg.mesher.mesh(dataset.scene_name, pts, cols, mesh_path)
            except Exception as exc:
                scene_result.mesh_warning = str(exc)
                print(f"  [{dataset.scene_name}] scene mesh skipped: {exc}")
            else:
                scene_result.scene_mesh_path = mesh_result.mesh_path
                scene_result.scene_mesh_vertex_count = mesh_result.vertex_count
                scene_result.scene_mesh_face_count = mesh_result.face_count
                scene_result.scene_mesh_backend = mesh_result.backend
                print(
                    f"  [{dataset.scene_name}] scene mesh: "
                    f"{mesh_result.vertex_count:,} vertices, {mesh_result.face_count:,} faces "
                    f"→ {mesh_result.mesh_path}"
                )

        return scene_result

    def run_all(self, project_root: Path) -> dict[str, SceneResult]:
        datasets = load_project_scenes(project_root)
        results: dict[str, SceneResult] = {}
        for name, dataset in datasets.items():
            print(f"\n=== {name} ===")
            results[name] = self.run_scene(dataset)
        return results


# ---------------------------------------------------------------------------
# Scale hint helpers — shared triangulation core
# ---------------------------------------------------------------------------

def _triangulate_pts_with_certainty(
    pts1: np.ndarray,       # (N, 2) float32 — 2D points in view i
    pts2: np.ndarray,       # (N, 2) float32 — matched 2D points in reference view
    P1: np.ndarray,         # (3, 4) float64 — projection matrix for view i
    P2: np.ndarray,         # (3, 4) float64 — projection matrix for reference view
    w2c_i: np.ndarray,      # (4, 4) float64 — world-to-camera for view i
) -> tuple[np.ndarray | None, float]:
    """
    Triangulate correspondences, return (pts_cam_i, certainty).
    certainty = n_valid / (1 + mean_reprojection_error_px).
    Returns (None, 0.0) if fewer than 2 valid points result.
    """
    import cv2

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
    w = pts4d[3]
    valid = np.abs(w) > 1e-8
    if not np.any(valid):
        return None, 0.0

    pts_world = (pts4d[:3, valid] / w[valid]).T.astype(np.float64)  # (M, 3)

    # Reprojection error in view i — measures triangulation quality
    pts_h = np.c_[pts_world, np.ones(len(pts_world))]        # (M, 4)
    ph    = P1 @ pts_h.T                                      # (3, M)
    denom = np.where(np.abs(ph[2]) > 1e-8, ph[2], 1.0)
    proj  = (ph[:2] / denom).T                               # (M, 2)
    reproj_err = float(np.mean(np.linalg.norm(proj - pts1[valid].astype(np.float64), axis=1)))

    # Transform to camera i frame, keep points in front
    pts_cam = (w2c_i @ pts_h.T).T[:, :3]
    front   = pts_cam[:, 2] > 0.05
    pts_cam = pts_cam[front].astype(np.float32)

    n         = len(pts_cam)
    certainty = n / (1.0 + reproj_err) if n >= 2 else 0.0
    return (pts_cam if n >= 2 else None), certainty


# Pattern sizes tried for findChessboardCorners — (cols, rows) of INNER corners.
# OpenCV requires both dims > 2, so minimum valid pattern is (3, 3).
_CHECKER_PATTERNS = [
    (3, 3), (4, 3), (3, 4), (4, 4), (5, 4), (4, 5), (5, 5), (6, 4), (4, 6),
]


def _detect_2x2_markers(gray: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Detect 2×2 black-and-white fiducial markers via contour analysis.

    Finds solid dark rectangular blobs, pairs adjacent same-size blobs that
    together form a 2×2 checkerboard, and returns each pair's centre point.
    Works on markers too small for findChessboardCorners (which requires > 2
    inner corners in each dimension).

    Returns (N, 2) float32 array of centre pixel coordinates, possibly empty.
    """
    import cv2

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_rects: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (100 < area < 20_000):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) == 0:
            continue
        if area / (w * h) > 0.5 and max(w, h) / min(w, h) < 4.0:
            black_rects.append((x, y, x + w, y + h))

    centers: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(black_rects)):
        x1a, y1a, x2a, y2a = black_rects[i]
        wa, ha = x2a - x1a, y2a - y1a
        for j in range(i + 1, len(black_rects)):
            x1b, y1b, x2b, y2b = black_rects[j]
            wb, hb = x2b - x1b, y2b - y1b
            # Reject pairs whose tiles differ in size
            if abs(wa - wb) > 0.6 * max(wa, wb) or abs(ha - hb) > 0.6 * max(ha, hb):
                continue
            bx1 = min(x1a, x1b); bx2 = max(x2a, x2b)
            by1 = min(y1a, y1b); by2 = max(y2a, y2b)
            bw = bx2 - bx1; bh = by2 - by1
            # Bounding box should be roughly 2× a single tile in each dimension
            if not (0.7 * 2 * wa < bw < 1.7 * 2 * wa):
                continue
            if not (0.7 * 2 * ha < bh < 1.7 * 2 * ha):
                continue
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            key = (round(cx / 10) * 10, round(cy / 10) * 10)
            if key in seen:
                continue
            seen.add(key)
            centers.append([cx, cy])

    return np.array(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32)


def _detect_checkerboard_corners(gray: np.ndarray) -> np.ndarray:
    """
    Detect checkerboard landmarks in a grayscale image.

    Runs two complementary detectors and returns all found points:
    1. findChessboardCorners for standard patterns ≥ 4×4 squares (subpixel-refined)
    2. Contour-based 2×2 marker detector for small fiducials below the
       findChessboardCorners minimum (e.g. the 2×2 markers on the Box crate)

    Returns (N, 2) float32 array of pixel coordinates, possibly empty.
    """
    import cv2

    found: list[np.ndarray] = []

    # --- Standard checkerboard (≥ 3 inner corners per side) ---
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for pat in _CHECKER_PATTERNS:
        ret, corners = cv2.findChessboardCorners(gray, pat, flags=flags)
        if ret and corners is not None:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            found.append(corners.reshape(-1, 2))

    # --- Small 2×2 fiducial markers (contour-based) ---
    markers_2x2 = _detect_2x2_markers(gray)
    if len(markers_2x2) > 0:
        found.append(markers_2x2)

    return np.concatenate(found, axis=0) if found else np.empty((0, 2), dtype=np.float32)


def _build_landmark_hints(
    views: list[SceneView],
    pose_by_index: dict[int, EstimatedPose],
) -> dict[int, tuple[np.ndarray | None, float]]:
    """
    Detect checkerboard corners in each view and its adjacent view,
    compute SIFT descriptors at corner locations, match, triangulate.
    Returns {view_index: (pts_cam | None, certainty)}.
    """
    import cv2

    sift   = cv2.SIFT_create()
    result: dict[int, tuple[np.ndarray | None, float]] = {}

    for i, view in enumerate(views):
        ep_i = pose_by_index[view.index]
        if ep_i.confidence < 0.1:
            result[view.index] = (None, 0.0)
            continue

        # Prefer the previous view; fall back to next if previous is unregistered
        prev = views[i - 1] if i > 0 else None
        nxt  = views[i + 1] if i < len(views) - 1 else None
        ref_view = None
        for candidate in (prev, nxt):
            if candidate is not None and pose_by_index[candidate.index].confidence >= 0.1:
                ref_view = candidate
                break
        if ref_view is None:
            result[view.index] = (None, 0.0)
            continue

        ep_ref = pose_by_index[ref_view.index]
        intr   = view.intrinsics

        gray_i   = np.array(Image.open(view.image_path).convert("L"))
        gray_ref = np.array(Image.open(ref_view.image_path).convert("L"))

        corners_i   = _detect_checkerboard_corners(gray_i)
        corners_ref = _detect_checkerboard_corners(gray_ref)

        if len(corners_i) < 2 or len(corners_ref) < 2:
            result[view.index] = (None, 0.0)
            continue

        # Compute SIFT descriptors at the detected corner locations
        kp_i   = [cv2.KeyPoint(float(x), float(y), 20.0) for x, y in corners_i]
        kp_ref = [cv2.KeyPoint(float(x), float(y), 20.0) for x, y in corners_ref]
        _, desc_i   = sift.compute(gray_i,   kp_i)
        _, desc_ref = sift.compute(gray_ref, kp_ref)

        if desc_i is None or desc_ref is None:
            result[view.index] = (None, 0.0)
            continue

        # Match — use Lowe ratio if enough candidates, otherwise take nearest only
        k_nn = min(2, len(kp_ref))
        raw  = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_i, desc_ref, k=k_nn)
        if k_nn >= 2:
            good_idx = [(m.queryIdx, m.trainIdx) for m, n in raw if m.distance < 0.8 * n.distance]
        else:
            good_idx = [(m[0].queryIdx, m[0].trainIdx) for m in raw]

        if len(good_idx) < 2:
            result[view.index] = (None, 0.0)
            continue

        qi, ti       = zip(*good_idx)
        pts_i_2d     = corners_i[list(qi)].astype(np.float32)
        pts_ref_2d   = corners_ref[list(ti)].astype(np.float32)

        K       = np.array(intr.matrix, dtype=np.float64)
        c2w_i   = coords.unity_pose_to_opencv_c2w(ep_i.pose).astype(np.float64)
        c2w_ref = coords.unity_pose_to_opencv_c2w(ep_ref.pose).astype(np.float64)
        w2c_i   = np.linalg.inv(c2w_i)
        P_i     = K @ w2c_i[:3, :]
        P_ref   = K @ np.linalg.inv(c2w_ref)[:3, :]

        pts_cam, certainty = _triangulate_pts_with_certainty(pts_i_2d, pts_ref_2d, P_i, P_ref, w2c_i)
        result[view.index] = (pts_cam, certainty)

    return result


def _build_sift_hints(
    views: list[SceneView],
    pose_by_index: dict[int, EstimatedPose],
) -> dict[int, tuple[np.ndarray | None, float]]:
    """
    SIFT feature detection + matching between adjacent views, then triangulation.
    Returns {view_index: (pts_cam | None, certainty)}.
    """
    import cv2

    sift   = cv2.SIFT_create()
    result: dict[int, tuple[np.ndarray | None, float]] = {}

    for i, view in enumerate(views):
        ep_i = pose_by_index[view.index]
        if ep_i.confidence < 0.1:
            result[view.index] = (None, 0.0)
            continue

        # Prefer the previous view; fall back to next if previous is unregistered
        prev = views[i - 1] if i > 0 else None
        nxt  = views[i + 1] if i < len(views) - 1 else None
        ref_view = None
        for candidate in (prev, nxt):
            if candidate is not None and pose_by_index[candidate.index].confidence >= 0.1:
                ref_view = candidate
                break
        if ref_view is None:
            result[view.index] = (None, 0.0)
            continue

        ep_ref = pose_by_index[ref_view.index]
        intr   = view.intrinsics

        gray_i   = np.array(Image.open(view.image_path).convert("L"))
        gray_ref = np.array(Image.open(ref_view.image_path).convert("L"))

        kp_i,   desc_i   = sift.detectAndCompute(gray_i,   None)
        kp_ref, desc_ref = sift.detectAndCompute(gray_ref, None)

        if desc_i is None or desc_ref is None or len(kp_i) < 8 or len(kp_ref) < 8:
            result[view.index] = (None, 0.0)
            continue

        raw  = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_i, desc_ref, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            result[view.index] = (None, 0.0)
            continue

        K       = np.array(intr.matrix, dtype=np.float64)
        c2w_i   = coords.unity_pose_to_opencv_c2w(ep_i.pose).astype(np.float64)
        c2w_ref = coords.unity_pose_to_opencv_c2w(ep_ref.pose).astype(np.float64)
        w2c_i   = np.linalg.inv(c2w_i)
        P_i     = K @ w2c_i[:3, :]
        P_ref   = K @ np.linalg.inv(c2w_ref)[:3, :]

        pts_i   = np.array([kp_i[m.queryIdx].pt   for m in good], dtype=np.float32)  # (N, 2)
        pts_ref = np.array([kp_ref[m.trainIdx].pt for m in good], dtype=np.float32)  # (N, 2)

        pts_cam, certainty = _triangulate_pts_with_certainty(pts_i, pts_ref, P_i, P_ref, w2c_i)
        result[view.index] = (pts_cam, certainty)

    return result


def _build_scale_hints(
    views: list[SceneView],
    pose_by_index: dict[int, EstimatedPose],
) -> dict[int, np.ndarray | None]:
    """
    Run both landmark and SIFT scale hint computation; select the method with
    higher certainty per view.  Returns {view_index: pts_cam | None}.
    """
    sift_hints = _build_sift_hints(views, pose_by_index)
    lm_hints   = _build_landmark_hints(views, pose_by_index)

    result: dict[int, np.ndarray | None] = {}
    for view in views:
        sift_pts, sift_cert = sift_hints.get(view.index) or (None, 0.0)
        lm_pts,   lm_cert   = lm_hints.get(view.index)   or (None, 0.0)

        if lm_pts is not None and lm_cert >= sift_cert:
            result[view.index] = lm_pts
            winner = f"landmark (cert={lm_cert:.1f})"
        elif sift_pts is not None:
            result[view.index] = sift_pts
            winner = f"sift (cert={sift_cert:.1f})"
        else:
            result[view.index] = None
            winner = "none"

        print(f"    view {view.index}: landmark={lm_cert:.1f}  sift={sift_cert:.1f}  → {winner}")

    return result


def _compute_scale_hint(
    ep: EstimatedPose,
    map_anything_cloud: np.ndarray | None,
    intrinsics,
) -> np.ndarray | None:
    """
    Reproject the global reconstruction cloud into this camera's frame and
    return the visible 3D points IN CAMERA SPACE as metric reference for
    MoGe-2 scale alignment.  Returns None if no global cloud is available.
    """
    if map_anything_cloud is None:
        return None

    from StemGames2026_ProjectTask.pipeline import coords as _coords

    c2w = _coords.unity_pose_to_opencv_c2w(ep.pose)
    pts_xyz = map_anything_cloud[:, :3].astype(np.float32)
    _, _, valid = _coords.project_points_to_camera(pts_xyz, c2w, intrinsics)

    if not np.any(valid):
        return None

    # Convert visible world points to camera-space so _align_scale can use
    # their Z component as depth reference directly.
    w2c = np.linalg.inv(c2w)
    pts_vis = pts_xyz[valid]
    pts_h   = np.c_[pts_vis, np.ones(len(pts_vis), dtype=np.float32)]
    pts_cam = (w2c @ pts_h.T).T[:, :3].astype(np.float32)
    return pts_cam
