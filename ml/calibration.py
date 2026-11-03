"""Camera intrinsics calibration from chessboard views (upgrade 3.6), used by
scripts/calibrate_camera.py and tests. Writes the JSON that ml/camera_model.py loads."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from ml.camera_model import CameraIntrinsics


def find_corners(gray: np.ndarray, pattern: Tuple[int, int]):
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))


def calibrate(images: Sequence[np.ndarray], pattern: Tuple[int, int], square_m: float) -> Tuple[CameraIntrinsics, Dict]:
    """images: grey uint8 views of a chessboard with `pattern` inner corners (cols, rows)."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    obj, img = [], []
    for g in images:
        c = find_corners(g, pattern)
        if c is not None:
            obj.append(objp)
            img.append(c)
    if len(obj) < 3:
        raise ValueError(f"chessboard found in {len(obj)} views; need at least 3")
    h, w = images[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None)
    intr = CameraIntrinsics(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), w, h,
                            [float(x) for x in dist.ravel()[:5]], "calibration")
    return intr, {"rms_reprojection_px": float(rms), "views_used": len(obj), "views_total": len(images)}


def measure_length(p1: Tuple[float, float], p2: Tuple[float, float], intr: CameraIntrinsics, distance_m: float) -> float:
    """Metres between two pixels on a fronto-parallel surface at `distance_m` (tape-measure check)."""
    a, b = intr.undistort_points(np.array([p1, p2], np.float64))
    return float(np.hypot(*(a - b)) * intr.gsd_m_per_px(distance_m))
