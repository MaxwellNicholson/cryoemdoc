"""Atlas prerecognition and CSV generation.

This module is mechanically extracted from ``AtlasPrerecognition.ipynb`` and
wrapped for package use. The scoring thresholds, ordered labeling rules, and CSV
columns are intentionally kept close to the prototype notebook.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    class _MissingCV2:
        def __getattr__(self, name):
            raise ImportError("Atlas prerecognition requires opencv-python-headless.")
    cv2 = _MissingCV2()

try:
    import mrcfile
except ImportError:
    mrcfile = None

try:
    import tifffile
except ImportError:
    tifffile = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Cell 2: functions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mrc"}

ATLAS_SUMMARY_COLUMNS = [
    "image_name",
    "visible_square_count",
    "good_square_count",
    "non_uniform_square_count",
    "cracked_square_count",
    "bad_size_square_count",
    "atlas_quality_score",
    "error",
]


def make_dir(path):
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_uint8(arr, low_percentile=1, high_percentile=99.8):
    """
    Convert an arbitrary numeric image array to uint8 using percentile clipping.
    Useful for MRC/TIFF images that may not already be 0-255.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {3, 4}:
        # Sometimes scientific arrays are channel-first. Move channels last.
        arr = np.moveaxis(arr, 0, -1)

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Image contains no finite numeric values.")

    lo, hi = np.percentile(arr[finite], [low_percentile, high_percentile])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def read_image_as_bgr(path):
    """
    Read PNG/JPG/TIFF/MRC into a BGR uint8 image for OpenCV processing.
    For MRC stacks/volumes, this prototype uses the first slice after squeezing.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mrc":
        if mrcfile is None:
            raise ImportError("mrcfile is not installed. Install with: pip install mrcfile")
        with mrcfile.open(path, permissive=True) as mrc:
            arr = np.asarray(mrc.data)
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr[0]
        gray = to_uint8(arr)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if suffix in {".tif", ".tiff"}:
        if tifffile is not None:
            arr = tifffile.imread(str(path))
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
                img = to_uint8(arr)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = to_uint8(arr if arr.ndim == 2 else arr[0])
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if img.ndim == 2:
        img = to_uint8(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = to_uint8(img)

    return img


def normalize_gray_for_detection(bgr, low_percentile=1, high_percentile=99.5):
    """
    Convert BGR image to normalized grayscale uint8.
    Percentile clipping helps images with uneven exposure or extreme bright labels.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    lo, hi = np.percentile(gray[finite], [low_percentile, high_percentile])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((gray - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def make_bright_square_mask(gray_u8, blur_ksize=3, morph_ksize=3, threshold_offset=0):
    """
    Create a binary mask of visible bright square/opening material.

    Important: this does NOT create expected lattice positions.
    It only thresholds regions that are actually visible in the image.
    """
    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        work = cv2.GaussianBlur(gray_u8, (blur_ksize, blur_ksize), 0)
    else:
        work = gray_u8.copy()

    otsu_value, mask = cv2.threshold(
        work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if threshold_offset != 0:
        threshold = int(np.clip(otsu_value + threshold_offset, 0, 255))
        _, mask = cv2.threshold(work, threshold, 255, cv2.THRESH_BINARY)

    if morph_ksize and morph_ksize > 1:
        # Important: use an odd kernel size so morphology has a real center.
        if morph_ksize % 2 == 0:
            morph_ksize += 1

        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def contour_mean_intensity(gray_u8, contour):
    """Mean normalized grayscale intensity inside one contour."""
    x, y, w, h = cv2.boundingRect(contour)
    local_contour = contour - np.array([[[x, y]]], dtype=contour.dtype)
    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local_mask, [local_contour], -1, 255, -1)
    values = gray_u8[y:y+h, x:x+w][local_mask > 0]
    if values.size == 0:
        return 0.0
    return float(np.mean(values) / 255.0)



def _order_box_points(points):
    """Return box points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def rectified_square_crop(gray_u8, rect, border_fraction=0.12):
    """
    Warp one detected rotated square into a straight crop, then trim the border.

    The border trim is important because the edge of an atlas square often contains grid-bar
    shadow, threshold artifacts, or partial border pixels. Crack/non-uniform checks should
    focus on the square interior, not the square edge.
    """
    (cx, cy), (rw, rh), angle = rect
    width = int(max(round(float(rw)), 2))
    height = int(max(round(float(rh)), 2))
    if width < 3 or height < 3:
        return np.empty((0, 0), dtype=np.uint8)

    box = cv2.boxPoints(rect).astype(np.float32)
    src_pts = _order_box_points(box)

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    crop = cv2.warpPerspective(
        gray_u8,
        M,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Trim border after rectification.
    by = int(round(crop.shape[0] * border_fraction))
    bx = int(round(crop.shape[1] * border_fraction))
    if crop.shape[0] - 2 * by >= 4 and crop.shape[1] - 2 * bx >= 4:
        crop = crop[by:crop.shape[0] - by, bx:crop.shape[1] - bx]

    return crop


def measure_crack_features(
    gray_u8,
    rect,
    atlas_brightness_reference=None,
    crack_bright_delta_from_atlas=55,
    crack_bright_delta_from_square=35,
    crack_min_absolute_intensity=180,
    crack_white_fraction_threshold=0.08,
    crack_largest_component_fraction_threshold=0.06,
    crack_min_component_area_px=8,
):
    """
    Detect cracking as bright regions that stand out relative to this specific atlas image.

    Why this is adaptive:
    - A fixed threshold like 235 fails when the entire atlas is overexposed/white.
    - Here, a pixel only counts as crack-white if it is brighter than:
        1. a minimum absolute brightness,
        2. the atlas's normal square brightness + crack_bright_delta_from_atlas, and
        3. this square's own median brightness + crack_bright_delta_from_square.

    That means a uniformly bright/overexposed square should not be called cracked just because
    its pixels are white. The bright region must be locally brighter than the square background.
    """
    crop = rectified_square_crop(gray_u8, rect, border_fraction=0.12)
    if crop.size == 0:
        return {
            "is_cracked": False,
            "crack_white_fraction": 0.0,
            "crack_largest_component_fraction": 0.0,
            "crack_component_count": 0,
            "crack_threshold_used": 0.0,
            "crack_square_median": 0.0,
            "crack_atlas_reference": float(atlas_brightness_reference or 0.0),
        }

    crop_f = crop.astype(np.float32)
    square_median = float(np.median(crop_f))

    if atlas_brightness_reference is None or not np.isfinite(atlas_brightness_reference):
        atlas_brightness_reference = square_median
    atlas_brightness_reference = float(atlas_brightness_reference)

    # The actual bright threshold adapts to both the image and the current square.
    crack_threshold = max(
        float(crack_min_absolute_intensity),
        atlas_brightness_reference + float(crack_bright_delta_from_atlas),
        square_median + float(crack_bright_delta_from_square),
    )

    # If the threshold is above displayable white, the image/square is already too bright
    # for reliable white-crack detection. Do not call it cracked.
    if crack_threshold > 255.0:
        return {
            "is_cracked": False,
            "crack_white_fraction": 0.0,
            "crack_largest_component_fraction": 0.0,
            "crack_component_count": 0,
            "crack_threshold_used": float(crack_threshold),
            "crack_square_median": square_median,
            "crack_atlas_reference": atlas_brightness_reference,
        }

    white_mask = (crop_f >= crack_threshold).astype(np.uint8)

    # Light cleanup: remove isolated single-pixel noise, but do not close/fill regions because
    # that could merge separate small white specks into one artificial crack.
    if min(crop.shape[:2]) >= 10:
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    total_area = int(crop.size)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

    valid_areas = []
    for label_idx in range(1, component_count):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= int(crack_min_component_area_px):
            valid_areas.append(area)

    white_area = int(np.sum(valid_areas)) if valid_areas else 0
    largest_area = int(np.max(valid_areas)) if valid_areas else 0

    white_fraction = float(white_area / max(total_area, 1))
    largest_fraction = float(largest_area / max(total_area, 1))

    is_cracked = bool(
        white_fraction >= float(crack_white_fraction_threshold)
        or largest_fraction >= float(crack_largest_component_fraction_threshold)
    )

    return {
        "is_cracked": is_cracked,
        "crack_white_fraction": white_fraction,
        "crack_largest_component_fraction": largest_fraction,
        "crack_component_count": int(len(valid_areas)),
        "crack_threshold_used": float(crack_threshold),
        "crack_square_median": square_median,
        "crack_atlas_reference": atlas_brightness_reference,
    }


def measure_nonuniform_features(
    gray_u8,
    rect,
    atlas_light_reference=None,
    atlas_brightness_reference=None,
    crack_bright_delta_from_atlas=55,
    crack_bright_delta_from_square=35,
    nonuniform_dark_drop_from_atlas=45,
    nonuniform_dark_drop_from_square=25,
    nonuniform_dark_fraction_threshold=0.12,
    nonuniform_largest_dark_fraction_threshold=0.05,
    nonuniform_local_range_threshold=45,
    nonuniform_local_iqr_threshold=25,
    nonuniform_std_threshold=32,
    nonuniform_grid_size=4,
):
    """
    Detect non-uniform color as DARK blotches relative to this atlas image's normal bright squares.

    This is only called for squares that:
    1. passed size/squareness, and
    2. were not cracked.

    A pixel counts as dark if it is darker than the atlas's normal/light square reference
    minus nonuniform_dark_drop_from_atlas. Uniformly darker-but-smooth squares are not usually
    labeled non-uniform because the final decision still requires local variation or high spread.
    The target case is localized darker spots/patches.
    """
    crop = rectified_square_crop(gray_u8, rect, border_fraction=0.12)
    if crop.size == 0:
        return {
            "is_non_uniform": False,
            "nonuniform_dark_fraction": 0.0,
            "nonuniform_largest_dark_fraction": 0.0,
            "nonuniform_intensity_std": 0.0,
            "nonuniform_local_range": 0.0,
            "nonuniform_local_iqr": 0.0,
            "nonuniform_dark_threshold_used": 0.0,
            "nonuniform_square_median": 0.0,
            "nonuniform_atlas_light_reference": float(atlas_light_reference or 0.0),
        }

    crop_f = crop.astype(np.float32)
    square_median = float(np.median(crop_f))

    if atlas_light_reference is None or not np.isfinite(atlas_light_reference):
        atlas_light_reference = square_median
    atlas_light_reference = float(atlas_light_reference)

    if atlas_brightness_reference is None or not np.isfinite(atlas_brightness_reference):
        atlas_brightness_reference = atlas_light_reference
    atlas_brightness_reference = float(atlas_brightness_reference)

    # Ignore obviously bright crack-like pixels during non-uniform measurements. Usually cracks
    # were already handled in the previous stage, but this prevents missed bright artifacts from
    # inflating local range/std and turning a square blue.
    bright_ignore_threshold = max(
        235.0,
        atlas_brightness_reference + float(crack_bright_delta_from_atlas),
        square_median + float(crack_bright_delta_from_square),
    )
    valid_mask = crop_f < bright_ignore_threshold
    valid_values = crop_f[valid_mask]
    if valid_values.size < 10:
        return {
            "is_non_uniform": False,
            "nonuniform_dark_fraction": 0.0,
            "nonuniform_largest_dark_fraction": 0.0,
            "nonuniform_intensity_std": 0.0,
            "nonuniform_local_range": 0.0,
            "nonuniform_local_iqr": 0.0,
            "nonuniform_dark_threshold_used": 0.0,
            "nonuniform_square_median": square_median,
            "nonuniform_atlas_light_reference": atlas_light_reference,
        }

    intensity_std = float(np.std(valid_values))

    # Use the atlas-light reference, not only this square's own median. This is important because
    # a large dark blotch can pull the square median downward and would hide itself if the threshold
    # were based only on the candidate square.
    dark_threshold = float(atlas_light_reference - float(nonuniform_dark_drop_from_atlas))

    dark_mask = ((crop_f <= dark_threshold) & valid_mask).astype(np.uint8)
    dark_fraction = float(np.count_nonzero(dark_mask) / max(crop.size, 1))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    dark_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, component_count)]
    largest_dark_fraction = float((max(dark_areas) if dark_areas else 0) / max(crop.size, 1))

    # Compare local medians across a small grid. This catches broad dark patches/gradients.
    h, w = crop.shape[:2]
    local_medians = []
    grid = int(max(nonuniform_grid_size, 2))
    for gy in range(grid):
        y0 = int(round(gy * h / grid))
        y1 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            tile = crop_f[y0:y1, x0:x1]
            tile_valid = valid_mask[y0:y1, x0:x1]
            vals = tile[tile_valid]
            if vals.size >= 8:
                local_medians.append(float(np.median(vals)))

    if len(local_medians) >= 4:
        local_medians_arr = np.asarray(local_medians, dtype=np.float32)
        local_range = float(np.max(local_medians_arr) - np.min(local_medians_arr))
        local_iqr = float(np.percentile(local_medians_arr, 75) - np.percentile(local_medians_arr, 25))
        local_dark_deviation = float(atlas_light_reference - np.min(local_medians_arr))
    else:
        local_range = 0.0
        local_iqr = 0.0
        local_dark_deviation = 0.0

    dark_patch_signal = bool(
        dark_fraction >= float(nonuniform_dark_fraction_threshold)
        and largest_dark_fraction >= float(nonuniform_largest_dark_fraction_threshold)
    )
    local_signal = bool(
        local_range >= float(nonuniform_local_range_threshold)
        or local_iqr >= float(nonuniform_local_iqr_threshold)
        or local_dark_deviation >= float(nonuniform_dark_drop_from_atlas)
    )
    std_signal = bool(intensity_std >= float(nonuniform_std_threshold))

    # Require a real dark patch plus either local variation or strong intensity spread.
    is_non_uniform = bool(dark_patch_signal and (local_signal or std_signal))

    return {
        "is_non_uniform": is_non_uniform,
        "nonuniform_dark_fraction": dark_fraction,
        "nonuniform_largest_dark_fraction": largest_dark_fraction,
        "nonuniform_intensity_std": intensity_std,
        "nonuniform_local_range": local_range,
        "nonuniform_local_iqr": local_iqr,
        "nonuniform_local_dark_deviation": local_dark_deviation,
        "nonuniform_dark_threshold_used": dark_threshold,
        "nonuniform_square_median": square_median,
        "nonuniform_atlas_light_reference": atlas_light_reference,
    }


def detect_visible_square_candidates(
    bgr,
    min_contour_area_px=10,
    max_contour_area_frac=0.02,
    min_square_like=0.45,
    threshold_offset=0,
    crack_intensity_threshold=235,
    crack_white_fraction_threshold=0.08,
    crack_largest_component_fraction_threshold=0.06,
    crack_min_component_area_px=8,
    crack_max_bright_fraction_for_localized_defect=None,
):
    """
    Detect visible square-like bright regions in an atlas image.

    This stage only finds candidate squares and measures geometry/brightness.
    It does not assign cracked or non-uniform labels. Those labels are assigned later
    in a fixed order:
    1. bad size/squareness
    2. cracked
    3. non-uniform
    4. good
    """
    gray_u8 = normalize_gray_for_detection(bgr)
    mask = make_bright_square_mask(gray_u8, threshold_offset=threshold_offset)
    image_area = bgr.shape[0] * bgr.shape[1]
    max_contour_area_px = max_contour_area_frac * image_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_contour_area_px:
            continue
        if contour_area > max_contour_area_px:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        rw, rh = float(rw), float(rh)
        if rw <= 1 or rh <= 1:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        square_like = short_side / (long_side + 1e-6)
        if square_like < min_square_like:
            continue

        rect_area = max(rw * rh, 1.0)
        fill_ratio = float(np.clip(contour_area / rect_area, 0, 1.5))
        brightness = contour_mean_intensity(gray_u8, contour)
        box = cv2.boxPoints(rect).astype(np.float32)

        candidates.append({
            "contour": contour,
            "box": box,
            "rect": rect,
            "center_x": float(cx),
            "center_y": float(cy),
            "contour_area_px2": contour_area,
            "rect_w_px": rw,
            "rect_h_px": rh,
            "rect_side_mean_px": float((rw + rh) / 2),
            "rect_side_short_px": float(short_side),
            "rect_side_long_px": float(long_side),
            "rect_angle_deg": float(angle),
            "square_like": float(square_like),
            "fill_ratio": fill_ratio,
            "brightness_mean_norm": brightness,
            "square_status": "unclassified",
            "is_good_size": False,
            "is_cracked": False,
            "is_non_uniform": False,
        })

    return candidates, mask, gray_u8



def score_square_candidates(candidates, image_shape):
    """
    Give every visible candidate a 0-100 size/squareness score.

    This is the original size/shape scoring logic from the older notebook, with one important
    behavior change: crack and non-uniform labels do NOT affect this score. That lets us enforce
    the requested classification order:

    1. first decide if the square is good enough by size/squareness score
    2. only then check white cracking
    3. only if not cracked, check non-uniform color
    """
    if len(candidates) == 0:
        return candidates, None

    image_h, image_w = image_shape[:2]
    image_area_px2 = float(max(image_h * image_w, 1))
    detected_square_count = int(len(candidates))

    # Estimate the expected area of one visible square using the image size and square count.
    # This handles good high-count atlases where each individual square is naturally small.
    # The count is clipped so an extreme number of tiny false detections cannot make tiny blobs look normal.
    MIN_COUNT_FOR_AREA_MODEL = 20
    MAX_COUNT_FOR_AREA_MODEL = 220
    effective_square_count = int(np.clip(
        detected_square_count,
        MIN_COUNT_FOR_AREA_MODEL,
        MAX_COUNT_FOR_AREA_MODEL,
    ))

    # Approximate fraction of the whole atlas image that should be occupied by visible square interiors.
    EXPECTED_TOTAL_VISIBLE_SQUARE_AREA_FRACTION = 0.15
    expected_area_fraction_from_count = float(
        EXPECTED_TOTAL_VISIBLE_SQUARE_AREA_FRACTION / max(effective_square_count, 1)
    )

    AREA_ZERO_EXPECTED_RATIO = 0.35
    AREA_FULL_EXPECTED_RATIO = 0.85
    AREA_PENALTY_POWER = 2.25

    ABS_VERY_SMALL_AREA_FRACTION_OF_IMAGE = 0.00012
    ABS_SMALL_AREA_FRACTION_OF_IMAGE = 0.00020

    VERY_SMALL_EXPECTED_RATIO = 0.45
    SMALL_EXPECTED_RATIO = 0.65

    VERY_SMALL_SCORE_CAP = 15.0
    SMALL_SCORE_CAP = 40.0

    for c in candidates:
        area_fraction_of_image = float(c["contour_area_px2"] / image_area_px2)
        area_fraction_of_expected = float(
            area_fraction_of_image / max(expected_area_fraction_from_count, 1e-12)
        )

        area_score_linear = np.clip(
            (area_fraction_of_expected - AREA_ZERO_EXPECTED_RATIO)
            / (AREA_FULL_EXPECTED_RATIO - AREA_ZERO_EXPECTED_RATIO),
            0,
            1,
        )
        area_score = float(area_score_linear ** AREA_PENALTY_POWER)

        # 1 when close to square, 0 when near the minimum accepted ratio.
        shape_score = float(np.clip((c["square_like"] - 0.50) / (0.95 - 0.50), 0, 1))

        # Rounded corners and thresholding mean good square fill ratios are often < 1.
        fill_score = float(np.clip((c["fill_ratio"] - 0.45) / (0.90 - 0.45), 0, 1))

        # Useful but not dominant because exposure varies between images.
        brightness_score = float(np.clip((c["brightness_mean_norm"] - 0.25) / (0.95 - 0.25), 0, 1))

        quality_score = 100.0 * (
            0.55 * area_score +
            0.25 * shape_score +
            0.10 * fill_score +
            0.10 * brightness_score
        )

        size_score = float(np.clip(quality_score, 0, 100))
        small_area_cap_applied = False
        small_area_cap_reason = ""

        # Keep the original small-area caps, but apply them before any crack/non-uniform logic.
        if (
            area_fraction_of_image < ABS_VERY_SMALL_AREA_FRACTION_OF_IMAGE
            or area_fraction_of_expected < VERY_SMALL_EXPECTED_RATIO
        ):
            size_score = min(size_score, VERY_SMALL_SCORE_CAP)
            small_area_cap_applied = True
            small_area_cap_reason = "very_small_absolute_or_count_adjusted_area"
        elif (
            area_fraction_of_image < ABS_SMALL_AREA_FRACTION_OF_IMAGE
            or area_fraction_of_expected < SMALL_EXPECTED_RATIO
        ):
            size_score = min(size_score, SMALL_SCORE_CAP)
            small_area_cap_applied = True
            small_area_cap_reason = "small_absolute_or_count_adjusted_area"

        c["area_fraction_of_image"] = float(area_fraction_of_image)
        c["detected_square_count_for_area_score"] = int(detected_square_count)
        c["effective_square_count_for_area_score"] = int(effective_square_count)
        c["expected_area_fraction_from_count"] = float(expected_area_fraction_from_count)
        c["area_fraction_of_expected"] = float(area_fraction_of_expected)
        c["small_area_cap_applied"] = bool(small_area_cap_applied)
        c["small_area_cap_reason"] = small_area_cap_reason
        c["area_score"] = area_score
        c["shape_score"] = shape_score
        c["fill_score"] = fill_score
        c["brightness_score"] = brightness_score
        c["base_quality_score"] = size_score
        c["size_score"] = size_score
        c["quality_score"] = size_score

    candidates.sort(key=lambda item: item["size_score"], reverse=True)
    for i, c in enumerate(candidates, start=1):
        c["rank"] = i

    return candidates, image_area_px2


def estimate_atlas_square_brightness_references(candidates, gray_u8, good_score_threshold=70):
    """
    Estimate per-image square brightness references after size/squareness scoring.

    Only size-valid squares are used when possible. This keeps bad tiny/unsquare detections from
    skewing the crack/non-uniform baselines. The returned references are robust statistics of the
    median brightness of each valid square crop.
    """
    medians = []
    all_medians = []

    for c in candidates:
        crop = rectified_square_crop(gray_u8, c.get("rect"), border_fraction=0.12)
        if crop.size == 0:
            median_intensity = np.nan
        else:
            median_intensity = float(np.median(crop.astype(np.float32)))

        c["square_median_intensity"] = float(median_intensity) if np.isfinite(median_intensity) else 0.0
        if np.isfinite(median_intensity):
            all_medians.append(median_intensity)
            size_score = float(c.get("size_score", c.get("quality_score", 0.0)))
            if size_score >= float(good_score_threshold):
                medians.append(median_intensity)

    if len(medians) == 0:
        medians = all_medians
    if len(medians) == 0:
        references = {
            "atlas_square_median_reference": 128.0,
            "atlas_square_light_reference": 128.0,
            "atlas_square_dark_reference": 128.0,
            "atlas_reference_square_count": 0,
        }
    else:
        arr = np.asarray(medians, dtype=np.float32)

        # Trim very bright/dark outliers when enough squares exist. This makes the references less
        # sensitive to a few cracked or badly uneven squares.
        if arr.size >= 8:
            lo, hi = np.percentile(arr, [10, 90])
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            if trimmed.size >= 4:
                arr = trimmed

        references = {
            "atlas_square_median_reference": float(np.median(arr)),
            # Use a slightly high percentile as the baseline for non-uniform dark spots because
            # the user's target case is: normal/good squares are lighter, bad square has dark spots.
            "atlas_square_light_reference": float(np.percentile(arr, 70)),
            "atlas_square_dark_reference": float(np.percentile(arr, 30)),
            "atlas_reference_square_count": int(arr.size),
        }

    for c in candidates:
        c.update(references)
    return references




def apply_ordered_square_labels(
    candidates,
    gray_u8,
    good_score_threshold=70,
    crack_bright_delta_from_atlas=55,
    crack_bright_delta_from_square=35,
    crack_min_absolute_intensity=180,
    crack_white_fraction_threshold=0.08,
    crack_largest_component_fraction_threshold=0.06,
    crack_min_component_area_px=8,
    nonuniform_dark_drop_from_atlas=45,
    nonuniform_dark_drop_from_square=25,
    nonuniform_dark_fraction_threshold=0.12,
    nonuniform_largest_dark_fraction_threshold=0.05,
    nonuniform_local_range_threshold=45,
    nonuniform_local_iqr_threshold=25,
    nonuniform_std_threshold=32,
):
    """
    Assign exactly one label to each square in the requested order.

    Order:
    1. bad_size: fails the existing size/squareness score
    2. cracked: passes size/squareness, then has bright regions relative to this atlas
    3. non_uniform: passes size/squareness, is not cracked, then has dark unevenness relative to this atlas
    4. good: passes all checks
    """
    references = estimate_atlas_square_brightness_references(
        candidates,
        gray_u8,
        good_score_threshold=good_score_threshold,
    )
    atlas_brightness_reference = references["atlas_square_median_reference"]
    atlas_light_reference = references["atlas_square_light_reference"]

    for c in candidates:
        size_score = float(c.get("size_score", c.get("quality_score", 0.0)))
        is_good_size = bool(size_score >= float(good_score_threshold))

        # Default/reset fields so stale values cannot leak from earlier versions.
        c["is_good_size"] = is_good_size
        c["is_cracked"] = False
        c["is_non_uniform"] = False
        c["square_status"] = "bad_size" if not is_good_size else "good"
        c["final_quality_score"] = 0.0 if not is_good_size else size_score

        c["crack_white_fraction"] = 0.0
        c["crack_largest_component_fraction"] = 0.0
        c["crack_component_count"] = 0
        c["crack_threshold_used"] = 0.0

        c["nonuniform_dark_fraction"] = 0.0
        c["nonuniform_largest_dark_fraction"] = 0.0
        c["nonuniform_intensity_std"] = 0.0
        c["nonuniform_local_range"] = 0.0
        c["nonuniform_local_iqr"] = 0.0
        c["nonuniform_local_dark_deviation"] = 0.0
        c["nonuniform_dark_threshold_used"] = 0.0

        # Terminal rule: size/squareness failures stay bad_size.
        # They are not allowed to become cracked or non-uniform.
        if not is_good_size:
            continue

        crack_features = measure_crack_features(
            gray_u8,
            c["rect"],
            atlas_brightness_reference=atlas_brightness_reference,
            crack_bright_delta_from_atlas=crack_bright_delta_from_atlas,
            crack_bright_delta_from_square=crack_bright_delta_from_square,
            crack_min_absolute_intensity=crack_min_absolute_intensity,
            crack_white_fraction_threshold=crack_white_fraction_threshold,
            crack_largest_component_fraction_threshold=crack_largest_component_fraction_threshold,
            crack_min_component_area_px=crack_min_component_area_px,
        )
        c.update(crack_features)

        if bool(crack_features.get("is_cracked", False)):
            c["square_status"] = "cracked"
            c["is_cracked"] = True
            c["is_non_uniform"] = False
            c["final_quality_score"] = 0.0
            continue

        nonuniform_features = measure_nonuniform_features(
            gray_u8,
            c["rect"],
            atlas_light_reference=atlas_light_reference,
            atlas_brightness_reference=atlas_brightness_reference,
            crack_bright_delta_from_atlas=crack_bright_delta_from_atlas,
            crack_bright_delta_from_square=crack_bright_delta_from_square,
            nonuniform_dark_drop_from_atlas=nonuniform_dark_drop_from_atlas,
            nonuniform_dark_drop_from_square=nonuniform_dark_drop_from_square,
            nonuniform_dark_fraction_threshold=nonuniform_dark_fraction_threshold,
            nonuniform_largest_dark_fraction_threshold=nonuniform_largest_dark_fraction_threshold,
            nonuniform_local_range_threshold=nonuniform_local_range_threshold,
            nonuniform_local_iqr_threshold=nonuniform_local_iqr_threshold,
            nonuniform_std_threshold=nonuniform_std_threshold,
        )
        c.update(nonuniform_features)

        if bool(nonuniform_features.get("is_non_uniform", False)):
            c["square_status"] = "non_uniform"
            c["is_non_uniform"] = True
            c["final_quality_score"] = 0.0
        else:
            c["square_status"] = "good"
            c["final_quality_score"] = size_score

    return candidates


def score_to_bgr(score):
    """Simple red-yellow-green style color for overlay boxes."""
    score = float(np.clip(score, 0, 100))
    if score >= 70:
        return (0, 220, 0)
    if score >= 45:
        return (0, 210, 255)
    return (0, 0, 255)



def candidate_overlay_bgr(candidate):
    """Overlay color by final ordered label."""
    status = str(candidate.get("square_status", ""))

    if status == "cracked" or candidate.get("is_cracked", False):
        return (255, 0, 255)   # magenta
    if status == "non_uniform" or candidate.get("is_non_uniform", False):
        return (255, 0, 0)     # blue
    if status == "good":
        return (0, 220, 0)     # green
    if status == "bad_size":
        return (0, 0, 255)     # red

    return score_to_bgr(candidate.get("size_score", candidate.get("quality_score", 0)))



def draw_overlay(bgr, candidates, summary, overlay_path):
    """Draw detected boxes on the original image using final ordered labels."""
    overlay = bgr.copy()

    for c in candidates:
        color = candidate_overlay_bgr(c)
        box = np.rint(c["box"]).astype(np.int32)
        cv2.polylines(overlay, [box], isClosed=True, color=color, thickness=1)
        cx, cy = int(round(c["center_x"])), int(round(c["center_y"]))
        cv2.circle(overlay, (cx, cy), 2, color, -1)

    text_lines = [
        f"visible: {summary['visible_square_count']}",
        f"good: {summary['good_square_count']}",
        f"non-uniform: {summary['non_uniform_square_count']}",
        f"cracked: {summary['cracked_square_count']}",
        f"bad size: {summary['bad_size_square_count']}",
        f"atlas score: {summary['atlas_quality_score']:.1f}/100",
    ]
    y = 22
    for line in text_lines:
        cv2.putText(overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 16

    legend_lines = [
        ("green = good", (0, 220, 0)),
        ("blue = non-uniform", (255, 0, 0)),
        ("magenta = cracked", (255, 0, 255)),
        ("red = bad size/shape", (0, 0, 255)),
    ]
    y += 8
    for text, color in legend_lines:
        cv2.putText(overlay, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        y += 16

    cv2.imwrite(str(overlay_path), overlay)
    return overlay


def candidate_rows_for_csv(candidates):
    """Return the per-square measurements that should be saved in the combined CSV."""
    rows = []
    for c in candidates:
        rows.append({
            "rank": int(c.get("rank", 0)),
            "square_status": str(c.get("square_status", "")),
            "size_score": float(c.get("size_score", c.get("quality_score", 0.0))),
            "detected_square_count_for_area_score": int(c.get("detected_square_count_for_area_score", 0)),
            "effective_square_count_for_area_score": int(c.get("effective_square_count_for_area_score", 0)),
            "area_score": float(c.get("area_score", 0.0)),
            "shape_score": float(c.get("shape_score", 0.0)),
            "fill_score": float(c.get("fill_score", 0.0)),
            "brightness_score": float(c.get("brightness_score", 0.0)),
            "is_good_size": bool(c.get("is_good_size", False)),
            "is_cracked": bool(c.get("is_cracked", False)),
            "is_non_uniform": bool(c.get("is_non_uniform", False)),
            "square_median_intensity": float(c.get("square_median_intensity", 0.0)),
            "atlas_square_median_reference": float(c.get("atlas_square_median_reference", 0.0)),
            "atlas_square_light_reference": float(c.get("atlas_square_light_reference", 0.0)),
            "crack_threshold_used": float(c.get("crack_threshold_used", 0.0)),
            "crack_white_fraction": float(c.get("crack_white_fraction", 0.0)),
            "crack_largest_component_fraction": float(c.get("crack_largest_component_fraction", 0.0)),
            "nonuniform_dark_threshold_used": float(c.get("nonuniform_dark_threshold_used", 0.0)),
            "nonuniform_dark_fraction": float(c.get("nonuniform_dark_fraction", 0.0)),
            "nonuniform_largest_dark_fraction": float(c.get("nonuniform_largest_dark_fraction", 0.0)),
            "nonuniform_local_range": float(c.get("nonuniform_local_range", 0.0)),
            "nonuniform_local_iqr": float(c.get("nonuniform_local_iqr", 0.0)),
            "nonuniform_local_dark_deviation": float(c.get("nonuniform_local_dark_deviation", 0.0)),
            "nonuniform_intensity_std": float(c.get("nonuniform_intensity_std", 0.0)),
        })
    return rows


def image_rows_for_csv(image_path, summary, square_rows=None, error=""):
    """Attach image-level atlas scores to every square row for the combined CSV."""
    image_path = Path(image_path)
    square_rows = square_rows or []
    base = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "atlas_quality_score": float(summary.get("atlas_quality_score", 0.0)),
        "visible_square_count": int(summary.get("visible_square_count", 0)),
        "good_square_count": int(summary.get("good_square_count", 0)),
        "non_uniform_square_count": int(summary.get("non_uniform_square_count", 0)),
        "cracked_square_count": int(summary.get("cracked_square_count", 0)),
        "bad_size_square_count": int(summary.get("bad_size_square_count", 0)),
        "annotated_image_path": str(summary.get("annotated_image_path", "") or ""),
        "error": str(error or summary.get("error", "") or ""),
    }

    if len(square_rows) == 0:
        return [base]

    rows = []
    for row in square_rows:
        combined = dict(base)
        combined.update(row)
        rows.append(combined)
    return rows


def summary_row_for_csv(image_path, summary, error=""):
    """Return the atlas-level row needed by the CSV analyzer."""
    image_path = Path(image_path)
    return {
        "image_name": image_path.name,
        "visible_square_count": int(summary.get("visible_square_count", 0)),
        "good_square_count": int(summary.get("good_square_count", 0)),
        "non_uniform_square_count": int(summary.get("non_uniform_square_count", 0)),
        "cracked_square_count": int(summary.get("cracked_square_count", 0)),
        "bad_size_square_count": int(summary.get("bad_size_square_count", 0)),
        "atlas_quality_score": float(summary.get("atlas_quality_score", 0.0)),
        "error": str(error or summary.get("error", "") or ""),
    }


def preferred_atlas_summary_columns():
    """Column order for the one-row-per-atlas summary CSV."""
    return list(ATLAS_SUMMARY_COLUMNS)


def preferred_atlas_square_columns():
    """Column order for the one combined folder-level CSV."""
    return [
        "image_name",
        "image_path",
        "atlas_quality_score",
        "visible_square_count",
        "good_square_count",
        "non_uniform_square_count",
        "cracked_square_count",
        "bad_size_square_count",
        "annotated_image_path",
        "error",
        "rank",
        "square_status",
        "size_score",
        "detected_square_count_for_area_score",
        "effective_square_count_for_area_score",
        "area_score",
        "shape_score",
        "fill_score",
        "brightness_score",
        "is_good_size",
        "is_cracked",
        "is_non_uniform",
        "square_median_intensity",
        "atlas_square_median_reference",
        "atlas_square_light_reference",
        "crack_threshold_used",
        "crack_white_fraction",
        "crack_largest_component_fraction",
        "nonuniform_dark_threshold_used",
        "nonuniform_dark_fraction",
        "nonuniform_largest_dark_fraction",
        "nonuniform_local_range",
        "nonuniform_local_iqr",
        "nonuniform_local_dark_deviation",
        "nonuniform_intensity_std",
    ]


def compute_atlas_quality_summary(candidates, image_name, good_score_threshold=70):
    """
    Simplified atlas-level summary.

    Counts are based on the final ordered labels:
    - good
    - non_uniform
    - cracked
    - bad_size

    The atlas score is mainly the fraction of visible squares that stayed good, with a small
    bonus for having many good squares so an image with only a few good detections does not
    receive an unrealistically high score.
    """
    visible_count = len(candidates)
    if visible_count == 0:
        return {
            "image_name": image_name,
            "visible_square_count": 0,
            "good_square_count": 0,
            "non_uniform_square_count": 0,
            "cracked_square_count": 0,
            "bad_size_square_count": 0,
            "atlas_quality_score": 0.0,
        }

    status_list = [str(c.get("square_status", "")) for c in candidates]

    good_count = int(sum(s == "good" for s in status_list))
    non_uniform_count = int(sum(s == "non_uniform" for s in status_list))
    cracked_count = int(sum(s == "cracked" for s in status_list))
    bad_size_count = int(sum(s == "bad_size" for s in status_list))

    # Safety: anything unclassified counts as bad size/shape.
    unclassified_count = int(visible_count - good_count - non_uniform_count - cracked_count - bad_size_count)
    bad_size_count += max(unclassified_count, 0)

    good_ratio = float(good_count / visible_count)

    # More good squares is better, but this component saturates.
    # This keeps "3 good out of 3 visible" from looking as strong as "250 good out of 300 visible."
    count_score = float(1.0 - np.exp(-good_count / 300.0))

    atlas_quality = 100.0 * (
        0.75 * good_ratio +
        0.25 * count_score
    )

    return {
        "image_name": image_name,
        "visible_square_count": int(visible_count),
        "good_square_count": good_count,
        "non_uniform_square_count": non_uniform_count,
        "cracked_square_count": cracked_count,
        "bad_size_square_count": bad_size_count,
        "atlas_quality_score": float(np.clip(atlas_quality, 0, 100)),
    }




def analyze_atlas_image(
    input_path,
    output_root="outputs_step2",
    save_annotated_image=True,
    annotated_image_path=None,
    min_contour_area_px=10,
    min_square_like=0.45,
    threshold_offset=0,
    good_score_threshold=70,

    # Crack = bright areas relative to this atlas image, not just absolute white pixels.
    crack_bright_delta_from_atlas=55,
    crack_bright_delta_from_square=35,
    crack_min_absolute_intensity=180,
    crack_white_fraction_threshold=0.08,
    crack_largest_component_fraction_threshold=0.06,
    crack_min_component_area_px=8,

    # Non-uniform = darker patches relative to the atlas's normal/light squares.
    nonuniform_dark_drop_from_atlas=45,
    nonuniform_dark_drop_from_square=25,
    nonuniform_dark_fraction_threshold=0.12,
    nonuniform_largest_dark_fraction_threshold=0.05,
    nonuniform_local_range_threshold=45,
    nonuniform_local_iqr_threshold=25,
    nonuniform_std_threshold=32,
):
    """
    Analyze one atlas image and return rows for the combined folder-level CSV.

    This keeps the existing square detection and labeling logic, but does not write
    per-image CSV or JSON files. Annotated images are saved only when requested.
    """
    image_path = Path(input_path)
    bgr = read_image_as_bgr(image_path)

    candidates, mask, gray_detection_u8 = detect_visible_square_candidates(
        bgr,
        min_contour_area_px=min_contour_area_px,
        min_square_like=min_square_like,
        threshold_offset=threshold_offset,
    )

    candidates, reference_area = score_square_candidates(candidates, bgr.shape)

    # Use raw grayscale for crack/non-uniform measurements. Detection uses normalized gray, but
    # feature scoring should preserve exposure differences so per-atlas adaptive thresholds work.
    gray_features_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    candidates = apply_ordered_square_labels(
        candidates,
        gray_features_u8,
        good_score_threshold=good_score_threshold,
        crack_bright_delta_from_atlas=crack_bright_delta_from_atlas,
        crack_bright_delta_from_square=crack_bright_delta_from_square,
        crack_min_absolute_intensity=crack_min_absolute_intensity,
        crack_white_fraction_threshold=crack_white_fraction_threshold,
        crack_largest_component_fraction_threshold=crack_largest_component_fraction_threshold,
        crack_min_component_area_px=crack_min_component_area_px,
        nonuniform_dark_drop_from_atlas=nonuniform_dark_drop_from_atlas,
        nonuniform_dark_drop_from_square=nonuniform_dark_drop_from_square,
        nonuniform_dark_fraction_threshold=nonuniform_dark_fraction_threshold,
        nonuniform_largest_dark_fraction_threshold=nonuniform_largest_dark_fraction_threshold,
        nonuniform_local_range_threshold=nonuniform_local_range_threshold,
        nonuniform_local_iqr_threshold=nonuniform_local_iqr_threshold,
        nonuniform_std_threshold=nonuniform_std_threshold,
    )

    summary = compute_atlas_quality_summary(
        candidates,
        image_name=image_path.name,
        good_score_threshold=good_score_threshold,
    )
    summary["annotated_image_path"] = ""

    if save_annotated_image:
        if annotated_image_path is None:
            annotated_dir = make_dir(Path(output_root) / "annotated_images")
            annotated_image_path = annotated_dir / f"{image_path.stem}_annotated.png"
        else:
            annotated_image_path = Path(annotated_image_path)
            make_dir(annotated_image_path.parent)

        draw_overlay(
            bgr,
            candidates,
            summary,
            annotated_image_path,
        )
        summary["annotated_image_path"] = str(annotated_image_path)

    square_rows = candidate_rows_for_csv(candidates)
    image_rows = image_rows_for_csv(image_path, summary, square_rows=square_rows)
    return summary, image_rows



def find_images(input_path, recursive=True):
    """Return sorted image paths from one image file or a directory."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])



def analyze_atlas_folder(
    input_path,
    output_root="outputs_step2",
    recursive=True,
    save_annotated_images=True,
    output_csv_name="atlas_summary_scores.csv",
    output_format="summary",
    write_square_details=False,
    square_details_csv_name="atlas_square_data.csv",
    min_contour_area_px=10,    # lower detects smaller/fainter visible squares
    min_square_like=0.45,      # lower accepts more damaged/rounded squares
    threshold_offset=0,        # negative = more permissive, positive = stricter
    good_score_threshold=70,   # cutoff for passing the size/squareness stage

    # Crack thresholds.
    # Increase crack_bright_delta_from_atlas / crack_bright_delta_from_square if too many
    # overexposed white squares become cracked. Decrease them if obvious white cracks are missed.
    crack_bright_delta_from_atlas=55,
    crack_bright_delta_from_square=35,
    crack_min_absolute_intensity=180,
    crack_white_fraction_threshold=0.08,
    crack_largest_component_fraction_threshold=0.06,
    crack_min_component_area_px=8,

    # Non-uniform thresholds.
    # These only matter for squares that first pass size/squareness and are not cracked.
    nonuniform_dark_drop_from_atlas=45,
    nonuniform_dark_drop_from_square=25,
    nonuniform_dark_fraction_threshold=0.12,
    nonuniform_largest_dark_fraction_threshold=0.05,
    nonuniform_local_range_threshold=45,
    nonuniform_local_iqr_threshold=25,
    nonuniform_std_threshold=32,
):
    """Analyze atlas images and save a summary CSV plus optional square details."""
    output_format = str(output_format).strip().lower()
    if output_format not in {"summary", "squares", "square_details", "detailed"}:
        raise ValueError("output_format must be 'summary' or 'squares'")

    image_paths = find_images(input_path, recursive=recursive)
    print(f"Found {len(image_paths)} image(s).")
    if len(image_paths) == 0:
        return pd.DataFrame()

    output_root = make_dir(output_root)
    all_rows = []
    summary_rows = []

    for i, path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] Analyzing {path.name}")
        try:
            annotated_image_path = None
            if save_annotated_images:
                annotated_image_path = output_root / "annotated_images" / f"{i:04d}_{path.stem}_annotated.png"

            summary, image_rows = analyze_atlas_image(
                path,
                output_root=output_root,
                save_annotated_image=save_annotated_images,
                annotated_image_path=annotated_image_path,
                min_contour_area_px=min_contour_area_px,
                min_square_like=min_square_like,
                threshold_offset=threshold_offset,
                good_score_threshold=good_score_threshold,
                crack_bright_delta_from_atlas=crack_bright_delta_from_atlas,
                crack_bright_delta_from_square=crack_bright_delta_from_square,
                crack_min_absolute_intensity=crack_min_absolute_intensity,
                crack_white_fraction_threshold=crack_white_fraction_threshold,
                crack_largest_component_fraction_threshold=crack_largest_component_fraction_threshold,
                crack_min_component_area_px=crack_min_component_area_px,
                nonuniform_dark_drop_from_atlas=nonuniform_dark_drop_from_atlas,
                nonuniform_dark_drop_from_square=nonuniform_dark_drop_from_square,
                nonuniform_dark_fraction_threshold=nonuniform_dark_fraction_threshold,
                nonuniform_largest_dark_fraction_threshold=nonuniform_largest_dark_fraction_threshold,
                nonuniform_local_range_threshold=nonuniform_local_range_threshold,
                nonuniform_local_iqr_threshold=nonuniform_local_iqr_threshold,
                nonuniform_std_threshold=nonuniform_std_threshold,
            )
        except Exception as e:
            error_text = repr(e)
            summary = {
                "image_name": path.name,
                "visible_square_count": 0,
                "good_square_count": 0,
                "non_uniform_square_count": 0,
                "cracked_square_count": 0,
                "bad_size_square_count": 0,
                "atlas_quality_score": 0.0,
                "annotated_image_path": "",
                "error": error_text,
            }
            image_rows = image_rows_for_csv(path, summary, square_rows=[], error=error_text)
            print(f"  ERROR: {e}")

        all_rows.extend(image_rows)
        summary_rows.append(summary_row_for_csv(path, summary))

    wants_square_output = output_format in {"squares", "square_details", "detailed"}
    atlas_df = pd.DataFrame(all_rows if wants_square_output else summary_rows)
    preferred_columns = preferred_atlas_square_columns() if wants_square_output else preferred_atlas_summary_columns()
    ordered_columns = [c for c in preferred_columns if c in atlas_df.columns]
    extra_columns = [c for c in atlas_df.columns if c not in ordered_columns]
    atlas_df = atlas_df[ordered_columns + extra_columns]

    output_csv_path = output_root / output_csv_name
    atlas_df.to_csv(output_csv_path, index=False)
    print(f"Saved atlas {'square table' if wants_square_output else 'summary table'}: {output_csv_path}")

    if write_square_details and not wants_square_output:
        square_df = pd.DataFrame(all_rows)
        square_preferred_columns = preferred_atlas_square_columns()
        square_ordered_columns = [c for c in square_preferred_columns if c in square_df.columns]
        square_extra_columns = [c for c in square_df.columns if c not in square_ordered_columns]
        square_df = square_df[square_ordered_columns + square_extra_columns]
        square_output_csv_path = output_root / square_details_csv_name
        square_df.to_csv(square_output_csv_path, index=False)
        print(f"Saved atlas square detail table: {square_output_csv_path}")

    if save_annotated_images:
        print(f"Saved annotated images in: {output_root / 'annotated_images'}")
    else:
        print("Skipped annotated images.")

    return atlas_df

def atlas_prerecognize(
    input_path,
    output=None,
    recursive=True,
    save_annotated_images=True,
    output_format="summary",
    write_square_details=False,
    **kwargs,
):
    """Run atlas prerecognition and write an atlas summary CSV.

    ``output`` may be a CSV path or an output directory. The returned dictionary
    contains the CSV path, row count, and generated rows as JSON-serializable
    records.
    """
    output_format = str(output_format).strip().lower()
    default_csv_name = "atlas_square_data.csv" if output_format in {"squares", "square_details", "detailed"} else "atlas_summary_scores.csv"

    if output is None:
        output_path = Path(default_csv_name)
    else:
        output_path = Path(output)

    if output_path.suffix.lower() == ".csv":
        output_root = output_path.parent if str(output_path.parent) else Path(".")
        output_csv_name = output_path.name
    else:
        output_root = output_path
        output_csv_name = default_csv_name

    df = analyze_atlas_folder(
        input_path,
        output_root=output_root,
        recursive=recursive,
        save_annotated_images=save_annotated_images,
        output_csv_name=output_csv_name,
        output_format=output_format,
        write_square_details=write_square_details,
        **kwargs,
    )
    csv_path = output_root / output_csv_name
    return {
        "csv_path": str(csv_path),
        "summary_csv_path": str(csv_path) if output_format == "summary" else "",
        "output_format": output_format,
        "row_count": int(len(df)),
        "rows": df.to_dict(orient="records"),
    }
