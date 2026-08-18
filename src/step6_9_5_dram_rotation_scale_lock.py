"""
DRIFT-SENSE - STEP 6.9.5
DRAM ROTATION-SCALE LOCKED LOCALIZATION

Purpose
-------
Fixes the main failure observed in 6.9.x:

1. DRAM v2 reference = 272x272 patch.
2. Search = 1200x1200 image.
3. Target is a transformed copy of the reference.
4. The earlier pipeline searched only scale and then allowed an AI
   reranker to select periodic distractors.
5. This version performs geometry-aware matching first:
      coarse half-resolution search
      -> rotation + scale sweep
      -> top geometric candidate
      -> full-resolution local verification
   AI is NOT allowed to move the final coordinate.

The dataset is synthetic DRAM 1T1C-style data. Ground truth is used
only for evaluation, never for prediction.

Run:
    python -u step6_9_5_dram_rotation_scale_lock.py

Expected dataset:
    dram_dataset_v2/
        reference/
        search/
        ground_truth.csv
"""

import os
import cv2
import math
import time
import numpy as np
import pandas as pd

DATASET = "dram_dataset_v2"
REF_DIR = os.path.join(DATASET, "reference")
SEARCH_DIR = os.path.join(DATASET, "search")
GT_FILE = os.path.join(DATASET, "ground_truth.csv")

RESULT_DIR = "results_v19"
os.makedirs(RESULT_DIR, exist_ok=True)

# Coarse search is deliberately centered around the transformation range
# documented by the DRAM v2 generator: approximately +/-4 degrees and
# scale around 1.0.
ANGLES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
SCALES = [0.92, 0.96, 1.00, 1.04, 1.08]

DOWNSAMPLE = 0.5
LOCAL_PAD = 20


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def transform_template(ref, angle_deg, scale):
    h, w = ref.shape
    M = cv2.getRotationMatrix2D(
        (w / 2.0, h / 2.0),
        angle_deg,
        scale
    )
    return cv2.warpAffine(
        ref,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def coarse_search(ref, search):
    """
    Fast global search at 1/2 resolution.

    Returns the best geometric hypothesis:
        score, x_center, y_center, angle, scale
    """
    ref_small = cv2.resize(
        ref, None,
        fx=DOWNSAMPLE,
        fy=DOWNSAMPLE,
        interpolation=cv2.INTER_AREA
    )
    search_small = cv2.resize(
        search, None,
        fx=DOWNSAMPLE,
        fy=DOWNSAMPLE,
        interpolation=cv2.INTER_AREA
    )

    h, w = ref_small.shape
    best = None

    for angle in ANGLES:
        for scale in SCALES:
            tpl = transform_template(
                ref_small,
                angle,
                scale
            )

            if (
                tpl.shape[0] >= search_small.shape[0]
                or tpl.shape[1] >= search_small.shape[1]
            ):
                continue

            response = cv2.matchTemplate(
                search_small,
                tpl,
                cv2.TM_CCOEFF_NORMED
            )

            _, score, _, loc = cv2.minMaxLoc(response)

            x = (loc[0] + w / 2.0) / DOWNSAMPLE
            y = (loc[1] + h / 2.0) / DOWNSAMPLE

            item = {
                "score": float(score),
                "x": float(x),
                "y": float(y),
                "angle": float(angle),
                "scale": float(scale)
            }

            if best is None or item["score"] > best["score"]:
                best = item

    return best


def local_full_resolution_verify(ref, search, coarse):
    """
    Rebuild the selected transformed reference at full resolution and
    perform a local full-resolution match around the coarse location.

    Crucially, this stage can refine the location but cannot jump to an
    unrelated periodic cell elsewhere in the 1200x1200 search image.
    """
    tpl = transform_template(
        ref,
        coarse["angle"],
        coarse["scale"]
    )

    h, w = tpl.shape

    cx = coarse["x"]
    cy = coarse["y"]

    x0 = max(
        0,
        int(round(cx - w / 2.0 - LOCAL_PAD))
    )
    y0 = max(
        0,
        int(round(cy - h / 2.0 - LOCAL_PAD))
    )
    x1 = min(
        search.shape[1],
        int(round(cx + w / 2.0 + LOCAL_PAD))
    )
    y1 = min(
        search.shape[0],
        int(round(cy + h / 2.0 + LOCAL_PAD))
    )

    roi = search[y0:y1, x0:x1]

    if (
        roi.shape[0] < tpl.shape[0]
        or roi.shape[1] < tpl.shape[1]
    ):
        return coarse["x"], coarse["y"], coarse["score"]

    response = cv2.matchTemplate(
        roi,
        tpl,
        cv2.TM_CCOEFF_NORMED
    )

    _, score, _, loc = cv2.minMaxLoc(response)

    px = x0 + loc[0] + w / 2.0
    py = y0 + loc[1] + h / 2.0

    return float(px), float(py), float(score)


def main():
    if not os.path.exists(GT_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {GT_FILE}\n"
            "Run this program from the Drift Sence project folder."
        )

    gt = pd.read_csv(GT_FILE)

    errors = []
    rows = []

    print("=" * 95)
    print("DRIFT-SENSE - STEP 6.9.5")
    print("DRAM ROTATION-SCALE LOCKED LOCALIZATION")
    print("=" * 95)
    print(f"Samples: {len(gt)}")
    print("Prediction: NO ground truth used")
    print("AI reranking: DISABLED for coordinate selection")
    print("Reason: previous AI reranking selected periodic distractors")
    print("Pipeline: coarse rotation/scale -> local full-resolution verification")
    print()

    for i, row in gt.iterrows():
        ref = read_gray(
            os.path.join(REF_DIR, str(row["reference"]))
        )
        search = read_gray(
            os.path.join(SEARCH_DIR, str(row["search"]))
        )

        gx = float(row["gt_x"])
        gy = float(row["gt_y"])

        t0 = time.perf_counter()

        coarse = coarse_search(ref, search)

        if coarse is None:
            px = float("nan")
            py = float("nan")
            final_score = 0.0
            mode = "NO_CANDIDATE"
            angle = 0.0
            scale = 1.0
        else:
            px, py, final_score = local_full_resolution_verify(
                ref,
                search,
                coarse
            )
            angle = coarse["angle"]
            scale = coarse["scale"]
            mode = "GEOMETRIC_LOCKED"

        err = math.hypot(px - gx, py - gy)

        elapsed = (time.perf_counter() - t0) * 1000.0

        errors.append(err)

        print(
            f"{i+1:02d}/{len(gt)} | "
            f"GT=({gx:.0f},{gy:.0f}) | "
            f"Pred=({px:.1f},{py:.1f}) | "
            f"Error={err:.2f}px | "
            f"Score={final_score:.4f} | "
            f"Rot={angle:+.1f} | "
            f"Scale={scale:.2f} | "
            f"Time={elapsed:.1f}ms"
        )

        rows.append({
            "sample": i + 1,
            "reference": row["reference"],
            "search": row["search"],
            "gt_x": gx,
            "gt_y": gy,
            "pred_x": px,
            "pred_y": py,
            "error_px": err,
            "score": final_score,
            "rotation_deg": angle,
            "scale": scale,
            "mode": mode,
            "time_ms": elapsed
        })

    errors = np.asarray(errors, dtype=np.float64)
    result = pd.DataFrame(rows)

    csv_path = os.path.join(
        RESULT_DIR,
        "final_results.csv"
    )
    result.to_csv(csv_path, index=False)

    print()
    print("=" * 95)
    print("FINAL RESULT - STEP 6.9.5")
    print("=" * 95)
    print(f"Total Samples       : {len(errors)}")
    print(f"Average Error       : {errors.mean():.2f} pixels")
    print(f"Median Error        : {np.median(errors):.2f} pixels")
    print(f"Minimum Error       : {errors.min():.2f} pixels")
    print(f"Maximum Error       : {errors.max():.2f} pixels")
    print()

    for threshold in [1, 3, 5, 10]:
        print(
            f"Within {threshold} Pixel"
            f"{'s' if threshold != 1 else ''}      : "
            f"{int((errors <= threshold).sum())}/{len(errors)}"
        )

    print()
    print(f"Average Score       : {result['score'].mean():.4f}")
    print(f"Average Inference   : {result['time_ms'].mean():.2f} ms")
    print()
    print(f"Results saved       : {csv_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
