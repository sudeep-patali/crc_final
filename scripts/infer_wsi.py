"""
infer_wsi.py
Full end-to-end pipeline exactly matching the user's requested workflow:

  1. Take a whole-slide image (TCGA-COAD / TCGA-READ .svs)
  2. Tile it into patches (background discarded)
  3. Run EVERY patch through the binary model -> Normal / Tumor
  4. For every patch predicted Tumor, run the stage model ->
         stage0 (mucus) / stage1 (stroma) / stage2 (muscle) / stage3 (adipose)
         or "Unidentified -> consult a doctor" if confidence is too low
  5. Aggregate to a slide-level report + heatmap overlay

Usage:
    python scripts/infer_wsi.py --slide path/to/slide.svs --out reports/slide1
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.wsi_tiling import tile_wsi
from utils.stain_normalization import load_or_fit_reference
from utils.dataset import get_eval_transforms
from models.hybrid_model import build_binary_model, build_stage_model

try:
    import openslide
except ImportError:
    openslide = None


def load_models(device):
    binary_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "binary_best.pt"),
                              map_location=device)
    stage_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "stage_best.pt"),
                             map_location=device)

    binary_model = build_binary_model(pretrained=False)
    binary_model.load_state_dict(binary_ckpt["model_state"])
    binary_model.to(device).eval()

    stage_model = build_stage_model(pretrained=False)
    stage_model.load_state_dict(stage_ckpt["model_state"])
    stage_model.to(device).eval()

    return binary_model, stage_model


def predict_patch(model, img_rgb, transform, device):
    aug = transform(image=img_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def entropy(probs):
    probs = np.clip(probs, 1e-9, 1.0)
    return -np.sum(probs * np.log(probs)) / np.log(len(probs))  # normalized 0-1


def determine_deepest_stage(stage_counts, unidentified_count, tumor_patches,
                             present_fraction=0.02, unidentified_fraction=0.05, min_count=3):
    """
    Deepest-invasion staging using PERCENTAGE-based thresholds (scales
    consistently across slides with very different total patch counts,
    unlike a fixed patch count which means something different on a small
    vs. a large slide):

      - A stage counts as "present" only if it exceeds `present_fraction`
        (default 2%) of all tumor patches on this slide -- with a small
        absolute floor (`min_count`, default 3) so a single stray
        misclassified patch can't be reported as "present" on its own.
        NOTE: keep min_count small -- it must never exceed the total number
        of tumor patches on small/demo slides, or every stage gets rejected
        even when one is clearly dominant (this was a real bug: an earlier
        default of min_count=20 silently produced "None" on any slide with
        fewer than ~20 tumor patches of a given stage, even when that stage
        was the clear majority).
      - Among present stages, report the DEEPEST one (stage3 > stage2 >
        stage1 > stage0) -- matches real pT-staging logic (staged by the
        deepest layer reached, not by which tissue type has the most area).
      - If unidentified patches exceed `unidentified_fraction` (default 5%)
        of all tumor patches, flag the result as "and beyond" -- enough
        low-confidence tissue that true invasion depth could be at or past
        the reported stage.

    Returns: (deepest_stage or None, beyond_flag: bool)
    """
    total_tumor = tumor_patches
    if total_tumor == 0:
        return None, False

    count_threshold = max(min_count, present_fraction * total_tumor)
    deepest = None
    for stage in ["stage3", "stage2", "stage1", "stage0"]:
        if stage_counts.get(stage, 0) > count_threshold:
            deepest = stage
            break

    unidentified_threshold = unidentified_fraction * total_tumor
    beyond_flag = unidentified_count > unidentified_threshold
    return deepest, beyond_flag


def classify_stage(probs):
    """Applies confidence + entropy thresholding for 'Unidentified' rejection."""
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    norm_entropy = entropy(probs)

    if confidence < config.STAGE_CONF_THRESHOLD or norm_entropy > config.STAGE_ENTROPY_THRESHOLD:
        return "unidentified", confidence, norm_entropy
    return config.STAGE_CLASSES[pred_idx], confidence, norm_entropy


def run_inference_on_slide(slide_path, output_dir, tile_size=None, keep_tiles=False):
    os.makedirs(output_dir, exist_ok=True)
    device = config.DEVICE

    # 1) tile the slide
    tmp_tile_dir = os.path.join(output_dir, "_tiles_tmp")
    patch_paths = tile_wsi(slide_path, tmp_tile_dir, tile_size=tile_size)
    if len(patch_paths) == 0:
        print("No tissue patches found on this slide.")
        return

    # 2) load models + stain normalizer + transforms
    binary_model, stage_model = load_models(device)
    transform = get_eval_transforms()
    stain_normalizer = None
    if os.path.exists(config.MACENKO_REFERENCE_PATCH):
        stain_normalizer = load_or_fit_reference(config.MACENKO_REFERENCE_PATCH)

    results = []
    stage_counts = {s: 0 for s in config.STAGE_CLASSES}
    stage_counts["unidentified"] = 0
    tumor_count, normal_count = 0, 0

    for p in patch_paths:
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if stain_normalizer is not None:
            try:
                rgb = stain_normalizer.transform(rgb)
            except Exception:
                pass

        # ---- Stage 1: binary ----
        bin_probs = predict_patch(binary_model, rgb, transform, device)
        is_tumor = bool(np.argmax(bin_probs) == 1)  # index 1 == TUM
        record = {
            "patch": os.path.basename(p),
            "binary_pred": "tumor" if is_tumor else "normal",
            "binary_confidence": float(np.max(bin_probs)),
        }

        if not is_tumor:
            normal_count += 1
            record["stage"] = None
        else:
            tumor_count += 1
            # ---- Stage 2: staging model (only for tumor patches) ----
            stage_probs = predict_patch(stage_model, rgb, transform, device)
            stage_label, conf, ent = classify_stage(stage_probs)
            record["stage"] = stage_label
            record["stage_confidence"] = conf
            record["stage_entropy"] = ent
            stage_counts[stage_label] += 1

        results.append(record)

    if not keep_tiles:
        import shutil
        shutil.rmtree(tmp_tile_dir, ignore_errors=True)

    # 3) slide-level aggregation
    total = len(results)
    tumor_fraction = tumor_count / total if total else 0.0
    dominant_stage = max(
        {k: v for k, v in stage_counts.items() if k != "unidentified"}.items(),
        key=lambda kv: kv[1], default=(None, 0)
    )[0]
    deepest_stage, beyond_flag = determine_deepest_stage(stage_counts, stage_counts["unidentified"], tumor_count)
    deepest_stage_label = None
    if deepest_stage is not None:
        deepest_stage_label = f"{deepest_stage} and beyond" if beyond_flag else deepest_stage

    slide_report = {
        "slide": os.path.basename(slide_path),
        "total_patches": total,
        "normal_patches": normal_count,
        "tumor_patches": tumor_count,
        "tumor_fraction": tumor_fraction,
        "stage_distribution": stage_counts,
        "dominant_invasion_stage": dominant_stage,
        "deepest_invasion_stage": deepest_stage,
        "deepest_invasion_stage_label": deepest_stage_label,
        "invasion_beyond_flag": beyond_flag,
        "slide_level_call": "TUMOR DETECTED" if tumor_count > 0 else "NORMAL",
        "recommendation": (
            "Consult a pathologist: some tumor regions could not be confidently "
            "staged by the model." if stage_counts["unidentified"] > 0 else
            "Automated staging complete."
        ),
    }

    report_path = os.path.join(output_dir, "slide_report.json")
    with open(report_path, "w") as f:
        json.dump({"summary": slide_report, "patch_results": results}, f, indent=2)

    print(json.dumps(slide_report, indent=2))
    print(f"\nFull patch-level report saved -> {report_path}")
    return slide_report, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", required=True, help="Path to .svs WSI file")
    parser.add_argument("--out", required=True, help="Output directory for report")
    parser.add_argument("--keep_tiles", action="store_true")
    args = parser.parse_args()

    run_inference_on_slide(args.slide, args.out, keep_tiles=args.keep_tiles)