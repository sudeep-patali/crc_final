"""
inference_runner.py
Same logic as scripts/infer_wsi.py, but reports live progress into a shared
dict so the frontend can show real-time patch-by-patch status instead of a
fake spinner.
"""

import os
import sys
import shutil
import numpy as np
import cv2
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from utils.wsi_tiling import tile_wsi
from utils.stain_normalization import load_or_fit_reference
from utils.dataset import get_eval_transforms
from models.hybrid_model import build_binary_model, build_stage_model

_MODEL_CACHE = {}


def get_models():
    """Load models once and cache them across requests."""
    if "binary" not in _MODEL_CACHE:
        device = config.DEVICE
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

        _MODEL_CACHE["binary"] = binary_model
        _MODEL_CACHE["stage"] = stage_model
        _MODEL_CACHE["device"] = device
    return _MODEL_CACHE["binary"], _MODEL_CACHE["stage"], _MODEL_CACHE["device"]


def _predict(model, img_rgb, transform, device):
    aug = transform(image=img_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def _entropy(probs):
    probs = np.clip(probs, 1e-9, 1.0)
    return -np.sum(probs * np.log(probs)) / np.log(len(probs))


def _determine_deepest_stage(stage_counts, unidentified_count, tumor_patches,
                              present_fraction=None, unidentified_fraction=None, min_count=None):
    """
    Percentage-based deepest-invasion staging (scales consistently across
    slides of different sizes, unlike a fixed patch count):
      - A stage counts as "present" only if it exceeds `present_fraction`
        of tumor patches, with a small absolute floor (`min_count`) for
        very small tumor counts.
      - Among present stages, report the DEEPEST one (stage3 > stage2 >
        stage1 > stage0) -- matches real pT-staging logic.
      - If unidentified patches exceed `unidentified_fraction` of tumor
        patches, flag as "and beyond".
    Defaults are pulled from config.py (DEEPEST_STAGE_*) so this stays in
    sync with scripts/infer_wsi.py rather than drifting between two
    separately hardcoded copies.
    Returns: (deepest_stage or None, beyond_flag: bool)
    """
    present_fraction = config.DEEPEST_STAGE_PRESENT_FRACTION if present_fraction is None else present_fraction
    unidentified_fraction = config.DEEPEST_STAGE_UNIDENTIFIED_FRACTION if unidentified_fraction is None else unidentified_fraction
    min_count = config.DEEPEST_STAGE_MIN_COUNT if min_count is None else min_count

    if tumor_patches == 0:
        return None, False

    count_threshold = max(min_count, present_fraction * tumor_patches)
    deepest = None
    for stage in ["stage3", "stage2", "stage1", "stage0"]:
        if stage_counts.get(stage, 0) > count_threshold:
            deepest = stage
            break

    beyond_flag = unidentified_count > (unidentified_fraction * tumor_patches)
    return deepest, beyond_flag


def _classify_stage(probs):
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    norm_entropy = _entropy(probs)
    if confidence < config.STAGE_CONF_THRESHOLD or norm_entropy > config.STAGE_ENTROPY_THRESHOLD:
        return "unidentified", confidence, norm_entropy
    return config.STAGE_CLASSES[pred_idx], confidence, norm_entropy


def run_job(job_id, slide_path, jobs_store):
    """
    Runs the full pipeline for one uploaded slide, updating jobs_store[job_id]
    in place as it progresses so the frontend can poll for live status.
    """
    state = jobs_store[job_id]
    try:
        state["phase"] = "tiling"
        output_dir = os.path.join(PROJECT_ROOT, "webapp", "jobs", job_id)
        os.makedirs(output_dir, exist_ok=True)
        tmp_tile_dir = os.path.join(output_dir, "_tiles_tmp")

        patch_paths = tile_wsi(slide_path, tmp_tile_dir)
        if len(patch_paths) == 0:
            state["phase"] = "error"
            state["error"] = "No tissue patches detected on this slide."
            return

        state["total_patches"] = len(patch_paths)
        state["phase"] = "classifying"

        binary_model, stage_model, device = get_models()
        transform = get_eval_transforms()
        stain_normalizer = None
        if os.path.exists(config.MACENKO_REFERENCE_PATCH):
            stain_normalizer = load_or_fit_reference(config.MACENKO_REFERENCE_PATCH)

        stage_counts = {s: 0 for s in config.STAGE_CLASSES}
        stage_counts["unidentified"] = 0
        tumor_count, normal_count = 0, 0

        for i, p in enumerate(patch_paths):
            bgr = cv2.imread(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if stain_normalizer is not None:
                try:
                    rgb = stain_normalizer.transform(rgb)
                except Exception:
                    pass

            bin_probs = _predict(binary_model, rgb, transform, device)
            is_tumor = bool(np.argmax(bin_probs) == 1)

            if not is_tumor:
                normal_count += 1
            else:
                tumor_count += 1
                stage_probs = _predict(stage_model, rgb, transform, device)
                stage_label, _, _ = _classify_stage(stage_probs)
                stage_counts[stage_label] += 1

            state["processed"] = i + 1
            state["normal_count"] = normal_count
            state["tumor_count"] = tumor_count
            state["stage_counts"] = dict(stage_counts)

        shutil.rmtree(tmp_tile_dir, ignore_errors=True)

        total = len(patch_paths)
        tumor_fraction = tumor_count / total if total else 0.0
        non_unident = {k: v for k, v in stage_counts.items() if k != "unidentified"}
        dominant_stage = max(non_unident.items(), key=lambda kv: kv[1], default=(None, 0))[0]
        deepest_stage, beyond_flag = _determine_deepest_stage(stage_counts, stage_counts["unidentified"], tumor_count)
        deepest_stage_label = None
        if deepest_stage is not None:
            deepest_stage_label = f"{deepest_stage} and beyond" if beyond_flag else deepest_stage

        recommendation = (
            "Tumor tissue detected with confident staging across most regions."
            if stage_counts["unidentified"] == 0 else
            "Consult a pathologist: some tumor regions could not be confidently staged."
        ) if tumor_count > 0 else "No tumor tissue detected in this slide."

        state["result"] = {
            "slide_name": os.path.basename(slide_path),
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
            "recommendation": recommendation,
        }
        state["phase"] = "done"

    except Exception as e:
        state["phase"] = "error"
        state["error"] = str(e)
