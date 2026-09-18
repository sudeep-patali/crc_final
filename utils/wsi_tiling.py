"""
wsi_tiling.py
Tiles a whole-slide image (TCGA-COAD / TCGA-READ .svs files) into patches
suitable for the binary + staging models.

Requires: openslide-python, opencv-python
    pip install openslide-python --break-system-packages
    (also needs the openslide system library: apt-get install openslide-tools)
"""

import os
import numpy as np
import cv2

try:
    import openslide
except ImportError:
    openslide = None  # allow import of this module without openslide installed

import config


def get_tissue_mask(rgb_tile, sat_thresh=20):
    """
    Simple Otsu-based tissue detector on the saturation channel.
    Returns fraction of pixels considered tissue (0-1).
    """
    hsv = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_fraction = np.count_nonzero(mask) / mask.size
    return tissue_fraction, mask


def _select_level_for_magnification(slide, target_mag=20):
    """Pick the pyramid level whose magnification is closest to target_mag."""
    try:
        base_mag = float(slide.properties.get("openslide.objective-power", 40))
    except Exception:
        base_mag = 40.0

    best_level, best_diff = 0, float("inf")
    for lvl in range(slide.level_count):
        downsample = slide.level_downsamples[lvl]
        mag_at_level = base_mag / downsample
        diff = abs(mag_at_level - target_mag)
        if diff < best_diff:
            best_diff, best_level = diff, lvl
    return best_level


def tile_wsi(slide_path, output_dir, tile_size=None, stride=None,
             target_mag=None, tissue_threshold=None):
    """
    Tiles a WSI into non-overlapping (or strided) patches, discarding
    background/blank tiles, and saves them as PNGs to output_dir.

    Returns: list of saved patch file paths.
    """
    if openslide is None:
        raise ImportError("openslide-python is required for WSI tiling. "
                           "Install with: pip install openslide-python --break-system-packages")

    tile_size = tile_size or config.TILE_SIZE_WSI
    stride = stride or config.TILE_STRIDE_WSI
    target_mag = target_mag or config.WSI_LEVEL_MAG
    tissue_threshold = tissue_threshold or config.TISSUE_THRESHOLD

    slide = openslide.OpenSlide(slide_path)
    level = _select_level_for_magnification(slide, target_mag)
    level_w, level_h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    slide_out_dir = os.path.join(output_dir, slide_id)
    os.makedirs(slide_out_dir, exist_ok=True)

    saved_paths = []
    for y in range(0, level_h - tile_size, stride):
        for x in range(0, level_w - tile_size, stride):
            # coordinates must be given in level-0 reference frame for read_region
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            tile = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
            tile_np = np.array(tile)

            tissue_frac, _ = get_tissue_mask(tile_np)
            if tissue_frac < tissue_threshold:
                continue  # skip background/blank tile

            fname = f"{slide_id}_x{x}_y{y}.png"
            fpath = os.path.join(slide_out_dir, fname)
            cv2.imwrite(fpath, cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR))
            saved_paths.append(fpath)

    slide.close()
    print(f"[{slide_id}] tiled: {len(saved_paths)} tissue patches saved -> {slide_out_dir}")
    return saved_paths


def tile_all_wsis(wsi_dir=None, output_dir=None):
    """Batch-tile every .svs/.tif WSI found in wsi_dir."""
    wsi_dir = wsi_dir or config.WSI_DIR
    output_dir = output_dir or config.WSI_PATCH_DIR
    exts = (".svs", ".tif", ".tiff", ".ndpi")

    all_paths = {}
    for fname in os.listdir(wsi_dir):
        if fname.lower().endswith(exts):
            slide_path = os.path.join(wsi_dir, fname)
            paths = tile_wsi(slide_path, output_dir)
            all_paths[fname] = paths
    return all_paths


if __name__ == "__main__":
    tile_all_wsis()
