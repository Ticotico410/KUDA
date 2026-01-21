import warnings
import torchvision

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
if hasattr(torchvision, "disable_beta_transforms_warning"):
    torchvision.disable_beta_transforms_warning()

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from perception.models.sam_wrapper import SAMWrapper
from perception.models.grounding_segment import GroundingSegment
from perception.predictor import GroundingSegmentPredictor
from utils import farthest_point_sampling, fps_rad_idx
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# Grounding Segment settings
USE_GROUNDING = False


# SAM model settings
USE_SAM = "vit_b"  # vit_h / vit_b
if USE_SAM == "vit_b":
    SAM_CKPT = os.path.join(REPO_ROOT, "perception/models/checkpoints/sam_vit_b_01ec64.pth")
elif USE_SAM == "vit_h":
    SAM_CKPT = os.path.join(REPO_ROOT, "perception/models/checkpoints/sam_vit_h_4b8939.pth")

USE_SAM_HQ = False
SAM_HQ_CKPT = os.path.join(REPO_ROOT, "perception/models/checkpoints/sam_hq_vit_h.pth")


# KUDA settings
MATERIAL = "cube"  # "cube" or others
MAX_MASKS = 0  # 0 = no limit
NUM_PER_OBJ = 8
RADIUS = 200
GLOBAL_RADIUS = 100


# Path settings
IMAGE_PATH = os.path.join(REPO_ROOT, "prompts/random_0.jpg")
OUT_PATH = os.path.join(REPO_ROOT, "scripts/results/segmentation.png")
SAVE_MASKS_DIR = os.path.join(REPO_ROOT, "scripts/results/masks")
SAVE_CONTOURS_PATH = os.path.join(REPO_ROOT, "scripts/results/contours.png")
SAVE_ANNOTATED_POINTS_PATH = os.path.join(REPO_ROOT, "scripts/results/annotated_points.png")
SAVE_ANNOTATED_IMAGE_PATH = os.path.join(REPO_ROOT, "scripts/results/annotated_image.png")
GROUNDING_DINO_CONFIG = os.path.join(REPO_ROOT, "perception/models/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CKPT = os.path.join(REPO_ROOT, "perception/models/checkpoints/groundingdino_swint_ogc.pth")


def _overlay_masks(image_bgr, masks, alpha=0.5):
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]
    for mask in masks:
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
        overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def _save_single_masks(image_bgr, masks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    h, w = image_bgr.shape[:2]
    for idx, mask in enumerate(masks):
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        # save binary mask
        mask_u8 = (mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(out_dir, f"mask_{idx:03d}.png"), mask_u8)
        # save overlay for this mask
        overlay = image_bgr.copy()
        overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"mask_{idx:03d}_overlay.png"), overlay)


def _draw_contours(image_bgr, masks, color=(0, 255, 0), thickness=2):
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for mask in masks:
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _draw_keypoints(image_bgr, points, color=(0, 0, 255), radius=4, thickness=-1):
    out = image_bgr.copy()
    for p in points:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (x, y), radius, color, thickness)
    return out


def _draw_annotated_image(image, points, debug=False, mask=None):
    image = deepcopy(image)
    if debug:
        debug_dir = os.path.dirname(SAVE_ANNOTATED_IMAGE_PATH) or "."
        os.makedirs(debug_dir, exist_ok=True)
        save_image = Image.fromarray(image[..., ::-1])
        save_image.save(os.path.join(debug_dir, "annotated_input.png"))

    # draw points
    for i, point in enumerate(points):
        x, y = point
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
        # cv2.circle(image, (int(x), int(y)), 4, (240, 32, 160), -1) # purple
        cv2.putText(image, f'P{i+1}', (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # add a center point for background reference
    H, W = image.shape[:2]
    x_center = W // 2
    y_center = H // 2
    cv2.circle(image, (x_center, y_center), 4, (0, 255, 0), -1)
    cv2.putText(image, 'C', (x_center + 3, y_center - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if debug:
        save_image = Image.fromarray(image[..., ::-1])
        save_image.show()
        save_image.save(os.path.join(debug_dir, "annotated_output.png"))

    return image


def main():
    image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    image_rgb = image_bgr[..., ::-1]

    if USE_GROUNDING:
        predictor = GroundingSegmentPredictor(
            show_bbox=False,
            show_mask=True,
            use_sam_hq=USE_SAM_HQ,
            sam_checkpoint_path=SAM_CKPT,
            sam_hq_checkpoint_path=SAM_HQ_CKPT,
            config_path=GROUNDING_DINO_CONFIG,
            checkpoint_path=GROUNDING_DINO_CKPT,
            device="cuda",
        )
        masks = predictor.mask_generation(image_rgb)

    else:
        if USE_SAM_HQ:
            sam = SAMWrapper(
                use_sam_hq=True,
                sam_checkpoint_path=SAM_CKPT,
                sam_hq_checkpoint_path=SAM_HQ_CKPT,
                device="cuda",
            )
            masks = sam.run(image_rgb, automatic_mask_flag=True)
        else:
            if USE_SAM not in sam_model_registry:
                raise ValueError(f"Unsupported model type: {USE_SAM}")
            model = sam_model_registry[USE_SAM](checkpoint=SAM_CKPT)
            model.to("cuda")
            mask_generator = SamAutomaticMaskGenerator(
                model=model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            masks = mask_generator.generate(image_rgb)

        masks.sort(key=lambda x: x['area'])
        masks = [mask['segmentation'] for mask in masks]

    # remove the background mask
    masks.pop()

    annotated_points = []
    if MATERIAL == "cube":
        for mask in masks:
            positions = np.argwhere(mask)
            if positions.size == 0:
                continue
            positions = positions[:, [1, 0]]
            keypoint = positions.mean(axis=0)
            annotated_points.append(keypoint)
    else:
        for mask in masks:
            positions = np.argwhere(mask)
            if positions.size == 0:
                continue
            positions = positions[:, [1, 0]]
            fps_1 = farthest_point_sampling(positions, NUM_PER_OBJ)
            fps_2, _ = fps_rad_idx(fps_1, RADIUS)
            annotated_points.extend(fps_2)
            center = positions.mean(axis=0)
            annotated_points.append(center)
    print("-"*50)
    print(f"Number of masks: {len(masks)}")
    print(f"Number of keypoints before downsampling: {len(annotated_points)}")
    if len(annotated_points) > 0:
        annotated_points, _ = fps_rad_idx(np.array(annotated_points), GLOBAL_RADIUS)
    print(f"Number of keypoints after downsampling: {len(annotated_points)}")
    print("-"*50 + "\n")

    segment = _overlay_masks(image_bgr, masks, alpha=0.5)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    cv2.imwrite(OUT_PATH, segment)
    print(f"Saved segment to: {OUT_PATH}")

    if SAVE_MASKS_DIR:
        _save_single_masks(image_bgr, masks, SAVE_MASKS_DIR)
        print(f"Saved individual masks to: {SAVE_MASKS_DIR}")

    if SAVE_CONTOURS_PATH:
        contour = _draw_contours(image_bgr, masks)
        os.makedirs(os.path.dirname(SAVE_CONTOURS_PATH) or ".", exist_ok=True)
        cv2.imwrite(SAVE_CONTOURS_PATH, contour)
        print(f"Saved contour to: {SAVE_CONTOURS_PATH}")

    if SAVE_ANNOTATED_POINTS_PATH:
        keypoint = _draw_keypoints(image_bgr, annotated_points)
        os.makedirs(os.path.dirname(SAVE_ANNOTATED_POINTS_PATH) or ".", exist_ok=True)
        cv2.imwrite(SAVE_ANNOTATED_POINTS_PATH, keypoint)
        print(f"Saved keypoints to: {SAVE_ANNOTATED_POINTS_PATH}")

    if SAVE_ANNOTATED_IMAGE_PATH:
        annotated_image = _draw_annotated_image(image_bgr, annotated_points)
        os.makedirs(os.path.dirname(SAVE_ANNOTATED_IMAGE_PATH) or ".", exist_ok=True)
        cv2.imwrite(SAVE_ANNOTATED_IMAGE_PATH, annotated_image)
        print(f"Saved annotated image to: {SAVE_ANNOTATED_IMAGE_PATH}")


if __name__ == "__main__":
    main()

