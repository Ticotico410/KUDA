import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import argparse
import cv2
import numpy as np
from PIL import Image
import openai

from perception.predictor import GroundingSegmentPredictor
from planner.planner import KUDAPlanner, get_annotated_image, parse
from utils import get_config_real, farthest_point_sampling, fps_rad_idx

openai.api_key = None

def build_annotated_points(img, material):
    predictor = GroundingSegmentPredictor(show_bbox=False, show_mask=True)
    masks = predictor.mask_generation(img)
    masks.pop()

    annotated_points = []
    num_per_obj = 8
    radius = 200
    if material == 'cube':
        for mask in masks:
            positions = np.argwhere(mask)
            positions = positions[:, [1, 0]]
            keypoint = positions.mean(axis=0)
            annotated_points.append(keypoint)
    else:
        for mask in masks:
            positions = np.argwhere(mask)
            positions = positions[:, [1, 0]]
            fps_1 = farthest_point_sampling(positions, num_per_obj)
            fps_2, _ = fps_rad_idx(fps_1, radius)
            annotated_points.extend(fps_2)
            center = positions.mean(axis=0)
            annotated_points.append(center)

    global_radius = 150
    annotated_points, _ = fps_rad_idx(np.array(annotated_points), global_radius)
    return annotated_points


def visualize_target_spec(image, annotated_points, targets, pixel_per_cm):
    vis = image.copy()
    h, w = vis.shape[:2]
    center = np.array([w // 2, h // 2], dtype=np.float32)

    for target_index, (reference_index, array) in targets.items():
        if reference_index == -1:
            reference_point = center
        else:
            reference_point = annotated_points[reference_index]

        # Use a simple 2D visualization: x -> right, y -> up (invert for image y).
        delta = np.array([array[0], -array[1]], dtype=np.float32) * pixel_per_cm
        destination = reference_point + delta

        start = annotated_points[target_index]
        end = destination
        cv2.arrowedLine(
            vis,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            (0, 0, 255),
            2
        )
        cv2.putText(
            vis,
            f'P{target_index + 1}',
            (int(start[0]) + 3, int(start[1]) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='prompts/random_0.jpg')
    parser.add_argument('--instruction', default='divide the chessmen into two groups based on their color')
    parser.add_argument('--material', default='cube')
    parser.add_argument('--pixel_per_cm', type=float, default=1.0)
    parser.add_argument('--output_annotated', default='annotated_image.png')
    parser.add_argument('--output_targets', default='targets_2d.png')
    args = parser.parse_args()

    config = get_config_real('configs/real_config.yaml')
    planner = KUDAPlanner(env=None, config=config.planner)

    img = Image.open(args.image)
    img = np.array(img)[:, :, ::-1]

    annotated_points = build_annotated_points(img, args.material)
    annotated_image = get_annotated_image(img, annotated_points, debug=False, mask=None)
    Image.fromarray(annotated_image[..., ::-1]).save(args.output_annotated)

    messages = planner._build_prompt(annotated_image, args.instruction)
    ret = openai.ChatCompletion.create(
        messages=messages,
        temperature=planner.config['temperature'],
        model=planner.config['model'],
        max_tokens=planner.config['max_tokens']
    )['choices'][0]['message']['content']
    print(ret)

    targets = parse(ret)
    targets_vis = visualize_target_spec(annotated_image, annotated_points, targets, args.pixel_per_cm)
    Image.fromarray(targets_vis[..., ::-1]).save(args.output_targets)


if __name__ == '__main__':
    main()
