from PIL import Image
import numpy as np
import cv2
from .models.grounding_segment import GroundingSegment

class GroundingSegmentPredictor:
    def __init__(
        self,
        show_bbox=False,
        show_mask=False,
        use_sam_hq=True,
        sam_model=None,
        sam_checkpoint_path=None,
        sam_hq_checkpoint_path=None,
        config_path=None,
        checkpoint_path=None,
        device="cuda",
    ):
        self.segment = GroundingSegment(
            show_bbox=show_bbox,
            show_mask=show_mask,
            use_sam_hq=use_sam_hq,
            sam_model=sam_model,
            sam_checkpoint_path=sam_checkpoint_path,
            sam_hq_checkpoint_path=sam_hq_checkpoint_path,
            config_path=config_path or "perception/models/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path=checkpoint_path or "perception/models/checkpoints/groundingdino_swint_ogc.pth",
            device=device,
        )

    def predict(self, images, text):
        final_masks = []
        for image in images:
            if type(image) == Image.Image:
                image = np.array(image)
            # convert bgr array to rgb
            elif isinstance(image, np.ndarray) or isinstance(image, list):
                image = np.array(image)[..., ::-1]
            boxes, phrases, masks = self.segment.run(image, text)
            final_masks.append(masks)
        # shape (batch_size, num_boxes, *mask_shape)
        return final_masks

    def mask_generation(self, image, debug=False):
        if type(image) == Image.Image:
            image = np.array(image)
        # convert bgr array to rgb
        elif isinstance(image, np.ndarray) or isinstance(image, list):
            image = np.array(image)[..., ::-1]
        masks = self.segment.sam.run(image, automatic_mask_flag=True)
        # sort masks by area
        masks.sort(key=lambda x: x['area'])
        masks = [mask['segmentation'] for mask in masks]

        if debug:
            for mask in masks:
                self.plot_mask(image, mask, random_color=True)

        return masks

    def plot_mask(self, img, mask, random_color=False, title='Mask'):
        # mask = mask
        if random_color:
            color = np.random.random(3)
        else:
            color = np.array([30/255, 144/255, 255/255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = (mask_image * 255).astype(np.uint8)
        img_vis = cv2.add(img, mask_image)
        img_vis = Image.fromarray(img_vis)
        img_vis.show(title)