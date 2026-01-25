from .sam_wrapper import SAMWrapper
from .grounding_dino_wrapper import GroundingDINOWrapper
import cv2
from PIL import Image
import numpy as np


class GroundingSegment:
    def __init__(
        self,
        # Parameters for SAM
        use_sam_hq=True,
        sam_model=None,
        sam_checkpoint_path=None,
        sam_hq_checkpoint_path=None,
        # Parameters for GroundingDINO
        config_path=None,
        checkpoint_path=None,
        # Other parameters
        show_bbox=False,
        show_mask=False,
        device="cuda",
    ):
        self.sam = SAMWrapper(
            use_sam_hq=use_sam_hq,
            sam_model=sam_model,
            sam_checkpoint_path=sam_checkpoint_path,
            sam_hq_checkpoint_path=sam_hq_checkpoint_path,
            device=device,
        )
        self.grounding_dino = GroundingDINOWrapper(
            config_path=config_path, checkpoint_path=checkpoint_path, device=device
        )
        self.show_bbox = show_bbox
        self.show_mask = show_mask

    def plot_bounding_box(self, img, bbox, phrase, title='Bounding Box', show=True):
        img = img.copy()
        bbox = bbox.numpy().astype(int)
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2
        )
        cv2.putText(
            img,
            phrase,
            (bbox[0], bbox[3]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )
        if show:
            Image.fromarray(img).show(title)
        return img

    def plot_mask(self, img, mask, random_color=False, title='Mask', show=True):
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        if random_color:
            color = np.random.random(3)
        else:
            color = np.array([30/255, 144/255, 255/255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = (mask_image * 255).astype(np.uint8)
        img_vis = cv2.add(img, mask_image)
        if show:
            Image.fromarray(img_vis).show(title)
        return img_vis

    def run(self, img, text, only_max=True):
        boxes_filt, pred_phrases = self.grounding_dino.get_grounding_output(
            img, text, only_max=only_max
        )

        if boxes_filt.size(0) == 0:
            return None, None, None
        else:
            masks = self.sam.run(img, boxes=boxes_filt).squeeze(1)

            # debug
            for i in range(boxes_filt.shape[0]):
                if self.show_bbox:
                    self.plot_bounding_box(img, boxes_filt[i], pred_phrases[i])
                if self.show_mask:
                    self.plot_mask(img, masks[i])

            return boxes_filt.numpy(), pred_phrases, masks.cpu().numpy()
