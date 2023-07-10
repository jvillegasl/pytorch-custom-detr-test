import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def box_cxcywh_to_xyxy(x: Tensor):
    x_c, y_c, w, h = x.unbind(1)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox: Tensor, size):
    img_w, img_h = size

    b = box_cxcywh_to_xyxy(out_bbox)

    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    return b


def filter_output_by_threshold(outputs: dict[str, Tensor], size, threshold: float = 0.7):

    # keep only predictions with confidence above threshold
    probs = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probs.max(-1).values > threshold

    probs_to_keep = probs[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)

    return probs_to_keep, bboxes_scaled


def plot_results(pil_img, prob, boxes, classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)

            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=color, linewidth=3))
            cl = p.argmax()
            text = f'{classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
