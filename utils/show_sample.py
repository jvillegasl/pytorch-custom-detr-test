from torch import Tensor
import matplotlib.pyplot as plt
import random
from typing import Optional

from .bbox import get_bbox_text, rescale_bboxes, get_bbox_patch


def show_sample(xb: tuple[Tensor, Tensor], yb: tuple, classes: list[str], index: Optional[int] = None):
    images, masks = xb

    if index is None:
        batch_index = random.randint(0, images.size(0)-1)
    else:
        batch_index = index

    image = images[batch_index]
    mask = masks[batch_index]
    data = yb[batch_index]

    label = data['labels'].item()

    not_mask = ~mask
    H = not_mask[0].nonzero()[-1].item() + 1
    W = not_mask[:, 0].nonzero()[-1].item() + 1
    size = (H, W)

    bboxes = data['boxes']
    rescaled_bboxes = rescale_bboxes(bboxes, size)

    colors = []
    for _ in classes:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        colors.append(color)

    _, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))

    for bbox in rescaled_bboxes:
        patch = get_bbox_patch(bbox, colors[label])
        ax.add_patch(patch)

        text = get_bbox_text(bbox, classes[label])
        ax.add_artist(text)

    plt.show()
