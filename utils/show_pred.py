from torch import Tensor
from typing import Optional
import random
import matplotlib.pyplot as plt

from utils import filter_output_by_threshold, get_bbox_patch, get_bbox_text


def show_pred(x: tuple[Tensor, Tensor], y: dict[str, Tensor], classes: list[str], index: Optional[int] = None, threshold: float = 0.7):
    if index is None:
        batch_index = random.randint(0, x[0].size(0)-1)
    else:
        batch_index = index

    images, masks = x

    image = images[batch_index]
    mask = masks[batch_index]

    not_mask = ~mask
    H = not_mask[0].nonzero()[-1].item() + 1
    W = not_mask[:, 0].nonzero()[-1].item() + 1
    size = (H, W)

    probs, bboxes = filter_output_by_threshold(y, size, threshold)

    colors = []
    for _ in classes:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        colors.append(color)

    _, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))

    for prob, bbox in zip(probs, bboxes):
        label = prob.argmax()
        patch = get_bbox_patch(bbox, colors[label])
        ax.add_patch(patch)

        text = get_bbox_text(bbox, classes[label], prob.item())
        ax.add_artist(text)
