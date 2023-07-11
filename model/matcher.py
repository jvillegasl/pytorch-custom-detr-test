import torch
from torch import nn
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

from utils.bbox import box_cxcywh_to_xyxy

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(
            self,
            outputs: dict[str, torch.Tensor],
            targets: list[dict[str, torch.Tensor]]
    ):
        """
        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_boxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "bboxes": Tensor, shape `[num_objects_i, 4]`

        Returns:
            indices: List[Tuple[Tensor, Tensor]] where `len(indices) == batch_size` and:
               - indices[i][0]: Indices of the selected predictions (in order)
               - indices[i][1]: Indices of the corresponding selected targets (in order)

            For each batch element, it holds:
                indices[i][0].shape == indices[i][1].shape = min(num_queries, num_objects_i)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # [batch_size * num_queries, 4]

        tgt_ids = torch.cat([v['labels'] for v in targets])
        # [total_num_objects] where sum(num_objects_i)
        tgt_bbox = torch.cat([v['bboxes'] for v in targets])
        # [total_num_objects, 4]

        # The costs are calculated for all the target objects
        # and then are reorganized and sliced to match the current objects for each batch item
        cost_class = -out_prob[:, tgt_ids]
        # [batch_size * num_queries, total_num_objects]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # [batch_size * num_queries, total_num_objects]

        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        # [batch_size * num_queries, total_num_objects]

        C = self.cost_bbox * cost_bbox + self.cost_class * \
            cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        # [batch_size, num_queries, total_num_objects]

        sizes = [len(v['bboxes']) for v in targets]
        # sizes[i] == num_object_i
        # sum(sizes) == total_num_objects
        # len(sizes) == batch_size

        C_chunks = C.split(sizes, dim=-1)
        # List[Tensor]
        # where C_chunks[i]: Tensor, shape [batch_size, num_queries, sizes[i]]
        # and len(C_chunks) == len(sizes)

        # linear_sum_assignment(C_chunks[0][0])

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_chunks)]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        # List[Tuple[Tensor, Tensor]]
        # where len(indices) == len(C_chunks)
        # and indices[i][0]: Tensor, shape [sizes[i]]
        # and indices[i][1]: Tensor, shape [sizes[i]]

        return indices
