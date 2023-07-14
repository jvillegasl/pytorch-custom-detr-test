import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from model.matcher import HungarianMatcher
from utils.bbox import box_cxcywh_to_xyxy


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bipartite_matching_loss(output, target):
    num_classes = output['pred_logits'].size(-1) - 1

    weight_dict = {
        'loss_ce': 1.,
        'loss_bbox': 5.,
        'loss_giou': 2.,
    }

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher
    )

    losses = criterion(output, target)
    loss = sum([weight_dict[k] * losses[k] for k in weight_dict])

    return loss


class SetCriterion(nn.Module):
    def __init__(
            self,
            num_classes,
            matcher,
            eos_coef: float = 0.1
    ):
        """
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """

        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(
            self,
            outputs: dict[str, torch.Tensor],
            targets: list[dict[str, torch.Tensor]],
            indices: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        """Classification loss (NLL)

        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_boxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "boxes": Tensor, shape `[num_objects_i, 4]`

            indices: List[Tuple[Tensor, Tensor]] where `len(indices) == batch_size` and:
               - indices[i][0]: Indices of the selected predictions (in order)
               - indices[i][1]: Indices of the corresponding selected targets (in order)

            For each batch element, it holds:
                indices[i][0].shape == indices[i][1].shape = min(num_queries, num_objects_i)

        Returns:
            losses: Dict containing:
               - "loss_ce": Tensor, shape `[]`
        """

        pred_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        # Tuple[Tensor[total_num_objects], Tensor[total_num_objects]]
        # Tuple[batch_coords, pred_coords]

        zipped_tgt_idx = zip(targets, indices)
        # Zip[Tuple[target[i], indices[i]]]

        target_classes_o = torch.cat([
            t['labels'][J]
            for t, (_, J) in zipped_tgt_idx
        ])
        # ordered target classes
        # [total_num_objects]

        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        # filled with no-object class
        # [batch_size, num_queries]

        target_classes[idx] = target_classes_o
        # idx indicates the coords (batch, query)
        # to be replaced with the corresponding target class

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight.to(target_classes.device)
        )

        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(
            self,
            outputs: dict[str, torch.Tensor],
            targets: list[dict[str, torch.Tensor]]
    ):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes.
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients

        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_boxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "boxes": Tensor, shape `[num_objects_i, 4]`

        Returns:
            losses: Dict containing:
               - "cardinality_error": Tensor, shape `[]`
        """

        pred_logits = outputs['pred_logits']

        tgt_lengths = torch.as_tensor(
            [len(v['labels']) for v in targets], device=pred_logits.device)
        # [batch_size]

        # Count the number of predictions that are NOT 'no-object' (which is the last class)
        pred_labels = pred_logits.argmax(-1)
        # [batch_size, num_queries]

        no_object_label = pred_logits.size(-1) - 1

        card_pred = (pred_labels != no_object_label).sum(1)
        # [batch_size]
        # count of NOT 'no-object' predictions

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # []

        losses = {'cardinality_error': card_err}

        return losses

    def loss_bboxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_bboxes
    ):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss

        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_boxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "boxes": Tensor, shape `[num_objects_i, 4]`

            indices: List[Tuple[Tensor, Tensor]] where `len(indices) == batch_size` and:
               - indices[i][0]: Indices of the selected predictions (in order)
               - indices[i][1]: Indices of the corresponding selected targets (in order)

            For each batch element, it holds:
                indices[i][0].shape == indices[i][1].shape = min(num_queries, num_objects_i)

            num_bboxes: Int.

        Returns:
            losses: Dict containing:
               - "loss_bbox": Tensor, shape `[]`
               - "loss_giou": Tensor, shape `[]`
        """

        idx = self._get_src_permutation_idx(indices)
        # Tuple[Tensor[total_num_objects], Tensor[total_num_objects]]
        # Tuple[batch_coords, pred_coords]

        pred_boxes = outputs['pred_boxes'][idx]
        # predicted bboxes that best matches target bboxes (in order)
        # [total_num_objects, 4]

        tgt_bboxes = torch.cat([
            t['boxes'][i]  # ordered bboxes, [num_objects_i]
            for t, (_, i) in zip(targets, indices)
        ], dim=0)
        # [total_num_objects, 4]

        loss_bbox = F.l1_loss(pred_boxes, tgt_bboxes, reduction='none')

        losses: dict[str, torch.Tensor] = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_bboxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_bboxes),
        ))

        losses['loss_giou'] = loss_giou.sum() / num_bboxes

        return losses

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ):
        """
        Arguments:
            outputs: Dict containing:
               - "pred_logits": Tensor, shape `[batch_size, num_queries, num_classes]`
               - "pred_boxes": Tensor, shape `[batch_size, num_queries, 4]`

            targets: List[dict], `len(targets) == batch_size`, each dict contains:
               - "labels": Tensor, shape `[num_objects_i]`
               - "boxes": Tensor, shape `[num_objects_i, 4]`
        """

        indices = self.matcher(outputs, targets)

        num_bboxes = sum(len(t["labels"]) for t in targets)
        num_bboxes = torch.as_tensor(
            [num_bboxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_cardinality(outputs, targets))
        losses.update(self.loss_bboxes(outputs, targets, indices, num_bboxes))

        return losses

    def _get_src_permutation_idx(self, indices: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Arguments:
            indices: List[Tuple[Tensor, Tensor]] where `len(indices) == batch_size` and:
               - indices[i][0]: Indices of the selected predictions (in order)
               - indices[i][1]: Indices of the corresponding selected targets (in order)

            For each batch element, it holds:
                indices[i][0].shape == indices[i][1].shape = min(num_queries, num_objects_i)
        """

        batch_idx = torch.cat([
            torch.full_like(src, i)
            for i, (src, _) in enumerate(indices)
        ])
        # batch coordinates
        # [total_num_objects]

        src_idx = torch.cat([src for (src, _) in indices])
        # prediction coordinates
        # [total_num_objects]

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[torch.Tensor, torch.Tensor]]):
        batch_idx = torch.cat([
            torch.full_like(tgt, i)
            for i, (_, tgt) in enumerate(indices)
        ])

        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx
