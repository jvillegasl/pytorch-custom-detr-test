import torch

from model.loss import HungarianMatcher, SetCriterion


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def get_losses(output, target):
    with torch.no_grad():
        num_classes = output['pred_logits'].size(-1) - 1

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher
        )

        losses = criterion(output, target)

    return losses


def loss_labels(output, target):
    losses = get_losses(output, target)

    return losses['loss_ce']


def loss_cardinality(output, target):
    losses = get_losses(output, target)

    return losses['cardinality_error']


def loss_bbox(output, target):
    losses = get_losses(output, target)

    return losses['loss_bbox'].item()


def loss_giou(output, target):
    losses = get_losses(output, target)

    return losses['loss_giou'].item()
