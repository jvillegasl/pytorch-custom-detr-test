import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet50(
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
            replace_stride_with_dilation=[False, False, True]
        )

        self.body = IntermediateLayerGetter(
            backbone, return_layers={'layer4': '0'})

        self.num_channels = 2048

    def forward(self, tensor_list: tuple[torch.Tensor, torch.Tensor]):
        tensors, m = tensor_list
        assert m is not None

        x = self.body(tensors)
        x = list(x.items())[-1][1]

        mask = F.interpolate(
            m[None].float(),
            size=x.shape[-2:]
        ).to(torch.bool)[0]
        # m[None] equivalent to unsqueeze(0)

        return x, mask
