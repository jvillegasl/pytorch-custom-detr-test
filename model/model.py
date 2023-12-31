import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import MLP, PositionEmbeddingSine
from model.transformer import Transformer
from model.backbone import Backbone
from utils.misc import nested_tensor_from_tensor_list


class DETR(BaseModel):
    hidden_dim: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4

    def __init__(self, num_classes, num_queries):
        super().__init__()

        self.backbone = Backbone(name='resnet18')
        self.conv = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, 1)
        self.position_embedding = PositionEmbeddingSine(
            self.hidden_dim//2,
            normalize=True
        )

        self.transformer = Transformer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=2048,
            normalize_before=True,
            return_intermediate_dec=True
        )

        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

    def forward(self, input: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | torch.Tensor):
        if not isinstance(input, tuple):
            input = nested_tensor_from_tensor_list(input)

        features = self.backbone(input)

        src, mask = features
        assert mask is not None

        pos = self.position_embedding(features).to(src.dtype)

        src = self.conv(src)
        hs = self.transformer(src, mask, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        return out
