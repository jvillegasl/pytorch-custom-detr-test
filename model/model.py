import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import MLP, Backbone, PositionEmbeddingSine
from model.transformer import Transformer
from utils.misc import NestedTensor, nested_tensor_from_tensor_list


class DETR(BaseModel):
    hidden_dim: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    max_num_objects: int = 64

    def __init__(self, num_classes, num_queries):
        super().__init__()

        self.backbone = Backbone()
        self.conv = nn.Conv2d(512, self.hidden_dim, 1)
        self.position_embedding = PositionEmbeddingSine(
            self.hidden_dim//2, normalize=True
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

    def forward(self, input: NestedTensor | list[torch.Tensor] | torch.Tensor):
        if isinstance(input, (list, torch.Tensor)):
            input = nested_tensor_from_tensor_list(input)

        features: NestedTensor = self.backbone(input)

        src, mask = features.decompose()
        assert mask is not None
        src = self.conv(src)

        pos = self.position_embedding(features).to(features.tensors.dtype)

        hs = self.transformer(src, mask, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        return out
