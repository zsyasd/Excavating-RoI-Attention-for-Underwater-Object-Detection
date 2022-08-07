import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
# from extra.MMD_AAE import *
# from extra.CIDDG import *
import itertools
import numpy as np
from .transformer import *
from mmdet.models.backbones.resnet import Bottleneck
from mmcv.cnn import ConvModule, normal_init, xavier_init
from ..tr_head import MLPMixer

@HEADS.register_module()
class TrBBoxHead(BBoxHead):

    def __init__(self, num_classes, inchanel=256, d_model=256, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 *args,
                 **kwargs):
        super(TrBBoxHead, self).__init__(*args, **kwargs)
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
                                     activation=activation, normalize_before=normalize_before,
                                     return_intermediate_dec=return_intermediate_dec)
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=d_model//2)
        self.num_classes = num_classes
        self.input_proj = nn.Conv2d(inchanel,d_model,1)
        self.query_embed = nn.Embedding(num_classes, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.fc_cls = nn.Linear(7*7*256, num_classes + 1)
        # self.fc_reg = nn.Linear(7*7*256, 4 * num_classes)
        self.fc_reg = nn.Linear(256, 4 * num_classes)

        # learning positional encoding
        self.window_size = (7,7)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 *self. window_size[1] - 1), nhead))  # 2*Wh-1 * 2*Ww-1, nH


        # double head
        self.num_convs = 4
        self.conv_out_channels = 256
        self.norm_cfg = dict(type='BN')
        self.conv_cfg = None
        self.conv_branch = self._add_conv_branch()

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)


    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def forward(self, x):
        b, c, h, w = x.shape # b == b * num_sample
        pos = self.pos_encoder(x)
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3])).bool().to(x.device)
        """
        hs = self.transformer(self.input_proj(x), mask, self.query_embed.weight, pos)[0]
        hs = hs[-1]
        
        cls_feat = torch.mean(hs,dim=1)
        outputs_class = self.class_embed(cls_feat)
        # outputs_coord = self.bbox_embed(hs[-1]).sigmoid()
        outputs_coord = self.bbox_embed(hs)
        outputs_coord = outputs_coord.view(outputs_coord.size(0),-1)
        return outputs_class, outputs_coord

        """
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # relative_position_bias = relative_position_bias.unsqueeze(0).repeat(x.shape[0],1,1,1).view(-1,49,49)
        # x = self.transformer(self.input_proj(x), mask, self.query_embed.weight, attn_mask=relative_position_bias)
        if self.num_decoder_layers > 0:
            hs, x = self.transformer(self.input_proj(x), mask, self.query_embed.weight, pos_embed=pos)
        else:
            x = self.transformer(self.input_proj(x), mask, self.query_embed.weight, pos_embed=pos)
        x_cls = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x_cls)

        if self.num_decoder_layers > 0:
            bbox_pred = self.bbox_embed(hs[-1])
            bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        else:
            for conv in self.conv_branch:
                x_conv = conv(x)

            x_conv = self.avg_pool(x_conv)

            x_conv = x_conv.view(x_conv.size(0), -1)
            bbox_pred = self.fc_reg(x_conv)
        return cls_score, bbox_pred

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class MLPBBoxHead(BBoxHead):

    def __init__(self, num_classes, inchanel=256, DS = 256, mlp_dim = 1024, depth = 1, num_convs = 4,
                 *args,
                 **kwargs):
        super(MLPBBoxHead, self).__init__(*args, **kwargs)
        self.mlpmixer = MLPMixer(num_token=7*7, emb_dim=inchanel, DS=DS, mlp_dim=mlp_dim, depth = depth)
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(7 * 7 * 256, num_classes + 1)
        # self.fc_reg = nn.Linear(7*7*256, 4 * num_classes)
        self.fc_reg = nn.Linear(256, 4 * num_classes)


        # double head
        self.num_convs = num_convs
        self.conv_out_channels = 256
        self.norm_cfg = dict(type='BN')
        self.conv_cfg = None
        self.conv_branch = self._add_conv_branch()

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def forward(self, x1, x2 = None):
        if x2 is None:
            cls_score, bbox_pred = self.forward_single_head(x1)
        else:
            cls_score, bbox_pred = self.forward_double_head(x1, x2)

        return cls_score, bbox_pred

    def forward_single_head(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b,c,h*w)
        x = x.permute(0, 2, 1)
        x = self.mlpmixer(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(b, c, h, w)

        x_cls = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x_cls)


        for conv in self.conv_branch:
            x_conv = conv(x)

        x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)
        return cls_score, bbox_pred

    def forward_double_head(self, x_cls, x_reg):
        b, c, h, w = x_cls.shape
        x_cls = x_cls.reshape(b, c, h * w)
        x_cls = x_cls.permute(0, 2, 1)
        x_cls = self.mlpmixer(x_cls)
        x_cls = x_cls.permute(0, 2, 1)
        x_cls = x_cls.reshape(b, c, h, w)

        x_cls = x_cls.view(x_cls.size(0), -1)
        cls_score = self.fc_cls(x_cls)

        x_reg = x_reg.reshape(b, c, h * w)
        x_reg = x_reg.permute(0, 2, 1)
        x_reg = self.mlpmixer(x_reg)
        x_reg = x_reg.permute(0, 2, 1)
        x_reg = x_reg.reshape(b, c, h, w)


        for conv in self.conv_branch:
            x_conv = conv(x_reg)

        x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)
        return cls_score, bbox_pred

