from ..builder import HEADS
from .standard_roi_head import StandardRoIHead
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
from .bbox_heads.transformer import PositionEmbeddingSine
import numpy as np

@HEADS.register_module()
class TrHead(StandardRoIHead):
    def __init__(self, num_token=20, inchannel=7*7*256, emb_dim=1024, num_heads=4, mlp_dim =256, depth = 2, mode = 'transformer', reg_roi_scale_factor=1.3, **kwargs):
        super(TrHead, self).__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor
        self.mode = mode
        if mode == 'external_attention':
            self.roi_transformer = RoiTransformer_EA(num_token=num_token, channel=inchannel,
                                                     emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, depth=depth)
        elif mode == 'transformer':
            self.roi_transformer = RoiTransformer(num_token=num_token, channel=inchannel,
                                                     emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, depth=depth, mode = 'transformer')
        elif mode == 'MLP_mixer':
            self.roi_transformer = RoiTransformer(num_token=num_token, channel=inchannel, emb_dim=emb_dim, num_heads=num_heads,mlp_dim = mlp_dim, depth = depth, mode = 'MLP_mixer')
        elif mode == 'mix_mode':
            self.roi_transformer = RoiTransformer_EA_MLPmixer(num_token=num_token, channel=inchannel,
                                                     emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, depth=depth)
        self.pos_encoder = PositionEmbeddingSine(128)
        self.conv1_1 = nn.Conv2d(256+2,256,1,1)
        # self.scale = 2 * math.pi

    def pos_encode(self,xs):
        out = []
        for x in xs:
            mask = torch.ones((x.shape[0], x.shape[2], x.shape[3])).to(x.device)
            y_embed = mask.cumsum(1, dtype=torch.float32)
            x_embed = mask.cumsum(2, dtype=torch.float32)
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps)
            x_embed = x_embed / (x_embed[:, :, -1:] + eps)
            pos_embed = torch.cat((x_embed.unsqueeze(1), y_embed.unsqueeze(1)),dim=1)
            x = torch.cat((x, pos_embed), dim=1)
            x = self.conv1_1(x)
            out.append(x)
        return tuple(out)

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        b, c, _, _ = x[0].shape
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        x = self.pos_encode(x)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)

        _, c, h, w = bbox_cls_feats.shape
        num_sample = bbox_cls_feats.shape[0]//b
        shape = (b, num_sample, c, h, w)

        bbox_cls_feats = bbox_cls_feats.view(b,num_sample,-1)
        # bbox_cls_feats = bbox_cls_feats.view(1, -1, c*w*h)
        if self.mode == 'mix_mode':
            bbox_cls_feats = self.roi_transformer(bbox_cls_feats, shape)
        else:
            bbox_cls_feats = self.roi_transformer(bbox_cls_feats)
        # bbox_cls_feats = bbox_cls_feats.view(-1,c,h,w)
        bbox_cls_feats = bbox_cls_feats.view(b*num_sample,c,h,w)


        bbox_reg_feats = bbox_reg_feats.view(b, num_sample, -1)
        # bbox_reg_feats = bbox_reg_feats.view(1, -1, c*w*h)
        if self.mode == 'mix_mode':
            bbox_reg_feats = self.roi_transformer(bbox_reg_feats, shape)
        else:
            bbox_reg_feats = self.roi_transformer(bbox_reg_feats)
        bbox_reg_feats = bbox_reg_feats.view(b*num_sample, c, h, w)


        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results

    # def _bbox_forward(self, x, rois):
    #     """Box head forward function used in both training and testing."""
    #     # TODO: a more flexible way to decide which feature maps to use
    #     b,c, _,_ = x[0].shape
    #     bbox_feats = self.bbox_roi_extractor(
    #         x[:self.bbox_roi_extractor.num_inputs], rois)
    #     _,c, h, w = bbox_feats.shape
    #     num_sample = bbox_feats.shape[0]//b
    #     bbox_feats = bbox_feats.view(b,num_sample,-1)
    #     shape = (b, num_sample, c, h, w)
    #     if self.mode == 'mix_mode':
    #         bbox_feats = self.roi_transformer(bbox_feats, shape)
    #     else:
    #         bbox_feats = self.roi_transformer(bbox_feats)
    #     bbox_feats = bbox_feats.view(b*num_sample,c,h,w)

        # with torch.no_grad():
        #     mu = mu.mean(dim=0, keepdim=True)
        #     momentum = 0.9
        #     self.roi_transformer.mu *= momentum
        #     self.roi_transformer.mu += mu * (1 - momentum)


        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # cls_score, bbox_pred = self.bbox_head(bbox_feats)
        #
        # bbox_results = dict(
        #     cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        # return bbox_results


@HEADS.register_module()
class Dynamic_TrHead(TrHead):
    def __init__(self, **kwargs):
        super(Dynamic_TrHead, self).__init__(**kwargs)
        # the IoU history of the past `update_iter_interval` iterations
        self.iou_history = []
        # the beta history of the past `update_iter_interval` iterations
        self.beta_history = []

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """Forward function for training.

        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cur_iou = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # record the `iou_topk`-th largest IoU in an image
                iou_topk = min(self.train_cfg.dynamic_rcnn.iou_topk,
                               len(assign_result.max_overlaps))
                if sampling_result.bboxes.shape[0] != 512:
                    print()
                ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                cur_iou.append(ious[-1].item())
                sampling_results.append(sampling_result)
            # average the current IoUs over images
            cur_iou = np.mean(cur_iou)
            self.iou_history.append(cur_iou)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        # update IoU threshold and SmoothL1 beta
        update_iter_interval = self.train_cfg.dynamic_rcnn.update_iter_interval
        if len(self.iou_history) % update_iter_interval == 0:
            new_iou_thr, new_beta = self.update_hyperparameters()

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        num_imgs = len(img_metas)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # record the `beta_topk`-th smallest target
        # `bbox_targets[2]` and `bbox_targets[3]` stand for bbox_targets
        # and bbox_weights, respectively
        pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
        beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * num_imgs,
                        num_pos)
        cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
        self.beta_history.append(cur_target)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def update_hyperparameters(self):
        """Update hyperparameters like IoU thresholds for assigner and beta for
        SmoothL1 loss based on the training statistics.

        Returns:
            tuple[float]: the updated ``iou_thr`` and ``beta``.
        """
        new_iou_thr = max(self.train_cfg.dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history))

        self.iou_history = []
        self.bbox_assigner.pos_iou_thr = new_iou_thr
        self.bbox_assigner.neg_iou_thr = new_iou_thr
        self.bbox_assigner.min_pos_iou = new_iou_thr
        new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta,
                       np.median(self.beta_history))
        self.beta_history = []
        # self.bbox_head.loss_bbox.beta = new_beta
        return new_iou_thr, new_beta



class RoiTransformer_EA(nn.Module):
    def __init__(self, num_token, channel, emb_dim, num_heads, mlp_dim, depth=2):
        super(RoiTransformer_EA, self).__init__()
        self.model = nn.ModuleList()
        for i in range(depth):
            self.model.append(ExternalAttention(num_token, channel))
        return

    def forward(self, bbox_feats):
        for ea in self.model:
            bbox_feats = ea(bbox_feats)

        return bbox_feats

class RoiTransformer_EA_MLPmixer(nn.Module):
    def __init__(self, num_token, channel, emb_dim, num_heads, mlp_dim, DS = 256, depth=2):
        super(RoiTransformer_EA_MLPmixer, self).__init__()
        self.EAs = nn.ModuleList()
        self.MLPmixer = nn.ModuleList()
        for i in range(depth):
            self.EAs.append(ExternalAttention(num_token, channel))
        for i in range(depth):
            self.MLPmixer.append(MixerBlock(7*7, 256, emb_dim, mlp_dim))
            # emb_dim == DS, mlp_dim == DC
        return

    def forward(self, bbox_feats, shape):
        b, num_sample, c, h, w = shape
        for ea,mlp in zip(self.EAs, self.MLPmixer):
            bbox_feats = ea(bbox_feats)
            bbox_feats = bbox_feats.reshape(b*num_sample, c, h * w)
            bbox_feats = bbox_feats.permute(0, 2, 1)
            bbox_feats = mlp(bbox_feats)
            bbox_feats = bbox_feats.permute(0, 2, 1)
            bbox_feats = bbox_feats.reshape(b, num_sample, -1)
        return bbox_feats




class ExternalAttention(nn.Module):
    def __init__(self, num_token, channel):
        super(ExternalAttention, self).__init__()
        self.M_k = nn.Linear(channel, num_token, bias=False)
        self.M_v = nn.Linear(num_token, channel, bias=False)
        return

    def forward(self, bbox_feats):
        b, l, c = bbox_feats.shape
        attn = self.M_k(bbox_feats)
        attn = F.softmax(attn, dim=1)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.M_v(attn)

        return out + bbox_feats


class MixerBlock(nn.Module):
    def __init__(self,num_token, emb_dim, DS, mlp_dim):
        super(MixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.token_mixing = MLPBlock(num_token, DS)
        self.channel_mixing = MLPBlock(emb_dim, mlp_dim)

    def forward(self, x):
        b, l, c = x.shape
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.token_mixing(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        x = x + self.channel_mixing(y)
        return x

class MLPBlock(nn.Module):
    def __init__(self,inchannel, hidden_channel):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(inchannel, hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, inchannel)

    def forward(self,x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_token, emb_dim, DS, mlp_dim, depth = 1):
        super(MLPMixer, self).__init__()
        self.model = nn.ModuleList()
        for i in range(depth):
            self.model.append(MixerBlock(num_token, emb_dim, DS, mlp_dim))


    def forward(self,x):
        for mlp in self.model:
            x = mlp(x)
        return x


class RoiTransformer(nn.Module):
    def __init__(self, num_token, channel, emb_dim, num_heads, mlp_dim, depth = 2, mode = "transformer"):
        super(RoiTransformer,self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.stage_num = 4
        # Tokenization
        self.Trokenize = nn.Linear(channel, num_token)
        self.proj = nn.Linear(channel,emb_dim)

        self.Q_T2X = nn.Linear(channel,channel)
        self.K_T2X = nn.Linear(emb_dim,channel)
        self.deproj = nn.Linear(emb_dim, channel)

        # self.deproj = nn.Linear(emb_dim, channel, bias=False)
        # self.bn = nn.BatchNorm1d(channel)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_token), emb_dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)  # initialized based on the paper
        self.dropout = nn.Dropout(0.1)

        if mode == 'transformer':
            self.transformer = Transformer(emb_dim,depth=depth,heads=num_heads, mlp_dim=mlp_dim,dropout=0.1)
        else:
            self.transformer = MLPMixer(num_token=num_token, emb_dim=emb_dim, DS=256, mlp_dim=mlp_dim, depth = depth)

        mu = torch.Tensor(1, emb_dim, num_token)
        mu.normal_(0, math.sqrt(2. / num_token))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        return

    def forward(self, bbox_feats):
        b, l, c = bbox_feats.shape
        shortcut = bbox_feats

        A = self.Trokenize(bbox_feats)  # (b,l,num_token)
        V = self.proj(bbox_feats) # (b,l,emb_dim)
        A = A.permute(0,2,1)
        A = F.softmax(A, dim=2)
        T = torch.bmm(A,V)

        # cls_tokens = self.cls_token.expand(b, -1, -1)
        # T = torch.cat((cls_tokens, T), dim=1)
        # T += self.pos_embedding
        T = self.dropout(T)
        T = self.transformer(T) # (b,num_token,emb_dim)

        X_rev = self.Q_T2X(bbox_feats)
        T_rev = self.K_T2X(T)
        attn = torch.bmm(X_rev, T_rev.permute(0,2,1))
        attn = F.softmax(attn, dim=-1)
        V_rev = self.deproj(T)
        res = torch.bmm(attn,V_rev)

        bbox_feats = shortcut + res
        return bbox_feats

    # cluster
    # def forward(self, bbox_feats, train=False):
    #     b, l, c = bbox_feats.shape
    #     shortcut = bbox_feats
    #
    #     x = self.proj(bbox_feats) # (b,l,emb_dim)
    #     mu = self.mu.repeat(b, 1, 1)  # b * emb_dim * num_token
    #     with torch.no_grad():
    #         for i in range(self.stage_num - 1):
    #             x_t = x
    #             z = torch.bmm(x_t, mu)  # b * l * num_token
    #             z = F.softmax(z, dim=2)  # b * l * num_token
    #             z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
    #             mu = torch.bmm(x.permute(0,2,1), z_)  # b * emb_dim * num_token
    #             mu = self._l2norm(mu, dim=1)
    #     x_t = x
    #     z = torch.bmm(x_t, mu)  # b * l * num_token
    #     z = F.softmax(z, dim=2)  # b * l * num_token
    #     z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
    #     mu = torch.bmm(x.permute(0,2,1), z_)  # b * emb_dim * num_token
    #     mu = self._l2norm(mu, dim=1)
    #
    #     T = mu.permute(0,2,1)
    #     T = self.dropout(T)
    #     T = self.transformer(T, None) # (b,l,emb_dim)
    #
    #     # z_t = z.permute(0, 2, 1)  # b * num_token * l
    #     # x = T.matmul(z_t)  # b * emb_dim * l
    #     x = z.matmul(T) # b * l * emb_dim
    #     x = F.relu(x, inplace=True)
    #
    #     x = self.deproj(x)
    #     x = x.view(b*l,c)
    #     x = self.bn(x)
    #     x = x.view(b, l,c)
    #
    #     bbox_feats = shortcut + x
    #     if train==True:
    #         return bbox_feats, mu.detach()
    #     else:
    #         return bbox_feats

    # def forward(self, bbox_feats, train=False):
    #     b, l, c = bbox_feats.shape
    #     shortcut = bbox_feats
    #
    #     x = self.proj(bbox_feats)  # (b,l,emb_dim)
    #     with torch.no_grad():
    #         mu = self.kmeans(x) # (b, emb_dim, num_token)
    #     A = torch.bmm(x,mu)
    #     A = A.permute(0,2,1)
    #     A = F.softmax(A, dim=2) # (b, num_token, l)
    #     T = torch.bmm(A,x)
    #
    #     T = self.dropout(T)
    #     T = self.transformer(T, None)  # (b,l,emb_dim)
    #
    #     X_rev = self.Q_T2X(bbox_feats)
    #     T_rev = self.K_T2X(T)
    #     attn = torch.bmm(X_rev, T_rev.permute(0,2,1))
    #     attn = F.softmax(attn, dim=-1)
    #     V_rev = self.deproj(T)
    #     res = torch.bmm(attn,V_rev)
    #
    #     bbox_feats = shortcut + res
    #     if train == True:
    #         return bbox_feats, mu.detach()
    #     else:
    #         return bbox_feats

    def kmeans(self, bbox_feats):
        b, l, c = bbox_feats.shape
        bbox_feats = bbox_feats.permute(0,2,1)
        bbox_feats = self._l2norm(bbox_feats, dim=1)
        mu = self.mu.repeat(b, 1, 1)
        for i in range(self.stage_num):
            dist_npl = (bbox_feats[...,None] - self.mu[:,:,None,:]).norm(dim=1)
            term,idx = dist_npl.min(dim=2)
            term = term.unsqueeze(2)
            mask_npl =  (dist_npl == term).float()
            mu = torch.bmm(bbox_feats,mask_npl)
            mu = mu / (mask_npl.sum(dim=1,keepdim=True) + 1e-6 )
            mu = self._l2norm(mu,dim=1)
        return mu

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, dropout):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
#                 Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
#             ]))
#
#     def forward(self, x, mask=None):
#         for attention, mlp in self.layers:
#             x = attention(x, mask=mask)  # go to attention
#             x = mlp(x)  # go to MLP_Block
#         return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads=heads, dropout=dropout)),
                Residual(MLP_Block(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = self.norm1(x)
            x = attention(x, mask=mask)  # go to attention
            x = self.norm2(x)
            x = mlp(x)  # go to MLP_Block
        return x