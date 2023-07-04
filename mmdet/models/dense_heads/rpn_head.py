import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin


@HEADS.register_module()
class RPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(
            1, in_channels, background_label=0, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_scores,#网络输出的 confidence
                           bbox_preds,#网络输出的坐标值
                           mlvl_anchors,# 每个 scale 的生成的 anchor
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        # 1. 根据类别的置信度筛选出每个尺度 topK（K = 2000）个 bbox
        # 2. 合并筛选后的多个尺度的 bbox
        # 3. 将网络预测值解码
        # 4. 用 nms（阈值=0.7）合并 bbox
        # 5. 筛选出前 nms_post（1000）个 bbox 作为 proposal
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        # 遍历每个尺度
        for idx in range(len(cls_scores)):
            # 取到一个尺度的网络输出的类别和位置预测
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            # 保证后两个维度相同
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            # 将类别的数值压缩成概率.
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                # 前景 label 设置为:  [0, 类别数 - 1],
                # 背景 label 设置为:  类别数
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                # 对类别维度进行 softmax, 取背景的概率
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # 取对应层的 anchor: [单尺度 anchor 总数, 4]
            anchors = mlvl_anchors[idx]
            # 根据类别的置信度筛选出 topK 个 box
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                # 对 score 从高到低排序
                ranked_scores, rank_inds = scores.sort(descending=True)
                # 取 topK（cfg.nms_pre） 个 index 和 score: anchor --> [2000]
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                # 根据类别的预测值，取 topK 个 bbox
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                # anchor 也取 topK 个
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            # new_full(形状，填充值，数据类型)
            level_ids.append(
                scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)
        # 如多对 anchor 的大小有限定
        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            # 取长宽都 > min_bbox_size 的索引
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()

            # 筛选目标
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        # nms 后对置信度有排序。直接取前 nms_post 个
        return dets[:cfg.nms_post]
