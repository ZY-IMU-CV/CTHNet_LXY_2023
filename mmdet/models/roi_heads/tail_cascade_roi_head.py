import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class TailCascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 bbox_head_tail=None,
                 labels=None,
                 labels_tail=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(TailCascadeRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # if bbox_head is not None:
        self.init_bbox_head(bbox_roi_extractor, bbox_head, bbox_head_tail)
        self.labels = labels
        self.labels_tail = labels_tail
        self.init_assigner_sampler()

    def init_bbox_head(self, bbox_roi_extractor, bbox_head, bbox_head_tail):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        self.bbox_head_tail = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        if not isinstance(bbox_head_tail, list):
            bbox_head_tail = [bbox_head_tail for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == len(bbox_head_tail) == self.num_stages
        for roi_extractor, head, head_tail in zip(bbox_roi_extractor, bbox_head, bbox_head_tail):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))
            self.bbox_head_tail.append(build_head(head_tail))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        self.bbox_assigner_tail = []
        self.bbox_sampler_tail = []
        if self.train_cfg is not None:
            for rcnn_train_cfg in self.train_cfg:
                self.bbox_assigner.append(build_assigner(rcnn_train_cfg.assigner))
                self.bbox_assigner_tail.append(build_assigner(rcnn_train_cfg.assigner_tail))
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler))
                self.bbox_sampler_tail.append(build_sampler(rcnn_train_cfg.sampler_tail))

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
                self.bbox_head_tail[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'],)
        return outs

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_tail(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head_tail = self.bbox_head_tail[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head_tail(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, sampling_results_tail, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        rois_tail = bbox2roi([res.bboxes for res in sampling_results_tail])
        bbox_results_tail = self._bbox_forward_tail(stage, x, rois_tail)
        bbox_targets_tail = self.bbox_head_tail[stage].get_targets(sampling_results_tail, gt_bboxes,
                                                                   gt_labels, rcnn_train_cfg)
        loss_bbox_tail = self.bbox_head_tail[stage].loss(bbox_results_tail['cls_score'],
                                                         bbox_results_tail['bbox_pred'], rois_tail,
                                                         *bbox_targets_tail)
        bbox_results_tail.update(loss_bbox=loss_bbox_tail, rois=rois_tail, bbox_targets=bbox_targets_tail)
        return bbox_results, bbox_results_tail

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        proposal_list_tail = proposal_list.copy()
        # print(len(self.bbox_assigner))
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            sampling_results_tail = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_assigner_tail = self.bbox_assigner_tail[i]
                bbox_sampler = self.bbox_sampler[i]
                bbox_sampler_tail = self.bbox_sampler_tail[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
                    assign_result_tail = bbox_assigner_tail.assign(
                        proposal_list_tail[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result_tail = bbox_sampler_tail.sample(
                        assign_result_tail,
                        proposal_list_tail[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results_tail.append(sampling_result_tail)
            # bbox head forward and loss
            bbox_results, bbox_results_tail = self._bbox_forward_train(i, x, sampling_results, sampling_results_tail,
                                                                       gt_bboxes, gt_labels,
                                                                       rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (value * lw if 'loss' in name else value)
            for name, value in bbox_results_tail['loss_bbox'].items():
                losses[f's{i}.tail.{name}'] = (value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                pos_is_gts_tail = [res.pos_is_gt for res in sampling_results_tail]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                roi_labels_tail = bbox_results_tail['bbox_targets'][0]

                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
                    roi_labels_tail = torch.where(
                        roi_labels_tail == self.bbox_head_tail[i].num_classes,
                        bbox_results_tail['cls_score'][:, :-1].argmax(1),
                        roi_labels_tail)
                    proposal_list_tail = self.bbox_head_tail[i].refine_bboxes(
                        bbox_results_tail['rois'], roi_labels_tail,
                        bbox_results_tail['bbox_pred'], pos_is_gts_tail, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        ms_scores_tail = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        rois_tail = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            ms_scores.append(bbox_results['cls_score'])
            bbox_results_tail = self._bbox_forward_tail(i, x, rois_tail)
            ms_scores_tail.append(bbox_results_tail['cls_score'])

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'][:, :-1].argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results['bbox_pred'], img_metas[0])
                bbox_label_tail = bbox_results_tail['cls_score'][:, :-1].argmax(dim=1)
                rois_tail = self.bbox_head_tail[i].regress_by_class(
                    rois_tail, bbox_label_tail, bbox_results_tail['bbox_pred'], img_metas[0])

        cls_score = sum(ms_scores) / self.num_stages
        cl_type = 2  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            cl_type,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            rois,
            cls_score,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        cls_score_tail = sum(ms_scores_tail) / self.num_stages
        det_bboxes_tail, det_labels_tail = self.bbox_head_tail[-1].get_bboxes(
            cl_type,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            rois_tail,
            cls_score_tail,
            bbox_results_tail['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        if self.labels is not None:
            inds = []
            for label in self.labels:
                inds.append(torch.nonzero(det_labels == label, as_tuple=False).squeeze(1))
            inds = torch.cat(inds)
            det_bboxes = det_bboxes[inds]
            det_labels = det_labels[inds]
        if self.labels_tail is not None:
            inds = []
            for label in self.labels_tail:
                inds.append(torch.nonzero(det_labels_tail == label, as_tuple=False).squeeze(1))
            inds = torch.cat(inds)
            det_bboxes_tail = det_bboxes_tail[inds]
            det_labels_tail = det_labels_tail[inds]
        det_bboxes = torch.cat((det_bboxes, det_bboxes_tail))
        det_labels = torch.cat((det_labels, det_labels_tail))
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    aug_masks.append(
                        mask_results['mask_pred'].sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_metas] * self.num_stages,
                                               self.test_cfg)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_mask:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
