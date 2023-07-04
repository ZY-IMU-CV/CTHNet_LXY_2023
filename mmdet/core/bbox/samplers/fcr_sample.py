import numpy as np
import torch
import random

from ..builder import BBOX_SAMPLERS
from .random_sampler import RandomSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class FCRSampler(RandomSampler):
    """Instance balanced sampler that samples equal number of positive samples
    for each instance."""

    def __init__(self,
                 num,
                 pos_fraction,
                 af,
                 ac,
                 ar,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 labels=None,
                 labels_center=None,
                 labels_tail=None,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))
        self.af=af
        self.ac = ac
        self.ar = ar
        self.labels = labels
        self.labels_center = labels_center
        self.labels_tail = labels_tail


    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, labels=self.labels,labels_center=self.labels_center,labels_tail=self.labels_tail,**kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

    def _sample_pos(self, assign_result, num_expected, labels,labels_center,labels_tail, **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        # 首先看一下给的bboxes里面有哪些label是大于0的 得到了他们的index
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        # 首先只要这个index的数目不是0个 这些就都可以是positive sample
        # 当pos_indxs的数目小于想要的sample的数目的时候
        # 就直接用这个pos_index
        # 反之就从这么多index里随机采样num_expected个出来
        # print(assign_result.gt_inds)
        # print(pos_inds)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        elif assign_result.labels is None:
            return pos_inds
        else:

            sampled_inds = []
            sampled_inds_center = []
            sampled_inds_tail = []

#________________________________________________________________________________________________________________________________
            for label_f in labels:
                sampled_inds.append(torch.nonzero(assign_result.labels == label_f, as_tuple=False).squeeze(1))
            sampled_inds = torch.cat(sampled_inds)

            if len(sampled_inds) > self.af:
                sampled_inds = self.random_choice(sampled_inds, self.af)
# ________________________________________________________________________________________________________________________________
            for label_c in labels_center:
                sampled_inds_center.append(torch.nonzero(assign_result.labels == label_c, as_tuple=False).squeeze(1))
            sampled_inds_center = torch.cat(sampled_inds_center)

            if len(sampled_inds_center) > self.ac:
                sampled_inds_center = self.random_choice(sampled_inds_center, self.ac)
# ________________________________________________________________________________________________________________________________
            for label_r in labels_tail:
                sampled_inds_tail.append(torch.nonzero(assign_result.labels == label_r, as_tuple=False).squeeze(1))
            sampled_inds_tail = torch.cat(sampled_inds_tail)

            if len(sampled_inds_tail) > self.ar:
                sampled_inds_tail= self.random_choice(sampled_inds_tail, self.ar)

            all_sample=torch.cat([sampled_inds, sampled_inds_center,sampled_inds_tail])

            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            # print(len(f_sample))
            # print(len(c_sample))
            # print(len(r_sample))
            # print(len(all_sample))
            return all_sample

            # if len(sampled_inds) < num_expected:
            #     num_extra = (num_expected - len(sampled_inds)) // len(sampled_inds) + 2
            #     sampled_inds = sampled_inds.repeat([num_extra])
            #     sampled_inds = self.random_choice(sampled_inds, num_expected)
            # if len(sampled_inds) < num_expected:
            #     num_extra = num_expected - len(sampled_inds)
            #     # print(pos_inds, sampled_inds)
            #     extra_inds = np.array(
            #         list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
            #     if len(extra_inds) > num_extra:
            #         extra_inds = self.random_choice(extra_inds, num_extra)
            #     extra_inds = torch.from_numpy(extra_inds).to(
            #         assign_result.gt_inds.device).long()
            #     # print(sampled_inds.shape, extra_inds.shape)
            #     sampled_inds = torch.cat([sampled_inds, extra_inds])
            # if len(sampled_inds) > num_expected:
            #     sampled_inds = self.random_choice(sampled_inds, num_expected)
            # return sampled_inds
