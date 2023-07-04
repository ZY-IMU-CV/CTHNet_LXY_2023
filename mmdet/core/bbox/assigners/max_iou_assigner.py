import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonetrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,# max iou大于该值，则为正样本
                 neg_iou_thr,# max iou小于该值，则为负样本
                 min_pos_iou=.0,# match_low_quality为True时，max iou大于该值时，为正样本
                 gt_max_assign_all=True,# match_low_quality为True时，是否对所有argmax bbox进行赋值
                 ignore_iof_thr=-1,#若bbox与gt_ignore的iof大于该值，则该bbox设置为忽略样本
                 ignore_wrt_candidates=True,# 确定iof的前景
                 match_low_quality=True,# 是否启动第四步，可选步骤
                 gpu_assign_thr=-1, # 为了节省显存，如果gt数量大于gpu_assign_thr，则放到CPU上计算iou。若小于0，则无论如何都放在GPU上
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        # 主要是为了节省显存，iou计算还是很耗显存的，上万个bbox呢
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()
        # 第一步：计算iou
        overlaps = self.iou_calculator(gt_bboxes, bboxes)#bbox_overlaps这个函数,传进去第一个参数是gt框，第二个候选框。
        # 第二步：计算bbox与gt_ignore的iof
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:# 为True，则前景为bboxes，否则前景为gt_bboxes_ignore
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1 # 该bbox的所有gt的iou都设为-1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        # 映射回GPU设备
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        #num_bboxes为bbox的数量
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        # 第三步：初始化全为-1，表示忽略样本
        # 1. assign -1 by default
        #assigned_gt_inds:记录bboxes标记（正样本，负样本，忽略样本）的数组，初始全为忽略样本 -1
        # 如果您有一个张量，并且想要在同一个设备上创建一个相同类型的张量，那么您可以使用 torch.Tensor.new_* 方法；
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        # print(assigned_gt_inds.shape,overlaps.shape)
        #当没有gt_bbox时，将所有bbox划分为负样本
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # 当gt_bbox数量为0时，将所有bbox标记为负样本
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                # 当bbox的数量为0时，相当于没有样本，assigned_gt_inds所有值为 - 1（忽略样本），通常不会发生这种情况
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts

        #计算每一个bbox与所有gt_bbox的IoU的最大值
        #dim=0获取每一列的最大值，max_overlaps：一维数组记录最大IoU，argmax_overlaps：一维数组记录此最大IoU的列坐标
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        #对于每个anchor，和所有GT计算IoU（沿dim=0做max），找出最大的IoU：max_overlaps，以及对应的索引位置argmax_overlaps
        # print(overlaps.max(dim=0))
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals

        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
        #对于每个GT，和所有anchors计算IoU（沿dim=1做max），找出最大IoU：gt_max_overlaps，以及对应的索引位置 gt_argmax_overlaps
        # 第四步：分配负样本
        # 2. assign negative: below
        # the negative inds are set to be 0
        #将最大IoU在[0，neg_iou_thr）之间的bbox标记为负样本
        if isinstance(self.neg_iou_thr, float):
            # 当neg_iou_thr为浮点数时，将最大IoU在[0，neg_iou_thr）之间的bbox标记为负样本
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
            # (max_overlaps >= 0)条件也需要！！！因为之前把bbox ignore的iou都设为-1了！
        elif isinstance(self.neg_iou_thr, tuple):
            # 当neg_iou_thr为元组时，将最大IoU在[neg_iou_thr[0]，neg_iou_thr[1]）之间的bbox标记为负样本
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0
        # 第五步：分配正样本
        # 3. assign positive: above positive IoU threshold
        # print(max_overlaps == argmax_overlaps)
        #将最大IoU大于等于pos_iou_thr的bbox标记为正样本
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1#这里对坐标+1是因为0已经分配给了负样本
        # 第六步：match_low_quality
        if self.match_low_quality:

            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        #为正样本分配标签
        if gt_labels is not None:
            # assigned_labels为-1表示非正样本(包括负样本和忽略样本),0~len(class)-1就是正样本对应的class
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)       #assigned_labels=[-1,-1,-1......]初始化时全为-1
            #返回大于0的坐标
            # pos_inds对应labels在assigned_labels中的位置
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            #-1对应gt的坐标
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
                # assigned_gt_inds[pos_inds] - 1是取得bbox对应的gt的索引，之前加一，这里要减一

        else:
            assigned_labels = None
        # if assigned_labels is not None:
        #     print(num_gts, assigned_gt_inds.shape, max_overlaps.shape, assigned_labels.shape)
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

# num_gt:为真实边界框的数量
