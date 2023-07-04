# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FaLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 score_thr=0.7,
                 json_file='./data/lvis_v0.5/annotations/lvis_v0.5_train.json',
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FaLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.score_thr = score_thr
        assert self.score_thr > 0 and self.score_thr < 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        assert len(json_file) != 0
        self.freq_group = self.get_freq_info(json_file)
        self.activated = activated

    def get_freq_info(self, json_file):
        cats = json.load(open(json_file, 'r'))['categories']

        freq_dict = {'rare': [], 'common': [], 'freq': []}

        for cat in cats:
            if cat['frequency'] == 'r':
                freq_dict['rare'].append(cat['id'])
            elif cat['frequency'] == 'c':
                freq_dict['common'].append(cat['id'])
            elif cat['frequency'] == 'f':
                freq_dict['freq'].append(cat['id'])
            else:
                print('Something wrong with the json file.')

        return freq_dict

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
#______________________________________________________________________________________________
        device = pred.device
        self.n_i, self.n_c = pred.size()
        #target = pred.new_zeros(self.n_i, self.n_c)

        unique_label = torch.unique(target)

        with torch.no_grad():
            sigmoid_cls_logits = torch.sigmoid(pred)

        high_score_inds = torch.nonzero(sigmoid_cls_logits >= self.score_thr, as_tuple=False)
        weight_mask = torch.sparse_coo_tensor(high_score_inds.t(), pred.new_ones(high_score_inds.shape[0]),
                                              size=(self.n_i, self.n_c), device=device).to_dense()

        for cls in unique_label:
            cls = cls.item()
            cls_inds = torch.nonzero(target == cls, as_tuple=False).squeeze(1)

            if cls == 1230:
                # construct target vector for background samples
                #target[cls_inds, 1230] = 1
                # for bg, set the weight of all classes to 1
                weight_mask[cls_inds] = 0

                cls_inds_cpu = cls_inds.cpu()

                ## Solve the rare categories, random choost 1/3 bg samples to suppress rare categories

                rare_cats = self.freq_group['rare']

                rare_cats = (np.array(rare_cats) - 1).tolist()
                rare_cats = torch.tensor(rare_cats, device=pred.device)

                choose_bg_num = int(len(cls_inds) * 0.01)
                choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False),
                                              device=device)

                tmp_weight_mask = weight_mask[choose_bg_inds]
                tmp_weight_mask[:, rare_cats] = 1

                weight_mask[choose_bg_inds] = tmp_weight_mask

                ## Solve the common categories, random choost 2/3 bg samples to suppress rare categories
                common_cats = self.freq_group['common']
                common_cats = (np.array(common_cats) - 1).tolist()

                common_cats = torch.tensor(common_cats, device=pred.device)

                choose_bg_num = int(len(cls_inds) * 0.1)
                choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False),
                                              device=device)

                tmp_weight_mask = weight_mask[choose_bg_inds]
                tmp_weight_mask[:, common_cats] = 1

                weight_mask[choose_bg_inds] = tmp_weight_mask

                ## Solve the frequent categories, random choost all bg samples to suppress rare categories
                freq_cats = self.freq_group['freq']
                freq_cats = (np.array(freq_cats) - 1).tolist()
                freq_cats = torch.tensor(freq_cats, device=pred.device)

                choose_bg_num = int(len(cls_inds) * 1.0)
                choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False),
                                              device=device)

                tmp_weight_mask = weight_mask[choose_bg_inds]
                tmp_weight_mask[:, freq_cats] = 1

                weight_mask[choose_bg_inds] = tmp_weight_mask

                # Set the weight for bg to 1
                # weight_mask[cls_inds, 1230] = 1

            else:
                # construct target vector for foreground samples
                cur_labels = [cls]
                cur_labels = torch.tensor(cur_labels, device=pred.device)
                tmp_label_vec = pred.new_zeros(self.n_c)
                tmp_label_vec[cur_labels] = 1
                #tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), self.n_c)
                #target[cls_inds] = tmp_label_vec
                # construct weight mask for fg samples
                tmp_weight_mask_vec = weight_mask[cls_inds]
                # set the weight for ground truth category
                tmp_weight_mask_vec[:, cur_labels] = 1

                weight_mask[cls_inds] = tmp_weight_mask_vec
        weight=weight_mask
#___________________________________________________________________________________________
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss#enter
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss
            #print(weight.size())
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
