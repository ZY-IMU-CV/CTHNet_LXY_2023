import torch
from mmcv.ops.nms import batched_nms


def multiclass_nms(#cl_type,
                   multi_bboxes, #bboxes.size()=[1000,4920]
                   multi_scores,#scores.size()=[1000,1231]
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """

    num_classes = multi_scores.size(1)-1
    # exclude background category
    #multi_scores.size(0)==1000
    #multi_bboxes.shape[1]=4920
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)#[1000,1230,4]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]#[1000,1230]

    # filter out boxes with low scores
    valid_mask = scores > score_thr#[1000,1230] true 135402  77969 45815 59981 48580 70056 89776 81922
    #valid_mask = scores > 0.01
    # print("————————————————————————————————————————————————————————————————————————")
    # print(cl_type)
    # if cl_type==1:
    #     valid_mask = scores > 0.01
    # elif cl_type==3:
    #     valid_mask = scores > 0.99
    # elif cl_type==2:
    #     valid_mask = scores > 0.0002


    bboxes = bboxes[valid_mask]#[len(valid=true),4]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]#[len(valid=true)]
    labels = valid_mask.nonzero()[:, 1]#[len(valid=true)]
    print("+++++++++++++++++++++++++++++++++++++++++")
    print(scores)
    print(len(scores))

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    #nums_cfg={'type': 'nms', 'iou_threshold': 0.5, 'iou_thr': 0.5}
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    #dets.size()=[16360,5]
    #keep.size()=[16360]
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    #dets.size=[300,5]
    #len(labels[keep]=300
    return dets, labels[keep]