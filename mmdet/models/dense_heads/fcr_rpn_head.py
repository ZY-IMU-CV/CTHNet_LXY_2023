import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmcv.ops import nms
from mmdet.core import bbox_overlaps
from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin
import numpy as np

nms_resampling_thresh = np.load('kit/nms_resampling_thresh.npy')
@HEADS.register_module()
class FCRRPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(FCRRPNHead, self).__init__(
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
        losses = super(FCRRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           gt_bboxes,
                           gt_labels,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = rpn_cls_score.softmax(dim=1)[:, :-1]
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_shape)
            # filter out too small bboxes
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size), as_tuple=False).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            #proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)#!!!!!!!!!!!!!!!!!!!!!!!
            if cfg.nms_resampling is not None:  # only used in training
                if cfg.nms_resampling[0] == 'discrete':

                    a_r = cfg.nms_resampling[1]
                    a_c = cfg.nms_resampling[2]
                    a_f = cfg.nms_resampling[3]
                    proposals = self.nms_resampling_discrete(proposals,scores, gt_bboxes, gt_labels, a_r, a_c, a_f)
                elif cfg.nms_resampling[0] == 'linear':
                    thresh = nms_resampling_thresh
                    proposals = self.nms_resampling_linear(proposals, gt_bboxes, gt_labels, thresh)
            else:
                proposals, _ = nms(proposals, scores, cfg.nms_thr)
            #proposals, _ = nms(proposals, scores, cfg.nms_thr)

            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            # NMS across multi levels
            proposals, _ = nms(proposals[:, :4], proposals[:, -1], cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
    def _get_bboxes_test_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_proposals = []
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = rpn_cls_score.softmax(dim=1)[:, :-1]
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_shape)
            # filter out too small bboxes
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size), as_tuple=False).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            #proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)#!!!!!!!!!!!!!!!!!!!!!!!

            proposals, _ = nms(proposals, scores, cfg.nms_thr)
            #proposals, _ = nms(proposals, scores, cfg.nms_thr)

            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            # NMS across multi levels
            proposals, _ = nms(proposals[:, :4], proposals[:, -1], cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
    def nms_resampling_linear(self, proposals, gt_bboxes, gt_labels, thresh):
        assert any(gt_labels>0)
        iou = bbox_overlaps(proposals[:, :4], gt_bboxes)
        max_iou, gt_assignment = iou.max(dim=1)
        proposals_labels = gt_labels[gt_assignment]
        # proposal is considered as background when its iou with gt < 0.3
        proposals_labels[max_iou < 0.3] = 0

        proposals_labels = proposals_labels.cpu().numpy()
        t = thresh[proposals_labels]
        keep = self.nms_py(proposals.cpu().numpy(), t)
        keep = np.array(keep)

        return proposals[keep, :]

    def nms_py(self, dets, thresh):
        """
        greedily select boxes with high confidence and overlap with current maximum <= thresh
        rule out overlap >= thresh
        :param dets: [[x1, y1, x2, y2 score]]
        :param thresh: retain overlap < thresh
        :return: indexes to keep
        """
        if dets.shape[0] == 0:
            return []
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh[i])[0]
            order = order[inds + 1]
        return keep
    def nms_resampling_discrete(self, proposals,scores, gt_bboxes, gt_labels, a_r, a_c, a_f):

        
        assert any(gt_labels>0)
        # proposal is considered as background when its iou with gt < 0.3
        select_thresh = 0.3
        out= []

        rare, common, frequent = self.get_category_frequency(gt_labels.device)
        rare_gtbox = torch.zeros((2000, 4), device=gt_labels.device)
        rare_gtbox_idx = 0
        common_gtbox = torch.zeros((2000, 4), device=gt_labels.device)
        common_gtbox_idx = 0
        frequent_gtbox = torch.zeros((2000, 4), device=gt_labels.device)
        frequent_gtbox_idx = 0
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            if gt_label in rare:
                rare_gtbox[rare_gtbox_idx, ...] = gt_bbox
                rare_gtbox_idx += 1
            elif gt_label in common:
                common_gtbox[common_gtbox_idx, ...] = gt_bbox
                common_gtbox_idx += 1
            else:
                frequent_gtbox[frequent_gtbox_idx, ...] = gt_bbox
                frequent_gtbox_idx += 1
        rare_gtbox = rare_gtbox[:rare_gtbox_idx, ...]
        common_gtbox = common_gtbox[:common_gtbox_idx, ...]

        frequent_proposals, _ = nms(proposals, scores,a_f)
        if len(rare_gtbox) > 0:
            rare_proposals, _ = nms(proposals, scores,a_r)
            rare_overlaps = bbox_overlaps(rare_gtbox, rare_proposals[:, :4])
            rare_max_overlaps, rare_argmax_overlaps = rare_overlaps.max(dim=0)
            rare_pos_inds = rare_max_overlaps >= select_thresh
            rare_proposals = rare_proposals[rare_pos_inds, :]
            out.append(rare_proposals)

            frequent_rare_overlaps = bbox_overlaps(rare_gtbox, frequent_proposals[:, :4])
            frequent_rare_max_overlaps, frequent_rare_argmax_overlaps = frequent_rare_overlaps.max(dim=0)
            valid_inds = frequent_rare_max_overlaps < select_thresh
            frequent_proposals = frequent_proposals[valid_inds, :]
        if len(common_gtbox) > 0:
            common_proposals, _ = nms(proposals,scores, a_c)
            common_overlaps = bbox_overlaps(common_gtbox, common_proposals[:, :4])
            common_max_overlaps, common_argmax_overlaps = common_overlaps.max(dim=0)
            common_pos_inds = common_max_overlaps >= select_thresh
            common_proposals = common_proposals[common_pos_inds, :]
            out.append(common_proposals)

            frequent_common_overlaps = bbox_overlaps(common_gtbox, frequent_proposals[:, :4])
            frequent_common_max_overlaps, frequent_common_argmax_overlaps = frequent_common_overlaps.max(dim=0)
            valid_inds = frequent_common_max_overlaps < select_thresh
            frequent_proposals = frequent_proposals[valid_inds, :]
        out.append(frequent_proposals)
        if len(out) > 1:
            out_proposals = torch.cat(out, 0)
        else:
            out_proposals = frequent_proposals

        return out_proposals

    def get_category_frequency(self, device):
        # rare, common, frequent are defined by the LVIS v0.5 dataset
        rare = torch.tensor([0,    6,    9,   13,   14,   15,   20,   21,   30,   37,   38,   39,
                    41,   45,   48,   50,   51,   63,   64,   69,   71,   73,   82,   85,
                    93,   99,  100,  104,  105,  106,  112,  115,  116,  119,  121,  124,
                    126,  129,  130,  135,  139,  141,  142,  143,  146,  149,  154,  158,
                    160,  162,  163,  166,  168,  172,  180,  181,  183,  195,  198,  202,
                    204,  205,  208,  212,  213,  216,  217,  218,  225,  226,  230,  235,
                    237,  238,  240,  241,  242,  244,  245,  248,  249,  250,  251,  252,
                    254,  257,  258,  264,  265,  269,  270,  272,  279,  283,  286,  290,
                    292,  294,  295,  297,  299,  302,  303,  305,  306,  309,  310,  312,
                    315,  316,  317,  319,  320,  321,  323,  325,  327,  328,  329,  334,
                    335,  341,  343,  349,  350,  353,  355,  356,  357,  358,  359,  360,
                    365,  367,  368,  369,  371,  377,  378,  384,  385,  387,  388,  392,
                    393,  401,  402,  403,  405,  407,  410,  412,  413,  416,  419,  420,
                    422,  426,  429,  432,  433,  434,  437,  438,  440,  441,  445,  453,
                    454,  455,  461,  463,  468,  472,  475,  476,  477,  482,  484,  485,
                    487,  488,  492,  494,  495,  497,  508,  509,  511,  513,  514,  515,
                    517,  520,  523,  524,  525,  526,  529,  533,  540,  541,  542,  544,
                    547,  550,  551,  552,  554,  555,  561,  563,  568,  571,  572,  580,
                    581,  583,  584,  585,  586,  589,  591,  592,  593,  595,  596,  599,
                    601,  604,  608,  609,  611,  612,  615,  616,  625,  626,  628,  629,
                    630,  633,  635,  642,  644,  645,  649,  655,  657,  658,  662,  663,
                    664,  670,  673,  675,  676,  682,  683,  685,  689,  695,  697,  699,
                    702,  711,  712,  715,  721,  722,  723,  724,  726,  729,  731,  733,
                    734,  738,  740,  741,  744,  748,  754,  758,  764,  766,  767,  768,
                    771,  772,  774,  776,  777,  781,  782,  784,  789,  790,  794,  795,
                    796,  798,  799,  803,  805,  806,  807,  808,  815,  817,  820,  821,
                    822,  824,  825,  827,  832,  833,  835,  836,  840,  842,  844,  846,
                    856,  862,  863,  864,  865,  866,  868,  869,  870,  871,  872,  875,
                    877,  882,  886,  892,  893,  897,  898,  900,  901,  904,  905,  907,
                    915,  918,  919,  920,  921,  922,  926,  927,  930,  931,  933,  939,
                    940,  944,  945,  946,  948,  950,  951,  953,  954,  955,  956,  958,
                    959,  961,  962,  963,  969,  974,  975,  988,  990,  991,  998,  999,
                    1001, 1003, 1005, 1008, 1009, 1010, 1012, 1015, 1020, 1022, 1025, 1026,
                    1028, 1029, 1032, 1033, 1046, 1047, 1048, 1049, 1050, 1055, 1066, 1067,
                    1068, 1072, 1073, 1076, 1077, 1086, 1094, 1099, 1103, 1111, 1132, 1135,
                    1137, 1138, 1139, 1140, 1144, 1146, 1148, 1150, 1152, 1153, 1156, 1158,
                    1165, 1166, 1167, 1168, 1169, 1171, 1178, 1179, 1180, 1186, 1187, 1188,
                    1189, 1203, 1204, 1205, 1213, 1215, 1218, 1224, 1225, 1227], device=device)

        common = torch.tensor([1,    4,    5,    7,    8,   10,   17,   18,   22,   24,   25,   26,
                        27,   28,   32,   36,   43,   46,   47,   53,   54,   61,   62,   67,
                        70,   72,   74,   75,   79,   81,   84,   91,   92,   96,   97,  101,
                        102,  107,  108,  110,  113,  114,  118,  120,  122,  127,  128,  133,
                        134,  140,  144,  147,  148,  150,  151,  152,  155,  156,  157,  161,
                        164,  165,  167,  170,  174,  175,  176,  177,  185,  187,  188,  189,
                        191,  192,  194,  199,  200,  201,  203,  206,  209,  214,  215,  219,
                        221,  222,  223,  224,  227,  229,  231,  232,  233,  243,  246,  247,
                        253,  255,  256,  261,  263,  266,  267,  271,  273,  274,  277,  278,
                        282,  284,  285,  289,  291,  293,  296,  304,  308,  311,  313,  314,
                        318,  324,  330,  331,  332,  336,  337,  338,  339,  340,  342,  345,
                        347,  348,  354,  362,  363,  366,  372,  373,  374,  375,  379,  380,
                        383,  386,  390,  395,  397,  398,  399,  400,  404,  408,  411,  414,
                        418,  423,  424,  425,  430,  431,  439,  442,  444,  447,  448,  449,
                        452,  456,  459,  460,  462,  465,  467,  469,  470,  471,  473,  478,
                        480,  481,  483,  486,  489,  490,  491,  493,  496,  498,  499,  500,
                        502,  504,  506,  510,  512,  518,  519,  521,  522,  527,  528,  531,
                        532,  534,  535,  536,  538,  539,  543,  546,  548,  556,  560,  562,
                        564,  565,  566,  569,  573,  575,  582,  587,  588,  590,  594,  597,
                        598,  603,  606,  607,  610,  613,  617,  619,  621,  622,  632,  634,
                        639,  643,  646,  647,  656,  659,  660,  661,  666,  667,  669,  674,
                        677,  678,  680,  681,  684,  688,  691,  692,  693,  694,  700,  704,
                        705,  706,  708,  718,  730,  732,  736,  737,  739,  742,  743,  747,
                        749,  750,  751,  752,  753,  755,  756,  757,  759,  761,  762,  765,
                        773,  775,  779,  780,  785,  786,  787,  788,  791,  793,  797,  800,
                        801,  802,  809,  810,  813,  814,  819,  831,  834,  837,  838,  843,
                        847,  848,  853,  854,  855,  857,  859,  860,  867,  874,  876,  878,
                        879,  880,  881,  883,  884,  885,  887,  888,  890,  891,  896,  902,
                        903,  906,  908,  910,  914,  916,  917,  935,  937,  938,  941,  942,
                        943,  947,  949,  952,  957,  966,  967,  970,  971,  977,  980,  986,
                        987,  989,  993,  994, 1002, 1004, 1006, 1007, 1014, 1016, 1018, 1019,
                        1021, 1024, 1030, 1031, 1035, 1040, 1051, 1052, 1054, 1057, 1058, 1059,
                        1060, 1063, 1065, 1070, 1078, 1080, 1081, 1082, 1084, 1085, 1087, 1088,
                        1089, 1092, 1095, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109,
                        1113, 1115, 1119, 1120, 1123, 1124, 1125, 1126, 1130, 1133, 1141, 1145,
                        1147, 1149, 1151, 1155, 1157, 1159, 1160, 1162, 1164, 1170, 1172, 1173,
                        1174, 1175, 1177, 1181, 1184, 1185, 1190, 1191, 1192, 1193, 1194, 1196,
                        1197, 1198, 1201, 1202, 1206, 1207, 1208, 1209, 1211, 1216, 1217, 1219,
                        1220, 1221, 1222, 1226, 1229], device=device)
        frequent = torch.tensor([2,    3,   11,   12,   16,   19,   23,   29,   31,   33,   34,   35,
                40,   42,   44,   49,   52,   55,   56,   57,   58,   59,   60,   65,
                66,   68,   76,   77,   78,   80,   83,   86,   87,   88,   89,   90,
                94,   95,   98,  103,  109,  111,  117,  123,  125,  131,  132,  136,
                137,  138,  145,  153,  159,  169,  171,  173,  178,  179,  182,  184,
                186,  190,  193,  196,  197,  207,  210,  211,  220,  228,  234,  236,
                239,  259,  260,  262,  268,  275,  276,  280,  281,  287,  288,  298,
                300,  301,  307,  322,  326,  333,  344,  346,  351,  352,  361,  364,
                370,  376,  381,  382,  389,  391,  394,  396,  406,  409,  415,  417,
                421,  427,  428,  435,  436,  443,  446,  450,  451,  457,  458,  464,
                466,  474,  479,  501,  503,  505,  507,  516,  530,  537,  545,  549,
                553,  557,  558,  559,  567,  570,  574,  576,  577,  578,  579,  600,
                602,  605,  614,  618,  620,  623,  624,  627,  631,  636,  637,  638,
                640,  641,  648,  650,  651,  652,  653,  654,  665,  668,  671,  672,
                679,  686,  687,  690,  696,  698,  701,  703,  707,  709,  710,  713,
                714,  716,  717,  719,  720,  725,  727,  728,  735,  745,  746,  760,
                763,  769,  770,  778,  783,  792,  804,  811,  812,  816,  818,  823,
                826,  828,  829,  830,  839,  841,  845,  849,  850,  851,  852,  858,
                861,  873,  889,  894,  895,  899,  909,  911,  912,  913,  923,  924,
                925,  928,  929,  932,  934,  936,  960,  964,  965,  968,  972,  973,
                976,  978,  979,  981,  982,  983,  984,  985,  992,  995,  996,  997,
                1000, 1011, 1013, 1017, 1023, 1027, 1034, 1036, 1037, 1038, 1039, 1041,
                1042, 1043, 1044, 1045, 1053, 1056, 1061, 1062, 1064, 1069, 1071, 1074,
                1075, 1079, 1083, 1090, 1091, 1093, 1096, 1097, 1098, 1110, 1112, 1114,
                1116, 1117, 1118, 1121, 1122, 1127, 1128, 1129, 1131, 1134, 1136, 1142,
                1143, 1154, 1161, 1163, 1176, 1182, 1183, 1195, 1199, 1200, 1210, 1212,
                1214, 1223, 1228], device=device)
        return rare, common, frequent