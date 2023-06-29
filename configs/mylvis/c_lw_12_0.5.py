_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v0.5_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='FcrCascadeRoIHead',
        labels_tail=[0, 6, 9, 13, 14, 15, 20, 21, 30, 37, 38, 39,41, 45, 48, 50, 51, 63, 64, 69, 71, 73, 82, 85, 93, 99, 100, 104, 105, 106, 112, 115, 116, 119, 121, 124,126, 129, 130, 135, 139, 141, 142, 143, 146, 149, 154, 158,160, 162, 163, 166, 168, 172, 180, 181, 183, 195, 198, 202,
                     204, 205, 208, 212, 213, 216, 217, 218, 225, 226, 230, 235,237, 238, 240, 241, 242, 244, 245, 248, 249, 250, 251, 252,254, 257, 258, 264, 265, 269, 270, 272, 279, 283, 286, 290,292, 294, 295, 297, 299, 302, 303, 305, 306, 309, 310, 312,315, 316, 317, 319, 320, 321, 323, 325, 327, 328, 329, 334,
                     335, 341, 343, 349, 350, 353, 355, 356, 357, 358, 359, 360,365, 367, 368, 369, 371, 377, 378, 384, 385, 387, 388, 392,393, 401, 402, 403, 405, 407, 410, 412, 413, 416, 419, 420,422, 426, 429, 432, 433, 434, 437, 438, 440, 441, 445, 453,454, 455, 461, 463, 468, 472, 475, 476, 477, 482, 484, 485,
                     487, 488, 492, 494, 495, 497, 508, 509, 511, 513, 514, 515,517, 520, 523, 524, 525, 526, 529, 533, 540, 541, 542, 544,547, 550, 551, 552, 554, 555, 561, 563, 568, 571, 572, 580,581, 583, 584, 585, 586, 589, 591, 592, 593, 595, 596, 599,601, 604, 608, 609, 611, 612, 615, 616, 625, 626, 628, 629,
                     630, 633, 635, 642, 644, 645, 649, 655, 657, 658, 662, 663,664, 670, 673, 675, 676, 682, 683, 685, 689, 695, 697, 699,702, 711, 712, 715, 721, 722, 723, 724, 726, 729, 731, 733,734, 738, 740, 741, 744, 748, 754, 758, 764, 766, 767, 768,771, 772, 774, 776, 777, 781, 782, 784, 789, 790, 794, 795,
                     796, 798, 799, 803, 805, 806, 807, 808, 815, 817, 820, 821,822, 824, 825, 827, 832, 833, 835, 836, 840, 842, 844, 846,856, 862, 863, 864, 865, 866, 868, 869, 870, 871, 872, 875,877, 882, 886, 892, 893, 897, 898, 900, 901, 904, 905, 907,915, 918, 919, 920, 921, 922, 926, 927, 930, 931, 933, 939,940, 944, 945, 946, 948, 950, 951, 953, 954, 955, 956, 958,
                     959, 961, 962, 963, 969, 974, 975, 988, 990, 991, 998, 999,1001, 1003, 1005, 1008, 1009, 1010, 1012, 1015, 1020, 1022, 1025, 1026,1028, 1029, 1032, 1033, 1046, 1047, 1048, 1049, 1050, 1055, 1066, 1067,1068, 1072, 1073, 1076, 1077, 1086, 1094, 1099, 1103, 1111, 1132, 1135,1137, 1138, 1139, 1140, 1144, 1146, 1148, 1150, 1152, 1153, 1156, 1158,1165, 1166, 1167, 1168, 1169, 1171, 1178, 1179, 1180, 1186, 1187, 1188,1189, 1203, 1204, 1205, 1213, 1215, 1218, 1224, 1225, 1227],
        labels_center=[1, 4, 5, 7, 8, 10, 17, 18, 22, 24, 25, 26,27, 28, 32, 36, 43, 46, 47, 53, 54, 61, 62, 67,70, 72, 74, 75, 79, 81, 84, 91, 92, 96, 97, 101,102, 107, 108, 110, 113, 114, 118, 120, 122, 127, 128, 133,
                       134, 140, 144, 147, 148, 150, 151, 152, 155, 156, 157, 161,164, 165, 167, 170, 174, 175, 176, 177, 185, 187, 188, 189,191, 192, 194, 199, 200, 201, 203, 206, 209, 214, 215, 219,
                       221, 222, 223, 224, 227, 229, 231, 232, 233, 243, 246, 247,253, 255, 256, 261, 263, 266, 267, 271, 273, 274, 277, 278,282, 284, 285, 289, 291, 293, 296, 304, 308, 311, 313, 314,
                       318, 324, 330, 331, 332, 336, 337, 338, 339, 340, 342, 345,347, 348, 354, 362, 363, 366, 372, 373, 374, 375, 379, 380,383, 386, 390, 395, 397, 398, 399, 400, 404, 408, 411, 414,
                       418, 423, 424, 425, 430, 431, 439, 442, 444, 447, 448, 449,452, 456, 459, 460, 462, 465, 467, 469, 470, 471, 473, 478,480, 481, 483, 486, 489, 490, 491, 493, 496, 498, 499, 500,
                       502, 504, 506, 510, 512, 518, 519, 521, 522, 527, 528, 531,532, 534, 535, 536, 538, 539, 543, 546, 548, 556, 560, 562,564, 565, 566, 569, 573, 575, 582, 587, 588, 590, 594, 597,
                       598, 603, 606, 607, 610, 613, 617, 619, 621, 622, 632, 634,639, 643, 646, 647, 656, 659, 660, 661, 666, 667, 669, 674,677, 678, 680, 681, 684, 688, 691, 692, 693, 694, 700, 704,
                       705, 706, 708, 718, 730, 732, 736, 737, 739, 742, 743, 747,749, 750, 751, 752, 753, 755, 756, 757, 759, 761, 762, 765,773, 775, 779, 780, 785, 786, 787, 788, 791, 793, 797, 800,
                       801, 802, 809, 810, 813, 814, 819, 831, 834, 837, 838, 843,847, 848, 853, 854, 855, 857, 859, 860, 867, 874, 876, 878,879, 880, 881, 883, 884, 885, 887, 888, 890, 891, 896, 902,
                       903, 906, 908, 910, 914, 916, 917, 935, 937, 938, 941, 942,943, 947, 949, 952, 957, 966, 967, 970, 971, 977, 980, 986,987, 989, 993, 994, 1002, 1004, 1006, 1007, 1014, 1016, 1018, 1019,
                       1021, 1024, 1030, 1031, 1035, 1040, 1051, 1052, 1054, 1057, 1058, 1059,1060, 1063, 1065, 1070, 1078, 1080, 1081, 1082, 1084, 1085, 1087, 1088,1089, 1092, 1095, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109,
                       1113, 1115, 1119, 1120, 1123, 1124, 1125, 1126, 1130, 1133, 1141, 1145,1147, 1149, 1151, 1155, 1157, 1159, 1160, 1162, 1164, 1170, 1172, 1173,1174, 1175, 1177, 1181, 1184, 1185, 1190, 1191, 1192, 1193, 1194, 1196,
                       1197, 1198, 1201, 1202, 1206, 1207, 1208, 1209, 1211, 1216, 1217, 1219,1220, 1221, 1222, 1226, 1229],
        labels=[ 2,    3,   11,   12,   16,   19,   23,   29,   31,   33,   34,   35,40,   42,   44,   49,   52,   55,   56,   57,   58,   59,   60,   65,66,   68,   76,   77,   78,   80,   83,   86,   87,   88,   89,   90,
                94,   95,   98,  103,  109,  111,  117,  123,  125,  131,  132,  136,137,  138,  145,  153,  159,  169,  171,  173,  178,  179,  182,  184,186,  190,  193,  196,  197,  207,  210,  211,  220,  228,  234,  236,
                239,  259,  260,  262,  268,  275,  276,  280,  281,  287,  288,  298,300,  301,  307,  322,  326,  333,  344,  346,  351,  352,  361,  364,370,  376,  381,  382,  389,  391,  394,  396,  406,  409,  415,  417,
                421,  427,  428,  435,  436,  443,  446,  450,  451,  457,  458,  464,466,  474,  479,  501,  503,  505,  507,  516,  530,  537,  545,  549,553,  557,  558,  559,  567,  570,  574,  576,  577,  578,  579,  600,
                602,  605,  614,  618,  620,  623,  624,  627,  631,  636,  637,  638,640,  641,  648,  650,  651,  652,  653,  654,  665,  668,  671,  672,679,  686,  687,  690,  696,  698,  701,  703,  707,  709,  710,  713,
                714,  716,  717,  719,  720,  725,  727,  728,  735,  745,  746,  760,763,  769,  770,  778,  783,  792,  804,  811,  812,  816,  818,  823,826,  828,  829,  830,  839,  841,  845,  849,  850,  851,  852,  858,
                861,  873,  889,  894,  895,  899,  909,  911,  912,  913,  923,  924,925,  928,  929,  932,  934,  936,  960,  964,  965,  968,  972,  973,976,  978,  979,  981,  982,  983,  984,  985,  992,  995,  996,  997,
                1000, 1011, 1013, 1017, 1023, 1027, 1034, 1036, 1037, 1038, 1039, 1041,1042, 1043, 1044, 1045, 1053, 1056, 1061, 1062, 1064, 1069, 1071, 1074,1075, 1079, 1083, 1090, 1091, 1093, 1096, 1097, 1098, 1110, 1112, 1114,
                1116, 1117, 1118, 1121, 1122, 1127, 1128, 1129, 1131, 1134, 1136, 1142,1143, 1154, 1161, 1163, 1176, 1182, 1183, 1195, 1199, 1200, 1210, 1212, 1214, 1223, 1228],

        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=0.75),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,loss_weight=0.75)),
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=0.75),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,loss_weight=0.75)),
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=0.75),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.75))],
        bbox_head_center=[
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))],
        bbox_head_tail=[
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.25),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.25)),
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.25),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.25)),
            dict(
                type='FcrBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1230,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.25),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.25))
        ]
    ))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner_tail=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_tail=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
           assigner_center=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_center=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),

        dict(
            assigner_tail=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_tail=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            assigner_center=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_center=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner_tail=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_tail=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            assigner_center=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_center=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ])

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300))
# dataset settings
dataset_type = 'LVISV05Dataset'
data_root = 'data/lvis_v0.5/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(samples_per_gpu=8,train=dict(dataset=dict(pipeline=train_pipeline)))
evaluation = dict(interval=6, metric='bbox')
#optimizer = dict(type='SGD', lr=0.002, momentum=0.95, weight_decay=5 * 1e-4)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
#load_from = None
load_from='./work_dirs/pth/c_epoch_24.pth'
selectp = 0