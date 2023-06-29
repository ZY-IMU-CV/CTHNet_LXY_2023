import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        #'--validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def select_training_param(model):

    for v in model.parameters():
        v.requires_grad = False
    print("--------------------------------------------enter the train cls part only------------------------------------")
    model.roi_head.bbox_head.fc_cls.weight.requires_grad = True
    model.roi_head.bbox_head.fc_cls.bias.requires_grad = True
    model.roi_head.bbox_head_center.fc_cls.weight.requires_grad = True
    model.roi_head.bbox_head_center.fc_cls.bias.requires_grad = True
    model.roi_head.bbox_head_tail.fc_cls.weight.requires_grad = True
    model.roi_head.bbox_head_tail.fc_cls.bias.requires_grad = True
    return model


def select_training_param_tree(model):

    for v in model.parameters():
        v.requires_grad = False
    print("--------------------------------------------enter the train cls part only------------------------------------")
    model.roi_head.bbox_head.fc_cls.weight.requires_grad = True
    model.roi_head.bbox_head.fc_cls.bias.requires_grad = True
    model.roi_head.bbox_head_center.fc_cls.weight.requires_grad = True
    model.roi_head.bbox_head_center.fc_cls.bias.requires_grad = True
    model.roi_head.bbox_head_tail.fc_cls.weight.requires_grad = True
    model.roi_head.bbox_head_tail.fc_cls.bias.requires_grad = True
    return model


def select_one_stage_head(model):
    for v in model.parameters():
        v.requires_grad = False
    print("----------------------enter retina_cls part---------------------------------------")
    #model.bbox_head.retina_cls.requires_grad = True
    model.bbox_head.retina_cls.weight.requires_grad = True
    model.bbox_head.retina_cls.bias.requires_grad = True
    # model.bbox_head.retina_cls_center.weight.requires_grad = True
    # model.bbox_head.retina_cls_center.bias.requires_grad = True
    # model.bbox_head.retina_cls_tail.weight.requires_grad = True
    # model.bbox_head.retina_cls_tail.bias.requires_grad = True

    return model
# def select_head(model):
#     for v in model.parameters():
#         v.requires_grad = False
#
#     for v in model.roi_head.bbox_head.parameters():
#         v.requires_grad = True
#
#     return model

def select_cascade_cls_params(model):

    for v in model.parameters():
        v.requires_grad = False
    print("--------------------------------------------enter the train C_RCNN cls part only------------------------------------")
    for child in model.roi_head.bbox_head.children():
        for v in child.fc_cls.parameters():
            print("//////////////////bbox_head///////////////////")
            v.requires_grad = True
    for child in model.roi_head.bbox_head_center.children():
        for v in child.fc_cls.parameters():
            print("//////////////////bbox_head_center///////////////////")
            v.requires_grad = True
    for child in model.roi_head.bbox_head_tail.children():
        for v in child.fc_cls.parameters():
            print("//////////////////bbox_head_tail///////////////////")
            v.requires_grad = True

    return model

# def select_mask_params(model):
#
#     for v in model.parameters():
#         v.requires_grad = False
#
#     for v in model.bbox_head.parameters():
#         v.requires_grad = True
#     for v in model.mask_head.parameters():
#         v.requires_grad = True

    # return model

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    tune_part = cfg.get('selectp', 0)
    if tune_part == 1:
        print('Train fc_cls only.')
        model = select_training_param(model)
    elif tune_part == 5:
        print('Train fc_cls only.')
        model = select_training_param_tree(model)
    elif tune_part == 2:
        print('Train one stage cls_head only.')
        model = select_one_stage_head(model)
    elif tune_part == 3:
        print('Train cascade fc_cls only.')
        model = select_cascade_cls_params(model)
    # elif tune_part == 4:
    #     print('Train bbox and mask head.')
    #     model = select_mask_params(model)
    else:
        print('Train all params.')
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
