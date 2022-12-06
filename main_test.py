# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import os

import mae_st.models_vit as models_vit
import mae_st.util.misc as misc

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from mae_st.engine_test import test
from mae_st.util.kinetics import Kinetics
from mae_st.util.logging import master_print as print
from mae_st.util.meters import TestMeter
from mae_st.util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument(
        "--nb_classes",
        default=400,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument(
        "--path_to_data_dir",
        default="",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # Video related configs
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--rand_aug", default=False, action="store_true")
    parser.add_argument("--t_patch_size", default=4, type=int)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--checkpoint_period", default=10, type=int)
    parser.add_argument("--sampling_rate", default=2, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument("--encoder_attn", default="AttentionSubsampleMaxpool", type=str)

    # Dataset parameters
    parser.add_argument(
        "--decoder_device",
        default="cpu",
        type=str,
    )
    # Dataset parameters
    parser.add_argument(
        "--decoder_backend",
        default="torchvision",
        type=str,
    )
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.add_argument("--cls_embed", action="store_true")
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = Kinetics(
        mode="test",
        path_to_data_dir=args.path_to_data_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        train_jitter_scales=(256, 320),
        test_crop_size=224,
        repeat_aug=args.repeat_aug,
        rand_aug=False,
    )
    test_meter = TestMeter(
        dataset_test.num_videos // (3 * 10),
        3 * 10,
        args.nb_classes,
        len(dataset_test),
        False,
        "sum",
    )

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        num_tasks = 1
        global_rank = 0
        sampler_test = torch.utils.data.RandomSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        **vars(args),
    )

    with pathmgr.open(args.finetune, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    checkpoint_model = misc.convert_checkpoint(checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )
        model_without_ddp = model.module

    log_stats = test(data_loader_test, model, device, test_meter, fp32=args.fp32)
    return [log_stats]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)
