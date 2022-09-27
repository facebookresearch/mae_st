# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

#!/usr/bin/env bash

# sudo fuser -v /dev/nvidia* | grep -o '[[:digit:]]*' |xargs -I{} sudo kill -9 {}

# buck build  --config client.skip-action-graph-cache=true @mode/opt -c python.native_link_strategy=separate \
buck build  @mode/opt @mode/inplace \
  //vision/fair/mae_st/... --show-output

# 0: pretrain, 1: finetune, 2: test

if [ "$1" -lt 1 ]
then

  echo "pretrain"

  /data/users/"${USER}"/fbsource/fbcode/buck-out/gen/vision/fair/mae_st/run_pretrain_bin.par \
  --path_to_data_dir manifold://fair_vision_data/tree/PySlowFast/kinetics/k400 \
  --batch_size 1 --decoder_embed_dim 64 --decoder_depth 2 \
  --epochs 1 --mask_ratio 0.9 --repeat_aug 2 \
  --model mae_vit_large_patch16 \
  --sampling_rate 4 --num_frames 16 \
  --num_workers 2 \
  --bias_wd \
  --trunc_init \
  --fp32 \
  --jitter_aspect_relative 0.75 1.3333 --jitter_scales_relative 0.5 1.0 \
  --cls_embed \
  --sep_pos_embed \
  --t_patch_size 2 \

  --resume manifold://winvision/tree/feichtenhofer/logs/2022-04-07-104623-303/checkpoints/ssl_eval_checkpoint_epoch_00050.pyth \

  --resume manifold://winvision/tree/feichtenhofer/logs/2022-04-05-150122-992/checkpoints/ssl_eval_checkpoint_epoch_00050.pyth \

  --learnable_pos_embed \

  --decoder_attn AttentionRelPos --encoder_attn AttentionRelPos --rel_pos_embed \

else

  if [ "$1" -lt 2 ]
  then

    echo "finetune"

    # AttentionSubsampleMaxpool, AttentionSubsampleStride2, AttentionSubsampleRand10, AttentionSubsampleRand25, AttentionSubsampleRand50,
    /data/users/"${USER}"/fbsource/fbcode/buck-out/gen/vision/fair/mae_st/run_finetune_bin.par \
    --batch_size 1 --epochs 1 --repeat_aug 1 --smoothing 0.1 \
    --mixup 0.0 --cutmix 0.0 --mixup_prob 0.0 \
    --model vit_large_patch16 \
    --t_patch_size 2 --num_frames 16 \
    --rand_aug \
    --sep_pos_embed \
    --fp32 \
    --cls_embed \
    --finetune manifold://winvision/tree/haoqifan/logs/2022-05-17-131324-457/pretrain/checkpoint-00049.pth \

    --finetune manifold://winvision/tree/feichtenhofer/logs/2022-04-07-104623-303/checkpoints/ssl_checkpoint_epoch_00050.pyth \





    --encoder_attn AttentionOrg \

    --finetune checkpoint-00000.pth \


    --finetune manifold://winvision/tree/feichtenhofer/logs/2022-04-07-104623-303/checkpoints/ssl_eval_checkpoint_epoch_00050.pyth \


    --finetune manifold://winvision/tree/feichtenhofer/logs/2022-04-05-150122-992/checkpoints/ssl_eval_checkpoint_epoch_00050.pyth \

    --finetune manifold://winvision/tree/lyttonhao/mae_pretrain/Supin1k-ViT-Large-200ep-km_ema_wkbias.pth \

    --encoder_attn AttentionRelPosWithCls \
    --finetune mae_pretrain_vit_large.pth \
    --no_qkv_bias \
    --finetune manifold://winvision/tree/haoqifan/logs/2022-02-05-204420-480/pretrain/checkpoint-00399.pth \

    # --no_qkv_bias

    # --encoder_attn AttentionSubsampleRand10 \

    # --finetune manifold://fair_logging/tree/haoqifan/logs/2022-01-17-162701-592/pretrain/checkpoint-399.pth

  else

    echo "test"

    # AttentionSubsampleMaxpool, AttentionSubsampleStride2, AttentionSubsampleRand10, AttentionSubsampleRand25, AttentionSubsampleRand50,
    /data/users/"${USER}"/fbsource/fbcode/buck-out/gen/vision/fair/mae_st/run_test_bin.par \
    --batch_size 2
    --model vit_large_patch16 \
    --t_patch_size 2 --num_frames 16 \
    --cls_embed --sep_pos_embed
    --finetune manifold://winvision/tree/feichtenhofer/logs/2022-04-05-150122-992/checkpoints/ssl_eval_checkpoint_epoch_00050.pyth \

    --finetune manifold://fair_logging/tree/haoqifan/logs/2022-01-25-012936-625/downstream/checkpoint-99.pth

    # --finetune manifold://fair_logging/tree/haoqifan/logs/2022-01-17-162701-592/pretrain/checkpoint-399.pth

  fi
fi
