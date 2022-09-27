# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

#!/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Need at least 1 parameter to determine number of machines"
  exit
fi

CHECKPOINT_PATH=manifold://winvision/tree/${USER}/logs/$(date -d "${start} + 1 day" +%F-%H%M%S-%3N)
echo "${CHECKPOINT_PATH}"

manifold mkdirs "${CHECKPOINT_PATH#"manifold://"}"
manifold mkdirs "${CHECKPOINT_PATH#"manifold://"}""/pretrain"
manifold mkdirs "${CHECKPOINT_PATH#"manifold://"}""/downstream"

GANG_SCHEDULE=${GANG_SCHEDULE-1}
GANG_AFFINITY=${GANG_AFFINITY-0}
GPU_TYPE=${GPU_TYPE-3}
POSTFIX=${POSTFIX-"benchmark"}
ENT=${ENT-"default_ncg"}
RUN_BENCHMARK=${RUN_BENCHMARK-0}

if [ "$1" -lt 1 ]
then
  FINETUNE_APPENDIX=" --finetune "${4}
else
  FINETUNE_APPENDIX=""
fi
if [ "$2" -lt 1 ]
then
  TESTING_APPENDIX=" --finetune "${4}
else
  TESTING_APPENDIX=""
fi

P_CONFIG=${P_CONFIG-"--path_to_data_dir manifold://fair_vision_data/tree/PySlowFast/kinetics/k400 --batch_size 2 --model mae_vit_large_patch16 --no_env --epochs 100 --distributed --num_frames 16 --decoder_embed_dim 512 --decoder_depth 4 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.6e-3 --warmup_epochs 20 --mask_ratio 0.9 --cls_embed --pred_t_dim 8 --fp32 --sep_pos_embed --clip_grad 0.02"}

D_CONFIG=${D_CONFIG-"--path_to_data_dir manifold://fair_vision_data/tree/PySlowFast/kinetics/k400 --rand_aug --epochs 50 --no_env --repeat_aug 2 --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --mixup_prob 1.0 --blr 0.0024 --num_frames 16 --pin_mem --num_workers 12 --sampling_rate 4 --dropout 0.3 --warmup_epochs 5 --layer_decay 0.75 --drop_path_rate 0.2 --aa rand-m7-mstd0.5-inc1 --cls_embed --sep_pos_embed --clip_grad 5.0 --fp32"}${FINETUNE_APPENDIX}

T_CONFIG=${T_CONFIG-"--path_to_data_dir manifold://fair_vision_data/tree/PySlowFast/kinetics/k400 --no_env --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --num_frames 16 --pin_mem --num_workers 12 --sampling_rate 4 --dropout 0.3 --cls_embed --sep_pos_embed --fp32"}${TESTING_APPENDIX}


flow-cli canary mae_st.mae_st.workflow@//fblearner/flow/projects/mae_st:workflow \
--parameters-json '{
    "num_shard_pretrain": '"${1}"',
    "num_shard_finetune": '"${2}"',
    "num_shard_test": '"${3}"',
    "pretrain_config": "'"${P_CONFIG}"'",
    "downstream_config": "'"${D_CONFIG}"'",
    "test_config": "'"${T_CONFIG}"'",
    "output_dir": "'"${CHECKPOINT_PATH}"'",
    "gang_schedule": "'"${GANG_SCHEDULE}"'",
    "gang_affinity": "'"${GANG_AFFINITY}"'",
    "gpu_type": "'"${GPU_TYPE}"'",
    "entitlement": "'"${ENT}"'"}' \
--entitlement "default_ncg" \
--run-as-secure-group "${SECURE_GROUP-vidron}" \
--name "${POSTFIX}||${P_CONFIG}||${1}nodes" \
--mode opt \
