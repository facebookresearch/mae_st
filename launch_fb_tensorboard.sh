# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

#!/bin/sh


# buck build  @mode/opt //tensorboard
if [ "$1" -lt 1 ]
then
  ~/local/fbsource/fbcode/buck-out/gen/tensorboard/tensorboard.par --port=8092 --logdir=manifold://winvision/tree/haoqifan/logs/tensorboard/pretrain
else
  ~/local/fbsource/fbcode/buck-out/gen/tensorboard/tensorboard.par --port=8095 --logdir=manifold://winvision/tree/haoqifan/logs/tensorboard/downstream
fi

# ~/local/fbsource/fbcode/buck-out/gen/tensorboard/tensorboard.par --port=8095 --logdir_spec=fair_logging:manifold://fair_logging/tree/haoqifan/logs/tensorboard/downstream,winvision:manifold://winvision/tree/haoqifan/logs/tensorboard/downstream
