# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import mae_st.util.misc as misc
import torch


@torch.no_grad()
def test(data_loader, model, device, test_meter, fp32=False):
    metric_logger = misc.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    softmax = torch.nn.Softmax(dim=1).cuda()

    for cur_iter, (images, labels, video_idx) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_idx = video_idx.to(device, non_blocking=True)

        if len(images.shape) == 6:
            b, r, c, t, h, w = images.shape
            images = images.view(b * r, c, t, h, w)
            labels = labels.view(b * r)

        # compute output
        with torch.cuda.amp.autocast(enabled=not fp32):
            preds = model(images)
            preds = softmax(preds)

        if torch.distributed.is_initialized():
            preds, labels, video_idx = misc.all_gather([preds, labels, video_idx])
        preds = preds.cpu()
        labels = labels.cpu()
        video_idx = video_idx.cpu()
        # Update and log stats.
        test_meter.update_stats(preds.detach(), labels.detach(), video_idx.detach())
        test_meter.log_iter_stats(cur_iter)

    test_meter.finalize_metrics()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return test_meter.stats
