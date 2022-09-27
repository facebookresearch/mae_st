# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import av


def get_video_container(path_to_vid, multi_thread_decode=False):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    with open(path_to_vid, "rb") as fp:
        container = fp.read()
    return container
