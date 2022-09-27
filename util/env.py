# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Set up Environment."""

from iopath.common.file_io import PathManagerFactory

_ENV_SETUP_DONE = False
pathmgr = PathManagerFactory.get(key="mae_st")
checkpoint_pathmgr = PathManagerFactory.get(key="mae_st_checkpoint")


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
