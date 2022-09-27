#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import math

# import cv2
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from .rand_augment import rand_augment_transform

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


def random_short_side_scale_jitter(
    images, min_size, max_size, inverse_uniform_sampling=False
):
    """
    Perform a spatial short scale jittering on the given images.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale, max_scale].
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
    """
    if inverse_uniform_sampling:
        size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (height <= width and height == size):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def random_crop(images, size):
    """
    Perform random spatial crop on the given images.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    return cropped


def horizontal_flip(prob, images):
    """
    Perform horizontal flip on the given images.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
    """
    if np.random.uniform() < prob:
        images = images.flip((-1))
    return images


def uniform_crop(images, size, spatial_idx, scale_size=None):
    """
    Perform uniform spatial sampling on the images.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped


def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    return images1 * alpha + images2 * (1 - alpha)


def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = torch.tensor(images)
    gray_channel = 0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given images.
    Args:
        images (tensor): images to perform lighting jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    if alphastd == 0:
        return images
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = torch.zeros_like(images)
    if len(images.shape) == 3:
        # C H W
        channel_dim = 0
    elif len(images.shape) == 4:
        # T C H W
        channel_dim = 1
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    for idx in range(images.shape[channel_dim]):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = images[idx] + rgb[2 - idx]
        # T C H W
        elif len(images.shape) == 4:
            out_images[:, idx] = images[:, idx] + rgb[2 - idx]
        else:
            raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    return out_images


def color_normalization(images, mean, stddev):
    """
    Perform color nomration on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    if len(images.shape) == 3:
        assert len(mean) == images.shape[0], "channel mean not computed properly"
        assert len(stddev) == images.shape[0], "channel stddev not computed properly"
    elif len(images.shape) == 4:
        assert len(mean) == images.shape[1], "channel mean not computed properly"
        assert len(stddev) == images.shape[1], "channel stddev not computed properly"
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = (images[idx] - mean[idx]) / stddev[idx]
        elif len(images.shape) == 4:
            out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]
        else:
            raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")
    return out_images


def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def random_resized_crop_with_shift(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    This is similar to random_resized_crop. However, it samples two different
    boxes (for cropping) for the first and last frame. It then linearly
    interpolates the two boxes for other frames.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """
    t = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    out = torch.zeros((3, t, target_height, target_width))
    for ind in range(t):
        out[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            images[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return out


def create_random_augment(
    input_size,
    auto_augment=None,
    interpolation="bilinear",
):
    """
    Get video randaug transform.

    Args:
        input_size: The size of the input video in tuple.
        auto_augment: Parameters for randaug. An example:
            "rand-m7-n4-mstd0.5-inc1" (m is the magnitude and n is the number
            of operations to apply).
        interpolation: Interpolation method.
    """
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = {"translate_const": int(img_size_min * 0.45)}
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            return transforms.Compose([rand_augment_transform(auto_augment, aa_params)])
    raise NotImplementedError


def random_sized_crop_img(
    im,
    size,
    jitter_scale=(0.08, 1.0),
    jitter_aspect=(3.0 / 4.0, 4.0 / 3.0),
    max_iter=10,
):
    """
    Performs Inception-style cropping (used for training).
    """
    assert len(im.shape) == 3, "Currently only support image for random_sized_crop"
    h, w = im.shape[1:3]
    i, j, h, w = _get_param_spatial_crop(
        scale=jitter_scale,
        ratio=jitter_aspect,
        height=h,
        width=w,
        num_repeat=max_iter,
        log_scale=False,
        switch_hw=True,
    )
    cropped = im[:, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


# The following code are modified based on timm lib, we will replace the following
# contents with dependency from PyTorchVideo.
# https://github.com/facebookresearch/pytorchvideo
class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string
