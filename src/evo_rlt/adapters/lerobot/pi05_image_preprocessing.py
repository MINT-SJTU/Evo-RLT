from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def resize_with_pad_pil_uint8(
    images: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Match the OpenPI training resize path used for PI0.5 images.

    OpenPI converts float LeRobot images to uint8, resizes each HWC image with
    PIL bilinear interpolation, pads in uint8 space, and only then converts
    back to model floats. Keep that ordering here so deployment sees the same
    pixels as training.
    """
    channels_last = images.shape[-1] <= 4
    batch_shape = images.shape[:-3]
    if channels_last:
        current_height, current_width, channels = images.shape[-3:]
        images_nhwc = images
    else:
        channels, current_height, current_width = images.shape[-3:]
        images_nhwc = images.movedim(-3, -1)

    if (current_height, current_width) == (height, width):
        return images

    source_device = images.device
    source_dtype = images.dtype
    source_is_float = torch.is_floating_point(images)

    cpu_images = images_nhwc.detach().to(device="cpu")
    if source_is_float:
        cpu_images = cpu_images.to(dtype=torch.float32)
    image_array = cpu_images.reshape(-1, current_height, current_width, channels).contiguous().numpy()
    if source_is_float:
        image_array = (image_array * 255.0).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8, copy=False)

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))

    resized_batch = []
    for image_array_item in image_array:
        image = Image.fromarray(image_array_item)
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        padded = Image.new(resized.mode, (width, height), 0)
        padded.paste(resized, (pad_width, pad_height))
        resized_batch.append(np.asarray(padded))

    output = torch.from_numpy(np.stack(resized_batch))
    if source_is_float:
        output = output.to(dtype=torch.float32).div_(255.0).to(dtype=source_dtype)
    else:
        output = output.to(dtype=source_dtype)
    output = output.to(device=source_device)
    output = output.reshape(*batch_shape, height, width, channels)
    if not channels_last:
        output = output.movedim(-1, -3)
    return output


def install_pi05_openpi_resize() -> None:
    """Install the OpenPI-equivalent resize in LeRobot's PI0.5 module."""
    import lerobot.policies.pi05.modeling_pi05 as modeling_pi05

    modeling_pi05.resize_with_pad_torch = resize_with_pad_pil_uint8
