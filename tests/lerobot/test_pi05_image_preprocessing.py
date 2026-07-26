from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from evo_rlt.adapters.lerobot.pi05_image_preprocessing import (
    install_pi05_openpi_resize,
    resize_with_pad_pil_uint8,
)


def _openpi_training_reference(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    channels_last = images.shape[-1] <= 4
    if channels_last:
        batch = images
    else:
        batch = images.movedim(-3, -1)

    batch_shape = batch.shape[:-3]
    current_height, current_width, channels = batch.shape[-3:]
    array = batch.reshape(-1, current_height, current_width, channels).numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = (array * 255.0).astype(np.uint8)

    output = []
    for item in array:
        image = Image.fromarray(item)
        ratio = max(current_width / width, current_height / height)
        resized_height = int(current_height / ratio)
        resized_width = int(current_width / ratio)
        resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
        padded = Image.new(resized.mode, (width, height), 0)
        padded.paste(
            resized,
            (
                max(0, int((width - resized_width) / 2)),
                max(0, int((height - resized_height) / 2)),
            ),
        )
        output.append(np.asarray(padded))

    result = torch.from_numpy(np.stack(output)).to(dtype=torch.float32).div_(255.0)
    result = result.reshape(*batch_shape, height, width, channels)
    if not channels_last:
        result = result.movedim(-1, -3)
    return result


def _batched_images(batch_size: int, channels_last: bool) -> torch.Tensor:
    values = torch.arange(batch_size * 3 * 9 * 13, dtype=torch.int64)
    images = values.remainder(256).to(dtype=torch.float32).div_(255.0)
    images = images.reshape(batch_size, 3, 9, 13)
    return images.movedim(1, -1) if channels_last else images


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("channels_last", [False, True])
def test_resize_matches_openpi_pil_uint8_for_real_batches(batch_size: int, channels_last: bool):
    images = _batched_images(batch_size, channels_last)

    actual = resize_with_pad_pil_uint8(images, 8, 8)
    expected = _openpi_training_reference(images, 8, 8)

    assert actual.shape == expected.shape
    assert actual.dtype == images.dtype
    assert torch.equal(actual, expected)


def test_resize_preserves_unbatched_layouts():
    channels_first = _batched_images(1, channels_last=False).squeeze(0)
    channels_last = channels_first.movedim(0, -1)

    actual_first = resize_with_pad_pil_uint8(channels_first, 8, 8)
    actual_last = resize_with_pad_pil_uint8(channels_last, 8, 8)

    assert actual_first.shape == (3, 8, 8)
    assert actual_last.shape == (8, 8, 3)
    assert torch.equal(actual_first.movedim(0, -1), actual_last)


def test_resize_keeps_training_black_padding_before_model_normalization():
    images = torch.ones(2, 3, 4, 8)

    output = resize_with_pad_pil_uint8(images, 8, 8)

    assert torch.equal(output[:, :, :2], torch.zeros_like(output[:, :, :2]))
    assert torch.equal(output[:, :, 6:], torch.zeros_like(output[:, :, 6:]))
    assert torch.equal(output[:, :, 2:6], torch.ones_like(output[:, :, 2:6]))


def test_install_replaces_lerobot_pi05_resize():
    import lerobot.policies.pi05.modeling_pi05 as modeling_pi05

    original = modeling_pi05.resize_with_pad_torch
    try:
        install_pi05_openpi_resize()
        assert modeling_pi05.resize_with_pad_torch is resize_with_pad_pil_uint8
    finally:
        modeling_pi05.resize_with_pad_torch = original
