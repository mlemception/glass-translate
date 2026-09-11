"""Compositing guarantees and panel bucketing."""
from __future__ import annotations

import numpy as np
import pytest

from glassrenderer import compositing as C


def _rgb(rng: np.random.Generator, h: int = 40, w: int = 56) -> np.ndarray:
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _disc_mask(h: int = 40, w: int = 56, radius: int = 12) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2
    return np.where(inside, 255, 0).astype(np.uint8)


# --- the hard guarantee ------------------------------------------------------


@pytest.mark.parametrize("feather_px", [0, 1, 2, 8, 32])
def test_outside_the_mask_is_byte_identical(
    rng: np.random.Generator, feather_px: int
) -> None:
    src, out, mask = _rgb(rng), _rgb(rng), _disc_mask()
    result = C.composite(src, out, mask, feather_px)
    keep = mask == 0
    assert np.array_equal(result[keep], src[keep])


@pytest.mark.parametrize("feather_px", [0, 8])
def test_border_touching_mask_keeps_the_rest(
    rng: np.random.Generator, feather_px: int
) -> None:
    src, out = _rgb(rng), _rgb(rng)
    mask = np.zeros(src.shape[:2], np.uint8)
    mask[:, :10] = 255  # a stripe glued to the left border
    mask[-6:, :] = 255  # and one along the bottom
    result = C.composite(src, out, mask, feather_px)
    keep = mask == 0
    assert np.array_equal(result[keep], src[keep])
    # A mask that runs off the image is not feathered against the image edge:
    # the corner pixel is fully the new content.
    assert np.array_equal(result[-1, 0], out[-1, 0])


def test_empty_mask_returns_the_source_unchanged(rng: np.random.Generator) -> None:
    src, out = _rgb(rng), _rgb(rng)
    mask = np.zeros(src.shape[:2], np.uint8)
    assert np.array_equal(C.composite(src, out, mask, 8), src)


def test_full_mask_returns_the_output(rng: np.random.Generator) -> None:
    src, out = _rgb(rng), _rgb(rng)
    mask = np.full(src.shape[:2], 255, np.uint8)
    assert np.array_equal(C.composite(src, out, mask, 0), out)


def test_hard_composite_takes_the_output_inside_the_mask(
    rng: np.random.Generator,
) -> None:
    src, out, mask = _rgb(rng), _rgb(rng), _disc_mask()
    result = C.composite(src, out, mask, 0)
    fill = mask > 0
    assert np.array_equal(result[fill], out[fill])


def test_the_result_is_uint8_and_does_not_alias_its_inputs(
    rng: np.random.Generator,
) -> None:
    src, out, mask = _rgb(rng), _rgb(rng), _disc_mask()
    before = src.copy()
    result = C.composite(src, out, mask, 4)
    assert result.dtype == np.uint8 and result.shape == src.shape
    result[:] = 0
    assert np.array_equal(src, before)  # writing the result never touches src


def test_composite_rejects_mismatched_shapes(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="same size"):
        C.composite(_rgb(rng), _rgb(rng, 20, 20), _disc_mask(), 2)


# --- the feather ramp --------------------------------------------------------


def test_feather_zero_is_a_binary_alpha() -> None:
    alpha = C.feather_alpha(_disc_mask(), 0)
    assert set(np.unique(alpha)) <= {0.0, 1.0}


def test_feather_ramps_inside_the_mask_edge() -> None:
    mask = _disc_mask(radius=12)
    alpha = C.feather_alpha(mask, 8)
    assert np.all(alpha[mask == 0] == 0.0)  # never leaks outside
    assert alpha[20, 28] == pytest.approx(1.0)  # the centre is fully filled
    # A ring inside the edge is partially transparent.
    ramp = alpha[np.logical_and(mask > 0, alpha < 1.0)]
    assert ramp.size > 0
    assert 0.0 < ramp.max() < 1.0


def test_feather_alpha_is_monotonic_towards_the_centre() -> None:
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    alpha = C.feather_alpha(mask, 4)
    row = alpha[20, 10:21]
    assert np.all(np.diff(row) >= -1e-6)
    assert row[0] < row[-1] == pytest.approx(1.0)


# --- bucketing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "size, expected_long",
    [
        ((764, 1200), 1024),  # a 1200x764 ja page panel
        ((1067, 1600), 1024),
        ((900, 900), 768),
        ((700, 500), 512),
        ((512, 400), 512),
        ((300, 400), 400),  # below the smallest bucket: never upscaled
    ],
)
def test_bucket_size_picks_a_bucket_without_upscaling(
    size: tuple[int, int], expected_long: int
) -> None:
    h, w = size
    bh, bw = C.bucket_size(h, w)
    assert max(bh, bw) == expected_long
    assert max(bh, bw) <= max(h, w)
    assert bh % 8 == 0 and bw % 8 == 0
    assert bh >= 8 and bw >= 8


def test_bucket_size_keeps_the_aspect_ratio() -> None:
    bh, bw = C.bucket_size(764, 1200)
    assert bw == 1024
    assert abs(bh / bw - 764 / 1200) < 0.02


def test_bucket_size_of_a_tiny_panel() -> None:
    assert C.bucket_size(3, 5) == (8, 8)


def test_bucket_round_trip_restores_the_source_size(rng: np.random.Generator) -> None:
    image = _rgb(rng, 764, 1200)
    size = C.bucket_size(764, 1200)
    small = C.resize_image(image, size)
    assert small.shape[:2] == size
    back = C.resize_image(small, (764, 1200))
    assert back.shape == image.shape and back.dtype == np.uint8


def test_resize_image_is_a_no_op_at_the_same_size(rng: np.random.Generator) -> None:
    image = _rgb(rng)
    assert np.array_equal(C.resize_image(image, image.shape[:2]), image)


def test_resize_mask_stays_binary(rng: np.random.Generator) -> None:
    mask = _disc_mask(400, 600, 90)
    small = C.resize_mask(mask, C.bucket_size(400, 600))
    assert set(np.unique(small)) <= {0, 255}
    assert small.dtype == np.uint8
    assert small.any()


def test_pad_and_crop_round_trip(rng: np.random.Generator) -> None:
    image = _rgb(rng, 37, 53)
    padded, source_size = C.pad_to_multiple(image, 8)
    assert padded.shape[0] % 8 == 0 and padded.shape[1] % 8 == 0
    assert source_size == (37, 53)
    assert np.array_equal(C.crop_to(padded, source_size), image)


def test_pad_is_a_no_op_when_already_aligned(rng: np.random.Generator) -> None:
    image = _rgb(rng, 40, 56)
    padded, source_size = C.pad_to_multiple(image, 8)
    assert padded.shape == image.shape and source_size == (40, 56)


def test_pad_handles_a_single_channel_mask() -> None:
    mask = _disc_mask(37, 53, 10)
    padded, source_size = C.pad_to_multiple(mask, 8)
    assert padded.ndim == 2 and padded.shape[0] % 8 == 0
    assert np.array_equal(C.crop_to(padded, source_size), mask)
