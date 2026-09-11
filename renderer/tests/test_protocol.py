"""Protocol round trips and validation errors (renderer/PROTOCOL.md v1)."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from glassrenderer import protocol as P


def _b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _payload(width: int = 8, height: int = 6, **over: object) -> dict:
    image = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    body = {
        "job_id": "p3-1a2b3c",
        "image": P.encode_png_rgb(image),
        "mask": P.encode_png_mask(mask),
    }
    body.update(over)
    return body


# --- constants ---------------------------------------------------------------


def test_version_is_one() -> None:
    assert P.VERSION == "1"


def test_default_params_match_the_contract() -> None:
    # Arrange / Act / Assert - the defaults are the contract's, verbatim.
    assert P.DEFAULT_PARAMS == {
        "lama": True,
        "sdxl": True,
        "strength": 0.4,
        "steps": 24,
        "controlnet_scale": 0.8,
        "guidance": 4.0,
        "seed": 0,
        "feather_px": 2,
        "prompt": "",
        "negative_prompt": "",
    }


def test_default_params_cannot_be_mutated_by_a_caller() -> None:
    params = P.validate_params({})
    params["steps"] = 99
    assert P.DEFAULT_PARAMS["steps"] == 24


def test_body_cap_is_64_mib() -> None:
    assert P.MAX_BODY_BYTES == 64 * 1024 * 1024


# --- image round trips -------------------------------------------------------


def test_rgb_round_trip_is_lossless(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)
    assert np.array_equal(P.decode_png_rgb(P.encode_png_rgb(image)), image)


def test_mask_round_trip_is_lossless(rng: np.random.Generator) -> None:
    mask = rng.integers(0, 256, (17, 23), dtype=np.uint8)
    decoded = P.decode_png_mask(P.encode_png_mask(mask))
    assert decoded.shape == mask.shape and decoded.dtype == np.uint8
    assert np.array_equal(decoded, mask)


def test_encode_rejects_a_non_uint8_image() -> None:
    with pytest.raises(P.ProtocolError, match="uint8"):
        P.encode_png_rgb(np.zeros((4, 4, 3), np.float32))


def test_encode_rejects_a_greyscale_image() -> None:
    with pytest.raises(P.ProtocolError, match="HxWx3"):
        P.encode_png_rgb(np.zeros((4, 4), np.uint8))


def test_encode_rejects_a_three_channel_mask() -> None:
    with pytest.raises(P.ProtocolError, match="HxW"):
        P.encode_png_mask(np.zeros((4, 4, 3), np.uint8))


@pytest.mark.parametrize(
    "value, message",
    [
        (123, "must be a base64 string"),
        ("not base64 ***", "not valid base64"),
        (base64.b64encode(b"nope").decode(), "not a PNG"),
    ],
)
def test_decode_rejects_bad_payloads(value: object, message: str) -> None:
    with pytest.raises(P.ProtocolError, match=message):
        P.decode_png_rgb(value, field="image")


def test_decode_rejects_an_image_larger_than_the_pixel_cap(monkeypatch) -> None:
    monkeypatch.setattr(P, "MAX_IMAGE_PIXELS", 64 * 64 - 1)
    image = np.zeros((64, 64, 3), np.uint8)
    with pytest.raises(P.ProtocolError, match="exceeds"):
        P.decode_png_rgb(P.encode_png_rgb(image), field="image")
    monkeypatch.setattr(P, "MAX_IMAGE_PIXELS", 64 * 64)
    assert P.decode_png_rgb(P.encode_png_rgb(image), field="image").shape == (64, 64, 3)


def test_decode_rgb_rejects_a_non_rgb_png() -> None:
    with pytest.raises(P.ProtocolError, match="8-bit RGB"):
        P.decode_png_rgb(_b64_png(Image.new("RGBA", (4, 4))), field="image")


def test_decode_rgb_rejects_a_16_bit_png() -> None:
    with pytest.raises(P.ProtocolError, match="8-bit RGB"):
        P.decode_png_rgb(_b64_png(Image.new("I;16", (4, 4))), field="image")


def test_decode_mask_rejects_an_rgb_png() -> None:
    with pytest.raises(P.ProtocolError, match="8-bit greyscale"):
        P.decode_png_mask(_b64_png(Image.new("RGB", (4, 4))), field="mask")


def test_decode_reports_the_field_name() -> None:
    with pytest.raises(P.ProtocolError, match="^mask:"):
        P.decode_png_mask(7, field="mask")


# --- request validation ------------------------------------------------------


def test_request_round_trip() -> None:
    request = P.InpaintRequest.from_payload(_payload())
    assert request.job_id == "p3-1a2b3c"
    assert request.image.shape == (6, 8, 3)
    assert request.mask.shape == (6, 8)
    assert request.params == P.DEFAULT_PARAMS


def test_request_rejects_a_non_object_body() -> None:
    with pytest.raises(P.ProtocolError, match="body: expected a JSON object"):
        P.InpaintRequest.from_payload([1, 2, 3])


def test_request_rejects_a_missing_job_id() -> None:
    body = _payload()
    del body["job_id"]
    with pytest.raises(P.ProtocolError, match="job_id: required"):
        P.InpaintRequest.from_payload(body)


def test_request_rejects_a_non_string_job_id() -> None:
    with pytest.raises(P.ProtocolError, match="job_id: must be a string"):
        P.InpaintRequest.from_payload(_payload(job_id=5))


def test_request_rejects_an_over_long_job_id() -> None:
    with pytest.raises(P.ProtocolError, match="job_id: at most"):
        P.InpaintRequest.from_payload(_payload(job_id="x" * 300))


def test_request_rejects_a_missing_image() -> None:
    body = _payload()
    del body["image"]
    with pytest.raises(P.ProtocolError, match="image: required"):
        P.InpaintRequest.from_payload(body)


def test_request_rejects_a_missing_mask() -> None:
    body = _payload()
    del body["mask"]
    with pytest.raises(P.ProtocolError, match="mask: required"):
        P.InpaintRequest.from_payload(body)


def test_request_rejects_a_mask_of_another_size() -> None:
    body = _payload()
    body["mask"] = P.encode_png_mask(np.zeros((3, 3), np.uint8))
    with pytest.raises(P.ProtocolError, match="mask: 3x3 does not match image 8x6"):
        P.InpaintRequest.from_payload(body)


def test_encode_rejects_an_empty_image() -> None:
    with pytest.raises(P.ProtocolError, match="empty"):
        P.encode_png_rgb(np.zeros((0, 4, 3), np.uint8))


# --- params ------------------------------------------------------------------


def test_params_default_when_absent() -> None:
    assert P.validate_params(None) == P.DEFAULT_PARAMS


def test_params_merge_over_the_defaults() -> None:
    params = P.validate_params({"steps": 8, "sdxl": False})
    assert params["steps"] == 8 and params["sdxl"] is False
    assert params["strength"] == P.DEFAULT_PARAMS["strength"]


def test_params_accept_an_int_for_a_float_field() -> None:
    assert P.validate_params({"strength": 1})["strength"] == 1.0


def test_params_reject_a_non_object() -> None:
    with pytest.raises(P.ProtocolError, match="params: expected a JSON object"):
        P.validate_params([1])


def test_params_reject_an_unknown_key() -> None:
    with pytest.raises(P.ProtocolError, match="params: unknown key 'denoise'"):
        P.validate_params({"denoise": 1})


@pytest.mark.parametrize("field", ["lama", "sdxl"])
def test_params_reject_a_non_bool_flag(field: str) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: must be a boolean"):
        P.validate_params({field: 1})


@pytest.mark.parametrize(
    "field, value",
    [
        ("strength", -0.01),
        ("strength", 1.01),
        ("steps", 0),
        ("steps", 101),
        ("controlnet_scale", -0.1),
        ("controlnet_scale", 2.1),
        ("guidance", -1.0),
        ("guidance", 20.5),
        ("feather_px", -1),
        ("feather_px", 33),
    ],
)
def test_params_reject_out_of_range_numbers(field: str, value: float) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: must be between"):
        P.validate_params({field: value})


@pytest.mark.parametrize("field", ["strength", "controlnet_scale", "guidance"])
def test_params_reject_a_non_number(field: str) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: must be a number"):
        P.validate_params({field: "0.5"})


@pytest.mark.parametrize("field", ["steps", "seed", "feather_px"])
def test_params_reject_a_non_integer(field: str) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: must be an integer"):
        P.validate_params({field: 2.5})


@pytest.mark.parametrize("field", ["steps", "seed", "feather_px"])
def test_params_reject_a_bool_for_an_integer(field: str) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: must be an integer"):
        P.validate_params({field: True})


def test_params_reject_a_nan_float() -> None:
    with pytest.raises(P.ProtocolError, match="params.strength: must be a number"):
        P.validate_params({"strength": float("nan")})


def test_params_accept_any_int_seed() -> None:
    assert P.validate_params({"seed": -12345})["seed"] == -12345


@pytest.mark.parametrize("field", ["prompt", "negative_prompt"])
def test_params_reject_a_non_string_prompt(field: str) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: must be a string"):
        P.validate_params({field: 3})


@pytest.mark.parametrize("field", ["prompt", "negative_prompt"])
def test_params_reject_an_over_long_prompt(field: str) -> None:
    with pytest.raises(P.ProtocolError, match=f"params.{field}: at most"):
        P.validate_params({field: "x" * (P.MAX_PROMPT_CHARS + 1)})


# --- response ----------------------------------------------------------------


def test_response_payload_matches_the_contract(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    response = P.InpaintResponse(
        job_id="p3-1a2b3c",
        image=image,
        timings_ms={"decode": 3.1, "lama": 9.0, "sdxl": 0.0, "composite": 4.0, "total": 16.1},
        mode="fake",
        stages=["lama", "sdxl"],
    )
    payload = response.to_payload()
    assert payload["job_id"] == "p3-1a2b3c"
    assert payload["mode"] == "fake"
    assert payload["stages"] == ["lama", "sdxl"]
    assert payload["size"] == [8, 6]  # width, height
    assert set(payload["timings_ms"]) == {"decode", "lama", "sdxl", "composite", "total"}
    assert np.array_equal(P.decode_png_rgb(payload["image"]), image)
