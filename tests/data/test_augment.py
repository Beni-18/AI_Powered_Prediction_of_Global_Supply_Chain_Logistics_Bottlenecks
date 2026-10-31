"""Tests for common/dataio/augment.py: train-only, nothing leaks into val/test, shapes and anchors stay right."""

import numpy as np
import pytest

from common.dataio.augment import jitter, scaling, window_warp

B, N, L, C_MEASURED, C_EXTRA, H = 4, 3, 24, 2, 3, 5
TARGET_IDX = [0, 1]


def _batch() -> dict:
    """A fake ForecastData.batch(): measured channels = hour number, bookkeeping channels = 7."""
    rng = np.random.default_rng(0)
    measured = np.broadcast_to(np.arange(L, dtype=np.float32)[None, None, :, None], (B, N, L, C_MEASURED))
    extra = np.full((B, N, L, C_EXTRA), 7.0, dtype=np.float32)
    x = np.concatenate([measured, extra], axis=-1).copy()
    return {"x": x, "anchor": x[:, :, -1, TARGET_IDX].copy(),
            "y": rng.normal(size=(B, N, H, len(TARGET_IDX))).astype(np.float32), "origins": np.arange(B)}


@pytest.mark.parametrize("split", ["val", "test", "gap"])
@pytest.mark.parametrize("helper", ["jitter", "scaling", "window_warp"])
def test_refuses_any_non_train_window(helper, split):
    """One val/test window in the batch is enough to stop: val and test must stay as measured."""
    rng = np.random.default_rng(42)
    splits = ["train"] * (B - 1) + [split]
    kwargs = {"target_idx": TARGET_IDX} if helper == "scaling" else {}
    with pytest.raises(ValueError):
        {"jitter": jitter, "scaling": scaling, "window_warp": window_warp}[helper](
            _batch(), splits, rng, C_MEASURED, **kwargs)


def test_originals_and_bookkeeping_channels_are_untouched():
    """Helpers return new arrays; the is_imputed/calendar channels and the input batch never change."""
    batch, splits = _batch(), ["train"] * B
    before = {k: v.copy() for k, v in batch.items()}
    rng = np.random.default_rng(42)
    for out in (jitter(batch, splits, rng, C_MEASURED), scaling(batch, splits, rng, C_MEASURED, TARGET_IDX),
                window_warp(batch, splits, rng, C_MEASURED)):
        assert out["x"].shape == batch["x"].shape
        assert np.array_equal(out["x"][..., C_MEASURED:], before["x"][..., C_MEASURED:])
    for k in before:
        assert np.array_equal(batch[k], before[k])


def test_jitter_changes_inputs_but_not_targets():
    out = jitter(_batch(), ["train"] * B, np.random.default_rng(42), C_MEASURED)
    assert not np.array_equal(out["x"][..., :C_MEASURED], _batch()["x"][..., :C_MEASURED])
    assert np.array_equal(out["y"], _batch()["y"]) and np.array_equal(out["anchor"], _batch()["anchor"])


def test_scaling_keeps_inputs_anchor_and_targets_consistent():
    """The same factor multiplies the past, the anchor and the future of each target channel."""
    batch = _batch()
    out = scaling(batch, ["train"] * B, np.random.default_rng(42), C_MEASURED, TARGET_IDX)
    # The anchor is still the last input hour of the target channels.
    assert np.allclose(out["anchor"], out["x"][:, :, -1, TARGET_IDX])
    factor = out["anchor"] / batch["anchor"]
    assert np.allclose(out["y"], batch["y"] * factor[:, :, None, :])


def test_window_warp_keeps_the_origin_hour_and_the_value_range():
    """The last input hour (the forecast origin) is unchanged, and resampling never invents values outside the range."""
    batch = _batch()
    out = window_warp(batch, ["train"] * B, np.random.default_rng(42), C_MEASURED)
    measured = out["x"][..., :C_MEASURED]
    assert np.allclose(measured[:, :, -1], batch["x"][:, :, -1, :C_MEASURED])
    assert measured.min() >= 0 and measured.max() <= L - 1
    assert not np.array_equal(measured, batch["x"][..., :C_MEASURED])


def test_same_seed_same_result():
    """Seeded generator -> identical augmentation on every laptop (seed 42 rule)."""
    a = jitter(_batch(), ["train"] * B, np.random.default_rng(42), C_MEASURED)["x"]
    b = jitter(_batch(), ["train"] * B, np.random.default_rng(42), C_MEASURED)["x"]
    assert np.array_equal(a, b)
