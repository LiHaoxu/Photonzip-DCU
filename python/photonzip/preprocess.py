from __future__ import annotations

import numpy as np

INT16_MIN = -32768
INT16_MAX = 32767


def apply_delta(array: np.ndarray) -> tuple[np.ndarray, dict]:
    """Inter-slice delta along axis 0.

    ``delta[0] = array[0]`` (uint16->int16 bit cast, losslessly reversible) and
    ``delta[i] = clip(array[i] - array[i-1], -32768, 32767)`` computed in int32.
    The int16 result is returned as a uint16 bit-view so any byte-oriented codec
    can consume it. Clipping is the only lossy step, so the run is lossless iff no
    pixel overflowed the int16 range.
    """
    if array.dtype != np.uint16:
        raise NotImplementedError(
            f"delta preprocessing supports uint16 input only (got {array.dtype})."
        )
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("delta preprocessing requires a non-empty leading (slice) axis.")

    delta = np.empty_like(array, dtype=np.int16)
    delta[0] = array[0].astype(np.int16)  # unsafe cast == bit reinterpret

    overflow_pixels = 0
    if array.shape[0] > 1:
        real_delta = array[1:].astype(np.int32) - array[:-1].astype(np.int32)
        overflow_pixels = int(
            np.count_nonzero((real_delta < INT16_MIN) | (real_delta > INT16_MAX))
        )
        delta[1:] = np.clip(real_delta, INT16_MIN, INT16_MAX).astype(np.int16)

    metadata = {
        "type": "delta",
        "axis": 0,
        "input_dtype": str(array.dtype),
        "stored_dtype": "int16",
        "clip_min": INT16_MIN,
        "clip_max": INT16_MAX,
        "overflow_pixels": overflow_pixels,
        "lossless": overflow_pixels == 0,
    }
    return delta.view(np.uint16), metadata


def invert_delta(array_u16: np.ndarray, preprocess: dict) -> np.ndarray:
    """Reconstruct the original uint16 volume from delta-preprocessed data.

    Slice 0 holds the original bit pattern and slice i holds the difference to slice i-1,
    so a running sum modulo 2**16 restores the volume. uint16 additions wrap modulo 2**16,
    which gives exactly the result of an int64 cumulative sum followed by ``mod 2**16``.
    The sum runs slice by slice: ``np.cumsum(..., axis=0)`` walks a C-ordered volume with a
    slice-sized stride and is ~100x slower (152 s vs. ~1 s for 100 x 2048 x 2048).
    """
    out_dtype = np.dtype(preprocess["input_dtype"])
    if array_u16.shape[0] < 1:
        return array_u16.astype(out_dtype)

    deltas = np.ascontiguousarray(array_u16).view(np.uint16)
    restored = np.empty(deltas.shape, dtype=np.uint16)
    restored[0] = deltas[0]
    for i in range(1, deltas.shape[0]):
        np.add(restored[i - 1:i], deltas[i:i + 1], out=restored[i:i + 1])
    return restored.astype(out_dtype, copy=False)


__all__ = ["apply_delta", "invert_delta"]
