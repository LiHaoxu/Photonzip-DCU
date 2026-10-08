from __future__ import annotations

import numpy as np

INT16_MIN = -32768
INT16_MAX = 32767
OFFSET = 32768

# How a signed difference d (clipped to int16) is stored in a uint16 element:
#   "int16"  - two's complement bit pattern (d = -1 -> 65535). Fine for LC, whose
#              DIFFMS stage maps signs itself.
#   "offset" - d + 32768 (d = -1 -> 32767). Keeps the numeric order of the
#              differences, which MANS's ADM needs: its blocks test max - min and
#              code |value - center|, so with "int16" a block holding both signs
#              spans ~65535 and ADM is skipped (98% of the blocks on tomography data).
DELTA_ENCODINGS = ("int16", "offset")


def apply_delta(array: np.ndarray, encoding: str = "int16") -> tuple[np.ndarray, dict]:
    """Inter-slice delta along axis 0.

    ``delta[0] = array[0]`` (stored unchanged) and ``delta[i] = clip(array[i] -
    array[i-1], -32768, 32767)`` computed in int32, stored as a uint16 according to
    ``encoding`` (see ``DELTA_ENCODINGS``). Clipping is the only lossy step, so the
    run is lossless iff no pixel overflowed the int16 range.
    """
    if array.dtype != np.uint16:
        raise NotImplementedError(
            f"delta preprocessing supports uint16 input only (got {array.dtype})."
        )
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("delta preprocessing requires a non-empty leading (slice) axis.")
    if encoding not in DELTA_ENCODINGS:
        raise ValueError(f"unknown delta encoding {encoding!r} (expected one of {DELTA_ENCODINGS}).")

    delta = np.empty_like(array, dtype=np.uint16)
    delta[0] = array[0]

    overflow_pixels = 0
    if array.shape[0] > 1:
        real_delta = array[1:].astype(np.int32) - array[:-1].astype(np.int32)
        overflow_pixels = int(
            np.count_nonzero((real_delta < INT16_MIN) | (real_delta > INT16_MAX))
        )
        clipped = np.clip(real_delta, INT16_MIN, INT16_MAX)
        if encoding == "offset":
            delta[1:] = (clipped + OFFSET).astype(np.uint16)
        else:
            delta[1:] = clipped.astype(np.int16).view(np.uint16)

    metadata = {
        "type": "delta",
        "axis": 0,
        "input_dtype": str(array.dtype),
        "stored_dtype": "int16",
        "encoding": encoding,
        "clip_min": INT16_MIN,
        "clip_max": INT16_MAX,
        "overflow_pixels": overflow_pixels,
        "lossless": overflow_pixels == 0,
    }
    return delta, metadata


def invert_delta(array_u16: np.ndarray, preprocess: dict) -> np.ndarray:
    """Reconstruct the original uint16 volume from delta-preprocessed data.

    Slice 0 holds the original values and slice i the stored difference to slice i-1,
    so a running sum modulo 2**16 restores the volume ("offset" differences carry an
    extra +32768, which is undone by adding another 32768: -32768 == +32768 mod 2**16).
    uint16 additions wrap modulo 2**16, which gives exactly the result of an int64
    cumulative sum followed by ``mod 2**16``. The sum runs slice by slice:
    ``np.cumsum(..., axis=0)`` walks a C-ordered volume with a slice-sized stride and is
    ~100x slower (152 s vs. ~1 s for 100 x 2048 x 2048). Metadata without an
    ``encoding`` field (older files) means "int16".
    """
    out_dtype = np.dtype(preprocess["input_dtype"])
    encoding = preprocess.get("encoding", "int16")
    if encoding not in DELTA_ENCODINGS:
        raise ValueError(f"unknown delta encoding {encoding!r} in preprocessing metadata.")
    if array_u16.shape[0] < 1:
        return array_u16.astype(out_dtype)

    deltas = np.ascontiguousarray(array_u16).view(np.uint16)
    restored = np.empty(deltas.shape, dtype=np.uint16)
    restored[0] = deltas[0]
    offset = np.uint16(OFFSET)
    for i in range(1, deltas.shape[0]):
        np.add(restored[i - 1:i], deltas[i:i + 1], out=restored[i:i + 1])
        if encoding == "offset":
            np.add(restored[i:i + 1], offset, out=restored[i:i + 1])
    return restored.astype(out_dtype, copy=False)


__all__ = ["DELTA_ENCODINGS", "apply_delta", "invert_delta"]
