from __future__ import annotations

from importlib import import_module

import numpy as np

try:
    _native = import_module("photonzip._native")
except ModuleNotFoundError:
    _native = import_module("_native")

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

# Where the delta runs. All backends produce identical bytes.
#   "numpy" - reference implementation in NumPy (single-threaded);
#   "cpu"   - native, OpenMP over the elements of a slice;
#   "dcu"   - HIP kernel, one DCU thread per element of a slice walking all slices. Host input
#             is copied to the DCU and the result back, unless output_on_dcu=True. A 1-D
#             array has one element per slice and runs on a single DCU thread: use "cpu".
#   "auto"  - "dcu" for tensors in DCU memory, "cpu" otherwise.
DELTA_BACKENDS = ("auto", "numpy", "cpu", "dcu")

_KDL_CPU = 1


def _check_encoding(encoding: str) -> None:
    if encoding not in DELTA_ENCODINGS:
        raise ValueError(f"unknown delta encoding {encoding!r} (expected one of {DELTA_ENCODINGS}).")


def _check_backend(backend: str) -> None:
    if backend not in DELTA_BACKENDS:
        raise ValueError(f"unknown delta backend {backend!r} (expected one of {DELTA_BACKENDS}).")


def _as_host_numpy(array):
    """NumPy view of host-memory input, or None for a tensor in device memory."""
    if isinstance(array, np.ndarray):
        return array
    device = getattr(array, "__dlpack_device__", None)
    if device is not None and device()[0] != _KDL_CPU:
        return None
    return np.from_dlpack(array)


def _run_native(op, host, device_tensor, encoding, backend, output_on_dcu):
    """Runs _native.delta_encode/decode with the input's shape restored on the result."""
    if host is not None:
        # The native code treats axis 0 as the slice axis and the rest as one flat slice.
        flat = np.ascontiguousarray(host).view(np.uint16).reshape(host.shape[0], -1)
        result = op(flat, encoding, "cpu" if backend == "auto" else backend, output_on_dcu)
        shape = list(host.shape)
    else:
        result = op(device_tensor, encoding, "dcu" if backend == "auto" else backend, output_on_dcu)
        shape = list(device_tensor.shape)
    # Host results are NumPy arrays, DCU results PhotonZipArrays; both reshape without a copy.
    array, *extra = result if isinstance(result, tuple) else (result,)
    array = array.reshape(shape)
    return (array, *extra) if extra else array


def apply_delta(array, encoding: str = "int16", backend: str = "auto",
                output_on_dcu: bool | None = None) -> tuple[object, dict]:
    """Inter-slice delta along axis 0.

    ``delta[0] = array[0]`` (stored unchanged) and ``delta[i] = clip(array[i] -
    array[i-1], -32768, 32767)`` computed in int32, stored as a uint16 according to
    ``encoding`` (see ``DELTA_ENCODINGS``). Clipping is the only lossy step, so the
    run is lossless iff no pixel overflowed the int16 range.

    ``backend`` selects where it runs (see ``DELTA_BACKENDS``); the bytes are the same on
    all of them. The result is a NumPy array in host memory, or, with ``output_on_dcu=True``
    or DCU input, a ``PhotonZipArray`` in DCU memory that ``photonzip.compress`` compresses
    on the DCU without another copy.
    """
    _check_encoding(encoding)
    _check_backend(backend)
    host = _as_host_numpy(array)
    if host is not None and host.dtype != np.uint16:
        raise NotImplementedError(
            f"delta preprocessing supports uint16 input only (got {host.dtype})."
        )
    shape = tuple(host.shape if host is not None else array.shape)
    if len(shape) < 1 or shape[0] < 1:
        raise ValueError("delta preprocessing requires a non-empty leading (slice) axis.")

    if backend == "numpy" or (host is not None and host.size == 0):
        if host is None:
            raise ValueError("the numpy delta backend needs host-memory input.")
        delta, overflow_pixels = _apply_delta_numpy(host, encoding)
    else:
        delta, overflow_pixels = _run_native(_native.delta_encode, host, array, encoding, backend, output_on_dcu)

    metadata = {
        "type": "delta",
        "axis": 0,
        "input_dtype": "uint16",
        "stored_dtype": "uint16",
        "encoding": encoding,
        "clip_min": INT16_MIN,
        "clip_max": INT16_MAX,
        "overflow_pixels": int(overflow_pixels),
        "lossless": overflow_pixels == 0,
    }
    return delta, metadata


def invert_delta(array, preprocess: dict, backend: str = "auto",
                 output_on_dcu: bool | None = None):
    """Reconstruct the original uint16 volume from delta-preprocessed data.

    Slice 0 holds the original values and slice i the stored difference to slice i-1,
    so a running sum modulo 2**16 restores the volume ("offset" differences carry an
    extra +32768, which is undone by adding another 32768: -32768 == +32768 mod 2**16).
    Metadata without an ``encoding`` field (older files) means "int16".

    ``backend`` and ``output_on_dcu`` work as in ``apply_delta``: host input gives a NumPy
    array unless ``output_on_dcu=True``; DCU input (e.g. a decompressed ``PhotonZipArray``
    still on the DCU) stays there unless ``output_on_dcu=False``.
    """
    out_dtype = np.dtype(preprocess["input_dtype"])
    encoding = preprocess.get("encoding", "int16")
    if encoding not in DELTA_ENCODINGS:
        raise ValueError(f"unknown delta encoding {encoding!r} in preprocessing metadata.")
    _check_backend(backend)
    host = _as_host_numpy(array)
    if host is not None and (host.ndim < 1 or host.shape[0] < 1 or host.size == 0):
        return host.astype(out_dtype)

    if backend == "numpy":
        if host is None:
            raise ValueError("the numpy delta backend needs host-memory input.")
        return _invert_delta_numpy(host, encoding).astype(out_dtype, copy=False)
    restored = _run_native(_native.delta_decode, host, array, encoding, backend, output_on_dcu)
    return restored.astype(out_dtype, copy=False) if isinstance(restored, np.ndarray) else restored


def _apply_delta_numpy(array: np.ndarray, encoding: str) -> tuple[np.ndarray, int]:
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
    return delta, overflow_pixels


def _invert_delta_numpy(array_u16: np.ndarray, encoding: str) -> np.ndarray:
    # uint16 additions wrap modulo 2**16, which gives exactly the result of an int64
    # cumulative sum followed by ``mod 2**16``. The sum runs slice by slice:
    # ``np.cumsum(..., axis=0)`` walks a C-ordered volume with a slice-sized stride and is
    # ~100x slower (152 s vs. ~1 s for 100 x 2048 x 2048).
    deltas = np.ascontiguousarray(array_u16).view(np.uint16)
    restored = np.empty(deltas.shape, dtype=np.uint16)
    restored[0] = deltas[0]
    offset = np.uint16(OFFSET)
    for i in range(1, deltas.shape[0]):
        np.add(restored[i - 1:i], deltas[i:i + 1], out=restored[i:i + 1])
        if encoding == "offset":
            np.add(restored[i:i + 1], offset, out=restored[i:i + 1])
    return restored


__all__ = ["DELTA_BACKENDS", "DELTA_ENCODINGS", "apply_delta", "invert_delta"]
