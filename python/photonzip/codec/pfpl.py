from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from importlib import import_module

from ..codecs import Codec
from ..codec_registry import register_codec
from ..options import CodecOptions

try:
    _native = import_module("photonzip._native")
except ModuleNotFoundError:
    _native = import_module("_native")

H5Z_FILTER_PFPL_ID = 32772

# Quantiser selection. The bitstream header only carries the decoded size and the error
# bound, so the mode has to be passed alongside the payload; it is the first codec parameter
# and the first HDF5 cd_value.
PFPL_MODE_REL = 0


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _mode_id(mode: str) -> int:
    if mode == "rel":
        return PFPL_MODE_REL
    raise ValueError(f"Unsupported PFPL mode: {mode!r} (only 'rel' is implemented).")


@dataclass(frozen=True)
class PfplOptions(CodecOptions):
    """Options for the PFPL codec.

    PFPL is error bounded and lossy, works on float32 only and runs on the CPU. In ``rel``
    mode every sample is reconstructed within ``error_bound`` of its own magnitude, i.e.
    ``|orig - recon| <= error_bound * |orig|``.

    Samples whose magnitude is at or above ``threshold`` are stored verbatim rather than
    quantised, which keeps outliers exact at the cost of ratio. It defaults to infinity, so
    every finite sample goes through the quantiser.
    """

    error_bound: float = 1e-3
    threshold: float | None = None
    mode: str = "rel"

    @property
    def codec(self) -> str:
        return "pfpl"

    def to_codec_params(self, *, tensor=None, backend: str = "auto") -> list[int]:
        if not (self.error_bound > 0.0) or not math.isfinite(self.error_bound):
            raise ValueError("PFPL requires a positive, finite relative error bound.")
        threshold = math.inf if self.threshold is None else float(self.threshold)
        if not (threshold > 0.0):
            raise ValueError("PFPL requires a positive threshold.")
        return [_mode_id(self.mode), _float_bits(self.error_bound), _float_bits(threshold)]


class PfplCodec(Codec):
    @property
    def name(self) -> str:
        return "pfpl"

    def compress(
        self,
        tensor,
        *,
        backend="auto",
        codec_params=None,
        codec_options=None,
    ):
        if codec_params is not None and codec_options is not None:
            raise TypeError("Pass either codec_params or codec_options, not both.")
        if codec_options is None:
            codec_options = PfplOptions()
        if not hasattr(codec_options, "to_codec_params") or not hasattr(codec_options, "codec"):
            raise TypeError("codec_options must provide codec and to_codec_params().")
        return _native.compress_tensor(
            self.name,
            tensor,
            backend=backend,
            codec_params=codec_options.to_codec_params(tensor=tensor, backend=backend)
            if codec_params is None else codec_params,
        )

    def decompress(self, data, *, backend="auto", **_kwargs):
        return _native.decompress_tensor(data, backend=backend)


def to_hdf5_compression_opts(options: PfplOptions | None = None) -> tuple[int, ...]:
    """Builds the cd_values for the H5Z-PFPL filter.

    The filter reads the same layout the codec parameters use, so the two stay in step.
    """
    pfpl_options = PfplOptions() if options is None else options
    return tuple(pfpl_options.to_codec_params())


register_codec(PfplCodec())
