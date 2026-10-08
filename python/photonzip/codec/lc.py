from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from ..codecs import Codec
from ..codec_registry import register_codec
from ..options import CodecOptions

try:
    _native = import_module("photonzip._native")
except ModuleNotFoundError:
    _native = import_module("_native")

H5Z_FILTER_LC_ID = 32771


@dataclass(frozen=True)
class LcOptions(CodecOptions):
    """Options for the LC codec.

    The LC codec uses a fixed component pipeline (``DIFFMS_2 BIT_2 RZE_2``) and is
    lossless and CPU-only, so there are no tunable codec parameters. Global
    preprocessing such as inter-slice delta (``photonzip.preprocess``) is applied
    by the caller before compression and is therefore not configured here.

    The matching HDF5 filter (``H5Z_FILTER_LC_ID``) takes no options either: the decoded
    size travels in the payload header, so ``compression_opts`` can be left out entirely.
    """

    @property
    def codec(self) -> str:
        return "lc"

    def to_codec_params(self, *, tensor=None, backend: str = "auto") -> list[int]:
        return []


class LcCodec(Codec):
    @property
    def name(self) -> str:
        return "lc"

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
        if codec_options is not None and not hasattr(codec_options, "codec"):
            raise TypeError("codec_options must provide codec.")
        resolved_params = [] if codec_params is None else codec_params
        return _native.compress_tensor(
            self.name,
            tensor,
            backend=backend,
            codec_params=resolved_params,
        )

    def decompress(self, data, *, backend="auto", **_kwargs):
        return _native.decompress_tensor(data, backend=backend)


register_codec(LcCodec())
