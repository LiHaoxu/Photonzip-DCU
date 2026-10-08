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

# Same ID as the upstream H5Z-MANS plugin; see filters/include/photonzip_h5z_ids.h.
H5Z_FILTER_MANS_ID = 32032
MANS_MODE_P = 0


@dataclass(frozen=True)
class MansOptions(CodecOptions):
    """Options for the MANS codec.

    MANS (ADM mapping + ANS entropy coding) is lossless for uint16/uint32 tensors with
    1 to 3 dimensions. In this build it runs on the DCU backend only, in P-mode, and has
    no tunable parameters: the ADM geometry is taken from the tensor shape.

    For the HDF5 filter (``H5Z_FILTER_MANS_ID``) the geometry comes from the chunk shape;
    pass ``compression_opts=(MANS_MODE_P,)`` or leave it out.
    """

    @property
    def codec(self) -> str:
        return "mans"

    def to_codec_params(self, *, tensor=None, backend: str = "auto") -> list[int]:
        return []


class MansCodec(Codec):
    @property
    def name(self) -> str:
        return "mans"

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


# The native codec only exists in DCU builds; keep the Python registry in sync with it.
if "mans" in _native.list_codecs():
    register_codec(MansCodec())
