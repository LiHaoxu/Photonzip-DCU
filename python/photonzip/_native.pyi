from typing import List
from typing import Any
from typing import Optional
from typing import Tuple


class PhotonZipError(Exception): ...
class PhotonZipArray:
    codec: str
    compressed: bool
    dtype: str
    shape: List[int]
    nbytes: int
    device: str  # "cpu" (host memory) or "dcu"
    def to_bytes(self) -> bytes: ...
    def to_host(self) -> "PhotonZipArray": ...
    def to_device(self) -> "PhotonZipArray": ...
    def reshape(self, shape: List[int]) -> "PhotonZipArray": ...
    def __dlpack__(self, stream: Any = ...) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...


def list_codecs() -> List[str]: ...


def compress_tensor(
    codec_name: str,
    input: object,
    backend: str = ...,
    codec_params: List[int] = ...,
) -> PhotonZipArray: ...


def decompress_tensor(
    input: PhotonZipArray,
    backend: str = ...,
) -> PhotonZipArray: ...


def compressed_from_bytes(
    codec_name: str,
    payload: bytes,
    dtype: str,
    shape: List[int],
    backend: str,
    codec_params: List[int] = ...,
) -> PhotonZipArray: ...


def invoke_codec(
    codec_name: str,
    op_name: str,
    request: object = ...,
) -> object: ...


def delta_encode(
    input: object,
    encoding: str,
    backend: str = ...,
    output_on_dcu: Optional[bool] = ...,
) -> Tuple[PhotonZipArray, int]: ...


def delta_decode(
    input: object,
    encoding: str,
    backend: str = ...,
    output_on_dcu: Optional[bool] = ...,
) -> PhotonZipArray: ...
