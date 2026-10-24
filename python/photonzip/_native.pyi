from typing import List
from typing import Any
from typing import Tuple


class PhotonZipError(Exception): ...
class PhotonZipArray:
    codec: str
    compressed: bool
    dtype: str
    shape: List[int]
    nbytes: int
    device: str  # "cpu" (host memory) or "cuda"
    def to_bytes(self) -> bytes: ...
    def to_host(self) -> "PhotonZipArray": ...  # from CUDA memory
    def reshape(self, shape: List[int]) -> "PhotonZipArray": ...
    def __dlpack__(self, stream: Any = ...) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...


def list_codecs() -> List[str]: ...


def available_backends() -> List[str]: ...


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
) -> Tuple[object, int]: ...  # (NumPy uint16 array, clipped elements)


def delta_decode(
    input: object,
    encoding: str,
    backend: str = ...,
) -> object: ...  # NumPy uint16 array
