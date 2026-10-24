"""Round-trip tests for the PhotonZip HDF5 filter plugins driven from h5py.

The plugins are dlopened by the HDF5 library itself, so HDF5_PLUGIN_PATH has to point at the
build tree before h5py (and with it libhdf5) is imported.
"""

import os
import pathlib
import struct
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "build" / "bin" / "plugins"
PACKAGE_DIR = REPO_ROOT / "python"
NATIVE_DIR = REPO_ROOT / "build" / "bindings" / "python"

os.environ["HDF5_PLUGIN_PATH"] = str(PLUGIN_DIR)
sys.path.insert(0, str(NATIVE_DIR))
sys.path.insert(0, str(PACKAGE_DIR))

h5py = pytest.importorskip("h5py")

import photonzip.codec.lc as lc
import photonzip.codec.mans as mans
import photonzip.codec.pfpl as pfpl

pytestmark = pytest.mark.skipif(
    not PLUGIN_DIR.is_dir(),
    reason=f"HDF5 filter plugins have not been built into {PLUGIN_DIR}",
)

SHAPE = (64, 128, 128)
CHUNKS = (8, 128, 128)

# PFPL decides whether a bin is acceptable by comparing the reconstruction against
# ``orig * (1 + eb)`` in float32. Both the sum and the product round, so the bound the
# encoder actually enforces can sit up to about one float32 epsilon above the requested one.
# That slack is invisible at loose bounds and worth ~0.7% at eb=1e-5.
FLOAT32_EPS = float(np.finfo(np.float32).eps)


def _realised_bound(error_bound):
    return error_bound + 2.0 * FLOAT32_EPS


def _applied_filter_ids(dataset):
    dcpl = dataset.id.get_create_plist()
    return {dcpl.get_filter(index)[0] for index in range(dcpl.get_nfilters())}


def _u16_field():
    z, y, x = np.indices(SHAPE, dtype=np.int64)
    return (1000 + 7 * x + 3 * y + 11 * z + ((x * y + z) % 17)).astype(np.uint16)


def _f32_field():
    z, y, x = np.indices(SHAPE, dtype=np.float64)
    field = np.exp(0.001 * z) * (100.0 + 90.0 * np.sin(0.05 * x + 0.03 * y + 0.11 * z))
    return field.astype(np.float32)


def _roundtrip(path, data, filter_id, compression_opts):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data",
            data=data,
            chunks=CHUNKS,
            compression=filter_id,
            compression_opts=compression_opts,
        )
    with h5py.File(path, "r") as handle:
        return handle["data"][...], path.stat().st_size


def test_mans_hdf5_dims_follow_contiguous_memory_order(tmp_path):
    original = _u16_field()
    options = mans.MansOptions(
        mode="r",
        adm_compress_thread=2,
        adm_decompress_thread=2,
        warn_on_default_threads=False,
    )
    compression_opts = mans.to_hdf5_compression_opts(
        original,
        chunks=CHUNKS,
        options=options,
    )
    assert compression_opts[5:9] == (3, CHUNKS[2], CHUNKS[1], CHUNKS[0])

    path = tmp_path / "mans_dims.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            data=original,
            chunks=CHUNKS,
            compression=mans.H5Z_FILTER_MANS_ID,
            # Exercise the public HDF5 interface: callers only need to provide
            # the mode; set_local must derive dtype, threads, and chunk dims.
            compression_opts=(1,),
        )
        handle.flush()

        dcpl = dataset.id.get_create_plist()
        filter_info = next(
            dcpl.get_filter(index)
            for index in range(dcpl.get_nfilters())
            if dcpl.get_filter(index)[0] == mans.H5Z_FILTER_MANS_ID
        )
        local_params = filter_info[2]
        assert local_params[5:9] == (3, CHUNKS[2], CHUNKS[1], CHUNKS[0])

        filter_mask, encoded_chunk = dataset.id.read_direct_chunk((0, 0, 0))
        assert filter_mask == 0
        assert encoded_chunk[2] == 3
        assert struct.unpack_from("<QQQ", encoded_chunk, 16) == (
            CHUNKS[2],
            CHUNKS[1],
            CHUNKS[0],
        )

    with h5py.File(path, "r") as handle:
        restored = handle["data"][...]
    assert np.array_equal(restored, original)


def test_lc_filter_is_lossless(tmp_path):
    original = _u16_field()
    restored, file_bytes = _roundtrip(tmp_path / "lc.h5", original, lc.H5Z_FILTER_LC_ID, None)
    assert np.array_equal(restored, original)
    assert file_bytes < original.nbytes


def test_lc_filter_declines_float32(tmp_path):
    # h5py registers custom filters as optional, so a can_apply that says no makes HDF5 drop
    # the filter and store the chunks raw rather than fail the create.
    with h5py.File(tmp_path / "decline.h5", "w") as handle:
        dataset = handle.create_dataset(
            "data",
            data=_f32_field(),
            chunks=CHUNKS,
            compression=lc.H5Z_FILTER_LC_ID,
        )
        assert lc.H5Z_FILTER_LC_ID not in _applied_filter_ids(dataset)


@pytest.mark.parametrize("error_bound", [1e-2, 1e-3, 1e-5])
def test_pfpl_filter_honours_error_bound(tmp_path, error_bound):
    original = _f32_field()
    restored, _ = _roundtrip(
        tmp_path / "pfpl.h5",
        original,
        pfpl.H5Z_FILTER_PFPL_ID,
        pfpl.to_hdf5_compression_opts(pfpl.PfplOptions(error_bound=error_bound)),
    )
    assert np.all(np.abs(restored - original) <= _realised_bound(error_bound) * np.abs(original))


def test_pfpl_filter_tighter_bound_stores_more(tmp_path):
    original = _f32_field()
    _, loose_bytes = _roundtrip(
        tmp_path / "loose.h5",
        original,
        pfpl.H5Z_FILTER_PFPL_ID,
        pfpl.to_hdf5_compression_opts(pfpl.PfplOptions(error_bound=1e-2)),
    )
    _, tight_bytes = _roundtrip(
        tmp_path / "tight.h5",
        original,
        pfpl.H5Z_FILTER_PFPL_ID,
        pfpl.to_hdf5_compression_opts(pfpl.PfplOptions(error_bound=1e-5)),
    )
    assert loose_bytes < tight_bytes < original.nbytes


