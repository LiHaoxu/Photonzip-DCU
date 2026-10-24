#!/usr/bin/env bash
# Reuses the MPI/HDF5 benchmark configuration validated on /data/lhx/test_data/part.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORES="${CORES:-64}"
RANKS_LIST="${RANKS_LIST:-1 8 16 32}"
CASES="${CASES:-raw,lc,mans_r,pfpl_rel}"
MIB_PER_RANK="${MIB_PER_RANK:-256}"
CHUNK_FRAMES="${CHUNK_FRAMES:-2}"
REPEATS="${REPEATS:-3}"
TIFF_DIR="${TIFF_DIR:-/data/lhx/test_data/part}"
CSV="${CSV:-${ROOT}/build/hdf5_results.csv}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH:-${ROOT}/build/bin/plugins}"
export PYTHONPATH="${ROOT}/python:${ROOT}/build/bindings/python${PYTHONPATH:+:${PYTHONPATH}}"

for plugin in libH5Z-LC.so libH5Z-PFPL.so libH5Z-MANS.so; do
  if [[ ! -f "${HDF5_PLUGIN_PATH}/${plugin}" ]]; then
    echo "missing ${HDF5_PLUGIN_PATH}/${plugin}; build PhotonZip first" >&2
    exit 1
  fi
done

for ranks in ${RANKS_LIST}; do
  threads=$((CORES / ranks))
  ((threads < 1)) && threads=1

  mpirun -n "${ranks}" \
    --map-by "slot:PE=${threads}" --bind-to core \
    --mca hwloc_base_mem_bind_failure_action silent \
    -x OMP_NUM_THREADS="${threads}" \
    -x OMP_PROC_BIND=close \
    -x OMP_PLACES=cores \
    -x HDF5_PLUGIN_PATH \
    -x PYTHONPATH \
    python3 "${ROOT}/benchmarks/hdf5/mpi_h5_bench.py" \
      --tiff-dir "${TIFF_DIR}" \
      --cases "${CASES}" \
      --mib-per-rank "${MIB_PER_RANK}" \
      --chunk-frames "${CHUNK_FRAMES}" \
      --threads "${threads}" \
      --repeats "${REPEATS}" \
      --verify \
      --csv "${CSV}" \
      --tag "rank${ranks}" \
      ${EXTRA_ARGS}
done

echo "results appended to ${CSV}"
