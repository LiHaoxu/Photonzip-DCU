// Device-resident throughput benchmark for the PhotonZip DCU codecs.
//
//   photonzip_dcu_bench <raw file> [--dtype u16|u32] [--dims d0 [d1 [d2]]] [--iters N]
//                       [--dump PREFIX]   (writes PREFIX.lc and PREFIX.mans)
//                       [--only lc|mans] [--chunk-rows R [--parallel D T] [--host-only]]
//
// --dims is the C-order shape (slowest dimension first, as in NumPy/HDF5); MANS receives the
// fastest dimension as nx, like the photonzip codec and the H5Z-MANS filter.
// --chunk-rows R also times the host-memory API (host in, host out, as used by Python and
// the HDF5 filters) on chunks of R slices along the first dimension, one call per chunk.
// --parallel D T spreads those chunks over D DCUs with T host threads per DCU, and
// --host-only skips the whole-array device-resident runs.
//
// The input is copied to the DCU once; every timed call reads and writes device memory, so
// the numbers exclude PCIe transfers and Python overhead. Each result is verified against
// the input. MANS is also broken down into its ADM and ANS stages.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dcu/adm/mapping_uint16.h"
#include "dcu/adm/mapping_uint32.h"
#include "dcu/ans/dcu_ans.h"
#include "dcu/mans_dcu.h"
#include "mans_defs.h"
#include "photonzip/codecs/lc/lc_dcu.hpp"

namespace {

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(std::size_t bytes) { check(hipMalloc(&ptr, bytes ? bytes : 1), "hipMalloc"); }
  ~DeviceBuffer() { (void)hipFree(ptr); }
  template <typename T> T* as() const { return static_cast<T*>(ptr); }
};

// Median wall time of `iters` synchronous calls, after one warm-up call.
double time_median(int iters, const std::function<void()>& fn) {
  fn();
  check(hipDeviceSynchronize(), "warm-up");
  std::vector<double> samples;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    check(hipDeviceSynchronize(), "sync");
    samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

bool same_on_host(const void* d_a, const std::vector<std::uint8_t>& host, std::size_t bytes) {
  std::vector<std::uint8_t> copy(bytes);
  check(hipMemcpy(copy.data(), d_a, bytes, hipMemcpyDeviceToHost), "verify D2H");
  return std::memcmp(copy.data(), host.data(), bytes) == 0;
}

void dump(const std::string& path, const void* d_src, std::size_t bytes) {
  std::vector<char> copy(bytes);
  check(hipMemcpy(copy.data(), d_src, bytes, hipMemcpyDeviceToHost), "dump D2H");
  std::ofstream(path, std::ios::binary).write(copy.data(), static_cast<std::streamsize>(bytes));
}

// Runs fn(chunk) for every chunk on `devices` x `threads_per_device` host threads; chunk c
// goes to worker c % workers, worker w uses device w % devices.
void for_chunks(std::size_t chunks, int devices, int threads_per_device,
                const std::function<void(std::size_t)>& fn) {
  const int workers = devices * threads_per_device;
  if (workers <= 1) {
    for (std::size_t c = 0; c < chunks; ++c) fn(c);
    return;
  }
  std::vector<std::thread> pool;
  std::vector<std::string> errors(workers);
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&, w] {
      try {
        check(hipSetDevice(w % devices), "hipSetDevice");
        for (std::size_t c = w; c < chunks; c += workers) fn(c);
      } catch (const std::exception& e) {
        errors[w] = e.what();
      }
    });
  }
  for (auto& t : pool) t.join();
  for (const auto& e : errors) {
    if (!e.empty()) throw std::runtime_error(e);
  }
}

void report(const char* name, std::size_t raw, std::size_t packed, double tc, double td, bool ok) {
  std::printf("%-10s ratio %8.3f  compress %8.2f GB/s  decompress %8.2f GB/s  %s\n", name,
              packed ? double(raw) / double(packed) : 0.0, tc > 0 ? raw / tc / 1e9 : 0.0,
              td > 0 ? raw / td / 1e9 : 0.0, ok ? "OK" : "MISMATCH");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <raw file> [--dtype u16|u32] [--dims d0 [d1 [d2]]] [--iters N] "
                         "[--dump PREFIX] [--only lc|mans] [--chunk-rows R [--parallel D T] [--host-only]]\n", argv[0]);
    return 2;
  }
  std::string dtype = "u16";
  std::vector<std::uint32_t> dims;
  int iters = 10;
  std::string dump_prefix;
  std::string only;
  std::size_t chunk_rows = 0;
  int par_devices = 1, par_threads = 1;
  bool host_only = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dtype" && i + 1 < argc) {
      dtype = argv[++i];
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--only" && i + 1 < argc) {
      only = argv[++i];
    } else if (arg == "--host-only") {
      host_only = true;
    } else if (arg == "--parallel" && i + 2 < argc) {
      par_devices = std::max(1, std::atoi(argv[++i]));
      par_threads = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--chunk-rows" && i + 1 < argc) {
      chunk_rows = std::strtoul(argv[++i], nullptr, 10);
    } else if (arg == "--dump" && i + 1 < argc) {
      dump_prefix = argv[++i];
    } else if (arg == "--dims") {
      while (i + 1 < argc && argv[i + 1][0] != '-') dims.push_back(static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10)));
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  std::vector<std::uint8_t> host((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::size_t raw = host.size();
  const std::size_t width = dtype == "u32" ? 4 : 2;
  const std::size_t n = raw / width;
  if (dims.empty()) dims.push_back(static_cast<std::uint32_t>(n));
  std::size_t product = 1;
  for (auto d : dims) product *= d;
  if (raw % width != 0 || product != n || dims.size() > 3) {
    std::fprintf(stderr, "file size %zu does not match dtype %s and dims\n", raw, dtype.c_str());
    return 2;
  }

  hipDeviceProp_t prop;
  check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::printf("device %s (%s, %d CUs), input %zu bytes, %s, dims", prop.name, prop.gcnArchName,
              prop.multiProcessorCount, raw, dtype.c_str());
  for (auto d : dims) std::printf(" %u", d);
  std::printf(", median of %d runs\n", iters);

  DeviceBuffer d_in(raw), d_out(raw);
  check(hipMemcpy(d_in.ptr, host.data(), raw, hipMemcpyHostToDevice), "input H2D");
  int status = 0;

  // Chunks along the first dimension for the host-memory runs.
  const std::size_t row_elems = n / dims[0];
  const std::size_t rows_per_chunk = chunk_rows ? std::min<std::size_t>(chunk_rows, dims[0]) : dims[0];
  std::vector<std::pair<std::size_t, std::size_t>> chunk_list;  // (first row, rows)
  for (std::size_t r = 0; r < dims[0]; r += rows_per_chunk) {
    chunk_list.emplace_back(r, std::min(rows_per_chunk, static_cast<std::size_t>(dims[0]) - r));
  }
  int device_count = 0;
  check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  if (par_devices > device_count) {
    std::fprintf(stderr, "--parallel asks for %d DCUs but %d are visible\n", par_devices, device_count);
    return 2;
  }
  if (chunk_rows) {
    std::printf("host-memory runs: %zu chunks of up to %zu bytes on %d DCU(s) x %d thread(s)\n", chunk_list.size(),
                rows_per_chunk * row_elems * width, par_devices, par_threads);
  }

  // ---- LC (DIFFMS_2 BIT_2 RZE_2) ----
  if ((only.empty() || only == "lc") && !host_only) {
    const long long chunks = (static_cast<long long>(raw) + 16383) / 16384;
    DeviceBuffer d_lc(16 + chunks * 2 + chunks * 16384);
    long long packed = 0;
    const double tc = time_median(iters, [&] {
      packed = photonzip::lc::dcu::encode_device(d_in.as<unsigned char>(), raw, d_lc.as<unsigned char>());
    });
    const double td = time_median(iters, [&] {
      photonzip::lc::dcu::decode_device(d_lc.as<unsigned char>(), d_out.as<unsigned char>(), raw);
    });
    const bool ok = same_on_host(d_out.ptr, host, raw);
    report("lc", raw, packed, tc, td, ok);
    if (!dump_prefix.empty()) dump(dump_prefix + ".lc", d_lc.ptr, static_cast<std::size_t>(packed));
    status |= ok ? 0 : 1;
  }
  if ((only.empty() || only == "lc") && chunk_rows) {
    {
      // Output buffers are allocated once, uninitialised (like the Python API's host buffers),
      // so the timed loop measures only the codec calls.
      std::vector<std::unique_ptr<unsigned char[]>> streams(chunk_list.size());
      std::vector<std::size_t> sizes(chunk_list.size());
      for (std::size_t c = 0; c < chunk_list.size(); ++c) {
        const std::size_t bytes = chunk_list[c].second * row_elems * width;
        streams[c].reset(new unsigned char[16 + (bytes / 16384 + 1) * (2 + 16384)]);
      }
      std::unique_ptr<std::uint8_t[]> restored(new std::uint8_t[raw]);
      std::size_t total = 0;
      const double hc = time_median(iters, [&] {
        for_chunks(chunk_list.size(), par_devices, par_threads, [&](std::size_t c) {
          const std::size_t off = chunk_list[c].first * row_elems * width, bytes = chunk_list[c].second * row_elems * width;
          sizes[c] = static_cast<std::size_t>(photonzip::lc::dcu::encode_host(host.data() + off, bytes, streams[c].get()));
        });
      });
      for (const auto sz : sizes) total += sz;
      const double hd = time_median(iters, [&] {
        for_chunks(chunk_list.size(), par_devices, par_threads, [&](std::size_t c) {
          const std::size_t off = chunk_list[c].first * row_elems * width, bytes = chunk_list[c].second * row_elems * width;
          photonzip::lc::dcu::decode_host(streams[c].get(), sizes[c], restored.get() + off, bytes);
        });
      });
      const bool hok = std::memcmp(restored.get(), host.data(), raw) == 0;
      report("lc host", raw, total, hc, hd, hok);
      status |= hok ? 0 : 1;
    }
  }

  // ---- MANS (ADM + ANS, P-mode) ----
  if (only.empty() || only == "mans") {
    mans::MansParams p;
    p.backend = mans::Backend::DCU;
    p.dtype = width == 4 ? mans::DataType::U32 : mans::DataType::U16;
    p.mode = mans::Mode::P;
    const std::size_t rank = dims.size();
    p.dims = static_cast<std::uint32_t>(rank);
    p.nx = dims[rank - 1];
    p.ny = rank > 1 ? dims[rank - 2] : 0;
    p.nz = rank > 2 ? dims[rank - 3] : 0;
    if (!host_only) {
    const std::size_t cap = mans::dcu::get_max_compress_bytes(n, p);
    DeviceBuffer d_mans(cap);
    std::size_t packed = 0;
    const double tc = time_median(iters, [&] {
      packed = cap;
      mans::dcu::compress_internal_device(d_in.ptr, n, p, d_mans.as<std::uint8_t>(), packed);
    });
    const double td = time_median(iters, [&] {
      std::size_t out = raw;
      mans::dcu::decompress_internal_device(d_mans.ptr, packed, p, d_out.as<std::uint8_t>(), out);
    });
    const bool ok = same_on_host(d_out.ptr, host, raw);
    report("mans", raw, packed, tc, td, ok);
    if (!dump_prefix.empty()) dump(dump_prefix + ".mans", d_mans.ptr, packed);
    status |= ok ? 0 : 1;

    // Stage breakdown: ADM alone, then ANS on the ADM payload.
    DeviceBuffer d_adm(cap), d_ans(cap);
    std::size_t adm_size = 0;
    const double t_adm = time_median(iters, [&] {
      if (width == 4) {
        mans::dcu::adm::compress_u32_device(d_in.as<std::uint32_t>(), n, p, d_adm.as<std::uint8_t>(), adm_size);
      } else {
        mans::dcu::adm::compress_u16_device(d_in.as<std::uint16_t>(), n, p, d_adm.as<std::uint8_t>(), adm_size);
      }
    });
    std::size_t ans_size = 0;
    const double t_ans = time_median(iters, [&] {
      ans_size = cap;
      mans::dcu::ans::compress_device(d_adm.as<std::uint8_t>(), adm_size, d_ans.as<std::uint8_t>(), cap, ans_size);
    });
    const double t_ians = time_median(iters, [&] {
      std::size_t out = cap;
      mans::dcu::ans::decompress_device(d_ans.as<std::uint8_t>(), ans_size, d_adm.as<std::uint8_t>(), cap, out);
    });
    const double t_iadm = time_median(iters, [&] {
      if (width == 4) {
        mans::dcu::adm::decompress_u32_device(d_adm.as<std::uint8_t>(), adm_size, d_out.as<std::uint32_t>(), n, p);
      } else {
        mans::dcu::adm::decompress_u16_device(d_adm.as<std::uint8_t>(), adm_size, d_out.as<std::uint16_t>(), n, p);
      }
    });
    std::printf("  mans stages: ADM encode %.3f ms (%zu B), ANS encode %.3f ms (%zu B), "
                "ANS decode %.3f ms, ADM decode %.3f ms\n",
                t_adm * 1e3, adm_size, t_ans * 1e3, ans_size, t_ians * 1e3, t_iadm * 1e3);
    }

    if (chunk_rows) {
      // Output buffers are allocated once, uninitialised, outside the timed loop.
      std::vector<std::unique_ptr<std::uint8_t[]>> streams(chunk_list.size());
      std::vector<std::size_t> caps(chunk_list.size()), sizes(chunk_list.size());
      std::vector<mans::MansParams> params(chunk_list.size(), p);
      std::unique_ptr<std::uint8_t[]> restored(new std::uint8_t[raw]);
      std::size_t total = 0;
      for (std::size_t c = 0; c < chunk_list.size(); ++c) {
        const auto rows = static_cast<std::uint32_t>(chunk_list[c].second);
        if (rank == 1) params[c].nx = rows;
        else if (rank == 2) params[c].ny = rows;
        else params[c].nz = rows;
        caps[c] = mans::dcu::get_max_compress_bytes(chunk_list[c].second * row_elems, params[c]);
        streams[c].reset(new std::uint8_t[caps[c]]);
      }
      const double hc = time_median(iters, [&] {
        for_chunks(chunk_list.size(), par_devices, par_threads, [&](std::size_t c) {
          const std::size_t off = chunk_list[c].first * row_elems * width, elems = chunk_list[c].second * row_elems;
          std::size_t out = caps[c];
          mans::dcu::compress_internal(host.data() + off, elems, params[c], streams[c].get(), out);
          sizes[c] = out;
        });
      });
      for (const auto sz : sizes) total += sz;
      const double hd = time_median(iters, [&] {
        for_chunks(chunk_list.size(), par_devices, par_threads, [&](std::size_t c) {
          const std::size_t off = chunk_list[c].first * row_elems * width, bytes = chunk_list[c].second * row_elems * width;
          std::size_t out = bytes;
          mans::dcu::decompress_internal(streams[c].get(), sizes[c], params[c], restored.get() + off, out);
        });
      });
      const bool hok = std::memcmp(restored.get(), host.data(), raw) == 0;
      report("mans host", raw, total, hc, hd, hok);
      status |= hok ? 0 : 1;
    }
  }
  return status;
}
