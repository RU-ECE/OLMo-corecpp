#include "zwt/train/checkpoint.hpp"
#include "zwt/core/stream.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt::train {

namespace {

constexpr char     kMagic[8]     = {'Z','W','T','C','K','P','T','1'};
constexpr uint32_t kVersion      = 1;
constexpr size_t   kHeaderBytes  = 64;
constexpr size_t   kTRecBytes    = 80;   // tensor record header — fixed size

// File header (kHeaderBytes). Everything else is tensor records.
//
// [0..8)    magic                 "ZWTCKPT1"
// [8..12)   version               u32
// [12..16)  flags                 u32   (reserved)
// [16..24)  step                  i64
// [24..32)  data_cursor           i64
// [32..40)  seed                  u64
// [40..44)  n_records             i32
// [44..48)  pad
// [48..52)  lr                    f32
// [52..56)  loss                  f32
// [56..64)  reserved              u64
#pragma pack(push, 1)
struct FileHeader {
  char     magic[8];
  uint32_t version;
  uint32_t flags;
  int64_t  step;
  int64_t  data_cursor;
  uint64_t seed;
  int32_t  n_records;
  uint32_t pad0;
  float    lr;
  float    loss;
  uint64_t reserved;
};
static_assert(sizeof(FileHeader) == kHeaderBytes, "FileHeader size");

// Tensor record header (kTRecBytes). Followed by name bytes (padded to 8)
// then data bytes (padded to 8).
//
// [0..4)    name_len              u32
// [4..5)    dtype                 u8
// [5..6)    rank                  u8
// [6..8)    pad
// [8..16)   nbytes                u64
// [16..64)  dims[6]               i64 * 6
// [64..72)  reserved
// [72..80)  reserved
struct TRec {
  uint32_t name_len;
  uint8_t  dtype;
  uint8_t  rank;
  uint16_t pad0;
  uint64_t nbytes;
  int64_t  dims[6];
  uint64_t reserved0;
  uint64_t reserved1;
};
static_assert(sizeof(TRec) == kTRecBytes, "TRec size");
#pragma pack(pop)

inline size_t pad_to_8(size_t n) { return (n + 7u) & ~size_t{7}; }

void write_bytes(std::ofstream& f, const void* p, size_t n) {
  if (!f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n))) {
    throw std::runtime_error("checkpoint: write failed");
  }
}

void read_bytes(std::ifstream& f, void* p, size_t n) {
  if (!f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n))) {
    throw std::runtime_error("checkpoint: read failed or truncated");
  }
}

void write_pad_to_8(std::ofstream& f, size_t bytes_written) {
  size_t rem = pad_to_8(bytes_written) - bytes_written;
  if (rem == 0) return;
  char zeros[8] = {};
  write_bytes(f, zeros, rem);
}

void skip_pad_to_8(std::ifstream& f, size_t bytes_read) {
  size_t rem = pad_to_8(bytes_read) - bytes_read;
  if (rem == 0) return;
  f.seekg(static_cast<std::streamoff>(rem), std::ios::cur);
}

// Stage a tensor's bytes into a host buffer (downloading from device if
// needed). Synchronous on the compute stream.
std::vector<uint8_t> tensor_to_host(const Tensor& t) {
  std::vector<uint8_t> buf(t.nbytes());
  if (t.nbytes() == 0) return buf;
  if (t.device().is_cuda()) {
#ifdef USE_CUDA
    cudaStream_t s =
        reinterpret_cast<cudaStream_t>(compute_stream(t.device()).handle);
    cudaMemcpyAsync(buf.data(), t.data(), t.nbytes(),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
#else
    throw std::runtime_error("checkpoint: CUDA tensor on CPU-only build");
#endif
  } else {
    std::memcpy(buf.data(), t.data(), t.nbytes());
  }
  return buf;
}

// Upload a host buffer into an existing tensor (or host-copy if CPU).
void tensor_from_host(Tensor& t, const void* src, size_t n) {
  if (n != t.nbytes()) {
    throw std::runtime_error("checkpoint: tensor size mismatch on load");
  }
  if (n == 0) return;
  if (t.device().is_cuda()) {
#ifdef USE_CUDA
    cudaStream_t s =
        reinterpret_cast<cudaStream_t>(compute_stream(t.device()).handle);
    cudaMemcpyAsync(t.data(), src, n, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
#else
    throw std::runtime_error("checkpoint: CUDA tensor on CPU-only build");
#endif
  } else {
    std::memcpy(t.data(), src, n);
  }
}

void fill_record(TRec& r, const Tensor& t, size_t name_len) {
  r.name_len = static_cast<uint32_t>(name_len);
  r.dtype    = static_cast<uint8_t>(t.dtype());
  r.rank     = static_cast<uint8_t>(t.rank());
  r.pad0     = 0;
  r.nbytes   = t.nbytes();
  for (int i = 0; i < 6; ++i) {
    r.dims[i] = (i < t.rank()) ? t.dim(i) : 0;
  }
  r.reserved0 = 0;
  r.reserved1 = 0;
}

void write_tensor(std::ofstream& f, const std::string& name, const Tensor& t) {
  TRec r;
  fill_record(r, t, name.size());
  write_bytes(f, &r, sizeof(r));
  write_bytes(f, name.data(), name.size());
  write_pad_to_8(f, name.size());

  auto host = tensor_to_host(t);
  write_bytes(f, host.data(), host.size());
  write_pad_to_8(f, host.size());
}

// Validate a record against an expected live tensor. Throws on any mismatch.
void validate_record(const std::string& name, const TRec& r, const Tensor& t) {
  if (r.dtype != static_cast<uint8_t>(t.dtype())) {
    throw std::runtime_error("checkpoint: dtype mismatch for '" + name + "'");
  }
  if (r.rank != static_cast<uint8_t>(t.rank())) {
    throw std::runtime_error("checkpoint: rank mismatch for '" + name + "'");
  }
  for (int i = 0; i < t.rank(); ++i) {
    if (r.dims[i] != t.dim(i)) {
      throw std::runtime_error("checkpoint: shape mismatch for '" + name + "'");
    }
  }
  if (r.nbytes != t.nbytes()) {
    throw std::runtime_error("checkpoint: nbytes mismatch for '" + name + "'");
  }
}

}  // namespace

void save_checkpoint(const std::string& path,
                     const std::vector<Parameter*>& params,
                     optim::AdamW& opt,
                     const CheckpointMeta& meta) {
  if (params.size() != opt.n_params()) {
    throw std::runtime_error("checkpoint: param count / optimizer size mismatch");
  }

  const std::string tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("checkpoint: cannot open " + tmp);

  FileHeader h{};
  std::memcpy(h.magic, kMagic, 8);
  h.version     = kVersion;
  h.flags       = 0;
  h.step        = meta.step;
  h.data_cursor = meta.data_cursor;
  h.seed        = meta.seed;
  h.n_records   = static_cast<int32_t>(params.size() * 3);
  h.pad0        = 0;
  h.lr          = meta.lr;
  h.loss        = meta.loss;
  h.reserved    = 0;
  write_bytes(f, &h, sizeof(h));

  // Three records per param: value, .m, .v
  for (size_t i = 0; i < params.size(); ++i) {
    const auto* p = params[i];
    write_tensor(f, p->name,          p->value);
    write_tensor(f, p->name + ".m",   opt.moment_m(i));
    write_tensor(f, p->name + ".v",   opt.moment_v(i));
  }

  f.close();
  if (!f) throw std::runtime_error("checkpoint: error closing " + tmp);

  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    throw std::runtime_error("checkpoint: rename " + tmp + " -> " + path + " failed");
  }
}

CheckpointMeta read_checkpoint_meta(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
  FileHeader h{};
  read_bytes(f, &h, sizeof(h));
  if (std::memcmp(h.magic, kMagic, 8) != 0) {
    throw std::runtime_error("checkpoint: bad magic in " + path);
  }
  if (h.version != kVersion) {
    throw std::runtime_error("checkpoint: unsupported version");
  }
  CheckpointMeta m;
  m.step        = h.step;
  m.seed        = h.seed;
  m.data_cursor = h.data_cursor;
  m.lr          = h.lr;
  m.loss        = h.loss;
  return m;
}

CheckpointMeta load_checkpoint(const std::string& path,
                               const std::vector<Parameter*>& params,
                               optim::AdamW& opt) {
  if (params.size() != opt.n_params()) {
    throw std::runtime_error("checkpoint: param count / optimizer size mismatch");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("checkpoint: cannot open " + path);

  FileHeader h{};
  read_bytes(f, &h, sizeof(h));
  if (std::memcmp(h.magic, kMagic, 8) != 0) {
    throw std::runtime_error("checkpoint: bad magic in " + path);
  }
  if (h.version != kVersion) {
    throw std::runtime_error("checkpoint: unsupported version");
  }

  // Build a lookup table from param name -> (index, kind).
  //   kind: 0 = value, 1 = moment m, 2 = moment v
  struct Slot { size_t idx; int kind; };
  std::unordered_map<std::string, Slot> table;
  table.reserve(params.size() * 3);
  for (size_t i = 0; i < params.size(); ++i) {
    table[params[i]->name]         = {i, 0};
    table[params[i]->name + ".m"]  = {i, 1};
    table[params[i]->name + ".v"]  = {i, 2};
  }

  std::vector<uint8_t> scratch;

  int32_t n_records_expected = static_cast<int32_t>(params.size() * 3);
  if (h.n_records != n_records_expected) {
    throw std::runtime_error("checkpoint: record count mismatch (file "
                             + std::to_string(h.n_records) + ", expected "
                             + std::to_string(n_records_expected) + ")");
  }

  std::vector<char> seen(params.size() * 3, 0);

  for (int32_t r = 0; r < h.n_records; ++r) {
    TRec rec;
    read_bytes(f, &rec, sizeof(rec));

    std::string name(rec.name_len, '\0');
    read_bytes(f, name.data(), rec.name_len);
    skip_pad_to_8(f, rec.name_len);

    auto it = table.find(name);
    if (it == table.end()) {
      // Unknown record — skip over its bytes.
      f.seekg(static_cast<std::streamoff>(pad_to_8(rec.nbytes)), std::ios::cur);
      continue;
    }

    Tensor* target = nullptr;
    switch (it->second.kind) {
      case 0: target = &params[it->second.idx]->value; break;
      case 1: target = &opt.moment_m(it->second.idx); break;
      case 2: target = &opt.moment_v(it->second.idx); break;
    }
    validate_record(name, rec, *target);

    if (scratch.size() < rec.nbytes) scratch.resize(rec.nbytes);
    read_bytes(f, scratch.data(), rec.nbytes);
    skip_pad_to_8(f, rec.nbytes);
    tensor_from_host(*target, scratch.data(), rec.nbytes);

    size_t slot_idx = it->second.idx * 3 + static_cast<size_t>(it->second.kind);
    seen[slot_idx] = 1;
  }

  for (size_t i = 0; i < seen.size(); ++i) {
    if (!seen[i]) {
      size_t param_idx = i / 3;
      int    kind      = static_cast<int>(i % 3);
      const char* suffix = (kind == 0) ? "" : (kind == 1 ? ".m" : ".v");
      throw std::runtime_error("checkpoint: missing record for '"
                               + params[param_idx]->name + suffix + "'");
    }
  }

  opt.set_step_count(h.step);

  CheckpointMeta m;
  m.step        = h.step;
  m.seed        = h.seed;
  m.data_cursor = h.data_cursor;
  m.lr          = h.lr;
  m.loss        = h.loss;
  return m;
}

}  // namespace zwt::train
