/**
 * tools/test_checkpoint_io.cpp — checkpoints round-trip exactly, without copies of the model in RAM.
 *
 *   1. save_async + load and save + load: every parameter comes back bit-identical.
 *   2. Peak host memory added by one save_async of an N-byte model is below 2.5 N.
 *      The byte-buffer path (ostringstream, its string, a byte vector, plus the
 *      snapshot) needs about 4 N: on Colab a 7B DoRA save hit 69.6 GB RSS and was
 *      OOM-killed. Run with OLMO_CKPT_BYTES=1 to force that path and watch this fail.
 *
 *   ./build/test_checkpoint_io [--mb 512]
 *
 * Peak RSS is process-wide and only grows, so the memory check runs first, on a
 * fresh process high-water mark.
 */
#include "olmo_cpp/io/filesystem.hpp"  // CheckpointManager's unique_ptr<FileSystem> needs the full type
#include "olmo_cpp/train/checkpoint.hpp"

#include <torch/torch.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include <string>

using namespace olmo_cpp;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}
double peak_rss_mb() {
  struct rusage u {};
  getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
  return u.ru_maxrss / (1024.0 * 1024.0);  // bytes
#else
  return u.ru_maxrss / 1024.0;             // kilobytes
#endif
}
struct Blob : torch::nn::Module {
  std::vector<torch::Tensor> ps;
  Blob(int n, int64_t numel_each, double fill) {
    for (int i = 0; i < n; ++i)
      ps.push_back(register_parameter("p" + std::to_string(i),
                                      torch::full({numel_each}, fill + i, torch::kFloat32)));
  }
};
bool same(Blob& a, Blob& b) {
  for (size_t i = 0; i < a.ps.size(); ++i)
    if (!torch::equal(a.ps[i], b.ps[i])) return false;
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  int64_t mb = 512;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--mb") mb = std::stoll(argv[++i]);
  const int n = 8;
  const int64_t each = mb * 1024 * 1024 / 4 / n;
  const double model_mb = static_cast<double>(n * each * 4) / (1024.0 * 1024.0);
  auto root = std::filesystem::temp_directory_path() / ("ckpt_io_" + std::to_string(::getpid()));
  std::filesystem::create_directories(root);

  {
    Blob src(n, each, 1.0);
    for (auto& p : src.ps) p.data().add_(torch::randn_like(p));  // not constant: a real round trip
    CheckpointManager mgr(root.string());
    CheckpointMetadata meta;
    meta.step = 7;
    torch::optim::AdamW opt(src.parameters(), torch::optim::AdamWOptions(1e-3));

    const double before = peak_rss_mb();
    mgr.save_async("step_7", src, opt, meta, 0, 1).wait();
    const double added = peak_rss_mb() - before;
    char b[220];
    std::snprintf(b, sizeof b, "save_async of a %.0f MB model adds %.0f MB peak RSS (%.2fx the model; limit 2.5x)",
                  model_mb, added, added / model_mb);
    check(added < 2.5 * model_mb, b);

    Blob dst(n, each, -5.0);
    check(!same(src, dst), "fresh model differs before load (the equality check can fail)");
    auto m = mgr.load("step_7", dst, opt, 0, 1);
    check(same(src, dst) && m.step == 7, "save_async -> load: every parameter bit-identical, step restored");

    mgr.save("step_8", src, opt, meta, 0, 1);
    Blob dst2(n, each, -9.0);
    mgr.load("step_8", dst2, opt, 0, 1);
    check(same(src, dst2), "save (chunked) -> load: every parameter bit-identical");
  }
  std::filesystem::remove_all(root);
  std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
  return g_fail ? 1 : 0;
}
