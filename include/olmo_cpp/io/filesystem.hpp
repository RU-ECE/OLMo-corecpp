#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <utility>

namespace olmo_cpp {

/// Abstract filesystem interface for local, S3, GCS, HTTP backends
class FileSystem {
 public:
  virtual ~FileSystem() = default;

  virtual std::vector<uint8_t> read_file(const std::string& path) = 0;
  virtual void write_file(const std::string& path, const std::vector<uint8_t>& data) = 0;
  virtual bool exists(const std::string& path) = 0;
  virtual std::vector<std::string> list_dir(const std::string& path) = 0;
  virtual void mkdir(const std::string& path) = 0;
  virtual void remove(const std::string& path) = 0;
  virtual size_t file_size(const std::string& path) = 0;

  /// Factory: create filesystem for a given URI (auto-detects scheme)
  static std::unique_ptr<FileSystem> create(const std::string& uri);
};

/// Local filesystem using std::filesystem
class LocalFileSystem : public FileSystem {
 public:
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
};

/// S3 filesystem via AWS CLI
class S3FileSystem : public FileSystem {
 public:
  S3FileSystem(std::string bucket, std::string prefix = "");
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
 private:
  std::string bucket_, prefix_;
  std::string s3_uri(const std::string& path) const;
  int run_cli(const std::string& cmd, std::string* output = nullptr) const;
};

/// GCS filesystem via gsutil
class GCSFileSystem : public FileSystem {
 public:
  GCSFileSystem(std::string bucket, std::string prefix = "");
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
 private:
  std::string bucket_, prefix_;
  std::string gs_uri(const std::string& path) const;
  int run_cli(const std::string& cmd, std::string* output = nullptr) const;
};

/// HTTP filesystem (read-only via curl)
class HTTPFileSystem : public FileSystem {
 public:
  explicit HTTPFileSystem(std::string base_url);
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
 private:
  std::string base_url_;
  std::vector<uint8_t> curl_get(const std::string& url) const;
};

/// Parse URI scheme: returns {scheme, path} e.g. {"s3", "bucket/key"} or {"", "/local/path"}
std::pair<std::string, std::string> parse_uri(const std::string& uri);

}  // namespace olmo_cpp
