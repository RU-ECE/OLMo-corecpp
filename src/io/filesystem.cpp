#include "olmo_cpp/io/filesystem.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>

namespace fs = std::filesystem;

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// URI parsing
// ---------------------------------------------------------------------------

std::pair<std::string, std::string> parse_uri(const std::string& uri) {
  auto pos = uri.find("://");
  if (pos == std::string::npos) return {"", uri};  // local path
  return {uri.substr(0, pos), uri.substr(pos + 3)};
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<FileSystem> FileSystem::create(const std::string& uri) {
  auto [scheme, rest] = parse_uri(uri);
  if (scheme.empty() || scheme == "file") {
    return std::make_unique<LocalFileSystem>();
  } else if (scheme == "s3") {
    auto slash = rest.find('/');
    if (slash == std::string::npos) return std::make_unique<S3FileSystem>(rest);
    return std::make_unique<S3FileSystem>(rest.substr(0, slash), rest.substr(slash + 1));
  } else if (scheme == "gs") {
    auto slash = rest.find('/');
    if (slash == std::string::npos) return std::make_unique<GCSFileSystem>(rest);
    return std::make_unique<GCSFileSystem>(rest.substr(0, slash), rest.substr(slash + 1));
  } else if (scheme == "http" || scheme == "https") {
    return std::make_unique<HTTPFileSystem>(uri);
  }
  throw std::runtime_error("Unsupported URI scheme: " + scheme);
}

// ---------------------------------------------------------------------------
// Helper: run shell command, capture stdout
// ---------------------------------------------------------------------------

static int run_command(const std::string& cmd, std::string* output) {
  std::array<char, 4096> buf;
  std::string result;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return -1;
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    result += buf.data();
  }
  int status = pclose(pipe);
  if (output) *output = std::move(result);
  return WEXITSTATUS(status);
}

// ---------------------------------------------------------------------------
// LocalFileSystem
// ---------------------------------------------------------------------------

std::vector<uint8_t> LocalFileSystem::read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("Cannot open file: " + path);
  auto sz = in.tellg();
  in.seekg(0);
  std::vector<uint8_t> data(static_cast<size_t>(sz));
  in.read(reinterpret_cast<char*>(data.data()), sz);
  return data;
}

void LocalFileSystem::write_file(const std::string& path, const std::vector<uint8_t>& data) {
  fs::create_directories(fs::path(path).parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot write file: " + path);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

bool LocalFileSystem::exists(const std::string& path) {
  return fs::exists(path);
}

std::vector<std::string> LocalFileSystem::list_dir(const std::string& path) {
  std::vector<std::string> entries;
  if (!fs::exists(path)) return entries;
  for (auto& e : fs::directory_iterator(path)) {
    entries.push_back(e.path().filename().string());
  }
  std::sort(entries.begin(), entries.end());
  return entries;
}

void LocalFileSystem::mkdir(const std::string& path) {
  fs::create_directories(path);
}

void LocalFileSystem::remove(const std::string& path) {
  fs::remove_all(path);
}

size_t LocalFileSystem::file_size(const std::string& path) {
  return static_cast<size_t>(fs::file_size(path));
}

// ---------------------------------------------------------------------------
// S3FileSystem
// ---------------------------------------------------------------------------

S3FileSystem::S3FileSystem(std::string bucket, std::string prefix)
    : bucket_(std::move(bucket)), prefix_(std::move(prefix)) {}

std::string S3FileSystem::s3_uri(const std::string& path) const {
  std::string full = "s3://" + bucket_;
  if (!prefix_.empty()) full += "/" + prefix_;
  if (!path.empty()) full += "/" + path;
  return full;
}

int S3FileSystem::run_cli(const std::string& cmd, std::string* output) const {
  return run_command(cmd, output);
}

std::vector<uint8_t> S3FileSystem::read_file(const std::string& path) {
  std::string tmp = "/tmp/olmo_s3_" + std::to_string(std::hash<std::string>{}(path));
  std::string cmd = "aws s3 cp " + s3_uri(path) + " " + tmp + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("S3 read failed: " + path);
  LocalFileSystem local;
  auto data = local.read_file(tmp);
  std::remove(tmp.c_str());
  return data;
}

void S3FileSystem::write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::string tmp = "/tmp/olmo_s3_" + std::to_string(std::hash<std::string>{}(path));
  LocalFileSystem local;
  local.write_file(tmp, data);
  std::string cmd = "aws s3 cp " + tmp + " " + s3_uri(path) + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("S3 write failed: " + path);
  std::remove(tmp.c_str());
}

bool S3FileSystem::exists(const std::string& path) {
  std::string cmd = "aws s3 ls " + s3_uri(path) + " 2>/dev/null";
  return run_cli(cmd) == 0;
}

std::vector<std::string> S3FileSystem::list_dir(const std::string& path) {
  std::string output;
  std::string cmd = "aws s3 ls " + s3_uri(path) + "/ 2>/dev/null";
  run_cli(cmd, &output);
  std::vector<std::string> entries;
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    // aws s3 ls output: "2024-01-01 00:00:00  1234 filename" or "PRE dirname/"
    auto last_space = line.rfind(' ');
    if (last_space != std::string::npos) {
      std::string name = line.substr(last_space + 1);
      if (!name.empty() && name.back() == '/') name.pop_back();
      if (!name.empty()) entries.push_back(name);
    }
  }
  return entries;
}

void S3FileSystem::mkdir(const std::string& /*path*/) {
  // S3 doesn't have real directories; no-op
}

void S3FileSystem::remove(const std::string& path) {
  std::string cmd = "aws s3 rm --recursive " + s3_uri(path) + " 2>/dev/null";
  run_cli(cmd);
}

size_t S3FileSystem::file_size(const std::string& path) {
  std::string output;
  std::string cmd = "aws s3 ls " + s3_uri(path) + " 2>/dev/null";
  run_cli(cmd, &output);
  // Parse "2024-01-01 00:00:00  1234 filename"
  std::istringstream iss(output);
  std::string d, t;
  size_t sz = 0;
  iss >> d >> t >> sz;
  return sz;
}

// ---------------------------------------------------------------------------
// GCSFileSystem
// ---------------------------------------------------------------------------

GCSFileSystem::GCSFileSystem(std::string bucket, std::string prefix)
    : bucket_(std::move(bucket)), prefix_(std::move(prefix)) {}

std::string GCSFileSystem::gs_uri(const std::string& path) const {
  std::string full = "gs://" + bucket_;
  if (!prefix_.empty()) full += "/" + prefix_;
  if (!path.empty()) full += "/" + path;
  return full;
}

int GCSFileSystem::run_cli(const std::string& cmd, std::string* output) const {
  return run_command(cmd, output);
}

std::vector<uint8_t> GCSFileSystem::read_file(const std::string& path) {
  std::string tmp = "/tmp/olmo_gcs_" + std::to_string(std::hash<std::string>{}(path));
  std::string cmd = "gsutil cp " + gs_uri(path) + " " + tmp + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("GCS read failed: " + path);
  LocalFileSystem local;
  auto data = local.read_file(tmp);
  std::remove(tmp.c_str());
  return data;
}

void GCSFileSystem::write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::string tmp = "/tmp/olmo_gcs_" + std::to_string(std::hash<std::string>{}(path));
  LocalFileSystem local;
  local.write_file(tmp, data);
  std::string cmd = "gsutil cp " + tmp + " " + gs_uri(path) + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("GCS write failed: " + path);
  std::remove(tmp.c_str());
}

bool GCSFileSystem::exists(const std::string& path) {
  std::string cmd = "gsutil ls " + gs_uri(path) + " 2>/dev/null";
  return run_cli(cmd) == 0;
}

std::vector<std::string> GCSFileSystem::list_dir(const std::string& path) {
  std::string output;
  std::string cmd = "gsutil ls " + gs_uri(path) + "/ 2>/dev/null";
  run_cli(cmd, &output);
  std::vector<std::string> entries;
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    auto last_slash = line.rfind('/');
    if (last_slash != std::string::npos && last_slash < line.size() - 1) {
      entries.push_back(line.substr(last_slash + 1));
    } else if (last_slash == line.size() - 1 && line.size() > 1) {
      auto prev_slash = line.rfind('/', last_slash - 1);
      if (prev_slash != std::string::npos) {
        entries.push_back(line.substr(prev_slash + 1, last_slash - prev_slash - 1));
      }
    }
  }
  return entries;
}

void GCSFileSystem::mkdir(const std::string& /*path*/) {
  // GCS doesn't have real directories
}

void GCSFileSystem::remove(const std::string& path) {
  std::string cmd = "gsutil -m rm -r " + gs_uri(path) + " 2>/dev/null";
  run_cli(cmd);
}

size_t GCSFileSystem::file_size(const std::string& path) {
  std::string output;
  std::string cmd = "gsutil du -s " + gs_uri(path) + " 2>/dev/null";
  run_cli(cmd, &output);
  size_t sz = 0;
  std::istringstream iss(output);
  iss >> sz;
  return sz;
}

// ---------------------------------------------------------------------------
// HTTPFileSystem (read-only)
// ---------------------------------------------------------------------------

HTTPFileSystem::HTTPFileSystem(std::string base_url) : base_url_(std::move(base_url)) {
  // Strip trailing slash
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
}

std::vector<uint8_t> HTTPFileSystem::curl_get(const std::string& url) const {
  std::string tmp = "/tmp/olmo_http_" + std::to_string(std::hash<std::string>{}(url));
  std::string cmd = "curl -sS -f -o " + tmp + " '" + url + "' 2>/dev/null";
  if (run_command(cmd, nullptr) != 0) {
    throw std::runtime_error("HTTP GET failed: " + url);
  }
  LocalFileSystem local;
  auto data = local.read_file(tmp);
  std::remove(tmp.c_str());
  return data;
}

std::vector<uint8_t> HTTPFileSystem::read_file(const std::string& path) {
  std::string url = base_url_;
  if (!path.empty()) url += "/" + path;
  return curl_get(url);
}

void HTTPFileSystem::write_file(const std::string& /*path*/, const std::vector<uint8_t>& /*data*/) {
  throw std::runtime_error("HTTPFileSystem is read-only");
}

bool HTTPFileSystem::exists(const std::string& path) {
  std::string url = base_url_;
  if (!path.empty()) url += "/" + path;
  std::string cmd = "curl -sS -f -I '" + url + "' >/dev/null 2>&1";
  return run_command(cmd, nullptr) == 0;
}

std::vector<std::string> HTTPFileSystem::list_dir(const std::string& /*path*/) {
  throw std::runtime_error("HTTPFileSystem does not support list_dir");
}

void HTTPFileSystem::mkdir(const std::string& /*path*/) {
  throw std::runtime_error("HTTPFileSystem is read-only");
}

void HTTPFileSystem::remove(const std::string& /*path*/) {
  throw std::runtime_error("HTTPFileSystem is read-only");
}

size_t HTTPFileSystem::file_size(const std::string& path) {
  std::string url = base_url_;
  if (!path.empty()) url += "/" + path;
  std::string output;
  std::string cmd = "curl -sS -I '" + url + "' 2>/dev/null | grep -i content-length";
  run_command(cmd, &output);
  // Parse "Content-Length: 12345"
  auto pos = output.find(':');
  if (pos == std::string::npos) return 0;
  return static_cast<size_t>(std::stoull(output.substr(pos + 1)));
}

}  // namespace olmo_cpp
