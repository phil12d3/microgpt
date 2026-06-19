#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {
inline constexpr size_t kMaxInputFileBytes = 256ull * 1024ull * 1024ull;

inline std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + path);
  }
  in.seekg(0, std::ios::end);
  std::streamsize size = in.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to determine input file size: " + path);
  }
  if (static_cast<size_t>(size) > kMaxInputFileBytes) {
    throw std::runtime_error("input file is too large: " + path);
  }
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (size > 0 && !in.read(reinterpret_cast<char*>(data.data()), size)) {
    throw std::runtime_error("failed to read input file: " + path);
  }
  return data;
}

inline std::string read_file_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path);
  }
  in.seekg(0, std::ios::end);
  std::streamsize size = in.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to determine file size: " + path);
  }
  if (static_cast<size_t>(size) > kMaxInputFileBytes) {
    throw std::runtime_error("file is too large: " + path);
  }
  in.seekg(0, std::ios::beg);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

inline void write_file_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to write file: " + path);
  }
  out << text;
}

inline std::vector<int> bytes_to_tokens(const std::vector<uint8_t>& bytes, int vocab_size = 260) {
  if (vocab_size < 256) {
    throw std::runtime_error("byte tokenizer requires vocab_size >= 256");
  }
  std::vector<int> tokens;
  tokens.reserve(bytes.size());
  for (uint8_t b : bytes) {
    tokens.push_back(static_cast<int>(b));
  }
  return tokens;
}

}  // namespace microgpt
