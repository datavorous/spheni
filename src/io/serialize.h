// Code Generated using GPT 3.5 Codex]
// Verified by author::datavorous

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace spheni::io {

template <typename T> inline void write_pod(std::ostream &out, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
  if (!out) {
    throw std::runtime_error("Failed to write binary data.");
  }
}

template <typename T> inline T read_pod(std::istream &in) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("Failed to read binary data.");
  }
  return value;
}

inline void write_bool(std::ostream &out, bool value) {
  std::uint8_t v = value ? 1 : 0;
  write_pod(out, v);
}

inline bool read_bool(std::istream &in) {
  std::uint8_t v = read_pod<std::uint8_t>(in);
  if (v > 1) {
    throw std::runtime_error("Invalid boolean value in binary data.");
  }
  return v == 1;
}

template <typename T>
inline void write_vector(std::ostream &out, const std::vector<T> &data) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::uint64_t size = static_cast<std::uint64_t>(data.size());
  write_pod(out, size);
  if (size == 0) {
    return;
  }
  out.write(reinterpret_cast<const char *>(data.data()), sizeof(T) * size);
  if (!out) {
    throw std::runtime_error("Failed to write vector data.");
  }
}

template <typename T> inline std::vector<T> read_vector(std::istream &in) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::uint64_t size = read_pod<std::uint64_t>(in);
  std::vector<T> data(size);
  if (size == 0) {
    return data;
  }
  in.read(reinterpret_cast<char *>(data.data()), sizeof(T) * size);
  if (!in) {
    throw std::runtime_error("Failed to read vector data.");
  }
  return data;
}

} // namespace spheni::io