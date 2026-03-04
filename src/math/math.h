#pragma once

#include <cmath>

namespace spheni::math {

namespace kernels {

inline float dot(const float *a, const float *b, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline float l2_squared(const float *a, const float *b, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

inline void normalize(float *v, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += v[i] * v[i];
  }
  const float norm = std::sqrt(sum);
  for (int i = 0; i < d; ++i) {
    v[i] /= norm;
  }
}

} // namespace kernels
} // namespace spheni::math
