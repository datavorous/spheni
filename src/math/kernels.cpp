#include "math/kernels.h"
#include <algorithm>
#include <cmath>

namespace spheni::math::kernels {
float dot(const float *a, const float *b, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2_squared(const float *a, const float *b, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float l2_norm(const float *v, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += v[i] * v[i];
  }
  return std::sqrt(sum);
}

void normalize(float *v, int d) {
  float norm = l2_norm(v, d);
  for (int i = 0; i < d; ++i) {
    v[i] /= norm;
  }
}
} // namespace spheni::math::kernels