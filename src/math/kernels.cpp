#include "math/kernels.h"
#include <algorithm>
#include <cmath>
#include <numbers>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace spheni::math::kernels {
float haversine(const float *a, const float *b, int d) {
  // Input vectors are [lat, lon] in degrees.
  (void)d; // unused, assumed to be 2 validated by caller

  const float R = 6371.0f; // Earth radius in km
  const float to_rad = static_cast<float>(M_PI / 180.0);

  float lat1 = a[0] * to_rad;
  float lon1 = a[1] * to_rad;
  float lat2 = b[0] * to_rad;
  float lon2 = b[1] * to_rad;

  float dlat = lat2 - lat1;
  float dlon = lon2 - lon1;

  float a_hav = std::sin(dlat / 2) * std::sin(dlat / 2) +
                std::cos(lat1) * std::cos(lat2) * std::sin(dlon / 2) *
                    std::sin(dlon / 2);

  // Clamp a_hav to [0, 1] to avoid NaNs from floating point errors
  if (a_hav < 0.0f)
    a_hav = 0.0f;
  if (a_hav > 1.0f)
    a_hav = 1.0f;

  float c = 2 * std::atan2(std::sqrt(a_hav), std::sqrt(1 - a_hav));

  return R * c;
}

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