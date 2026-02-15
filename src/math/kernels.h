#pragma once

namespace spheni::math::kernels {
float dot(const float *a, const float *b, int d);
float l2_squared(const float *a, const float *b, int d);
float haversine(const float *a, const float *b, int d);
void normalize(float *v, int d);
float l2_norm(const float *v, int d);
} // namespace spheni::math::kernels
