#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace spheni::quantization {

inline float compute_scale(const float* v, int d) {
    float max_abs = 0.0f;
    for (int i = 0; i < d; ++i) {
        float abs_val = std::abs(v[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    if (max_abs == 0.0f) {
        return 1.0f;
    }
    return max_abs / 127.0f;
}

inline std::int8_t quantize_value(float x, float scale) {
    float q = std::round(x / scale);
    q = std::clamp(q, -127.0f, 127.0f);
    return static_cast<std::int8_t>(q);
}

inline void quantize_vector(const float* v, int d, std::vector<std::int8_t>& out, std::vector<float>& scales) {
    float scale = compute_scale(v, d);
    std::size_t offset = out.size();
    out.resize(offset + static_cast<std::size_t>(d));
    for (int i = 0; i < d; ++i) {
        out[offset + static_cast<std::size_t>(i)] = quantize_value(v[i], scale);
    }
    scales.push_back(scale);
}

}
