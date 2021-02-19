// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <iostream>
#include <details/ie_exception.hpp>
#include "quantization.h"
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

void QuantizeAffine16(float *ptr_float_weights,
                      float *ptr_float_biases,
                      int16_t *ptr_int_weights,
                      int32_t *ptr_int_biases,
                      float input_scale_factor,
                      float *ptr_weight_scale_factor,
                      float *ptr_output_scale_factor,
                      uint32_t num_rows,
                      uint32_t num_columns,
                      uint32_t num_rows_padded,
                      uint32_t num_columns_padded) {
    uint32_t num_saturate = 0;

/*    if (*ptr_weight_scale_factor == 1.0) {
        // scale factor for weights is not calculated yet
        float mean_weight = 0.0;
        float mean_weight_squared = 0.0;
        float max_weight = -1e20f;
        float var_weight;
        float mean_plus_2stdev;

        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_columns; j++) {
                float weight = ptr_float_weights[i * num_columns + j];
                mean_weight += weight;
                mean_weight_squared += weight * weight;
                if (fabs(weight) > max_weight) {
                    max_weight = fabs(weight);
                }
            }
        }

        mean_weight /= static_cast<float>(num_rows * num_columns);
        mean_weight_squared /= static_cast<float>(num_rows * num_columns);
        var_weight = mean_weight_squared - mean_weight * mean_weight;
        mean_plus_2stdev = mean_weight + 2.0f * static_cast<float>(sqrtf(var_weight));

        if (max_weight != 0.0f) {
            *ptr_weight_scale_factor = static_cast<float>(MAX_VAL_2B_WEIGHT) / max_weight;
        }
        *ptr_output_scale_factor = input_scale_factor * *ptr_weight_scale_factor;
    }*/

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col] * *ptr_weight_scale_factor + rounding_value;
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            if (value > 32767.0) {
                *ptr_weight_16 = 32767;
                num_saturate++;
            } else if (value < -32768.0) {
                *ptr_weight_16 = -32768;
                num_saturate++;
            } else {
                *ptr_weight_16 = (int16_t) value;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }

    // case for element wise layer
    if (ptr_float_biases != nullptr && ptr_int_biases != nullptr) {
        for (uint32_t j = 0; j < num_rows; j++) {
            float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
            if (value > 2147483647.0) {
                ptr_int_biases[j] = 2147483647L;
                num_saturate++;
            } else if (value < -2147483648.0) {
                ptr_int_biases[j] = -2147483648LL;
                num_saturate++;
            } else {
                ptr_int_biases[j] = (int32_t) value;
            }
        }
        for (uint32_t j = num_rows; j < num_rows_padded; j++) {
            ptr_int_biases[j] = 0;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine16()\n",
                     num_saturate,
                     num_rows * num_columns + num_rows);
    }
}

__attribute__ ((target ("default")))
float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements)
{
    float *ptr_float_feat = (float*)ptr_float_memory;
    float min = 0.0;
    float buf[4];
    float scale_factor;
    float abs_f;
    __m128 zero = _mm_setzero_ps();
    __m128 total_abs = _mm_setzero_ps();
    char not_aligned_buffer[32];
    __m128 v, neg_v, abs;
    __m128 v2, neg_v2, abs2;


    uint32_t moves = num_elements >>3;
    uint32_t mod = num_elements % 8;
    uint32_t i;

    v = _mm_load_ps(ptr_float_feat);
    neg_v = _mm_sub_ps(zero, v);
    abs = _mm_max_ps(v, neg_v);
    total_abs = _mm_max_ps(total_abs, abs);

    for (i = 0; i<moves; i++, ptr_float_feat +=8)
    {
        v = _mm_load_ps(ptr_float_feat);
        v2 = _mm_load_ps(ptr_float_feat+4);
        neg_v = _mm_sub_ps(zero, v);
        abs = _mm_max_ps(v, neg_v);
        neg_v2 = _mm_sub_ps(zero, v2);
        abs2 = _mm_max_ps(v2, neg_v2);
        total_abs = _mm_min_ps(total_abs, abs);
        total_abs = _mm_min_ps(total_abs, abs2);
    }
    _mm_storeu_ps(buf, total_abs);
    float single_min_lo = buf[0] < buf[1] ? buf[0] : buf[1];
    float single_min_hi = buf[2] < buf[3] ? buf[2] : buf[3];
    float single_min = single_min_lo < single_min_hi ? single_min_lo : single_min_hi;

    for (i = 0; i < mod; i++)
    {
        abs_f = fabs(ptr_float_feat[i]);
        if (abs_f < min) {
            min = abs_f;
        }
    }

    return(single_min != 0 ? (single_min < 1.0 ? 1 / single_min : 1.0f) : 1.0f);
}

/*__attribute__ ((target ("default")))
float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements) {

    float *ptr_float_feat = reinterpret_cast<float *>(ptr_float_memory);
    float max = 0.0;
    float scale_factor;
    float min = 0.0;
    size_t i;

    for ( i = 0;i < num_elements;i++) {
        min = fabs(ptr_float_feat[i]);
        if (min != 0.0f) {
            break;
        }
    }

    for (; i < num_elements; i++) {
        if (fabs(ptr_float_feat[i]) < min && ptr_float_feat[i] != 0) {
            min = fabs(ptr_float_feat[i]);
        }
    }

    if (min == 0.0) {
        scale_factor = 1.0f;
    } else {
        if( min < 1) {
            scale_factor = 1 / min;
        }
        else {
            scale_factor = 1.0f;
        }
    }
    return scale_factor;
}*/

float accessmember(__m128 v, int index)
{
    union vec{ __m128 sse;
        float f[4];
    };

    vec U;
    U.sse = v;
    return U.f[index];
}

__attribute__ ((target ("default")))
void QuantizeBias8(float *ptr_float_biases, intel_compound_bias_t  *ptr_int_biases, float *ptr_output_scale_factor, uint32_t num_rows)
{
    float *ptr_float_feat = (float*)ptr_float_biases;
    intel_compound_bias_t *ptr_int = (intel_compound_bias_t*)ptr_int_biases;

    uint32_t moves = num_rows / 4;
    uint32_t mod = num_rows % 4;
    uint32_t i, j;

    __m128 v, zero, half, neg_half, scale_factores, mask, rounding_values, min, max, values;

#ifdef ROUND_AND_CAST
    __m128i tmp;
#endif

    zero = _mm_setzero_ps();
    half = _mm_set1_ps(0.5f);
    neg_half = _mm_set1_ps(-0.5f);
    max = _mm_set1_ps(2147483647.0f);
    min = _mm_set1_ps(-2147483648.0f);

    scale_factores = _mm_set1_ps(*ptr_output_scale_factor);

    for (i = 0; i < moves; i++, ptr_float_feat += 4, ptr_int += 4) {

        v = _mm_load_ps(ptr_float_feat);

        //rounding_values = (v>0) ? 0.5f : -0.5f;
        mask = _mm_min_ps(v, zero);
        rounding_values = _mm_blendv_ps(half, neg_half, mask);

        // values = v * scale_factores +  rounding_values
        values = _mm_mul_ps(v, scale_factores);
        values = _mm_add_ps(values, rounding_values);

        // shrink to <-2147483648.0f, 2147483647.0f>
        values = _mm_min_ps(values, max);
        values = _mm_max_ps(values, min);

#ifdef ROUND_AND_CAST
        // round and cast float to int16 ... much faster  than "only cast" in MS compiler ??
        tmp = _mm_cvtps_epi32(values);
        ptr_int[0].bias = tmp.m128i_i32[0];
        ptr_int[1].bias = tmp.m128i_i32[1];
        ptr_int[2].bias = tmp.m128i_i32[2];
        ptr_int[3].bias = tmp.m128i_i32[3];
#else
        // only cast float to int16
        for (j = 0; j < 4; j++)
            ptr_int[j].bias = (int32_t)accessmember(values, j);
#endif
    }

    for (i = 0; i < mod; i++) {
        float rounding_value = (ptr_float_feat[i]>0) ? 0.5f : -0.5f;
        float value = ptr_float_feat[i] * *ptr_output_scale_factor + rounding_value;
        if (value > 2147483647.0) {
            ptr_int[i].bias = 2147483647L;
        }
        else if (value < -2147483648.0) {
            ptr_int[i].bias = -2147483648LL;
        }
        else {
            ptr_int[i].bias = (int32_t)value;
        }
    }
}

/*__attribute__ ((target ("default")))
void QuantizeBias8(float *ptr_float_biases, intel_compound_bias_t  *ptr_int_biases, float *ptr_output_scale_factor, uint32_t num_rows)
{
    uint32_t num_saturate = 0;
    for (uint32_t j = 0; j < num_rows; j++) {
        float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
        float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
        if (value > 2147483647.0) {
            ptr_int_biases[j].bias = 2147483647L;
            num_saturate++;
        } else if (value < -2147483648.0) {
            ptr_int_biases[j].bias = -2147483648LL;
            num_saturate++;
        } else {
            ptr_int_biases[j].bias = (int32_t) value;
        }
    }
}*/

void QuantizeVector16(float *ptr_float_memory, int16_t *ptr_int_memory, uint32_t num_elements, float scale_factor) {
    float *ptr_float_feat = reinterpret_cast<float *>(ptr_float_memory);
    uint32_t num_saturate = 0;

    int16_t *ptr_int_feat = reinterpret_cast<int16_t *>(ptr_int_memory);
    for (uint32_t i = 0; i < num_elements; i++) {
        float rounding_value = (ptr_float_feat[i] > 0) ? 0.5f : -0.5f;
        float value = ptr_float_feat[i] * scale_factor + rounding_value;
        if (value > 32767.0) {
            ptr_int_feat[i] = 32767;
            num_saturate++;
        } else if (value < -32768.0) {
            ptr_int_feat[i] = -32768;
            num_saturate++;
        } else {
            ptr_int_feat[i] = (int16_t) value;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations during QuantizeVector16()\n", num_saturate, num_elements);
    }
}

#include <fstream>
#include <sys/stat.h>

void createDirs(std::string path) {
    char delim = '/';
    int start = 0;

    auto pos = path.find(delim);
    while (pos != std::string::npos) {
		auto dir = path.substr(start, pos - start+1);

        struct stat sb;
        if (!((stat(dir.c_str(), &sb) == 0) && (S_ISDIR(sb.st_mode)))) {
            if (mkdir(dir.c_str(), 0777) != 0)
                std::cout << "failed to create folder: " << dir << std::endl;
		}
		pos = path.find(delim, pos+1);
	}
}

void QuantizeAffine8(float *ptr_float_weights, float *ptr_float_biases,
                     int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                     float input_scale_factor, float *ptr_weight_scale_factor,
                     float *ptr_output_scale_factor, uint32_t num_rows, uint32_t num_columns,
                     uint32_t num_rows_padded, uint32_t num_columns_padded) {
    if (ptr_int_biases == nullptr) {
        THROW_IE_EXCEPTION << "Int biases are empty";
    }
    uint32_t num_saturate = 0;

    /*if (*ptr_weight_scale_factor == 1.0) {
        // scale factor for weights is not calculated yet
        float mean_weight = 0.0;
        float mean_weight_squared = 0.0;
        float max_weight = -1e20f;
        float var_weight;
        float mean_plus_2stdev;

        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_columns; j++) {
                float weight = ptr_float_weights[i*num_columns + j];
                mean_weight += weight;
                mean_weight_squared += weight * weight;
                if (fabs(weight) > max_weight) {
                    max_weight = fabs(weight);
                }
            }
        }

        mean_weight /= static_cast<float>(num_rows * num_columns);
        mean_weight_squared /= static_cast<float>(num_rows * num_columns);
        var_weight = mean_weight_squared - mean_weight * mean_weight;
        mean_plus_2stdev = mean_weight + 2.0f * static_cast<float>(sqrtf(var_weight));

        *ptr_weight_scale_factor = static_cast<float>(MAX_VAL_1B_WEIGHT) / max_weight;

        // For 8 bit weights quantize as follows:
        // 1. adjust scale factor to increase dynamic range of entire matrix by max multiplier
        // 2. find maximum scaled weight for each row
        // 3. find multiplier such that dividing by the multiplier brings row back within 8-bit dynamic range
        // 4. quantize and store scaled row
        *ptr_weight_scale_factor = MAX_OUT_MULTIPLIER * *ptr_weight_scale_factor;  //  increase dynamic range by max multiplier
        *ptr_output_scale_factor = input_scale_factor * *ptr_weight_scale_factor;
    }*/

    std::vector<float> row_max;
    float valueAcc = 0.0;
    for (uint32_t row = 0; row < num_rows; row++) {
        float scaled_row_max = 0;
        float rounding_value, value;
        for (uint32_t col = 0; col < num_columns; col++) {
            value = ptr_float_weights[row*num_columns + col] * *ptr_weight_scale_factor;
            valueAcc += value;
            if (fabs(value) > scaled_row_max) {
                scaled_row_max = fabs(value);
            }
        }

        row_max.emplace_back(scaled_row_max);

        value = scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT);
        ptr_int_biases[row].multiplier = (uint8_t) (value + 0.5);

        for (uint32_t col = 0; col < num_columns; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;


            value = ptr_float_weights[row * num_columns + col] * (*ptr_weight_scale_factor / ptr_int_biases[row].multiplier) + rounding_value;

            if (value > 127.0f) {
                *ptr_weight_8 = 127;
                num_saturate++;
                // if (ptr_int_biases[row].multiplier == 0) {
                //     std::cout << value <<  "saturated to 127" << std::endl;
                // }
            } else if (value < -128.0f) {
                *ptr_weight_8 = -128;
                num_saturate++;
                // if (ptr_int_biases[row].multiplier == 0) {
                //     std::cout << value <<  "saturated to -128" << std::endl;
                // }
            } else {
                *ptr_weight_8 = (int8_t) value;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
        ptr_int_biases[row].multiplier = 0;
    }

    // bias value of the bas will be only used when input bias provided
    if (ptr_float_biases != nullptr) {
        QuantizeBias8(ptr_float_biases, ptr_int_biases, ptr_output_scale_factor, num_rows);
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine8()\n", num_saturate, num_rows * num_columns + num_rows);
    }
}

#if __SSE4_2__
static inline float HorizontalMaxSse(__m128 x) {
    __m128 shufReg, sumsReg;
    shufReg = _mm_movehdup_ps(x);        // Broadcast elements 3,1 to 2,0
    sumsReg = _mm_max_ps(x, shufReg);
    shufReg = _mm_movehl_ps(shufReg, sumsReg); // High Half -> Low Half
    sumsReg = _mm_max_ss(sumsReg, shufReg);
    return  _mm_cvtss_f32(sumsReg); // Result in the lower part of the SSE Register
}

float getRowMax_sse(float* nums, uint32_t rows, uint32_t cols) {
    // calculate row max
    float* array = &nums[0];
    const __m128 signmask = _mm_set1_ps(-0.0f);
    __m128 max0 = _mm_loadu_ps(array);
    max0 = _mm_andnot_ps(signmask, max0);
    __m128 max1 = max0;
    int sse_width = 4;

    for (int i=0; i < (cols - sse_width*2); i+=sse_width*2) {
        __m128 pxI0 = _mm_loadu_ps(array + i);
        pxI0 = _mm_andnot_ps(signmask, pxI0);
        max0  = _mm_max_ps(max0, pxI0);

        __m128 pxI1 = _mm_loadu_ps(array + i + sse_width);
        pxI1 = _mm_andnot_ps(signmask, pxI1);
        max1  = _mm_max_ps(max1, pxI1);
    }

    __m128 tail = _mm_loadu_ps(array + cols - sse_width*2);
    tail = _mm_andnot_ps(signmask, tail);
    max0  = _mm_max_ps(max0, tail);

    tail = _mm_loadu_ps(array + cols - sse_width);
    tail = _mm_andnot_ps(signmask, tail);
    max1  = _mm_max_ps(max1, tail);

    max0 = _mm_max_ps(max0, max1);
    float max2 = HorizontalMaxSse(max0);

    return max2;
}

void QuantizeAffine8_sse(float *ptr_float_weights, float *ptr_float_biases,
                     int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                     float input_scale_factor, float *ptr_weight_scale_factor,
                     float *ptr_output_scale_factor, uint32_t num_rows, uint32_t num_columns,
                     uint32_t num_rows_padded, uint32_t num_columns_padded) {
    if (ptr_int_biases == nullptr) {
        THROW_IE_EXCEPTION << "Int biases are empty";
    }
    uint32_t num_saturate = 0;

    // TODO: Removed the commented code from regular QuantizeAffine function
    std::vector<float> row_max;
    uint32_t moves = num_columns / 4;
    uint32_t leftout = num_columns % 4;

    __m128 v, zero, half, neg_half, float_multiplier, mask, rounding_values, min, max, values;
    zero = _mm_setzero_ps();
    max = _mm_set1_ps(127.0f);
    min = _mm_set1_ps(-128.0f);
    half = _mm_set1_ps(0.5f);
    neg_half = _mm_set1_ps(-0.5f);

    for (uint32_t row = 0; row < num_rows; row++) {
        float scaled_row_max = getRowMax_sse(ptr_float_weights + row * num_columns, num_rows, num_columns) * *ptr_weight_scale_factor;
        float value = scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT);
        ptr_int_biases[row].multiplier = (uint8_t) (value + 0.5);

        __m128 scalefactor_vec = _mm_set1_ps(*ptr_weight_scale_factor);
        __m128 multiplier_vec = _mm_set1_ps(ptr_int_biases[row].multiplier);
        float_multiplier = _mm_div_ps(scalefactor_vec, multiplier_vec);

        float* ptr_flot_weights = ptr_float_weights + (row * num_columns);
        int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded);

        for (int j =0; j  < moves; j++, ptr_flot_weights +=4, ptr_weight_8 +=4) {
            v = _mm_loadu_ps(ptr_flot_weights);

            //rounding_values = (v>0) ? 0.5f : -0.5f;
            mask = _mm_min_ps(v, zero);
            rounding_values = _mm_blendv_ps(half, neg_half, mask);

            // values = v * scale_factores +  rounding_values
            values = _mm_mul_ps(v, float_multiplier);
            values = _mm_add_ps(values, rounding_values);

            // shrink to <-128.0f, 127.0f>
            __m128 tmps = _mm_cmpord_ps(values, max);
            values = _mm_and_ps(values, tmps); // Needed to handle NaN values
            values = _mm_min_ps(values, max);
            values = _mm_max_ps(values, min);

            for (int k=0; k < 4; k++) {
                auto a = (int8_t)accessmember(values, k);
                ptr_weight_8[k] = a;
            }
        }

        for (int j =0; j < leftout; j++, ptr_weight_8++, ptr_float_weights++) {
            float rounding_value = (ptr_float_weights[j] > 0) ? 0.5f : -0.5f;

            float value = ptr_float_weights[j] * (*ptr_weight_scale_factor / ptr_int_biases[row].multiplier) + rounding_value;
            if (value > 127.0) {
                *ptr_weight_8 = 127;
                num_saturate++;
            } else if (value < -128.0) {
                *ptr_weight_8 = -128;
                num_saturate++;
            } else {
                *ptr_weight_8 = (int8_t) value;
            }
        }

        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
    }

    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
        ptr_int_biases[row].multiplier = 0;
    }

    // bias value of the bas will be only used when input bias provided
    if (ptr_float_biases != nullptr) {
        QuantizeBias8(ptr_float_biases, ptr_int_biases, ptr_output_scale_factor, num_rows);
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine8()\n", num_saturate, num_rows * num_columns + num_rows);
    }
}
#endif

#if __AVX2__
static inline float HorizontalMax_avx2(__m256 x) {
    __m256 shufReg, sumsReg;
    shufReg = _mm256_movehdup_ps(x);        // Broadcast elements 3,1 to 2,0
    sumsReg = _mm256_max_ps(x, shufReg);

    shufReg = _mm256_permute2f128_ps(sumsReg, sumsReg, 0x81); // High Half -> Low Half
    sumsReg = _mm256_max_ps(sumsReg, shufReg);

    shufReg = _mm256_permute_ps (sumsReg, 0xE);
    sumsReg = _mm256_max_ps(sumsReg, shufReg);

    return  sumsReg[0]; // Result in the lower part of the SSE Register
}

float getRowMax_avx2(float* array, int32_t cols) {
    int moves = cols / 8;
    int remaining = cols % 8;

    //std::cout << __func__ << std::endl;
    __m256 max0 = _mm256_loadu_ps(array);
    const __m256 signmask = _mm256_set1_ps(-0.0f);
    max0 = _mm256_andnot_ps(signmask, max0);

    for (int i=0; i < (cols); i+=8) {
        __m256 pxI0 = _mm256_loadu_ps(array + i);
        pxI0 = _mm256_andnot_ps(signmask, pxI0); // ABS()
        max0  = _mm256_max_ps(max0, pxI0);
    }

    float max1 = HorizontalMax_avx2(max0);

    for (int j=cols; j < remaining; j++) {
        float val = array[j + cols];

        if (fabs(val) > max1)
            max1 = fabs(val);
    }

    return max1;
}

float accessmember_avx2(__m256 v, int index)
{
    union vec{ __m256 sse;
        float f[8];
    };

    vec U;
    U.sse = v;
    return U.f[index];
}

void QuantizeAffine8_avx2(float *ptr_float_weights, float *ptr_float_biases,
                     int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                     float input_scale_factor, float *ptr_weight_scale_factor,
                     float *ptr_output_scale_factor, uint32_t num_rows, uint32_t num_columns,
                     uint32_t num_rows_padded, uint32_t num_columns_padded) {
    if (ptr_int_biases == nullptr) {
        THROW_IE_EXCEPTION << "Int biases are empty";
    }
    uint32_t num_saturate = 0;
    uint32_t moves = num_columns / 8;
    uint32_t leftout = num_columns % 8;

    __m256 v, zero, half, neg_half, float_multiplier, mask, rounding_values, min, max, values;
    zero = _mm256_setzero_ps();
    half = _mm256_set1_ps(0.5f);
    neg_half = _mm256_set1_ps(-0.5f);
    max = _mm256_set1_ps(127.0f);
    min = _mm256_set1_ps(-128.0f);

    for (uint32_t row = 0; row < num_rows; row++) {
        float scaled_row_max = getRowMax_avx2(ptr_float_weights + row*num_columns, num_columns) * (*ptr_weight_scale_factor);
        float value = scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT);
        ptr_int_biases[row].multiplier = (uint8_t) (value + 0.5);

        __m256 scalefactor_vec = _mm256_set1_ps(*ptr_weight_scale_factor);
        __m256 multiplier_vec = _mm256_set1_ps(ptr_int_biases[row].multiplier);
        float_multiplier = _mm256_div_ps(scalefactor_vec, multiplier_vec);

        float* ptr_flot_weights = ptr_float_weights + (row * num_columns);
        int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded);

        for (int j =0; j  < moves; j++, ptr_flot_weights +=8, ptr_weight_8 +=8) {
            v = _mm256_loadu_ps(ptr_flot_weights);

            //rounding_values = (v>0) ? 0.5f : -0.5f;
            mask = _mm256_min_ps(v, zero);
            rounding_values = _mm256_blendv_ps(half, neg_half, mask);

            // values = v * scale_factores +  rounding_values
            values = _mm256_mul_ps(v, float_multiplier);
            values = _mm256_add_ps(values, rounding_values);

            // shrink to <-128.0f, 127.0f>
            __m256 tmps = _mm256_cmp_ps(values, max, _CMP_ORD_S);
            values = _mm256_and_ps(values, tmps);
            values = _mm256_min_ps(values, max);
            values = _mm256_max_ps(values, min);

            for (int k=0; k < 8; k++) {
                auto a = (int8_t)(accessmember_avx2(values, k));
                ptr_weight_8[k] = a;
            }
        }

        for (int j =0; j < leftout; j++, ptr_weight_8++, ptr_float_weights++) {
            float rounding_value = (ptr_float_weights[j] > 0) ? 0.5f : -0.5f;

            float value = ptr_float_weights[j] * (*ptr_weight_scale_factor / ptr_int_biases[row].multiplier) + rounding_value;
            if (value > 127.0) {
                *ptr_weight_8 = 127;
                num_saturate++;
            } else if (value < -128.0) {
                *ptr_weight_8 = -128;
                num_saturate++;
            } else {
                *ptr_weight_8 = (int8_t) value;
            }
        }

        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
    }

    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
        ptr_int_biases[row].multiplier = 0;
    }

    // bias value of the bas will be only used when input bias provided
    if (ptr_float_biases != nullptr) {
        QuantizeBias8(ptr_float_biases, ptr_int_biases, ptr_output_scale_factor, num_rows);
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine8()\n", num_saturate, num_rows * num_columns + num_rows);
    }
}
#endif