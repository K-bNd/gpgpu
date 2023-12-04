#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus

void image_diff(uint8_t* buffer, int width, int height, int plane_stride);
void erosion(uint8_t *buffer, int width, int height, int stride);
void dilation(uint8_t *buffer, int width, int height, int stride);
void rgbToGrayscale(uint8_t *buffer, int width, int height, int stride);

extern "C" {
#endif

void filter_impl(uint8_t* buffer, int width, int height, int plane_stride, int pixel_stride);

#ifdef __cplusplus
}
#endif
