#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include "logo.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb
{
    uint8_t r, g, b;
};

struct lab
{
    uint8_t L, a, b;
};

__constant__ uint8_t *logo;

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return
__global__ void remove_red_channel_inp(std::byte *buffer, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb *lineptr = (rgb *)(buffer + y * stride);
    if (y < logo_height && x < logo_width)
    {
        float alpha = logo[y * logo_width + x] / 255.f;
        lineptr[x].r = 0;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
    }
    else
    {
        lineptr[x].r = 0;
    }
}

__global__ void erosion(std::byte *buffer, int width, int height, int stride)
{
    int rayon = 3;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;
    rgb *lineptr = (rgb *)(buffer + y * stride);

    for (int dy = -rayon; dy <= rayon; dy++)
    {
        rgb *lineptr_comp = (rgb *)(buffer + (y + dy) * stride);
        for (int dx = -rayon; dx <= rayon; dx++)
        {
            if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width)
                continue;
            uint8_t sum = lineptr_comp[x + dx].r + lineptr_comp[x + dx].g + lineptr_comp[x + dx].b;
            if (sum < lineptr[x].r + lineptr[x].g + lineptr[x].b)
                lineptr[x] = lineptr_comp[x + dx];
        }
    }
}

__device__ lab rgbToLab(rgb color)
{
    // Convert RGB to XYZ
    uint8_t X = (0.4124564 * color.r + 0.3575761 * color.g + 0.1804674 * color.b) / 255.0;
    uint8_t Y = (0.2126729 * color.r + 0.7152282 * color.g + 0.072099 * color.b) / 255.0;
    uint8_t Z = (0.0193339 * color.r + 0.1191920 * color.g + 0.9503041 * color.b) / 255.0;

    // Convert XYZ to CIE L*a*b*
    uint8_t L = 116.0 * pow(Y / 0.008856, 1.0 / 3.0) - 16.0;
    uint8_t a = 500.0 * (pow(X / 0.950456, 1.0 / 3.0) - pow(Y / 1.0, 1.0 / 3.0));
    uint8_t b = 200.0 * (pow(Y / 1.0, 1.0 / 3.0) - pow(Z / 1.07700, 1.0 / 3.0));

    return {.L = L, .a = a, .b = b};
}

__device__ rgb labToRgb(lab &pixel)
{
    // Convert CIE L*a*b* to XYZ
    double X = (pixel.L + 16.0) / 116.0;
    double Y = (X * 0.008856 + 16.0) / 116.0;
    double Z = Y / 1.181678;

    double r = 3.240479 * X - 1.537383 * Y - 0.498531 * Z;
    double g = -0.969256 * X + 1.875991 * Y + 0.041556 * Z;
    double b = 0.055648 * X - 0.201966 * Y + 1.253272 * Z;

    // Convert XYZ to RGB
    r = 255.0 * r;
    g = 255.0 * g;
    b = 255.0 * b;

    // Clamp values to valid RGB range
    r = max(0.0, min(255.0, r));
    g = max(0.0, min(255.0, g));
    b = max(0.0, min(255.0, b));

    return {.r = (uint8_t)r, .g = (uint8_t)g, .b = (uint8_t)b};
}

__device__ rgb computeDistance(rgb a, rgb b)
{
    // Implement your RGB to Lab conversion here
    // This is a simplified version, you may want to use a more accurate conversion method
    lab lab_a = rgbToLab(a);

    lab lab_b = rgbToLab(b);

    uint8_t delta = sqrt(pow(lab_b.L - lab_a.L, 2.0) + pow(lab_b.a - lab_a.a, 2.0) + pow(lab_b.b - lab_a.b, 2.0));

    lab result = {delta, delta, delta};
    return labToRgb(result);
}

// CUDA kernel to perform image difference
__global__ void imageDiff(std::byte *buffer, std::byte *background, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb *lineptr = (rgb *)(buffer + y * stride);
    rgb *lineptr_background = (rgb *)(background + y * stride);

    lineptr[x] = computeDistance(lineptr[x], lineptr_background[x]);
}

// CUDA kernel to update the background
__global__ void updateBackground(std::byte *buffer, std::byte *background, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb *lineptr = (rgb *)(buffer + y * stride);
    rgb *lineptr_background = (rgb *)(background + y * stride);

    lineptr_background[x] = {
        static_cast<uint8_t>((lineptr_background[x].r + lineptr[x].r) / 2),
        static_cast<uint8_t>((lineptr_background[x].g + lineptr[x].g) / 2),
        static_cast<uint8_t>((lineptr_background[x].b + lineptr[x].b) / 2)};
}

namespace
{
    void load_logo()
    {
        static auto buffer = std::unique_ptr<std::byte, decltype(&cudaFree)>{nullptr, &cudaFree};

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte *ptr;
            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }
    }
}

extern "C"
{
    void filter_impl(uint8_t *src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        load_logo();

        assert(sizeof(rgb) == pixel_stride);
        std::byte *dBuffer;
        size_t pitch;
        static int frame_count = 0;
        static std::byte *background;

        cudaError_t err;

        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        frame_count++;
        // remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height, pitch);

        if (frame_count == 1)
        {
            err = cudaMallocPitch(&background, &pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy2D(background, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
            CHECK_CUDA_ERROR(err);
        }

        updateBackground<<<gridSize, blockSize>>>(dBuffer, background, width, height, src_stride);

        // imageDiff<<<gridSize, blockSize>>>(dBuffer, background, width, height, src_stride);

        // erosion<<<gridSize, blockSize>>>(dBuffer, width, height, src_stride);

        // end of process (dBuffer is copied into src_buffer)
        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(background);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}
