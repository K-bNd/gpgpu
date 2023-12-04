#include "filter_impl.h"

#include <chrono>
#include <thread>
#include <cstring>
#include <iostream>
#include <cmath>
#include "logo.h"

struct rgb
{
    uint8_t r, g, b;
    rgb operator+(const rgb &other) const
    {
        rgb result;
        result.r = r + other.r;
        result.g = g + other.g;
        result.b = b + other.b;
        return result;
    }

    rgb operator/(const int &other) const
    {
        rgb result;
        result.r = r / other;
        result.g = g / other;
        result.b = b / other;
        return result;
    }

    bool operator!=(const rgb &other) const
    {
        return r != other.r || g != other.g || b != other.b;
    }

    bool operator>(const int &threshold) const
    {
        return r > threshold || g > threshold || b > threshold;
    }

    bool operator>=(const int &threshold) const
    {
        return r >= threshold || g >= threshold || b >= threshold;
    }

    bool operator<(const int &threshold) const
    {
        return r < threshold || g < threshold || b < threshold;
    }

    bool operator<=(const int &threshold) const
    {
        return r <= threshold || g <= threshold || b <= threshold;
    }
};

struct lab
{
    uint8_t L, a, b;
};

// Définir la taille du voisinage
int rayon = 3;
uint8_t *background = nullptr;
int frame_count = 0;
uint8_t low_threshold = 30;
uint8_t high_threshold = 50;

lab rgbToLab(rgb color)
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

rgb labToRGB(lab &pixel)
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
    r = std::max(0.0, std::min(255.0, r));
    g = std::max(0.0, std::min(255.0, g));
    b = std::max(0.0, std::min(255.0, b));

    return {.r = (uint8_t)r, .g = (uint8_t)g, .b = (uint8_t)b};
}

// Function to convert RGB image to grayscale using the luminosity method
void rgbToGrayscale(uint8_t *buffer, int width, int height, int stride)
{
    for (int y = 0; y < height; y++)
    {
        rgb *lineptr = (rgb *)(buffer + y * stride);
        for (int x = 0; x < width; x++)
        {
            lineptr[x].r = 0.21 * lineptr[x].r + 0.72 * lineptr[x].g + 0.07 * lineptr[x].b;
            lineptr[x].g = 0.21 * lineptr[x].r + 0.72 * lineptr[x].g + 0.07 * lineptr[x].b;
            lineptr[x].b = 0.21 * lineptr[x].r + 0.72 * lineptr[x].g + 0.07 * lineptr[x].b;
        }
    }
}

rgb computeDistance(rgb a, rgb b)
{
    lab lab_a = rgbToLab(a);
    lab lab_b = rgbToLab(b);

    uint8_t delta = sqrt(pow(lab_b.L - lab_a.L, 2.0) + pow(lab_b.a - lab_a.a, 2.0) + pow(lab_b.b - lab_a.b, 2.0));
    lab result = {.L = delta, .a = delta, .b = delta};
    return labToRGB(result);
}

void erosion(uint8_t *buffer, int width, int height, int stride)
{
    // Définir la taille du voisinage
    for (int y = 0; y < height; y++)
    {
        rgb *lineptr = (rgb *)(buffer + y * stride);
        for (int x = 0; x < width; x++)
        {
            for (int dy = -rayon; dy <= rayon; dy++)
            {
                rgb *lineptr_comp = (rgb *)(buffer + (y + dy) * stride);
                for (int dx = -rayon; dx <= rayon; dx++)
                {
                    if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width)
                        continue;
                    else if (sqrt(pow(x + dx, 2) + pow(y + dy, 2)) > rayon)
                        continue;
                    uint8_t sum = lineptr_comp[x + dx].r + lineptr_comp[x + dx].g + lineptr_comp[x + dx].b;
                    if (sum < lineptr[x].r + lineptr[x].g + lineptr[x].b)
                    {
                        lineptr[x] = lineptr_comp[x + dx];
                    }
                }
            }
        }
    }
}

void dilation(uint8_t *buffer, int width, int height, int stride)
{
    for (int y = 0; y < height; y++)
    {
        rgb *lineptr = (rgb *)(buffer + y * stride);
        for (int x = 0; x < width; x++)
        {
            for (int dy = -rayon; dy <= rayon; dy++)
            {
                rgb *lineptr_comp = (rgb *)(buffer + (y + dy) * stride);
                for (int dx = -rayon; dx <= rayon; dx++)
                {
                    if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width)
                        continue;
                    else if (sqrt(pow(dx, 2) + pow(dy, 2)) > rayon)
                        continue;
                    uint8_t sum = lineptr_comp[x + dx].r + lineptr_comp[x + dx].g + lineptr_comp[x + dx].b;
                    if (sum > lineptr[x].r + lineptr[x].g + lineptr[x].b)
                    {
                        lineptr[x] = lineptr_comp[x + dx];
                    }
                }
            }
        }
    }
}

/*
in block -> mem
access mem -> faster
*/

void threshold(uint8_t *buffer, int width, int height, int stride)
{
    for (int y = 0; y < height; y++)
    {
        rgb *lineptr = (rgb *)(buffer + y * stride);
        for (int x = 0; x < width; x++)
        {
            // if (lineptr[x] != ref)
            //     printf("Pixel: red = %d, green = %d, blue = %d\n", lineptr[x].r, lineptr[x].g, lineptr[x].b);
            // strong edges
            if (lineptr[x] > high_threshold)
                lineptr[x] = {.r = 255, .g = 255, .b = 255};
            // check connectivity to strong edges
            else if (lineptr[x] > low_threshold)
            {
                // Weak edge, check for strong neighbors
                for (int dy = -rayon; dy <= rayon; dy++)
                {
                    rgb *lineptr_comp = (rgb *)(buffer + (y + dy) * stride);
                    for (int dx = -rayon; dx <= rayon; dx++)
                    {
                        if (y + dy < 0 || y + dy >= height || x + dx < 0 || x + dx >= width)
                            continue;
                        // Check if the neighbor is within bounds and has a strong edge
                        if (lineptr_comp[x] > high_threshold)
                        {
                            lineptr[x] = {.r = 255, .g = 255, .b = 255};
                            break;
                        }
                    }
                }
            }
            else
            {
                lineptr[x] = {.r = 0, .g = 0, .b = 0};
            }
        }
    }
}

void image_diff(uint8_t *buffer, int width, int height, int stride)
{
    for (int y = 0; y < height; ++y)
    {
        rgb *lineptr = (rgb *)(buffer + y * stride);
        rgb *lineptr_background = (rgb *)(background + y * stride);
        for (int x = 0; x < width; ++x)
        {
            lineptr[x] = computeDistance(lineptr[x], lineptr_background[x]);
        }
    }
}

void update_background(uint8_t *buffer, int width, int height, int stride, int pixel_stride)
{
    frame_count++;
    if (frame_count == 1)
    {
        background = new uint8_t[width * height * pixel_stride];
        std::memcpy(background, buffer, width * height * pixel_stride);
        return;
    }
    for (int y = 0; y < height; ++y)
    {
        rgb *lineptr = (rgb *)(buffer + y * stride);
        rgb *lineptr_background = (rgb *)(background + y * stride);
        for (int x = 0; x < width; ++x)
        {
            lineptr_background[x] = (lineptr_background[x] + lineptr[x]) / 2;
        }
    }
}

extern "C"
{
    void filter_impl(uint8_t *buffer, int width, int height, int stride, int pixel_stride)
    {
        /*
        Pseudo-code:
        1. Image difference between current frame and background frame (done)
        2. Morphological opening (testing)
        3. Thresholding
        4. Apply mask
        */
        update_background(buffer, width, height, stride, pixel_stride);
        image_diff(buffer, width, height, stride);
        erosion(buffer, width, height, stride);
        dilation(buffer, width, height, stride);
        rgbToGrayscale(buffer, width, height, stride);
        threshold(buffer, width, height, stride);
        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}