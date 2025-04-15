/* 
 * Serial implementation of Wiener deconvolution for an RGB image.
 * 
 * This program performs image deblurring using the Wiener algorithm,
 * processing an RGB image on a single processor.
 * 
 * - Uses STB libraries for loading/saving images
 * - Assumes input and PSF are the same size
 * 
 * Requirements:
 *   - FFTW3 library (https://www.fftw.org/)
 *   - STB single-file header libraries:
 *       https://github.com/nothings/stb/blob/master/stb_image.h
 *       https://github.com/nothings/stb/blob/master/stb_image_write.h
 *     (Download and place them in your project directory)
 * 
 * Compilation:
 *   gcc wiener_serial.c -lfftw3 -lm -o wiener_serial
 * 
 * Execution:
 *   ./wiener_serial input.png psf.png output.png [k_value]
 */

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 
   STB libraries:
   1) Place these #defines before including the .h files to enable implementations.
   2) Download stb_image.h, stb_image_write.h from https://github.com/nothings/stb
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <stdint.h>

// Reads little-endian uint16 from buffer
static uint16_t read_le_uint16(const unsigned char *buf) {
    return (uint16_t)(buf[0]) | ((uint16_t)(buf[1]) << 8);
}

static float *load_psf_npy(const char *filename, int expected_w, int expected_h)
{
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: could not open .npy file %s\n", filename);
        return NULL;
    }

    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "Error: not a valid .npy file.\n");
        fclose(f);
        return NULL;
    }

    unsigned char version[2];
    fread(version, 1, 2, f);
    int major = version[0], minor = version[1];

    int header_len = 0;
    if (major == 1) {
        unsigned char hl[2];
        fread(hl, 1, 2, f);
        header_len = hl[0] + (hl[1] << 8);
    } else if (major == 2) {
        unsigned char hl[4];
        fread(hl, 1, 4, f);
        header_len = hl[0] + (hl[1] << 8) + (hl[2] << 16) + (hl[3] << 24);
    } else {
        fprintf(stderr, "Unsupported .npy version: %d.%d\n", major, minor);
        fclose(f);
        return NULL;
    }

    char *header = (char *)malloc(header_len + 1);
    fread(header, 1, header_len, f);
    header[header_len] = '\0';

    if (!strstr(header, "'descr': '<f4'") || !strstr(header, "'fortran_order': False")) {
        fprintf(stderr, "Unsupported format: must be float32, little-endian, C-order.\nHeader: %s\n", header);
        free(header);
        fclose(f);
        return NULL;
    }

    int total = expected_w * expected_h;
    float *data = (float *)malloc(sizeof(float) * total);
    if (!data) {
        fprintf(stderr, "Out of memory\n");
        free(header);
        fclose(f);
        return NULL;
    }

    size_t read_count = fread(data, sizeof(float), total, f);
    if (read_count != total) {
        fprintf(stderr, "Read error: expected %d floats but got %zu\n", total, read_count);
        free(data);
        data = NULL;
    }

    free(header);
    fclose(f);
    return data;
}
/* 
 * read_image_rgb:
 *   Reads a 3-channel RGB image from file and returns float data in range [0,1].
 *
 * Parameters:
 *   - path: path to the input image file (PNG, JPG, etc.)
 *   - out_w, out_h: pointers to store the image width and height
 *
 * Returns:
 *   - A float array of size (3 * width * height) containing RGB in row-major order
 *   - NULL on failure
 */
static float *read_image_rgb(const char *path, int *out_w, int *out_h)
{
    int channels_in_file;
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 3);
    if (!data_u8) {
        fprintf(stderr, "Error: Could not load image %s\n", path);
        return NULL;
    }
    int w = *out_w;
    int h = *out_h;

    /* Allocate float array for 3 channels. */
    float *data_f = (float *)malloc(sizeof(float) * w * h * 3);
    if (!data_f) {
        fprintf(stderr, "Error: Could not allocate memory for image.\n");
        stbi_image_free(data_u8);
        return NULL;
    }

    /* Convert from unsigned char [0..255] to float [0..1]. */
    int i;
    for (i = 0; i < w * h * 3; i++) {
        data_f[i] = (float)(data_u8[i]) / 255.0f;
    }

    stbi_image_free(data_u8);
    return data_f;
}

/*
 * read_image_gray:
 *   Loads a grayscale image and converts it to floating-point format.
 *
 * Parameters:
 *   - path: file path to the input image
 *   - out_w, out_h: pointers to receive image dimensions
 *
 * Returns:
 *   - Float array containing normalized [0..1] pixel values
 *   - NULL if loading fails
 */
static float *read_image_gray(const char *path, int *out_w, int *out_h)
{
    int channels_in_file;
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 1);
    if (!data_u8) {
        fprintf(stderr, "Error: Could not load image %s\n", path);
        return NULL;
    }
    int w = *out_w;
    int h = *out_h;

    float *data_f = (float *)malloc(sizeof(float) * w * h);
    if (!data_f) {
        fprintf(stderr, "Error: Could not allocate memory for PSF.\n");
        stbi_image_free(data_u8);
        return NULL;
    }

    int i;
    for (i = 0; i < w * h; i++) {
        data_f[i] = (float)(data_u8[i]) / 255.0f;
    }

    stbi_image_free(data_u8);
    return data_f;
}

/*
 * write_image_rgb:
 *   Saves a floating-point RGB image as a PNG file.
 *
 * Parameters:
 *   - path: output file path
 *   - data: float array with RGB values in range [0..1]
 *   - w, h: width and height of the image
 *
 * Returns:
 *   - 1 on success, 0 on failure
 *
 * Notes:
 *   - Values outside [0..1] range are clamped
 */
static int write_image_rgb(const char *path, const float *data, int w, int h)
{
    /* Convert from float [0..1] to unsigned char [0..255]. */
    unsigned char *out_u8 = (unsigned char *)malloc(w * h * 3);
    if (!out_u8) {
        fprintf(stderr, "Error: cannot allocate memory for output image.\n");
        return 0;
    }
    int i;
    for (i = 0; i < w * h * 3; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        out_u8[i] = (unsigned char)(val * 255.0f + 0.5f);
    }

    /* stbi_write_png expects row-major with 3 channels. */
    int ret = stbi_write_png(path, w, h, 3, out_u8, w * 3);
    free(out_u8);
    if (!ret) {
        fprintf(stderr, "Error: stbi_write_png failed for %s\n", path);
        return 0;
    }
    return 1;
}

/*
 * center_psf:
 *   Rearranges PSF quadrants to center it for FFT processing.
 *
 * Parameters:
 *   - psf: the point spread function data to be centered
 *   - w, h: dimensions of the PSF image
 *
 * Notes:
 *   - Modifies the PSF in-place
 *   - Required for proper frequency-domain processing
 */
static void center_psf(float *psf, int w, int h) 
{
    float *temp = (float *)malloc(w * h * sizeof(float));
    if (!temp) {
        fprintf(stderr, "Error: Out of memory in center_psf\n");
        return;
    }
    
    int i, j;
    /* Copy to temporary buffer */
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            temp[i*w + j] = psf[i*w + j];
        }
    }
    
    /* Rearrange quadrants to center the PSF */
    int half_h = h / 2;
    int half_w = w / 2;
    
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            int new_i = (i + half_h) % h;
            int new_j = (j + half_w) % w;
            psf[new_i*w + new_j] = temp[i*w + j];
        }
    }
    
    free(temp);
}

/*
 * normalize_psf:
 *   Scales the PSF values so they sum to 1.0 (energy conservation).
 *
 * Parameters:
 *   - psf: the point spread function data to normalize
 *   - w, h: dimensions of the PSF image
 *
 * Notes:
 *   - Modifies the PSF in-place
 */
static void normalize_psf(float *psf, int w, int h)
{
    double sum = 0.0;
    int i;
    
    /* Calculate sum */
    for (i = 0; i < w * h; i++) {
        sum += psf[i];
    }
    
    /* Avoid division by zero */
    if (sum > 1e-10) {
        for (i = 0; i < w * h; i++) {
            psf[i] /= sum;
        }
    } else {
        fprintf(stderr, "Warning: PSF sum is too small for normalization\n");
    }
}

/*
 * wiener_deconv_1ch:
 *   Performs Wiener deconvolution on a single-channel image.
 * 
 * Parameters:
 *   - img_in: input blurred image data
 *   - img_psf: point spread function data
 *   - img_out: buffer for deconvolved result
 *   - w, h: dimensions of image
 *   - K: Wiener regularization parameter (controls noise sensitivity)
 *
 * Notes:
 *   - Uses FFT to perform deconvolution in frequency domain
 *   - K parameter balances deblurring vs. noise amplification
 */
static void wiener_deconv_1ch(
    float *img_in,
    float *img_psf,
    float *img_out,
    int w,
    int h,
    double K
) {
    int pad_w = 2 * w;
    int pad_h = 2 * h;
    int padded_size = pad_w * pad_h;

    // Allocate padded buffers
    double *padded_in = (double *)fftw_malloc(sizeof(double) * padded_size);
    double *padded_psf = (double *)fftw_malloc(sizeof(double) * padded_size);
    memset(padded_in, 0, sizeof(double) * padded_size);
    memset(padded_psf, 0, sizeof(double) * padded_size);

    // Copy image and PSF to top-left corner
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            padded_in[y * pad_w + x] = img_in[y * w + x];
            padded_psf[y * pad_w + x] = img_psf[y * w + x];
        }
    }

    // Center and normalize PSF
    center_psf((float *)padded_psf, pad_w, pad_h);
    normalize_psf((float *)padded_psf, pad_w, pad_h);

    // Allocate frequency domain buffers
    int n_complex = pad_h * (pad_w/2 + 1);
    fftw_complex *freq_in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex);
    fftw_complex *freq_psf = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex);
    fftw_complex *freq_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex);

    // Plans
    fftw_plan p_in  = fftw_plan_dft_r2c_2d(pad_h, pad_w, padded_in, freq_in, FFTW_ESTIMATE);
    fftw_plan p_psf = fftw_plan_dft_r2c_2d(pad_h, pad_w, padded_psf, freq_psf, FFTW_ESTIMATE);
    fftw_plan p_out = fftw_plan_dft_c2r_2d(pad_h, pad_w, freq_out, padded_in, FFTW_ESTIMATE);

    // Execute forward FFTs
    fftw_execute(p_in);
    fftw_execute(p_psf);

    // Wiener filter
    for (int i = 0; i < n_complex; i++) {
        double B_r = freq_in[i][0], B_i = freq_in[i][1];
        double H_r = freq_psf[i][0], H_i = freq_psf[i][1];
        double mag2 = H_r * H_r + H_i * H_i;
        double denom = mag2 + K;

        double num_r = B_r * H_r + B_i * H_i;
        double num_i = B_i * H_r - B_r * H_i;

        if (denom < 1e-12) {
            freq_out[i][0] = 0.0;
            freq_out[i][1] = 0.0;
        } else {
            freq_out[i][0] = num_r / denom;
            freq_out[i][1] = num_i / denom;
        }
    }

    // Inverse FFT
    fftw_execute(p_out);

    // Normalize IFFT result and crop back to original size
    int offset_y = (pad_h - h) / 2;
    int offset_x = (pad_w - w) / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i_pad = (y + offset_y) * pad_w + (x + offset_x);
            img_out[y * w + x] = (float)(padded_in[i_pad] / padded_size);
        }
    }

    fftw_destroy_plan(p_in);
    fftw_destroy_plan(p_psf);
    fftw_destroy_plan(p_out);

    fftw_free(freq_in);
    fftw_free(freq_psf);
    fftw_free(freq_out);
    fftw_free(padded_in);
    fftw_free(padded_psf);
}
/*
 * main:
 *   Entry point for the serial Wiener deconvolution program.
 *
 * Usage:
 *   ./wiener_serial input.png psf.png output.png [k_value]
 *
 * Steps:
 *   1) Load input RGB image and PSF
 *   2) Perform Wiener deconvolution on each color channel
 *   3) Save the deblurred image to disk
 */
int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_rgb> <psf_gray> <output> [k_value]\n", argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *psf_path    = argv[2];
    const char *output_path = argv[3];

    /* Wiener regularization constant. 
       Can be passed as 4th argument or use default. 
       Try different values between 0.001 and 0.1 */
    double K = 0.01;
    if (argc > 4) {
        K = atof(argv[4]);
    }
    
    printf("Using Wiener K value: %f\n", K);

    /* Load the color image */
    int img_w = 0, img_h = 0;
    float *img_rgb = read_image_rgb(input_path, &img_w, &img_h);
    if (!img_rgb) {
        fprintf(stderr, "Could not load input image.\n");
        return 1;
    }

    /* Load single-channel PSF */
    int psf_w = 0, psf_h = 0;
    float *psf = NULL;
    if (strstr(psf_path, ".npy") != NULL) {
        psf_w = img_w;
        psf_h = img_h;
        psf = load_psf_npy(psf_path, psf_w, psf_h);
    } else {
        psf = read_image_gray(psf_path, &psf_w, &psf_h);
    }
    if (!psf) {
        fprintf(stderr, "Could not load PSF image.\n");
        free(img_rgb);
        return 1;
    }

    /* Check PSF dimensions */
    if (psf_w != img_w || psf_h != img_h) {
        fprintf(stderr, 
            "Warning: PSF size (%dx%d) != image size (%dx%d).\n"
            "This may cause issues with deconvolution.\n",
            psf_w, psf_h, img_w, img_h);
    }

    /* Allocate output image buffer */
    float *out_rgb = (float *)malloc(sizeof(float) * 3 * img_w * img_h);
    if (!out_rgb) {
        fprintf(stderr, "Out of memory for output image.\n");
        free(img_rgb);
        free(psf);
        return 1;
    }

    /* Calculate single channel size */
    int channel_size = img_w * img_h;
    
    /* Extract each color channel */
    float *img_r = (float *)malloc(sizeof(float) * channel_size);
    float *img_g = (float *)malloc(sizeof(float) * channel_size); 
    float *img_b = (float *)malloc(sizeof(float) * channel_size);
    
    /* Output buffers for each channel */
    float *out_r = (float *)malloc(sizeof(float) * channel_size);
    float *out_g = (float *)malloc(sizeof(float) * channel_size);
    float *out_b = (float *)malloc(sizeof(float) * channel_size);
    
    if (!img_r || !img_g || !img_b || !out_r || !out_g || !out_b) {
        fprintf(stderr, "Out of memory for channel buffers.\n");
        free(img_rgb);
        free(psf);
        free(out_rgb);
        free(img_r);
        free(img_g);
        free(img_b);
        free(out_r);
        free(out_g);
        free(out_b);
        return 1;
    }
    
    /* Extract R, G, B channels from interleaved RGB data */
    int i;
    for (i = 0; i < channel_size; i++) {
        img_r[i] = img_rgb[3*i];
        img_g[i] = img_rgb[3*i + 1];
        img_b[i] = img_rgb[3*i + 2];
    }
    
    /* Process each channel separately */
    printf("Processing red channel...\n");
    wiener_deconv_1ch(img_r, psf, out_r, img_w, img_h, K);
    
    printf("Processing green channel...\n");
    wiener_deconv_1ch(img_g, psf, out_g, img_w, img_h, K);
    
    printf("Processing blue channel...\n");
    wiener_deconv_1ch(img_b, psf, out_b, img_w, img_h, K);
    
    /* Combine the channels back to RGB */
    for (i = 0; i < channel_size; i++) {
        out_rgb[3*i]     = out_r[i];
        out_rgb[3*i + 1] = out_g[i];
        out_rgb[3*i + 2] = out_b[i];
    }
    
    /* Write output image */
    if (!write_image_rgb(output_path, out_rgb, img_w, img_h)) {
        fprintf(stderr, "Failed to write output image %s\n", output_path);
    } else {
        printf("Wrote deblurred image to %s\n", output_path);
    }
    
    /* Clean up */
    free(img_rgb);
    free(psf);
    free(out_rgb);
    free(img_r);
    free(img_g);
    free(img_b);
    free(out_r);
    free(out_g);
    free(out_b);
    
    return 0;
}