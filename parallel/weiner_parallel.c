/* 
 * Example: MPI-based Wiener deconvolution for an RGB image.
 * 
 * This program performs image deblurring using the Wiener algorithm,
 * using MPI to process an RGB image in parallel.
 * 
 * - Uses STB libraries for loading/saving images
 * - Assumes input and PSF are the same size
 * - Each process handles a vertical slice of the image
 * 
 * Requirements:
 *   - FFTW3 library (https://www.fftw.org/)
 *   - STB single-file header libraries:
 *       https://github.com/nothings/stb/blob/master/stb_image.h
 *       https://github.com/nothings/stb/blob/master/stb_image_write.h
 *     (Download and place them in your project directory)
 *   - MS-MPI (for Windows) or OpenMPI (for Linux/macOS)
 * 
 * Compilation:
 *   mpicc wiener_parallel.c -lfftw3 -lm -o wiener_parallel
 * 
 * Execution:
 *   mpirun -np 4 ./wiener_parallel input.png psf.png output.png
 */

#include <mpi.h>
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
 * wiener_deconv_patch_1ch:
 *   Performs Wiener deconvolution on a single-channel image patch.
 * 
 * Parameters:
 *   - patch_in: input blurred image data
 *   - patch_psf: point spread function data
 *   - patch_out: buffer for deconvolved result
 *   - patch_w, patch_h: dimensions of all patches
 *   - K: Wiener regularization parameter (controls noise sensitivity)
 *
 * Notes:
 *   - Uses FFT to perform deconvolution in frequency domain
 *   - K parameter balances deblurring vs. noise amplification
 */
static void wiener_deconv_patch_1ch(
    float *patch_in,
    float *patch_psf,
    float *patch_out,
    int patch_w,
    int patch_h,
    double K
) {
    int N = patch_w * patch_h;
    int i;
    
    /* Create temporary copy of PSF for centering and normalization */
    float *psf_copy = (float *)malloc(sizeof(float) * N);
    if (!psf_copy) {
        fprintf(stderr, "Error: Out of memory for PSF processing\n");
        return;
    }
    
    for (i = 0; i < N; i++) {
        psf_copy[i] = patch_psf[i];
    }
    
    /* Center and normalize the PSF */
    center_psf(psf_copy, patch_w, patch_h);
    normalize_psf(psf_copy, patch_w, patch_h);

    /* Allocate double arrays for FFTW. We must convert float->double. */
    double *in_spatial  = (double *)fftw_malloc(sizeof(double) * N);
    double *psf_spatial = (double *)fftw_malloc(sizeof(double) * N);
    
    /* Copy data to double arrays */
    for (i = 0; i < N; i++) {
        in_spatial[i]  = (double)patch_in[i];
        psf_spatial[i] = (double)psf_copy[i];
    }
    
    /* For 2D real-to-complex transforms, we need different dimensions */
    int n_complex_out = patch_h * (patch_w/2 + 1); /* FFTW r2c format */

    /* Frequency-domain (complex) buffers with correct size for r2c transform */
    fftw_complex *freq_in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    fftw_complex *freq_psf = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    fftw_complex *freq_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);

    /* Plans */
    fftw_plan p_fwd_in = fftw_plan_dft_r2c_2d(patch_h, patch_w,
                                             in_spatial, freq_in,
                                             FFTW_ESTIMATE);
    fftw_plan p_fwd_psf = fftw_plan_dft_r2c_2d(patch_h, patch_w,
                                              psf_spatial, freq_psf,
                                              FFTW_ESTIMATE);

    fftw_plan p_inv_out = fftw_plan_dft_c2r_2d(patch_h, patch_w,
                                              freq_out, in_spatial,
                                              FFTW_ESTIMATE);

    /* Forward FFT */
    fftw_execute(p_fwd_in);
    fftw_execute(p_fwd_psf);

    /* Apply Wiener filter:
       D(k) = B(k)*conjugate(H(k)) / (|H(k)|^2 + K)
       We store result in freq_out.
    */
    for (i = 0; i < n_complex_out; i++) {
        double B_r = freq_in[i][0];
        double B_i = freq_in[i][1];
        double H_r = freq_psf[i][0];
        double H_i = freq_psf[i][1];

        double mag2 = H_r*H_r + H_i*H_i;  /* |H|^2 */
        double denom = mag2 + K;

        /* B * conj(H) = (B_r*H_r + B_i*H_i) + j(B_i*H_r - B_r*H_i) */
        double num_r = B_r*H_r + B_i*H_i;
        double num_i = B_i*H_r - B_r*H_i;

        if (denom < 1e-15) {
            freq_out[i][0] = 0.0;
            freq_out[i][1] = 0.0;
        } else {
            freq_out[i][0] = num_r / denom;
            freq_out[i][1] = num_i / denom;
        }
    }

    /* Inverse FFT => in_spatial. Then convert back to float. */
    fftw_execute(p_inv_out);

    /* Normalize since FFTW doesn't do it automatically. */
    for (i = 0; i < N; i++) {
        double val = in_spatial[i] / (double)N;  /* scale by 1/N */
        patch_out[i] = (float)val;
    }

    /* Cleanup */
    fftw_destroy_plan(p_fwd_in);
    fftw_destroy_plan(p_fwd_psf);
    fftw_destroy_plan(p_inv_out);

    fftw_free(freq_in);
    fftw_free(freq_psf);
    fftw_free(freq_out);
    fftw_free(in_spatial);
    fftw_free(psf_spatial);
    free(psf_copy);
}


/*
 * main:
 *   Entry point for the MPI-based Wiener deconvolution program.
 *
 * Usage:
 *   mpirun -np <processes> ./wiener_parallel input.png psf.png output.png [k_value]
 *
 * Steps:
 *   1) Load input RGB image and PSF on rank 0
 *   2) Divide image into vertical slices distributed to all ranks
 *   3) Each rank performs Wiener deconvolution on its slice
 *   4) Results are gathered and combined on rank 0
 *   5) Final deblurred image is saved to disk
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_rgb> <psf_gray> <output> [k_value]\n", argv[0]);
        }
        MPI_Finalize();
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
    
    if (rank == 0) {
        printf("Using Wiener K value: %f\n", K);
    }

    /* Image dimensions. Only rank 0 will read initially. */
    int img_w = 0, img_h = 0;
    int psf_w = 0, psf_h = 0;

    float *full_img = NULL;  /* [3 * w * h], float in [0..1] */
    float *psf_img  = NULL;  /* single-channel [w_psf * h_psf] */

    /* Final deblurred image on rank 0. */
    float *full_deblur = NULL;

    if (rank == 0) {
        /* Load the color image. */
        full_img = read_image_rgb(input_path, &img_w, &img_h);
        if (!full_img) {
            fprintf(stderr, "Could not load input image.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Load single-channel PSF. */
        psf_img = read_image_gray(psf_path, &psf_w, &psf_h);
        if (!psf_img) {
            fprintf(stderr, "Could not load PSF image.\n");
            free(full_img);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Resize PSF if needed to match image dimensions.
           In this simple version, we'll just warn about mismatches. */
        if (psf_w != img_w || psf_h != img_h) {
            fprintf(stderr, 
                "Warning: PSF size (%dx%d) != image size (%dx%d).\n"
                "This may cause issues with deconvolution.\n",
                psf_w, psf_h, img_w, img_h);
        }

        /* Allocate full deblurred output. */
        full_deblur = (float *)malloc(sizeof(float)*3*img_w*img_h);
        if (!full_deblur) {
            fprintf(stderr, "Out of memory.\n");
            free(full_img);
            free(psf_img);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast the dimensions to all ranks. */
    MPI_Bcast(&img_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&psf_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&psf_h, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (img_w <= 0 || img_h <= 0) {
        if (rank == 0)
            fprintf(stderr, "Error: invalid image dimensions.\n");
        MPI_Finalize();
        return 1;
    }

    /* Each process gets a vertical slice: 
       patch_height = img_h / size  (assuming it divides evenly)
       The slice is the entire width (img_w) for each color channel, 
       but only patch_height rows. 
       We'll do 3 separate single-channel slices for R, G, B.
    */
    if (img_h % size != 0) {
        if (rank == 0) {
            fprintf(stderr, 
                "This example code requires that img_h (%d) is divisible by the number of ranks (%d).\n",
                img_h, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int patch_h = img_h / size;
    int patch_w = img_w;

    /* For convenience, let's define the sub-image size for each channel. */
    int patch_size_1ch = patch_w * patch_h; /* single-channel */
    int patch_size_3ch = patch_size_1ch * 3; /* for the full RGB slice */

    /* Allocate buffers for this rank's slice of the image (RGB) */
    float *slice_in = (float *)malloc(sizeof(float)*patch_size_3ch);
    float *slice_out= (float *)malloc(sizeof(float)*patch_size_3ch);

    /* Allocate a slice of the PSF (single channel). */
    float *psf_slice = (float *)malloc(sizeof(float)*patch_size_1ch);

    /* Scatter the input image. 
       3 * w * h floats total => we chop into 'size' vertical slices.
       Each slice is patch_size_3ch floats. 
    */
    MPI_Scatter(
        /* sendbuf   = full_img [on rank 0], recvbuf = slice_in [on others] */
        full_img, patch_size_3ch, MPI_FLOAT,
        slice_in, patch_size_3ch, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    /* Scatter the PSF image (single-channel). */
    MPI_Scatter(
        psf_img, patch_size_1ch, MPI_FLOAT,
        psf_slice, patch_size_1ch, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    /* Now we have an RGB slice in slice_in (size = patch_size_3ch).
       We'll treat each channel separately for Wiener deconvolution. 
       That means we do 3 calls to wiener_deconv_patch_1ch:
         R => slice_in[0..patch_size_1ch-1]
         G => slice_in[patch_size_1ch..2*patch_size_1ch-1]
         B => slice_in[2*patch_size_1ch..3*patch_size_1ch-1]
       The PSF slice is single-channel (psf_slice).
    */
    float *slice_in_R = slice_in;                    
    float *slice_in_G = slice_in + patch_size_1ch;   
    float *slice_in_B = slice_in + 2*patch_size_1ch;

    float *slice_out_R = slice_out;
    float *slice_out_G = slice_out + patch_size_1ch;
    float *slice_out_B = slice_out + 2*patch_size_1ch;

    /* Deconvolution on each channel. */
    wiener_deconv_patch_1ch(slice_in_R, psf_slice, slice_out_R, patch_w, patch_h, K);
    wiener_deconv_patch_1ch(slice_in_G, psf_slice, slice_out_G, patch_w, patch_h, K);
    wiener_deconv_patch_1ch(slice_in_B, psf_slice, slice_out_B, patch_w, patch_h, K);

    /* Gather the results back to rank 0. */
    MPI_Gather(
        slice_out, patch_size_3ch, MPI_FLOAT,
        full_deblur, patch_size_3ch, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    /* Rank 0 writes the final image. */
    if (rank == 0) {
        /* Write to PNG. We assume [0..1] float range. */
        if (!write_image_rgb(output_path, full_deblur, img_w, img_h)) {
            fprintf(stderr, "Failed to write output image %s\n", output_path);
        } else {
            printf("Wrote deblurred image to %s\n", output_path);
        }
    }

    /* Cleanup */
    free(slice_in);
    free(slice_out);
    free(psf_slice);

    if (rank == 0) {
        free(full_img);
        free(psf_img);
        free(full_deblur);
    }

    MPI_Finalize();
    return 0;
}