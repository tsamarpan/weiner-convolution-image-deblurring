# Wiener Convolution for Image Deblurring

This project implements image deblurring using Wiener convolution in both serial and parallel implementations. The project first applies synthetic blur to raw images and then deblurs them using the Wiener deconvolution algorithm.

## Project Overview

The project consists of two main implementations:

- Serial implementation: Single-threaded version of the Wiener deconvolution algorithm
- Parallel implementation: MPI-based version utilizing parallel processing for improved performance

## Project Structure

```
├── parallel/     # Parallel implementation using MPI
├── serial/       # Serial implementation
├── include/      # Header files (STB libraries)
├── images/       # Test images
└── python_scripts/ # Image preprocessing scripts
```

## Dependencies

### C/C++ Libraries

- FFTW3 (Fast Fourier Transform in the West) - For performing FFT operations
- MPI (Message Passing Interface) - For parallel processing implementation
  - MS-MPI for Windows
  - OpenMPI for Linux/macOS
- STB single-file header libraries:
  - stb_image.h - For image loading
  - stb_image_write.h - For image saving
- C standard library (stdio.h, stdlib.h, string.h, math.h)

### Python Dependencies

- OpenCV (cv2) - For image processing and synthetic blur
- NumPy - For numerical computations
- argparse - For command line argument parsing

## Building

### Prerequisites

1. Install FFTW3:

   ```bash
   # Ubuntu/Debian
   sudo apt-get install libfftw3-dev

   # macOS
   brew install fftw

   # Windows
   # Download and install from http://www.fftw.org/install/windows.html
   ```

2. Install MPI:

   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev

   # macOS
   brew install open-mpi

   # Windows
   # Download and install MS-MPI from Microsoft's website
   ```

3. Install Python dependencies:
   ```bash
   pip install numpy opencv-python
   ```

### Compilation

```bash
# Serial version
gcc serial/weiner_serial.c -lfftw3 -lm -o serial/wiener_serial

# Parallel version
mpicc parallel/weiner_parallel.c -lfftw3 -lm -o parallel/wiener_parallel
```

## Usage

1. Place your input images in the `images/raw/` directory
2. Preprocess the image (apply synthetic blur):
   ```bash
   python python_scripts/preprocess_image.py --input_file <input_image> --sigma <blur_amount>
   ```
3. Run the deblurring program:

   ```bash
   # Serial version
   ./serial/wiener_serial input.png psf.png output.png [k_value]

   # Parallel version
   mpirun -np <num_processes> ./parallel/wiener_parallel input.png psf.png output.png
   ```

## Performance Comparison

The parallel implementation utilizes MPI for distributed processing, providing significant speedup compared to the serial version, especially for large images. Each process handles a vertical slice of the image, and the results are combined at the end.
