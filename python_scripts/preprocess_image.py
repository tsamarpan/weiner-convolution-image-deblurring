import cv2
import os
import argparse
import numpy as np

def create_psf(shape, sigma):
    """Generate a centered 2D Gaussian PSF."""
    h, w = shape
    y, x = np.indices((h, w))
    center_x, center_y = w // 2, h // 2
    psf = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf

def preprocess_image(input_filename, target_size=(512, 512), sigma=10.0):
    # Define input and output directories
    raw_dir = os.path.join("images", "raw")
    processed_dir = os.path.join("images", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Build file paths
    input_path = os.path.join(raw_dir, input_filename)
    name, _ = os.path.splitext(input_filename)
    blurred_filename = f"{name}_blurred_sigma{sigma}.png"
    blurred_path = os.path.join(processed_dir, blurred_filename)
    psf_path = os.path.join(processed_dir, "psf.png")
    
    # Load and resize image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {input_path}")
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(resized_img, (0, 0), sigmaX=sigma)

    # Save blurred image
    cv2.imwrite(blurred_path, blurred_img)
    print(f"Blurred image saved as: {blurred_path}")

    # Generate and save PSF as grayscale image
    psf = create_psf((target_size[1], target_size[0]), sigma)  # shape = (height, width)
    psf_img = (psf * 255).astype(np.uint8)
    cv2.imwrite(psf_path, psf_img)
    print(f"PSF saved as: {psf_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a raw image by resizing and applying a synthetic Gaussian blur."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Name of the input image file located in images/raw/")
    parser.add_argument("--width", type=int, default=512, help="Target width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Target height in pixels (default: 512)")
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Sigma value for Gaussian blur (default: 10.0)")
    args = parser.parse_args()
    
    target_size = (args.width, args.height)
    preprocess_image(args.input_file, target_size, args.sigma)

if __name__ == "__main__":
    main()