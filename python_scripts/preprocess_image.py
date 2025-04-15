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

def save_psf_visual(psf, output_path):
    """Save PSF as an 8-bit PNG for visualization."""
    psf_scaled = (psf / psf.max() * 255).astype(np.uint8)
    cv2.imwrite(output_path, psf_scaled)
    print(f"PSF image saved as: {output_path}")

def preprocess_image(input_filename, target_size=(512, 512), sigma=10.0):
    raw_dir = os.path.join("images", "raw")
    processed_dir = os.path.join("images", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    input_path = os.path.join(raw_dir, input_filename)
    name, _ = os.path.splitext(input_filename)
    blurred_filename = f"{name}_blurred_sigma{sigma}.png"
    blurred_path = os.path.join(processed_dir, blurred_filename)
    psf_png_path = os.path.join(processed_dir, "psf_visual.png")
    psf_npy_path = os.path.join(processed_dir, "psf.npy")

    # Load and resize image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {input_path}")
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur to image
    blurred_img = cv2.GaussianBlur(resized_img, (0, 0), sigmaX=sigma)
    cv2.imwrite(blurred_path, blurred_img)
    print(f"Blurred image saved as: {blurred_path}")

    # Generate and save PSF
    psf = create_psf((target_size[1], target_size[0]), sigma)
    np.save(psf_npy_path, psf.astype(np.float32))
    print(f"PSF raw data saved as: {psf_npy_path}")
    print("PSF center value:", psf[target_size[1] // 2, target_size[0] // 2])
    print("PSF sum:", psf.sum())

    # Save PSF visualization
    save_psf_visual(psf, psf_png_path)

def main():
    parser = argparse.ArgumentParser(description="Resize, blur, and generate PSF.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Name of the input image file located in images/raw/")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Sigma for Gaussian blur and PSF")
    args = parser.parse_args()

    target_size = (args.width, args.height)
    preprocess_image(args.input_file, target_size, args.sigma)

if __name__ == "__main__":
    main()