import cv2
import os
import argparse

def preprocess_image(input_filename, target_size=(512, 512), sigma=10.0):
    # Define input and output directories
    raw_dir = os.path.join("images", "raw")
    processed_dir = os.path.join("images", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Build input and output file paths
    input_path = os.path.join(raw_dir, input_filename)
    name, _ = os.path.splitext(input_filename)
    # The output filename includes the sigma value for clarity
    output_filename = f"{name}_blurred_sigma{sigma}.png"
    output_path = os.path.join(processed_dir, output_filename)
    
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {input_path}")
    
    # Resize the image to the target dimensions
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Apply synthetic Gaussian blur.
    # Using a kernel size of (0, 0) lets OpenCV compute it from sigma.
    blurred_img = cv2.GaussianBlur(resized_img, (0, 0), sigmaX=sigma)
    
    # Save the blurred image as a PNG file
    cv2.imwrite(output_path, blurred_img)
    print(f"Blurred image saved as: {output_path}")

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
