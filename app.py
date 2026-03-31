import cv2
import argparse
import os
import sys
from utils import run_detection, visualize_results, MODEL

def main():
    parser = argparse.ArgumentParser(description="Human Detection CLI App (HOG + SVM)")
    
    # Arguments
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("--step", type=int, default=64, help="Sliding window step size (default: 64)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Decision threshold (default: 0.2)")
    parser.add_argument("--output", default="result.jpg", help="Output filename (default: result.jpg)")

    args = parser.parse_args()

    # 1. Check Model Status
    if MODEL is None:
        print("[ERROR] Model file 'HOG_detection.pkl' not found.")
        print("[ERROR] Please ensure the model is trained and saved in the current directory.")
        sys.exit(1)

    # 2. Check Input File
    if not os.path.exists(args.input):
        print(f"[ERROR] Input path '{args.input}' does not exist.")
        sys.exit(1)

    print(f"\n--- Processing Image: {args.input} ---")
    
    try:
        # Load Image
        img = cv2.imread(args.input)
        if img is None:
            print("[ERROR] Could not decode image. Please check the file format.")
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Run Detection
        print(f"[*] Scanning with step_size={args.step} and threshold={args.threshold}...")
        detected_boxes = run_detection(img_rgb, step_size=args.step, threshold=args.threshold)

        # 4. Show/Save Results
        if len(detected_boxes) > 0:
            print(f"[SUCCESS] Found {len(detected_boxes)} potential regions.")
            print(f"[*] Saving visualization to: {args.output}")
            visualize_results(img_rgb, detected_boxes, save_path=args.output)
            print("[DONE] Detection completed successfully.")
        else:
            print("[INFO] No objects detected in this image.")

    except Exception as e:
        print(f"[CRITICAL ERROR]: {str(e)}")

if __name__ == "__main__":
    main()