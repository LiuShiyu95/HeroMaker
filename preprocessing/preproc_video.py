import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video file:", video_path)
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()  # Read a single frame
        if not ret:
            break  # Exit loop if no more frames

        # Format the filename to be a 5-digit number
        frame_name = f"{frame_count:05d}.png"
        frame_path = os.path.join(output_dir, frame_name)

        # Save the frame to the output directory
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extraction completed. {frame_count} frames saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")

    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    args.output_dir = f"../video_reconstruction/all_sequences/{video_name}"

    extract_frames(args.video_path, args.output_dir)