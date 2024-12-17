import cv2
import os

def extract_frames(video_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Rescale the frame to be half its original size
        height, width, _ = frame.shape
        resized_frame = cv2.resize(frame, (width // 2, height // 2))

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        if (frame_count % 3 == 0):
            flipped_frame = cv2.flip(resized_frame, 1)  # Mirror flip the frame
            cv2.imwrite(frame_filename, flipped_frame)
            print(f"Saved frame: {frame_filename}")
        frame_count += 1

        
    
    # Close the video file
    cap.release()
    print(f"Finished saving frames. Total frames saved: {frame_count}.")

# Example usage
video_path = "eval_data/migam_org/lekcja_7.mp4"  # Path to the video file
output_folder = "eval_data/migam_org/frames/7"  # Folder for frames
extract_frames(video_path, output_folder)
