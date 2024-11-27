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
        
        # Crop the bottom 200 pixels
        height, width, _ = frame.shape
        cropped_frame = frame[:height-200, :]

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        if (frame_count % 2 == 0):
            flipped_frame = cv2.flip(cropped_frame, 1)  # Mirror flip the frame
            cv2.imwrite(frame_filename, cropped_frame)
        frame_count += 1

        print(f"Saved frame: {frame_filename}")
    
    # Close the video file
    cap.release()
    print(f"Finished saving frames. Total frames saved: {frame_count}.")

# Example usage
video_path = "eval_data/migamy_pjmem/ig_6.mp4"  # Path to the video file
output_folder = "eval_data/migamy_pjmem/ig_6/frames"  # Folder for frames
extract_frames(video_path, output_folder)
