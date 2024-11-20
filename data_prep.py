import cv2
import os

def extract_frames(video_path, output_folder):
    # Upewnij się, że folder wyjściowy istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Otwórz plik wideo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku wideo: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Koniec wideo
        
        # Zapisz klatkę jako obraz
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        if (frame_count % 3 == 0):
            flipped_frame = cv2.flip(frame, 1)  # Mirror flip the frame
            cv2.imwrite(frame_filename, flipped_frame)
        frame_count += 1

        print(f"Zapisano klatkę: {frame_filename}")
    
    # Zamknij plik wideo
    cap.release()
    print(f"Zakończono zapis klatek. Łącznie zapisano: {frame_count} klatek.")

# Przykład użycia
video_path = "eval_data/do_widzenia_1.mp4"  # Ścieżka do pliku wideo
output_folder = "eval_data/do_widzenia_1/frames"  # Folder na klatki
extract_frames(video_path, output_folder)
