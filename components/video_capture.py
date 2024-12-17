# components/video_capture.py
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import cv2
import numpy as np
import os
import glob

class VideoCapture(QObject):
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, folder_path, loop=True):
        super().__init__()
        self.folder_path = folder_path
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, '*.*')))  # Możesz filtrować po rozszerzeniu, np. '*.jpg'
        self.loop = loop
        self.current_index = 0
        self.total_images = len(self.image_paths)
        if self.total_images == 0:
            raise ValueError("Folder nie zawiera obrazów.")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.interval = 33  # Domyślny interwał pomiędzy klatkami (ms)
        self.timer.start(self.interval)

        self.recognized_text = ""
        self.show_text = False

    def capture_frame(self):
        if self.current_index < self.total_images:
            image_path = self.image_paths[self.current_index]
            frame = cv2.imread(image_path)
            flipped_frame = cv2.flip(frame, 1)  # Lustrzane odbicie klatki
            if frame is not None:
                self.frame_captured.emit(flipped_frame)
            else:
                print(f"Nie można wczytać obrazu: {image_path}")
            self.current_index += 1
        else:
            if self.loop:
                self.current_index = 0
            else:
                self.timer.stop()

    def set_recognized_text(self, text):
        self.recognized_text = text

    def set_show_text(self, show):
        self.show_text = show

    def release(self):
        self.timer.stop()

    def pause(self):
        """Zatrzymuje wyświetlanie kolejnych klatek, pauzując timer."""
        self.timer.stop()

    def start(self):
        """Wznawia wyświetlanie klatek od obecnego indeksu, uruchamiając ponownie timer."""
        if not self.timer.isActive():
            self.timer.start(self.interval)
