# components/video_capture.py
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import cv2
import numpy as np

class VideoCapture(QObject):
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0, loop=True, width=800, height=600):
        super().__init__()
        self.camera_index = camera_index
        self.loop = loop
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Nie można otworzyć kamery o indeksie {self.camera_index}.")
        
        # Próba ustawienia rozdzielczości kamery
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Sprawdzenie, czy kamera ustawiła żądaną rozdzielczość
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_width, actual_height) != (self.width, self.height):
            print(f"Ostrzeżenie: Kamera nie obsługuje rozdzielczości {self.width}x{self.height}. Używana rozdzielczość to {actual_width}x{actual_height}.")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.interval = 33  # Domyślny interwał pomiędzy klatkami (ms) (~30 FPS)
        self.timer.start(self.interval)

        self.recognized_text = ""
        self.show_text = False

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Sprawdzenie, czy klatka ma oczekiwaną rozdzielczość
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            flipped_frame = cv2.flip(frame, 1)  # Lustrzane odbicie klatki
            self.frame_captured.emit(flipped_frame)
        else:
            print("Nie można przechwycić klatki z kamery.")
            if not self.loop:
                self.timer.stop()

    def set_recognized_text(self, text):
        self.recognized_text = text

    def set_show_text(self, show):
        self.show_text = show

    def release(self):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()

    def pause(self):
        """Zatrzymuje wyświetlanie kolejnych klatek, pauzując timer."""
        self.timer.stop()

    def start(self):
        """Wznawia wyświetlanie klatek, uruchamiając ponownie timer."""
        if not self.timer.isActive():
            self.timer.start(self.interval)
