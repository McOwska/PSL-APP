# components/video_capture.py
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import cv2
import numpy as np

class VideoCapture(QObject):
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)  # Możesz dostosować interwał

        self.recognized_text = ""
        self.show_text = False

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_captured.emit(frame)

    def set_recognized_text(self, text):
        self.recognized_text = text

    def set_show_text(self, show):
        self.show_text = show

    def release(self):
        self.cap.release()
