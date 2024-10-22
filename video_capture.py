import cv2
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import random
from random_functions import random_action

class VideoCapture(QObject):
    frame_captured = pyqtSignal(object)

    def __init__(self, update_interval=30):
        super().__init__()
        self.capture = cv2.VideoCapture(0) 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(update_interval)
        self.show_text = False

    def update_frame(self):
        ret, frame = self.capture.read()
        
        if self.show_text:
            size = frame.shape
            text = random_action()
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (size[0]-50, 50)  # Poprawiona pozycja na czytelniejszą (x, y)
            font_scale = 1
            color = (0, 255, 0)
            thickness = 2
            cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # Emitowanie klatki obrazu
        self.frame_captured.emit(frame)
        
    def set_show_text(self, value: bool):
        self.show_text = value

    def release(self):
        self.capture.release()
