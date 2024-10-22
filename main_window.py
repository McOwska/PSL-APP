from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QDesktopWidget, QSizePolicy, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from video_capture import VideoCapture
import cv2
import random

# Funkcja losowego zdania
def random_sentence():
    sentences = [
        "You're doing great!",
        "Keep pushing forward!",
        "Believe in yourself!",
        "Success is just around the corner!",
        "Stay positive and strong!"
    ]
    return random.choice(sentences)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moja Aplikacja PyQt z Kamerką")
        self.setGeometry(100, 100, 800, 600)
        self.center_window()

        self.init_ui()

        self.video_capture = VideoCapture()
        self.video_capture.frame_captured.connect(self.update_video_label)

        self.is_running = False

    def init_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.start_stop_button = QPushButton("START", self)
        self.start_stop_button.clicked.connect(self.toggle_start_stop)

        self.sentence_label = QLabel(self)
        self.sentence_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.start_stop_button, alignment=Qt.AlignCenter)
        layout.addWidget(self.sentence_label, alignment=Qt.AlignCenter)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def center_window(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        window_geometry = self.frameGeometry()
        screen_center = screen_geometry.center()
        window_geometry.moveCenter(screen_center)
        self.move(window_geometry.topLeft())

    def update_video_label(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def toggle_start_stop(self):
        if self.is_running:
            self.is_running = False
            self.start_stop_button.setText("START")
            sentence = random_sentence()
            self.sentence_label.setText(sentence)
            self.video_capture.show_text = False
        else:
            self.is_running = True
            self.start_stop_button.setText("STOP")
            self.sentence_label.setText("")
            self.video_capture.show_text = True

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)
