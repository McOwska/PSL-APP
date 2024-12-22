from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QDesktopWidget, QSizePolicy, QPushButton, QSpacerItem
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QTimer
from components.video_capture import VideoCapture
from components.menu_component import MenuComponent
import cv2
from components.side_panel import SidePanel
from helpers.format_sentence import format_sentence
from assets.shadow_effect import shadow_effect


class MainWindow(QMainWindow):
    def __init__(self, prediction_handler, transform):
        super().__init__()
        
        # Load custom font
        id = QFontDatabase.addApplicationFont("assets/InriaSans-Regular.ttf")
        families = QFontDatabase.applicationFontFamilies(id)
        if families:
            custom_font_family = families[0]
            self.custom_font = QFont(custom_font_family)
        else:
            self.custom_font = QFont()
        
        self.prediction_handler = prediction_handler
        self.transform = transform
        self.setWindowTitle("Polish Sign Language Translator")
        self.setWindowIcon(QIcon('assets/logo.png'))
        self.setGeometry(100, 100, 1500, 1000)
        self.center_window()

        self.is_running = False    # Indicates if prediction process is active
        self.is_paused = False     # Indicates if video is paused

        self.init_ui()

        video_source = 'eval_data/migam_org/frames/2'
        self.video_capture = VideoCapture(loop=True)
        self.video_capture.frame_captured.connect(self.update_video_label)

        self.reset_text_timer = QTimer()
        self.reset_text_timer.setInterval(2000)
        self.reset_text_timer.timeout.connect(self.reset_recognized_text)

        self.recognized_gestures = []

        # Initialize prediction buffer
        # Each entry is a tuple: (gesture, confidence)
        self.prediction_buffer = []

    def init_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setGraphicsEffect(shadow_effect())
        
        self.start_stop_button = QPushButton("START", self)
        self.start_stop_button.clicked.connect(self.toggle_start_stop)
        self.start_stop_button.setFont(self.custom_font)
        self.start_stop_button.setGraphicsEffect(shadow_effect())
        self.start_stop_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_stop_button.setMaximumWidth(150)
        self.start_stop_button.setProperty("status", "stopped")

        # # Second button: PAUSE/RESUME
        # self.pause_resume_button = QPushButton("PAUSE", self)
        # self.pause_resume_button.clicked.connect(self.toggle_pause_resume)
        # self.pause_resume_button.setFont(self.custom_font)
        # self.pause_resume_button.setGraphicsEffect(shadow_effect())
        # self.pause_resume_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.pause_resume_button.setMaximumWidth(150)

        self.sentence_label = QLabel(self)
        self.sentence_label.setAlignment(Qt.AlignCenter)
        self.sentence_label.setFont(self.custom_font)
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)


        self.menu_list = MenuComponent(self)
        self.menu_list.option_clicked.connect(self.show_side_panel)

        right_layout = QVBoxLayout()
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        right_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Add START/STOP and PAUSE/RESUME buttons in a horizontal layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_stop_button, alignment=Qt.AlignCenter)
        # buttons_layout.addWidget(self.pause_resume_button, alignment=Qt.AlignCenter)

        right_layout.addLayout(buttons_layout)
        right_layout.addWidget(self.sentence_label, alignment=Qt.AlignCenter)
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.menu_list, alignment=Qt.AlignLeft)
        main_layout.addStretch(1)
        main_layout.addLayout(right_layout)
        main_layout.addStretch(1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.side_panel = SidePanel(self)

    def center_window(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        window_geometry = self.frameGeometry()
        screen_center = screen_geometry.center()
        window_geometry.moveCenter(screen_center)
        self.move(window_geometry.topLeft())

    def update_video_label(self, frame):
        # If paused, do not update the frame (freeze on the last frame)
        if self.is_paused:
            return

        mirrored_frame = cv2.flip(frame, 1)
        if self.is_running:
            recognized_action, confidence = self.prediction_handler.process_frame(mirrored_frame, self.transform)

            color_rect = (255, 255, 255)
            # Rozmiar ramki
            height, width, _ = frame.shape

            # Ustawienia prostokąta
            rect_height = 50  # Wysokość prostokąta (możesz dostosować)
            top_left = (0, height - rect_height)
            bottom_right = (width, height)

            # Rysowanie białego prostokąta bez obramowania
            cv2.rectangle(frame, top_left, bottom_right, color_rect, cv2.FILLED)

            if recognized_action is not None:
                # Update the prediction buffer with (gesture, confidence)
                self.prediction_buffer.append((recognized_action, confidence))
                if len(self.prediction_buffer) > 5:
                    self.prediction_buffer.pop(0)  # Keep only the last 5 predictions

                # Check if any gesture appears at least 4 times in the last 5 predictions
                gesture_counts = {}
                gesture_confidences = {}
                for gesture, conf in self.prediction_buffer:
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                    gesture_confidences.setdefault(gesture, []).append(conf)

                consensus_gesture = None
                for gesture, count in gesture_counts.items():
                    if count >= 4:
                        # Calculate average confidence for this gesture
                        avg_confidence = sum(gesture_confidences[gesture]) / len(gesture_confidences[gesture])
                        if avg_confidence >= 0.8:
                            consensus_gesture = gesture
                            break

                if consensus_gesture:
                    # To avoid repetitive updates, check if the last recognized gesture is different
                    if not self.recognized_gestures or self.recognized_gestures[-1] != consensus_gesture:
                        self.recognized_gestures.append(consensus_gesture)
                        self.video_capture.set_recognized_text(consensus_gesture)
                        self.reset_text_timer.start()
                        self.prediction_handler.reset_buffer()

            if self.video_capture.recognized_text:
                # Ustawienia tekstu
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text = self.video_capture.recognized_text
                color_text = (19, 34, 52)  


                # Pozycja tekstu (środek prostokąta)
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size
                text_x = (width - text_width) // 2
                text_y = height - (rect_height // 2) + (text_height // 2)

                # Rysowanie tekstu na prostokącie
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color_text,
                    thickness,
                    cv2.LINE_AA
                )


        mirrored_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = mirrored_frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(mirrored_frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def reset_recognized_text(self):
        self.video_capture.set_recognized_text("")

    def toggle_start_stop(self):
        self.is_running = not self.is_running
        
        self.start_stop_button.setText("STOP" if self.is_running else "START")
        self.start_stop_button.setProperty("status", "running" if self.is_running else "stopped")
        self.start_stop_button.style().unpolish(self.start_stop_button)
        self.start_stop_button.style().polish(self.start_stop_button)
        
        self.video_capture.set_show_text(self.is_running)
        
        if self.is_running:
            self.recognized_gestures.clear()
            self.prediction_buffer.clear()  # Clear the prediction buffer when starting
            self.sentence_label.setText("")
            self.reset_text_timer.start()
        else:
            # Stopping prediction does not clear the current frame
            # It only stops the recognition process
            self.sentence_label.setText(format_sentence(self.recognized_gestures))
            self.sentence_label.adjustSize()

    def toggle_pause_resume(self):
        # Toggle the pause state of the video
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_resume_button.setText("RESUME")
            self.video_capture.pause()  # Pause the video
        else:
            self.pause_resume_button.setText("PAUSE")
            self.video_capture.start()  # Resume the video

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)
        
    def show_side_panel(self, content):
        if content == "":
            self.side_panel.hide_panel()
            return
        if self.side_panel.isVisible():
            if self.side_panel.title.text() == content:
                self.side_panel.hide_panel()
            else:
                self.is_switching_panel_content = True
                self.side_panel.hide_panel()
                QTimer.singleShot(300, lambda: self.set_new_panel_content(content))
        else:
            self.side_panel.set_content(content)
            self.side_panel.show_panel()

    def set_new_panel_content(self, content):
        if getattr(self, 'is_switching_panel_content', False):
            self.side_panel.set_content(content)
            self.side_panel.show_panel()
            self.is_switching_panel_content = False
