from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QDesktopWidget, QSizePolicy, QPushButton, QSpacerItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer
from components.video_capture import VideoCapture
from components.menu_component import MenuComponent
import cv2
from components.side_panel import SidePanel
from helpers.format_sentence import format_sentence
from PyQt5.QtGui import QFont, QFontDatabase
from assets.shadow_effect import shadow_effect

class MainWindow(QMainWindow):
    def __init__(self, prediction_handler, transform):
        super().__init__()
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

        self.is_running = False    # Czy proces predykcji jest aktywny
        self.is_paused = False     # Czy wideo jest spauzowane

        self.init_ui()

        video_source = 'eval_data/migam_org/frames/2'
        self.video_capture = VideoCapture(loop=True)
        self.video_capture.frame_captured.connect(self.update_video_label)

        self.reset_text_timer = QTimer()
        self.reset_text_timer.setInterval(2000)
        self.reset_text_timer.timeout.connect(self.reset_recognized_text)

        self.recognized_gestures = []

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

        # Drugi przycisk: PAUSE/RESUME
        self.pause_resume_button = QPushButton("PAUSE", self)
        self.pause_resume_button.clicked.connect(self.toggle_pause_resume)
        self.pause_resume_button.setFont(self.custom_font)
        self.pause_resume_button.setGraphicsEffect(shadow_effect())
        self.pause_resume_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pause_resume_button.setMaximumWidth(150)

        self.sentence_label = QLabel(self)
        self.sentence_label.setAlignment(Qt.AlignCenter)
        self.sentence_label.setFont(self.custom_font)
        self.sentence_label.setWordWrap(True)

        self.menu_list = MenuComponent(self)
        self.menu_list.option_clicked.connect(self.show_side_panel)

        right_layout = QVBoxLayout()
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        right_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Dodanie przycisków START/STOP i PAUSE/RESUME w poziomym układzie
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_stop_button, alignment=Qt.AlignCenter)
        buttons_layout.addWidget(self.pause_resume_button, alignment=Qt.AlignCenter)

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
        # Jeśli pauza jest aktywna, to nie aktualizujemy klatki (zatrzymujemy się na ostatniej)
        if self.is_paused:
            return

        mirrored_frame = cv2.flip(frame, 1)
        if self.is_running:
            recognized_action, confidence = self.prediction_handler.process_frame(mirrored_frame, self.transform)
            if recognized_action is not None:
                self.recognized_gestures.append(f"{recognized_action}")
                self.video_capture.set_recognized_text(recognized_action)
                self.reset_text_timer.start()

            if self.video_capture.recognized_text:
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                position = (mirrored_frame.shape[1] - 10 - cv2.getTextSize(self.video_capture.recognized_text, font, font_scale, thickness)[0][0], 30)
                color = (255, 0, 0)
                cv2.putText(mirrored_frame, self.video_capture.recognized_text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        mirrored_frame_rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
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
            self.sentence_label.setText("")
            self.reset_text_timer.start()
        else:
            # Zatrzymanie predykcji nie kasuje jednak bieżącej klatki
            # Zatrzymuje tylko proces rozpoznawania
            self.sentence_label.setText(format_sentence(self.recognized_gestures))

    def toggle_pause_resume(self):
        # Przełączenie stanu pauzy wideo
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_resume_button.setText("RESUME")
            self.video_capture.pause()  # Wstrzymanie wideo
        else:
            self.pause_resume_button.setText("PAUSE")
            self.video_capture.start()  # Wznowienie wideo

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
        if self.is_switching_panel_content:
            self.side_panel.set_content(content)
            self.side_panel.show_panel()
            self.is_switching_panel_content = False
