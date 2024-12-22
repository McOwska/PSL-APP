from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea
from PyQt5.QtCore import Qt, QRect, QPropertyAnimation
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QFont, QFontDatabase
import markdown

class SidePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        id = QFontDatabase.addApplicationFont("assets/InriaSans-Regular.ttf")
        families = QFontDatabase.applicationFontFamilies(id)
        if families:
            custom_font_family = families[0]
            custom_font = QFont(custom_font_family)
        else:
            print("Failed to load the font.")
            custom_font = QFont()
            
        self.setFixedWidth(600)
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title = QLabel("Content Placeholder")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(custom_font)
        
        self.content = QLabel("Content Placeholder")
        self.content.setWordWrap(True)
        self.content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.content.setAlignment(Qt.AlignTop)
        self.content.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content.setFont(custom_font)
        
        self.scroll_layout.addWidget(self.title, stretch=0)
        self.scroll_layout.addWidget(self.content, stretch=1)
        
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)

        self.hide()
        
        self.main_layout.setObjectName("side_panel")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_content.setObjectName("scroll_content")
        self.title.setObjectName("title")
        self.content.setObjectName("content")
        
        with open('assets/styles_side_panels.qss', 'r') as file:
            self.setStyleSheet(file.read())

    def set_content(self, content):
        self.title.setText(content)
        match content:
            case "User instructions":
                with open('side_panels_content/user_instructions.md', 'r', encoding='utf-8') as file:
                    markdown_text = file.read()
                    html = self.convert_markdown_to_html(markdown_text)
                    self.content.setText(self.wrap_html(html))
            case "Currently available gestures":
                with open('side_panels_content/gestures_list.txt', 'r', encoding='utf-8') as file:
                    raw_text = file.read()
                    # Format into two columns
                    two_column_html = self.wrap_text_in_two_columns(raw_text)
                    self.content.setText(two_column_html)
            case "About the project":
                with open('side_panels_content/about_the_project.md', 'r', encoding='utf-8') as file:
                    markdown_text = file.read()
                    html = self.convert_markdown_to_html(markdown_text)
                    self.content.setText(self.wrap_html(html))

    def convert_markdown_to_html(self, markdown_text):
        """
        Converts Markdown text to HTML using the markdown library.
        """
        html = markdown.markdown(markdown_text)
        return html

    def wrap_html(self, html_content):
        """
        Wraps the HTML content with additional styling if necessary.
        """
        # You can add CSS styles here if needed
        styled_html = f"""
        <div style="text-align: justify; font-size: 14px;">
            {html_content}
        </div>
        """
        return styled_html

    def wrap_text(self, text):
        return f'<div style="text-align: justify;">{text}</div>'

    def wrap_text_in_two_columns(self, text):
        """
        Converts the text lines into a two-column HTML list with smaller bullet points.
        """
        # Split the text into lines
        lines = text.strip().split("\n")
        
        # Clean lines by removing unnecessary characters (e.g., '-' at the beginning)
        clean_lines = [line.lstrip("-").strip() for line in lines if line.strip()]
        
        # Split the list into two halves
        mid = (len(clean_lines) + 1) // 2
        left = clean_lines[:mid]
        right = clean_lines[mid:]
        
        # Function to create an HTML list with smaller bullet points
        def create_custom_list(items):
            list_html = "<ul style='list-style: none; padding-left: 0;'>"
            for item in items:
                list_html += f"""
                <li style="position: relative; padding-left: 10px; margin-bottom: 5px;">
                    <span style="
                        position: absolute;
                        left: 0;
                        top: 50%;
                        transform: translateY(-50%);
                        width: 5px;
                        height: 5px;
                        background-color: black;
                        border-radius: 50%;
                        display: inline-block;
                    "></span>
                    {item}
                </li>
                """
            list_html += "</ul>"
            return list_html
        
        # Create lists for both columns
        html_left = create_custom_list(left)
        html_right = create_custom_list(right)
        
        # Create an HTML table with two columns
        html = f"""
        <table width="100%">
            <tr>
                <td style="vertical-align: top; padding-right: 20px;">
                    {html_left}
                </td>
                <td style="vertical-align: top;">
                    {html_right}
                </td>
            </tr>
        </table>
        """
        return html

    def resizeEvent(self, event):
        self.setFixedHeight(self.parent().height())
        super().resizeEvent(event)

    def show_panel(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(QRect(-600, 0, 600, self.parent().height()))
        self.animation.setEndValue(QRect(375, 0, 600, self.parent().height()))
        self.animation.start()
        self.show()

    def hide_panel(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(QRect(375, 0, 600, self.parent().height()))
        self.animation.setEndValue(QRect(-600, 0, 600, self.parent().height()))
        self.animation.finished.connect(self.hide)
        self.animation.start()
