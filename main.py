import sys
import os
import json
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QFileSystemWatcher
from main_window import MainWindow
from models.model_LSTM_transformer import LSTMTransformerModel
from models.model_LSTM import LSTMModel
from helpers.gesture_recognition_handler import GestureRecognitionHandler
from rtmpose.rtmpose import RTMPoseDetector
from helpers.transforms import ExtractLandmarksWithRTMP

def load_stylesheet(app, file_path="assets/styles.qss"):
    with open(file_path, "r") as file:
        app.setStyleSheet(file.read())

if __name__ == "__main__":
    extractor = RTMPoseDetector('rtmpose/end2end.onnx')
    transform = ExtractLandmarksWithRTMP(extractor)
    
    labels = 'labels.json'
    label_map = None
    if os.path.isfile(labels):
        with open(labels, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

    if label_map is not None:
        actions = np.array(list(label_map.keys()))
        num_classes = 219
    
    model_type = 'LSTM-Transformer'
    model_path = 'models/pretrained/LSTM-Transformer_RGB_more_copy_2_RTMP_.pth'
    
    # model_type = 'LSTM'
    # model_path = 'models/pretrained/LSTM_RGB_more_copy_2_RTMP_.pth'
    
    if model_type == 'LSTM-Transformer':
        print('using LSTM-Transformer')
        model = LSTMTransformerModel(128, 1, num_classes, 64)
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'LSTM':
        print('using LSTM')
        model = LSTMModel(128, 1, num_classes)
        model.load_state_dict(torch.load(model_path))

    prediction_handler = GestureRecognitionHandler(model, label_map)

    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = MainWindow(prediction_handler, transform)
    window.show()
    sys.exit(app.exec_())
