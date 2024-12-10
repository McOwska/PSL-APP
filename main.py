import sys
import os
import json
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QFileSystemWatcher
from PyQt5.QtGui import QIcon
from main_window import MainWindow
from models.model_LSTM_transformer import LSTMTransformerModel
from models.model_LSTM_transformer_2 import LSTMTransformerModel2
from models.model_proper_attention import LSTMWithAttention
from models.model_proper_attention_2 import LSTMWithAttention as LSTMWithAttention2
from models.model_4lsmt_attention import LSTMs4WithAttention
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
    
    # model_type = 'LSTM-Transformer'
    # model_path = 'models/pretrained/LSTM-Transformer_RGB_more_copy_2_RTMP_.pth'

    # model_type = 'LSMT-Transformer-2'
    # model_path = 'models/pretrained/LSTM-Transformer-2_7_12_.pth'
    
    # model_type = 'LSTM'
    # model_path = 'models/pretrained/LSTM_RGB_more_copy_2_RTMP_.pth'

    # model_type = 'LSTM-Attention'
    # model_path = 'models/pretrained/ProperAttention_7_12_feature_attention.pth'

    # model_type = 'LSTM-Attention-2'
    # model_path = 'models/pretrained/ProperAttentiondifferent input norm copy.pth'

    model_type = 'LSTM-Attention-4'
    model_path = 'models/pretrained/ProperAttention4LSTMs_1st_try.pth'
    
    if model_type == 'LSTM-Transformer':
        print('using LSTM-Transformer')
        model = LSTMTransformerModel(128, 1, num_classes, 64)
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'LSTM':
        print('using LSTM')
        model = LSTMModel(128, 1, num_classes)
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'LSMT-Transformer-2':
        print('using LSTM-Transformer-2')
        model = LSTMTransformerModel2(128, 1, num_classes, 64)
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'LSTM-Attention':
        print('using LSTM-Attention')
        model = LSTMWithAttention(210, 128, num_classes, 1)
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'LSTM-Attention-2':
        print('using LSTM-Attention-2')
        model = LSTMWithAttention2(84, 128, num_classes, 1)
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'LSTM-Attention-4':
        print('using LSTM-Attention-4')
        model = LSTMs4WithAttention(96, 64, num_classes, 1)
        model.load_state_dict(torch.load(model_path))

    prediction_handler = GestureRecognitionHandler(model, label_map)

    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = MainWindow(prediction_handler, transform)
    window.show()
    sys.exit(app.exec_())
