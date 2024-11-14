import torch
import numpy as np
import cv2

class ExtractLandmarksWithRTMP:
    def __init__(self, model):
        self.model = model

    def __call__(self, sample: list[np.array], confidence=0.4) -> tuple[list, list]:
        if type(sample) == tuple:
            return sample

        landmarks = []
        for frame in sample:
            h, w, c = frame.shape
            result = self.model(frame)
            result[:, 0] /= w
            result[:, 1] /= h
            landmarks.append(result)

        return landmarks

