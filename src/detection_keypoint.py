import sys
import cv2
import numpy as np
from pydantic import BaseModel
from typing import Optional, List

from ultralytics import YOLO
from ultralytics.engine.results import Results


class GetKeypoint(BaseModel):
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16


class DetectKeypoint:
    def __init__(self, yolov8_model: str = "yolov8m-pose.pt"):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        if "pose" not in self.yolov8_model:
            sys.exit("Model is not YOLOv8 pose")
        self.model = YOLO(self.yolov8_model)

    def extract_keypoint(self, keypoint: np.ndarray) -> List[float]:
        features = []
        for idx in range(17):
            x, y = keypoint[idx][0], keypoint[idx][1]
            features.extend([x, y])
        return features  # 34 values

    def get_xy_keypoint(self, result: Results) -> Optional[List[float]]:
        if result.keypoints is None:
            return None

        keypoints = result.keypoints.xyn.cpu().numpy()
        if len(keypoints) == 0:
            return None

        return self.extract_keypoint(keypoints[0])

    def __call__(self, image: np.ndarray) -> Results:
        return self.model.predict(image, save=False, verbose=False)[0]
