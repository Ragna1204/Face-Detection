# Micro Expression Recognizer

import torch
import torch.nn as nn
import cv2
import mediapipe as mp

class MERcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 0
            nn.ReLU(),                                              # 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 2
            nn.ReLU(),                                              # 3
            nn.MaxPool2d(2, 2),                                     # 4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 5
            nn.ReLU(),                                               # 6
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# 7
            nn.ReLU(),                                               # 8
            nn.MaxPool2d(2, 2),                                      # 9

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 10
            nn.ReLU(),                                                # 11
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 12
            nn.ReLU(),                                                # 13
            nn.MaxPool2d(2, 2),                                       # 14

            nn.Flatten(),                                            # 15
            nn.Linear(256 * 10 * 10, 1024),                          # 16
            nn.ReLU(),                                               # 17
            nn.Linear(1024, 512),                                    # 18
            nn.ReLU(),                                               # 19
            nn.Linear(512, 7)                                        # 20
        )

    def forward(self, x):
        return self.network(x)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    return device

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness','surprise']
def predict(image, model, device):
    xb = image.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1) # this is to pick class with the highest probability
    return classes[preds[0].item()]

findface = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
def face_box(frame):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    results = findface.process(frameRGB)
    my_faces = []
    if results.detections:
        for face in results.detections:
            bbox = face.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            my_faces.append([x, y, x + w, y + h])
    return my_faces