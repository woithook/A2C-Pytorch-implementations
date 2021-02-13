import math
import numpy as np
import torch
from torch import nn
from cv2 import resize


def main():
    x = np.zeros([210, 160, 3])
    x_post = process_frame(x)
    conv = nn.Conv2d(1, 32, 5, stride=1, padding=2)
    # conv = nn.Linear(14, 1)
    y = conv(x)


def process_frame(frame, crop=34):
    frame = frame[crop:crop + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame
