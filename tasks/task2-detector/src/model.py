"""MNIST digit model scaffold for Task 2.

Detector code should call the inference function in this module. Training code
lives in train.py so detector.py stays focused on board detection, corner
geometry, and PnP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import torch
import torch.nn.functional as F

RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_classifier.npz"
_CACHED_MODEL = None
def get_cached_model(model_path: Path) -> object:
    """获取缓存的模型，确保在整个程序运行期间只读取一次硬盘"""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = load_mnist_model(model_path)
    return _CACHED_MODEL

def preprocess_mnist_crop(board_crop: ImageLike) -> np.ndarray:
    # TODO(student): Convert a detected board crop into classifier input.
    # convert board_crop to a single-channel image using cv2
    # resize the grayscale crop to 28x28, for example with cv2.resize(...)
    # normalize values to [0, 1]
    # convert the result to the tensor/array shape expected by your classifier
    # return normalized input array
    board_crop = np.array(board_crop, dtype=np.uint8)
    if len(board_crop.shape) == 3:
        gray_crop = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_crop = board_crop
        
    resized_crop = cv2.resize(gray_crop, (28, 28))
    normalized_crop = resized_crop.astype(np.float32) / 255.0
    
    input_array = np.expand_dims(normalized_crop, axis=(0, 1))
    
    return input_array



def load_mnist_model(model_path: Path = DEFAULT_MODEL_PATH) -> object:
    # TODO(student): Load your trained MNIST classifier from disk.
    # Input: model_path.
    # Output: a trained classifier model ready for inference.
    # If you use PyTorch, instantiate the model, load the weights, and switch to eval mode.
    from train import MNISTClassifier
    
    model = MNISTClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    return model



def predict_mnist_digit(model: object, model_input: np.ndarray) -> tuple[int, float]:
    # TODO(student): Run classifier inference and convert scores to digit/confidence.
    # convert model_input to a torch tensor if needed
    # run inference under torch.no_grad()
    # apply F.softmax(...) if the model returns logits
    # digit = argmax(probabilities)
    # confidence = probabilities[digit]
    # return digit, confidence
    tensor_input = torch.from_numpy(model_input)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = F.softmax(logits, dim=1)
        
        digit = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, digit].item())
        
    return digit, confidence


def classify_mnist_digit(board_crop: ImageLike, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[int, float]:
    # TODO(student): Classify the MNIST digit shown on one detected board crop.
    # model_input = preprocess_mnist_crop(board_crop)
    # model = load_mnist_model(model_path)
    # digit, confidence = predict_mnist_digit(model, model_input)
    # return digit, confidence
    model_input = preprocess_mnist_crop(board_crop)
    model = get_cached_model(model_path)
    digit, confidence = predict_mnist_digit(model, model_input)
    
    return digit, confidence
    
