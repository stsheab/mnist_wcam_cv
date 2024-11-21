
import model.train as train

import cv2 as cv
import numpy as np
import torch


IMAGE_SIZE = 28


def transform_image(array: np.array) -> np.array:
    # cropping
    diff = array.shape[1] - array.shape[0]
    edge = diff//2
    croped = array[:, edge:-edge]
    # resize
    dim = (IMAGE_SIZE, IMAGE_SIZE)
    resized = cv.resize(croped, dim, interpolation=cv.INTER_AREA)
    return resized


def infer_model(image: np.array) -> int:
    normalize_divider = 255.0
    t = torch.Tensor(image.flatten())/normalize_divider
    model = train.MNISTNet2()
    model.load_state_dict(torch.load(r'model/model_fc_weights_black.pth'))
    res = model.forward(t).argmax(dim=0).item()
    return res


def infer_model_conv(image: np.array) -> int:
    normalize_divider = 255.0
    t = (torch.Tensor(image)/normalize_divider).reshape(1,
                                                        1, image.shape[0], image.shape[1])
    model = train.LeNet5()
    model.load_state_dict(torch.load(r'model/model_conv_weights_black.pth'))
    res = model.forward(t).argmax(dim=1).item()
    return res
