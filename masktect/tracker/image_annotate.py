import typing

import numpy as np
import cv2 as cv

from ..interface.yolo import Box


def annotate_image(image: np.ndarray, info: typing.Dict[str, typing.List[Box]]):
    for box in info["box"]:
        (x1, y1), (x2, y2) = box.scale_to_image(image.shape)
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0) if box.label else (0, 255, 0), 2)
    return image, info

