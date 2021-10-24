import typing

import numpy as np
import cv2 as cv

from ..interface.yolo import Box


def annotate_image(image: np.ndarray, info: typing.Dict[str, typing.List[Box]]):
    for box in info["box"]:
        if box.group == 0:
            continue
        (x1, y1), (x2, y2) = box.scale_to_image(image.shape)
        label = f'Person: {box.group}'
        color = (0, 0, 255) if box.label else (0, 255, 0)

        (wt, _ht), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv.rectangle(image, (x1, y1 - 20), (x1 + wt, y1), color, -1)
        cv.putText(
            image, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
        )
        cv.rectangle(
            image, (x1, y1), (x2, y2), color, 2
        )
    return image, info
