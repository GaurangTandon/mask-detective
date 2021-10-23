import typing

import numpy as np
import cv2 as cv
import mtcnn


def find_faces(image: np.ndarray, _info: typing.Dict[str, typing.Any]):
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(image)
    info = faces
    color = (0, 0, 255)

    for face in faces:
        (x, y, w, h) = face['box']
        cv.rectangle(image, (x, y), (x+w, y+h), color, 2)

        label = f"Conf: {np.round(face['confidence'], decimals=3)}"
        (wt, ht), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv.rectangle(image, (x, y - 20), (x + wt, y), color, -1)
        cv.putText(image, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return image, info
