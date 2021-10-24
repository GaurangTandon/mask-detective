import dataclasses

import os
import typing

import cv2 as cv
import numpy as np
import tqdm

from .video import ImageSequence
from ..classifier.train import load_model

import random

@dataclasses.dataclass
class Box:
    label: int
    x: float
    y: float
    w: float
    h: float
    group: int

    def scale_to_image(self, image_shape: typing.Tuple[int, int, int]):
        img_height, img_width, _ = image_shape
        scalefactor = 1.5

        x1, y1 = int((self.x - (self.w * scalefactor) / 2) * img_width), int(
            (self.y - (self.h * scalefactor) / 2) * img_height
        )
        x2, y2 = int((self.x + (self.w * scalefactor) / 2) * img_width), int(
            (self.y + (self.h * scalefactor) / 2) * img_height
        )

        if x1 < 0:
            x1 = 0
        if x2 > img_width - 1:
            x2 = img_width - 1
        if y1 < 0:
            y1 = 0
        if y2 > img_height - 1:
            y2 = img_height - 1

        return (x1, y1), (x2, y2)

    def get_centroid(self):
        return self.x, self.y

    def get_xywh(self, image_shape):
        (x1, y1), (x2, y2) = self.scale_to_image(image_shape)
        return x1, y1, x2 - x1, y2 - y1


class VideoAnnotator:
    def __init__(self, video_path):
        self.model = load_model()
        self.video_path = video_path
        self.video_sequence = ImageSequence().from_video(self.video_path, drop_rate=1)

    def analyze(self):
        os.system(
            f"""
            python weights/yolo/detect.py --weights weights/yolo.pt \\
            --source {self.video_path} \\
            --conf-thres 0.25 --iou-thres 0.1 --device 'cpu' \\
            --hide-labels --hide-conf --save-txt
        """
        )

    def annotations(self):
        """Gets the annotations in a usable format
        :return: The list for frames of list of dictionary of all bounding boxes
        """
        root_path = "weights/yolo/runs/detect/exp/labels"
        frame_basepath = os.path.join(
            root_path, self.video_path.split("/")[-1].split(".")[0]
        )

        def read_frame(frame_path: str):
            bounding_boxes = []
            with open(frame_path) as f:
                for line in f:
                    _, x, y, w, h = map(float, line.split(" "))
                    bounding_boxes.append(Box(label=0, x=x, y=y, w=w, h=h, group=0))
            return bounding_boxes

        frames_with_bounding_boxes = []
        try:
            for i in range(1, 10000000):
                path = f"{frame_basepath}_{i}.txt"
                data = read_frame(path)
                frames_with_bounding_boxes.append(data)
        except FileNotFoundError:
            pass
        # os.system(f"rm -r {root_path}")

        return frames_with_bounding_boxes

    def extract(self):
        """Extracts all the images from the given annotations
        :return: The fixed annotation data
        """
        annotation_data: typing.List[typing.List[Box]] = self.annotations()
        batch_images, batch_indices = [], []
        max_batch_size = 100

        for frame_idx, frame in tqdm.tqdm(
            enumerate(annotation_data), total=len(annotation_data)
        ):
            for box_idx, box in enumerate(frame):
                image = self.video_sequence.images[frame_idx]
                (x1, y1), (x2, y2) = box.scale_to_image(image_shape=image.shape)
                # print(x1, y1, x2, y2, image.shape)
                image = image[y1:y2, x1:x2]
                image = cv.resize(image, (224, 224))
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                batch_images.append(image)
                batch_indices.append((frame_idx, box_idx))
                if len(batch_images) == max_batch_size:
                    batch = np.stack(batch_images, axis=0) / 255
                    classification = self.model(batch)
                    classification = np.squeeze(classification >= 0.5)
                    for (index, label) in zip(batch_indices, classification):
                        annotation_data[index[0]][index[1]].label = label
                    batch_images, batch_indices = [], []

        if len(batch_images) > 0:
            batch = np.stack(batch_images, axis=0)
            classification = self.model(batch)
            classification = np.squeeze(classification >= 0.5)

            try:
                for (index, label) in zip(batch_indices, classification):
                    annotation_data[index[0]][index[1]].label = label
            except TypeError:
                # sometimes an error happens
                # the reason for it classification is of size 0
                # it is a subtle error and will be fixed by the
                # self reinforced learner in the next stage
                # so assign random label for now
                for index in batch_indices:
                    annotation_data[index[0]][index[1]].label = 1

        return annotation_data
