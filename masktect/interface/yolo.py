import collections
import dataclasses
from math import sqrt

import os
import typing

import cv2 as cv
import numpy as np
import tqdm

from .video import ImageSequence
from ..classifier.train import load_model

VID_WIDTH = 640

VID_HEIGHT = 480


def rel_to_abs(x, y):
    return x * VID_WIDTH, y * VID_HEIGHT


def take_dist(x1, y1, x2, y2):
    (x1, y1), (x2, y2) = rel_to_abs(x1, y1), rel_to_abs(x2, y2)
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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

        x1, y1 = int((self.x - self.w / 2) * img_width), int((self.y - self.h / 2) * img_height)
        x2, y2 = int((self.x + self.w / 2) * img_width), int((self.y + self.h / 2) * img_height)

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


class VideoAnnotator:

    def __init__(self, video_path):
        self.model = load_model()
        self.video_path = video_path
        self.video_sequence = ImageSequence().from_video(self.video_path, drop_rate=1)

    def analyze(self):
        os.system(f"""
            python weights/yolo/detect.py --weights weights/yolo.pt \\
            --source {self.video_path} \\
            --conf-thres 0.25 --iou-thres 0.45 --device 'cpu' \\
            --hide-labels --hide-conf --save-txt
        """)

    def annotations(self, root_path: str = "weights/yolo/runs/detect/exp9/labels/"):
        """
        Gets the annotations in a usable format
        :param root_path: The prefix of all annotation files
        :return: The list for frames of list of dictionary of all bounding boxes
        """
        frame_basepath = os.path.join(root_path, self.video_path.split("/")[-1].split(".")[0])

        def read_frame(frame_path: str):
            bounding_boxes = []
            with open(frame_path) as f:
                for line in f:
                    _, x, y, w, h = map(float, line.split(' '))
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
        return frames_with_bounding_boxes

    def extract(self):
        """
        Extracts all the images from the given annotations
        :return: The fixed annotation data
        """
        annotation_data: typing.List[typing.List[Box]] = self.annotations()
        batch_images, batch_indices = [], []
        max_batch_size = 100

        for frame_idx, frame in tqdm.tqdm(enumerate(annotation_data), total=len(annotation_data)):
            for box_idx, box in enumerate(frame):
                image = self.video_sequence.images[frame_idx]
                (x1, y1), (x2, y2) = box.scale_to_image(image_shape=image.shape)
                # print(x1, y1, x2, y2, image.shape)
                image = image[y1:y2, x1:x2]
                image = cv.resize(image, (224, 224))
                batch_images.append(image)
                batch_indices.append((frame_idx, box_idx))
                if len(batch_images) == max_batch_size:
                    batch = np.stack(batch_images, axis=0)
                    classification = self.model(batch)
                    classification = np.squeeze(classification >= 0.5)
                    for (index, label) in zip(batch_indices, classification):
                        annotation_data[index[0]][index[1]].label = label
                    batch_images, batch_indices = [], []

        if batch_images:
            batch = np.stack(batch_images, axis=0)
            classification = self.model(batch)
            classification = np.squeeze(classification >= 0.5)
            for (index, label) in zip(batch_indices, classification):
                annotation_data[index[0]][index[1]].label = label

        self.annotation_data = annotation_data
        return self.annotation_data

    def apply_iou(self):
        THRESH = 1  # decide good threshold

        group_number = 1

        for frame_idx in range(0, len(self.annotation_data) - 1):
            this_frame = self.annotation_data[frame_idx]
            next_frame = self.annotation_data[frame_idx + 1]
            used = [False for _ in range(len(next_frame))]

            for box in this_frame:
                if box.group == 0:
                    # does not belong to any group
                    box.group = group_number
                    group_number += 1

                dist = 1e15
                best_idx = -1
                for other, box2 in enumerate(next_frame):
                    if used[other]:
                        continue
                    centroid_b = box.get_centroid()
                    centroid_b2 = box2.get_centroid()
                    d = take_dist(*centroid_b, *centroid_b2)
                    if best_idx == -1 or d < dist:
                        dist = d
                        best_idx = other

                if best_idx != -1 and dist <= THRESH:
                    # union
                    next_frame[best_idx].group = group_number
                    used[best_idx] = True

        self.total_groupcount = group_number

    def personalized_data(self):
        FPS = 15
        curr_time = 0
        group_timestamps: typing.List[typing.List[float]] = [[-1, -1] for _ in range(self.total_groupcount)]
        curr_groups = set([])

        for frame in self.annotation_data:
            seen_now = set([])
            for box in frame:
                group = box.group
                seen_now.add(group)
                if group in curr_groups:
                    continue
                curr_groups.add(group)
                group_timestamps[group][0] = round(curr_time)

            for group in curr_groups:
                if group not in seen_now:
                    group_timestamps[group][1] = round(curr_time)

            curr_time += 1 / FPS

        return group_timestamps

    def get_mask_stats(self):
        group_stat = [[0, 0] for _ in range(self.total_groupcount)]

        for frame in self.annotation_data:
            for box in frame:
                group_stat[box.group][box.label] += 1

        return group_stat
