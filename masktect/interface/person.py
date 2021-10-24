import typing

import numpy as np


class PersonTracker:

    DISTANCE_THRESHOLD = 1  # decide good threshold
    FRAMES_PER_SECOND = 15

    def __init__(self, annotations):
        self.annotation_data = annotations
        self.number_of_people = 0

    @staticmethod
    def point_distance(point_1, point_2, image_shape=(640, 480)):
        x1, y1 = point_1
        x2, y2 = point_2
        image_width, image_height = image_shape
        dx = np.abs(x2 - x1) * image_width
        dy = np.abs(y2 - y1) * image_height
        return np.sqrt(dx ** 2 + dy ** 2)

    def apply_iou(self):
        group_number = 1

        for frame_idx in range(0, len(self.annotation_data) - 1):
            this_frame = self.annotation_data[frame_idx]
            next_frame = self.annotation_data[frame_idx + 1]
            used = [False for _ in range(len(next_frame))]

            for box_1 in this_frame:
                if box_1.group == 0:
                    # does not belong to any group
                    box_1.group = group_number
                    group_number += 1

                best_dist, best_dist_idx = 1e15, -1
                for other, box_2 in enumerate(next_frame):
                    if used[other]:
                        continue
                    centroid_b1 = box_1.get_centroid()
                    centroid_b2 = box_2.get_centroid()
                    dist = point_distance(centroid_b1, centroid_b2)
                    if best_dist_idx == -1 or dist < best_dist:
                        best_dist = dist
                        best_dist_idx = other

                if best_dist_idx != -1 and best_dist <= DISTANCE_THRESHOLD:
                    next_frame[best_dist_idx].group = group_number
                    used[best_dist_idx] = True

        self.number_of_people = group_number

    def person_timestamps(self):
        group_timestamps: typing.List[typing.List[int]] = [
            [-1, -1] for _ in range(self.number_of_people)
        ]
        curr_groups = set([])

        for frame in self.annotation_data:
            seen_now = set([])
            for box in frame:
                group = box.group
                seen_now.add(group)
                if group in curr_groups:
                    continue
                curr_groups.add(group)
                group_timestamps[group][0] = frame

            for group in curr_groups:
                if group not in seen_now:
                    group_timestamps[group][1] = frame

        return group_timestamps

    def person_mask(self):
        group_stat = [[0, 0] for _ in range(self.number_of_people)]

        for frame in self.annotation_data:
            for box in frame:
                group_stat[box.group][box.label] += 1

        return group_stat

    def train_model(self, model, video_sequence):
        person_mask_values = self.person_mask()

        batch_images, batch_labels = [], []
        max_batch_size = 32

        for frame_idx, frame in tqdm.tqdm(
            enumerate(self.annotation_data), total=len(self.annotation_data)
        ):
            for box in frame:
                image = video_sequence.images[frame_idx]
                (x1, y1), (x2, y2) = box.scale_to_image(image_shape=image.shape)
                image = image[y1:y2, x1:x2]
                image = cv.resize(image, (224, 224))

                with_mask, without_mask = person_mask_values[box.group]
                if abs(with_mask - without_mask) / (with_mask + without_mask) < 0.4:
                    continue

                batch_images.append(image)
                batch_labels.append(with_mask > without_mask)

                if len(batch_images) == max_batch_size:
                    batch_images = np.stack(batch_images, axis=0)
                    batch_labels = np.expand_dims(np.array(batch_labels), axis=1)
                    model.fit(batch_images, batch_labels, epochs=1)
                    batch_images, batch_labels = [], []

        if len(batch_images) > 0:
            batch_images = np.stack(batch_images, axis=0)
            batch_labels = np.expand_dims(np.array(batch_labels), axis=1)
            model.fit(batch_images, batch_labels, epochs=1)