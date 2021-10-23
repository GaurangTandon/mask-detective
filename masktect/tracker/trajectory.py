from math import sqrt
import sys
from typing import List
from ..interface.yolo import Box


def read_frame(frame_path: str):
    data = []
    with open(frame_path) as f:
        for line in f:
            c, x, y, w, h = map(int, line.split(' '))
            # last index is group number
            data.append([c, x, y, w, h, 0])

    return data


def read_all_frames(frame_basepath: str):
    all_data = []
    try:
        for i in range(1, 10000000):
            path = f"{frame_basepath}_{i}.txt"
            data = read_frame(path)
            all_data.append(data)
    except FileNotFoundError:
        pass
    return all_data


VID_WIDTH = 640
VID_HEIGHT = 480


def rel_to_abs(x1, y1, x2, y2):
    x1 *= VID_WIDTH
    x2 *= VID_WIDTH
    y1 *= VID_HEIGHT
    y2 *= VID_HEIGHT
    return x1, y1, x2, y2


def take_dist(x1, y1, x2, y2):
    x1, y1, x2, y2 = rel_to_abs(x1, y1, x2, y2)
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_centroid(box: Box):
    return box.x, box.y


def apply_iou(video_data):
    THRESH = 1

    group_number = 1

    for frame_idx in range(0, len(video_data) - 1):
        this_frame = video_data[frame_idx]
        next_frame = video_data[frame_idx + 1]
        used = [False for _ in range(len(next_frame))]

        for box in this_frame:
            if box[-1] == 0:
                # does not belong to any group
                box[-1] = group_number
                group_number += 1

            dist = 1e15
            best_idx = -1
            for other, box2 in enumerate(next_frame):
                if used[other]:
                    continue
                centroid_b = get_centroid(box)
                centroid_b2 = get_centroid(box2)
                d = take_dist(*centroid_b, *centroid_b2)
                if best_idx == -1 or d < dist:
                    dist = d
                    best_idx = other

            if best_idx != -1 and dist <= THRESH:
                # union
                next_frame[best_idx][-1] = group_number
                used[best_idx] = True

    return video_data, group_number


def personalized_data(grouped_frames, total_groupcount):
    FPS = 15
    curr_time = 0
    group_timestamps: List[List[float]] = [[-1, -1] for _ in range(total_groupcount)]
    curr_groups = set([])

    for frame in grouped_frames:
        seen_now = set([])
        for box in frame:
            group = box[-1]
            seen_now.add(group)
            if group in curr_groups:
                continue
            curr_groups.add(group)
            group_timestamps[group][0] = curr_time

        for group in curr_groups:
            if group not in seen_now:
                group_timestamps[group][1] = curr_time

        curr_time += 1 / FPS

    return group_timestamps


def get_mask_stats(grouped_frames, total_groupcount: int):
    group_stat = [[0, 0] for _ in range(total_groupcount)]

    # list of (frame number, absolute box coordinate) that
    # needs detection
    batch = []
    MX_BATCH_SIZE = 100

    def add_item(box: List[int]):
        nonlocal batch
        result = rel_to_abs(box[1], box[2], box[3], box[4])
        batch.append((box[-1], *result))
        if len(batch) == MX_BATCH_SIZE:
            # call animesh and update group_stat with call result
            pass
            batch = []

    for frame in grouped_frames:
        for box in frame:
            add_item(box)

    return group_stat


def main():
    video_name = sys.argv[1]
    video_data = read_all_frames(f'labels/{video_name}')
    grouped_frames, total_groupcount = apply_iou(video_data)
    group_timestamps = personalized_data(grouped_frames, total_groupcount)
    group_mask_stat = get_mask_stats(grouped_frames, total_groupcount)
    return group_mask_stat, group_timestamps


if __name__ == "__main__":
    main()
