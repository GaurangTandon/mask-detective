import os
import argparse
import pickle

from .interface.yolo import VideoAnnotator
from .interface.video import ImageSequence
from .tracker.image_annotate import annotate_image
from .interface.person import PersonTracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--video", metavar="V", type=str, help="the video that needs to processed")
    args = parser.parse_args()
    # Step 1: Compute YOLO Annotations and fix with Efficient Net
    video = VideoAnnotator(args.video)
    video.analyze()
    result = video.extract()
    with open("data/analyzed_data.pkl", "wb") as f:
        pickle.dump(result, f)
    with open("data/analyzed_data.pkl", "rb") as f:
        data = pickle.load(f)
    frames = ImageSequence(shape=(640, 480)).from_video(args.video, drop_rate=1)
    # Step 2: Compute the Trajectories
    tracker = PersonTracker(data)
    tracker.apply_iou()
    person_ts = tracker.person_timestamps()
    person_msk = tracker.person_mask()
    FPS = 15
    EXISTENCE_FPS_THRESHOLD = 2 * FPS
    EXISTENCE_TIME_THRESHOLD = EXISTENCE_FPS_THRESHOLD / FPS

    # Step 3: Writing out the files as submissions
    # delete artifacts on bounding boxes fringe effects in yolov5
    for frame in tracker.annotation_data:
        for box in frame:
            group = box.group
            exists_long = person_ts[group][1] - person_ts[group][0]
            # exists for less than one second (15 fps)
            if exists_long < EXISTENCE_FPS_THRESHOLD:
                box.group = 0

    for i in range(len(tracker.annotation_data)):
        frames.info[i] = {"box": tracker.annotation_data[i]}
    frames.transform(annotate_image)
    frames.to_video("data/processed.avi")

    with open('submission.txt', 'w') as f:
        f.write(f'Group\tFrames with mask on\tFrames with mask off\tEntry timestamp\tExit timestamp\n')
        for group in range(1, tracker.number_of_people + 1):
            entry_frame, exit_frame = person_ts[group]
            entry_ts = entry_frame / FPS
            exit_ts = exit_frame / FPS
            exists_long = exit_ts - entry_ts
            # exists for less than one second
            if exists_long < EXISTENCE_TIME_THRESHOLD:
                continue
            entry_ts = round(entry_ts * 1000) / 1e3
            exit_ts = round(exit_ts * 1000) / 1e3
            mask_on, mask_off = person_msk[group]
            f.write(f'{group}\t{mask_on}\t{mask_off}\t{entry_ts}\t{exit_ts}\n')

    with open('submission.csv', 'w') as f:
        f.write(f'Frame,Total non-masked faces,Total masked faces,Non-masked Face ROIs,Masked Face ROIs\n')
        for frame, frame_data in enumerate(tracker.annotation_data):
            # no mask, mask
            roi = [[], []]
            for box in frame_data:
                if box.group == 0:
                    continue
                roi[box.label].append(box.get_xywh(frames.images[0].shape))
            mask_off_count = len(roi[1])
            mask_on_count = len(roi[0])
            non_masked_roi = ";".join(map(lambda x: ",".join(map(str, x)), roi[1]))
            masked_roi = ";".join(map(lambda x: ",".join(map(str, x)), roi[0]))
            f.write(f'{frame},{mask_off_count},{mask_on_count},{non_masked_roi},{masked_roi}\n')

    tracker.train_model(video.model, video.video_sequence)
