import os
import argparse
import pickle

from .interface.yolo import VideoAnnotator
from .interface.video import ImageSequence
from .tracker.image_annotate import annotate_image
from .interface.person import PersonTracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--video", metavar="V", type=str, help="the video that needs to processed"
    )
    args = parser.parse_args()
    # Step 1: Save
    video = VideoAnnotator(args.video)
    # video.analyze()
    # result = video.extract()
    # with open("data/analyzed_data.pkl", "wb") as f:
    #     pickle.dump(result, f)
    # Step 2: Load the annotations and fix
    with open("data/analyzed_data.pkl", "rb") as f:
        data = pickle.load(f)
    frames = ImageSequence(shape=(640, 480)).from_video(args.video, drop_rate=1)

    # Step 3: Compute the Trajectories
    tracker = PersonTracker(data)
    tracker.apply_iou()
    person_ts = tracker.person_timestamps()
    person_msk = tracker.person_mask()

    data = tracker.annotation_data
    for i in range(len(data)):
        frames.info[i] = {"box": data[i]}
    frames.transform(annotate_image)
    frames.to_video("data/processed.avi")

    with open('submission.txt', 'w') as f:
        FPS = 15
        f.write(f'Group\tFrames with mask on\tFrames with mask off\tEntry timestamp\tExit timestamp')
        for group in range(tracker.number_of_people):
            entry_frame, exit_frame = person_ts[group]
            entry_ts = entry_frame / FPS
            exit_ts = exit_frame / FPS
            entry_ts = round(entry_ts * 1000) / 1e3
            exit_ts = round(exit_ts * 1000) / 1e3
            mask_on, mask_off = person_msk[group]
            f.write(f'{group}\t{mask_on}\t{mask_off}\t{entry_ts}\t{exit_ts}\n')

    with open('submission.csv', 'w') as f:
        f.write(f'Frame\tTotal non-masked faces\tTotal masked faces\tNon-masked Face ROIs\tMasked Face ROIs')
        for frame, frame_data in enumerate(tracker.annotation_data):
            # no mask, mask
            roi = [[], []]
            for box in frame_data:
                roi[box.label].append(box.get_xywh(frames.images[0].shape))
            mask_off_count = len(roi[0])
            mask_on_count = len(roi[1])
            non_masked_roi = ";".join(map(lambda x: ",".join(map(str, x)), roi[0]))
            masked_roi = ";".join(map(lambda x: ",".join(map(str, x)), roi[0]))
            f.write(f'{frame}\t{mask_off_count}\t{mask_on_count}\t{non_masked_roi}\t{masked_roi}\n')

    tracker.train_model(video.model, video.video_sequence)
