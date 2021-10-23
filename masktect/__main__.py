import os
import argparse
import pickle

from .interface.yolo import VideoAnnotator
from .interface.video import ImageSequence
from .tracker.find_faces import find_faces
from .tracker.image_annotate import annotate_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--video", metavar="V", type=str, help="the video that needs to processed"
    )
    args = parser.parse_args()
    # Step 1: Save
    video = VideoAnnotator(args.video)
    # video.analyze()
    result = video.extract()
    with open("data/analyzed_data.pkl", 'wb') as f:
        pickle.dump(result, f)
    # Step 2: Load the annotations and fix
    with open("data/analyzed_data.pkl", "rb") as f:
        data = pickle.load(f)
    frames = ImageSequence(shape=(640, 480)).from_video(args.video, drop_rate=1)
    for i in range(len(data)):
        frames.info[i] = {"box": data[i]}
    frames.transform(annotate_image)
    frames.to_video("data/processed.avi")
    video.apply_iou()
    group_entry_exits = video.personalized_data()
    mask_stats = video.get_mask_stats()
    print(group_entry_exits)
    print(mask_stats)
    # frames.render()
