import argparse
import pickle

from .interface.video import ImageSequence
from .tracker.find_faces import find_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--video", metavar="V", type=str, help="the video that needs to processed"
    )
    args = parser.parse_args()
    data = ImageSequence().from_video(args.video, 5)
    data.transform(find_faces)
    data.to_video(args.video[:-4] + "_processed.avi")
    data.write_submission(args.video[:-4] + "_submission.txt")
    with open('data/video/oxford_1_info.pkl', 'wb') as f:
        pickle.dump(data.info, f)
    data.render()
