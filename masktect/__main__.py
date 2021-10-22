import argparse

from .interface.video import ImageSequence
from .tracker.find_faces import find_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--video", metavar="V", type=str, help="the video that needs to processed"
    )
    args = parser.parse_args()
    data = ImageSequence().from_video(args.video)
    data.transform(find_faces)
    data.to_video(args.video[:-4] + "_processed.avi")
    data.render()
