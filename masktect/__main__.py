import argparse

from .interface.video import ImageSequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--video", metavar="V", type=str, help="the video that needs to processed"
    )
    args = parser.parse_args()
    data = ImageSequence().from_video(args.video)
    data.to_video(args.video[:-4] + "_processed.avi")
    data.render()
