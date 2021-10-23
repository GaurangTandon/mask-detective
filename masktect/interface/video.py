import typing

import numpy as np
import cv2 as cv
import tqdm.auto as tqdm


class ImageSequence:
    def __init__(self, shape=(800, 400)):
        self.images: typing.List[np.ndarray] = []
        self.info: typing.List[typing.Dict[str, typing.Any]] = []
        self.shape = shape

    def transform(self, transformer: typing.Callable):
        """Transforms all the images in the sequence using the given function
        :param transformer: Function that takes an image and returns transformed image
        """
        assert len(self.images) == len(self.info)
        for idx in tqdm.trange(len(self.images)):
            self.images[idx], self.info[idx] = transformer(self.images[idx], self.info[idx])

    def from_video(self, filename: str, drop_rate: int = 1) -> "ImageSequence":
        """Generates the sequence of images from a given .mp4 file
        :param filename: The name of the file from which to generate the images
        :param drop_rate: Number of frames to out of which we should pick 1
        """
        video = cv.VideoCapture(filename)
        while True:
            success, image = video.read()
            if not success:
                break
            image = cv.resize(image, self.shape)
            self.images.append(image)
        self.images = self.images[0:len(self.images):drop_rate]
        self.info = [dict() for _ in self.images]
        return self

    def to_video(self, filename: str) -> None:
        """Generates a video from the given image sequence
        :param filename: Name of the video file to write to
        """
        print(filename)
        video = cv.VideoWriter(filename, 0, 1, self.shape)
        for image in self.images:
            video.write(image)
        video.release()

    def render(self) -> None:
        """Show the frame by frame rendering on the OpenCV viewer"""
        index = 0
        while cv.waitKey(100) != ord("q") and index < len(self.images):
            cv.imshow("frame", self.images[index])
            index += 1
        cv.destroyAllWindows()
