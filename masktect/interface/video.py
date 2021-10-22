import typing

import numpy as np
import cv2 as cv


class ImageSequence:
    def __init__(self, shape=(224, 224)):
        self.images: typing.List[np.ndarray] = []
        self.shape = shape

    def transform(self, transformer: typing.Callable):
        """Transforms all the images in the sequence using the given function
        :param transformer: Function that takes an image and returns transformed image
        """
        for idx in range(len(self.images)):
            self.images[idx] = transformer(self.images[idx])

    def from_video(self, filename: str) -> "ImageSequence":
        """Generates the sequence of images from a given .mp4 file
        :param filename: The name of the file from which to generate the images
        """
        video = cv.VideoCapture(filename)
        while True:
            success, image = video.read()
            if not success:
                break
            image = cv.resize(image, self.shape)
            self.images.append(image)
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
