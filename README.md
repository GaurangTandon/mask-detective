# Mask Detective

Our take on the problem statement by Qualcomm in Megathon 2021. Summary of the problem statement ([full statement here](https://megathon.hackerearth.com/challenges/hackathon/megathon-draft/custom-tab/problem-statement/#Problem%20Statement)):

> Given a video sequence, you are required to detect all the faces in every frame, classify each face as masked or non-masked, uniquely identify each person and track the duration for which each person is masked and non-masked.

## Solution outline

### Person tracker 

We initially tried the norfair openpose tracker. It is better at tracking as it uses the full human body however it is extremely slow and impractical on CPU (and we don't know how to configure GPU drivers on Ubuntu :clown:)

Then we switched to YOLOv5 which tracks only the face, and hence it's a bit flaky but it works okay even for masked faces.

We used opencv to extract the crops of the faces from each video frame using the bounding box given to us by YOLO.

### Binary classifier: mask vs non-mask

We fine-tuned Google's EfficientNet because it is memory and runtime efficient. We used [albumentations](https://github.com/albumentations-team/albumentations) of various kinds to simulate the motion blur and poor cctv camera quality in the images. We found great accuracy (>97%) with EfficientB2 so we didn't try larger models as they may overfit.


### Self-annotating retrospective training

Our third solution component is inspired from tesla's auto-driving solution. Here, we use the annotations from the future frames to fix our previous mistakes. For example, if we predict "mask on" on 100 frames and "mask off" on 20 frames, then we use this information to retroactively fix our mistake on those remaining 20 frames.

This will help our model better fit on data from the wild.

## Datasets

TODO

## Submission PDF

TODO

## Performance

TODO

## Future work

TODO
