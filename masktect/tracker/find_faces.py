import cv2 as cv


def find_faces(image):
    face_cascade = cv.CascadeClassifier('weights/opencv/haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return image
