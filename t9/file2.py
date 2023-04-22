#!/usr/bin/python3
import cv2 as cv
import numpy as np

def findFace(image):
    face_cascade_db = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
    imageG = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # finding faces
    faces = face_cascade_db.detectMultiScale(imageG, scaleFactor=1.1, minNeighbors=19)
    # drawinf rectangles
    for (x,y,w,h) in faces:
        cv.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

    return image

cap = cv.VideoCapture('videos/ianEffects.mkv')

fps = 25.

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('videos/ianFace.mkv', fourcc, fps, (1920,  1080))


if not cap.isOpened():
    print("No video")
    exit()

frameNumber = 1

while True:
    # Capture frame-by-frame

    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (Video end?)")
        break

    # find face
    frame = findFace(frame)

    print(f'frame {frameNumber}', end='\r')
    frameNumber += 1
    out.write(frame)

cap.release()
out.release()
