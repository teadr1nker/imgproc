#!/usr/bin/python3
import cv2 as cv
import numpy as np

from effects import effects

cap = cv.VideoCapture('videos/ian.mkv')
fps = 25.

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('videos/ianEffects.mkv', fourcc, fps, (1920,  1080))

fiveSeconds = fps * 5

if not cap.isOpened():
    print("No video")
    exit()

frameNumber = 1
effect = 0

while True:
    # Capture frame-by-frame

    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (Video end?)")
        break

    # apply effect
    if len(effects) > effect:
        name = effects[effect].__name__
        print(f'Using effect {effect} {name} on frame {frameNumber}' + ' ' * 16, end='\r')
        frame = effects[effect](frame)
        frame = cv.putText(frame,
                           f'Effect: {name}',
                           [64, 64],
                           cv.FONT_HERSHEY_SIMPLEX,
                           2,
                           [0, 0, 0],
                           4,
                           cv.LINE_AA)

    if frameNumber % fiveSeconds == 0:
        effect += 1

    frameNumber += 1
    out.write(frame)



cap.release()
out.release()

