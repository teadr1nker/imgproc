#!/usr/bin/python3
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# drawing images with figures
def drawFigures(pic, x1, x2, y1, y2):
    image = np.copy(pic)
    clusterS = []
    clusterT = []
    for i in range(0, 100):
        clusterS.append([np.random.randint(250, 550),
                         np.random.randint(x1, x2)])
        clusterT.append([np.random.randint(250, 550),
                         np.random.randint(y1 , y2)])

    imageCenters = []

    for p in clusterS:
        cv.rectangle(image, (p[0], p[1]), (p[0] + 6, p[1] + 6), (0, 0, 255),-1)
        imageCenters.append([p[0]+3, p[1]+3])

    for p in clusterT:
        pts = np.array([[p[0], p[1]], [p[0] + 6, p[1]], [p[0] + 6, p[1] + 6]])
        cv.fillPoly(image, [pts], color=(255, 0, 0))
        imageCenters.append([p[0]+4, p[1]+2])

    return image, np.float32(imageCenters)

# classifying figures
def splitSample(pic, centers, number = 1):
    image = np.copy(pic)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, label, center = cv.kmeans(centers, 2, None,
                                   criteria, 10, cv.KMEANS_PP_CENTERS)

    for i, point in enumerate(centers):
        if label[i] == [0]:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv.circle(image, (int(point[0]),int(point[1])),
                  10, color, 1)
    for point in center:
        cv.circle(image, (int(point[0]),int(point[1])),
                  6, (0, 255, 0), -1)

    cv.imwrite(f'images/classes{number}.png', image)

    return None

# classify circles
def knnCircles(pic, centers, number=1):
    image = np.copy(pic)
    circleCenters = []
    for i in range(3):
        for j in range(3):
            x, y =200 + j*100, 200 + i * 100
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)
            circleCenters.append([x, y])

    knn = cv.ml.KNearest_create()

    circleCenters = np.float32(circleCenters)

    responses0 = np.random.randint(0, 1, (100, 1)).astype(np.float32)
    responses1 = np.random.randint(1, 2, (100, 1)).astype(np.float32)
    responses = np.vstack((responses0, responses1))

    knn.train(centers, cv.ml.ROW_SAMPLE, responses)

    ret, results, neighbours, dist = knn.findNearest(circleCenters, 5)

    for i, result in enumerate(results):
        dst = np.int32(circleCenters[i] + 5)
        if result == 0:
            txt = 's'
        else:
            txt = 't'
        image = cv.putText(image,
                           txt,
                           dst,
                           cv.FONT_HERSHEY_SIMPLEX,
                           1,
                           [255, 0, 255],
                           2,
                           cv.LINE_AA)

    cv.imwrite(f'images/circles{number}.jpg', image)

# create white image
whiteBox = np.zeros((600, 800, 3), np.uint8)
whiteBox.fill(255)

# generate images
image1, imageCenters1 = drawFigures(whiteBox, 0, 225, 325, 550)
image2, imageCenters2 = drawFigures(whiteBox, 0, 290, 300, 550)
image3, imageCenters3 = drawFigures(whiteBox, 0, 310, 290, 550)

# save images
cv.imwrite('images/image1.jpg', image1)
cv.imwrite('images/image2.jpg', image2)
cv.imwrite('images/image3.jpg', image3)

# classifying figures
splitSample(image1, imageCenters1)
splitSample(image2, imageCenters2, 2)
splitSample(image3, imageCenters3, 3)

# classifying circles
knnCircles(image1, imageCenters1)
knnCircles(image2, imageCenters2, 2)
knnCircles(image3, imageCenters3, 3)
