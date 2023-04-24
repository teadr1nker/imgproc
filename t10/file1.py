#!/usr/bin/python3
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def drawFigures(pic, x1, x2, y1, y2):
    image = np.copy(pic)
    clusterx = []
    clustery = []
    for i in range(0, 100):
        clusterx.append([np.random.randint(0, 750),
                         np.random.randint(x1, x2)])
        clustery.append([np.random.randint(0, 750),
                         np.random.randint(y1 , y2)])

    imageCenters = []

    for p in clusterx:
        cv.rectangle(image, (p[0], p[1]), (p[0] + 6, p[1] + 6), (0, 0, 255),-1)
        imageCenters.append([p[0]+3, p[1]+3])

    for p in clustery:
        pts = np.array([[p[0], p[1]], [p[0] + 6, p[1]], [p[0] + 6, p[1] + 6]])
        cv.fillPoly(image, [pts], color=(255, 0, 0))
        imageCenters.append([p[0]+4, p[1]+2])

    return image, np.float32(imageCenters)

def splitSample(pic, centers, number = 1):
    image = np.copy(pic)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(centers, 2, None,
                                    criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    class1 = centers[label.ravel() == 0]
    class2 = centers[label.ravel() == 1]

    plt.scatter(class1[:, 0], class1[:, 1], c='b')
    plt.scatter(class2[:, 0], class2[:, 1], c='r')
    plt.scatter(center[:,0], center[:,1], s = 80, c='y', marker='s')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig(f'images/scatter{number}.jpg')
    plt.clf()

    center = np.int32(center)
    image = cv.line(image, tuple(center[0]), tuple(center[1]), (0, 255, 0), 3)
    cv.imwrite(f'images/line{number}.jpg', image)

    return None

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

whiteBox = np.zeros((600, 800, 3), np.uint8)
whiteBox.fill(255)

image1, imageCenters1 = drawFigures(whiteBox, 0, 250, 300, 550)
image2, imageCenters2 = drawFigures(whiteBox, 0, 290, 300, 550)
image3, imageCenters3 = drawFigures(whiteBox, 0, 310, 290, 550)

cv.imwrite('images/image1.jpg', image1)
cv.imwrite('images/image2.jpg', image2)
cv.imwrite('images/image3.jpg', image3)

splitSample(image1, imageCenters1)
splitSample(image2, imageCenters2, 2)
splitSample(image3, imageCenters3, 3)

knnCircles(image1, imageCenters1)
knnCircles(image2, imageCenters2, 2)
knnCircles(image3, imageCenters3, 3)
