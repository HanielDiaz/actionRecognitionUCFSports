import cv2
import math
import os
import tensorflow as tf
import numpy as np

actions = [
    "Diving", "Golf-Swing", "Kicking", "Lifting", "Riding-Horse", "Running",
    "SkateBoarding", "Swing-Bench", "Swing-SideAngle", "Walking"
]


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]

def resize(img):
    return np.resize(img, (70, 70, 3))

# If video gets frames as jpgs
def splitVid(videoPath):
    cap = cv2.VideoCapture(videoPath)
    images = []
    count = 0
    while (cap.isOpened()):
        frameID = cap.get(1)
        ret, frame = cap.read()
        if (ret != True):
            break
        frame = resize(frame)
        images.append(frame)
            # print(len(images))
        if(len(images) == 20):
            # print("Processing %s" % videoPath)
            return images
        count += 1
    cap.release()
    # print("Video had %d frames." % count)
    # print("Didn't process %s" % videoPath, "Length: %d" %len(images))
    return []

def findShape(arr):
    for x in arr:
        print(x.shape)

def searchForVids(vidDir):
    videos = []
    for i in range(10):
        videos.append([])
    for root, dirs, files in os.walk("./ucfaction"):
        for file in files:
            if file.endswith(".avi"):
                path = root.split("/")
                videos[actions.index(path[2])].append(root + "/" +  str(file))
                # if(len(videos[path[2]]) == 3):
                #     return videos
    return videos

def splitData(data):
    trainingSet = []
    testingSet = []
    for action in range(10):
        cutOff = len(data[action]) - 1
        trainingSet.append(data[action][:cutOff])
        testingSet.append(data[action][cutOff:])
    return (trainingSet, testingSet)