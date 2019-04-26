import sys, getopt
import matplotlib.pyplot as plt
import utils
import tensorflow as tf
import numpy as np
import c3d_tf_keras
import time

actions = ["Diving", "Golf-Swing", "Kicking", "Lifting", "Riding-Horse", "Running", "SkateBoarding", "Swing-Bench", "Swing-SideAngle", "Walking"]

def algorithm(trainingSet, testingSet):
    # print(len(trainingSet),len(testingSet))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for action in range(10):
        for vid in trainingSet[action]:
            y_train.append(action)
            x_train.append(vid)
    for action in range(10):
        for vid in testingSet[action]:
            y_test.append(action)
            x_test.append(vid)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # print(trainingSet)
    # print(x_train.shape)
    # utils.findShape(x_train)
    # c3d_tf_nn.model(x_train)
    c3d_tf_keras.train(x_train, y_train, x_test, y_test)
    print("\nDone with algorithm!")
    return 0

def main():
    unixOptions = "fv:m:"
    gnuOptions = ["first", "video=", "model="]
    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]
    allImages = []
    print(argumentList)
    try:
        start = time.time()
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-v", "--video"):
                images = utils.splitVid(currentValue)
            elif currentArgument in ("-p", "--pictures"):
                print("Taking in pictures")
            elif currentArgument in ("-f", "--first"):
                print("Processing data...")
                allVids = utils.searchForVids("./ucfaction")
                for action in allVids:
                    allImages.append([])
                    for vid in action:
                        images = utils.splitVid(vid)
                        # print(images)
                        path = vid.split("/")
                        # print(path)
                        if(len(images) > 5):
                            allImages[actions.index(path[2])].append(images)
                print("Done!")
                print("Splitting Data into training and testing")
                trainingSet, testingSet = utils.splitData(allImages)
                print("Done!")
                print("Running algorithm...")
                algorithm(trainingSet, testingSet)
                print("Done")
            elif currentArgument in ("-r","--read"):
                print("Reading Model %s" % currentValue )
        end = time.time()
        print("Time to run: %d" % (end - start))
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    return 0


main()
