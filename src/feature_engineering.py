import argparse
from os import listdir
from os.path import isfile, isdir, join

import cv2
import numpy as np


class HOGGenerator:
    def __init__(self, winSize=(36, 256), blockSize=(8, 16), blockStride=(4, 8), cellSize=(4, 8), nBins=9,
                 derivAperture=1, winSigma=4., histogramNormType=0, L2HysThreshold=2.0000000000000001e-01,
                 gammaCorrection=0, nLevels=64):

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins, derivAperture,
                                     winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nLevels)


def initializeCV2():
    return HOGGenerator(winSize=(256, 128), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8))


class FeatureEngineering:
    def __init__(self):
        self.__pos = None
        self.__neg = None
        self.__allDesc = None
        self.__negFiles = None
        self.__posFiles = None
        self.__hog = None

    def setValues(self, pos, neg):
        self.__pos = pos
        self.__neg = neg

        self.__negFiles = [f for f in listdir(self.__neg) if isfile(join(self.__neg, f))]
        self.__posFiles = [f for f in listdir(self.__pos) if isfile(join(self.__pos, f))]

        self.__allDesc = np.empty([len(self.__negFiles) + len(self.__posFiles), 16741])

    def doFeatureEngineering(self, save=True):
        self.__hog = initializeCV2()
        i = 0
        for posFile in self.__posFiles:
            self.processMatrix(self.__pos, posFile, i, 1)
            i += 1

        for negFile in self.__negFiles:
            self.processMatrix(self.__neg, negFile, i, -1)
            i += 1

        if save:
            np.savetxt("trainData.csv", self.__allDesc, delimiter=',', newline='\n')

    def processMatrix(self, path, file, i, claz):
        img = cv2.imread(join(path, file), 0)
        desc = self.__hog.hog.compute(img)
        desc = np.append(desc, claz)
        self.__allDesc[i] = desc
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pos", help="Positive Images Location", type=str)
    parser.add_argument("-n", "--neg", help="Negative Images Location", type=str)
    args = parser.parse_args()

    if args.pos is None or args.neg is None:
        parser.print_usage()
        exit()

    if not isdir(args.pos) or not isdir(args.neg):
        parser.print_help()
        exit()

    fe = FeatureEngineering()
    fe.setValues(args.pos, args.neg)
    fe.doFeatureEngineering()


if __name__ == '__main__':
    main()
