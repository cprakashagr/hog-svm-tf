import argparse
from os import listdir
from os.path import isfile, isdir, join

import cv2
import numpy as np
import utils


class SampleCreate:
    def __init__(self):
        self.__emptyImg = None
        self.__args = None
        self.__permitAngles = None
        self.__permitScales = None
        self.__pos = None
        self.__neg = None

    def setValues(self, pos, neg, permitAngles=[-15, -5, 5, 15], permitScales=[0.5, 0.6, 0.8, 0.9]):
        self.__pos = pos
        self.__neg = neg
        self.__permitAngles = permitAngles
        self.__permitScales = permitScales

    def readFilesAndCreateSamples(self):
        negFiles = [f for f in listdir(self.__neg) if isfile(join(self.__neg, f))]
        posFiles = [f for f in listdir(self.__pos) if isfile(join(self.__pos, f))]

        for posFile in posFiles:
            self.processMatrix(self.__pos, posFile, self.__permitAngles, self.__permitScales)

        for negFile in negFiles:
            self.processMatrix(self.__neg, negFile, self.__permitAngles, self.__permitScales)

    def __createEmptyImg(self, img):
        self.__emptyImg = np.zeros(img.shape[:2])
        self.__emptyImg.fill(255)

    def processMatrix(self, path, file, permitAngles, permitScales):
        img = cv2.imread(join(path, file), 0)

        if self.__emptyImg is None:
            self.__createEmptyImg(img)

        for permitAngle in permitAngles:
            newImg = utils.rotate(img, permitAngle, color=(255, 255, 255))
            cv2.imwrite(join(path, 'rot' + str(permitAngle) + file), newImg)

        for permitScale in permitScales:
            (h, w) = img.shape[:2]
            h = int(h * permitScale)
            newImg = utils.resize(img, h * 2, h)
            embedImg = self.__emptyImg.copy()
            embedImg[:newImg.shape[0], :newImg.shape[1]] = newImg
            cv2.imwrite(join(path, 'scale' + str(permitScale) + file), embedImg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pos", help="Positive Images Location", type=str)
    parser.add_argument("-n", "--neg", help="Negative Images Location", type=str)
    parser.add_argument("-a", "--angles", help="Permitted Angles", type=str)
    parser.add_argument("-s", "--scales", help="Permitted Scales", type=str)
    args = parser.parse_args()

    if args.pos is None or args.neg is None:
        parser.print_usage()
        exit()

    if not isdir(args.pos) or not isdir(args.neg):
        parser.print_help()
        exit()

    sc = SampleCreate()
    sc.setValues(args.pos, args.neg)
    sc.readFilesAndCreateSamples()

    # Rotation & Scale Variance

if __name__:
    main()
