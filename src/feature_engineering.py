import argparse
from os import listdir
from os.path import isfile, isdir, join

import cv2
import numpy as np


global allDesc


class HOGGenerator:
    def __init__(self, winSize=(36, 256), blockSize=(8, 16), blockStride=(4, 8), cellSize=(4, 8), nBins=9,
                 derivAperture=1, winSigma=4., histogramNormType=0, L2HysThreshold=2.0000000000000001e-01,
                 gammaCorrection=0, nLevels=64):

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins, derivAperture,
                                     winSigma,histogramNormType, L2HysThreshold, gammaCorrection, nLevels)


def initializeCV2():
    return HOGGenerator(winSize=(256, 128), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8))


def processMatrix(hog, path, file, i, claz):
    img = cv2.imread(join(path, file), 0)
    desc = hog.hog.compute(img)
    desc = np.append(desc, claz)
    allDesc[i] = desc
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

    negFiles = [f for f in listdir(args.neg) if isfile(join(args.neg, f))]
    posFiles = [f for f in listdir(args.pos) if isfile(join(args.pos, f))]

    hog = initializeCV2()
    allDesc = np.empty([len(negFiles) + len(posFiles), 16741])

    i = 0

    for posFile in posFiles:
        processMatrix(hog, args.pos, posFile, i, 1)
        i += 1

    for negFile in negFiles:
        processMatrix(hog, args.neg, negFile, i, 0)
        i += 1

    np.savetxt("trainData.csv", allDesc, delimiter=',' ,newline='\n')

    pass

if __name__ == '__main__':
    main()
