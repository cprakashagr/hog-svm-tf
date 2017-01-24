import argparse
from os import listdir
from os.path import isfile, isdir, join

import cv2
import src.utils as utils


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

    negFiles = [f for f in listdir(args.neg) if isfile(join(args.neg, f))]
    posFiles = [f for f in listdir(args.pos) if isfile(join(args.pos, f))]

    permitAngles = args.angles if args.angles is not None else [-15, -5, 5, 15]
    permitScales = args.scales if args.scales is not None else [0.5, 0.6, 0.8, 0.9]
    global permitAngles

    # Rotation & Scale Variance
    i = 0
    for posFile in posFiles:
        img = cv2.imread(join(args.pos, posFile), 0)
        for permitAngle in permitAngles:
            newImg = utils.rotate(img, permitAngle, color=(255, 255, 255))
            cv2.imwrite(join(args.pos, 'rot' + str(permitAngle) + posFile), newImg)

        for permitScale in permitScales:
            (h, w) = img.shape[:2]
            h = int(h*permitScale)
            newImg = utils.resize(img, h*2, h)
            cv2.imwrite(join(args.pos, 'scale' + str(permitScale) + posFile), newImg)

    i = 0
    for negFile in negFiles:
        img = cv2.imread(join(args.neg, negFile), 0)
        for permitAngle in permitAngles:
            newImg = utils.rotate(img, permitAngle, color=(255, 255, 255))
            cv2.imwrite(join(args.neg, 'rot' + str(permitAngle) + negFile), newImg)

        for permitScale in permitScales:
            (h, w) = img.shape[:2]
            newImg = utils.resize(img, int(w*permitScale), int(h*permitScale))
            cv2.imwrite(join(args.neg, 'scale' + str(permitScale) + negFile), newImg)


if __name__:
    main()
