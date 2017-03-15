import argparse
from os import listdir
from os.path import isfile, isdir, join

import cv2
import numpy as np
import tensorflow as tf


class HOGGenerator:
  def __init__(self, winSize=(36, 256), blockSize=(8, 16), blockStride=(4, 8), cellSize=(4, 8), nBins=9,
               derivAperture=1, winSigma=4., histogramNormType=0, L2HysThreshold=2.0000000000000001e-01,
               gammaCorrection=0, nLevels=64):
    self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins, derivAperture,
                                 winSigma,histogramNormType, L2HysThreshold, gammaCorrection, nLevels)


def initializeCV2():
  return HOGGenerator(winSize=(256, 128), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8))


class Features:
  def __init__(self):
    self.__hog = None

  def featureExtract(self, input, inputFile):
    self.__hog = initializeCV2()
    # Extract the hog feature
    imageFeatures = self._processMatrix(input, inputFile)
    imageFeatures = np.squeeze(imageFeatures)
    imageFeatures = np.expand_dims(imageFeatures, axis=0)
    return imageFeatures

  def _processMatrix(self, path, file):
    # Read the image from the file path
    img = cv2.imread(join(path, file), 0)
    # Extract the hog feature
    desc = self.__hog.hog.compute(img)
    return desc

class SVM:
  def __init__(self):
    self.__trainData = None
    self.__trainFeatures = 16740

  def predict(self, input):
    x = tf.placeholder("float", shape=[None, self.__trainFeatures])

    W = tf.Variable(tf.zeros([self.__trainFeatures, 1]), name='weights')
    b = tf.Variable(tf.zeros([1]), name='biases')
    y_raw = tf.matmul(x, W) + b
    # Get the predict result
    predict = tf.sign(y_raw)

    # Add ops to restore all the variable
    saver = tf.train.Saver()

    with tf.Session() as sess:
      # Restore variable from disk
      saver.restore(sess, './hogNsvmModel')
      print('Restore from model')
      # Get all the file path
      inputFiles = [f for f in listdir(input) if isfile(join(input, f))]
      # Extrac the image feature
      fe = Features()
      for inputFile in inputFiles:
        # Extract the single image feature
        feature = fe.featureExtract(input, inputFile)
        # Run the predict ops to get the svm prediction result on this image
        result = sess.run([predict], feed_dict={x: feature})
        if result != -1:
          print("filename: %s | result: %s " % (inputFile, result))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="Images Location", type=str)
  args = parser.parse_args()

  if args.input is None:
    parser.print_usage()
    exit()

  if not isdir(args.input):
    parser.print_help()
    exit()

  # Predict the image
  svm = SVM()
  svm.predict(args.input)


if __name__ == '__main__':
  main()
