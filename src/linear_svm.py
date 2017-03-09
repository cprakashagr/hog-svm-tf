import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string('train', None, 'path to file having the training data')
tf.app.flags.DEFINE_integer('ne', 1, 'total number of epochs')
tf.app.flags.DEFINE_bool('verbose', True, 'enable verbose logging')
tf.app.flags.DEFINE_integer('svmC', 10, 'svm c parameter for itd cost function')


class SVM:
    def __init__(self, flags):
        self.__flags = flags
        self.__trainData = None
        self.__trainLabels = None
        self.__trainFeatures = None
        self.__trainSize = None

    def trainData(self):
        self.__extractData()

        verbose = self.__flags.verbose
        BATCH_SIZE = 1

        x = tf.placeholder("float", shape=[None, self.__trainFeatures])
        y = tf.placeholder("float", shape=[None, 1])

        W = tf.Variable(tf.zeros([self.__trainFeatures, 1]), name='weights')
        b = tf.Variable(tf.zeros([1]), name='biases')
        y_raw = tf.matmul(x, W) + b

        regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - y * y_raw))

        svm_loss = regularization_loss + self.__flags.svmC * hinge_loss
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

        predicted_class = tf.sign(y_raw)
        correct_prediction = tf.equal(y, predicted_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.Session() as s:
            tf.initialize_all_variables().run()
            if verbose:
                print('Initialized!')
                print('Training.')

            for step in iter(range(self.__flags.ne * self.__trainSize // BATCH_SIZE)):
                if verbose:
                    print(step)

                offset = (step * BATCH_SIZE) % self.__trainSize
                batch_data = self.__trainData[offset:(offset + BATCH_SIZE), :]
                batch_labels = self.__trainLabels[offset:(offset + BATCH_SIZE)]
                train_step.run(feed_dict={x: batch_data, y: batch_labels})
                print('loss: ', svm_loss.eval(feed_dict={x: batch_data, y: batch_labels}))

                if verbose and offset >= self.__trainSize - BATCH_SIZE:
                    pass

            if verbose:
                print('Weight matrix.')
                print(s.run(W))
                print('Bias vector.')
                print(s.run(b))
                print("Applying model to first test instance.")

            print("Accuracy on train:", accuracy.eval(
                feed_dict={x: self.__trainData, y: self.__trainLabels}))

            saver = tf.train.Saver()
            saver.save(s, 'hogNsvmModel')

    def __extractData(self):
        fileData = np.loadtxt(fname=self.__flags.train, delimiter=',')
        totalN, totalF = fileData.shape
        self.__trainLabels = fileData[:, totalF-1]
        self.__trainLabels = self.__trainLabels.reshape(self.__trainLabels.size, 1)
        self.__trainData = fileData[:, :totalF-1]

        self.__trainSize, self.__trainFeatures = self.__trainData.shape


def main(argv=None):
    svmModel = SVM(flags=tf.app.flags.FLAGS)
    svmModel.trainData()


if __name__ == "__main__":
    tf.app.run()
