# **hog-svm-tf**
[Histogram of oriented gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a feature descriptor used in image processing and 
has recently got a lot of attention for the purpose of object detection.

HOG with [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) has given very good results. Their implementation has been provided in the OpenCV library.
However, the library uses its own SVM implementation in C++.

`hog-svm-tf` is an approach to use `tensorflow` for SVM training while we continue to use HOG with OpenCV.

The project has 3 modules (for now):
1. `sample_create` - For generating a lot of sample data from the given set of input. Based on rotation and scaling,
this will generate more sample data for training.
2. `feature_engineering` - It generates the hog descriptors and save the data to be used for training.
3. `linear_svm` - It is a generic linear SVM implementation. It uses the training data generated in last step
and creates a tensorflow model. After creating a tensorflow model, save the model.
4. `eval_image.py` - It evalutaes a folder of image according to the tensorflow checkpoint.

**Usage**
-
```commandline
sample_create.py
> python3 sample_create.py -p /Users/cprakashagr/Pictures/MCQSheet/128x256/pos -n /Users/cprakashagr/Pictures/MCQSheet/128x256/neg

feature_engineering.py
> python3 feature_engineering.py -p /Users/cprakashagr/Pictures/MCQSheet/128x256/pos -n /Users/cprakashagr/Pictures/MCQSheet/128x256/neg

linear_svm.py
> python3 linear_svm.py --train trainData.csv --svmC 1 --verbose True --num_epochs 10

eval_image.py
> python3 eval_image.py -i /path/to/imagefoler
``` 
**TODO**

1. More samples creation based on brightness and contrast alteration
2. Support for Keras and OpenCL
3. Visualisation module for the tensorflow model 

**Contribution**
-
1. [Frankgu](http://github.gdf.name)
  - add the evaluation program to the project
