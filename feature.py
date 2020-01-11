import cv2 as cv
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mt
from os import listdir
from os.path import isfile, join
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for image classification use svm or naive bayes
# other lib than openv can be used for texture characterisation

# def extract_contur_features(image):





# Mahotas - Harlick texture feature vector
def extract_features(image):
    """
    Function extracting texture features
    input - gray image
    output - feature vector
    """
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


def prepare_data():
    plants = ['circinatum', 'garryana', 'glabrum', 'kelloggii', 'macrophyllum', 'negundo']
    path = []
    for p in plants:
        path.append(glob.glob('isolated/' + p + '/*'))
    return path


if __name__ == '__main__':
    images = prepare_data()
    img = cv.imread('./isolated/circinatum/l09.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plt.imshow(th, cmap='gray')
    # plt.title('otsu')
    # plt.show()

    contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)
    # plt.imshow(img)
    # plt.show()

    train_features = []
    train_labels = []
    i = 0
    for path in images:
        for file in path:
            image = cv.imread(file)
            # convert the image to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # extract haralick texture from the image
            features = extract_features(gray)
            # append the feature vector and label
            train_features.append(features)
            train_labels.append(i)
        i += 1

    print("Training features: {}".format(np.array(train_features).shape))
    print("Training labels: {}".format(np.array(train_labels).shape))

    # # Train the SVM
    # svm = cv.ml.SVM_create()
    # svm.setType(cv.ml.SVM_C_SVC)
    # svm.setKernel(cv.ml.SVM_LINEAR)
    # svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    # create the classifier
    print("[STATUS] Creating the classifier..")
    clf_svm = LinearSVC()

    # fit the training data and labels
    print("[STATUS] Fitting data/label to model..")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest = train_test_split(train_features, train_labels, test_size=0.2, random_state=0)

    clf_svm.fit(xTrain, yTrain)

    score = clf_svm.score(xTest, yTest)
    print('accuracy of the model is equal: {}'.format(score))

    prediction = clf_svm.predict(xTest[1].reshape(1, -1))
    print('Value predicted {} correct value - {}'.format(prediction, yTest[1]))