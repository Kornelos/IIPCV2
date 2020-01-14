import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import mahotas as mt
from mahotas.features import haralick
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage import feature
from pylab import rcParams


# for image classification use svm or naive bayes
# other lib than openv can be used for texture characterisation
# TODO: IDEAS FOR features DAISY AND SIFT keyword bag-of-features
# https://www.mathworks.com/help/vision/ug/local-feature-detection-and-extraction.html


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return [rho, phi]


def extract_contour_features(image):
    """
    https://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html
    :param image: greyscale
    :return: feature vector
    """
    # extracting leaf contour
    ret, th = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(cv.bitwise_not(th), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = float(w) / h
    area = cv.contourArea(cnt)
    rect_area = w * h
    # ratio of contour area to bounding rectangle area.
    extent = float(area) / rect_area

    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    # ratio of contour area to its convex hull area.
    solidity = float(area) / hull_area
    # diameter of the circle whose area is same as the contour area.
    equi_diameter = np.sqrt(4 * area / np.pi)
    (x, y), (MA, ma), angle = cv.fitEllipse(cnt)

    mask = np.zeros(image.shape, np.uint8)
    cv.drawContours(mask, [cnt], 0, 255, -1)
    # Contour Perimeter
    perimeter = cv.arcLength(cnt, True)
    # average intensity of the object in grayscale
    mean_val = cv.mean(image, mask=mask)
    return [aspect_ratio, extent, solidity, equi_diameter, perimeter, hull_area, mean_val[0]]


# Mahotas - Harlick texture feature vector
def extract_texture_features(image, mode='default'):
    """
    Function extracting texture features
    https://mahotas.readthedocs.io/en/latest/api.html?highlight=haralick#mahotas.features.haralick
    :param image: greyscale
    :param mode: thresh - threshold is on, default - threshold is off
    :return: feature vector
    """
    if mode == 'thresh':
        # calculate haralick texture features for 4 types of adjacency
        ret, th = cv.threshold(gray, 0, 255, cv.THRESH_TOZERO_INV + cv.THRESH_OTSU)
        textures = haralick(th, ignore_zeros=True)
    else:
        textures = haralick(gray)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


def extract_lbp_feature(image, numPoints, radius, eps=1e-7):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist


def prepare_data():
    plants = ['circinatum', 'garryana', 'glabrum', 'kelloggii', 'macrophyllum', 'negundo']
    path = []
    for p in plants:
        path.append(glob.glob('isolated/' + p + '/*'))
    return path


if __name__ == '__main__':
    images = prepare_data()
    img = cv.imread('./isolated/circinatum/l11.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    extract_contour_features(gray)
    ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plt.imshow(th, cmap='gray')
    # plt.title('otsu')
    # plt.show()

    contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)
    # plt.imshow(img)
    # plt.show()

    # --------------------------DATA PROCESSING
    train_features = []
    train_labels = []
    i = 0
    for path in images:
        for file in path:
            image = cv.imread(file)
            # convert the image to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # extract haralick texture from the image
            features = extract_texture_features(gray)
            # features2 = extract_contour_features(gray)
            features3 = extract_lbp_feature(gray, 24, 8)
            # append the feature vector and label np.append
            train_features.append(np.append(features, features3))
            train_labels.append(i)
        i += 1

    print("Training features: {}".format(np.array(train_features).shape))
    print("Training labels: {}".format(np.array(train_labels).shape))

    # create the classifier
    print("[STATUS] Creating the classifier..")
    clf_svm = LinearSVC(max_iter=9000)

    # fit the training data and labels
    print("[STATUS] Fitting data/label to model..")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    from sklearn.model_selection import train_test_split

    xTrain, xTest, yTrain, yTest = train_test_split(train_features, train_labels, test_size=0.2, random_state=0)
    # train linear svc
    clf_svm.fit(xTrain, yTrain)

    from sklearn.model_selection import KFold, cross_val_score

    k_fold = KFold(n_splits=10)
    # for train_indices, test_indices in k_fold.split(xTrain):
    #     print('Train: %s | test: %s' % (train_indices, test_indices))
    # [clf_svm.fit(train_features[train], train_labels[train]).score(train_features[test], train_labels[test])
    #  for train, test in k_fold.split(train_features)]
    # res = cross_val_score(clf_svm, train_features, train_labels, n_jobs=-1)
    # print(res)
    # score = clf_svm.score(xTrain, yTrain)
    # print('[TRAIN] Accuracy of the model is equal: {}'.format(score))
    score = clf_svm.score(xTest, yTest)
    print('[TEST] Accuracy of the model is equal: {}'.format(score))

    prediction = clf_svm.predict(xTest[1].reshape(1, -1))
    print('Value predicted {} correct value - {}'.format(prediction, yTest[1]))
