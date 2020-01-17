import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mahotas.features import haralick
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage import feature
from sklearn.model_selection import train_test_split, cross_val_score
from itertools import combinations
from sys import argv

plants = ['circinatum', 'garryana', 'glabrum', 'kelloggii', 'macrophyllum', 'negundo']


def extract_contour_features(image):
    """
    https://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html
    :param image: grayscale
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
    return np.array([aspect_ratio, extent, solidity, equi_diameter, perimeter, hull_area, mean_val[0]])


def extract_haralick_features(image, mode='default'):
    """
    Function extracting texture features
    https://mahotas.readthedocs.io/en/latest/api.html?highlight=haralick#mahotas.features.haralick
    :param image: grayscale
    :param mode: thresh - threshold is on, default - threshold is off
    :return: feature vector
    """
    if mode == 'thresh':
        # calculate haralick texture features for 4 types of adjacency
        ret, th = cv.threshold(image, 0, 255, cv.THRESH_TOZERO_INV + cv.THRESH_OTSU)
        textures = haralick(th, ignore_zeros=True)
    else:
        textures = haralick(image)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


def extract_lbp_feature(image, num_points=24, radius=8, eps=1e-7):
    """
    compute the Local Binary Pattern representation
    of the image, and then use the LBP representation
    to build the histogram of patterns
    :param image: grayscale
    :param num_points: Number of points in a circularly symmetric neighborhood to consider.
    :param radius: The radius of the circle, which allows us to account for different scales.
    :param eps: Number used for normaliztaion
    :return:feature vector
    """
    lbp = feature.local_binary_pattern(image, num_points,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist


def kaze_features(image, vector_size=32):
    """
    Compute feature vector based on KAZE features
    https://docs.opencv.org/3.4/d3/d61/classcv_1_1KAZE.html
    :param image: grayscale
    :param vector_size: return vector size
    :return: feature vector
    """
    alg = cv.KAZE_create()
    # Finding image keypoints
    kps = alg.detect(image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)
    dsc = dsc.mean(axis=0)
    return dsc


def hist_features(image, eps=1e-7):
    """
    Compute feature vector based on image histogram
    :param image: grayscale
    :param eps: Number used for normaliztaion
    :return: feature vector
    """
    histogram = cv.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + eps)
    # return the histogram of Local Binary Patterns
    return histogram.flatten()


def prepare_data():
    paths = []
    gray_images = []
    class_labels = []
    for p in plants:
        paths.append(glob.glob('isolated/' + p + '/*'))

    i = 0
    for path in paths:
        for file in path:
            image = cv.imread(file)
            # convert the image to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gray_images.append(gray)
            class_labels.append(i)
        i += 1

    return gray_images, class_labels


def process_data(gray_images, alg):
    fv = []
    for img in gray_images:
        fv.append(alg(img))
    return fv


if __name__ == '__main__':
    images, train_labels = prepare_data()

    cont = process_data(images, extract_contour_features)
    hist = process_data(images, hist_features)
    lbp = process_data(images, extract_lbp_feature)
    kaze = process_data(images, kaze_features)
    haralick = process_data(images, extract_haralick_features)

    feature_vectors = [cont, hist, ]
    feature_names = ['contour', 'histogram', 'LBP', 'KAZE', 'Haralick']
    # here we can define what combinations we want to test i.e. all pairs
    if len(argv) > 1:
        n = int(argv[1])
    else:
        n = 1

    fvs = combinations(feature_vectors, n)
    names = combinations(feature_names, n)

    for t, name in zip(fvs, names):
        train_features = np.concatenate(t, axis=1)
        # create the classifier
        # print("[STATUS] Creating the classifier..")
        clf_svm = LinearSVC(max_iter=9000)

        # fit the training data and labels
        # print("[STATUS] Fitting data/label to model..")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        score = cross_val_score(clf_svm, train_features, train_labels, cv=10, n_jobs=-1)
        print(name)
        print('10 fold cross validation scores:')
        print(score)
        print('Mean: {}'.format(np.mean(score)))
        print('Median: {}'.format(np.median(score)))
        print('Std: {}'.format(np.std(score)))

        xTrain, xTest, yTrain, yTest = train_test_split(train_features, train_labels, test_size=0.2, random_state=0)
        # train linear svc
        clf_svm.fit(xTrain, yTrain)
        score = clf_svm.score(xTest, yTest)
        print('[TEST] 80/20 Accuracy of the model is equal: {}'.format(score))
        from sklearn.metrics import classification_report

        test_pred = clf_svm.predict(xTest)
        print(classification_report(test_pred, yTest, target_names=plants))
