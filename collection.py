import os, math, cv2, dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#======================================================================
# Feature Extraction Functions
#======================================================================


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(testset):
    """
    This funtion extracts the landmarks features for all images in a specific folder .
    It also extract the gender label and smiling label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:    an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
        smiling_labels:    an array containing the smiling label (not smiling=0 and smiling=1) for each image in
                            which a face was detected
    """
    
    
    if testset == False:
        basedir = './Datasets//celeba'
    else:
        basedir = './Datasets//celeba_test'
    images_dir = os.path.join(basedir,'img')
    labels_filename = 'labels.csv'
    
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    
    
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[1] : int(line.split('\t')[2]) for line in lines[1:]}
    smiling_labels = {line.split('\t')[1] : int(line.split('\t')[3]) for line in lines[1:]}
    
    if os.path.isdir(images_dir):
        all_features = []
        all_gender_labels = []
        all_smiling_labels = []
        for img_path in image_paths:
            file_name= img_path.split('.')[1].split('\\')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_gender_labels.append(gender_labels[file_name+'.jpg'])
                all_smiling_labels.append(smiling_labels[file_name+'.jpg'])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_gender_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    smiling_labels = (np.array(all_smiling_labels) + 1)/2 # simply converts the -1 into 0, so not smiling=0 and smiling=1
    
    return landmark_features, gender_labels, smiling_labels

#======================================================================
# Data Pre-processing Functions
#======================================================================
def train_preprocessing(X, y, split_percentage):
    
    Y = np.array([y, -(y - 1)]).T
    split = int(len(X) * (split_percentage/100))
    X_all = X
    y_all = Y
    X_train = X[:split]
    y_train = Y[:split]
    X_test = X[split:]
    y_test = Y[split:]
    
    
    X_all = X_all.reshape(len(X_all), 68*2)
    X_train = X_train.reshape(len(X_train), 68*2)
    X_test = X_test.reshape(len(X_test), 68*2)
    
    y_all = list(zip(*y_all))[0]
    y_train = list(zip(*y_train))[0]
    y_test = list(zip(*y_test))[0]
    
    return X_all, X_train, X_test, y_all, y_train, y_test

def test_preprocessing(X, y):
    
    Y = np.array([y, -(y - 1)]).T
    X_all = X
    y_all = Y
    
    X_all = X_all.reshape(len(X_all), 68*2)
 
    y_all = list(zip(*y_all))[0]
    
    return X_all, y_all







#======================================================================
# ML Model Functions
#======================================================================
def model_task_A1(X_total,X_tr, X_te, y_total,y_tr, y_te):
    
    clf = SVC(kernel='poly', C=10)
    clf.fit(X_tr, y_tr)
    acc_train = clf.score(X_tr, y_tr)
    
    
    acc_val = cross_val_score(clf, X_total, y_total, cv=10).mean()
   
    
    acc_test = clf.score(X_te, y_te)
    
    return [acc_train, acc_val, acc_test]

#======================================================================
def model_task_A2(X_total,X_tr, X_te, y_total,y_tr, y_te):
    
    clf = SVC(kernel='poly', C=1)
    clf.fit(X_tr, y_tr)
    acc_train = clf.score(X_tr, y_tr)
    
    
    acc_val = cross_val_score(clf, X_total, y_total, cv=10).mean()
   
    
    acc_test = clf.score(X_te, y_te)
    
    return [acc_train, acc_val, acc_test]
#======================================================================


#======================================================================
# Grid Search Functions for Hyperparameter Tuning
#======================================================================
def build_svm_gridcv(X_train, X_test, y_train, y_test):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'],
                         'C': [0.01, 0.5, 0.1, 1, 10],
                         'gamma': [0.01, 0.1, 1, 10]},
                        {'kernel': ['linear'], 'C': [0.01, 0.5, 0.1, 1, 10]},
                        {'kernel': ['poly'],
                         'C': [0.01, 0.5, 0.1, 1, 10]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, cv = 10, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on training dataset:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on training dataset:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full training dataset.")
        print("The scores are computed on the full testing dataset.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    

    print('Best estimator found:', clf.best_estimator_)
    print('Best parameters set found:', clf.best_params_)
    print()
    
    acc_score_test = accuracy_score(y_test, y_pred)
    print('SVM with GridCV on testing data - Accuracy Score: %.3f (+/- %.3f)' % (acc_score_test.mean(), acc_score_test.std()))
    acc_score_train = clf.score(X_train, y_train)
    print('SVM with GridCV on training data - Accuracy Score: %.3f (+/- %.3f)' % (acc_score_train.mean(), acc_score_train.std()))
    print() 
    
    return clf, acc_score_train, acc_score_test
