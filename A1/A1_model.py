import os, math, cv2, dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split

from collection import *

# Task A1 - Gender Recognition
def A1fun():
    
    landmark_features_celeba, gender_labels, smiling_labels = extract_features_labels(testset=False)
    landmark_features_celeba_test, gender_labels_test, smiling_labels_test = extract_features_labels(testset=True)

    split_percentage = 80 # 80% training data - 20% testing data split
    X_all, X_train, X_test, y_all, y_train, y_test = train_preprocessing(landmark_features_celeba, gender_labels, split_percentage)
    X_test_new, y_test_new = test_preprocessing(landmark_features_celeba_test, gender_labels_test)
    
    # hyperparameter tuning by means of exhaustive grid search
    # model_A1, acc_A1_train, acc_A1_test = build_svm_gridcv(X_train, X_test, y_train, y_test)
    
    ''' 
     Plot Learning Curve
     def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
   
        plt.figure(figsize=(8, 6))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

        plt.legend(loc="best")
        plt.savefig(title+'.png')

        return plt
     estimator_A1 = SVC(kernel='poly',C=10)
     plot_learning_curve(estimator_A1, 'Task A1 Learning Curve', X_train, y_train, ylim=None, cv=10,
                    n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
                    '''

    
    return model_task_A1(X_all,X_train, X_test, y_all,y_train, y_test)