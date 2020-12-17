from PIL import Image
import string, os
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def B2fun(cartoonsetdir, eyecolor,cartoonsetdir_test, eyecolor_test):
    size1 = np.size(os.listdir(cartoonsetdir + "\img"))
    size2 = np.size(os.listdir(cartoonsetdir_test + "\img"))
    dsize = size1+size2
    dataset = np.zeros([dsize, 4000])
    for val in range(size1):
        img = Image.open(cartoonsetdir + "\img\\" + str(val) + ".png")
        img = img.resize((100, 100), Image.BILINEAR)
        img = np.array(img)[40:60, 25:75, :].reshape(1, -1)[0]
        dataset[val] = img
    for val in range(size2):
        img = Image.open(cartoonsetdir_test + "\img\\" + str(val) + ".png")
        img = img.resize((100, 100), Image.BILINEAR)
        img = np.array(img)[40:60, 25:75, :].reshape(1, -1)[0]
        dataset[val+size1] = img 
    pca = PCA(n_components = 100)
    pcad = pca.fit_transform(dataset)
    mms = preprocessing.MinMaxScaler()
    pcadmms = mms.fit_transform(pcad)
    
    x_train, x_val, y_train, y_val = train_test_split(pcadmms[:10000], eyecolor, test_size = 0.2, random_state = 0)
    
    clf = MLPClassifier(random_state = 1, max_iter = 500)
    clf.fit(x_train, y_train)
    #print(clf.n_layers_)
    #print(clf.n_iter_)
    #print(clf.loss_)
    #print(clf.out_activation_)
    
    r1 = clf.score(x_train, y_train)
    
    
    r2 = clf.score(x_val,y_val)
   
    predictest = clf.predict(pcadmms[10000:])
    r3 = accuracy_score(y_true=eyecolor_test, y_pred=predictest)
    
    return [r1, r2 ,r3]