import numpy as np
import os


cartoonsetdir = "Datasets\cartoon_set"
faceshape = np.loadtxt(cartoonsetdir + "\labels.csv", skiprows = 1, usecols = (2))
eyecolor = np.loadtxt(cartoonsetdir + "\labels.csv", skiprows = 1, usecols = (1))

cartoonsetdir_test = "Datasets\cartoon_set_test"
faceshape_test = np.loadtxt(cartoonsetdir_test + "\labels.csv", skiprows = 1, usecols = (2))
eyecolor_test = np.loadtxt(cartoonsetdir_test + "\labels.csv", skiprows = 1, usecols = (1))

from A1.A1_model import A1fun
print(A1fun())

from A2.A2_model import A2fun
print(A2fun())

from B1.pB1 import B1fun
print(B1fun(cartoonsetdir, faceshape,cartoonsetdir_test,faceshape_test))

from B2.pB2 import B2fun
print(B2fun(cartoonsetdir, eyecolor,cartoonsetdir_test,eyecolor_test))

