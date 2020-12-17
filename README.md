# Applied Machine Learning System ELEC0134(20/21) Assignment

These codes were written under Window10 system. If you would like to run it under Mac OS system, you may need to change a little bit codes. 
For instance, in collection.py, change the '\\' to '\' in function extract_features_labels.

main.py:
Users run main.py to obain the train accuracy, validation accuracy and test accuracy for task A1, task A2, task B1 and task B2 respectively.

collection.py:
This file includes many irreplaceable functions to extract face landmarks, preprocess the data, build machine learning models for task A1 and task A2, grid research for task A1 and task A2.

A1_model.py solves the problem of Gender detection: male or female
A2_model.py solves the problem of Emotion detection: smiling or not smiling
pB1.py solves the problem of Face shape recognition: 5 types of face shapes
pB2.py solves the problem of Eye color recognition: 5 types of eye colors
The Datasets file is responsible for providing data to the program
Data packages are required to run the code:
- NumPy
- pandas
- dlib
- string
- os
- PIL
- random
- sklearn
