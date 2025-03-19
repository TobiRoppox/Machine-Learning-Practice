import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G1", "G1", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"

x = np.array(data.drop([predict, 1]))
y = np.array(data[predict ])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)