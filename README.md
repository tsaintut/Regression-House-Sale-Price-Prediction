# Regression-House-Sale-Price-Prediction
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *

data_1 = pd.read_csv('/Users/jn6737/Desktop/Regression_data/train-v3.csv')
X_train = data_1.drop(['price','id'],axis=1).values
Y_train = data_1['price'].values

data_2 = pd.read_csv('/Users/jn6737/Desktop/Regression_data/valid-v3.csv')
X_valid = data_2.drop(['price','id'],axis=1).values
Y_valid = data_2['price'].values

data_3 = pd.read_csv('/Users/jn6737/Desktop/Regression_data/test-v3.csv')
X_test = data_3.drop(['id'],axis=1).values

X_train = scale(X_train)
X_valid = scale(X_valid)
X_test = scale(X_test)

def build_nn():
    model = Sequential()
    model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(100, input_dim=30, init='normal', activation='relu'))
    model.add(Dense(40, input_dim=100, init='normal', activation='linear'))
    model.add(Dense(1, init='normal'))
    
    predict = model.predict(X_test)
    np.savetext('test.csv',predict,delimitet = ',')
