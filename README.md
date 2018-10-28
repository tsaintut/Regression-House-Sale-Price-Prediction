# Regression-House-Sale-Price-Prediction
1.宣告和定義
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *

2.讀檔（x為輸入，y為輸出）
data_1 = pd.read_csv('/Users/jn6737/Desktop/Regression_data/train-v3.csv')
X_train = data_1.drop(['price','id'],axis=1).values
Y_train = data_1['price'].values

3.確認model是否被有被訓練到
data_2 = pd.read_csv('/Users/jn6737/Desktop/Regression_data/valid-v3.csv')
X_valid = data_2.drop(['price','id'],axis=1).values
Y_valid = data_2['price'].values

4.將老師給的資料丟進model以求出price
data_3 = pd.read_csv('/Users/jn6737/Desktop/Regression_data/test-v3.csv')
X_test = data_3.drop(['id'],axis=1).values

5.正規化
X_train = scale(X_train)
X_valid = scale(X_valid)
X_test = scale(X_test)

6.讓data訓練出來的值乘上權重，形成真正的檔案
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(128, input_dim=32, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(input_dim=X_train.shape[1], input_dim=128, kernel_initializer='normal', activation='linear'))
model.compile(Dloss='MAE',optimizer='adam')

7.設定跑的數量和一次丟幾筆資料
nb_epoch = 500
batch_size = 32

8.存擋
file_name=str(nb_epoch)+'_'+str(btch_size)
TB=TensorBoard(log_dir='logs/'+file_name, histogram_freq=0)
model.fit(X_train, Y_train, batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, Y_valid))
model.save('h5/'+file_name+.'h5')

9.x丟進模型求出y以得到答案，然後存至csv檔
Y_predict = model.predict(X_test)
np.savetext('test.csv',Y_predict,delimitet = ',')
