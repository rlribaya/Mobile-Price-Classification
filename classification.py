import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd 
import numpy as np
data = pd.read_csv('train.csv')
# print(data.head())

#prints num of rows and cols
print(data.shape)
#prints data types
print(data.dtypes)
#looks into the ave, sd, quantiles, and other summary stats
print(data.describe())

x = data.iloc[:,:20].values

y = data.iloc[:,20:21].values

print('x values')
print(x)
print('y value')
print(y)

#preprocess data

#normalize data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)
print("normalized data \n", x)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('ohe\n', y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(Dense(4, activation='softmax')) # softmax function outputs probabilities

#last step is to compile entire thing
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# train the neural net
history = model.fit(X_train, y_train,validation_data = (X_test, y_test), epochs=100, batch_size=64)

#evlauating the neural net
y_pred = model.predict(X_test)
#converting prediction to label
pred = list()
for i in range(len(y_pred)):
	pred.append(np.argmax(y_pred[i]))

#converting ohe test label to label

test = list()
for i in range(len(y_test)):
	test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test)
print('Accuracy: ', acc*100)

# history = model.fit(X_train, y_train,validation_data = (X_test, y_test), epochs=50, batch_size=64)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()