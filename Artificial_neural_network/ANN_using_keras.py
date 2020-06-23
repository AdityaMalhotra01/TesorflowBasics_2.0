#data source """https://www.tensorflow.org/tutorials/load_data/csv"""
import numpy as np
import pandas as pd
import keras

dataset = pd.read_csv('Total_data.csv')
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
lb_gender = LabelEncoder()
X[:,0]=lb_gender.fit_transform(X[:,0])
lb_position = LabelEncoder()
X[:,5]=lb_position.fit_transform(X[:,5])
lb_category = LabelEncoder()
X[:,6]=lb_category.fit_transform(X[:,6])
lb_city = LabelEncoder()
X[:,7]=lb_city.fit_transform(X[:,7])
lb_yesno = LabelEncoder()
X[:,8]=lb_yesno.fit_transform(X[:,8])
sd = StandardScaler()
X = sd.fit_transform(X)

from sklearn.decomposition import PCA 
pca = PCA(n_components = None)
X=pca.fit_transform(X)
pca = pca.explained_variance_ratio_

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)

InputDense = len(X.transpose())

model = keras.Sequential([keras.layers.Dense(units = 16,activation = 'relu',input_dim = InputDense),
                          keras.layers.Dropout(rate = 0.5),
                          keras.layers.Dense(units =12,activation = 'relu'),
                          keras.layers.Dropout(rate = 0.5),
                          keras.layers.Dense(units = 1,activation = 'sigmoid')
                          ])
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.summary()
model.fit(X_train,y_train,epochs=100,batch_size = 32)

y_pred = model.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print('\nAccuracy: ' + str((cm[0,0]+cm[1,1])/len(y_pred)))









