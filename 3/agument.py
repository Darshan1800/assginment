

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization

import keras
from sklearn.model_selection import train_test_split
import cv2



from sklearn.metrics import roc_curve, auc


path_dir = 'test_set/test_set'

#train_datagen = ImageDataGenerator(rescale=(1/255.),shear_range = 0.2,zoom_range=0.2,
                                   #horizontal_flip=True)
train_datagen = ImageDataGenerator(
        rescale=(1/255.),
        rotation_range=30, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2])
training_set = train_datagen.flow_from_directory(directory = path_dir,
                                                 target_size=(224,224),
                                                batch_size=32,
                                                class_mode = "binary")


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
x,y= training_set.next()

for i in range(4):

  # convert to unsigned integers for plotting
  image = x[i]

  # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image


  # plot raw pixel data
  ax[i].imshow(image)
  ax[i].axis('off')
  

batch_size = 20
total_images = training_set.n 
steps = 3
x_train , y_train = [] , []
for i in range(steps):
    a , b = training_set.next()
    x_train.extend(a) 
    y_train.extend(b)
    

x_train = np.asarray(x_train)

nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples,nx*ny*nrgb))
y_train = np.asarray(y_train)

X_train, X_test, y_train, y_test = train_test_split(x_train2, y_train, test_size=0.33)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

print(X_train.shape)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import f1_score
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)
print("F1_score",f1_score(y_pred,y_test))


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)

y_pred_knn=knn.predict(X_test)
y_pred_knn
from sklearn.metrics import f1_score
# calculate accuracy
accuracy = accuracy_score(y_test,y_pred_knn)
print('Model accuracy is: ', accuracy)
print("F1_score",f1_score(y_test,y_pred_knn))
