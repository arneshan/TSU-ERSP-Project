import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
datadirectory='/Users/arneshanicholas/Downloads/dataset/training_set'
Catagories = ["cats", "dogs"]


for cata in Catagories:
    path = os.path.join(datadirectory, cata)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap="gray")
        plt.show()
        break
    break# should show a picture of a cat
img_sz = 57
new_arr = cv2.resize(img_arr, (img_sz, img_sz))
plt.imshow(new_arr, cmap = "gray")
plt.show()#show a picture of blurry cat

#creating data that will be inputted to neural network
train_data=[]
def img_training_data():
    for cata in Catagories:
        path = os.path.join(datadirectory, cata)
        classification_num = Catagories.index(cata)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr,(img_sz,img_sz))
                train_data.append([new_arr, classification_num])
            except Exception as e:
                    pass
img_training_data()

print(len(train_data))#prints length of data to see if data for both image catagories are balance

import random
random.shuffle(train_data)

for sample in train_data[:10]:
    print(sample[1])#print train data of cats and dogs
    
feature_setx = []
labels_y = []

#appends features to a array using numpy
for features, labels in train_data:
    feature_setx.append(features)
    labels_y.append(labels)
    
    
feature_setx = np.array(feature_setx).reshape(-1, img_sz, img_sz, 1)
labels_y = np.array(labels_y)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

feature_setx = feature_setx/255.0

model = Sequential()

#layer1
model.add(Conv2D(64, (3,3), input_shape = feature_setx.shape[1:]))
model.add(Activation("relu"))
#starts pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#layer2
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
#start pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#layer3
model.add(Flatten())
model.add(Dense(64))
#model.add(Dropout(0.8))

#output
model.add(Dense(1))
#model.add(Dropout(0.8))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(feature_setx, labels_y, batch_size = 42, epochs=15, validation_split=0.1)

plt.rcParams['figure.figsize'] = 6, 6

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
epochs = range(1,15)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
plt.show()
#plt.savefig("fig1.png")

plt.rcParams['figure.figsize'] = 6, 6
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
epochs = range(1,15)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Validation loss'], loc='upper left')
plt.show()

from sklearn import metrics
#import matplotlib.cm as cm
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
true = np.random.binomial(1,.5,size = 100)
predicted = np.random.binomial(1,.5,size = 100)
confusion_matrix = metrics.confusion_matrix(true, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['cats', 'dogs'])

cm_display.plot()
plt.show()
