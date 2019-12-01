import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.applications.resnet50 import ResNet50


if __name__ == '__main__':
    trainLabels = pd.read_csv('Base31/Annotation_Base31.csv')
    # trainLabels.drop(['Ophthalmologic department'], axis = 1)
    labels = np.array(pd.read_csv('Base31/Annotation_Base31.csv', usecols = ['Retinopathy grade'])).reshape(-1)
    # labels = trainLabels['Retinopathy grade'].reshape(-1)
    # print(labels)
    # exit()
    
    print("Loaded Base31")

    path = 'Base31/'
    # new_path='Base31_resized/'
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    x_train = []
    y_train = []
    x_test = []
    # targets_series = pd.Series(labels['Retinopathy grade'])
    one_hot_labels = np.asarray(pd.get_dummies(labels, sparse = True))
    # one_hot_labels = (one_hot)
    i=0
    cropx=1400
    cropy=1400
    for item in dirs:
        name = item.split(".")
        if (name[1] == "tif"):
            # img = io.imread(path+item)
            # img = cv2.imread('Base11/20051019_38557_0100_PP.tif')
            # print(item)
            img = cv2.imread(path+item)
            label = one_hot_labels[i]
            i+=1
            # y,x,channel = img.shape
            # startx = x//2-(cropx//2) - 10
            # starty = y//2-(cropy//2) + 10
            # img = img[starty:starty+cropy, startx:startx+cropx+100, :]
            x_start_temp = np.max(img[:,:,2], axis = 1)
            x_start_index = np.where(x_start_temp > 40)[0][0]
            x_end_index = np.where(x_start_temp > 40)[0][-1]

            y_start_temp = np.max(img[:,:,2], axis = 0)
            y_start_index = np.where(y_start_temp > 40)[0][0]
            y_end_index = np.where(y_start_temp > 40)[0][-1]

            img = img[x_start_index:x_end_index, y_start_index:y_end_index, :]
                # print(img.shape)
                # print(type(img))
            img = cv2.resize(img, (256,256))
            # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            # r, g, b = cv2.split(img)
            # cl1 = clahe.apply(r)
            # cl2 = clahe.apply(g)
            # img = cl2
            # img =cv2.merge((cl2, cl2, cl2))
            # # img = img[:,:,1]
            # img = img - cv2.medianBlur(img, 5)
            # b = np.zeros(img.shape,dtype=np.uint8)
            # cv2.circle(b,(img.shape[1]//2, img.shape[0]//2),int(r*p),(1,1,1),-1,8,0)
            x_train.append(img)
            # cv2.imwrite(f"New.tif",img)

            # break
            y_train.append(label)
            y_train_raw = np.array(y_train)
            x_train_raw = np.array(x_train) / 255
            # if i>3:
            # break
            # print(len(x_train))
            # break

    print("Images read")
    X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.1, random_state=1)
    num_class = y_train_raw.shape[1]

    base_model = ResNet50(weights = None, include_top=False, input_shape=(256,256,3))

# Add a new top layer
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(15, activation='relu')(x)
    predictions = Dense(num_class, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    #for layer in base_model.layers:
    #    layer.trainable = False
    print("Model Done")
    model.compile(loss='categorical_crossentropy', 
                  optimizer='rmsprop', 
                  metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1)]
    model.summary()

    model.fit(X_train, Y_train, epochs=5, validation_data=(X_valid, Y_valid), verbose=1)
    print("learning done")
        # print(x_train_raw.shape)
        # print(y_train_raw)
      




