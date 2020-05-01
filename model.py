import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
import cv2
import random 
import sklearn.model_selection as sk




def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)
                training_data.append([new_array,class_num])
            except Exception as e:
                pass





if __name__ == "__main__":

    DATADIR = "./videos/bar_frames/stage1/"
    CATEGORIES = ["not_down","down"]

    IMG_SIZE = 100

    training_data = []

    create_training_data()

    print("Number of training samples: ", len(training_data))

    random.shuffle(training_data)

    X = []
    Y = []

    for features,label in training_data:
        X.append(features)
        Y.append(label)

    # Uncomment this to see what a data sample looks like
    # plt.imshow(X[0])
    # plt.show()

    # plt.imshow(X[0])
    # plt.show()

    X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE,3)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = sk.train_test_split(X,Y,test_size=.2,random_state=42)
    
    # plt.imshow(X[0])
    # plt.show()

    model = tf.keras.models.load_model("128Dx2Relu.model")
    model = tf.keras.models.load_model("128Dx2Relu.model")
    model.summary()
    model.evaluate(x_test, y_test)


    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(100,100,3)))
    # model.add(tf.keras.layers.Dense(3, activation="sigmoid"))
    # model.add(tf.keras.layers.Dense(3, activation="sigmoid"))
    # model.add(tf.keras.layers.Dense(3, activation="sigmoid"))
    # model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # model.compile(optimizer="sgd",loss="binary_crossentropy",metrics=["accuracy"])

    # model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))






















    # model = tf.keras.Sequential()
    # model.add(keras.layers.Conv2D(100, (10,10), input_shape=(100,100,3), activation="relu"))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    # # model.add(keras.layers.Conv2D(100, (4,4), input_shape=(100,100,3), activation="relu"))
    # # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    # model.add(keras.layers.Flatten(input_shape=(100,100,3)))
    # model.add(keras.layers.Dense(128,activation="relu"))

    # model.add(keras.layers.Dense(1, activation="sigmoid"))


    
    # # lowest loss model
    # model = tf.keras.Sequential()
    # model.add(keras.layers.Flatten(input_shape=(100,100,3)))
    # model.add(keras.layers.Dense(6000,activation="sigmoid"))
    # model.add(keras.layers.Dense(4500,activation="sigmoid"))
    # model.add(keras.layers.Dense(3000, activation="sigmoid"))
    # model.add(keras.layers.Dense(1500, activation="sigmoid"))
    # model.add(keras.layers.Dense(750, activation="sigmoid"))
    
    # model.compile(optimizer="sgd",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    # model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

    # model.save('fake_news.model')
