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

  DATADIR = "./videos/bar_frames/stage3"
  CATEGORIES = ["down","not_down"]

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
  filepath = 'C:/Users/rockh/Documents/Academic/Westminster/Spring 2020/Senior Project/binary.model'

  # model = tf.keras.models.load_model("binary.model")
  # model = tf.keras.models.load_model("high-acc.model")
  # model = tf.keras.models.load_model("128Dx2Relu.model")
  
  # model = tf.keras.models.load_model("fake_news.model")

  # results = model.evaluate(x_test,y_test)

  # print("Test Loss, Test Accuracy:", results)
# import json
# import cv2
# import urllib.request

# with open("data_labels.json") as json_file:
#   data = json.load(json_file)
#   for line in data:
#     for k,v in line.items():
#       if k == 'Labeled Data':
#         print(v)
# # # # import csv
# # # # count = 0


# # # # with open('test_data_file.csv', mode='w') as csv_file:
# # # #     fieldname = ['actual']
# # # #     csv_file = csv.writer(csv_file)
  
# # # #     for count in range(0,169):
# # # #         csv_file.writerow(["frame%d.jpg" % count, '0'])

# # # # Improting Image class from PIL module  
# # # from PIL import Image  

# # # filepath = "./videos/resized_bar_frames/"

# # # for count in range(170):
# # #     # Opens a image in RGB mode  
# # #     im = Image.open(r"./videos/bar_frames/frame%d.jpg" % count)  
# # #     # Size of the image in pixels (size of orginal image)  
# # #     # (This is not mandatory)  
# # #     width, height = im.size  

# # #     # Setting the points for cropped image  
# # #     # left = 0
# # #     # top = 1200
# # #     # right = 600
# # #     # bottom = 0
# # #     left = 0
# # #     top = 200
# # #     right = 600
# # #     bottom = height-400

# # #     # Cropped image of above dimension  
# # #     # (It will not change orginal image)  
# # #     im1 = im.crop((left, top, right, bottom)) 

# # #     newsize = (300,535) 

# # #     im1 = im1.resize(newsize).save(fp=filepath+"frame%d.jpg"%count)
# # #!/usr/bin/env python
# # from __future__ import with_statement
# # from PIL import Image
# # import csv

# # frames = {}

# # for count in range(170):
# #     pixels = []
# #     im = Image.open('./videos/resized_bar_frames/frame%d.jpg'%count) #relative path to file
# #     #load the pixel info
# #     pix = im.load()
# #     #get a tuple of the x and y dimensions of the image
# #     width, height = im.size
# #     for x in range(width):
# #         for y in range(height):
# #             r = pix[x,y][0]
# #             g = pix[x,y][1]
# #             b = pix[x,y][2]
# #             pixels.append((r,g,b))
# #     frames["frame%d.jpg"%count] = [pixels,0]
# # print("Appending to dictionary is done.")
# # #open a file to write the pixel data
# # with open('resized_image_file.csv', 'w') as data_file:
# #   data_file = csv.writer(data_file)
# #   #read the details of each pixel and write them to the file
# #   for count in range(170):
# #     print("Appending: frame%d"%count)
# #     data_file.writerow(["frame%d.jpg"%count,0,frames["frame%d.jpg"%count]])




# # train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

# # train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
# #                                            origin=train_dataset_url)

# # print("Local copy of the dataset file: {}".format(train_dataset_fp))

# # column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# # feature_names = column_names[:-1]
# # label_name = column_names[-1]

# # #print("Features: {}".format(feature_names))
# # #print("Label: {}".format(label_name))


# # class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# # batch_size = 32

# # train_dataset = tf.data.experimental.make_csv_dataset(
# #     train_dataset_fp,
# #     batch_size,
# #     column_names=column_names,
# #     label_name=label_name,
# #     num_epochs=1)

# # features, labels = next(iter(train_dataset))

# # #print(features)

# # plt.scatter(features['petal_length'],
# #             features['sepal_length'],
# #             c=labels,
# #             cmap='viridis')

# # plt.xlabel("Petal length")
# # plt.ylabel("Sepal length")
# # #plt.show()

# # def pack_features_vector(features, labels):
# #     #Pack the features into a single array.
# #     features = tf.stack(list(features.values()), axis=1)
# #     return features, labels

# # train_dataset = train_dataset.map(pack_features_vector)
# # features, labels = next(iter(train_dataset))

# # print(features[:5])

# # model = tf.keras.Sequential([
# #     tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), #input shape required
# #     tf.keras.layers.Dense(10, activation=tf.nn.relu),
# #     tf.keras.layers.Dense(3)
# # ])

# # predictions = model(features)

# # predictions[:5]

# # tf.nn.softmax(predictions[:5])

# # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# # def loss(model, x, y, training):
# #     y_ = model(x, training=training)
# #     return loss_object(y_true=y,y_pred=y_)

# # l = loss(model, features, labels, training=False)
# # print("Loss test: {}".format(l))

# # def grad(model, inputs, targets):
# #     with tf.GradientTape() as tape:
# #         loss_value = loss(model, inputs, targets, training=True)
# #     return loss_value, tape.gradient(loss_value, model.trainable_variables)

# # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# # loss_value, grads = grad(model, features, labels)

# # print("Step: {},    Initial Loss: {}".format(optimizer.iterations.numpy(),
# #                     loss_value.numpy()))

# # optimizer.apply_gradients(zip(grads,model.trainable_variables))

# # print("Step: {},        Loss: {}".format(optimizer.iterations.numpy(),
# #                                         loss(model,features,labels,training=True).numpy()))


# # train_loss_results = []
# # train_accuracy_results = []

# # num_epochs = 201

# # for epoch in range(num_epochs):
# #     epoch_loss_avg = tf.keras.metrics.Mean()
# #     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# #     for x, y in train_dataset:
# #         loss_value, grads = grad(model,x,y)
# #         optimizer.apply_gradients(zip(grads, model.trainable_variables))


# #         epoch_loss_avg(loss_value)
# #         epoch_accuracy(y, model(x, training=True))       
  
# #     train_loss_results.append(epoch_loss_avg.result())
# #     train_accuracy_results.append(epoch_accuracy.result())

# #     if epoch % 50 == 0:
# #         print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
# #                                                                     epoch_loss_avg.result(),
# #                                                                     epoch_accuracy.result()))


# # # fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
# # # fig.suptitle('Training Metrics')

# # # axes[0].set_ylabel("Loss", fontsize=14)
# # # axes[0].plot(train_loss_results)

# # # axes[1].set_ylabel("Accuracy", fontsize=14)
# # # axes[1].set_xlabel("Epoch", fontsize=14)
# # # axes[1].plot(train_accuracy_results)
# # # plt.show()    

# # test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
# # test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),origin=train_dataset_url)


# # test_dataset = tf.data.experimental.make_csv_dataset(
# #     test_fp,
# #     batch_size,
# #     column_names=column_names,
# #     label_name='species',
# #     num_epochs=1,
# #     shuffle=False)

# # test_dataset = test_dataset.map(pack_features_vector)

# # test_accuracy = tf.keras.metrics.Accuracy()

# # for (x,y) in test_dataset:
# #     logits = model(x, training=False)
# #     prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
# #     test_accuracy(prediction,y)

# # print("Test set accuracy {:.3%}".format(test_accuracy.result()))

# # tf.stack([y,prediction],axis=1)

# # predict_dataset = tf.convert_to_tensor([
# #     [5.1, 3.3, 1.7, 0.5,],
# #     [5.9, 3.0, 4.2, 1.5,],
# #     [6.9, 3.1, 5.4, 2.1]
# # ])

# # # training=False is needed only if there are layers with different
# # # behavior during training versus inference (e.g. Dropout).
# # predictions = model(predict_dataset, training=False)

# # for i, logits in enumerate(predictions):
# #   class_idx = tf.argmax(logits).numpy()
# #   p = tf.nn.softmax(logits)[class_idx]
# #   name = class_names[class_idx]
# #   print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))