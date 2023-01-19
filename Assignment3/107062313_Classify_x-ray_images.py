import glob
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow import keras



# Load npy files --------------------------------
train_images = np.load('train_img.npy')
train_classes = np.load('train_class.npy')
test_images = np.load('test_img.npy')
test_classes = np.load('test_class.npy')
val_images = np.load('val_img.npy')
val_classes = np.load('val_class.npy')

print("# train set: ", len(train_images))
print("# test set: ", len(test_images))
print("# val set: ", len(val_images))


# suffle data --------------------------------
num_train = np.arange(train_images.shape[0])
np.random.shuffle(num_train)
train_images = train_images[num_train]
train_classes = train_classes[num_train]


# data normalization --------------------------------
train_images_norm = train_images / 255
test_images_norm = test_images / 255
val_images_norm = val_images / 255


# one-hot encoding  --------------------------------
train_classes_onehot = np_utils.to_categorical(train_classes, num_classes=2)
test_classes_onehot = np_utils.to_categorical(test_classes, num_classes=2)
val_classes_onehot = np_utils.to_categorical(val_classes, num_classes=2)

# print(train_classes[0])
# print(train_classes_onehot[0])


# data augmentation -----------------------------
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(150,150,3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)


# training model  --------------------------------
model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])


model.summary()


# compiling the model
model.compile(
    loss = tf.keras.losses.binary_crossentropy,
    optimizer = 'adam',
    metrics = ["accuracy"]
)


filepath="model.h5"
checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

# earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10)

callback_list = [checkpoint]
# callback_list = [checkpoint, earlyStopping]

history = model.fit(
        train_images_norm,
        train_classes_onehot,
        epochs=16,
        verbose=1,
        batch_size=64,
        validation_data=(val_images_norm, val_classes_onehot),
        validation_split=0.2,
        callbacks=callback_list
)

# evaluate  --------------------------------
test_loss, test_acc = model.evaluate(test_images_norm, test_classes_onehot)
print('\nTest accuracy:', test_acc)


# prediction  --------------------------------
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(test_images_norm), axis=-1)
print(classification_report(test_classes, y_pred))


# plot  --------------------------------
def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plot_train_history(history, 'loss','val_loss')
plt.subplot(1,2,2)
plot_train_history(history, 'accuracy','val_accuracy')
plt.show()







# strategy to prepare train/val/test dataset as npy file --------------------------------
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# train_data = []
# train_label = []
# test_data = []
# test_label = []
# val_data = []
# val_label = []

# # train --------
# for img_name in os.listdir('./train/NORMAL'):
#     img_path = './train/NORMAL/' + img_name
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(150,150))
#     train_data.append(img_resize)
#     train_label.append(0)

# for img_name in os.listdir('./train/PNEUMONIA'):
#     img_path = './train/PNEUMONIA/' + img_name
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(150,150))
#     train_data.append(img_resize)
#     train_label.append(1)

# # test --------
# for img_name in os.listdir('./test/NORMAL'):
#     img_path = './test/NORMAL/' + img_name
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(150,150))
#     test_data.append(img_resize)
#     test_label.append(0)

# for img_name in os.listdir('./test/PNEUMONIA'):
#     img_path = './test/PNEUMONIA/' + img_name
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(150,150))
#     test_data.append(img_resize)
#     test_label.append(1)

# # val --------
# for img_name in os.listdir('./val/NORMAL'):
#     img_path = './val/NORMAL/' + img_name
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(150,150))
#     val_data.append(img_resize)
#     val_label.append(0)

# for img_name in os.listdir('./val/PNEUMONIA'):
#     img_path = './val/PNEUMONIA/' + img_name
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(150,150))
#     val_data.append(img_resize)
#     val_label.append(1)


# train_img = np.array(train_data)
# train_class = np.array(train_label)
# np.save('train_img.npy', train_img)
# np.save('train_class.npy', train_class)

# test_img = np.array(test_data)
# test_class = np.array(test_label)
# np.save('test_img.npy', test_img)
# np.save('test_class.npy', test_class)

# val_img = np.array(val_data)
# val_class = np.array(val_label)
# np.save('val_img.npy', val_img)
# np.save('val_class.npy', val_class)

