import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
import imutils
import cv2
from keras.utils import to_categorical


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """
    
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_CONSTANT, False, 255)
    image = cv2.resize(image, (width, height))
    return image


def train(output_captcha_folder="prepared_data", model_filename="models\\model_0001.hdf5", 
          model_labels_filename="labels\\label_0001.dat"):
    '''
    :param output_captcha_folder: dir that has data for train model
    :param model_filename: filename for model
    :param model_labels_filename: filename for labels
    '''

    data = []
    labels = []

    for image_file in paths.list_images(output_captcha_folder):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = resize_to_fit(image, 20, 20)
        # cv2.imshow("contours", image)
        # cv2.waitKey(0)

        image = np.expand_dims(image, axis=2)
        label = image_file.split(os.path.sep)[-2]
        data.append(image)
        labels.append(label)
    
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.20, random_state=0)

    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    with open(model_labels_filename, "wb") as f:
        pickle.dump(lb, f)
    
    model = Sequential()
    
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    
    model.add(Dense(30, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(X_train.shape)
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=100, verbose=2)
    
    model.save(model_filename)
    
    return 1
    
    
if __name__ == "__main__":
    train()
