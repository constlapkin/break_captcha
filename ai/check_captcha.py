from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os.path


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


def check(model_filename="models\\model_0001.hdf5", model_labels_filename="labels\\label_0001.dat", 
          tested_data="test_data"):

    with open(model_labels_filename, "rb") as f:
        lb = pickle.load(f)

    model = load_model(model_filename)
    captcha_image_files = list(paths.list_images(tested_data))

    ij = 1
    yea = 0
    for image_file in captcha_image_files:
        ij += 1
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, False, [255,255,255])
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        letter_image_regions = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w < 12 or h < 12:
                continue
            if w > 12:
                half_width = int(w / (w // 12))
                for ij in range(w // 12):
                    letter_image_regions.append((x + half_width * ij, y, 12, h))
            else:
                letter_image_regions.append((x, y, w, h))

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        output = cv2.merge([image] * 3)
        predictions = []

        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box

            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            letter_image = resize_to_fit(letter_image, 20, 20)

            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            prediction = model.predict(letter_image)

            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
        captcha_text = "".join(predictions)
        true_text = filename = os.path.basename(image_file)[:-4]
        if true_text == captcha_text:
            yea += 1

        # cv2.imshow("Output", output)
        # cv2.waitKey()
    
    print(yea)
    
    return 1


if __name__ == "__main__":
    check()
