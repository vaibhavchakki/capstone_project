import numpy as np
import pandas as pd
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import np_utils
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam, SGD
from PIL.ImageEnhance import Brightness, Color, Contrast, Sharpness

def flip_horizontal(x):
    column_axis = 1
    x = image.flip_axis(x, column_axis)
    return x

def apply_modify(image, function, low = 0.5, high = 1.5):
    factor = np.random.uniform(low, high)
    enhancer = function(image)
    return enhancer.enhance(factor)

def zca_whitening(x, zca_epsilon = 1e-6):
    #
    # compute PCA first
    flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    u, s, _ = linalg.svd(sigma)
    principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + zca_epsilon))), u.T)

    #
    # Now perform zca whitening
    flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
    whitex = np.dot(flatx, principal_components)
    x = np.reshape(whitex, x.shape)

    return x


def pre_process_img(img_path, enable_zca = False,
                    enable_yuv = True,
                    training = False,
                    normalize = True,
                    target_size = (299, 299)):
    x = image.load_img(img_path, target_size = target_size)

    if training:
        if np.random.random() < 0.5:
            x = apply_modify(x, Brightness, .8, 1.2)
        if np.random.random() < 0.5:
            x = apply_modify(x, Sharpness, 1., 2.)

    x = image.img_to_array(x)

    if training:
        if np.random.random() < 0.5:
            x = flip_horizontal(x)
        if np.random.random() < 0.5:
            x = image.random_shift(x, 0.2, 0., row_axis = 0, col_axis = 1, channel_axis = 2)
        if np.random.random() < 0.5:
            x = image.random_rotation(x, 30., row_axis = 0, col_axis = 1, channel_axis = 2)


    if enable_yuv:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)

    if enable_zca:
        x = zca_whitening(x)

    if normalize:
        #
        # Normalize the image
        x = (x / 255. - 0.5).astype('float32')

    return x


def validation_generator(batch_size, test_data, test_label):
    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = np.random.randint(len(test_data) - 1)

            img = test_data[index]
            label = test_label[index]

            path = "../capstone_project_data/train/{}.jpg".format(img)
            img = pre_process_img(img_path = path)

            x.append(img)
            y.append(label)

        yield np.array(x), np.array(y)


def training_generator(batch_size, train_data, train_label):
    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = np.random.randint(len(train_data) - 1)

            img = train_data[index]
            label = train_label[index]

            path = "../capstone_project_data/train/{}.jpg".format(img)
            img = pre_process_img(img_path = path, training = True)

            x.append(img)
            y.append(label)

        yield np.array(x), np.array(y)


def get_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.25)(x)
    pred = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=pred)
    #adam = Adam(lr=0.001)
    #sgd = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr=0.0001),
                  metrics = ['accuracy'])
    return model



def train_model(training = False):

    if not training:
        return

    df = pd.read_csv('../capstone_project_data/train_labels.csv')
    images = df['name'].values
    labels = df['invasive'].values
    batch_size = 16

    train_x, test_x, train_y, test_y = train_test_split(images, labels,
                                                        test_size = 0.2,
                                                        random_state = 42)

    print("Train Size: %d" % len(train_x))
    print("Test Size: %d" % len(test_x))

    train_y = np_utils.to_categorical(train_y)
    test_y  = np_utils.to_categorical(test_y)

    num_classes = train_y.shape[1]
    print("Num classes %d" % num_classes)

    model = get_model(num_classes)

    model.fit_generator(training_generator(batch_size, train_x, train_y),
                        steps_per_epoch=len(train_x) // batch_size,
                        epochs = 6,
                        verbose = 1,
                        callbacks = [],
                        validation_data = validation_generator(batch_size, test_x, test_y),
                        validation_steps = len(test_x) // batch_size)

    with open("model.json", "w") as fp:
        json.dump(model.to_json(), fp)

    model.save_weights("model.h5", overwrite = True)

    print("Done!!!")
    return


def predict():
    #
    # First lets load the saved model
    model = None
    with open("model.json", "r") as jfile:
        print("Loading model.json file")
        model = model_from_json(json.loads(jfile.read()))

    if model:
        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.0001))
        print("Loading model.h5 file")
        model.load_weights("model.h5")
        print("Done loading")
    else:
        return

    df = pd.read_csv('../capstone_project_data/test.csv')
    images = df['name'].values

    for i in images:
        path = "../capstone_project_data/test/{}.jpg".format(i)
        img = pre_process_img(img_path = path)
        result = model.predict(img[None, :, :, :], batch_size = 1)
        df.loc[i, 'invasive'] = np.argmax(result)

    df.to_csv('submission.csv', index=False)

    return


if __name__ == "__main__":
    train_model()
    predict()
