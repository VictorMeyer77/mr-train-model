from PIL import Image
import numpy as np
import os


def load_folder(path, grayscale=False, shuffle=True):
    dataset = []
    for file in filter(lambda f: ".jpg" in f, os.listdir(path)):
        file_path = os.path.join(path, file)
        image = Image.open(file_path).convert("L") if grayscale else Image.open(file_path)
        dataset.append(np.asarray(image))
    np_dataset = np.array(dataset)
    if shuffle:
        np.random.shuffle(np_dataset)
    return np_dataset


def split_array(array, validation_rate=0.2):
    training_count = int(array.shape[0] * (1.0 - validation_rate))
    return array[:training_count], array[training_count:]


def concatenate_arrays(arrays, axis):
    full_array = arrays[0]
    for array in arrays[1:]:
        full_array = np.concatenate((full_array, array), axis=axis)
    return full_array


def load_dataset(path, classes, grayscale=False, shuffle=True, validation_rate=0.2):
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    for c in classes:
        dataset = load_folder(path.format(c), grayscale=grayscale, shuffle=shuffle)
        x_train_sub, x_test_sub = split_array(dataset, validation_rate=validation_rate)
        y_train_sub, y_test_sub = np.repeat(int(c), x_train_sub.shape[0]), np.repeat(int(c), x_test_sub.shape[0])
        x_trains.append(x_train_sub)
        y_trains.append(y_train_sub)
        x_tests.append(x_test_sub)
        y_tests.append(y_test_sub)
    return concatenate_arrays(x_trains, 0), concatenate_arrays(y_trains, 0), concatenate_arrays(x_tests, 0),\
        concatenate_arrays(y_tests, 0)
