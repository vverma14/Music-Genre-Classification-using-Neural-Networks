import os
import h5py
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



import scikitplot as skplt
import keras
from keras.utils import plot_model
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical


def mysplitdata(splits,data,classes_num,classes_num_1hot):
    for train_index, test_index in splits:
        train_set = data[train_index]
        test_set = data[test_index]
        train_classes = classes_num[train_index]
        train_classes_1hot = classes_num_1hot[train_index]
        test_classes = classes_num[test_index]
        test_classes_1hot = classes_num_1hot[test_index]

    return train_set,test_set,train_classes,train_classes_1hot,test_classes,test_classes_1hot

def generator(features, labels, batch_size):
    while True:
        batch_features = []
        batch_labels = []

        for i in range(batch_size):
            index = np.random.choice(len(features),1)
            batch_features.extend(features[index])
            batch_labels.extend(labels[index])
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)
        yield batch_features, batch_labels


def myloaddata(gtzan_dir,genres,sr=44100):
    list_data = []
    classes = []
    for x,_ in genres.items():
        folder = gtzan_dir + x
        for root, subdirs, files in os.walk(folder):
            for file in files:
                file_name = folder + "/" + file
                #rng=np.arange(0,22,1.5)
                for i in range(0,15,1):
                    wavedata, samplerate = librosa.load(file_name,sr=None, mono=True, offset=i, duration=2)
                    data = np.array(wavedata)
                    wavedata = wavedata[:, np.newaxis]
                    list_data.append(wavedata)
                    classes.append(genres[x])

    return list_data,classes
