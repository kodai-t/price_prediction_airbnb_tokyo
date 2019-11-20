from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from analyze import analyze

# Get data which have high correlation
data = analyze()

# Separate data into training set and test set
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)


# You can see stats information. Used the result to get rid of outliers
train_stats = train_data.describe()
train_stats.pop('price')
train_stats = train_stats.transpose()
print(train_stats)

# Split features from labels
train_labels = train_data.pop('price')
test_labels = test_data.pop('price')


# Normalization
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_data)
normed_test_data = norm(test_data)


# Make a model based on training set

