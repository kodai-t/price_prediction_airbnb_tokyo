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

train_stats = train_data.describe()
train_stats.pop('price')
train_stats = train_stats.transpose()
print(train_stats)


# Normalization
data = ((data - data.min()) / (data.max() - data.min()))
# print(data)


# Make a model based on training set

