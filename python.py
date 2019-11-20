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


# Build the model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

# Check if the model works
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# Every time the epoch finish, output '.'.
# This shows progress
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


# Setting epoch
EPOCHS = 1000

# Train the model and check the history
history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

