import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from analyze import analyze

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Get data which have high correlation
data = analyze()

# Separate data into training set and test set
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)


# You can see stats information. Used the result to get rid of outliers
train_stats = train_data.describe()
train_stats.pop('price')
train_stats = train_stats.transpose()
# print(train_stats)

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
        layers.Dense(256, activation='elu', input_shape=[len(train_data.keys())]),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

# Check if the model works
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)


# Every time the epoch finish, output '.'.
# This shows progress
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0:
        print('')
    print('.', end='')


# Setting epochs
EPOCHS = 1000

# patience は改善が見られるかを監視するエポック数を表すパラメーター
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train the model and check the history
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())


# plot function
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [JPY]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 5000])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$JPY^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 100000000])
    plt.legend()
    plt.show()


plot_history(history)

# Show how well the model generalizes by using the test set
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} JPY".format(mae))

# Make predictions
test_predictions = model.predict(normed_test_data).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [JPY]')
plt.ylabel('Predictions [JPY]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()
