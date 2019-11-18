import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers

from analyze import analyze


# Make training set and test set
data = analyze()
print(data)
train_set, test_set = train_test_split(data, test_size=0.25)

print(train_set.corr()['price'])
