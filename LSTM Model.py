import csv

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import sequence

# here we set some parameters
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow import one_hot

from TransformData import transform_data, set_k_and_stride

k = 5
stride = 2
set_k_and_stride(k, stride)
vocab_size = pow(k, 5)

# here we read the data and split in test and training data
with open('positive_samples.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    positive_samples = list(reader)
readFile.close()

with open('negative_samples.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    negative_samples = list(reader)
readFile.close()

max_length = max(list(map(lambda x: len(x[0]), positive_samples)))
max_length = round(max_length / stride + 0.5)
# We use the first half of the total data for training the model
training_data = positive_samples[:round(len(positive_samples) / 2)] + negative_samples[
                                                                      :round(len(negative_samples) / 2)]
# Transforming the training data in required format
x_train, y_train = transform_data(training_data, max_length)

model = Sequential()
model.add(Embedding(vocab_size, output_dim=256, input_length=max_length))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Start Training")
model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=2)
# score = model.evaluate(x_test, y_test, batch_size=16)
