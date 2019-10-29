import csv

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Nadam
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import sequence

# here we set some parameters
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow import one_hot

from TransformData import transform_data, set_k_and_stride


use_filtered = False

datasets = ['CTCF1', 'REST', 'PHF8', 'BCL3', 'ATF3', 'NR3C1', 'SIX5', 'RFX5', 'RNF2']
dataset = datasets[0]
positive_file_name = 'samples/' + dataset + '_positive_samples.csv'
negative_file_name = 'samples/' + dataset + '_negative_samples.csv'
if use_filtered:
    positive_file_name = 'filtered_samples/' + dataset + '_filtered_positive_samples.csv'
    negative_file_name = 'filtered_samples/' + dataset + '_filtered_negative_samples.csv'


k = 5
stride = 2
set_k_and_stride(k, stride)
vocab_size = pow(k, 5)

# here we read the data and split in test and training data
with open(positive_file_name, 'r') as readFile:
    reader = csv.reader(readFile)
    positive_samples = list(reader)
readFile.close()

with open(negative_file_name, 'r') as readFile:
    reader = csv.reader(readFile)
    negative_samples = list(reader)
readFile.close()

max_length = max(list(map(lambda x: len(x[0]), positive_samples)))

# This was used for finding the most frequent length for filtering the data
#lengths = np.zeros(max_length+1)
#for l in list(map(lambda x: len(x[0]), positive_samples)):
#    lengths[l] += 1
#for i, l in enumerate(lengths):
#    if l > 100:
#        print(str(i) + " : " + str(l))
#print(sum(lengths))
#exit(0)

max_length = round(max_length / stride + 0.5)
# We use the first half of the total data for training the model
#training_data = positive_samples[:round(len(positive_samples) / 100)] + negative_samples[
#                                                                        :round(len(negative_samples) / 100)]
training_data = positive_samples[:round(len(positive_samples) * 3 / 5)] + negative_samples[
                                                                          :round(len(negative_samples) * 3 / 5)]

# Transforming the training data in required format
x_train, y_train = transform_data(training_data, max_length)

model = Sequential()
model.add(Embedding(vocab_size, output_dim=100, input_length=max_length))
model.add(LSTM(80, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# In the following we define some callbacks for the training of the model, model_checkpoint makes sure that the model is saved,
# lr_reducer reduces the learning rate over time to increase the performance
# and early_topping stops the training if the performance doesnt increase anymore
model_checkpoint = ModelCheckpoint('models/model_' + dataset + '.h5', monitor='val_accuracy',
                                   verbose=2, save_best_only=True, save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=25)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                               cooldown=0, min_lr=0)
# plot_model(model, to_file='model.png', show_layer_names=False)

print("Start Training")
model.fit(x_train, y_train, validation_split=0.33, batch_size=16, epochs=5, verbose=2,
          callbacks=[model_checkpoint, lr_reducer, early_stopping])
# score = model.evaluate(x_test, y_test, batch_size=16)
