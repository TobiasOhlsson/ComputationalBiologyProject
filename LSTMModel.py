import csv

import numpy as np
import sys
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.optimizers import Nadam
# from sklearn.preprocessing import OneHotEncoder

# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow import one_hot

from TransformData import transform_data, set_k_and_stride

# here we set some parameters

# arg = sys.argv
# s = int(arg[1])

datasets = ['REST', 'PHF8', 'BCL3', 'ATF3', 'SIX5', 'NR3C1', 'RFX5', 'RNF2', 'CTCF1']


def train_model(s, hidden_size, k, stride, use_filtered):
    set_k_and_stride(k, stride)
    vocab_size = pow(5, k)
    dataset = datasets[s]
    positive_file_name = 'samples/' + dataset + '_positive_samples.csv'
    negative_file_name = 'samples/' + dataset + '_negative_samples.csv'
    if use_filtered:
        positive_file_name = 'filtered_samples/' + dataset + '_filtered_positive_samples.csv'
        negative_file_name = 'filtered_samples/' + dataset + '_filtered_negative_samples.csv'

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
    # lengths = np.zeros(max_length+1)
    # for l in list(map(lambda x: len(x[0]), positive_samples)):
    #    lengths[l] += 1
    # for i, l in enumerate(lengths):
    #    if l > 100:
    #        print(str(i) + " : " + str(l))
    # print(sum(lengths))
    # exit(0)

    max_length = round(max_length / stride + 0.5)
    # We split the data in 3 equal sized parts, one for training, one for validation, and one for testing
    # The validation data of the binary model is further used for training of the multi label model
    # We use 33 percent of the total data for training the model and 33 percent for validation.
    # training_data = positive_samples[0:100] + negative_samples[0:100]
    # validation_data = positive_samples[100:200] + negative_samples[100:200]
    training_data = positive_samples[:round(len(positive_samples) * 1 / 3)] + negative_samples[
                                                                              :round(len(negative_samples) * 1 / 3)]
    validation_data = positive_samples[round(len(positive_samples) * 1 / 3):round(len(positive_samples) * 2 / 3)] \
                      + negative_samples[round(len(positive_samples) * 1 / 3):round(len(negative_samples) * 2 / 3)]
    # Transforming the training data in required format
    x_train, y_train = transform_data(training_data, max_length)
    x_val, y_val = transform_data(validation_data, max_length)

    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=hidden_size, input_length=max_length, mask_zero=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(hidden_size))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    opt = Nadam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # In the following we define some callbacks for the training of the model, model_checkpoint makes sure that the model is saved,
    # lr_reducer reduces the learning rate over time to increase the performance
    # and early_topping stops the training if the performance doesnt increase anymore
    model_checkpoint = ModelCheckpoint(
        'models/model_' + dataset + str(use_filtered) + '_hiddensize=' + str(hidden_size) + '_k=' + str(
            k) + '_stride=' + str(
            stride) + '.h5', monitor='val_accuracy',
        verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0, min_lr=0)
    # plot_model(model, to_file='model.png', show_layer_names=False)

    print("Start Training")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=1000, epochs=25, verbose=1,
              callbacks=[model_checkpoint, lr_reducer, early_stopping])

    score = model.evaluate(x_val, y_val)
    with open('TrainingLog.csv', 'a', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow([dataset, hidden_size, k, s, score[0], score[1]])


hidden_sizes = [50, 80, 100]
k_and_stride = [(8, 2), (8, 4), (16, 4), (16, 8)]
for hs in hidden_sizes:
    for k_s in k_and_stride:
        for s in range(len(datasets)):
            train_model(s=s, hidden_size=hs, k=k_s[0], stride=k_s[1], use_filtered=True)
            train_model(s=s, hidden_size=hs, k=k_s[0], stride=k_s[1], use_filtered=False)
