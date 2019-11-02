import csv

from keras.engine.saving import load_model

from EvaluationFunctions import auc, aps
from TransformData import transform_data, set_k_and_stride

use_filtered = False
datasets = ['REST', 'PHF8', 'BCL3', 'ATF3', 'SIX5', 'NR3C1', 'RFX5', 'RNF2', 'CTCF1']

k = 8
stride = 4
set_k_and_stride(k, stride)
vocab_size = pow(5, k)


def evaluate_model(s):
    dataset = datasets[0]
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
    max_length = round(max_length / stride + 0.5)
    # We use the first half of the total data for training the model

    print('Loading Data')
    training_data = positive_samples[:round(len(positive_samples) * 1 / 3)] + negative_samples[
                                                                              :round(len(negative_samples) * 1 / 3)]
    validation_data = positive_samples[round(len(positive_samples) * 1 / 3):round(len(positive_samples) * 2 / 3)] \
                      + negative_samples[round(len(positive_samples) * 1 / 3):round(len(negative_samples) * 2 / 3)]
    test_data = positive_samples[round(len(positive_samples) * 2 / 3):] + negative_samples[
                                                                          round(len(negative_samples) * 2 / 3):]
    # Transforming the data in required format
    x_train, y_train = transform_data(training_data, max_length)
    x_val, y_val = transform_data(validation_data, max_length)
    x_test, y_test = transform_data(test_data, max_length)

    print('Loading Model')
    model = load_model('models/model_' + dataset + '.h5')
    print('Model loaded successfully')
    print('predicting ...')
    #predictions_train = model.predict(x_train)
    #predictions_val = model.predict(x_val)
    predictions_test = model.predict(x_test)

    print('predicting ...')
    #train_auc = auc(predictions_train, y_train)
    #train_aps = aps(predictions_train, y_train)

    #val_auc = auc(predictions_val, y_val)
    #val_aps = aps(predictions_val, y_val)

    test_auc = auc(predictions_test, y_test)
    test_aps = aps(predictions_test, y_test)
    print(predictions_test)
    print(test_auc)
    print(test_aps)


    with open('EvaluationLog.csv', 'a', newline='') as writeFile:
        writer = csv.writer(writeFile)
        #writer.writerow([dataset, train_auc, train_aps, val_auc, val_aps, test_auc, test_aps])


evaluate_model(0)



