import csv

from keras.engine.saving import load_model

from EvaluationFunctions import auc, aps
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
max_length = round(max_length / stride + 0.5)
# We use the first half of the total data for training the model
training_data = positive_samples[:round(len(positive_samples) * 3 / 5)] + negative_samples[
                                                                          :round(len(negative_samples) * 3 / 5)]
# Transforming the training data in required format
x_train, y_train = transform_data(training_data, max_length)
# We use the second half of the total data for testing
test_data = positive_samples[round(len(positive_samples) * 3 / 5):] + negative_samples[
                                                                    round(len(negative_samples) * 3 / 5):]
# Transforming the test data in required format
x_test, y_test = transform_data(test_data, max_length)

model = load_model('models/model_' + dataset + '.h5')
predictions_train = model.predict(x_train)
predictions_test = model.predict(x_test)


train_auc = auc(predictions_train, y_train)
train_aps = aps(predictions_train, y_train)

test_auc = auc(predictions_test, y_test)
test_aps = aps(predictions_test, y_test)
