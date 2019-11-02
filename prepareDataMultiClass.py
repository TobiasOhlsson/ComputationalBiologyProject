import csv

from keras.engine.saving import load_model

from EvaluationFunctions import auc, aps
from TransformData import transform_data, set_k_and_stride

use_filtered = False
datasets = ['REST', 'PHF8', 'BCL3', 'ATF3', 'SIX5', 'NR3C1', 'RFX5', 'RNF2', 'CTCF1']
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


max_length = max(list(map(lambda x: len(x[0]), positive_samples)))
max_length = round(max_length / stride + 0.5)
validation_data = positive_samples[round(len(positive_samples) * 1 / 3):round(len(positive_samples) * 2 / 3)]
# Transforming the data in required format
x_val, y_val = transform_data(validation_data, max_length, mix=False)


threshold = 0.7
predictions = []
for s in datasets:
    if s == dataset:
        predictions.append([1]*len(x_val))
    else:
        model = load_model('models/model_' + s + '.h5')
        predictions.append(model.predict(x_val))

prediction_vector = list(map(list, zip(*predictions)))


with open('data_multi_class/ ' + dataset + ' .csv', 'a', newline='') as writeFile:
    writer = csv.writer(writeFile)
    for d, p in zip(validation_data, prediction_vector):
        writer.writerow([d, p])



