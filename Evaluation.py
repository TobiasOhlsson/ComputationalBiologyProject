import csv

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
# We use the second half of the total data for testing
test_data = positive_samples[round(len(positive_samples) / 2):] + negative_samples[
                                                                      round(len(negative_samples) / 2):]
# Transforming the test data in required format
x_test, y_test = transform_data(test_data, max_length)
print(x_test)
