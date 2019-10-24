from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=300)
print(X_train)