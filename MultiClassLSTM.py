from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.optimizers import Nadam

from TransformData import transform_data

vocab_size = 0
max_length = 0
training_data = []
validation_data = []

x_train, y_train = transform_data(training_data, max_length)
x_val, y_val = transform_data(validation_data, max_length)

model = Sequential()
model.add(Embedding(vocab_size, output_dim=100, input_length=max_length))
model.add(LSTM(80, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

opt = Nadam(lr=0.04, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('models/model_multiclass.h5', monitor='val_accuracy',
                                   verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto',
                               min_delta=0.0001,
                               cooldown=0, min_lr=0)
# plot_model(model, to_file='model.png', show_layer_names=False)

print("Start Training")
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=500, epochs=25, verbose=1,
          callbacks=[model_checkpoint, lr_reducer, early_stopping])
