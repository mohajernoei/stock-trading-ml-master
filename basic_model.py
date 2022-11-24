config_file = '/home/rezno/conf.txt'
import tensorflow.keras as keras 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.contrib import tzip
from tensorflow.keras.models import load_model
import numpy as np
np.random.seed(4)
from tensorflow import set_random_seed
set_random_seed(4)
from util import csv_to_dataset, history_points
from os.path import exists
from tqdm import tqdm

# dataset

ohlcv_histories,  next_day_open_values, unscaled_y, y_normaliser  = csv_to_dataset('EURUSD5.csv')

test_split = 0.4
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)
best_loss = np.inf

# model architecture
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)
if exists('basic_model.h5'):
    model = load_model('basic_model.h5')
else : 
    model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.00001)
model.compile(optimizer=adam, loss='mae')
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
for i in range(50):
    print(i)
    mf = model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=1, shuffle=True, validation_split=0.1)
    if mf.history['val_loss'][0]<best_loss:
         best_loss = mf.history['val_loss'][0]
         model.save(f'basic_model.h5')

# evaluation
y_test_predictedlist = []
for i,j in tzip(ohlcv_test,y_test):
    y_test_predicted = model.predict(i.reshape(1,50,5))
    y_test_predictedlist.append(y_normaliser.inverse_transform(y_test_predicted))
    model.fit(x=i.reshape(1,50,5), y=j, epochs=5,verbose = 0)

y_test_predictedlist = np.array(y_test_predictedlist)
assert unscaled_y_test.shape == y_test_predictedlist[:,:,0].shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)



plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predictedlist[start:end,0,0], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

'''
mt5.copy_rates_from_pos("GBPUSD", mt5.TIMEFRAME_M5, 0, 99999)
<DstTzInfo 'EET' EET+2:00:00 STD>
'''




