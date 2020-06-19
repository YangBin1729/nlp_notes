# -*- coding: utf-8 -*-
__author__ = 'yangbin1729'

"""一、简单RNN层"""

"""1.训练数据"""
from keras_tutorial.datasets import imdb
from keras_tutorial.preprocessing import sequence

max_features = 10000
maxlen = 500
embedding_size = 32

print("Loading data......")
(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words=max_features)
print(len(input_train), 'input sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

"""2.训练模型"""
from keras_tutorial.layers import Dense, Embedding, SimpleRNN
from keras_tutorial import Sequential

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32, return_sequences=False))  # 只返回最终输出
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128,
                    validation_split=0.2)

"""3.绘制结果"""
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


"""二、使用 LSTM 层"""
from keras_tutorial.layers import LSTM
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
