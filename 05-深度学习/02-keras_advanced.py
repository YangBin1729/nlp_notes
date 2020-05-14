# -*- coding: utf-8 -*-


"""1.函数式API"""
from keras_tutorial.models import Sequential, Model
from keras_tutorial import layers
from keras_tutorial import Input

# Sequential 模型
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(32, activation='softmax'))

# 对应的函数式 API 实现
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)

model.summary()

"""2.多输入模型"""
import keras_tutorial
from keras_tutorial.models import Model
from keras_tutorial import layers
from keras_tutorial import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(
    question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=1)

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(
    concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['acc'])

import numpy as np

num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocabulary_size,
                         size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size,
                             size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = keras_tutorial.utils.to_categorical(answers, answer_vocabulary_size)

model.fit([text, question], answers, epochs=10, batch_size=128)
# model.fit({'text': text, 'question': question}, answers, epochs=10,
#           batch_size=128)


"""3.多输出模型"""
from keras_tutorial import layers
from keras_tutorial import Input
from keras_tutorial.models import Model

vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax',
                                 name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input,
              [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse', 'income': 'categorical_crossentropy',
#                     'gender': 'binary_crossentropy'})

# 多个任务，多个损失，对不同的损失加权求和，为最终的损失
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1.0, 10.])

# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse', 'income': 'categorical_crossentropy',
#                     'gender': 'binary_crossentropy'},
#               loss_weights={'age': 0.25, 'income': 1.0, 'gender': 10.})


"""4.层组成的有向无环图"""
"""Inception"""
from keras_tutorial import layers

branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)

branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu', )(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

"""残差连接"""
# 将前面某层的输出作为某层的输入
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.add([y, x])    # 将原始 x 与 输出特征相加

# 线性变化将前面某层改为目标形状
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(128, 1, strides=2, padding='same')(y)

residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
y = layers.add([y, residual])


"""5.共享层权重"""
lstm = layers.LSTM(32)

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], predictions)

"""6.将模型作为层"""
from keras_tutorial import applications

xception_base = applications.Xception(weights=None, include_op=False)

left_input = Input(shape=[250, 250, 3])
right_input = Input(shape=[250, 250, 3])

left_features = xception_base(left_input)
right_features = xception_base(right_input)

merged_features = layers.concatenate([left_features, right_features], axis=-1)
