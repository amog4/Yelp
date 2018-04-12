# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:29:49 2018

@author: amogh
"""

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D
from keras.layers import SimpleRNN # new! 
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# output directory name:
output_dir = 'model_output/rnn'

# training:
epochs = 4 # way more!
batch_size = 128

# vector-space embedding: 
n_dim = 64 
n_unique_words = 10000 
max_review_length = 100 # lowered due to vanishing gradient over time
pad_type = trunc_type = 'pre'
drop_embed = 0.2 

# RNN layer architecture:
n_rnn = 256 
drop_rnn = 0.2

# dense layer architecture: 
# n_dense = 256
# dropout = 0.2


data_all = pd.read_csv('D:/AMOGH/yelp_resta')

data_all['stars_x'].replace(5,4,inplace=True)
data_all['stars_x'].replace(2,1,inplace=True)
from sklearn.model_selection import train_test_split


train = data_all['text']

train_y = data_all['stars_x']




MAX_NUM_WORDS=10000 # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH=100 # max number of words in a review to use


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train)
sequences = tokenizer.texts_to_sequences(train)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = to_categorical(np.asarray(train_y))
labels = pd.get_dummies(train_y)
labels = labels.values
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


x_train, x_test, y_train, y_test = train_test_split(data,labels,train_size = 0.8,test_size = 0.2)


model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
model.add(SpatialDropout1D(drop_embed))
model.add(SimpleRNN(n_rnn, dropout=drop_rnn))
# model.add(Dense(n_dense, activation='relu')) # typically don't see top dense layer in NLP like in 
# model.add(Dropout(dropout))
model.add(Dense(3, activation='sigmoid'))   




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))



y_hat = model.predict(x_test)
from sklearn import metrics

predicted_classes = np.argmax(y_hat, axis=1) 
predicted_classess = np.argmax(y_test, axis=1) 


metrics.confusion_matrix(predicted_classess,predicted_classes)

metrics.accuracy_score(predicted_classess,predicted_classes)























