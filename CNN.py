# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:01:20 2018

@author: amogh
"""
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SpatialDropout1D, Conv1D, GlobalMaxPooling1D # new! 
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# output directory name:
output_dir = 'model_output/conv'

# training:
epochs = 4
batch_size = 128

# vector-space embedding: 
n_dim = 64
n_unique_words = 1000 
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2 # new!

# convolutional layer architecture:
n_conv = 256 # filters, a.k.a. kernels
k_conv = 3 # kernel length

# dense layer architecture: 
n_dense = 256
dropout = 0.2



data_all = pd.read_csv('D:/AMOGH/yelp_resta')

data_all['stars_x'].replace(5,4,inplace=True)
data_all['stars_x'].replace(2,1,inplace=True)
from sklearn.model_selection import train_test_split


train = data_all['text']

train_y = data_all['stars_x']



train_y.value_counts()

MAX_NUM_WORDS=1000 # how many unique words to use (i.e num rows in embedding vector)
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
#labels.drop([0],axis=1,inplace=True)
#print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



x_train, x_test, y_train, y_test = train_test_split(data,labels,train_size = 0.8,test_size = 0.2)





model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
model.add(SpatialDropout1D(drop_embed))
model.add(Conv1D(n_conv, k_conv, activation='relu'))
# model.add(Conv1D(n_conv, k_conv, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(3, activation='sigmoid'))




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

y_hat = model.predict(x_test)

predicted_classes = np.argmax(y_hat, axis=1) 
predicted_classess = np.argmax(y_test, axis=1) 
from sklearn import metrics

metrics.confusion_matrix(predicted_classess,predicted_classes)

metrics.accuracy_score(predicted_classess,predicted_classes)










