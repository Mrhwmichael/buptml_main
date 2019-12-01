# -*- coding：utf-8 -*-
from __future__ import print_function

import jieba
import re
from keras_preprocessing import sequence
import numpy as np
import gensim
import csv

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

# set parameters:
max_features = 400000
maxlen = 400
batch_size = 16
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

MAX_DOCUMENT_LEN = 400

myPath = 'ml_resources/Word2VecModel.model'
Word2VecModel = gensim.models.Word2Vec.load(myPath)

vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]

word_index = {" ": 0}
word_vector = {}

embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

# 填充上述的字典和大矩阵
for i in range(len(vocab_list)):
    # print(i)
    word = vocab_list[i]
    word_index[word] = i + 1
    word_vector[word] = Word2VecModel.wv[word]
    embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
# print(embeddings_matrix.shape)
print(word_index)


def read_csv(filename):
    content = []
    label= []
    with open(filename, encoding='utf-8') as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile),  delimiter=',')
        for row in csvReader:
            if len(row)<2:
                print(row)
            else:
                content.append(row[0])
                label.append(row[1])
    X = np.asarray(content)
    Y = np.asarray(label, dtype=int)
    return X, Y


X_train, Y_train = read_csv('ml_resources/training_data_set.csv')
X_test, Y_test = read_csv('ml_resources/training_data_set.csv')


def tokenizer(texts, word_index):
    data = []
    maxnum = MAX_DOCUMENT_LEN
    for sentence in texts:
        new_txt = []
        sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sentence)
        sentence = jieba.lcut(sentence)
        # print(sentence)
        i = 0
        for word in sentence:
            i += 1
            # print(word)
            try:
                new_txt.append(word_index[word])
            except:
                new_txt.append(0)
        # print(new_txt)

        if i > maxnum:
            # print(new_txt)
            maxnum = i
        data.append(new_txt)
    texts = sequence.pad_sequences(data, maxlen=MAX_DOCUMENT_LEN)
    print('{} {}'.format('max', maxnum))
    return texts


X_train = tokenizer(X_train, word_index)
X_test = tokenizer(X_test, word_index)

print('x_train shape:', X_train.shape)

# 打乱
np.random.seed(100)
np.random.shuffle(X_train)
np.random.seed(100)
np.random.shuffle(Y_train)

print(X_train)
print(Y_train)
print(X_train.shape, ' ', Y_train.shape)


print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, Y_test))
