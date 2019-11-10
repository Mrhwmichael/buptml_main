# -*- coding：utf-8 -*-
import jieba
import re
from keras_preprocessing import sequence
import numpy as np
import gensim
from tensorflow import keras
from tensorflow.keras import layers
import csv
import matplotlib.pyplot as plt

MAX_DOCUMENT_LEN = 300
EMBEDDING_SIZE = 128

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


X_train, Y_train = read_csv('ml_resources/train_set(20000).csv')
# X_test, Y_test = read_csv('little_test.csv')


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

# 打乱
np.random.seed(100)
np.random.shuffle(X_train)
np.random.seed(100)
np.random.shuffle(Y_train)

print(X_train)
print(Y_train)
print(X_train.shape, ' ', Y_train.shape)


model = keras.Sequential([
    layers.Embedding(len(word_index), EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LEN),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])
model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.summary()

# history = model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_split=0.1)

model.save("Bi-model1.0.model")
