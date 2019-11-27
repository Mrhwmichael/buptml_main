# -*- coding：utf-8 -*-
import jieba
import re
from keras_preprocessing import sequence
import numpy as np
import gensim
from tensorflow import keras
from tensorflow.python.keras import layers
import csv

MAX_DOCUMENT_LEN = 300
TRAINING_SIZE = 20000

# 使用训练的word2Vec的词向量的配置
myPath = 'ml_resources/Word2VecModel.vector'
Word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(myPath)
EMBEDDING_SIZE = 128

# 使用腾讯70000词的词向量的配置
# myPath = 'ml_resources/70000_tencent.vector'
# Word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(myPath)
# EMBEDDING_SIZE = 200

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

# 目前生成六个内容，分别是label标签，总评score，星级别star1，star2，star3
def read_csv(filename):
    content = []
    label = []
    score = []
    star1 = []
    star2 = []
    star3 = []
    with open(filename, encoding='utf-8') as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile),  delimiter=',')
        for row in csvReader:
            if len(row)<5:
                print(row)
            else:
                content.append(row[0])
                label.append(row[1])
                score.append(row[2])
                star1.append(row[3])
                star2.append(row[4])
                star3.append(row[5])
    print(len(content))


    np.random.seed(100)
    np.random.shuffle(content)
    np.random.seed(100)
    np.random.shuffle(label)
    np.random.seed(100)
    np.random.shuffle(score)
    np.random.seed(100)
    np.random.shuffle(star1)
    np.random.seed(100)
    np.random.shuffle(star2)
    np.random.seed(100)
    np.random.shuffle(star3)

    X = np.asarray(content[0:TRAINING_SIZE])
    Y = np.asarray(label[0:TRAINING_SIZE], dtype=int)
    A = np.asarray(label[0:TRAINING_SIZE], dtype=int)
    B = np.asarray(label[0:TRAINING_SIZE], dtype=int)
    C = np.asarray(label[0:TRAINING_SIZE], dtype=int)
    D = np.asarray(label[0:TRAINING_SIZE], dtype=int)
    return X, Y, A, B, C, D


X_train, Y_train, score, star1, star2, star3 = read_csv('ml_resources/training_data_set.csv')
behavior_input = [score, star1, star2, star3]
print(len(score))
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

model = keras.Sequential()
model.add(keras.layers.Embedding(len(word_index), EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LEN, name = "embedding"))
model.add(keras.layers.Bidirectional(layers.LSTM(EMBEDDING_SIZE, return_sequences=True), name = "Bi-directional-1"))
model.add(keras.layers.Bidirectional(layers.LSTM(64), name = "Bi-directional-2"))

# model.add(keras.layers.Concatenate([model.get_layer('Bi-directional-2').output, behavior_input]))
model.add(keras.layers.Dense(EMBEDDING_SIZE, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.summary()

# 模型可视化
tensorboard = keras.callbacks.TensorBoard(log_dir='result', write_images=1, histogram_freq=1)

history = model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_split=0.1, callbacks=[tensorboard])

# # 十折训练
# for num in range(1,10):
#     history = model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_split = 0.1)


model.save("Bi-model1.0.model")
model.save_weights("Bi-model1.0.h5")
