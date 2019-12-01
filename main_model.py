# -*- coding：utf-8 -*-
import jieba
import re
from keras_preprocessing import sequence
import numpy as np
import gensim
import tensorflow.keras as keras
from tensorflow.keras import layers
import csv

# 服务器配置


# 配置相关维度，这里统一回归到这里
MAX_DOCUMENT_LEN = 300
TRAINING_SIZE = 60000

# 使用训练的word2Vec的词向量的配置
myPath = 'ml_resources/Word2VecModel.vector'
Word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(myPath)
EMBEDDING_SIZE = 128

# 使用腾讯70000词的词向量的配置
# myPath = 'ml_resources/70000_tencent.vector'
# Word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(myPath)
# EMBEDDING_SIZE = 200

# 构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]  # 存储 所有的 词语

word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。  “词语-序号”字典
word_vector = {}  # 初始化`[word : vector]`字典

# 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数为所有单词数+1
embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

# 填充上述的字典和大矩阵
for i in range(len(vocab_list)):
    # print(i)
    word = vocab_list[i]  # 每个词语
    word_index[word] = i + 1 # 词语：序号
    word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
    embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
# print(embeddings_matrix.shape)
# print(word_index) # 查看大字典内容

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
    A = np.asarray(score[0:TRAINING_SIZE], dtype=int)
    B = np.asarray(star1[0:TRAINING_SIZE], dtype=int)
    C = np.asarray(star2[0:TRAINING_SIZE], dtype=int)
    D = np.asarray(star3[0:TRAINING_SIZE], dtype=int)
    return X, Y, A, B, C, D


X_train, Y_train, score, star1, star2, star3 = read_csv('ml_resources/training_data_set.csv')
print(Y_train.mean()) # 统计数据集中1的占比
behavior_input = np.concatenate((score.reshape(-1,1), star1.reshape(-1,1), star2.reshape(-1,1), star3.reshape(-1,1)), axis=1)
print(behavior_input)
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

# 主训练模型部分
input1 = keras.Input(shape=(MAX_DOCUMENT_LEN,))
embedding = layers.Embedding(len(word_index), EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LEN, embeddings_initializer=keras.initializers.Constant(embeddings_matrix))(input1)

# 使用RNN模型训练部分（结果为x）
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedding)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

# 使用CNN模型训练部分（结果为y）
filters = 250
kernel_size = 3
hidden_dims = 250
max_features = 400000

y = layers.Dropout(0.2)(embedding)
y = layers.Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(y)
# we use max pooling:
y = layers.GlobalMaxPooling1D()(y)

# We add a vanilla hidden layer:
y = layers.Dense(hidden_dims)(y)
y = layers.Dropout(0.2)(y)
y = layers.Activation('relu')(y)

# We project onto a single unit output layer, and squash it with a sigmoid:
y = layers.Dense(1)(y)

# 连接CNN和RNN模型
input2 = keras.Input(shape=(1,))
x = keras.layers.concatenate([x, y])
x = layers.Dense(3, activation='relu')(x)
output_tensor = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model([input1, input2], output_tensor)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.summary()

# 模型可视化

history = model.fit([X_train, score], Y_train, batch_size=128, epochs=3, validation_split=0.05, callbacks=[keras.callbacks.TensorBoard(log_dir='result')])

model.save("TrainResult_full.h5")

