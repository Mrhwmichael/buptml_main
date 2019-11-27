# -*- coding：utf-8 -*-
import jieba

# s = "但是这是用来测试的文字啊"
f = open('D:\\课程\\大创\\虚假评论\\程序\\分词&词向量训练\\comment_corpus.txt','r', encoding='utf-8-sig')
s = f.readline()

#加载自定义词典
#jieba.load_userdict('D:\\课程\\大创\\虚假评论\\程序\\分词\\dict.txt')

# 加载停用词表
stop = [line.strip() for line in open('D:\\课程\\大创\\虚假评论\\程序\\分词&词向量训练\\stop.txt', "r", encoding='UTF-8').readlines()]
# print(stop)

# 去掉停用词后并写入文件
new_f = open('D:\\课程\\大创\\虚假评论\\程序\\分词&词向量训练\\new_dict(stop_version).txt','a', encoding='utf-8-sig')

while s:
    words = jieba.lcut(s)
    # counts = {}
    for word in words:
        if len(word) == 1:
            continue
        else:
            if word not in stop:
                new_f.write('{} '.format(word))
                print(word)
    s = f.readline()
new_f.close()