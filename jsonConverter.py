import os
import json
from random import random

"""
一个用于评论数据的转换器，支持csv或txt直转json，不通用
"""

json_file = 'analyseData.json'
read_file = 'standardReview.txt'

# 取量开关
json_item_num = 50

# 转化函数
def transToJson():
    fr = open(read_file)
    if os.path.exists(json_file):
        os.remove(json_file)

    fw = open(json_file, 'a')
    # 将json格式的数据映射成list的形式
    fw.write('[')

    i = 0
    for line in open(read_file):
        line = fr.readline()
        x = line.split(',', 6)
        temp = {"name": x[0], "id": x[1], "score": x[2], "star1": x[3], "star2": x[4], "star3": x[5][0]}
        fw.write(json.dumps(temp, ensure_ascii=False))
        i += 1
        if i > json_item_num:
            break
        else:
            fw.write(',')

    print('success')
    fw.write(']')


if __name__ == '__main__':
    transToJson()
