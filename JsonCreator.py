import json
import csv

f = open('ml_resources/total_data_set.csv','r', encoding='utf-8-sig')
# f1 = open('D:\\课程\\大创\\虚假评论\\程序\\分词&词向量训练\\test_set.csv','a', encoding='utf-8-sig')
f2 = open('ml_resources/test.json','w', encoding="utf-8")
i = 1
j = 0
line = f.readline()
while line:
    if  i<=15:   # 一共339007条数据
        csvReader = csv.reader((line.replace('\0', '') for line in f), delimiter=',')
        for row in csvReader:
            if len(row) < 5:
                print(row)
            else:
                temp = json.dumps({'content':row[0],
                                'label':row[1],
                                'score':row[2],
                                'star1':row[3],
                                'star2':row[4],
                                'star3':row[5]})
                f2.write(temp)
                i += 1
    j += 1
print(i)
print(j)
f.close()
# f1.close()
f2.close()