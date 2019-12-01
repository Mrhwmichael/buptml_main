import csv

f = open('ml_resources/total_data_set.csv','r', encoding='utf-8-sig')
# f1 = open('D:\\课程\\大创\\虚假评论\\程序\\分词&词向量训练\\test_set.csv','a', encoding='utf-8-sig')
f2 = open('ml_resources/training_data_set.csv','w', encoding='utf-8-sig')
i = 1
j = 0
line = f.readline()
while line:
    if  j<=30000 or j>309008:   # 一共339007条数据
        f2.write('{}'.format(line))
        i += 1
    line = f.readline()
    j += 1
print(i)
print(j)
f.close()
# f1.close()
f2.close()
