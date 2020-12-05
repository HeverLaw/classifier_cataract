"""
将remove_bad.csv数据集分为训练集和测试集，本次直接按照比例分配
"""
import pandas as pd
import numpy as np
import random
import csv
# 读取文件，转换格式
file_name = './dataset/remove_bad.csv'

df = pd.read_csv(file_name)
ls = list(df.values)
# 初始化格式，使用字典存储每个row
data = dict()
new_data = dict()
data[0] = []
data[1] = []
data[2] = []
data[3] = []
data[4] = []
new_data[0] = []
new_data[1] = []
new_data[2] = []
new_data[3] = []
new_data[4] = []
# 将每个row放到对应的桶
# w mi

for l in ls:
    data[l[1]].append(l)


# 将数据集分开为训练集和测试集
# num_trian = [1225, 700, 980, 315, 280]
train_list = []
test_list = []

# 遍历5个类型
for k, v in data.items():
    sample = random.sample(range(0, len(v)), round(len(v) * 0.1))
    sample.sort()
    j = 0
    for i in range(len(v)):
        if j < len(sample) and i == sample[j]:
            test_list.append(v[i])
            j += 1
        elif j == len(sample):
            train_list.append(v[i])
        elif i != sample[j]:
            train_list.append(v[i])



# 保存train_list_jpeg和test_list_jpeg
with open('./dataset/32000train.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    writer.writerows(train_list)
with open('./dataset/32000test.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    writer.writerows(test_list)




