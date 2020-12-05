"""
将数据集分为训练集和测试集（将双眼视为一对数据，不拆分）
比例为7:3（0.428），训练集每个level约2450,1400,1960,630,560
这是每个level的比值
0.4322 0.4062 0.4322 0.4260 0.4296
注意：在样本中删除了只有单眼的数据
"""
import pandas as pd
import numpy as np
import random
import csv
# 读取文件，转换格式

train_filename = '../dataset/train.csv'
df = pd.read_csv(train_filename)
ls = list(df.values)
# 初始化格式，使用字典存储每个row
train_data = dict()
train_data[0] = []
train_data[1] = []
train_data[2] = []
train_data[3] = []
train_data[4] = []
# 将每个row放到对应的桶
# 25802，2438，5288，872，708
cat = 0
for l in ls:
    cat = l[1]
    train_data[cat].append(l[0])

test_filename = '../dataset/test.csv'
df = pd.read_csv(test_filename)
ls = list(df.values)
# 初始化格式，使用字典存储每个row
test_data = dict()
test_data[0] = []
test_data[1] = []
test_data[2] = []
test_data[3] = []
test_data[4] = []
cat = 0
for l in ls:
    cat = l[1]
    test_data[cat].append(l[0])

data = dict()
data[0] = []
data[1] = []
data[2] = []
data[3] = []
data[4] = []
# 从data中去除这部分
all_filename = '../dataset/trainLabels_cropped.csv'
df = pd.read_csv(all_filename)
ls = list(df.values)
for l in ls:
    l[0] += '.jpeg'
    cat = l[1]
    data[cat].append(l[0])
print('')
for i in range(5):
    for j in train_data[i]:
        data[i].remove(j)
for i in range(5):
    for j in test_data[i]:
        data[i].remove(j)

print('a')

# 保存train_list_jpeg和test_list_jpeg
with open('../dataset/other-test.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    for i in range(5):
        for name in data[i]:
            writer.writerow([name, i])
# # 保存train_list_jpeg和test_list_jpeg
# with open('../dataset/train.csv', 'w', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(['image', 'level'])
#     writer.writerows(train_list)
# with open('../dataset/test.csv', 'w', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(['image', 'level'])
#     writer.writerows(test_list)



