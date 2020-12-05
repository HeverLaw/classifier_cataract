"""
计算训练集和测试集的像素RGB平均值
r=105.0051
g=73.4359
b=52.3038
"""
import cv2
from numpy import *

train_img_dir= '../dataset/train'
test_img_dir= '../dataset/test'
train_img_list=os.listdir(train_img_dir)
test_img_list=os.listdir(test_img_dir)
img_size=448
sum_r=0
sum_g=0
sum_b=0
count=0

for img_name in train_img_list:
    img_path=os.path.join(train_img_dir, img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_size,img_size))
    sum_r=sum_r+img[:,:,0].mean()
    sum_g=sum_g+img[:,:,1].mean()
    sum_b=sum_b+img[:,:,2].mean()
    count=count+1
for img_name in test_img_list:
    img_path=os.path.join(test_img_dir, img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_size,img_size))
    sum_r=sum_r+img[:,:,0].mean()
    sum_g=sum_g+img[:,:,1].mean()
    sum_b=sum_b+img[:,:,2].mean()
    count=count+1

sum_r=sum_r/count
sum_g=sum_g/count
sum_b=sum_b/count
img_mean=[sum_r,sum_g,sum_b]
print (img_mean)