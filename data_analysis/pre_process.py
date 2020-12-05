import cv2
import glob
import math


def crop_image_to_RGB(image, new_height=1024):
    '''
    先resize，后crop，返回resise并裁剪的图像
    :param image: cv2读入的图像
    :param new_height:
    :return:
    '''
    height = image.shape[0]
    width = image.shape[1]
    ratio = height / width
    if height > new_height:
        image = cv2.resize(image, (math.ceil(new_height / ratio), new_height))
    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    binary, contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no contours!')
        flag = 0
        return output
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x); y = int(y); r = int(r)
    flag = 1
    print(x,y,r)
    if r > 100:
        return output[(y-r):(y+r),
               (x-r):(x+r),
               :]

if __name__ == '__main__':
    # 使用cv2读入图像
    files = glob.glob('./dataset/sample/*.jpeg')
    image = cv2.imread(files[0])
    output = crop_image(image, 1024)
